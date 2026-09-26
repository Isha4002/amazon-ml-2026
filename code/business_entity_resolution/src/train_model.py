from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_loader import load_tsv
from preprocessing import add_normalized_columns
from blocking import create_blocking_keys
from features import build_pair_features


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset"

S1_SAMPLE_SIZE = 500
CHUNK_SIZE = 100_000

MAX_S1_BLOCK_SIZE = 500

NEGATIVE_MULTIPLIER = 5
NO_MATCH_NEGATIVES = 10

RANDOM_STATE = 42


# ============================================================
# F0.5
# ============================================================

def f05_score(precision, recall):
    beta_squared = 0.25

    if precision == 0 and recall == 0:
        return 0.0

    return (
        (1 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
    )


def calculate_macro_f05(
    candidate_df,
    probabilities,
    validation_s1_ids,
    ground_truth,
    threshold,
):
    """
    Calculate the actual challenge-style macro F0.5.

    Important:
    - Ground truth comes from the complete GT file.
    - Predictions come only from generated candidates.
    - Every S1 is included.
    - No-match S1 with no predictions gets score 1.
    """

    candidate_df = candidate_df.copy()
    candidate_df["probability"] = probabilities

    scores = []

    grouped_candidates = {
        s1_id: group
        for s1_id, group in candidate_df.groupby("s1_id")
    }

    for s1_id in validation_s1_ids:

        true_ids = ground_truth.get(s1_id, set())

        if s1_id in grouped_candidates:
            group = grouped_candidates[s1_id]

            predicted_ids = set(
                group.loc[
                    group["probability"] >= threshold,
                    "candidate_id",
                ]
            )
        else:
            predicted_ids = set()

        true_ids = set(true_ids)

        if not true_ids and not predicted_ids:
            score = 1.0

        elif not predicted_ids:
            score = 0.0

        elif not true_ids:
            score = 0.0

        else:
            intersection = true_ids & predicted_ids

            precision = len(intersection) / len(predicted_ids)
            recall = len(intersection) / len(true_ids)

            score = f05_score(precision, recall)

        scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))


# ============================================================
# GROUND TRUTH
# ============================================================

def build_ground_truth_lookup(ground_truth_df):

    lookup = {}

    for _, row in ground_truth_df.iterrows():

        s1_id = row["source1_entity_id"]
        raw_matches = row["matched_entity_ids"]

        if not raw_matches:
            lookup[s1_id] = set()
            continue

        matches = {
            value.strip()
            for value in str(raw_matches).split(",")
            if value.strip()
        }

        lookup[s1_id] = matches

    return lookup


# ============================================================
# SELECT S1 SAMPLE
# ============================================================

def select_s1_sample(source1):

    if len(source1) <= S1_SAMPLE_SIZE:
        sample = source1.copy()
    else:
        sample = source1.sample(
            n=S1_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        ).copy()

    return sample


# ============================================================
# BUILD S1 BLOCKING INDEX
# ============================================================

def build_s1_block_index(source1_sample):

    block_index = defaultdict(set)

    for _, row in source1_sample.iterrows():

        keys = create_blocking_keys(row)

        for key in keys:
            block_index[key].add(row["entity_id"])

    filtered_index = {
        key: values
        for key, values in block_index.items()
        if len(values) <= MAX_S1_BLOCK_SIZE
    }

    print(
        f"S1 blocking index: "
        f"{len(block_index):,} raw blocks -> "
        f"{len(filtered_index):,} usable blocks"
    )

    return filtered_index


# ============================================================
# FEATURE ROW
# ============================================================

def make_feature_row(
    s1_row,
    candidate_row,
    s1_id,
    candidate_id,
    label,
):
    features = build_pair_features(
        s1_row,
        candidate_row,
    )

    features["s1_id"] = s1_id
    features["candidate_id"] = candidate_id
    features["label"] = label

    return features


def process_source_file(
    file_path,
    source1_lookup,
    s1_block_index,
    ground_truth,
    source_name,
):
    """
    Stream one source file in chunks.

    Memory strategy:
    - Keep every positive candidate.
    - Keep only the strongest NEGATIVE_LIMIT negatives per S1.
    - This prevents millions of negative candidate rows from accumulating
      in RAM.
    """

    import heapq

    NEGATIVE_LIMIT = 100

    print()
    print("=" * 70)
    print(f"Processing {source_name}")
    print(file_path)
    print("=" * 70)

    # --------------------------------------------------------
    # Store positives separately.
    # These are important and must not be discarded.
    # --------------------------------------------------------

    positive_rows = []

    # --------------------------------------------------------
    # For each S1 we keep a small heap of hard negatives.
    #
    # Structure:
    # {
    #     s1_id: [
    #         (hardness, counter, feature_row),
    #         ...
    #     ]
    # }
    # --------------------------------------------------------

    negative_heaps = defaultdict(list)

    counter = 0

    total_rows = 0
    total_candidate_pairs = 0
    total_positive_pairs = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        print(
            f"{source_name}: processing chunk "
            f"{chunk_number} ({len(chunk):,} rows)"
        )

        chunk = add_normalized_columns(chunk)

        for _, candidate_row in chunk.iterrows():

            total_rows += 1

            keys = create_blocking_keys(candidate_row)

            matched_s1_ids = set()

            for key in keys:

                ids = s1_block_index.get(key)

                if ids:
                    matched_s1_ids.update(ids)

            if not matched_s1_ids:
                continue

            candidate_id = candidate_row["entity_id"]

            for s1_id in matched_s1_ids:

                s1_row = source1_lookup[s1_id]

                true_ids = ground_truth.get(
                    s1_id,
                    set(),
                )

                label = int(
                    candidate_id in true_ids
                )

                feature_row = make_feature_row(
                    s1_row,
                    candidate_row,
                    s1_id,
                    candidate_id,
                    label,
                )

                feature_row["source"] = source_name

                total_candidate_pairs += 1

                # ------------------------------------------------
                # TRUE MATCH
                # ------------------------------------------------

                if label == 1:

                    positive_rows.append(
                        feature_row
                    )

                    total_positive_pairs += 1

                # ------------------------------------------------
                # NEGATIVE CANDIDATE
                # ------------------------------------------------

                else:

                    # A simple "hardness" score.
                    #
                    # Higher similarity means this negative is
                    # more dangerous for the final matcher.
                    hardness = max(
                        float(
                            feature_row.get(
                                "name_ratio",
                                0.0,
                            )
                        ),
                        float(
                            feature_row.get(
                                "name_partial_ratio",
                                0.0,
                            )
                        ),
                        float(
                            feature_row.get(
                                "name_token_set",
                                0.0,
                            )
                        ),
                        float(
                            feature_row.get(
                                "address_ratio",
                                0.0,
                            )
                        ),
                        float(
                            feature_row.get(
                                "address_partial_ratio",
                                0.0,
                            )
                        ),
                        float(
                            feature_row.get(
                                "address_token_set",
                                0.0,
                            )
                        ),
                    )

                    counter += 1

                    heap = negative_heaps[s1_id]

                    item = (
                        hardness,
                        counter,
                        feature_row,
                    )

                    # Keep the strongest NEGATIVE_LIMIT negatives.
                    if len(heap) < NEGATIVE_LIMIT:

                        heapq.heappush(
                            heap,
                            item,
                        )

                    elif hardness > heap[0][0]:

                        heapq.heapreplace(
                            heap,
                            item,
                        )

        print(
            f"  Rows scanned: "
            f"{total_rows:,}"
        )

        print(
            f"  Candidate pairs seen: "
            f"{total_candidate_pairs:,}"
        )

        print(
            f"  Positive pairs found: "
            f"{total_positive_pairs:,}"
        )

    # --------------------------------------------------------
    # Flatten the bounded negative heaps.
    # --------------------------------------------------------

    negative_rows = []

    for heap in negative_heaps.values():

        for _, _, feature_row in heap:

            negative_rows.append(
                feature_row
            )

    feature_rows = (
        positive_rows
        + negative_rows
    )

    print()
    print(f"{source_name} rows scanned: {total_rows:,}")
    print(
        f"{source_name} candidate pairs seen: "
        f"{total_candidate_pairs:,}"
    )
    print(
        f"{source_name} positive pairs kept: "
        f"{total_positive_pairs:,}"
    )
    print(
        f"{source_name} hard negative pairs kept: "
        f"{len(negative_rows):,}"
    )
    print(
        f"{source_name} total pairs kept in RAM: "
        f"{len(feature_rows):,}"
    )

    return feature_rows


# ============================================================
# SAMPLE TRAINING NEGATIVES
# ============================================================

def sample_training_data(
    full_candidate_df,
    training_s1_ids,
):

    training_df = full_candidate_df[
        full_candidate_df["s1_id"].isin(training_s1_ids)
    ].copy()

    selected_parts = []

    for s1_id, group in training_df.groupby("s1_id"):

        positives = group[group["label"] == 1]
        negatives = group[group["label"] == 0]

        selected_parts.append(positives)

        if len(negatives) == 0:
            continue

        if len(positives) > 0:

            number_of_negatives = min(
                len(negatives),
                len(positives) * NEGATIVE_MULTIPLIER,
            )

        else:

            number_of_negatives = min(
                len(negatives),
                NO_MATCH_NEGATIVES,
            )

        sampled_negatives = negatives.sample(
            n=number_of_negatives,
            random_state=RANDOM_STATE,
        )

        selected_parts.append(sampled_negatives)

    if not selected_parts:
        return training_df.iloc[0:0].copy()

    return pd.concat(
        selected_parts,
        ignore_index=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML 2026 - PROPER ENTITY RESOLUTION TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    train_dir = DATASET_DIR / "train"

    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    print()
    print("Loading Source 1...")

    source1 = load_tsv(s1_path)

    print(f"S1 total rows: {len(source1):,}")

    # --------------------------------------------------------
    # Load Ground Truth
    # --------------------------------------------------------

    print()
    print("Loading ground truth...")

    ground_truth_df = load_tsv(gt_path)

    ground_truth = build_ground_truth_lookup(
        ground_truth_df
    )

    print(
        f"Ground-truth S1 records: "
        f"{len(ground_truth):,}"
    )

    # --------------------------------------------------------
    # Select S1 sample
    # --------------------------------------------------------

    source1_sample = select_s1_sample(source1)

    print()
    print(
        f"S1 training/evaluation sample: "
        f"{len(source1_sample):,}"
    )

    # Normalize S1 sample
    source1_sample = add_normalized_columns(
        source1_sample
    )

    # --------------------------------------------------------
    # Build S1 lookup
    # --------------------------------------------------------

    source1_lookup = {
        row["entity_id"]: row
        for _, row in source1_sample.iterrows()
    }

    # --------------------------------------------------------
    # Build blocking index
    # --------------------------------------------------------

    print()
    print("Building S1 blocking index...")

    s1_block_index = build_s1_block_index(
        source1_sample
    )

    # --------------------------------------------------------
    # Process S2
    # --------------------------------------------------------

    s2_features = process_source_file(
        s2_path,
        source1_lookup,
        s1_block_index,
        ground_truth,
        "S2",
    )

    # --------------------------------------------------------
    # Process S3
    # --------------------------------------------------------

    s3_features = process_source_file(
        s3_path,
        source1_lookup,
        s1_block_index,
        ground_truth,
        "S3",
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    print()
    print("Combining candidate pairs...")

    all_rows = s2_features + s3_features

    candidate_df = pd.DataFrame(all_rows)

    print(
        f"Total candidate pairs: "
        f"{len(candidate_df):,}"
    )

    if candidate_df.empty:
        raise RuntimeError(
            "No candidate pairs were generated."
        )

    # --------------------------------------------------------
    # Blocking recall
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BLOCKING RECALL")
    print("=" * 70)

    sample_s1_ids = set(
        source1_sample["entity_id"]
    )

    total_true_matches = 0
    found_true_matches = 0

    for s1_id in sample_s1_ids:

        true_ids = ground_truth.get(
            s1_id,
            set(),
        )

        total_true_matches += len(true_ids)

        predicted_candidate_ids = set(
            candidate_df.loc[
                candidate_df["s1_id"] == s1_id,
                "candidate_id",
            ]
        )

        found_true_matches += len(
            true_ids & predicted_candidate_ids
        )

    if total_true_matches > 0:

        blocking_recall = (
            found_true_matches
            / total_true_matches
        )

    else:
        blocking_recall = 0.0

    print(
        f"True matches in sampled S1: "
        f"{total_true_matches:,}"
    )

    print(
        f"True matches recovered by blocking: "
        f"{found_true_matches:,}"
    )

    print(
        f"Blocking recall: "
        f"{blocking_recall:.4%}"
    )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print()
    print("Candidate label distribution:")

    print(
        candidate_df["label"]
        .value_counts()
        .to_string()
    )

    positive_count = int(
        candidate_df["label"].sum()
    )

    if positive_count == 0:
        raise RuntimeError(
            "No positive candidate pairs were found. "
            "Blocking is missing the true matches."
        )

    # --------------------------------------------------------
    # Split by S1 entity
    # --------------------------------------------------------

    sample_ids = np.array(
        list(sample_s1_ids)
    )

    train_ids, validation_ids = train_test_split(
        sample_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_ids = set(train_ids)
    validation_ids = set(validation_ids)

    print()
    print(
        f"Training S1 entities: "
        f"{len(train_ids):,}"
    )

    print(
        f"Validation S1 entities: "
        f"{len(validation_ids):,}"
    )

    # --------------------------------------------------------
    # Training negative sampling
    # --------------------------------------------------------

    print()
    print("Sampling training negatives...")

    training_df = sample_training_data(
        candidate_df,
        train_ids,
    )

    print(
        f"Training pairs after sampling: "
        f"{len(training_df):,}"
    )

    print(
        f"Training positives: "
        f"{training_df['label'].sum():,}"
    )

    print(
        f"Training negatives: "
        f"{(training_df['label'] == 0).sum():,}"
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    excluded_columns = {
        "s1_id",
        "candidate_id",
        "label",
        "source",
    }

    feature_columns = [
        column
        for column in training_df.columns
        if column not in excluded_columns
    ]

    print()
    print(
        f"Number of model features: "
        f"{len(feature_columns)}"
    )

    print("Features:")

    for feature in feature_columns:
        print(f"  - {feature}")

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    X_train = training_df[
        feature_columns
    ].astype(float)

    y_train = training_df[
        "label"
    ].astype(int)

    print()
    print("=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Model trained successfully.")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_df = candidate_df[
        candidate_df["s1_id"].isin(validation_ids)
    ].copy()

    X_validation = validation_df[
        feature_columns
    ].astype(float)

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # Threshold tuning using ACTUAL macro F0.5
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MACRO F0.5 THRESHOLD SEARCH")
    print("=" * 70)

    results = []

    thresholds = np.arange(
        0.50,
        0.96,
        0.05,
    )

    for threshold in thresholds:

        score = calculate_macro_f05(
            validation_df,
            validation_probabilities,
            validation_ids,
            ground_truth,
            threshold,
        )

        results.append(
            (threshold, score)
        )

        print(
            f"Threshold {threshold:.2f} "
            f"-> Macro F0.5 = {score:.4f}"
        )

    # --------------------------------------------------------
    # Best threshold
    # --------------------------------------------------------

    best_threshold, best_score = max(
        results,
        key=lambda x: x[1],
    )

    print()
    print("=" * 70)
    print("BEST VALIDATION RESULT")
    print("=" * 70)

    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation Macro F0.5: "
        f"{best_score:.4f}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    print()
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()