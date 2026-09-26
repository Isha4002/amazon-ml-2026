from pathlib import Path
from collections import defaultdict

import pandas as pd

from data_loader import load_tsv
from preprocessing import add_normalized_columns
from blocking import create_blocking_keys


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"

S1_SAMPLE_SIZE = 500
CHUNK_SIZE = 100_000

NEGATIVES_PER_POSITIVE = 5
NO_MATCH_NEGATIVES = 10

RANDOM_STATE = 42


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

    sample_size = min(
        S1_SAMPLE_SIZE,
        len(source1),
    )

    return source1.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).copy()


# ============================================================
# COLLECT TRUE MATCH IDS
# ============================================================

def collect_required_ids(
    source1_sample,
    ground_truth,
):

    required_s2_ids = set()
    required_s3_ids = set()

    for s1_id in source1_sample["entity_id"]:

        true_ids = ground_truth.get(
            s1_id,
            set(),
        )

        for entity_id in true_ids:

            if entity_id.startswith("S2-"):
                required_s2_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                required_s3_ids.add(entity_id)

    return (
        required_s2_ids,
        required_s3_ids,
    )


# ============================================================
# FIND RECORDS BY ENTITY ID
# ============================================================

def find_records_by_ids(
    file_path,
    required_ids,
):

    found = {}

    if not required_ids:
        return found

    print()
    print("=" * 70)
    print(f"Finding records in {file_path.name}")
    print("=" * 70)

    print(
        f"Required IDs: "
        f"{len(required_ids):,}"
    )

    rows_scanned = 0

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

        rows_scanned += len(chunk)

        matches = chunk[
            chunk["entity_id"].isin(
                required_ids
            )
        ]

        for _, row in matches.iterrows():

            found[
                row["entity_id"]
            ] = row.copy()

        if chunk_number % 10 == 0:

            print(
                f"  Scanned: "
                f"{rows_scanned:,} | "
                f"Found: "
                f"{len(found):,}/"
                f"{len(required_ids):,}"
            )

        if len(found) == len(required_ids):
            break

    print(
        f"Finished: "
        f"{len(found):,}/"
        f"{len(required_ids):,} found"
    )

    return found


# ============================================================
# CHEAP HARDNESS SCORE
# ============================================================

def cheap_similarity(
    s1_row,
    candidate_row,
):
    """
    Cheap score used ONLY for selecting hard negatives.

    This is NOT the ML model.

    Expensive fuzzy features will be calculated later
    on the final training-pair dataset.
    """

    score = 0

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    s1_name = s1_row.get(
        "name_matching",
        "",
    )

    candidate_name = candidate_row.get(
        "name_matching",
        "",
    )

    if (
        s1_name
        and candidate_name
    ):

        if s1_name == candidate_name:
            score += 100

        elif s1_name[:4] == candidate_name[:4]:
            score += 40

        s1_tokens = set(
            s1_name.split()
        )

        candidate_tokens = set(
            candidate_name.split()
        )

        shared_name_tokens = len(
            s1_tokens & candidate_tokens
        )

        score += min(
            shared_name_tokens * 10,
            30,
        )

    # --------------------------------------------------------
    # Address numbers
    # --------------------------------------------------------

    s1_numbers = s1_row.get(
        "address_numbers",
        [],
    )

    candidate_numbers = candidate_row.get(
        "address_numbers",
        [],
    )

    if not isinstance(
        s1_numbers,
        list,
    ):
        s1_numbers = []

    if not isinstance(
        candidate_numbers,
        list,
    ):
        candidate_numbers = []

    if (
        set(s1_numbers)
        & set(candidate_numbers)
    ):
        score += 30

    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    s1_tokens = s1_row.get(
        "address_tokens",
        [],
    )

    candidate_tokens = candidate_row.get(
        "address_tokens",
        [],
    )

    if not isinstance(
        s1_tokens,
        list,
    ):
        s1_tokens = []

    if not isinstance(
        candidate_tokens,
        list,
    ):
        candidate_tokens = []

    shared_address_tokens = len(
        set(s1_tokens)
        & set(candidate_tokens)
    )

    score += min(
        shared_address_tokens * 5,
        25,
    )

    return score


# ============================================================
# BUILD S1 BLOCK INDEX
# ============================================================

def build_s1_block_index(
    source1_sample,
):

    block_index = defaultdict(set)

    for _, row in source1_sample.iterrows():

        keys = create_blocking_keys(
            row
        )

        for key in keys:

            block_index[key].add(
                row["entity_id"]
            )

    return block_index


# ============================================================
# FIND HARD NEGATIVES
# ============================================================

def find_hard_negatives(
    source1_sample,
    ground_truth,
):
    """
    Scan S2/S3 and keep only hard negative candidates.

    Result:

        [
            {
                "s1_id": ...,
                "candidate_id": ...,
                "source": ...,
                "candidate_row": ...,
                "hardness": ...
            }
        ]
    """

    print()
    print("=" * 70)
    print("BUILDING HARD NEGATIVES")
    print("=" * 70)

    # --------------------------------------------------------
    # S1 lookup
    # --------------------------------------------------------

    s1_lookup = {
        row["entity_id"]: row
        for _, row in source1_sample.iterrows()
    }

    # --------------------------------------------------------
    # Blocking index
    # --------------------------------------------------------

    s1_block_index = build_s1_block_index(
        source1_sample
    )

    print(
        f"S1 blocking keys: "
        f"{len(s1_block_index):,}"
    )

    # --------------------------------------------------------
    # Candidate heaps
    #
    # We store only a limited number of negatives per S1.
    # --------------------------------------------------------

    negative_candidates = defaultdict(list)

    # Maximum number of candidates kept per S1
    MAX_NEGATIVES_PER_S1 = 25

    source_files = [
        (
            DATASET_DIR
            / "train"
            / "train_source2.tsv",
            "S2",
        ),
        (
            DATASET_DIR
            / "train"
            / "train_source3.tsv",
            "S3",
        ),
    ]

    total_candidate_pairs = 0

    # --------------------------------------------------------
    # Scan S2 and S3
    # --------------------------------------------------------

    for file_path, source_name in source_files:

        print()
        print(
            f"Scanning {source_name}..."
        )

        rows_scanned = 0

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

            rows_scanned += len(chunk)

            chunk = add_normalized_columns(
                chunk
            )

            for _, candidate_row in chunk.iterrows():

                candidate_id = candidate_row[
                    "entity_id"
                ]

                keys = create_blocking_keys(
                    candidate_row
                )

                matched_s1_ids = set()

                for key in keys:

                    s1_ids = (
                        s1_block_index.get(
                            key
                        )
                    )

                    if s1_ids:
                        matched_s1_ids.update(
                            s1_ids
                        )

                if not matched_s1_ids:
                    continue

                for s1_id in matched_s1_ids:

                    # ------------------------------------------------
                    # Never use a true match as a negative.
                    # ------------------------------------------------

                    true_ids = ground_truth.get(
                        s1_id,
                        set(),
                    )

                    if candidate_id in true_ids:
                        continue

                    s1_row = s1_lookup[
                        s1_id
                    ]

                    hardness = cheap_similarity(
                        s1_row,
                        candidate_row,
                    )

                    total_candidate_pairs += 1

                    negative_candidates[
                        s1_id
                    ].append(
                        (
                            hardness,
                            candidate_id,
                            source_name,
                            candidate_row.copy(),
                        )
                    )

                    # Keep only the strongest candidates.
                    negative_candidates[
                        s1_id
                    ].sort(
                        key=lambda x: x[0],
                        reverse=True,
                    )

                    if len(
                        negative_candidates[s1_id]
                    ) > MAX_NEGATIVES_PER_S1:

                        negative_candidates[
                            s1_id
                        ] = negative_candidates[
                            s1_id
                        ][:MAX_NEGATIVES_PER_S1]

            if chunk_number % 10 == 0:

                print(
                    f"  Scanned: "
                    f"{rows_scanned:,} rows | "
                    f"Candidate negatives seen: "
                    f"{total_candidate_pairs:,}"
                )

        print(
            f"{source_name} complete: "
            f"{rows_scanned:,} rows scanned"
        )

    # --------------------------------------------------------
    # Convert to output list
    # --------------------------------------------------------

    selected_negatives = []

    for s1_id, candidates in (
        negative_candidates.items()
    ):

        true_count = len(
            ground_truth.get(
                s1_id,
                set(),
            )
        )

        if true_count > 0:

            limit = min(
                len(candidates),
                true_count
                * NEGATIVES_PER_POSITIVE,
            )

        else:

            limit = min(
                len(candidates),
                NO_MATCH_NEGATIVES,
            )

        for item in candidates[:limit]:

            hardness = item[0]
            candidate_id = item[1]
            source_name = item[2]
            candidate_row = item[3]

            selected_negatives.append({
                "s1_id": s1_id,
                "candidate_id": candidate_id,
                "source": source_name,
                "candidate_row": candidate_row,
                "hardness": hardness,
            })

    print()
    print(
        f"Hard negatives selected: "
        f"{len(selected_negatives):,}"
    )

    return selected_negatives


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PREPARING TRAINING PAIRS")
    print("=" * 70)

    train_dir = DATASET_DIR / "train"

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    print()
    print("Loading Source 1...")

    source1 = load_tsv(
        train_dir / "train_source1.tsv"
    )

    source1_sample = select_s1_sample(
        source1
    )

    source1_sample = add_normalized_columns(
        source1_sample
    )

    print(
        f"S1 sample: "
        f"{len(source1_sample):,}"
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    print()
    print("Loading ground truth...")

    ground_truth_df = load_tsv(
        train_dir / "train_ground_truth.tsv"
    )

    ground_truth = build_ground_truth_lookup(
        ground_truth_df
    )

    # --------------------------------------------------------
    # Find required true-match IDs
    # --------------------------------------------------------

    print()
    print(
        "Collecting true-match IDs..."
    )

    required_s2_ids, required_s3_ids = (
        collect_required_ids(
            source1_sample,
            ground_truth,
        )
    )

    print(
        f"Required S2 IDs: "
        f"{len(required_s2_ids):,}"
    )

    print(
        f"Required S3 IDs: "
        f"{len(required_s3_ids):,}"
    )

    # --------------------------------------------------------
    # Retrieve actual positive records
    # --------------------------------------------------------

    s2_positive_records = (
        find_records_by_ids(
            train_dir / "train_source2.tsv",
            required_s2_ids,
        )
    )

    s3_positive_records = (
        find_records_by_ids(
            train_dir / "train_source3.tsv",
            required_s3_ids,
        )
    )

    positive_records = {}

    positive_records.update(
        s2_positive_records
    )

    positive_records.update(
        s3_positive_records
    )

    expected_positive_count = (
        len(required_s2_ids)
        + len(required_s3_ids)
    )

    print()
    print("=" * 70)
    print("POSITIVE RECORD CHECK")
    print("=" * 70)

    print(
        f"Expected: "
        f"{expected_positive_count:,}"
    )

    print(
        f"Retrieved: "
        f"{len(positive_records):,}"
    )

    # --------------------------------------------------------
    # Find hard negatives
    # --------------------------------------------------------

    selected_negatives = (
        find_hard_negatives(
            source1_sample,
            ground_truth,
        )
    )

    # --------------------------------------------------------
    # Build final training pair table
    # --------------------------------------------------------

    print()
    print(
        "Building training-pair table..."
    )

    rows = []

    # --------------------------------------------------------
    # Positive pairs
    # --------------------------------------------------------

    for _, s1_row in (
        source1_sample.iterrows()
    ):

        s1_id = s1_row[
            "entity_id"
        ]

        true_ids = ground_truth.get(
            s1_id,
            set(),
        )

        for candidate_id in true_ids:

            candidate_row = (
                positive_records.get(
                    candidate_id
                )
            )

            if candidate_row is None:
                continue

            source_name = (
                "S2"
                if candidate_id.startswith(
                    "S2-"
                )
                else "S3"
            )

            rows.append({
                "s1_id": s1_id,
                "candidate_id": candidate_id,
                "label": 1,
                "source": source_name,
            })

    # --------------------------------------------------------
    # Negative pairs
    # --------------------------------------------------------

    for item in selected_negatives:

        rows.append({
            "s1_id": item["s1_id"],
            "candidate_id": item["candidate_id"],
            "label": 0,
            "source": item["source"],
        })

    output_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "training_pairs.tsv"
    )

    output_df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING PAIRS READY")
    print("=" * 70)

    print(
        f"Total pairs: "
        f"{len(output_df):,}"
    )

    print(
        f"Positive pairs: "
        f"{(output_df['label'] == 1).sum():,}"
    )

    print(
        f"Negative pairs: "
        f"{(output_df['label'] == 0).sum():,}"
    )

    print()
    print(
        f"Saved to:"
    )

    print(output_path)

    print()
    print(
        "Next step: build fuzzy features only "
        "for these training pairs."
    )


if __name__ == "__main__":
    main()