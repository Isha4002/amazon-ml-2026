from pathlib import Path

import pandas as pd

from data_loader import load_tsv
from preprocessing import add_normalized_columns
from features import build_pair_features


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAINING_PAIRS_PATH = OUTPUT_DIR / "training_pairs.tsv"
OUTPUT_PATH = OUTPUT_DIR / "training_features.tsv"

CHUNK_SIZE = 100_000


# ============================================================
# LOAD TRAINING PAIRS
# ============================================================

def load_training_pairs():

    print("=" * 70)
    print("LOADING TRAINING PAIRS")
    print("=" * 70)

    training_pairs = pd.read_csv(
        TRAINING_PAIRS_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    training_pairs["label"] = (
        training_pairs["label"].astype(int)
    )

    print(f"Training pairs: {len(training_pairs):,}")
    print(
        f"Positive pairs: "
        f"{(training_pairs['label'] == 1).sum():,}"
    )
    print(
        f"Negative pairs: "
        f"{(training_pairs['label'] == 0).sum():,}"
    )

    return training_pairs


# ============================================================
# GET REQUIRED IDS
# ============================================================

def get_required_ids(training_pairs):

    s1_ids = set(
        training_pairs["s1_id"].unique()
    )

    s2_ids = set(
        training_pairs.loc[
            training_pairs["candidate_id"].str.startswith("S2-"),
            "candidate_id"
        ].unique()
    )

    s3_ids = set(
        training_pairs.loc[
            training_pairs["candidate_id"].str.startswith("S3-"),
            "candidate_id"
        ].unique()
    )

    print("\nRequired records:")
    print(f"S1: {len(s1_ids):,}")
    print(f"S2: {len(s2_ids):,}")
    print(f"S3: {len(s3_ids):,}")

    return s1_ids, s2_ids, s3_ids


# ============================================================
# FIND ONLY REQUIRED RECORDS
# ============================================================

def find_records_by_ids(file_path, required_ids, source_name):

    print("\n" + "=" * 70)
    print(f"RETRIEVING {source_name} RECORDS")
    print("=" * 70)

    print(
        f"Required {source_name} IDs: "
        f"{len(required_ids):,}"
    )

    found = {}

    rows_scanned = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        rows_scanned += len(chunk)

        # Only keep records needed for our training pairs
        matches = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for _, row in matches.iterrows():

            found[row["entity_id"]] = row.to_dict()

        if chunk_number % 10 == 0:

            print(
                f"Scanned: {rows_scanned:,} | "
                f"Found: {len(found):,}/"
                f"{len(required_ids):,}"
            )

        # Stop as soon as everything is found
        if len(found) == len(required_ids):
            break

    print(
        f"{source_name} complete."
    )

    print(
        f"Found: {len(found):,}/"
        f"{len(required_ids):,}"
    )

    if len(found) != len(required_ids):

        missing = required_ids - set(found.keys())

        print(
            f"WARNING: Missing {len(missing):,} "
            f"{source_name} records."
        )

        for entity_id in list(missing)[:10]:
            print(f"  Missing: {entity_id}")

    return found


# ============================================================
# NORMALIZE ONLY REQUIRED RECORDS
# ============================================================

def normalize_lookup(raw_lookup):

    if not raw_lookup:
        return {}

    df = pd.DataFrame(
        list(raw_lookup.values())
    )

    df = add_normalized_columns(df)

    lookup = {}

    for _, row in df.iterrows():

        lookup[row["entity_id"]] = row

    return lookup


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(
    training_pairs,
    s1_lookup,
    s2_lookup,
    s3_lookup
):

    print("\n" + "=" * 70)
    print("BUILDING FUZZY FEATURES")
    print("=" * 70)

    total = len(training_pairs)

    feature_rows = []

    for i, (_, pair) in enumerate(
        training_pairs.iterrows(),
        start=1
    ):

        s1_id = pair["s1_id"]
        candidate_id = pair["candidate_id"]
        label = int(pair["label"])

        # ----------------------------------------------------
        # S1
        # ----------------------------------------------------

        s1_row = s1_lookup.get(s1_id)

        if s1_row is None:

            print(
                f"WARNING: Missing S1: {s1_id}"
            )

            continue

        # ----------------------------------------------------
        # Candidate
        # ----------------------------------------------------

        if candidate_id.startswith("S2-"):

            candidate_row = s2_lookup.get(
                candidate_id
            )

            source = "S2"

        elif candidate_id.startswith("S3-"):

            candidate_row = s3_lookup.get(
                candidate_id
            )

            source = "S3"

        else:

            print(
                f"WARNING: Unknown candidate: "
                f"{candidate_id}"
            )

            continue

        if candidate_row is None:

            print(
                f"WARNING: Missing candidate: "
                f"{candidate_id}"
            )

            continue

        # ----------------------------------------------------
        # Calculate features
        # ----------------------------------------------------

        features = build_pair_features(
            s1_row,
            candidate_row
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "s1_id": s1_id,
            "candidate_id": candidate_id,
            "source": source,
            "label": label,
        }

        result.update(features)

        feature_rows.append(result)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if i % 500 == 0 or i == total:

            percentage = (
                i / total
            ) * 100

            print(
                f"Processed: "
                f"{i:,}/{total:,} "
                f"({percentage:.1f}%)"
            )

    return pd.DataFrame(feature_rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING TRAINING FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Check training pairs
    # --------------------------------------------------------

    if not TRAINING_PAIRS_PATH.exists():

        raise FileNotFoundError(
            f"\nTraining pairs file not found:\n"
            f"{TRAINING_PAIRS_PATH}\n\n"
            f"Run prepare_training_pairs.py first."
        )

    # --------------------------------------------------------
    # Load pairs
    # --------------------------------------------------------

    training_pairs = load_training_pairs()

    # --------------------------------------------------------
    # Required IDs
    # --------------------------------------------------------

    s1_ids, s2_ids, s3_ids = (
        get_required_ids(training_pairs)
    )

    train_dir = DATASET_DIR / "train"

    # --------------------------------------------------------
    # Retrieve only required S1 records
    # --------------------------------------------------------

    s1_raw = find_records_by_ids(
        train_dir / "train_source1.tsv",
        s1_ids,
        "S1"
    )

    # --------------------------------------------------------
    # Retrieve only required S2 records
    # --------------------------------------------------------

    s2_raw = find_records_by_ids(
        train_dir / "train_source2.tsv",
        s2_ids,
        "S2"
    )

    # --------------------------------------------------------
    # Retrieve only required S3 records
    # --------------------------------------------------------

    s3_raw = find_records_by_ids(
        train_dir / "train_source3.tsv",
        s3_ids,
        "S3"
    )

    # --------------------------------------------------------
    # Normalize only the records we need
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("NORMALIZING REQUIRED RECORDS")
    print("=" * 70)

    print("Normalizing S1...")
    s1_lookup = normalize_lookup(s1_raw)

    print("Normalizing S2...")
    s2_lookup = normalize_lookup(s2_raw)

    print("Normalizing S3...")
    s3_lookup = normalize_lookup(s3_raw)

    print("\nNormalized records ready.")

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    features_df = build_features(
        training_pairs,
        s1_lookup,
        s2_lookup,
        s3_lookup
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    features_df.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FEATURES READY")
    print("=" * 70)

    print(
        f"Rows: {len(features_df):,}"
    )

    print(
        f"Columns: {len(features_df.columns):,}"
    )

    if "label" in features_df.columns:

        print(
            f"Positive: "
            f"{(features_df['label'] == 1).sum():,}"
        )

        print(
            f"Negative: "
            f"{(features_df['label'] == 0).sum():,}"
        )

    print("\nColumns:")

    for column in features_df.columns:

        print(f"  - {column}")

    print("\nSaved to:")

    print(OUTPUT_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()