from pathlib import Path
import sys
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "code" / "business_entity_resolution")
)

from src.preprocessing import add_normalized_columns
from src.blocking import create_blocking_keys


TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"

# Number of S1 records to use for this validation test.
SAMPLE_SIZE = 100

# Read large files in chunks so we don't load the entire
# 5-million-row files into RAM.
CHUNK_SIZE = 100_000


# ============================================================
# LOAD S1 + GROUND TRUTH
# ============================================================

def load_s1_and_ground_truth():

    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=SAMPLE_SIZE
    )

    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    return s1, ground_truth


# ============================================================
# BUILD TRUE MATCH LOOKUP
# ============================================================

def build_true_match_lookup(s1, ground_truth):
    """
    Create:

        S1 ID -> set of true S2/S3 IDs

    Example:

        S1-123 -> {
            S2-456,
            S2-789,
            S3-111
        }
    """

    sampled_s1_ids = set(
        s1["entity_id"]
    )

    lookup = {}

    for _, row in ground_truth.iterrows():

        s1_id = row["source1_entity_id"]

        # Only keep the 100 S1 records we're testing.
        if s1_id not in sampled_s1_ids:
            continue

        matched_string = row["matched_entity_ids"]

        if not matched_string:
            lookup[s1_id] = set()

        else:
            lookup[s1_id] = {
                entity_id.strip()
                for entity_id in matched_string.split(",")
                if entity_id.strip()
            }

    return lookup


# ============================================================
# BUILD S1 BLOCKING KEYS
# ============================================================

def build_s1_block_keys(s1):

    result = {}

    for _, row in s1.iterrows():

        s1_id = row["entity_id"]

        result[s1_id] = set(
            create_blocking_keys(row)
        )

    return result


# ============================================================
# REVERSE TRUE-MATCH LOOKUP
# ============================================================

def build_reverse_lookup(true_lookup):
    """
    Convert:

        S1 -> {S2, S3, ...}

    into:

        S2/S3 -> {S1, ...}

    Example:

        S1-A -> {S2-X, S3-Y}
        S1-B -> {S2-X}

    becomes:

        S2-X -> {S1-A, S1-B}
        S3-Y -> {S1-A}
    """

    reverse_lookup = {}

    for s1_id, matched_ids in true_lookup.items():

        for entity_id in matched_ids:

            if entity_id not in reverse_lookup:
                reverse_lookup[entity_id] = set()

            reverse_lookup[entity_id].add(
                s1_id
            )

    return reverse_lookup


# ============================================================
# SCAN SOURCE FILE
# ============================================================

def scan_source_for_matches(
    file_path,
    true_lookup,
    s1_keys
):
    """
    Scan one large source file in chunks.

    We only inspect rows whose entity_id is known to be
    a TRUE match for one of our sampled S1 records.

    This means we are NOT generating millions of candidate
    pairs. We are simply testing whether our blocking keys
    would have captured the true matches.
    """

    # --------------------------------------------------------
    # Build reverse lookup:
    #
    # S2/S3 ID -> S1 IDs
    # --------------------------------------------------------

    reverse_lookup = build_reverse_lookup(
        true_lookup
    )

    true_ids = set(
        reverse_lookup.keys()
    )

    found_true_ids = set()

    blocked_true_ids = set()

    total_true_ids = len(true_ids)

    print(
        f"True IDs expected in this source: "
        f"{total_true_ids:,}"
    )

    # --------------------------------------------------------
    # Read the large file in chunks
    # --------------------------------------------------------

    reader = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE
    )

    processed = 0

    for chunk in reader:

        processed += len(chunk)

        # ----------------------------------------------------
        # Only keep rows that are known true matches.
        # ----------------------------------------------------

        relevant = chunk[
            chunk["entity_id"].isin(true_ids)
        ]

        if len(relevant) == 0:
            continue

        # ----------------------------------------------------
        # Normalize only these relevant rows.
        # ----------------------------------------------------

        relevant = add_normalized_columns(
            relevant
        )

        # ----------------------------------------------------
        # Check every true matching record.
        # ----------------------------------------------------

        for _, row in relevant.iterrows():

            record_id = row["entity_id"]

            found_true_ids.add(
                record_id
            )

            # Create blocking keys for S2/S3 record.
            record_keys = set(
                create_blocking_keys(row)
            )

            # Which S1 records should match this record?
            matching_s1_ids = reverse_lookup.get(
                record_id,
                set()
            )

            for s1_id in matching_s1_ids:

                # Safety check.
                if s1_id not in s1_keys:
                    continue

                s1_record_keys = s1_keys[s1_id]

                # If there is at least one shared blocking key,
                # our blocking system would be able to generate
                # this true pair.
                if s1_record_keys & record_keys:

                    blocked_true_ids.add(
                        record_id
                    )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if processed % 500_000 == 0:

            print(
                f"Processed {processed:,} rows | "
                f"true records found: "
                f"{len(found_true_ids):,}/"
                f"{total_true_ids:,} | "
                f"blocked: "
                f"{len(blocked_true_ids):,}"
            )

    return (
        found_true_ids,
        blocked_true_ids
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    s1, ground_truth = (
        load_s1_and_ground_truth()
    )

    print(
        f"S1 sample loaded: "
        f"{len(s1):,}"
    )

    # --------------------------------------------------------
    # Normalize S1
    # --------------------------------------------------------

    print()
    print("Normalizing S1...")

    s1 = add_normalized_columns(
        s1
    )

    # --------------------------------------------------------
    # Build ground truth lookup
    # --------------------------------------------------------

    print(
        "Building ground-truth lookup..."
    )

    true_lookup = build_true_match_lookup(
        s1,
        ground_truth
    )

    # --------------------------------------------------------
    # Build S1 blocking keys
    # --------------------------------------------------------

    print(
        "Building S1 blocking keys..."
    )

    s1_keys = build_s1_block_keys(
        s1
    )

    # --------------------------------------------------------
    # Count true matches
    # --------------------------------------------------------

    all_true_ids = set()

    for matched_ids in true_lookup.values():

        all_true_ids.update(
            matched_ids
        )

    print()
    print("=" * 70)
    print("TRUE MATCH INFORMATION")
    print("=" * 70)

    print(
        f"S1 records checked       : "
        f"{len(s1):,}"
    )

    print(
        f"S1 records with matches  : "
        f"{sum(bool(x) for x in true_lookup.values()):,}"
    )

    print(
        f"Total true S2/S3 IDs     : "
        f"{len(all_true_ids):,}"
    )

    # --------------------------------------------------------
    # SOURCE 2
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SCANNING SOURCE 2")
    print("=" * 70)

    s2_path = (
        TRAIN_DIR /
        "train_source2.tsv"
    )

    s2_found, s2_blocked = (
        scan_source_for_matches(
            file_path=s2_path,
            true_lookup=true_lookup,
            s1_keys=s1_keys
        )
    )

    # --------------------------------------------------------
    # SOURCE 3
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SCANNING SOURCE 3")
    print("=" * 70)

    s3_path = (
        TRAIN_DIR /
        "train_source3.tsv"
    )

    s3_found, s3_blocked = (
        scan_source_for_matches(
            file_path=s3_path,
            true_lookup=true_lookup,
            s1_keys=s1_keys
        )
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    found = (
        s2_found |
        s3_found
    )

    blocked = (
        s2_blocked |
        s3_blocked
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL BLOCKING RECALL")
    print("=" * 70)

    print(
        f"Total true matches       : "
        f"{len(all_true_ids):,}"
    )

    print(
        f"True matches found       : "
        f"{len(found):,}"
    )

    print(
        f"True matches blocked     : "
        f"{len(blocked):,}"
    )

    if len(all_true_ids) > 0:

        recall = (
            len(blocked) /
            len(all_true_ids)
        )

        print(
            f"Blocking recall          : "
            f"{recall:.4%}"
        )

    missed = (
        all_true_ids -
        blocked
    )

    print(
        f"True matches missed      : "
        f"{len(missed):,}"
    )

    # --------------------------------------------------------
    # Show missed IDs
    # --------------------------------------------------------

    if missed:

        print()
        print("=" * 70)
        print("FIRST MISSED TRUE MATCHES")
        print("=" * 70)

        for entity_id in list(missed)[:20]:

            print(entity_id)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()