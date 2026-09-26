from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "code" / "business_entity_resolution")
)

from src.preprocessing import add_normalized_columns
from src.blocking import create_blocking_keys


def load_train_sample(n=100):
    train_dir = PROJECT_ROOT / "dataset" / "train"

    s1 = pd.read_csv(
        train_dir / "train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=n
    )

    gt = pd.read_csv(
        train_dir / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    return s1, gt


def main():

    print("Loading S1 sample and ground truth...")

    s1, gt = load_train_sample(100)

    s1 = add_normalized_columns(s1)

    # Ground truth indexed by S1 ID
    gt_lookup = dict(
        zip(
            gt["source1_entity_id"],
            gt["matched_entity_ids"]
        )
    )

    print("Checking blocking keys...")
    print()

    total_matches = 0
    exact_name_hits = 0
    any_block_hits = 0
    missed = []

    for _, row in s1.iterrows():

        s1_id = row["entity_id"]

        matched_string = gt_lookup.get(
            s1_id,
            ""
        )

        if not matched_string:
            continue

        true_ids = [
            x.strip()
            for x in matched_string.split(",")
            if x.strip()
        ]

        total_matches += len(true_ids)

        s1_keys = set(
            create_blocking_keys(row)
        )

        # We cannot yet compare against the complete S2/S3
        # index here. Instead, this reports how informative
        # the S1 blocking keys are.

        if any(
            key[0] == "exact_name"
            for key in s1_keys
        ):
            exact_name_hits += 1

        if s1_keys:
            any_block_hits += 1

        if not s1_keys:
            missed.append(s1_id)

    print("=" * 70)
    print("BLOCKING KEY CHECK")
    print("=" * 70)

    print(f"S1 records checked       : {len(s1):,}")
    print(f"True matches represented  : {total_matches:,}")
    print(f"S1 with exact-name key    : {exact_name_hits:,}")
    print(f"S1 with blocking keys     : {any_block_hits:,}")
    print(f"S1 with no blocking keys  : {len(missed):,}")

    if missed:
        print()
        print("S1 records with no keys:")
        for x in missed[:20]:
            print(x)


if __name__ == "__main__":
    main()