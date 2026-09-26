from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "code" / "business_entity_resolution")
)

from src.preprocessing import add_normalized_columns
from src.blocking import generate_candidate_pairs


def load_sample(file_name, n=5000):
    path = PROJECT_ROOT / "dataset" / "train" / file_name

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=n
    )


def main():

    print("Loading samples...")

    s1 = load_sample("train_source1.tsv", 1000)
    s2 = load_sample("train_source2.tsv", 5000)
    s3 = load_sample("train_source3.tsv", 5000)

    print("Normalizing...")

    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)

    print("Generating candidates...")

    candidates = generate_candidate_pairs(
        s1,
        s2,
        s3
    )

    print()
    print("=" * 70)
    print("BLOCKING RESULT")
    print("=" * 70)

    print(f"S1 records       : {len(s1):,}")
    print(f"S2 records       : {len(s2):,}")
    print(f"S3 records       : {len(s3):,}")
    print(f"Candidate pairs  : {len(candidates):,}")

    if len(s1) > 0:
        print(
            f"Average candidates per S1: "
            f"{len(candidates) / len(s1):.2f}"
        )


if __name__ == "__main__":
    main()