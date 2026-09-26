from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "code" / "business_entity_resolution")
)


# ============================================================
# IMPORT OUR FUNCTIONS
# ============================================================

from src.preprocessing import add_normalized_columns
from src.features import build_pair_features


# ============================================================
# MAIN TEST
# ============================================================

def main():

    train_dir = PROJECT_ROOT / "dataset" / "train"

    # --------------------------------------------------------
    # Load a very small sample
    # --------------------------------------------------------

    print("Loading sample data...")

    s1 = pd.read_csv(
        train_dir / "train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=5
    )

    s2 = pd.read_csv(
        train_dir / "train_source2.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=5
    )

    # --------------------------------------------------------
    # Add normalized/helper columns
    # --------------------------------------------------------

    print("Normalizing data...")

    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)

    # --------------------------------------------------------
    # Select one S1 and one S2 record
    # --------------------------------------------------------

    row1 = s1.iloc[0]
    row2 = s2.iloc[0]

    # --------------------------------------------------------
    # Build matching features
    # --------------------------------------------------------

    print("Building features...")

    features = build_pair_features(
        row1,
        row2
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE TEST")
    print("=" * 70)

    print()
    print(f"S1 ID: {row1['entity_id']}")
    print(f"S1 Name: {row1['business_name']}")

    print()

    print(f"S2 ID: {row2['entity_id']}")
    print(f"S2 Name: {row2['business_name']}")

    print()
    print("-" * 70)
    print("FEATURE VALUES")
    print("-" * 70)

    for name, value in features.items():

        print(
            f"{name:30s}: {value}"
        )

    print()
    print("-" * 70)

    print(
        f"Total features: {len(features)}"
    )

    print()
    print("Feature test completed successfully!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()