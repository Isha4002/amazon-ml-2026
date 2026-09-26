from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"


def show_samples(file_name, n=10):
    path = TRAIN_DIR / file_name

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=n,
    )

    print("\n" + "=" * 80)
    print(file_name)
    print("=" * 80)

    for _, row in df.iterrows():
        print(f"\nID      : {row['entity_id']}")
        print(f"NAME    : {row['business_name']}")
        print(f"ADDRESS : {row['business_address']}")
        print(f"COUNTRY : {row['country']}")


def main():
    show_samples("train_source1.tsv")
    show_samples("train_source2.tsv")
    show_samples("train_source3.tsv")


if __name__ == "__main__":
    main()