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


TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"


MISSED_IDS = {
    "S2-837366545",
    "S3-250788566",
    "S2-904010001",
    "S2-659407377",
    "S3-426660831",
    "S3-807299919",
    "S3-805705209",
    "S2-156312637",
    "S2-975207252",
    "S3-317748253",
    "S3-685337179",
    "S2-185860935",
    "S3-355590735",
    "S2-494260820",
}


def load_ground_truth():

    path = TRAIN_DIR / "train_ground_truth.tsv"

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def find_s1_for_missed_ids():

    gt = load_ground_truth()

    results = []

    for _, row in gt.iterrows():

        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        ids = {
            x.strip()
            for x in matched_ids.split(",")
            if x.strip()
        }

        overlap = ids & MISSED_IDS

        for missed_id in overlap:

            results.append(
                (
                    missed_id,
                    row["source1_entity_id"]
                )
            )

    return results


def load_records(ids):

    ids = set(ids)

    found = {}

    for filename in [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv"
    ]:

        path = TRAIN_DIR / filename

        # Read in chunks because files are huge.
        reader = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=100_000
        )

        for chunk in reader:

            matches = chunk[
                chunk["entity_id"].isin(ids)
            ]

            for _, row in matches.iterrows():

                found[
                    row["entity_id"]
                ] = row.to_dict()

            if len(found) == len(ids):
                return found

    return found


def main():

    print("=" * 80)
    print("INSPECTING MISSED TRUE MATCHES")
    print("=" * 80)

    pairs = find_s1_for_missed_ids()

    missed_ids = set(
        missed_id
        for missed_id, _ in pairs
    )

    s1_ids = set(
        s1_id
        for _, s1_id in pairs
    )

    print(
        f"Missed matches found : {len(missed_ids)}"
    )

    print(
        f"Corresponding S1 IDs : {len(s1_ids)}"
    )

    print()
    print("Loading records...")

    records = load_records(
        missed_ids | s1_ids
    )

    print()

    for missed_id, s1_id in pairs:

        s1 = records.get(s1_id)
        matched = records.get(missed_id)

        print("=" * 80)

        print(f"S1 ID      : {s1_id}")
        print(f"MISSED ID  : {missed_id}")

        print()

        if s1:

            print("SOURCE 1")
            print(
                f"Name      : "
                f"{s1['business_name']}"
            )
            print(
                f"Address   : "
                f"{s1['business_address']}"
            )
            print(
                f"Country   : "
                f"{s1['country']}"
            )

            # Show normalized values.
            s1_df = pd.DataFrame([s1])
            s1_df = add_normalized_columns(s1_df)
            s1_row = s1_df.iloc[0]

            print()
            print(
                f"Normalized name: "
                f"{s1_row.get('name_normalized', '')}"
            )
            print(
                f"Matching name  : "
                f"{s1_row.get('name_matching', '')}"
            )
            print(
                f"Normalized addr: "
                f"{s1_row.get('address_normalized', '')}"
            )
            print(
                f"Address number : "
                f"{s1_row.get('address_number', '')}"
            )
            print(
                f"Postal code    : "
                f"{s1_row.get('postal_code', '')}"
            )

            print()
            print("S1 BLOCKING KEYS:")

            for key in create_blocking_keys(
                s1_row
            ):
                print(f"  {key}")

        print()

        if matched:

            print("MATCHED SOURCE RECORD")
            print(
                f"Name      : "
                f"{matched['business_name']}"
            )
            print(
                f"Address   : "
                f"{matched['business_address']}"
            )
            print(
                f"Country   : "
                f"{matched['country']}"
            )

            matched_df = pd.DataFrame(
                [matched]
            )

            matched_df = add_normalized_columns(
                matched_df
            )

            matched_row = matched_df.iloc[0]

            print()
            print(
                f"Normalized name: "
                f"{matched_row.get('name_normalized', '')}"
            )
            print(
                f"Matching name  : "
                f"{matched_row.get('name_matching', '')}"
            )
            print(
                f"Normalized addr: "
                f"{matched_row.get('address_normalized', '')}"
            )
            print(
                f"Address number : "
                f"{matched_row.get('address_number', '')}"
            )
            print(
                f"Postal code    : "
                f"{matched_row.get('postal_code', '')}"
            )

            print()
            print("MATCHED RECORD BLOCKING KEYS:")

            for key in create_blocking_keys(
                matched_row
            ):
                print(f"  {key}")


if __name__ == "__main__":
    main()