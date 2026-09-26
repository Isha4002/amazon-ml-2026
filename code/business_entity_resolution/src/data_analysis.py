from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset"


def analyze_source_file(file_path, chunk_size=100_000):
    """
    Analyze a large source TSV in chunks.
    """

    total_rows = 0
    missing_name = 0
    missing_address = 0
    missing_country = 0
    country_counts = {}

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        total_rows += len(chunk)

        missing_name += (chunk["business_name"].str.strip() == "").sum()
        missing_address += (chunk["business_address"].str.strip() == "").sum()
        missing_country += (chunk["country"].str.strip() == "").sum()

        counts = chunk["country"].value_counts()

        for country, count in counts.items():
            country_counts[country] = (
                country_counts.get(country, 0) + int(count)
            )

    return {
        "rows": total_rows,
        "missing_name": int(missing_name),
        "missing_address": int(missing_address),
        "missing_country": int(missing_country),
        "countries": country_counts,
    }


def analyze_ground_truth(file_path, chunk_size=100_000):
    """
    Analyze the number of true matches for each Source 1 entity.
    """

    total_rows = 0
    singleton_count = 0
    no_match_count = 0
    match_distribution = {}

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        total_rows += len(chunk)

        for value in chunk["matched_entity_ids"]:
            value = value.strip()

            if not value:
                match_count = 0
            else:
                match_count = len(
                    [x for x in value.split(",") if x.strip()]
                )

            match_distribution[match_count] = (
                match_distribution.get(match_count, 0) + 1
            )

            if match_count == 0:
                no_match_count += 1

            elif match_count == 1:
                singleton_count += 1

    return {
        "rows": total_rows,
        "no_match": no_match_count,
        "exactly_one_match": singleton_count,
        "match_distribution": match_distribution,
    }


def print_source_summary(name, result):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Rows: {result['rows']:,}")

    print(f"Missing business names: {result['missing_name']:,}")

    print(
        f"Missing addresses: "
        f"{result['missing_address']:,}"
    )

    print(
        f"Missing countries: "
        f"{result['missing_country']:,}"
    )

    print("\nCountry distribution:")

    for country, count in sorted(
        result["countries"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        percentage = count / result["rows"] * 100

        print(
            f"  {country or '[EMPTY]':15s} "
            f"{count:>12,} "
            f"({percentage:6.2f}%)"
        )


def print_ground_truth_summary(result):
    print()
    print("=" * 70)
    print("GROUND TRUTH")
    print("=" * 70)

    print(f"Source 1 entities: {result['rows']:,}")

    print(
        f"No matches / singletons: "
        f"{result['no_match']:,}"
    )

    print(
        f"Exactly one match: "
        f"{result['exactly_one_match']:,}"
    )

    print("\nNumber of true matches per Source 1 entity:")

    for match_count, entity_count in sorted(
        result["match_distribution"].items()
    ):
        print(
            f"  {match_count:>4} matches: "
            f"{entity_count:>12,} entities"
        )


def main():
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"

    print("BUSINESS ENTITY RESOLUTION - DATASET ANALYSIS")
    print("=" * 70)

    print("\nAnalyzing training Source 1...")
    s1_train = analyze_source_file(
        train_dir / "train_source1.tsv"
    )
    print_source_summary(
        "TRAIN SOURCE 1",
        s1_train,
    )

    print("\nAnalyzing training Source 2...")
    s2_train = analyze_source_file(
        train_dir / "train_source2.tsv"
    )
    print_source_summary(
        "TRAIN SOURCE 2",
        s2_train,
    )

    print("\nAnalyzing training Source 3...")
    s3_train = analyze_source_file(
        train_dir / "train_source3.tsv"
    )
    print_source_summary(
        "TRAIN SOURCE 3",
        s3_train,
    )

    print("\nAnalyzing test Source 1...")
    s1_test = analyze_source_file(
        test_dir / "test_source1.tsv"
    )
    print_source_summary(
        "TEST SOURCE 1",
        s1_test,
    )

    print("\nAnalyzing test Source 2...")
    s2_test = analyze_source_file(
        test_dir / "test_source2.tsv"
    )
    print_source_summary(
        "TEST SOURCE 2",
        s2_test,
    )

    print("\nAnalyzing test Source 3...")
    s3_test = analyze_source_file(
        test_dir / "test_source3.tsv"
    )
    print_source_summary(
        "TEST SOURCE 3",
        s3_test,
    )

    print("\nAnalyzing ground truth...")
    ground_truth = analyze_ground_truth(
        train_dir / "train_ground_truth.tsv"
    )
    print_ground_truth_summary(
        ground_truth
    )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()