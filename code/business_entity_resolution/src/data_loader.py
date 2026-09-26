
from pathlib import Path
from typing import Optional

import pandas as pd


def load_tsv(file_path: str | Path) -> pd.DataFrame:
    """
    Load a tab-separated (.tsv) file into a pandas DataFrame.

    Parameters
    ----------
    file_path:
        Path to the TSV file.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".tsv":
        raise ValueError(f"Expected a .tsv file, got: {file_path}")

    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)

    return df


def load_training_data(dataset_dir: str | Path):
    """
    Load all training files when they are available.

    Expected structure:

    dataset/
        train/
            train_source1.tsv
            train_source2.tsv
            train_source3.tsv
            train_ground_truth.tsv
    """
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"

    source1 = load_tsv(train_dir / "train_source1.tsv")
    source2 = load_tsv(train_dir / "train_source2.tsv")
    source3 = load_tsv(train_dir / "train_source3.tsv")
    ground_truth = load_tsv(train_dir / "train_ground_truth.tsv")

    return source1, source2, source3, ground_truth


def load_test_data(dataset_dir: str | Path):
    """
    Load all test files when they are available.

    Expected structure:

    dataset/
        test/
            test_source1.tsv
            test_source2.tsv
            test_source3.tsv
    """
    dataset_dir = Path(dataset_dir)
    test_dir = dataset_dir / "test"

    source1 = load_tsv(test_dir / "test_source1.tsv")
    source2 = load_tsv(test_dir / "test_source2.tsv")
    source3 = load_tsv(test_dir / "test_source3.tsv")

    return source1, source2, source3