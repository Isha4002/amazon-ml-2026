from pathlib import Path

from .data_loader import load_training_data, load_test_data
from .preprocessing import add_normalized_columns
from .blocking import generate_candidate_pairs
from .features import build_feature_matrix


def prepare_training_data(dataset_dir):
    """
    Load and preprocess all training datasets.
    """

    source1, source2, source3, ground_truth = load_training_data(
        dataset_dir
    )

    source1 = add_normalized_columns(source1)
    source2 = add_normalized_columns(source2)
    source3 = add_normalized_columns(source3)

    return source1, source2, source3, ground_truth


def prepare_test_data(dataset_dir):
    """
    Load and preprocess all test datasets.
    """

    source1, source2, source3 = load_test_data(
        dataset_dir
    )

    source1 = add_normalized_columns(source1)
    source2 = add_normalized_columns(source2)
    source3 = add_normalized_columns(source3)

    return source1, source2, source3


def generate_features(
    source1,
    source2,
    source3,
):
    """
    Generate blocking candidates and pairwise features.
    """

    candidate_pairs = generate_candidate_pairs(
        source1,
        source2,
        source3,
    )

    X, metadata = build_feature_matrix(
        source1,
        source2,
        source3,
        candidate_pairs,
    )

    return X, metadata, candidate_pairs


def run_training_pipeline(dataset_dir):
    """
    Prepare training data and generate candidate-pair features.

    The actual ground-truth-to-feature-label mapping will be added
    after the real train_ground_truth.tsv format is inspected.
    """

    source1, source2, source3, ground_truth = (
        prepare_training_data(dataset_dir)
    )

    X, metadata, candidate_pairs = generate_features(
        source1,
        source2,
        source3,
    )

    return {
        "source1": source1,
        "source2": source2,
        "source3": source3,
        "ground_truth": ground_truth,
        "features": X,
        "metadata": metadata,
        "candidate_pairs": candidate_pairs,
    }


def run_test_pipeline(dataset_dir):
    """
    Prepare test data and generate candidate-pair features.
    """

    source1, source2, source3 = prepare_test_data(
        dataset_dir
    )

    X, metadata, candidate_pairs = generate_features(
        source1,
        source2,
        source3,
    )

    return {
        "source1": source1,
        "source2": source2,
        "source3": source3,
        "features": X,
        "metadata": metadata,
        "candidate_pairs": candidate_pairs,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    dataset_dir = project_root / "dataset"

    print("Business Entity Resolution Pipeline")
    print("-----------------------------------")
    print(f"Dataset directory: {dataset_dir}")

    if not dataset_dir.exists():
        print("Dataset directory does not exist yet.")
        print("Waiting for the challenge dataset.")
    else:
        print("Dataset directory found.")
        print("Training/test files will be loaded when available.")