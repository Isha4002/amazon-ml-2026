from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = PROJECT_ROOT / "output"

FEATURES_PATH = OUTPUT_DIR / "training_features.tsv"
MODEL_PATH = OUTPUT_DIR / "matcher_model.joblib"
THRESHOLD_PATH = OUTPUT_DIR / "best_threshold.txt"

RANDOM_STATE = 42


# ============================================================
# CHALLENGE METRIC
# ============================================================

def fbeta_score_custom(precision, recall, beta=0.5):

    if precision + recall == 0:
        return 0.0

    beta_squared = beta ** 2

    return (
        (1 + beta_squared)
        * precision
        * recall
        / (
            beta_squared * precision
            + recall
        )
    )


def calculate_macro_f05(
    validation_df,
    predictions
):

    temp = validation_df[
        ["s1_id", "candidate_id", "label"]
    ].copy()

    temp["prediction"] = predictions

    s1_scores = []

    for s1_id, group in temp.groupby("s1_id"):

        true_ids = set(
            group.loc[
                group["label"] == 1,
                "candidate_id"
            ]
        )

        predicted_ids = set(
            group.loc[
                group["prediction"] == 1,
                "candidate_id"
            ]
        )

        true_positive = len(
            true_ids & predicted_ids
        )

        false_positive = len(
            predicted_ids - true_ids
        )

        false_negative = len(
            true_ids - predicted_ids
        )

        if (
            true_positive == 0
            and false_positive == 0
            and false_negative == 0
        ):
            score = 1.0

        elif true_positive == 0:

            score = 0.0

        else:

            precision = (
                true_positive
                / (
                    true_positive
                    + false_positive
                )
                if (
                    true_positive
                    + false_positive
                ) > 0
                else 0.0
            )

            recall = (
                true_positive
                / (
                    true_positive
                    + false_negative
                )
                if (
                    true_positive
                    + false_negative
                ) > 0
                else 0.0
            )

            score = fbeta_score_custom(
                precision,
                recall,
                beta=0.5
            )

        s1_scores.append(score)

    if not s1_scores:
        return 0.0

    return float(
        np.mean(s1_scores)
    )


# ============================================================
# S1-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

def split_by_s1(df):

    unique_s1 = (
        df["s1_id"]
        .drop_duplicates()
        .sample(
            frac=1.0,
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )

    split_point = int(
        len(unique_s1) * 0.8
    )

    train_s1_ids = set(
        unique_s1.iloc[:split_point]
    )

    validation_s1_ids = set(
        unique_s1.iloc[split_point:]
    )

    train_df = df[
        df["s1_id"].isin(train_s1_ids)
    ].copy()

    validation_df = df[
        df["s1_id"].isin(validation_s1_ids)
    ].copy()

    return train_df, validation_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TRAINING ENTITY MATCHER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    if not FEATURES_PATH.exists():

        raise FileNotFoundError(
            f"\nFeature file not found:\n"
            f"{FEATURES_PATH}\n\n"
            f"Run build_training_features.py first."
        )

    print("\nLoading training features...")

    df = pd.read_csv(
        FEATURES_PATH,
        sep="\t",
        dtype={
            "s1_id": str,
            "candidate_id": str,
            "source": str,
            "label": int,
        }
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Positive: "
        f"{(df['label'] == 1).sum():,}"
    )

    print(
        f"Negative: "
        f"{(df['label'] == 0).sum():,}"
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    metadata_columns = [
        "s1_id",
        "candidate_id",
        "source",
        "label",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns
    ]

    print(
        f"\nNumber of features: "
        f"{len(feature_columns)}"
    )

    print("\nFeatures:")

    for column in feature_columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    X = df[feature_columns].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    y = df["label"].astype(int)

    # --------------------------------------------------------
    # Split by S1
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLITTING BY S1 ENTITY")
    print("=" * 70)

    train_df, validation_df = split_by_s1(df)

    print(
        f"Training S1 entities: "
        f"{train_df['s1_id'].nunique():,}"
    )

    print(
        f"Validation S1 entities: "
        f"{validation_df['s1_id'].nunique():,}"
    )

    print(
        f"Training pairs: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation pairs: "
        f"{len(validation_df):,}"
    )

    # --------------------------------------------------------
    # Training matrices
    # --------------------------------------------------------

    X_train = (
        train_df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    y_train = train_df["label"].astype(int)

    X_validation = (
        validation_df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Model training complete.")

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    print("\nGenerating validation probabilities...")

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # Search threshold
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEARCHING FOR BEST F0.5 THRESHOLD")
    print("=" * 70)

    best_threshold = 0.50
    best_f05 = -1.0

    threshold_results = []

    thresholds = np.arange(
        0.05,
        0.991,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        macro_f05 = calculate_macro_f05(
            validation_df,
            predictions
        )

        precision = precision_score(
            validation_df["label"],
            predictions,
            zero_division=0
        )

        recall = recall_score(
            validation_df["label"],
            predictions,
            zero_division=0
        )

        threshold_results.append(
            (
                threshold,
                macro_f05,
                precision,
                recall
            )
        )

        if macro_f05 > best_f05:

            best_f05 = macro_f05
            best_threshold = threshold

    # --------------------------------------------------------
    # Best threshold
    # --------------------------------------------------------

    print("\nBest threshold:")
    print(
        f"Threshold: {best_threshold:.2f}"
    )

    print(
        f"Macro F0.5: {best_f05:.6f}"
    )

    # Find corresponding precision/recall
    best_result = min(
        threshold_results,
        key=lambda x: abs(
            x[0] - best_threshold
        )
    )

    print(
        f"Pair precision: "
        f"{best_result[2]:.6f}"
    )

    print(
        f"Pair recall: "
        f"{best_result[3]:.6f}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    for _, row in importance.iterrows():

        print(
            f"{row['feature']:<35} "
            f"{row['importance']:.6f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Save threshold
    # --------------------------------------------------------

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"{best_threshold:.6f}\n"
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL READY")
    print("=" * 70)

    print(
        f"Model saved to:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"\nThreshold saved to:\n"
        f"{THRESHOLD_PATH}"
    )

    print(
        f"\nBest validation Macro F0.5: "
        f"{best_f05:.6f}"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()