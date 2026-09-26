import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score


def calculate_f05(
    y_true,
    y_pred,
) -> float:
    """
    Calculate F0.5 score.

    F0.5 gives more importance to precision than recall.
    """

    return fbeta_score(
        y_true,
        y_pred,
        beta=0.5,
        zero_division=0,
    )


def calculate_precision(
    y_true,
    y_pred,
) -> float:
    """Calculate precision."""

    return precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )


def calculate_recall(
    y_true,
    y_pred,
) -> float:
    """Calculate recall."""

    return recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )


def evaluate_thresholds(
    y_true,
    probabilities,
    thresholds=None,
):
    """
    Evaluate multiple probability thresholds.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.

    probabilities:
        Model probability for the pair being a match.

    thresholds:
        Thresholds to evaluate.

    Returns
    -------
    list[dict]
        Metrics for every threshold.
    """

    if thresholds is None:
        thresholds = np.arange(
            0.50,
            1.00,
            0.05,
        )

    results = []

    for threshold in thresholds:

        predictions = (
            np.asarray(probabilities) >= threshold
        ).astype(int)

        precision = calculate_precision(
            y_true,
            predictions,
        )

        recall = calculate_recall(
            y_true,
            predictions,
        )

        f05 = calculate_f05(
            y_true,
            predictions,
        )

        results.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f0.5": float(f05),
            }
        )

    return results


def find_best_threshold(
    y_true,
    probabilities,
    thresholds=None,
):
    """
    Find the threshold producing the highest F0.5 score.

    Returns
    -------
    dict
        Best threshold and its metrics.
    """

    results = evaluate_thresholds(
        y_true,
        probabilities,
        thresholds,
    )

    best_result = max(
        results,
        key=lambda result: result["f0.5"],
    )

    return best_result