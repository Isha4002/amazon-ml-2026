import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


class EntityMatchingModel:
    """
    Machine-learning model for business entity matching.

    The model predicts whether a candidate pair is:
        1 -> Match
        0 -> Non-match
    """

    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = 42,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

        self.feature_columns = None
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Train the matching model.

        Parameters
        ----------
        X:
            Numerical feature matrix.

        y:
            Binary labels:
                1 = true match
                0 = non-match
        """

        if X.empty:
            raise ValueError("Training feature matrix is empty.")

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of rows."
            )

        self.feature_columns = list(X.columns)

        self.model.fit(X, y)

        self.is_fitted = True

        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return probability of being a true match.
        """

        self._check_fitted()

        X = self._prepare_features(X)

        # Probability of class 1 = match probability
        probabilities = self.model.predict_proba(X)

        class_index = list(
            self.model.classes_
        ).index(1)

        return probabilities[:, class_index]

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Convert match probabilities into binary predictions.

        A pair is predicted as a match when:

            probability >= threshold
        """

        probabilities = self.predict_proba(X)

        return (
            probabilities >= threshold
        ).astype(int)

    def feature_importance(
        self,
    ) -> pd.DataFrame:
        """
        Return feature importance values.
        """

        self._check_fitted()

        importance = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": self.model.feature_importances_,
            }
        )

        return importance.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

    def _prepare_features(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Make sure prediction features have the same
        columns and ordering as training features.
        """

        missing_columns = set(
            self.feature_columns
        ) - set(X.columns)

        if missing_columns:
            raise ValueError(
                f"Missing feature columns: {missing_columns}"
            )

        return X[self.feature_columns]

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been trained yet. "
                "Call fit() first."
            )