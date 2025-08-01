from warnings import warn

import numpy as np
import pandas as pd


class DuckVotingClassifier:
    """
    Sklearn-like voting classifier that only requires sub-classifiers implement a sklearn-like API,
    specifically, the predict and predict_proba methods and the classes_ attribute.
    """

    # Hyperparameters initialized in the constructor
    estimators: list[tuple]
    voting: str

    # Initialized to None in constructor, set during fit(), used during predict_proba()/predict()
    classes_: np.ndarray | None = None

    def __init__(self, estimators: list[tuple], voting: str = 'hard', n_jobs: int | None = None):
        if n_jobs is not None and n_jobs > 1:
            warn('DuckVotingClassifier does not support parallel execution; n_jobs will be ignored.')

        self.estimators = estimators
        self.voting = voting
        self.classes_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, *args, **kwargs) -> None:
        self.classes_ = np.unique(y)  # np.unique also sorts the classes
        for (_, estimator) in self.estimators:
            estimator.fit(X, y, *args, **kwargs)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate multi-class prediction probabilities.

        Args:
            X_test: Test features

        Returns:
            Array of predicted label probabilities, shape (n_samples, n_classes), where n_classes and
            the order of classes correspond to self.classes_ .
        """
        if self.classes_ is None:
            raise Exception('Classifier has not been fit yet. Call fit() first.')

        if self.voting == 'hard':
            # Compute scores as sum of votes for each class
            sub_preds = np.stack([
                estimator.predict(X)
                for (_, estimator) in self.estimators
            ])  # n_estimators x n_samples
            # pred_is_class: n_estimators x n_samples x n_classes, bool
            pred_is_class = np.expand_dims(sub_preds, axis=-1) == self.classes_
            class_scores = pred_is_class.sum(axis=0)  # n_samples x n_classes, int (number of votes)

        elif self.voting == 'soft':
            # Compute scores as sum of sub-estimator probabilities for each class
            sub_probs = np.stack([
                estimator.predict_proba(X)[..., [estimator.classes_.tolist().index(c) for c in self.classes_]]
                for (_, estimator) in self.estimators
            ])  # n_estimators x n_samples x n_classes
            class_scores = sub_probs.sum(axis=0)  # n_samples x n_classes, float (sum of sub probabilities)
        else:
            raise ValueError(f'Unrecognized voting strategy {self.voting}')

        # Compute probabilities as normalized class scores
        probs = class_scores / class_scores.sum(axis=1, keepdims=True)  # n_samples x n_classes, float (probabilities)
        return probs

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate multi-class predictions.

        Args:
            X_test: Test features

        Returns:
            Array of predicted labels.
        """
        if self.classes_ is None:
            raise Exception('Classifier has not been fit yet. Call fit() first.')

        probs = self.predict_proba(X)
        return np.array([self.classes_[i] for i in probs.argmax(axis=1)])
