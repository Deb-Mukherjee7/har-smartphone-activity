"""Classifiers and a subject-aware stacking ensemble."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder


class LabelEncodedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Adapt XGBoost's integer-label requirement to the dataset's activity strings."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def fit(self, X, y):
        from xgboost import XGBClassifier

        self.label_encoder_ = LabelEncoder().fit(y)
        encoded = self.label_encoder_.transform(y)
        self.model_ = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        self.model_.fit(X, encoded)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        encoded = self.model_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(encoded)


def build_classifier(name: str, random_seed: int = 42):
    """Create a fresh baseline estimator with fixed, reproducible parameters."""
    if name == "logistic_regression":
        return LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
    if name == "svm":
        return SVC(C=10.0, kernel="rbf", probability=True, random_state=random_seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=random_seed,
            n_jobs=-1,
        )
    if name == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("XGBoost is required. Install project requirements first.") from exc
        return LabelEncodedXGBClassifier(random_seed=random_seed)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=350,
            early_stopping=False,
            random_state=random_seed,
        )
    raise ValueError(f"Unknown baseline model: {name}")


class GroupedStackingClassifier(ClassifierMixin, BaseEstimator):
    """RF + XGBoost + logistic regression with GroupKFold out-of-fold meta features.

    `fit` requires a `groups` array. Every row used to create a meta-training
    prediction comes from a subject held out from the corresponding base fit.
    """

    def __init__(self, random_seed: int = 42, n_splits: int = 5):
        self.random_seed = random_seed
        self.n_splits = n_splits

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError("GroupedStackingClassifier.fit requires subject IDs in groups.")
        groups = np.asarray(groups)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        splitter = GroupKFold(n_splits=self.n_splits)
        if len(np.unique(groups)) < self.n_splits:
            raise ValueError("Not enough distinct subjects for the configured GroupKFold splits.")
        base_names = ("random_forest", "xgboost", "logistic_regression")
        out_of_fold = np.zeros((len(y), len(base_names) * len(self.classes_)), dtype=float)

        for train_idx, held_idx in splitter.split(X, y, groups):
            for model_index, model_name in enumerate(base_names):
                estimator = build_classifier(model_name, self.random_seed)
                estimator.fit(X[train_idx], y[train_idx])
                probabilities = estimator.predict_proba(X[held_idx])
                aligned = np.zeros((len(held_idx), len(self.classes_)), dtype=float)
                for source_column, class_name in enumerate(estimator.classes_):
                    destination = np.flatnonzero(self.classes_ == class_name)
                    aligned[:, destination[0]] = probabilities[:, source_column]
                start = model_index * len(self.classes_)
                out_of_fold[held_idx, start : start + len(self.classes_)] = aligned

        self.meta_model_ = LogisticRegression(max_iter=2000, random_state=self.random_seed)
        self.meta_model_.fit(out_of_fold, y)
        self.base_models_ = [
            build_classifier(model_name, self.random_seed).fit(X, y)
            for model_name in base_names
        ]
        return self

    def predict_proba(self, X):
        meta_features = np.hstack([model.predict_proba(X) for model in self.base_models_])
        raw = self.meta_model_.predict_proba(meta_features)
        aligned = np.zeros((len(X), len(self.classes_)), dtype=float)
        for source_column, class_name in enumerate(self.meta_model_.classes_):
            destination = np.flatnonzero(self.classes_ == class_name)
            aligned[:, destination[0]] = raw[:, source_column]
        return aligned

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
