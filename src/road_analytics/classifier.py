from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import ROAD_FEATURE_COLUMNS


@dataclass
class RoadTypeClassifier:
    """Small, dependency-free logistic regression with standardized inputs."""

    regularization: float = 0.01
    learning_rate: float = 0.1
    iterations: int = 2500
    feature_names: list[str] | None = None
    means_: np.ndarray | None = None
    scales_: np.ndarray | None = None
    weights_: np.ndarray | None = None
    bias_: float = 0.0

    def fit(self, features: pd.DataFrame, labels: pd.Series | np.ndarray) -> "RoadTypeClassifier":
        self.feature_names = self.feature_names or ROAD_FEATURE_COLUMNS.copy()
        x = features[self.feature_names].to_numpy(dtype=float)
        y = np.asarray(labels, dtype=float)
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("At least two labeled samples are required")
        if len(np.unique(y)) < 2:
            raise ValueError("Training labels must contain both road types")

        self.means_ = x.mean(axis=0)
        self.scales_ = x.std(axis=0)
        self.scales_[self.scales_ == 0] = 1.0
        x_scaled = (x - self.means_) / self.scales_
        self.weights_ = np.zeros(x_scaled.shape[1], dtype=float)
        self.bias_ = 0.0

        for _ in range(self.iterations):
            scores = np.clip(x_scaled @ self.weights_ + self.bias_, -30, 30)
            probabilities = 1 / (1 + np.exp(-scores))
            error = probabilities - y
            weight_gradient = (
                x_scaled.T @ error / len(y) + self.regularization * self.weights_
            )
            bias_gradient = float(error.mean())
            self.weights_ -= self.learning_rate * weight_gradient
            self.bias_ -= self.learning_rate * bias_gradient
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.weights_ is None or self.means_ is None or self.scales_ is None:
            raise RuntimeError("Classifier has not been fitted")
        x = features[self.feature_names].to_numpy(dtype=float)
        scores = np.clip(((x - self.means_) / self.scales_) @ self.weights_ + self.bias_, -30, 30)
        return 1 / (1 + np.exp(-scores))

    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(features) >= threshold).astype(int)

    def coefficients(self) -> pd.DataFrame:
        if self.weights_ is None:
            raise RuntimeError("Classifier has not been fitted")
        return pd.DataFrame(
            {"feature": self.feature_names, "coefficient": self.weights_}
        ).sort_values("coefficient", key=np.abs, ascending=False)

    def to_dict(self) -> dict[str, object]:
        if self.weights_ is None:
            raise RuntimeError("Classifier has not been fitted")
        return {
            "regularization": self.regularization,
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "feature_names": self.feature_names,
            "means": self.means_.tolist(),
            "scales": self.scales_.tolist(),
            "weights": self.weights_.tolist(),
            "bias": self.bias_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RoadTypeClassifier":
        model = cls(
            regularization=float(payload["regularization"]),
            learning_rate=float(payload["learning_rate"]),
            iterations=int(payload["iterations"]),
            feature_names=list(payload["feature_names"]),
        )
        model.means_ = np.asarray(payload["means"], dtype=float)
        model.scales_ = np.asarray(payload["scales"], dtype=float)
        model.weights_ = np.asarray(payload["weights"], dtype=float)
        model.bias_ = float(payload["bias"])
        return model


def leave_one_out_evaluate(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    regularization: float = 0.01,
    learning_rate: float = 0.1,
    iterations: int = 2500,
) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    labels = labels.reset_index(drop=True)
    features = features.reset_index(drop=True)
    for index in range(len(features)):
        train_mask = np.arange(len(features)) != index
        y_train = labels.loc[train_mask]
        if y_train.nunique() < 2:
            probability = float(labels.loc[train_mask].mean())
        else:
            model = RoadTypeClassifier(
                regularization=regularization,
                learning_rate=learning_rate,
                iterations=iterations,
            ).fit(features.loc[train_mask], y_train)
            probability = float(model.predict_proba(features.loc[[index]])[0])
        actual = int(labels.iloc[index])
        results.append(
            {
                "row": index,
                "actual": actual,
                "predicted": int(probability >= 0.5),
                "intersection_probability": probability,
            }
        )
    return pd.DataFrame(results)

