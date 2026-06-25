from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning import clean_detections
from .counting import count_line_crossings


def classification_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    y_true = np.asarray(actual, dtype=int)
    y_pred = np.asarray(predicted, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate_counts(predicted: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    required = {"video_id", "class_name", "vehicle_count"}
    for name, frame in [("predicted", predicted), ("ground_truth", ground_truth)]:
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} counts must contain {sorted(required)}")
    merged = ground_truth.merge(
        predicted,
        on=["video_id", "class_name"],
        how="outer",
        suffixes=("_actual", "_predicted"),
    ).fillna(0)
    merged["absolute_error"] = (
        merged["vehicle_count_predicted"] - merged["vehicle_count_actual"]
    ).abs()
    merged["percent_error"] = np.where(
        merged["vehicle_count_actual"] > 0,
        merged["absolute_error"] / merged["vehicle_count_actual"] * 100,
        np.where(merged["vehicle_count_predicted"] == 0, 0.0, 100.0),
    )
    return merged


def confidence_sensitivity(
    detections: pd.DataFrame,
    ground_truth: pd.DataFrame,
    thresholds: list[float],
    *,
    minimum_track_frames: int = 5,
    minimum_displacement_px: float = 12.0,
    counting_options: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        cleaned, audit = clean_detections(
            detections,
            minimum_confidence=threshold,
            minimum_track_frames=minimum_track_frames,
            minimum_displacement_px=minimum_displacement_px,
        )
        _, counts = count_line_crossings(cleaned, **(counting_options or {}))
        errors = evaluate_counts(counts, ground_truth)
        rows.append(
            {
                "confidence_threshold": threshold,
                "mean_absolute_error": float(errors["absolute_error"].mean()),
                "mean_percent_error": float(errors["percent_error"].mean()),
                "tracks_kept": int(audit["kept"].sum()),
            }
        )
    return pd.DataFrame(rows)

