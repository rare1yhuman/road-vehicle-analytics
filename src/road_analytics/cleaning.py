from __future__ import annotations

import pandas as pd

from .schema import validate_detections


def clean_detections(
    detections: pd.DataFrame,
    *,
    minimum_confidence: float = 0.35,
    minimum_track_frames: int = 5,
    minimum_displacement_px: float = 12.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter weak/short/stationary tracks and return data plus an audit table."""
    data = validate_detections(detections)
    data = data[data["confidence"] >= minimum_confidence].copy()
    audit_columns = [
        "video_id",
        "track_id",
        "class_name",
        "observation_count",
        "unique_frames",
        "mean_confidence",
        "first_x",
        "first_y",
        "last_x",
        "last_y",
        "net_displacement",
        "passes_frame_filter",
        "passes_movement_filter",
        "kept",
    ]
    if data.empty:
        return data.reset_index(drop=True), pd.DataFrame(columns=audit_columns)
    if not data.empty:
        class_scores = (
            data.groupby(["video_id", "track_id", "class_name"], as_index=False)[
                "confidence"
            ]
            .sum()
            .sort_values(
                ["video_id", "track_id", "confidence", "class_name"],
                ascending=[True, True, False, True],
            )
            .drop_duplicates(["video_id", "track_id"])
            .rename(columns={"class_name": "smoothed_class_name"})
        )
        data = data.merge(
            class_scores[["video_id", "track_id", "smoothed_class_name"]],
            on=["video_id", "track_id"],
            how="left",
        )
        data["class_name"] = data.pop("smoothed_class_name")

    grouped = data.groupby(["video_id", "track_id"], sort=False)
    audit = grouped.agg(
        class_name=("class_name", lambda values: values.mode().iat[0]),
        observation_count=("frame_id", "size"),
        unique_frames=("frame_id", "nunique"),
        mean_confidence=("confidence", "mean"),
        first_x=("centroid_x", "first"),
        first_y=("centroid_y", "first"),
        last_x=("centroid_x", "last"),
        last_y=("centroid_y", "last"),
    ).reset_index()
    audit["net_displacement"] = (
        (audit["last_x"] - audit["first_x"]) ** 2
        + (audit["last_y"] - audit["first_y"]) ** 2
    ) ** 0.5
    audit["passes_frame_filter"] = audit["unique_frames"] >= minimum_track_frames
    audit["passes_movement_filter"] = audit["net_displacement"] >= minimum_displacement_px
    audit["kept"] = audit["passes_frame_filter"] & audit["passes_movement_filter"]

    valid = audit.loc[audit["kept"], ["video_id", "track_id"]]
    cleaned = data.merge(valid, on=["video_id", "track_id"], how="inner")
    audit = audit.reindex(columns=audit_columns)
    return (
        cleaned.sort_values(["video_id", "track_id", "frame_id"]).reset_index(drop=True),
        audit,
    )
