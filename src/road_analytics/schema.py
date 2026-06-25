from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DETECTION_COLUMNS = [
    "video_id",
    "frame_id",
    "track_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "centroid_x",
    "centroid_y",
    "timestamp_s",
    "frame_width",
    "frame_height",
]

NUMERIC_COLUMNS = [
    "frame_id",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "centroid_x",
    "centroid_y",
    "timestamp_s",
    "frame_width",
    "frame_height",
]


class DataValidationError(ValueError):
    pass


def validate_detections(df: pd.DataFrame, *, allow_empty: bool = False) -> pd.DataFrame:
    missing = [column for column in DETECTION_COLUMNS if column not in df.columns]
    if missing:
        raise DataValidationError(f"Missing detection columns: {', '.join(missing)}")
    if df.empty and not allow_empty:
        raise DataValidationError("Detection dataset is empty")

    clean = df.copy()
    for column in NUMERIC_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    if clean[NUMERIC_COLUMNS].isna().any().any():
        bad = clean[NUMERIC_COLUMNS].columns[clean[NUMERIC_COLUMNS].isna().any()].tolist()
        raise DataValidationError(f"Non-numeric or missing values in: {', '.join(bad)}")
    if ((clean["confidence"] < 0) | (clean["confidence"] > 1)).any():
        raise DataValidationError("Confidence values must be between 0 and 1")
    if (clean[["frame_width", "frame_height"]] <= 0).any().any():
        raise DataValidationError("Frame dimensions must be positive")

    clean["video_id"] = clean["video_id"].astype(str)
    clean["class_name"] = clean["class_name"].astype(str).str.lower()
    clean["frame_id"] = clean["frame_id"].astype(int)
    clean["track_id"] = clean["track_id"].astype(int)
    return clean.sort_values(["video_id", "track_id", "frame_id"]).reset_index(drop=True)


def read_detections(path: str | Path) -> pd.DataFrame:
    return validate_detections(pd.read_csv(path))


@dataclass(frozen=True)
class VideoMetadata:
    video_id: str
    path: str
    road_type: str | None = None
    location: str | None = None
    time_of_day: str | None = None
    weather: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "VideoMetadata":
        return cls(**{key: value.get(key) for key in cls.__dataclass_fields__})

