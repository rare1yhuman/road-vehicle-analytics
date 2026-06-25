from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METERS_PER_PIXEL = 0.05

MATERIAL = {
    "blue": "#2196F3",
    "red": "#F44336",
    "green": "#4CAF50",
    "amber": "#FFC107",
    "purple": "#9C27B0",
    "cyan": "#00BCD4",
    "deep_orange": "#FF5722",
    "indigo": "#3F51B5",
    "teal": "#009688",
    "light_green": "#8BC34A",
    "background": "#FAFAFA",
    "grid": "#EEEEEE",
    "text": "#212121",
    "subtext": "#757575",
    "dark_background": "#212121",
    "dark_card": "#303030",
    "dark_text": "#FAFAFA",
}

VEHICLE_TYPE_COLORS = {
    "car": MATERIAL["blue"],
    "truck": MATERIAL["red"],
    "motorcycle": MATERIAL["green"],
    "bus": MATERIAL["amber"],
    "bicycle": MATERIAL["purple"],
    "van": MATERIAL["cyan"],
    "auto": MATERIAL["deep_orange"],
    "other": MATERIAL["indigo"],
}


def color_for_vehicle(class_name: str) -> str:
    return VEHICLE_TYPE_COLORS.get(str(class_name).lower(), MATERIAL["indigo"])


def ensure_road_segment(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        if "road_segment" not in data.columns:
            data["road_segment"] = pd.Series(dtype="object")
        return data
    if "road_segment" not in data.columns:
        data["road_segment"] = data["video_id"].astype(str)
    else:
        data["road_segment"] = data["road_segment"].fillna(data["video_id"]).astype(str)
    return data


def add_speed_columns(
    track_features: pd.DataFrame,
    *,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
) -> pd.DataFrame:
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive")
    data = ensure_road_segment(track_features)
    if data.empty:
        for column in ["avg_speed_px_s", "avg_speed_kmh"]:
            if column not in data.columns:
                data[column] = pd.Series(dtype="float64")
        return data

    if "avg_speed_px_s" not in data.columns:
        if {"path_length", "duration_s"}.issubset(data.columns):
            duration = pd.to_numeric(data["duration_s"], errors="coerce").fillna(0)
            path = pd.to_numeric(data["path_length"], errors="coerce").fillna(0)
            data["avg_speed_px_s"] = np.where(duration > 0, path / duration, 0.0)
        else:
            data["avg_speed_px_s"] = 0.0
    data["avg_speed_px_s"] = pd.to_numeric(data["avg_speed_px_s"], errors="coerce").fillna(0)
    data["avg_speed_kmh"] = data["avg_speed_px_s"] * meters_per_pixel * 3.6
    return data


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_analysis_frames(analysis_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(analysis_dir)
    return {
        "detections": ensure_road_segment(read_csv_if_exists(root / "cleaned_detections.csv")),
        "tracks": add_speed_columns(read_csv_if_exists(root / "track_features.csv")),
        "counts": read_csv_if_exists(root / "vehicle_counts.csv"),
        "events": read_csv_if_exists(root / "crossing_events.csv"),
    }


def filter_analysis_frames(
    frames: dict[str, pd.DataFrame],
    *,
    road_segments: list[str] | None = None,
    vehicle_types: list[str] | None = None,
    time_range: tuple[float, float] | None = None,
) -> dict[str, pd.DataFrame]:
    road_segments = road_segments or []
    vehicle_types = vehicle_types or []
    filtered: dict[str, pd.DataFrame] = {}

    for key, frame in frames.items():
        data = ensure_road_segment(frame) if key != "counts" else frame.copy()
        if data.empty:
            filtered[key] = data
            continue
        if road_segments:
            segment_column = "road_segment" if "road_segment" in data.columns else "video_id"
            data = data[data[segment_column].astype(str).isin(road_segments)]
        if vehicle_types and "class_name" in data.columns:
            data = data[data["class_name"].astype(str).str.lower().isin(vehicle_types)]
        if time_range:
            time_column = "timestamp_s" if "timestamp_s" in data.columns else "crossing_timestamp_s"
            if time_column in data.columns:
                data = data[
                    (pd.to_numeric(data[time_column], errors="coerce") >= time_range[0])
                    & (pd.to_numeric(data[time_column], errors="coerce") <= time_range[1])
                ]
        filtered[key] = data.copy()
    return filtered


def prepare_traffic_volume(
    detections: pd.DataFrame,
    events: pd.DataFrame,
    *,
    time_bin_s: int = 5,
) -> pd.DataFrame:
    if time_bin_s < 1:
        raise ValueError("time_bin_s must be at least 1")
    if not events.empty and "crossing_timestamp_s" in events.columns:
        data = ensure_road_segment(events)
        data["time_s"] = pd.to_numeric(data["crossing_timestamp_s"], errors="coerce")
        data = data.dropna(subset=["time_s"])
        data["time_bin_s"] = (data["time_s"] // time_bin_s * time_bin_s).astype(int)
        return (
            data.groupby(["road_segment", "class_name", "time_bin_s"], as_index=False)
            .size()
            .rename(columns={"size": "vehicle_count"})
            .sort_values("time_bin_s")
        )

    if detections.empty:
        return pd.DataFrame(
            columns=["road_segment", "class_name", "time_bin_s", "vehicle_count"]
        )
    data = ensure_road_segment(detections)
    data["time_s"] = pd.to_numeric(data["timestamp_s"], errors="coerce")
    data = data.dropna(subset=["time_s"])
    data["time_bin_s"] = (data["time_s"] // time_bin_s * time_bin_s).astype(int)
    return (
        data.groupby(["road_segment", "class_name", "time_bin_s"])["track_id"]
        .nunique()
        .rename("vehicle_count")
        .reset_index()
        .sort_values("time_bin_s")
    )


def prepare_vehicle_breakdown(
    counts: pd.DataFrame,
    track_features: pd.DataFrame,
) -> pd.DataFrame:
    if not counts.empty and {"class_name", "vehicle_count"}.issubset(counts.columns):
        return (
            counts.groupby("class_name", as_index=False)["vehicle_count"]
            .sum()
            .sort_values("vehicle_count", ascending=False)
        )
    tracks = add_speed_columns(track_features)
    if tracks.empty:
        return pd.DataFrame(columns=["class_name", "vehicle_count"])
    return (
        tracks.groupby("class_name")["track_id"]
        .nunique()
        .rename("vehicle_count")
        .reset_index()
        .sort_values("vehicle_count", ascending=False)
    )


def prepare_speed_volume(
    detections: pd.DataFrame,
    track_features: pd.DataFrame,
    *,
    time_bin_s: int = 5,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
) -> pd.DataFrame:
    if detections.empty or track_features.empty:
        return pd.DataFrame(
            columns=["road_segment", "time_bin_s", "vehicle_count", "avg_speed_kmh"]
        )
    detections = ensure_road_segment(detections)
    tracks = add_speed_columns(track_features, meters_per_pixel=meters_per_pixel)
    speed_lookup = tracks[["video_id", "track_id", "avg_speed_kmh"]].drop_duplicates()
    data = detections.merge(speed_lookup, on=["video_id", "track_id"], how="left")
    data["time_s"] = pd.to_numeric(data["timestamp_s"], errors="coerce")
    data = data.dropna(subset=["time_s"])
    data["time_bin_s"] = (data["time_s"] // time_bin_s * time_bin_s).astype(int)
    return (
        data.groupby(["road_segment", "time_bin_s"])
        .agg(vehicle_count=("track_id", "nunique"), avg_speed_kmh=("avg_speed_kmh", "mean"))
        .reset_index()
        .sort_values("time_bin_s")
    )


def prepare_speed_distribution(
    track_features: pd.DataFrame,
    *,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
    group_by: str = "road_segment",
) -> pd.DataFrame:
    tracks = add_speed_columns(track_features, meters_per_pixel=meters_per_pixel)
    if tracks.empty:
        return pd.DataFrame(columns=[group_by, "class_name", "avg_speed_kmh"])
    if group_by not in tracks.columns:
        group_by = "road_segment"
    return tracks[[group_by, "class_name", "avg_speed_kmh", "avg_speed_px_s"]].copy()


def prepare_trajectory_points(
    detections: pd.DataFrame,
    track_features: pd.DataFrame,
    *,
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL,
) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame()
    data = ensure_road_segment(detections)
    tracks = add_speed_columns(track_features, meters_per_pixel=meters_per_pixel)
    if not tracks.empty:
        data = data.merge(
            tracks[["video_id", "track_id", "avg_speed_kmh", "avg_speed_px_s"]],
            on=["video_id", "track_id"],
            how="left",
        )
    else:
        data["avg_speed_kmh"] = 0.0
        data["avg_speed_px_s"] = 0.0
    return data.sort_values(["video_id", "track_id", "frame_id"])
