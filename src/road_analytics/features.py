from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .schema import validate_detections


ROAD_FEATURE_COLUMNS = [
    "angle_variance",
    "direction_entropy",
    "unique_entry_zones",
    "angle_deviation_proportion",
    "mean_track_displacement",
    "vehicle_count",
]
TRACK_FEATURE_COLUMNS = [
    "video_id",
    "road_segment",
    "track_id",
    "class_name",
    "first_frame",
    "last_frame",
    "track_duration_frames",
    "duration_s",
    "mean_confidence",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "movement_angle",
    "net_displacement",
    "path_length",
    "avg_speed_px_s",
    "avg_speed_kmh",
    "straightness",
    "entry_zone",
]
VIDEO_FEATURE_COLUMNS = [
    "video_id",
    *ROAD_FEATURE_COLUMNS,
    "mean_avg_speed_kmh",
    "dominant_angle",
]


def _circular_mean_deg(angles: np.ndarray) -> float:
    radians = np.deg2rad(angles)
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))


def _circular_deviation_deg(angles: np.ndarray, center: float) -> np.ndarray:
    return np.abs((angles - center + 180) % 360 - 180)


def _entry_zone(x: float, y: float, width: float, height: float, margin: float) -> str:
    left, right = width * margin, width * (1 - margin)
    top, bottom = height * margin, height * (1 - margin)
    distances = {
        "left": x / max(left, 1),
        "right": (width - x) / max(width - right, 1),
        "top": y / max(top, 1),
        "bottom": (height - y) / max(height - bottom, 1),
    }
    candidates = {k: v for k, v in distances.items() if v <= 1}
    return min(candidates or distances, key=(candidates or distances).get)


def build_track_features(
    detections: pd.DataFrame,
    *,
    zone_margin_ratio: float = 0.15,
    meters_per_pixel: float = 0.05,
) -> pd.DataFrame:
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive")
    if detections.empty:
        return pd.DataFrame(columns=TRACK_FEATURE_COLUMNS)
    data = validate_detections(detections)
    rows: list[dict[str, object]] = []
    for (video_id, track_id), track in data.groupby(["video_id", "track_id"], sort=False):
        track = track.sort_values("frame_id")
        first, last = track.iloc[0], track.iloc[-1]
        dx = float(last["centroid_x"] - first["centroid_x"])
        dy = float(last["centroid_y"] - first["centroid_y"])
        step_dx = track["centroid_x"].diff()
        step_dy = track["centroid_y"].diff()
        path_length = float(np.hypot(step_dx.fillna(0), step_dy.fillna(0)).sum())
        duration_s = float(last["timestamp_s"] - first["timestamp_s"])
        avg_speed_px_s = path_length / duration_s if duration_s > 0 else 0.0
        road_segment = (
            track["road_segment"].mode().iat[0]
            if "road_segment" in track.columns and not track["road_segment"].dropna().empty
            else video_id
        )
        rows.append(
            {
                "video_id": video_id,
                "road_segment": str(road_segment),
                "track_id": int(track_id),
                "class_name": track["class_name"].mode().iat[0],
                "first_frame": int(first["frame_id"]),
                "last_frame": int(last["frame_id"]),
                "track_duration_frames": int(track["frame_id"].nunique()),
                "duration_s": duration_s,
                "mean_confidence": float(track["confidence"].mean()),
                "start_x": float(first["centroid_x"]),
                "start_y": float(first["centroid_y"]),
                "end_x": float(last["centroid_x"]),
                "end_y": float(last["centroid_y"]),
                "movement_angle": float(math.degrees(math.atan2(dy, dx)) % 360),
                "net_displacement": float(math.hypot(dx, dy)),
                "path_length": path_length,
                "avg_speed_px_s": float(avg_speed_px_s),
                "avg_speed_kmh": float(avg_speed_px_s * meters_per_pixel * 3.6),
                "straightness": float(math.hypot(dx, dy) / path_length) if path_length else 0.0,
                "entry_zone": _entry_zone(
                    float(first["centroid_x"]),
                    float(first["centroid_y"]),
                    float(first["frame_width"]),
                    float(first["frame_height"]),
                    zone_margin_ratio,
                ),
            }
        )
    return pd.DataFrame(rows, columns=TRACK_FEATURE_COLUMNS)


def build_video_features(
    track_features: pd.DataFrame, *, angle_deviation_threshold_deg: float = 30.0
) -> pd.DataFrame:
    if track_features.empty:
        return pd.DataFrame(columns=VIDEO_FEATURE_COLUMNS)
    rows: list[dict[str, object]] = []
    for video_id, tracks in track_features.groupby("video_id", sort=False):
        angles = tracks["movement_angle"].to_numpy(dtype=float)
        dominant = _circular_mean_deg(angles)
        deviations = _circular_deviation_deg(angles, dominant)
        bins = np.histogram(angles, bins=np.linspace(0, 360, 9))[0].astype(float)
        probabilities = bins[bins > 0] / max(bins.sum(), 1)
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        rows.append(
            {
                "video_id": video_id,
                "angle_variance": float(np.var(deviations)),
                "direction_entropy": entropy,
                "unique_entry_zones": int(tracks["entry_zone"].nunique()),
                "angle_deviation_proportion": float(
                    (deviations > angle_deviation_threshold_deg).mean()
                ),
                "mean_track_displacement": float(tracks["net_displacement"].mean()),
                "vehicle_count": int(len(tracks)),
                "mean_avg_speed_kmh": float(tracks["avg_speed_kmh"].mean()),
                "dominant_angle": dominant % 360,
            }
        )
    return pd.DataFrame(rows, columns=VIDEO_FEATURE_COLUMNS)
