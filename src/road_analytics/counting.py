from __future__ import annotations

import pandas as pd

from .schema import validate_detections

EVENT_COLUMNS = [
    "video_id",
    "track_id",
    "class_name",
    "crossing_frame",
    "crossing_timestamp_s",
    "direction",
    "line_axis",
    "line_position",
]
COUNT_COLUMNS = ["video_id", "class_name", "vehicle_count"]


def count_confirmed_tracks(
    detections: pd.DataFrame,
    *,
    minimum_track_frames: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count each sufficiently persistent track once for moving-camera footage."""
    if detections.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS), pd.DataFrame(columns=COUNT_COLUMNS)
    data = validate_detections(detections)
    rows: list[dict[str, object]] = []
    for (video_id, track_id), track in data.groupby(
        ["video_id", "track_id"], sort=False
    ):
        track = track.sort_values("frame_id").drop_duplicates("frame_id")
        if len(track) < minimum_track_frames:
            continue
        confirmation = track.iloc[minimum_track_frames - 1]
        rows.append(
            {
                "video_id": video_id,
                "track_id": int(track_id),
                "class_name": track["class_name"].mode().iat[0],
                "crossing_frame": int(confirmation["frame_id"]),
                "crossing_timestamp_s": float(confirmation["timestamp_s"]),
            }
        )
    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS), pd.DataFrame(columns=COUNT_COLUMNS)
    events["direction"] = "confirmed_track"
    events["line_axis"] = "none"
    events["line_position"] = float("nan")
    events = events[EVENT_COLUMNS]
    summary = (
        events.groupby(["video_id", "class_name"])
        .size()
        .rename("vehicle_count")
        .reset_index()
    )
    return events, summary


def count_line_crossings(
    detections: pd.DataFrame,
    *,
    line_axis: str = "y",
    line_position_ratio: float = 0.55,
    hysteresis_px: float = 4.0,
    direction: str = "both",
    anchor: str = "bottom_center",
    roi_x_min_ratio: float = 0.0,
    roi_x_max_ratio: float = 1.0,
    roi_y_min_ratio: float = 0.0,
    roi_y_max_ratio: float = 1.0,
    maximum_frame_gap: int = 5,
    maximum_crossing_distance_ratio: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count a track once when it moves across a virtual line."""
    if line_axis not in {"x", "y"}:
        raise ValueError("line_axis must be 'x' or 'y'")
    if direction not in {"both", "positive", "negative"}:
        raise ValueError("direction must be both, positive, or negative")
    if anchor not in {"centroid", "bottom_center"}:
        raise ValueError("anchor must be centroid or bottom_center")
    ratios = (roi_x_min_ratio, roi_x_max_ratio, roi_y_min_ratio, roi_y_max_ratio)
    if any(value < 0 or value > 1 for value in ratios):
        raise ValueError("ROI ratios must be between 0 and 1")
    if roi_x_min_ratio >= roi_x_max_ratio or roi_y_min_ratio >= roi_y_max_ratio:
        raise ValueError("ROI minimum ratios must be smaller than maximum ratios")
    if maximum_frame_gap < 1:
        raise ValueError("maximum_frame_gap must be at least 1")
    if not 0 < maximum_crossing_distance_ratio <= 1:
        raise ValueError("maximum_crossing_distance_ratio must be between 0 and 1")

    if detections.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS), pd.DataFrame(columns=COUNT_COLUMNS)
    data = validate_detections(detections)
    data = data.copy()
    data["_anchor_x"] = (data["x1"] + data["x2"]) / 2
    data["_anchor_y"] = data["y2"] if anchor == "bottom_center" else data["centroid_y"]
    if anchor == "centroid":
        data["_anchor_x"] = data["centroid_x"]
    inside_roi = (
        (data["_anchor_x"] >= data["frame_width"] * roi_x_min_ratio)
        & (data["_anchor_x"] <= data["frame_width"] * roi_x_max_ratio)
        & (data["_anchor_y"] >= data["frame_height"] * roi_y_min_ratio)
        & (data["_anchor_y"] <= data["frame_height"] * roi_y_max_ratio)
    )
    data = data[inside_roi].copy()
    coordinate = "_anchor_x" if line_axis == "x" else "_anchor_y"
    dimension = "frame_width" if line_axis == "x" else "frame_height"
    events: list[dict[str, object]] = []

    for (video_id, track_id), track in data.groupby(["video_id", "track_id"], sort=False):
        track = track.sort_values("frame_id")
        line = float(track[dimension].median() * line_position_ratio)
        crossing_direction = None
        event_row = None

        stable_row = None
        stable_side = 0
        for _, current in track.iterrows():
            current_value = float(current[coordinate])
            if current_value < line - hysteresis_px:
                current_side = -1
            elif current_value > line + hysteresis_px:
                current_side = 1
            else:
                continue
            if stable_row is None:
                stable_row = current
                stable_side = current_side
                continue
            if current_side == stable_side:
                stable_row = current
                continue
            frame_gap = int(current["frame_id"] - stable_row["frame_id"])
            crossing_distance = abs(current_value - float(stable_row[coordinate]))
            maximum_distance = float(current[dimension] * maximum_crossing_distance_ratio)
            if frame_gap <= maximum_frame_gap and crossing_distance <= maximum_distance:
                crossing_direction = "positive" if stable_side < current_side else "negative"
                if direction in {"both", crossing_direction}:
                    event_row = current
                    break
            stable_row = current
            stable_side = current_side

        if event_row is not None:
            events.append(
                {
                    "video_id": video_id,
                    "track_id": int(track_id),
                    "class_name": track["class_name"].mode().iat[0],
                    "crossing_frame": int(event_row["frame_id"]),
                    "crossing_timestamp_s": float(event_row["timestamp_s"]),
                    "direction": crossing_direction,
                    "line_axis": line_axis,
                    "line_position": line,
                }
            )

    event_df = pd.DataFrame(
        events,
        columns=EVENT_COLUMNS,
    )
    if event_df.empty:
        summary = pd.DataFrame(columns=COUNT_COLUMNS)
    else:
        summary = (
            event_df.groupby(["video_id", "class_name"])
            .size()
            .rename("vehicle_count")
            .reset_index()
        )
    return event_df, summary
