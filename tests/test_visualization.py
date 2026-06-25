import pandas as pd

from road_analytics.annotation import annotate_video
from road_analytics.visualization import (
    add_speed_columns,
    prepare_speed_distribution,
    prepare_speed_volume,
    prepare_traffic_volume,
    prepare_vehicle_breakdown,
)


def test_annotated_video_can_be_written_as_webm(tmp_path):
    import cv2
    import numpy as np

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (64, 64),
    )
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    detections = pd.DataFrame(
        {
            "frame_id": [0],
            "track_id": [1],
            "class_name": ["car"],
            "confidence": [0.9],
            "x1": [8],
            "y1": [8],
            "x2": [32],
            "y2": [32],
        }
    )
    events = pd.DataFrame(
        {"track_id": [1], "crossing_frame": [0], "class_name": ["car"]}
    )

    output = annotate_video(source, tmp_path / "annotated.webm", detections, events)
    capture = cv2.VideoCapture(str(output))
    ok, _ = capture.read()
    capture.release()

    assert output.exists()
    assert output.stat().st_size > 0
    assert ok


def _detections():
    return pd.DataFrame(
        {
            "video_id": ["v1", "v1", "v1", "v1"],
            "frame_id": [0, 1, 2, 3],
            "track_id": [1, 1, 2, 2],
            "class_name": ["car", "car", "truck", "truck"],
            "confidence": [0.9, 0.9, 0.8, 0.8],
            "x1": [0, 10, 20, 25],
            "y1": [0, 10, 20, 25],
            "x2": [20, 30, 40, 45],
            "y2": [20, 30, 40, 45],
            "centroid_x": [10, 20, 30, 35],
            "centroid_y": [10, 20, 30, 35],
            "timestamp_s": [0.0, 1.0, 0.0, 1.0],
            "frame_width": [100, 100, 100, 100],
            "frame_height": [100, 100, 100, 100],
        }
    )


def _tracks():
    return pd.DataFrame(
        {
            "video_id": ["v1", "v1"],
            "track_id": [1, 2],
            "class_name": ["car", "truck"],
            "path_length": [20.0, 10.0],
            "duration_s": [1.0, 1.0],
        }
    )


def test_add_speed_columns_recalibrates_and_adds_segment():
    tracks = add_speed_columns(_tracks(), meters_per_pixel=0.1)
    assert set(["road_segment", "avg_speed_px_s", "avg_speed_kmh"]).issubset(tracks.columns)
    assert tracks.loc[tracks["track_id"] == 1, "road_segment"].iat[0] == "v1"
    assert round(tracks.loc[tracks["track_id"] == 1, "avg_speed_kmh"].iat[0], 2) == 7.2


def test_prepare_visualization_frames_have_expected_columns():
    detections = _detections()
    tracks = _tracks()
    volume = prepare_traffic_volume(detections, pd.DataFrame(), time_bin_s=1)
    breakdown = prepare_vehicle_breakdown(pd.DataFrame(), tracks)
    speed_volume = prepare_speed_volume(detections, tracks, time_bin_s=1)
    distribution = prepare_speed_distribution(tracks)

    assert {"road_segment", "class_name", "time_bin_s", "vehicle_count"}.issubset(volume.columns)
    assert {"class_name", "vehicle_count"}.issubset(breakdown.columns)
    assert {"road_segment", "time_bin_s", "vehicle_count", "avg_speed_kmh"}.issubset(speed_volume.columns)
    assert {"road_segment", "class_name", "avg_speed_kmh"}.issubset(distribution.columns)
