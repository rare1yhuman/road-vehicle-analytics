import numpy as np
import pandas as pd

from road_analytics.classifier import RoadTypeClassifier
from road_analytics.cleaning import clean_detections
from road_analytics.counting import count_confirmed_tracks, count_line_crossings
from road_analytics.features import build_track_features, build_video_features
from road_analytics.rule_classification import classify_road_rules
from road_analytics.review import build_track_review_template, evaluate_track_review
from road_analytics.synthetic import generate_demo_dataset


def _track(track_id, ys, confidence=0.9):
    rows = []
    for frame, y in enumerate(ys):
        rows.append(
            {
                "video_id": "test",
                "frame_id": frame,
                "track_id": track_id,
                "class_name": "car",
                "confidence": confidence,
                "x1": 80,
                "y1": y - 10,
                "x2": 120,
                "y2": y + 10,
                "centroid_x": 100,
                "centroid_y": y,
                "timestamp_s": frame / 10,
                "frame_width": 200,
                "frame_height": 200,
            }
        )
    return rows


def test_cleaning_filters_short_tracks():
    data = pd.DataFrame(_track(1, [20, 40, 60, 80, 120]) + _track(2, [10, 11]))
    cleaned, audit = clean_detections(
        data, minimum_track_frames=5, minimum_displacement_px=5
    )
    assert set(cleaned["track_id"]) == {1}
    assert audit.set_index("track_id").loc[2, "kept"] == np.False_


def test_line_crossing_counts_each_track_once():
    data = pd.DataFrame(_track(1, [20, 60, 105, 150, 180]))
    events, counts = count_line_crossings(
        data, line_position_ratio=0.5, hysteresis_px=2
    )
    assert len(events) == 1
    assert counts.iloc[0]["vehicle_count"] == 1


def test_line_crossing_uses_bottom_center_and_roi():
    data = pd.DataFrame(_track(1, [20, 60, 85, 105, 140]))
    events, _ = count_line_crossings(
        data,
        line_position_ratio=0.5,
        hysteresis_px=2,
        anchor="bottom_center",
        roi_x_min_ratio=0.0,
        roi_x_max_ratio=0.4,
    )
    assert events.empty


def test_confirmed_track_counting_ignores_short_fragments():
    data = pd.DataFrame(
        _track(1, [20, 40, 60, 80, 100, 120, 140, 160])
        + _track(2, [20, 30, 40])
    )
    events, counts = count_confirmed_tracks(data, minimum_track_frames=8)
    assert set(events["track_id"]) == {1}
    assert counts.iloc[0]["vehicle_count"] == 1
    assert events.iloc[0]["crossing_frame"] == 7


def test_line_crossing_rejects_implausible_jump():
    data = pd.DataFrame(_track(1, [20, 20, 180, 180]))
    events, counts = count_line_crossings(
        data, line_position_ratio=0.5, hysteresis_px=2
    )
    assert events.empty
    assert counts.empty


def test_cleaning_smooths_class_name_across_track():
    data = pd.DataFrame(_track(1, [20, 40, 60, 80, 120]))
    data.loc[0, "class_name"] = "motorcycle"
    data.loc[0, "confidence"] = 0.4
    cleaned, _ = clean_detections(
        data, minimum_track_frames=5, minimum_displacement_px=5
    )
    assert set(cleaned["class_name"]) == {"car"}


def test_feature_pipeline(tmp_path):
    detections_path, _, _ = generate_demo_dataset(tmp_path, videos_per_type=2)
    detections = pd.read_csv(detections_path)
    cleaned, _ = clean_detections(detections)
    tracks = build_track_features(cleaned)
    videos = build_video_features(tracks)
    assert len(videos) == 4
    assert videos["unique_entry_zones"].between(1, 4).all()
    assert "mean_avg_speed_kmh" in videos.columns


def test_track_features_include_calibrated_speed():
    data = pd.DataFrame(_track(1, [20, 40, 60, 80, 120]))
    tracks = build_track_features(data, meters_per_pixel=0.05)
    assert tracks.iloc[0]["road_segment"] == "test"
    assert tracks.iloc[0]["avg_speed_px_s"] == 250.0
    assert tracks.iloc[0]["avg_speed_kmh"] == 45.0


def test_track_speed_handles_zero_duration():
    data = pd.DataFrame(_track(1, [20, 40, 60, 80, 120]))
    data["timestamp_s"] = 1.0
    tracks = build_track_features(data)
    assert tracks.iloc[0]["duration_s"] == 0.0
    assert tracks.iloc[0]["avg_speed_px_s"] == 0.0
    assert tracks.iloc[0]["avg_speed_kmh"] == 0.0


def test_classifier_learns_separable_features():
    features = pd.DataFrame(
        {
            "angle_variance": [1, 2, 3, 100, 120, 150],
            "direction_entropy": [0.1, 0.2, 0.15, 1.5, 1.8, 2.0],
            "unique_entry_zones": [2, 2, 2, 3, 4, 4],
            "angle_deviation_proportion": [0, 0.1, 0, 0.7, 0.8, 0.9],
            "mean_track_displacement": [200] * 6,
            "vehicle_count": [10] * 6,
        }
    )
    labels = pd.Series([0, 0, 0, 1, 1, 1])
    model = RoadTypeClassifier(iterations=1000).fit(features, labels)
    assert np.array_equal(model.predict(features), labels.to_numpy())


def test_rule_classifier_is_disabled_until_validated():
    features = pd.DataFrame(
        [
            {
                "video_id": "straight",
                "direction_entropy": 0.2,
                "angle_deviation_proportion": 0.05,
                "unique_entry_zones": 2,
                "vehicle_count": 10,
            },
            {
                "video_id": "short",
                "direction_entropy": 1.5,
                "angle_deviation_proportion": 0.6,
                "unique_entry_zones": 4,
                "vehicle_count": 2,
            },
        ]
    )
    result = classify_road_rules(features).set_index("video_id")
    assert result.loc["straight", "predicted_road_type"] == "unknown"
    assert result.loc["short", "predicted_road_type"] == "unknown"


def test_track_review_metrics():
    tracks = pd.DataFrame(
        {
            "video_id": ["v", "v"],
            "track_id": [1, 2],
            "class_name": ["car", "truck"],
        }
    )
    template = build_track_review_template(tracks)
    truth = template.copy()
    truth.loc[truth["track_id"] == 2, "corrected_class"] = "bus"
    truth.loc[truth["track_id"] == 2, "should_count"] = False
    comparison, metrics = evaluate_track_review(template, truth)
    assert len(comparison) == 2
    assert metrics["class_accuracy"] == 1.0
    assert metrics["ground_truth_count"] == 1
    assert metrics["count_precision"] == 0.5
    assert metrics["count_recall"] == 1.0
    assert metrics["count_absolute_error"] == 1
