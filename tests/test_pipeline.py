import json

import pandas as pd

from road_analytics.config import load_config
from road_analytics.pipeline import run_analysis
from road_analytics.synthetic import generate_demo_dataset


def test_full_pipeline_writes_outputs(tmp_path):
    detections, truth, inventory = generate_demo_dataset(
        tmp_path / "data", videos_per_type=2
    )
    output = tmp_path / "analysis"
    report = run_analysis(
        detections,
        output,
        inventory_path=inventory,
        ground_truth_path=truth,
    )
    assert report["tracks_after_cleaning"] > 0
    assert (output / "vehicle_counts.csv").exists()
    assert (output / "road_rule_predictions.csv").exists()
    assert (output / "track_review_template.csv").exists()
    assert (output / "road_classifier.json").exists()
    summary = json.loads((output / "analysis_summary.json").read_text())
    assert summary["input_rows"] > 0
    assert summary["classifier_status"] == "trained"


def test_moving_camera_pipeline_uses_confirmed_tracks(tmp_path):
    detections, _, _ = generate_demo_dataset(tmp_path / "data", videos_per_type=1)
    output = tmp_path / "moving"
    report = run_analysis(detections, output, camera_mode="moving")
    assert report["counting_method"] == "confirmed_tracks"
    assert report["rule_road_predictions"][0]["predicted_road_type"] == "unknown"


def test_pipeline_handles_no_reliable_tracks(tmp_path):
    detections, _, _ = generate_demo_dataset(tmp_path / "data", videos_per_type=1)
    frame = pd.read_csv(detections)
    frame["confidence"] = 0.1
    low_confidence = tmp_path / "low_confidence.csv"
    frame.to_csv(low_confidence, index=False)
    output = tmp_path / "empty-analysis"
    report = run_analysis(low_confidence, output, camera_mode="moving")
    assert report["analysis_status"] == "no_reliable_tracks"
    assert report["tracks_after_cleaning"] == 0
    assert pd.read_csv(output / "vehicle_counts.csv").empty


def test_pipeline_removes_stale_optional_outputs(tmp_path):
    detections, _, _ = generate_demo_dataset(tmp_path / "data", videos_per_type=1)
    output = tmp_path / "analysis"
    output.mkdir()
    for name in [
        "count_evaluation.csv",
        "confidence_sensitivity.csv",
        "track_review_evaluation.csv",
    ]:
        (output / name).write_text("stale")
    plots = output / "plots"
    plots.mkdir()
    (plots / "stale.png").write_text("stale")
    run_analysis(detections, output, camera_mode="moving")
    assert not (output / "count_evaluation.csv").exists()
    assert not (output / "confidence_sensitivity.csv").exists()
    assert not (output / "track_review_evaluation.csv").exists()
    assert not (plots / "stale.png").exists()


def test_moving_camera_disables_trained_classifier(tmp_path):
    detections, _, inventory = generate_demo_dataset(
        tmp_path / "data", videos_per_type=2
    )
    output = tmp_path / "moving"
    report = run_analysis(
        detections, output, camera_mode="moving", inventory_path=inventory
    )
    assert report["classifier_status"] == "disabled_for_moving_camera"
    assert not (output / "road_type_predictions.csv").exists()


def test_packaged_default_config_loads():
    assert load_config()["extraction"]["model"] == "yolov8m.pt"
