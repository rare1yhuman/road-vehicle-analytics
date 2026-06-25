from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from .classifier import RoadTypeClassifier, leave_one_out_evaluate
from .cleaning import clean_detections
from .config import load_config
from .counting import count_confirmed_tracks, count_line_crossings
from .eda import create_eda_plots
from .evaluation import classification_metrics, confidence_sensitivity, evaluate_counts
from .features import build_track_features, build_video_features
from .rule_classification import classify_road_rules
from .review import build_track_review_template, evaluate_track_review
from .schema import read_detections

GENERATED_FILES = {
    "analysis_summary.json",
    "cleaned_detections.csv",
    "confidence_sensitivity.csv",
    "count_evaluation.csv",
    "crossing_events.csv",
    "road_classifier.json",
    "road_classifier_coefficients.csv",
    "road_classifier_loocv.csv",
    "road_rule_predictions.csv",
    "road_type_predictions.csv",
    "track_cleaning_audit.csv",
    "track_features.csv",
    "track_review_evaluation.csv",
    "track_review_template.csv",
    "vehicle_counts.csv",
    "video_features.csv",
}


def _clean_generated_outputs(output: Path) -> None:
    for filename in GENERATED_FILES:
        (output / filename).unlink(missing_ok=True)
    shutil.rmtree(output / "plots", ignore_errors=True)


def run_analysis(
    detections_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    inventory_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    counting_options: dict[str, object] | None = None,
    camera_mode: str = "fixed",
    track_ground_truth_path: str | Path | None = None,
) -> dict[str, object]:
    if camera_mode not in {"fixed", "moving"}:
        raise ValueError("camera_mode must be fixed or moving")
    config = load_config(config_path)
    if counting_options:
        config["counting"].update(counting_options)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _clean_generated_outputs(output)

    raw = read_detections(detections_path)
    cleaned, audit = clean_detections(raw, **config["cleaning"])
    tracks = build_track_features(
        cleaned,
        zone_margin_ratio=config["classification"]["zone_margin_ratio"],
    )
    videos = build_video_features(
        tracks,
        angle_deviation_threshold_deg=config["classification"][
            "angle_deviation_threshold_deg"
        ],
    )
    if camera_mode == "moving":
        events, counts = count_confirmed_tracks(
            cleaned,
            minimum_track_frames=max(
                8, int(config["cleaning"]["minimum_track_frames"])
            ),
        )
    else:
        events, counts = count_line_crossings(cleaned, **config["counting"])
    rule_options = config["classification"]
    rule_predictions = classify_road_rules(
        videos,
        minimum_tracks=rule_options["rule_minimum_tracks"],
        intersection_entropy=rule_options["rule_intersection_entropy"],
        intersection_deviation=rule_options["rule_intersection_deviation"],
        straight_entropy=rule_options["rule_straight_entropy"],
        straight_deviation=rule_options["rule_straight_deviation"],
    )
    if camera_mode == "moving":
        rule_predictions["predicted_road_type"] = "unknown"
        rule_predictions["rule_confidence"] = 0.0
        rule_predictions["reason"] = (
            "Automatic trajectory road classification is disabled for moving-camera "
            "footage because image-space motion is not road geometry."
        )
    review_template = build_track_review_template(tracks)

    cleaned.to_csv(output / "cleaned_detections.csv", index=False)
    audit.to_csv(output / "track_cleaning_audit.csv", index=False)
    tracks.to_csv(output / "track_features.csv", index=False)
    videos.to_csv(output / "video_features.csv", index=False)
    events.to_csv(output / "crossing_events.csv", index=False)
    counts.to_csv(output / "vehicle_counts.csv", index=False)
    rule_predictions.to_csv(output / "road_rule_predictions.csv", index=False)
    review_template.to_csv(output / "track_review_template.csv", index=False)
    plot_paths = create_eda_plots(cleaned, tracks, output / "plots")

    report: dict[str, object] = {
        "input_rows": len(raw),
        "cleaned_rows": len(cleaned),
        "tracks_before_cleaning": len(audit),
        "tracks_after_cleaning": int(audit["kept"].sum()),
        "crossing_events": len(events),
        "plots": [str(path) for path in plot_paths],
        "classifier_status": "no_inventory",
        "camera_mode": camera_mode,
        "counting_method": (
            "confirmed_tracks" if camera_mode == "moving" else "line_crossing"
        ),
        "rule_road_predictions": rule_predictions.to_dict(orient="records"),
        "analysis_status": (
            "complete" if len(tracks) else "no_reliable_tracks"
        ),
    }

    if track_ground_truth_path:
        comparison, track_metrics = evaluate_track_review(
            review_template,
            pd.read_csv(track_ground_truth_path),
        )
        comparison.to_csv(output / "track_review_evaluation.csv", index=False)
        report["track_review_metrics"] = track_metrics

    if inventory_path and camera_mode == "moving":
        report["classifier_status"] = "disabled_for_moving_camera"
    elif inventory_path:
        inventory = pd.read_csv(inventory_path)
        if {"video_id", "road_type"}.issubset(inventory.columns):
            labeled = videos.merge(
                inventory[["video_id", "road_type"]].drop_duplicates(),
                on="video_id",
                how="inner",
            )
            labeled = labeled[labeled["road_type"].isin(["straight", "intersection"])].copy()
            labeled["target"] = (labeled["road_type"] == "intersection").astype(int)
            if len(labeled) >= 3 and labeled["target"].nunique() == 2:
                options = config["classification"]
                loo = leave_one_out_evaluate(
                    labeled,
                    labeled["target"],
                    regularization=options["regularization"],
                    learning_rate=options["learning_rate"],
                    iterations=options["iterations"],
                )
                loo.insert(0, "video_id", labeled["video_id"].to_numpy())
                loo.to_csv(output / "road_classifier_loocv.csv", index=False)
                classifier = RoadTypeClassifier(
                    regularization=options["regularization"],
                    learning_rate=options["learning_rate"],
                    iterations=options["iterations"],
                ).fit(labeled, labeled["target"])
                with (output / "road_classifier.json").open("w", encoding="utf-8") as handle:
                    json.dump(classifier.to_dict(), handle, indent=2)
                classifier.coefficients().to_csv(
                    output / "road_classifier_coefficients.csv", index=False
                )
                predictions = videos.copy()
                predictions["intersection_probability"] = classifier.predict_proba(videos)
                predictions["predicted_road_type"] = predictions[
                    "intersection_probability"
                ].map(lambda value: "intersection" if value >= 0.5 else "straight")
                predictions.to_csv(output / "road_type_predictions.csv", index=False)
                report["classifier_metrics"] = classification_metrics(
                    loo["actual"], loo["predicted"]
                )
                report["classifier_status"] = "trained"
            else:
                report["classifier_status"] = "insufficient_labeled_videos"
        else:
            report["classifier_status"] = "invalid_inventory"

    if ground_truth_path:
        truth = pd.read_csv(ground_truth_path)
        count_errors = evaluate_counts(counts, truth)
        count_errors.to_csv(output / "count_evaluation.csv", index=False)
        if camera_mode == "fixed":
            sensitivity = confidence_sensitivity(
                raw,
                truth,
                [0.3, 0.4, 0.5, 0.6, 0.7],
                minimum_track_frames=config["cleaning"]["minimum_track_frames"],
                minimum_displacement_px=config["cleaning"]["minimum_displacement_px"],
                counting_options=config["counting"],
            )
            sensitivity.to_csv(output / "confidence_sensitivity.csv", index=False)
        report["mean_count_percent_error"] = float(count_errors["percent_error"].mean())

    with (output / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
