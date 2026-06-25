from __future__ import annotations

import pandas as pd


def build_track_review_template(track_features: pd.DataFrame) -> pd.DataFrame:
    """Create a user-editable ground-truth sheet with one row per predicted track."""
    columns = [
        "video_id",
        "track_id",
        "predicted_class",
        "corrected_class",
        "should_count",
        "notes",
    ]
    if track_features.empty:
        return pd.DataFrame(columns=columns)
    review = track_features[["video_id", "track_id", "class_name"]].copy()
    review = review.rename(columns={"class_name": "predicted_class"})
    review["corrected_class"] = review["predicted_class"]
    review["should_count"] = True
    review["notes"] = ""
    return review[columns]


def evaluate_track_review(
    review_template: pd.DataFrame,
    reviewed_ground_truth: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    required = {"video_id", "track_id", "corrected_class", "should_count"}
    missing = required - set(reviewed_ground_truth.columns)
    if missing:
        raise ValueError(f"Track ground truth is missing columns: {', '.join(sorted(missing))}")
    truth = reviewed_ground_truth.copy()
    predicted = review_template[["video_id", "track_id", "predicted_class"]].copy()
    predicted["video_id"] = predicted["video_id"].astype(str)
    predicted["track_id"] = predicted["track_id"].astype(str)
    truth["video_id"] = truth["video_id"].astype(str)
    truth["track_id"] = truth["track_id"].astype(str)
    truth["should_count"] = (
        truth["should_count"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    )
    merged = predicted.merge(
        truth[["video_id", "track_id", "corrected_class", "should_count"]],
        on=["video_id", "track_id"],
        how="outer",
        indicator=True,
    )
    merged["is_predicted"] = merged["_merge"].isin(["both", "left_only"])
    merged["is_ground_truth"] = merged["_merge"].isin(["both", "right_only"])
    merged["should_count"] = merged["should_count"].fillna(False).astype(bool)
    merged["class_correct"] = (
        (merged["_merge"] == "both")
        & merged["should_count"]
        & (merged["predicted_class"] == merged["corrected_class"])
    )
    merged["review_status"] = "matched"
    merged.loc[merged["_merge"] == "left_only", "review_status"] = "unreviewed_prediction"
    merged.loc[merged["_merge"] == "right_only", "review_status"] = "missed_vehicle"
    merged.loc[
        (merged["_merge"] == "both") & ~merged["should_count"], "review_status"
    ] = "false_positive"
    reviewed = merged[merged["_merge"] == "both"]
    countable_matched = reviewed[reviewed["should_count"]]
    ground_truth_count = int(merged["should_count"].sum())
    predicted_count = int(len(predicted))
    true_positive_count = int(len(countable_matched))
    metrics: dict[str, float | int] = {
        "reviewed_tracks": int(len(reviewed)),
        "class_accuracy": (
            float(countable_matched["class_correct"].mean())
            if len(countable_matched)
            else 0.0
        ),
        "ground_truth_count": ground_truth_count,
        "predicted_track_count": predicted_count,
        "count_precision": (
            float(true_positive_count / predicted_count) if predicted_count else 0.0
        ),
        "count_recall": (
            float(true_positive_count / ground_truth_count)
            if ground_truth_count
            else 0.0
        ),
        "missed_vehicles": int((merged["_merge"] == "right_only").sum()),
        "false_positive_tracks": int(
            ((merged["_merge"] == "both") & ~merged["should_count"]).sum()
        ),
    }
    actual = ground_truth_count
    metrics["count_net_error"] = predicted_count - actual
    metrics["count_absolute_error"] = (
        metrics["missed_vehicles"] + metrics["false_positive_tracks"]
    )
    metrics["count_percent_error"] = (
        float(metrics["count_absolute_error"] / actual * 100) if actual else 0.0
    )
    return merged.drop(columns="_merge"), metrics
