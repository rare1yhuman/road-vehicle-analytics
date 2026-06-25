from __future__ import annotations

import pandas as pd


def classify_road_rules(
    video_features: pd.DataFrame,
    *,
    minimum_tracks: int = 5,
    intersection_entropy: float = 1.15,
    intersection_deviation: float = 0.30,
    straight_entropy: float = 0.80,
    straight_deviation: float = 0.20,
) -> pd.DataFrame:
    """Return a conservative placeholder until the heuristic is validated."""
    columns = ["video_id", "predicted_road_type", "rule_confidence", "reason"]
    rows: list[dict[str, object]] = []
    for row in video_features.itertuples(index=False):
        rows.append(
            {
                "video_id": row.video_id,
                "predicted_road_type": "unknown",
                "rule_confidence": 0.0,
                "reason": (
                    "Automatic rule-based road classification is disabled because the "
                    "heuristic has not met validation requirements."
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)
