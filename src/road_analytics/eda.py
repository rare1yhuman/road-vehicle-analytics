from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_eda_plots(
    detections: pd.DataFrame, track_features: pd.DataFrame, output_dir: str | Path
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if detections.empty or track_features.empty:
        return paths

    for video_id, data in detections.groupby("video_id"):
        safe_id = str(video_id).replace("/", "_")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for class_name, values in data.groupby("class_name"):
            ax.hist(values["confidence"], bins=15, alpha=0.6, label=class_name)
        ax.set(title=f"Detection confidence — {video_id}", xlabel="Confidence", ylabel="Rows")
        ax.legend()
        path = output / f"{safe_id}_confidence.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        class_counts = (
            track_features[track_features["video_id"] == video_id]["class_name"]
            .value_counts()
            .sort_values()
        )
        class_counts.plot.barh(ax=ax, color="#2563eb")
        ax.set(title=f"Unique tracks by class — {video_id}", xlabel="Tracks")
        path = output / f"{safe_id}_classes.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(8, 5))
        for _, track in data.groupby("track_id"):
            ax.plot(track["centroid_x"], track["centroid_y"], alpha=0.65, linewidth=1.2)
        ax.invert_yaxis()
        ax.set(title=f"Vehicle trajectories — {video_id}", xlabel="X (px)", ylabel="Y (px)")
        path = output / f"{safe_id}_trajectories.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        per_second = (
            data.assign(second=data["timestamp_s"].astype(int))
            .groupby("second")["track_id"]
            .nunique()
        )
        per_second.plot(ax=ax, color="#0f766e")
        ax.set(title=f"Active vehicles over time — {video_id}", xlabel="Second", ylabel="Tracks")
        path = output / f"{safe_id}_timeline.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    track_features["track_duration_frames"].plot.hist(bins=20, ax=ax, color="#7c3aed")
    ax.set(title="Track length distribution", xlabel="Unique frames")
    path = output / "track_lengths.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths
