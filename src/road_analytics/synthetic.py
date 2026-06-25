from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import DETECTION_COLUMNS


CLASSES = ["car", "motorcycle", "truck", "bus"]


def generate_demo_dataset(
    output_dir: str | Path,
    *,
    videos_per_type: int = 5,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """Create deterministic trajectory data for exercising the full DS workflow."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    width, height, fps = 1280, 720, 15

    for road_type in ["straight", "intersection"]:
        for video_number in range(1, videos_per_type + 1):
            video_id = f"{road_type}_{video_number:02d}"
            track_count = int(rng.integers(15, 27))
            crossing_counts: dict[str, int] = {}
            for track_id in range(1, track_count + 1):
                class_name = str(rng.choice(CLASSES, p=[0.58, 0.22, 0.14, 0.06]))
                box_w = {"motorcycle": 32, "car": 58, "truck": 82, "bus": 94}[class_name]
                box_h = box_w * 0.75
                duration = int(rng.integers(18, 46))
                start_frame = int(rng.integers(0, 75))

                if road_type == "straight":
                    direction = int(rng.choice([-1, 1]))
                    start_x = float(rng.uniform(width * 0.25, width * 0.75))
                    start_y = height * (0.08 if direction == 1 else 0.92)
                    end_x = start_x + float(rng.normal(0, 45))
                    end_y = height * (0.92 if direction == 1 else 0.08)
                else:
                    side_in, side_out = rng.choice(4, size=2, replace=False)
                    points = [
                        (float(rng.uniform(300, 980)), height * 0.05),
                        (width * 0.95, float(rng.uniform(180, 540))),
                        (float(rng.uniform(300, 980)), height * 0.95),
                        (width * 0.05, float(rng.uniform(180, 540))),
                    ]
                    start_x, start_y = points[int(side_in)]
                    end_x, end_y = points[int(side_out)]

                counting_line = height * 0.55
                start_anchor_y = start_y + box_h / 2
                end_anchor_y = end_y + box_h / 2
                crosses_line = (
                    start_anchor_y < counting_line - 4
                    and end_anchor_y > counting_line + 4
                ) or (
                    start_anchor_y > counting_line + 4
                    and end_anchor_y < counting_line - 4
                )
                if crosses_line:
                    crossing_counts[class_name] = crossing_counts.get(class_name, 0) + 1

                for offset in range(duration):
                    t = offset / max(duration - 1, 1)
                    curve = np.sin(t * np.pi) * (rng.normal(0, 10) if road_type == "straight" else 35)
                    x = start_x + (end_x - start_x) * t + curve + rng.normal(0, 2)
                    y = start_y + (end_y - start_y) * t + rng.normal(0, 2)
                    rows.append(
                        {
                            "video_id": video_id,
                            "frame_id": start_frame + offset,
                            "track_id": track_id,
                            "class_name": class_name,
                            "confidence": float(np.clip(rng.normal(0.82, 0.08), 0.4, 0.99)),
                            "x1": x - box_w / 2,
                            "y1": y - box_h / 2,
                            "x2": x + box_w / 2,
                            "y2": y + box_h / 2,
                            "centroid_x": x,
                            "centroid_y": y,
                            "timestamp_s": (start_frame + offset) / fps,
                            "frame_width": width,
                            "frame_height": height,
                        }
                    )

            # Short, low-motion tracks simulate tracker fragments/ghost detections.
            for ghost_number in range(2):
                ghost_id = track_count + ghost_number + 1
                x, y = float(rng.uniform(100, width - 100)), float(rng.uniform(100, height - 100))
                for offset in range(2):
                    rows.append(
                        {
                            "video_id": video_id,
                            "frame_id": 100 + offset,
                            "track_id": ghost_id,
                            "class_name": "car",
                            "confidence": 0.45,
                            "x1": x - 25,
                            "y1": y - 18,
                            "x2": x + 25,
                            "y2": y + 18,
                            "centroid_x": x + offset,
                            "centroid_y": y + offset,
                            "timestamp_s": (100 + offset) / fps,
                            "frame_width": width,
                            "frame_height": height,
                        }
                    )

            for class_name, count in crossing_counts.items():
                truth_rows.append(
                    {
                        "video_id": video_id,
                        "class_name": class_name,
                        "vehicle_count": count,
                    }
                )
            inventory_rows.append(
                {
                    "video_id": video_id,
                    "file_path": f"data/raw/{video_id}.mp4",
                    "road_type": road_type,
                    "resolution": f"{width}x{height}",
                    "fps": fps,
                    "time_of_day": str(rng.choice(["morning", "afternoon", "evening"])),
                    "traffic_density": "medium",
                    "ground_truth_available": True,
                }
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detections_path = output / "demo_detections.csv"
    truth_path = output / "demo_ground_truth.csv"
    inventory_path = output / "video_inventory.csv"
    pd.DataFrame(rows, columns=DETECTION_COLUMNS).to_csv(detections_path, index=False)
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    return detections_path, truth_path, inventory_path
