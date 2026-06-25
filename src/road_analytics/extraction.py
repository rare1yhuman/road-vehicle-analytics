from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from uuid import uuid4

import pandas as pd

from .schema import DETECTION_COLUMNS, validate_detections


def extract_video(
    video_path: str | Path,
    output_csv: str | Path,
    *,
    video_id: str | None = None,
    model_name: str = "yolov8n.pt",
    tracker: str = "botsort.yaml",
    confidence: float = 0.3,
    iou: float = 0.5,
    image_size: int = 960,
    vehicle_classes: list[str] | None = None,
    inference_roi: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """Run Ultralytics tracking and persist one row per tracked detection."""
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Video extraction requires `pip install -e '.[vision]'`"
        ) from exc
    if importlib.util.find_spec("lap") is None:
        raise RuntimeError(
            f"The Python interpreter running this app ({sys.executable}) does not have "
            "`lap` installed. Start the dashboard with "
            "`.venv/bin/python -m streamlit run streamlit_app.py`, or install the vision "
            "dependencies into that interpreter."
        )

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    video_id = video_id or path.stem
    vehicle_classes = vehicle_classes or ["car", "motorcycle", "bus", "truck", "bicycle"]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    source_path = path
    offset_x = offset_y = 0
    cropped_path: Path | None = None
    if inference_roi is not None:
        x_min, x_max, y_min, y_max = inference_roi
        if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
            raise ValueError("Inference ROI ratios must satisfy 0 <= min < max <= 1")
        offset_x, offset_y = int(frame_width * x_min), int(frame_height * y_min)
        crop_x2, crop_y2 = int(frame_width * x_max), int(frame_height * y_max)
        crop_width = crop_x2 - offset_x
        crop_height = crop_y2 - offset_y
        crop_width -= crop_width % 2
        crop_height -= crop_height % 2
        cropped_path = output.with_name(f".inference-crop-{uuid4().hex}.mp4")
        capture = cv2.VideoCapture(str(path))
        writer = cv2.VideoWriter(
            str(cropped_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (crop_width, crop_height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create the temporary inference crop")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(
                frame[offset_y : offset_y + crop_height, offset_x : offset_x + crop_width]
            )
        capture.release()
        writer.release()
        source_path = cropped_path

    rows: list[dict[str, object]] = []
    try:
        model = YOLO(model_name)
        allowed_class_ids = [
            int(class_id)
            for class_id, class_name in model.names.items()
            if str(class_name).lower() in vehicle_classes
        ]
        if not allowed_class_ids:
            raise ValueError("None of the configured vehicle classes exist in this model")
        results = model.track(
            source=str(source_path),
            tracker=tracker,
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            classes=allowed_class_ids,
            persist=True,
            stream=True,
            verbose=False,
        )
        for frame_id, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()
            for track_id, class_id, score, (x1, y1, x2, y2) in zip(
                ids, classes, confidences, coordinates
            ):
                x1, x2 = x1 + offset_x, x2 + offset_x
                y1, y2 = y1 + offset_y, y2 + offset_y
                class_name = str(model.names[class_id]).lower()
                if class_name not in vehicle_classes:
                    continue
                rows.append(
                    {
                        "video_id": video_id,
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "class_name": class_name,
                        "confidence": score,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "centroid_x": (x1 + x2) / 2,
                        "centroid_y": (y1 + y2) / 2,
                        "timestamp_s": frame_id / fps,
                        "frame_width": frame_width,
                        "frame_height": frame_height,
                    }
                )
    finally:
        if cropped_path is not None:
            cropped_path.unlink(missing_ok=True)

    detections = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
    if not detections.empty:
        detections = validate_detections(detections)
    detections.to_csv(output, index=False)
    return detections
