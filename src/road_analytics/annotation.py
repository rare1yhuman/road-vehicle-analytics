from __future__ import annotations

from pathlib import Path

import pandas as pd


def _video_writer_codes(output: Path) -> list[str]:
    if output.suffix.lower() == ".webm":
        return ["VP80", "VP90"]
    return ["avc1", "mp4v"]


def annotate_video(
    source_path: str | Path,
    output_path: str | Path,
    detections: pd.DataFrame,
    events: pd.DataFrame,
    *,
    line_axis: str = "y",
    line_position_ratio: float = 0.55,
    roi_x_min_ratio: float = 0.0,
    roi_x_max_ratio: float = 1.0,
    roi_y_min_ratio: float = 0.0,
    roi_y_max_ratio: float = 1.0,
    max_output_dimension: int = 1280,
    camera_mode: str = "fixed",
) -> Path:
    """Render tracked boxes, IDs, ROI, counting line, and counted state."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video annotation requires opencv-python") from exc

    source = Path(source_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_output_dimension / max(width, height))
    output_width = max(2, int(round(width * scale / 2) * 2))
    output_height = max(2, int(round(height * scale / 2) * 2))
    writer = None
    for code in _video_writer_codes(output):
        candidate = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*code),
            fps,
            (output_width, output_height),
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        capture.release()
        raise RuntimeError(f"Could not create annotated video: {output}")

    by_frame = {int(key): value for key, value in detections.groupby("frame_id")}
    event_frame = (
        events.set_index("track_id")["crossing_frame"].astype(int).to_dict()
        if not events.empty
        else {}
    )
    frame_id = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (output_width, output_height))
        x1 = int(output_width * roi_x_min_ratio)
        x2 = int(output_width * roi_x_max_ratio)
        y1 = int(output_height * roi_y_min_ratio)
        y2 = int(output_height * roi_y_max_ratio)
        thickness = max(2, output_width // 900)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), thickness)
        if camera_mode == "fixed":
            line = int(
                (output_width if line_axis == "x" else output_height)
                * line_position_ratio
            )
            if line_axis == "x":
                cv2.line(
                    frame, (line, y1), (line, y2), (0, 255, 255), max(3, thickness)
                )
            else:
                cv2.line(
                    frame, (x1, line), (x2, line), (0, 255, 255), max(3, thickness)
                )

        for row in by_frame.get(frame_id, pd.DataFrame()).itertuples(index=False):
            counted = row.track_id in event_frame and frame_id >= event_frame[row.track_id]
            color = (0, 220, 0) if counted else (255, 80, 40)
            start = (int(row.x1 * scale), int(row.y1 * scale))
            end = (int(row.x2 * scale), int(row.y2 * scale))
            cv2.rectangle(frame, start, end, color, thickness)
            label = f"{row.class_name} #{row.track_id} {row.confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (start[0], max(24, start[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.45, output_width / 1800),
                color,
                max(1, thickness - 1),
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                (int((row.x1 + row.x2) / 2 * scale), int(row.y2 * scale)),
                max(3, output_width // 600),
                color,
                -1,
            )
        cv2.putText(
            frame,
            (
                f"Confirmed tracks: {sum(frame_id >= value for value in event_frame.values())}"
                if camera_mode == "moving"
                else f"Counted crossings: {sum(frame_id >= value for value in event_frame.values())}"
            ),
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.7, output_width / 1400),
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_id += 1

    capture.release()
    writer.release()
    return output
