from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .extraction import extract_video
from .pipeline import run_analysis
from .synthetic import generate_demo_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="road-analytics",
        description="Extract and analyze road-vehicle trajectories.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Generate demo data and run the full analysis")
    demo.add_argument("--output", default="outputs/demo")
    demo.add_argument("--videos-per-type", type=int, default=5)

    extract = subparsers.add_parser("extract", help="Extract tracked detections from a video")
    extract.add_argument("video")
    extract.add_argument("--output", required=True)
    extract.add_argument("--video-id")
    extract.add_argument("--config")
    extract.add_argument(
        "--inference-roi",
        nargs=4,
        type=float,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
    )

    analyze = subparsers.add_parser("analyze", help="Run cleaning, EDA, counting and modeling")
    analyze.add_argument("detections")
    analyze.add_argument("--output", default="outputs/analysis")
    analyze.add_argument("--config")
    analyze.add_argument("--inventory")
    analyze.add_argument("--ground-truth")
    analyze.add_argument("--track-ground-truth")
    analyze.add_argument("--camera-mode", choices=["fixed", "moving"], default="fixed")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "demo":
        root = Path(args.output)
        detections, truth, inventory = generate_demo_dataset(
            root / "data", videos_per_type=args.videos_per_type
        )
        report = run_analysis(
            detections,
            root,
            inventory_path=inventory,
            ground_truth_path=truth,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "extract":
        config = load_config(args.config)["extraction"]
        detections = extract_video(
            args.video,
            args.output,
            video_id=args.video_id,
            model_name=config["model"],
            tracker=config["tracker"],
            confidence=config["confidence"],
            iou=config["iou"],
            image_size=config["image_size"],
            vehicle_classes=config["vehicle_classes"],
            inference_roi=tuple(args.inference_roi) if args.inference_roi else None,
        )
        print(f"Wrote {len(detections):,} detection rows to {args.output}")
    elif args.command == "analyze":
        report = run_analysis(
            args.detections,
            args.output,
            config_path=args.config,
            inventory_path=args.inventory,
            ground_truth_path=args.ground_truth,
            track_ground_truth_path=args.track_ground_truth,
            camera_mode=args.camera_mode,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
