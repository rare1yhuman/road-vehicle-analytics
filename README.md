# Road Vehicle Analytics

An end-to-end data science project that converts road video into structured
trajectory data, counts vehicles by class, classifies the road as a straight
street or intersection, and evaluates both outputs against ground truth.

## What is implemented

- YOLOv8 detection with BoT-SORT tracking
- A stable CSV schema for frame-level detections
- Confidence, short-track, and stationary-track filtering with an audit trail
- Track features: movement angle, duration, displacement, path length,
  straightness, and entry zone
- Video features: angle variance, directional entropy, entry-zone count, and
  angle-deviation proportion
- Virtual-line crossing counts with one count per track
- Dependency-free logistic regression for road-type classification
- Leave-one-video-out cross-validation and feature coefficients
- Count error analysis and confidence-threshold sensitivity analysis
- EDA plots and an interactive Streamlit dashboard
- Deterministic demo data and automated tests

## Project structure

```text
streamlit_app.py              Streamlit dashboard (main entrypoint)
configs/                      Tracker configurations
src/road_analytics/           Extraction and data science package
tests/                        Unit and end-to-end tests
data/                         Templates and local video/data directories
outputs/                      Generated reports, tables, plots, and models
```

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[vision,dashboard,dev]"
```

Run the complete workflow without needing a video:

```bash
road-analytics demo --output outputs/demo
```

This creates ten labeled synthetic videos' worth of trajectories, runs all
analysis stages, and writes the outputs under `outputs/demo`.

Launch the dashboard:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

The application will be available at http://localhost:8501 in your browser.

The dashboard provides two input options:

1. **Traffic video** — upload one video file. YOLOv8 and BoT-SORT automatically
   extract tracked detections before the analytical pipeline runs.
2. **Detection CSV** — upload one detection CSV and run the analytical pipeline
   directly.

After video analysis, the extracted detection CSV can be downloaded from the
dashboard and reused with the second option.

Choose **Moving camera** for handheld or vehicle-mounted footage. This mode
uses confirmed persistent tracks for counting instead of an invalid fixed
image line, and suppresses trajectory-based road classification. Each
dashboard analysis is written to an isolated run directory so previous or
concurrent results cannot contaminate it.

## Analyze real video

Place videos in `data/raw/`, then extract tracked detections:

```bash
road-analytics extract data/raw/clip_001.mp4 \
  --output data/processed/clip_001.csv \
  --video-id clip_001
```

The first run may download the configured YOLO weights. To combine multiple
clips into one dataset:

```bash
python -c "import pandas as pd, glob; pd.concat([pd.read_csv(p) for p in glob.glob('data/processed/*.csv')]).to_csv('data/processed/all_detections.csv', index=False)"
```

Copy and complete the supplied inventory and ground-truth templates:

- `data/video_inventory_template.csv`
- `data/ground_truth/counts_template.csv`

Then run:

```bash
road-analytics analyze data/processed/all_detections.csv \
  --inventory data/video_inventory.csv \
  --ground-truth data/ground_truth/counts.csv \
  --output outputs/real_analysis
```

## Detection CSV contract

Every row is one tracked detection in one frame:

| Column | Meaning |
|---|---|
| `video_id` | Stable clip identifier |
| `frame_id`, `timestamp_s` | Temporal position |
| `track_id` | Tracker-assigned vehicle identity |
| `class_name`, `confidence` | Detector output |
| `x1`, `y1`, `x2`, `y2` | Bounding box |
| `centroid_x`, `centroid_y` | Trajectory coordinate |
| `frame_width`, `frame_height` | Needed for normalized zones and lines |

## Main outputs

| File | Purpose |
|---|---|
| `cleaned_detections.csv` | Valid frame-level observations |
| `track_cleaning_audit.csv` | Why each track was kept or removed |
| `track_features.csv` | One engineered row per vehicle |
| `video_features.csv` | One model-ready row per clip |
| `crossing_events.csv` | Counted vehicles and crossing time |
| `vehicle_counts.csv` | Counts by video and class |
| `road_type_predictions.csv` | Road prediction and probability |
| `road_classifier_loocv.csv` | Held-out prediction for each labeled video |
| `count_evaluation.csv` | Actual/predicted count errors |
| `confidence_sensitivity.csv` | Threshold impact on count accuracy |
| `plots/` | Confidence, class, trajectory, time-series, and track-length EDA |

## Tests

```bash
pytest
```
