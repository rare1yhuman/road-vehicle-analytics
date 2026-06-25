# Road Vehicle Analytics — PPT Slide Content

> Copy each slide's content into your PowerPoint / Google Slides.
> Speaker notes are provided below each slide.

---

## Slide 1 — Title Slide

**Road Vehicle Analytics**
*An End-to-End Video-to-Insight Pipeline for Traffic Intelligence*

- Your Name
- College Name · Internship Final Project
- Date: June 2026

> **Speaker Notes:** Introduce yourself. This project converts raw traffic video into structured data, counts vehicles, measures speed, and classifies road type — all through an automated pipeline with a live dashboard.

---

## Slide 2 — Problem Statement

### Why This Matters

- Manual traffic surveys are **expensive, slow, and error-prone**
- Existing tools require proprietary hardware or cloud APIs
- No open-source pipeline covers **detection → tracking → counting → classification → dashboard** in one package

### Our Solution
- A fully local, end-to-end Python pipeline
- Works with any traffic video (fixed or moving camera)
- Produces structured analytics + interactive dashboard

> **Speaker Notes:** Traffic departments still rely on humans sitting at intersections and counting cars. This project automates the entire workflow using computer vision and statistical analysis, producing the same output in minutes instead of hours.

---

## Slide 3 — Objectives

1. **Detect & track** vehicles in traffic video using YOLOv8 + BoT-SORT
2. **Clean** noisy tracker output — remove false positives, short tracks, and stationary objects
3. **Engineer features** — movement angle, speed, displacement, entry zone, path straightness
4. **Count vehicles** by class using virtual line crossing (fixed camera) or confirmed tracks (moving camera)
5. **Classify road type** — straight road vs. intersection using trajectory statistics
6. **Visualize results** through an interactive Streamlit dashboard with live video annotation

> **Speaker Notes:** Each objective maps to a module in the codebase. The pipeline is modular — you can use individual stages independently or run the full workflow.

---

## Slide 4 — System Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Raw Video   │ →  │  YOLOv8 +    │ →  │  Detection   │ →  │  Cleaning &  │
│  Input       │    │  BoT-SORT    │    │  CSV         │    │  Audit Trail │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Streamlit   │ ←  │  Road Type   │ ←  │  Vehicle     │ ←  │  Feature     │
│  Dashboard   │    │  Classifier  │    │  Counting    │    │  Engineering │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Input:** Traffic video (MP4) or pre-extracted detection CSV
**Output:** Dashboard with KPIs, charts, annotated video, downloadable CSVs

> **Speaker Notes:** The architecture follows a clean data pipeline pattern. Each box is an independent Python module. Data flows as CSV files between stages so any stage can be re-run independently.

---

## Slide 5 — Technology Stack

| Layer | Technology |
|---|---|
| **Object Detection** | YOLOv8 (Ultralytics) — pre-trained on COCO |
| **Multi-Object Tracking** | BoT-SORT / ByteTrack |
| **Data Processing** | Pandas, NumPy |
| **Video I/O** | OpenCV (cv2) |
| **Classification** | Custom logistic regression (zero dependencies) |
| **Visualization** | Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Configuration** | YAML |
| **Testing** | Pytest |
| **Language** | Python 3.10+ |

> **Speaker Notes:** We deliberately avoided heavy ML frameworks for the analytics stages. The logistic regression classifier was built from scratch to demonstrate understanding of the math rather than calling sklearn.

---

## Slide 6 — Stage 1: Detection & Tracking

### What Happens
- YOLOv8 runs on every frame, detecting: **car, motorcycle, bus, truck, bicycle**
- BoT-SORT assigns persistent track IDs across frames
- Each detection → one row in `video_detections.csv`

### Key Parameters
| Parameter | Value |
|---|---|
| Confidence threshold | 0.30 |
| IoU threshold | 0.50 |
| Inference size | 960 px |
| Supported classes | car, motorcycle, bus, truck, bicycle |

### Output Schema
`video_id, frame_id, track_id, class_name, confidence, x1, y1, x2, y2, centroid_x, centroid_y, timestamp_s`

> **Speaker Notes:** BoT-SORT is a state-of-the-art tracker that combines appearance features with motion prediction. It handles occlusion better than simple IoU trackers. The output is a standardized CSV — every row is one bounding box in one frame.

---

## Slide 7 — Stage 2: Track Cleaning

### Problem
- Raw tracker output contains noise: false positives, flickering detections, stationary objects (parked cars)

### Cleaning Rules
| Filter | Threshold | Tracks Removed |
|---|---|---|
| Low confidence | < 0.35 avg | Weak detections |
| Short tracks | < 5 frames | Noise/artifacts |
| Stationary objects | < 12 px displacement | Parked vehicles |

### Audit Trail
- Every track gets a **kept/removed** verdict with a reason
- `track_cleaning_audit.csv` — full transparency, no black box

> **Speaker Notes:** The audit trail is critical for reproducibility. If a vehicle was miscounted, you can trace back to which track was removed and why. This is a data science best practice.

---

## Slide 8 — Stage 3: Feature Engineering

### Track-Level Features (per vehicle)
- **Movement angle** — dominant direction of travel (degrees)
- **Duration** — how long the vehicle was visible (seconds)
- **Displacement** — straight-line distance from entry to exit (pixels)
- **Path length** — total distance traveled including curves (pixels)
- **Straightness ratio** — displacement ÷ path length (1.0 = perfectly straight)
- **Entry zone** — which edge of frame the vehicle entered from
- **Average speed** — pixels/second → converted to km/h

### Video-Level Features (per clip)
- **Angle variance** — spread of travel directions
- **Directional entropy** — how many different directions vehicles travel
- **Entry-zone count** — number of distinct approach directions
- **Angle deviation proportion** — fraction of tracks not aligned with the majority

> **Speaker Notes:** These features serve two purposes: they feed the road type classifier, and they appear directly in the dashboard as analytics. Speed is estimated from pixel displacement — for real calibration, you'd need ground-truth distance markers.

---

## Slide 9 — Stage 4: Vehicle Counting

### Two Counting Modes

**Fixed Camera** — Virtual Line Crossing
- A horizontal/vertical line is placed at a configurable position
- Each track is counted once when its centroid crosses the line
- Hysteresis prevents double-counting from jitter

**Moving Camera** — Confirmed Track Counting
- No fixed reference line is valid for moving footage
- Counts all tracks that survive cleaning (minimum frames + displacement)

### Output
- `crossing_events.csv` — when each vehicle crossed
- `vehicle_counts.csv` — totals by class (car: 25, motorcycle: 58, truck: 4, etc.)

> **Speaker Notes:** The counting mode is selected per-analysis. For dashcam footage, line crossing doesn't work because the camera itself is moving, so we fall back to counting confirmed persistent tracks.

---

## Slide 10 — Stage 5: Road Classification

### Rule-Based Heuristic
- Uses directional entropy and angle deviation to classify:
  - **Straight road:** low entropy, vehicles mostly travel in 1–2 directions
  - **Intersection:** high entropy, vehicles approach from many angles

### Statistical Classifier (when labeled data available)
- Custom logistic regression trained on video-level features
- Leave-one-video-out cross-validation
- Feature coefficients exported for interpretability

### Limitations
- Requires multiple labeled videos of both types to train
- Disabled for moving-camera footage (image-space motion ≠ road geometry)

> **Speaker Notes:** The classifier is a pedagogical choice — we implemented logistic regression from scratch using gradient descent rather than importing sklearn. This demonstrates understanding of the optimization math.

---

## Slide 11 — Dashboard Demo

### Interactive Streamlit Dashboard

| Section | What It Shows |
|---|---|
| **KPI Cards** | Total vehicles, unique tracks, crossing events, avg speed, camera mode |
| **Annotated Video** | Bounding boxes, track IDs, counting line, counted status — played inline |
| **Vehicle Breakdown** | Horizontal bar chart color-coded by vehicle type |
| **Traffic Volume** | Time-series line chart (5-second bins) |
| **Speed Distribution** | Box plot per vehicle class |
| **Trajectory Map** | Scatter plot of all vehicle paths |
| **Road Classification** | Predicted road type with confidence |
| **Data Downloads** | All CSVs available for download |

### Live Processing
- Upload a video → watch live progress bar: Extracting → Analysing → Annotating → Done
- Results appear automatically when complete

> **Speaker Notes:** [DEMO THE DASHBOARD LIVE HERE] — Upload a video, show the progress bar, then walk through each dashboard section.

---

## Slide 12 — Results & Metrics

### Sample Analysis (from test video)

| Metric | Value |
|---|---|
| Raw detections | 16,343 |
| After cleaning | 14,968 (91.6% retained) |
| Tracks detected | 157 |
| Tracks after cleaning | 88 (56% retained) |
| Crossing events | 87 |
| Vehicle classes found | car (25), motorcycle (58), truck (4) |
| Processing time | ~2–3 min on GPU |

### What the Cleaning Found
- 69 tracks removed as noise (short tracks, stationary objects, low confidence)
- Full audit trail preserved in `track_cleaning_audit.csv`

> **Speaker Notes:** The 56% track retention rate is expected — many short-lived false positives are produced by the tracker, especially in crowded scenes. The cleaning module filters these systematically.

---

## Slide 13 — Challenges & Learnings

### Technical Challenges
- **Video codec compatibility** — browser players reject MPEG-4 Part 2; solved by encoding to WebM (VP8)
- **Moving camera detection** — image-space trajectories are meaningless for road classification; implemented separate counting mode
- **Track fragmentation** — BoT-SORT sometimes splits one vehicle into multiple tracks; displacement filter catches most
- **Matplotlib API changes** — `boxplot(labels=...)` removed in v3.9+; adapted to use `set_xticklabels()`

### Key Learnings
- Modular pipeline architecture makes debugging 10x easier
- Audit trails are essential — you can't debug ML outputs without knowing what was filtered
- CSV as intermediate format = maximum transparency and portability

> **Speaker Notes:** Each challenge has a corresponding code solution. The modular architecture was the single biggest win — when detection was failing, we could isolate the problem to one module without touching the rest.

---

## Slide 14 — Future Scope & Conclusion

### Future Enhancements
- **Speed calibration** — use known landmark distances for real-world speed (km/h)
- **Multi-camera fusion** — combine overlapping camera views for wider coverage
- **Real-time streaming** — process RTSP feeds instead of recorded files
- **Anomaly detection** — flag wrong-way driving, sudden stops, unusual speed
- **Cloud deployment** — containerize with Docker for scalable traffic monitoring

### Conclusion
- Built a **complete, working pipeline** from raw video to interactive dashboard
- **13 Python modules**, automated tests, CLI interface, and Streamlit UI
- Demonstrates: computer vision, data engineering, statistical modeling, and software engineering
- **Fully open-source**, runs locally, no cloud dependencies

> **Speaker Notes:** The project demonstrates breadth across the data science stack — from GPU-accelerated inference to hand-coded logistic regression to interactive visualization. Every stage is testable and auditable.

---

## Slide 15 — Thank You / Q&A

**Thank You**

*Questions?*

Repository: `github.com/your-username/road-vehicle-analytics`

Demo command:
```bash
streamlit run streamlit_app.py
```

> **Speaker Notes:** Open the floor for questions. Have the dashboard running in the background to demo specific features if asked.
