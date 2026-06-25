"""Road Vehicle Analytics — Streamlit Dashboard.

Run with:
    .venv/bin/python -m streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Road Vehicle Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        color: white;
    }
    .kpi-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .kpi-value { font-size: 2.2rem; font-weight: 700; color: #38bdf8; line-height: 1; }
    .kpi-sub   { font-size: 0.75rem; color: #64748b; margin-top: 4px; }
    .section-header {
        font-size: 1.05rem; font-weight: 600; color: #e2e8f0;
        border-left: 3px solid #38bdf8; padding-left: 10px; margin: 24px 0 12px;
    }
    .status-pill {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Lazy imports (heavy — only when needed)
# ---------------------------------------------------------------------------
@st.cache_resource
def _import_pipeline():
    from road_analytics.pipeline import run_analysis
    from road_analytics.extraction import extract_video
    from road_analytics.annotation import annotate_video
    from road_analytics.config import load_config
    from road_analytics.visualization import (
        load_analysis_frames,
        prepare_traffic_volume,
        prepare_vehicle_breakdown,
        prepare_speed_distribution,
        prepare_trajectory_points,
        VEHICLE_TYPE_COLORS,
    )
    return dict(
        run_analysis=run_analysis,
        extract_video=extract_video,
        annotate_video=annotate_video,
        load_config=load_config,
        load_analysis_frames=load_analysis_frames,
        prepare_traffic_volume=prepare_traffic_volume,
        prepare_vehicle_breakdown=prepare_vehicle_breakdown,
        prepare_speed_distribution=prepare_speed_distribution,
        prepare_trajectory_points=prepare_trajectory_points,
        VEHICLE_TYPE_COLORS=VEHICLE_TYPE_COLORS,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRACKER_MAP = {
    "Moving-camera BoT-SORT": str(ROOT / "configs" / "botsort_moving.yaml"),
    "Standard BoT-SORT": "botsort.yaml",
    "ByteTrack": "bytetrack.yaml",
}

ST_RUNS = ROOT / "outputs" / "st_runs"
ST_RUNS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "analysis_dir": None,       # Path to current analysis output
        "summary": None,            # dict from analysis_summary.json
        "processing": False,
        "progress_msg": "",
        "progress_val": 0.0,
        "error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_frames(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    mods = _import_pipeline()
    return mods["load_analysis_frames"](analysis_dir)


def _find_latest_run() -> Path | None:
    runs_dir = ROOT / "outputs" / "web_runs"
    if not runs_dir.exists():
        return None
    candidates = []
    for d in runs_dir.iterdir():
        if (d / "analysis_summary.json").exists():
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        return None
    return max(candidates)[1]


def _load_summary(analysis_dir: Path) -> dict:
    p = analysis_dir / "analysis_summary.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _set_analysis(path: Path):
    st.session_state["analysis_dir"] = path
    st.session_state["summary"] = _load_summary(path)
    st.session_state["error"] = None


# ---------------------------------------------------------------------------
# Processing thread — runs pipeline steps and updates progress via file flag
# ---------------------------------------------------------------------------
def _run_video_pipeline(
    upload_path: Path,
    output_dir: Path,
    *,
    model_name: str,
    tracker: str,
    confidence: float,
    image_size: int,
    camera_mode: str,
    status_path: Path,
):
    mods = _import_pipeline()
    try:
        cfg = mods["load_config"]()

        status_path.write_text(json.dumps({"step": "extracting", "pct": 10}))
        detections = mods["extract_video"](
            upload_path,
            output_dir / "video_detections.csv",
            video_id=upload_path.stem,
            model_name=model_name,
            tracker=TRACKER_MAP.get(tracker, "botsort.yaml"),
            confidence=confidence,
            iou=cfg["extraction"]["iou"],
            image_size=image_size,
            vehicle_classes=cfg["extraction"]["vehicle_classes"],
        )

        if detections.empty:
            status_path.write_text(json.dumps({"step": "error", "msg": "No vehicles detected in this video."}))
            return

        status_path.write_text(json.dumps({"step": "analysing", "pct": 55}))
        mods["run_analysis"](
            output_dir / "video_detections.csv",
            output_dir,
            counting_options={"line_position_ratio": 0.55, "direction": "both"},
            camera_mode=camera_mode,
        )

        status_path.write_text(json.dumps({"step": "annotating", "pct": 80}))
        det_csv = output_dir / "cleaned_detections.csv"
        evt_csv = output_dir / "crossing_events.csv"
        if det_csv.exists() and evt_csv.exists():
            mods["annotate_video"](
                upload_path,
                output_dir / "annotated_video.webm",
                pd.read_csv(det_csv),
                pd.read_csv(evt_csv),
                line_position_ratio=0.55,
                camera_mode=camera_mode,
            )

        status_path.write_text(json.dumps({"step": "done", "pct": 100}))

    except Exception as exc:
        status_path.write_text(json.dumps({"step": "error", "msg": str(exc)}))


def _run_csv_pipeline(
    csv_path: Path,
    output_dir: Path,
    *,
    camera_mode: str,
    status_path: Path,
):
    mods = _import_pipeline()
    try:
        status_path.write_text(json.dumps({"step": "analysing", "pct": 30}))
        mods["run_analysis"](csv_path, output_dir, camera_mode=camera_mode)
        status_path.write_text(json.dumps({"step": "done", "pct": 100}))
    except Exception as exc:
        status_path.write_text(json.dumps({"step": "error", "msg": str(exc)}))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚗 Road Vehicle Analytics")
    st.markdown("---")

    source = st.radio(
        "Data source",
        ["Upload Video", "Upload CSV", "Load Latest Run"],
        index=0,
    )

    st.markdown("---")
    # --- Settings ---
    st.markdown("### ⚙️ Settings")
    confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.30, 0.05)
    model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8m.pt"])
    tracker = st.selectbox("Tracker", list(TRACKER_MAP.keys()))
    camera_mode = st.radio("Camera mode", ["moving", "fixed"])
    image_size = st.select_slider("Inference size (px)", options=[640, 960, 1280], value=960)

    st.markdown("---")

    # --- Trigger buttons per source ---
    if source == "Upload Video":
        video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if st.button("▶ Run Analysis", disabled=video_file is None, use_container_width=True):
            run_id = uuid4().hex[:12]
            out_dir = ST_RUNS / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = Path(video_file.name).suffix or ".mp4"
            upload_path = out_dir / f"uploaded{suffix}"
            upload_path.write_bytes(video_file.read())
            status_path = out_dir / "_status.json"
            status_path.write_text(json.dumps({"step": "starting", "pct": 0}))
            t = threading.Thread(
                target=_run_video_pipeline,
                kwargs=dict(
                    upload_path=upload_path,
                    output_dir=out_dir,
                    model_name=model_name,
                    tracker=tracker,
                    confidence=confidence,
                    image_size=image_size,
                    camera_mode=camera_mode,
                    status_path=status_path,
                ),
                daemon=True,
            )
            t.start()
            st.session_state["processing"] = True
            st.session_state["analysis_dir"] = out_dir
            st.session_state["summary"] = None
            st.session_state["error"] = None
            st.rerun()

    elif source == "Upload CSV":
        csv_file = st.file_uploader("Upload detections CSV", type=["csv"])
        if st.button("▶ Run Analysis", disabled=csv_file is None, use_container_width=True):
            run_id = uuid4().hex[:12]
            out_dir = ST_RUNS / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "uploaded.csv"
            csv_path.write_bytes(csv_file.read())
            status_path = out_dir / "_status.json"
            status_path.write_text(json.dumps({"step": "starting", "pct": 0}))
            t = threading.Thread(
                target=_run_csv_pipeline,
                kwargs=dict(
                    csv_path=csv_path,
                    output_dir=out_dir,
                    camera_mode=camera_mode,
                    status_path=status_path,
                ),
                daemon=True,
            )
            t.start()
            st.session_state["processing"] = True
            st.session_state["analysis_dir"] = out_dir
            st.session_state["summary"] = None
            st.session_state["error"] = None
            st.rerun()

    elif source == "Load Latest Run":
        if st.button("📂 Load Latest", use_container_width=True):
            latest = _find_latest_run()
            if latest:
                _set_analysis(latest)
                st.session_state["processing"] = False
                st.rerun()
            else:
                st.warning("No completed runs found in outputs/web_runs/")

    # Filters shown only when data is loaded
    if st.session_state["analysis_dir"] and not st.session_state["processing"]:
        analysis_dir = Path(st.session_state["analysis_dir"])
        frames = _load_frames(analysis_dir)
        det = frames.get("detections", pd.DataFrame())
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        all_classes = sorted(det["class_name"].dropna().unique().tolist()) if not det.empty and "class_name" in det.columns else []
        all_segments = sorted(det["video_id"].dropna().unique().tolist()) if not det.empty and "video_id" in det.columns else []
        sel_classes = st.multiselect("Vehicle types", all_classes, default=all_classes)
        sel_segments = st.multiselect("Road segments", all_segments, default=all_segments)
        st.session_state["sel_classes"] = sel_classes
        st.session_state["sel_segments"] = sel_segments

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.markdown("# 🚗 Road Vehicle Analytics Dashboard")
st.markdown("Upload a video or CSV to begin. Results appear here as soon as processing completes.")
st.markdown("---")

analysis_dir = st.session_state.get("analysis_dir")
processing = st.session_state.get("processing", False)

# ── Live progress during processing ──────────────────────────────────────────
if processing and analysis_dir:
    analysis_dir = Path(analysis_dir)
    status_path = analysis_dir / "_status.json"

    step_labels = {
        "starting":   ("⏳ Starting up…",           5),
        "extracting": ("🔍 Running YOLO tracking…", 35),
        "analysing":  ("📊 Analysing detections…",  70),
        "annotating": ("🎬 Rendering annotated video…", 90),
        "done":       ("✅ Complete!",               100),
    }

    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    log_placeholder = st.empty()

    # Poll until done
    import time
    while True:
        if status_path.exists():
            try:
                info = json.loads(status_path.read_text())
            except Exception:
                info = {"step": "starting"}
            step = info.get("step", "starting")

            if step == "error":
                st.session_state["processing"] = False
                st.session_state["error"] = info.get("msg", "Unknown error")
                st.rerun()

            label, pct = step_labels.get(step, ("⏳ Processing…", 10))
            status_placeholder.markdown(f"### {label}")
            progress_bar.progress(pct / 100)
            log_placeholder.caption(f"Step: `{step}` | Output: `{analysis_dir.name}`")

            if step == "done":
                st.session_state["processing"] = False
                _set_analysis(analysis_dir)
                time.sleep(0.5)
                st.rerun()

        time.sleep(1.5)
        st.rerun()

# ── Error state ──────────────────────────────────────────────────────────────
if st.session_state.get("error"):
    st.error(f"❌ **Processing failed:** {st.session_state['error']}")
    if st.button("Clear error"):
        st.session_state["error"] = None
        st.session_state["analysis_dir"] = None
        st.rerun()

# ── Blank state ──────────────────────────────────────────────────────────────
if not analysis_dir or processing:
    st.info("👈 Use the sidebar to upload a video or CSV file and start an analysis.")
    st.stop()

# ── Dashboard ─────────────────────────────────────────────────────────────────
analysis_dir = Path(analysis_dir)
summary = st.session_state.get("summary") or _load_summary(analysis_dir)
frames = _load_frames(analysis_dir)

# Apply filters
sel_classes = st.session_state.get("sel_classes", [])
sel_segments = st.session_state.get("sel_segments", [])

def _filter(df: pd.DataFrame, class_col="class_name", seg_col="video_id") -> pd.DataFrame:
    if df.empty:
        return df
    if sel_classes and class_col in df.columns:
        df = df[df[class_col].isin(sel_classes)]
    if sel_segments and seg_col in df.columns:
        df = df[df[seg_col].isin(sel_segments)]
    return df

det = _filter(frames.get("detections", pd.DataFrame()))
tracks = _filter(frames.get("tracks", pd.DataFrame()))
counts = _filter(frames.get("counts", pd.DataFrame()))
events = _filter(frames.get("events", pd.DataFrame()))

mods = _import_pipeline()
VEHICLE_TYPE_COLORS = mods["VEHICLE_TYPE_COLORS"]

# ── Section 1: KPI Cards ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Summary</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

total_vehicles = int(counts["vehicle_count"].sum()) if not counts.empty and "vehicle_count" in counts.columns else 0
unique_tracks = summary.get("tracks_after_cleaning", len(tracks) if not tracks.empty else 0)
crossing_events = summary.get("crossing_events", len(events) if not events.empty else 0)
cam_mode = summary.get("camera_mode", camera_mode).capitalize()
counting_method = summary.get("counting_method", "—").replace("_", " ").title()

# Avg speed across all tracks
if not tracks.empty and "avg_speed_kmh" in tracks.columns:
    avg_speed_val = f"{tracks['avg_speed_kmh'].mean():.1f}"
    avg_speed_sub = "km/h across all tracks"
else:
    avg_speed_val = "—"
    avg_speed_sub = "no speed data"

for col, label, val, sub in [
    (c1, "Total Vehicles", total_vehicles, counting_method),
    (c2, "Unique Tracks", unique_tracks, "after cleaning"),
    (c3, "Crossing Events", crossing_events, "line crossings"),
    (c4, "Avg Speed", avg_speed_val, avg_speed_sub),
    (c5, "Camera Mode", cam_mode, summary.get("analysis_status", "complete").replace("_", " ")),
]:
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{val}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Annotated Video (right below summary) ─────────────────────────────────────
_video_path_webm = analysis_dir / "annotated_video.webm"
_video_path_mp4  = analysis_dir / "annotated_video.mp4"

if _video_path_webm.exists():
    st.markdown('<div class="section-header">🎬 Annotated Video</div>', unsafe_allow_html=True)
    st.video(_video_path_webm.read_bytes(), format="video/webm")
elif _video_path_mp4.exists():
    st.markdown('<div class="section-header">🎬 Annotated Video</div>', unsafe_allow_html=True)
    st.caption("⚠️ MP4 playback may be limited by browser codec support.")
    st.video(_video_path_mp4.read_bytes(), format="video/mp4")

# ── Vehicle Trajectories (below video) ────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Vehicle Trajectories</div>', unsafe_allow_html=True)
traj = mods["prepare_trajectory_points"](det, tracks)

if not traj.empty and "centroid_x" in traj.columns:
    sample = traj.sample(min(6000, len(traj)), random_state=42) if len(traj) > 6000 else traj
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    for cls, grp in sample.groupby("class_name"):
        color = VEHICLE_TYPE_COLORS.get(str(cls).lower(), "#6366f1")
        ax.scatter(grp["centroid_x"], grp["centroid_y"], s=2, alpha=0.45, color=color, label=cls)
    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)", color="#94a3b8")
    ax.set_ylabel("Y (pixels)", color="#94a3b8")
    ax.tick_params(colors="#e2e8f0")
    ax.spines[:].set_visible(False)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", markerscale=5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No trajectory data available.")

# ── Section 2: Vehicle Breakdown ──────────────────────────────────────────────
st.markdown('<div class="section-header">🚙 Vehicle Breakdown</div>', unsafe_allow_html=True)
breakdown = mods["prepare_vehicle_breakdown"](counts, tracks)

if not breakdown.empty:
    fig, ax = plt.subplots(figsize=(9, max(2.5, len(breakdown) * 0.55)))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    colors = [VEHICLE_TYPE_COLORS.get(c.lower(), "#6366f1") for c in breakdown["class_name"]]
    bars = ax.barh(breakdown["class_name"], breakdown["vehicle_count"], color=colors, height=0.55)
    ax.bar_label(bars, fmt="%d", padding=4, color="#e2e8f0", fontsize=11)
    ax.set_xlabel("Vehicles", color="#94a3b8")
    ax.tick_params(colors="#e2e8f0")
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", color="#334155", linewidth=0.6)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No vehicle count data available.")

# ── Section 3: Traffic Volume Over Time ───────────────────────────────────────
st.markdown('<div class="section-header">📈 Traffic Volume Over Time</div>', unsafe_allow_html=True)
volume = mods["prepare_traffic_volume"](det, events, time_bin_s=5)

if not volume.empty and "time_bin_s" in volume.columns:
    pivot = volume.pivot_table(index="time_bin_s", columns="class_name", values="vehicle_count", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    for cls in pivot.columns:
        color = VEHICLE_TYPE_COLORS.get(cls.lower(), "#6366f1")
        ax.plot(pivot.index, pivot[cls], label=cls, color=color, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Time (seconds)", color="#94a3b8")
    ax.set_ylabel("Vehicle count", color="#94a3b8")
    ax.tick_params(colors="#e2e8f0")
    ax.spines[:].set_visible(False)
    ax.grid(color="#334155", linewidth=0.5)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No time-series data available.")

# ── Section 4: Speed Distribution ────────────────────────────────────────────
st.markdown('<div class="section-header">⚡ Speed Distribution by Vehicle Type</div>', unsafe_allow_html=True)
speed_df = mods["prepare_speed_distribution"](tracks)

if not speed_df.empty and "avg_speed_kmh" in speed_df.columns:
    classes = speed_df["class_name"].dropna().unique()
    data_by_class = [speed_df[speed_df["class_name"] == c]["avg_speed_kmh"].dropna().values for c in classes]
    data_by_class = [d for d in data_by_class if len(d) > 0]
    valid_classes = [c for c, d in zip(classes, [speed_df[speed_df["class_name"] == c]["avg_speed_kmh"].dropna().values for c in classes]) if len(d) > 0]

    if data_by_class:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        bp = ax.boxplot(
            data_by_class,
            patch_artist=True,
            medianprops=dict(color="#f8fafc", linewidth=2),
            whiskerprops=dict(color="#94a3b8"),
            capprops=dict(color="#94a3b8"),
            flierprops=dict(markerfacecolor="#ef4444", markersize=3, alpha=0.4),
        )
        ax.set_xticks(range(1, len(valid_classes) + 1))
        ax.set_xticklabels(valid_classes)
        for patch, cls in zip(bp["boxes"], valid_classes):
            patch.set_facecolor(VEHICLE_TYPE_COLORS.get(cls.lower(), "#6366f1"))
            patch.set_alpha(0.75)
        ax.set_ylabel("Speed (km/h)", color="#94a3b8")
        ax.tick_params(colors="#e2e8f0")
        ax.spines[:].set_visible(False)
        ax.grid(axis="y", color="#334155", linewidth=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("No speed data available.")



# ── Section 7: Road Type Predictions ─────────────────────────────────────────
road_pred_path = analysis_dir / "road_rule_predictions.csv"
if road_pred_path.exists():
    st.markdown('<div class="section-header">🛣️ Road Type Predictions</div>', unsafe_allow_html=True)
    road_df = pd.read_csv(road_pred_path)
    st.dataframe(road_df, use_container_width=True)

# ── Section 8: Raw Data & Downloads ──────────────────────────────────────────
st.markdown('<div class="section-header">📥 Data & Downloads</div>', unsafe_allow_html=True)
csv_files = sorted(analysis_dir.glob("*.csv"))

if csv_files:
    with st.expander("View & download CSVs", expanded=False):
        tabs = st.tabs([f.name for f in csv_files])
        for tab, csv_path in zip(tabs, csv_files):
            with tab:
                df = pd.read_csv(csv_path)
                st.dataframe(df.head(500), use_container_width=True)
                st.download_button(
                    label=f"⬇ Download {csv_path.name}",
                    data=csv_path.read_bytes(),
                    file_name=csv_path.name,
                    mime="text/csv",
                    key=f"dl_{csv_path.name}",
                )
else:
    st.info("No CSV outputs found.")

# ── Section 9: EDA Plots ──────────────────────────────────────────────────────
plots_dir = analysis_dir / "plots"
if plots_dir.exists():
    plot_files = sorted(plots_dir.glob("*.png"))
    if plot_files:
        st.markdown('<div class="section-header">📉 EDA Plots</div>', unsafe_allow_html=True)
        with st.expander("View generated plots", expanded=False):
            cols = st.columns(2)
            for i, pf in enumerate(plot_files):
                cols[i % 2].image(str(pf), use_container_width=True, caption=pf.stem.replace("_", " "))
