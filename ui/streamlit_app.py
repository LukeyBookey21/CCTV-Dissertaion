"""Streamlit UI for the CCTV dissertation project."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cctv_dissertation.analysis import summarize_detection_report
from cctv_dissertation.detection import run_yolo_detection, write_detection_report
from cctv_dissertation.ingest import ingest_video
from cctv_dissertation.storage import (
    DEFAULT_DB_PATH,
    import_detection_report,
    query_detections,
    query_tracks,
    store_tracks,
)
from cctv_dissertation.tracking import generate_tracks

DETECTIONS_DIR = Path("data/detections")
UPLOADS_DIR = Path("data/uploads")


def list_detection_files() -> List[Path]:
    if not DETECTIONS_DIR.exists():
        return []
    return sorted(DETECTIONS_DIR.glob("*.json"))


@st.cache_data(show_spinner=False)
def cached_summary(path_str: str) -> dict:
    return summarize_detection_report(path_str)


@st.cache_data(show_spinner=False)
def load_detection_report(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text())


@st.cache_data(show_spinner=False)
def render_frame_image(
    video_path: str,
    frame_index: int,
    detections: Tuple[Tuple[float, ...], ...],
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {frame_index}")
    for det in detections:
        x1, y1, x2, y2, r, g, b, label = det
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (int(b), int(g), int(r)),
            2,
        )
        cv2.putText(
            frame,
            label,
            (int(x1), max(0, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (int(b), int(g), int(r)),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    st.set_page_config(
        page_title="Forensic Video Toolkit",
        page_icon="🎥",
        layout="wide",
    )
    st.title("Forensic Video Analysis Dashboard")
    st.caption(
        "Upload footage, run YOLO detections, and query results "
        "without leaving the browser."
    )

    detection_files = list_detection_files()
    saved_path = st.session_state.get("selected_report_path")
    selected_idx = 0
    for idx, path in enumerate(detection_files):
        if saved_path and str(path.resolve()) == saved_path:
            selected_idx = idx
            break

    with st.sidebar:
        st.header("Context")
        if detection_files:
            selected_file = st.selectbox(
                "Detection report",
                options=detection_files,
                format_func=lambda p: p.name,
                index=selected_idx,
            )
        else:
            selected_file = None
            st.info("No detection reports yet. Upload a video below.")
        db_path = st.text_input("SQLite database", str(DEFAULT_DB_PATH))
        st.divider()
        st.subheader("Upload & Process")
        uploaded = st.file_uploader("Add new video", type=["mp4", "avi", "mov", "mkv"])
        auto_store = st.checkbox("Auto-store detections", value=True)
        auto_track = st.checkbox("Auto-track objects", value=True)
        if st.button("Process upload", disabled=uploaded is None, use_container_width=True):
            if uploaded is None:
                st.warning("Select a file first.")
            else:
                handle_upload(uploaded, db_path, auto_store, auto_track)
                detection_files[:] = list_detection_files()
        st.divider()
        st.subheader("Actions")
        col_a, col_b = st.columns(2)
        if col_a.button("Store detections", disabled=selected_file is None, use_container_width=True):
            try:
                result = import_detection_report(selected_file, db_path)
                st.success(
                    f"Stored {result['frames_imported']} frames / "
                    f"{result['detections_imported']} detections."
                )
            except Exception as exc:
                st.error(f"Failed to store detections: {exc}")
        if col_b.button("Generate tracks", disabled=selected_file is None, use_container_width=True):
            try:
                tracks = generate_tracks(selected_file)
                store_tracks(tracks["sha256"], tracks["tracks"], db_path)
                st.success(f"Stored {len(tracks['tracks'])} tracks.")
            except Exception as exc:
                st.error(f"Failed to generate tracks: {exc}")

    if not detection_files or selected_file is None:
        return

    st.session_state["selected_report_path"] = str(selected_file.resolve())

    summary = cached_summary(str(selected_file))
    display_summary(summary)
    report_data = load_detection_report(str(selected_file))
    display_visual_preview(report_data)

    st.divider()
    st.subheader("Investigator Query")
    query_mode = st.radio(
        "Mode",
        options=["Detections", "Tracks"],
        horizontal=True,
    )
    class_filter = st.text_input("Class filter (optional)", placeholder="e.g. car")
    col1, col2, col3 = st.columns(3)
    with col1:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.3, 0.05)
    with col2:
        time_start = st.number_input("Start time (s)", min_value=0.0, value=0.0)
    with col3:
        time_end = st.number_input(
            "End time (s)",
            min_value=0.0,
            value=float(summary["time_bounds"]["end_seconds"] or 0.0),
        )
    run_query = st.button("Run query", type="primary")
    payload = None
    if run_query:
        try:
            if query_mode == "Tracks":
                data = query_tracks(
                    summary["sha256"],
                    db_path=db_path,
                    class_name=class_filter or None,
                )
                display_tracks(data)
                payload = {"mode": "tracks", "data": data}
            else:
                time_range = (
                    time_start if time_start > 0 else None,
                    time_end if time_end > 0 else None,
                )
                data = query_detections(
                    summary["sha256"],
                    db_path=db_path,
                    class_name=class_filter or None,
                    min_conf=min_conf,
                    time_range=time_range,
                )
                display_detections(data)
                payload = {"mode": "detections", "data": data}
        except Exception as exc:
            st.error(f"Query failed: {exc}")
        else:
            st.session_state["last_query"] = payload
            render_export_controls(payload)
    else:
        render_export_controls(st.session_state.get("last_query"))


def display_summary(summary: dict) -> None:
    st.subheader("Detection Summary")
    metrics = st.columns(4)
    metrics[0].metric("Frames analyzed", summary["frames_in_report"])
    metrics[1].metric("Frames w/ detections", summary["frames_with_detections"])
    metrics[2].metric("Total detections", summary["detections_total"])
    duration = summary["time_bounds"]
    span = (
        f"{duration['start_seconds']:.2f}s – {duration['end_seconds']:.2f}s"
        if duration["start_seconds"] is not None and duration["end_seconds"] is not None
        else "N/A"
    )
    metrics[3].metric("Time span", span)

    class_stats = summary.get("class_stats", {})
    if class_stats:
        records = []
        for name, stats in class_stats.items():
            records.append(
                {
                    "Class": name,
                    "Count": stats.get("count", 0),
                    "First seen (s)": stats.get("first_seen_time"),
                    "Last seen (s)": stats.get("last_seen_time"),
                }
            )
        df = pd.DataFrame(records).sort_values("Count", ascending=False)
        st.dataframe(df, hide_index=True)
    else:
        st.info("No detections found in this report.")


def display_detections(data: List[dict]) -> None:
    if not data:
        st.warning("No detections matched your filters.")
        return
    df = pd.DataFrame(data)
    st.success(f"{len(df)} detections returned.")
    st.dataframe(df, hide_index=True)


def display_tracks(data: List[dict]) -> None:
    if not data:
        st.warning("No tracks matched your filters.")
        return
    summary_rows = [
        {
            "Track": item["track_label"],
            "Class": item["class_name"],
            "Detections": item["detections_count"],
            "Start (s)": item["start_time"],
            "End (s)": item["end_time"],
        }
        for item in data
    ]
    st.success(f"{len(summary_rows)} tracks returned.")
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
    with st.expander("View first track details"):
        first = data[0]
        st.write(f"{first['track_label']} ({first['class_name']})")
        st.dataframe(pd.DataFrame(first["points"]), hide_index=True)


def display_visual_preview(report: dict) -> None:
    st.subheader("Visual Preview")
    video_path = report.get("source_path")
    frames = [
        frame for frame in report.get("detections", []) if frame.get("detections")
    ]
    if not video_path or not Path(video_path).exists():
        st.info("Source video unavailable on this workspace.")
        return
    if not frames:
        st.info("No detections to visualize.")
        return
    frame_indices = [frame.get("frame_index", 0) for frame in frames]
    selected_frame = st.slider(
        "Frame index",
        min_value=min(frame_indices),
        max_value=max(frame_indices),
        value=frame_indices[0],
    )
    frame_data = next(
        (frame for frame in frames if frame.get("frame_index") == selected_frame),
        frames[0],
    )
    detection_payload = detection_draw_payload(frame_data["detections"])
    try:
        image = render_frame_image(
            video_path,
            frame_data.get("frame_index", 0),
            detection_payload,
        )
        timestamp = frame_data.get("timestamp_seconds") or 0.0
        st.image(
            image,
            caption=f"Frame {frame_data.get('frame_index')} @ {timestamp:.2f}s",
            use_column_width=True,
        )
    except Exception as exc:
        st.error(f"Unable to render frame: {exc}")
    st.dataframe(pd.DataFrame(frame_data["detections"]), hide_index=True)


def detection_draw_payload(detections: List[dict]) -> Tuple[Tuple[float, ...], ...]:
    payload = []
    for det in detections:
        bbox = det.get("bbox_xyxy") or [0, 0, 0, 0]
        class_name = det.get("class_name") or str(det.get("class_id"))
        r, g, b = class_color(class_name)
        label = f"{class_name} {det.get('confidence', 0):.2f}"
        payload.append(
            (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
                float(r),
                float(g),
                float(b),
                label,
            )
        )
    return tuple(payload)


def class_color(name: str) -> Tuple[int, int, int]:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    r = int(digest[0:2], 16)
    g = int(digest[2:4], 16)
    b = int(digest[4:6], 16)
    return r, g, b


def render_export_controls(payload: Optional[dict]) -> None:
    if not payload or not payload.get("data"):
        return
    st.subheader("Export Results")
    data = payload["data"]
    mode = payload["mode"]
    json_bytes = json.dumps(data, indent=2).encode("utf-8")
    st.download_button(
        "Download JSON",
        data=json_bytes,
        file_name=f"{mode}_results.json",
        mime="application/json",
    )
    df = pd.DataFrame(data)
    if mode == "tracks" and "points" in df.columns:
        df = df.drop(columns=["points"])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"{mode}_results.csv",
        mime="text/csv",
    )


def handle_upload(
    uploaded_file,
    db_path: str,
    auto_store: bool,
    auto_track: bool,
) -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOADS_DIR / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    ingest_video(target)
    with st.spinner("Running detections..."):
        report = run_yolo_detection(str(target))
    report_path = write_detection_report(report)
    st.success(f"Detections saved to {report_path.name}")
    if auto_store:
        result = import_detection_report(report_path, db_path)
        st.info(
            f"Stored {result['frames_imported']} frames / "
            f"{result['detections_imported']} detections."
        )
    if auto_track:
        tracks = generate_tracks(report_path)
        store_tracks(tracks["sha256"], tracks["tracks"], db_path)
        st.info(f"Stored {len(tracks['tracks'])} tracks.")
    st.session_state["selected_report_path"] = str(report_path.resolve())
    st.rerun()


if __name__ == "__main__":
    main()
