"""Streamlit UI for the CCTV dissertation project."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cctv_dissertation.analysis import summarize_detection_report  # noqa: E402
from cctv_dissertation.ingest import ingest_video, verify_video_integrity  # noqa: E402
from cctv_dissertation.storage import (  # noqa: E402
    DEFAULT_DB_PATH,
    query_detections,
    query_tracks,
)

DETECTIONS_DIR = Path("data/detections")
UPLOADS_DIR = Path("data/uploads")
PREVIEWS_DIR = Path("data/previews")
TRACKER_DB = PROJECT_ROOT / "data" / "tracker.db"
TRACKED_OUTPUT = PROJECT_ROOT / "data" / "tracked_output"
CLIPS_DIR = PROJECT_ROOT / "data" / "clips"


# ── Database query helpers ────────────────────────────────────────


def _query_tracked_entities(
    db_path: Path,
    camera_label: Optional[str] = None,
    run_id: Optional[str] = None,
) -> list:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    clauses: list = []
    params: list = []
    if camera_label:
        clauses.append("camera_label = ?")
        params.append(camera_label)
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM tracked_entities{where} "
        "ORDER BY camera_label, entity_type, track_id",
        params,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _query_cameras(db_path: Path, run_id: Optional[str] = None) -> list:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    if run_id:
        rows = conn.execute(
            "SELECT DISTINCT camera_label FROM tracked_entities "
            "WHERE run_id = ? ORDER BY camera_label",
            (run_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT DISTINCT camera_label FROM tracked_entities "
            "ORDER BY camera_label"
        ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _search_entities(
    db_path: Path,
    cameras: Optional[List[str]] = None,
    entity_type: Optional[str] = None,
    description_query: Optional[str] = None,
    upper_color: Optional[str] = None,
    lower_color: Optional[str] = None,
    vehicle_color: Optional[str] = None,
    vehicle_type: Optional[str] = None,
    run_id: Optional[str] = None,
) -> list:
    """Search tracked entities with filters."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    clauses: list = []
    params: list = []
    if cameras:
        ph = ",".join("?" for _ in cameras)
        clauses.append(f"camera_label IN ({ph})")
        params.extend(cameras)
    if entity_type:
        clauses.append("entity_type = ?")
        params.append(entity_type)
    if description_query:
        clauses.append("description LIKE ?")
        params.append(f"%{description_query}%")
    if upper_color:
        clauses.append("upper_color = ?")
        params.append(upper_color)
    if lower_color:
        clauses.append("lower_color = ?")
        params.append(lower_color)
    if vehicle_color:
        clauses.append("color = ?")
        params.append(vehicle_color)
    if vehicle_type:
        clauses.append("vehicle_type = ?")
        params.append(vehicle_type)
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = (
        "SELECT * FROM tracked_entities"
        + where
        + " ORDER BY camera_label, entity_type, track_id"
    )
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_image(path_str: str) -> Optional[np.ndarray]:
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return None
    img = cv2.imread(str(p))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _ensure_start_time(camera: str) -> None:
    """Auto-extract video start time via OCR if not already set."""
    key = f"start_time_{camera}"
    if st.session_state.get(key):
        return  # already set
    # Find video path from DB
    entities = _query_tracked_entities(TRACKER_DB, camera)
    if not entities:
        return
    vp = entities[0].get("video_path", "")
    if not vp or not Path(vp).exists():
        return
    try:
        from cctv_dissertation.tracker import extract_video_timestamp

        ts = extract_video_timestamp(vp)
        if ts:
            parts = ts.split(" ")
            st.session_state[key] = parts[1]
            st.session_state[f"start_date_{camera}"] = parts[0]
    except Exception:
        pass


def _get_unified_identities(cam_a: str, cam_b: str) -> dict:
    """Return cached unified identities, rebuilding when cameras or DB changes."""
    cache_key = "unified_identities_cache"
    cam_key = "unified_identities_cameras"
    mtime_key = "unified_identities_mtime"

    # Invalidate cache when DB file changes (re-processing, code changes)
    db_mtime = TRACKER_DB.stat().st_mtime if TRACKER_DB.exists() else 0
    cached_cams = st.session_state.get(cam_key)
    cached_mtime = st.session_state.get(mtime_key)

    if (
        cached_cams == (cam_a, cam_b)
        and cached_mtime == db_mtime
        and cache_key in st.session_state
    ):
        return st.session_state[cache_key]

    from cctv_dissertation.tracker import build_unified_identities

    result = build_unified_identities(
        str(TRACKER_DB),
        cam_a,
        cam_b,
        person_threshold=0.45,
        vehicle_threshold=0.80,
    )
    st.session_state[cache_key] = result
    st.session_state[cam_key] = (cam_a, cam_b)
    st.session_state[mtime_key] = db_mtime
    return result


# ── Single-camera pages ──────────────────────────────────────────


def page_single_tracking(camera: str) -> None:
    """Full tracking view — annotated video with bounding boxes."""
    st.header(f"Tracking View — {camera}")
    entities = _query_tracked_entities(TRACKER_DB, camera)
    if not entities:
        st.info("No tracking data for this camera.")
        return

    # Check for an annotated video
    ann_video = TRACKED_OUTPUT / camera / "annotated.mp4"
    if ann_video.exists():
        st.video(str(ann_video))
    else:
        # Offer to generate one
        if st.button("Generate annotated video", key=f"gen_ann_{camera}"):
            with st.spinner("Rendering annotated video..."):
                from cctv_dissertation.tracker import generate_annotated_video

                # Get video path from first entity
                video_path = entities[0].get("video_path", "")
                generate_annotated_video(
                    video_path,
                    str(TRACKER_DB),
                    str(ann_video),
                    camera_label=camera,
                )
            st.rerun()

    # Show frame-by-frame viewer from the source video
    if entities:
        video_path = entities[0].get("video_path", "")
        if Path(video_path).exists():
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()

            ts = st.slider(
                "Scrub through video",
                0.0,
                total / fps,
                0.0,
                step=1.0 / fps,
                key=f"scrub_{camera}",
            )
            frame_idx = int(ts * fps)
            frame = _read_frame(video_path, frame_idx)
            if frame is not None:
                # Overlay boxes from DB
                frame = _overlay_tracked_boxes(frame, frame_idx, camera)
                st.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption=_format_absolute_time(
                        ts, st.session_state.get(f"start_time_{camera}")
                    ),
                    use_container_width=True,
                )


def page_single_summary(camera: str) -> None:
    """Condensed summary of all tracked entities."""
    st.header(f"Summary — {camera}")
    _ensure_start_time(camera)
    entities = _query_tracked_entities(TRACKER_DB, camera)
    if not entities:
        st.info("No tracking data. Upload and process a video first.")
        return

    # Show auto-detected start time; allow manual override
    start_key = f"start_time_{camera}"
    date_key = f"start_date_{camera}"
    auto_time = st.session_state.get(start_key, "")
    auto_date = st.session_state.get(date_key, "")
    if auto_time:
        st.info(f"Timestamp detected: **{auto_date} {auto_time}**")
    with st.expander("Override start time", expanded=not auto_time):
        start_val = st.text_input(
            f"{camera} start (HH:MM:SS)",
            value=auto_time,
            key=f"input_start_single_{camera}",
        )
        if start_val:
            st.session_state[start_key] = start_val
    cam_start = st.session_state.get(start_key)

    persons = [e for e in entities if e["entity_type"] == "person"]
    vehicles = [e for e in entities if e["entity_type"] == "vehicle"]

    # ── People ──
    if persons:
        st.subheader(f"People ({len(persons)})")
        cols = st.columns(min(len(persons), 4))
        for i, p in enumerate(persons):
            with cols[i % len(cols)]:
                img = _load_image(p["crop_path"])
                if img is not None:
                    st.image(img, use_container_width=True)
                t0 = _format_absolute_time(p["first_ts"], cam_start)
                t1 = _format_absolute_time(p["last_ts"], cam_start)
                st.markdown(
                    f"**Person {p['track_id']}**  \n"
                    f"{p['description']}  \n"
                    f"Visible: {t0} - {t1}  \n"
                    f"Confidence: {p['confidence']:.2f}"
                )
        st.divider()

    # ── Vehicles ──
    if vehicles:
        st.subheader(f"Vehicles ({len(vehicles)})")

        # Summary table
        table = []
        for v in vehicles:
            vt0 = _format_absolute_time(v["first_ts"], cam_start)
            vt1 = _format_absolute_time(v["last_ts"], cam_start)
            table.append(
                {
                    "ID": v["track_id"],
                    "Type": v["vehicle_type"] or "car",
                    "Color": v["color"] or "unknown",
                    "Description": v["description"] or "",
                    "Plate": v["plate_text"] or "-",
                    "Visible": f"{vt0} - {vt1}",
                }
            )
        st.dataframe(pd.DataFrame(table), hide_index=True)

        # Gallery
        cols = st.columns(min(len(vehicles), 4))
        for i, v in enumerate(vehicles):
            with cols[i % len(cols)]:
                img = _load_image(v["crop_path"])
                if img is not None:
                    st.image(img, use_container_width=True)
                plate_str = (
                    f"Plate: {v['plate_text']}"
                    if v["plate_text"]
                    else "No plate detected"
                )
                st.markdown(
                    f"**Vehicle {v['track_id']}** — "
                    f"{v['description']}  \n"
                    f"{plate_str}"
                )


# ── Cross-camera pages ───────────────────────────────────────────


def page_cross_dual_view(cam_a: str, cam_b: str) -> None:
    """Side-by-side synchronised view of two camera feeds."""
    st.header("Dual Camera View")

    entities_a = _query_tracked_entities(TRACKER_DB, cam_a)
    entities_b = _query_tracked_entities(TRACKER_DB, cam_b)

    if not entities_a or not entities_b:
        st.info("Need tracking data from both cameras.")
        return

    # Show annotated videos side-by-side
    ann_a = TRACKED_OUTPUT / cam_a / "annotated.mp4"
    ann_b = TRACKED_OUTPUT / cam_b / "annotated.mp4"

    if ann_a.exists() or ann_b.exists():
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader(cam_a)
            if ann_a.exists():
                st.video(str(ann_a))
            else:
                st.info("No annotated video.")
        with col_right:
            st.subheader(cam_b)
            if ann_b.exists():
                st.video(str(ann_b))
            else:
                st.info("No annotated video.")

    # Frame scrubber
    st.divider()
    st.subheader("Synchronised Frame Scrubber")

    video_a = entities_a[0].get("video_path", "")
    video_b = entities_b[0].get("video_path", "")

    if not Path(video_a).exists() or not Path(video_b).exists():
        st.warning("Source video files not found.")
        return

    cap_a = cv2.VideoCapture(video_a)
    cap_b = cv2.VideoCapture(video_b)
    fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 25.0
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 25.0
    total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    total_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_a.release()
    cap_b.release()

    max_dur = max(total_a / fps_a, total_b / fps_b)

    ts = st.slider(
        "Synchronised timeline (seconds)",
        0.0,
        max_dur,
        0.0,
        step=0.1,
        key="cross_timeline",
    )

    col_left, col_right = st.columns(2)

    with col_left:
        fidx_a = min(int(ts * fps_a), total_a - 1)
        frame_a = _read_frame(video_a, fidx_a)
        if frame_a is not None:
            frame_a = _overlay_tracked_boxes(frame_a, fidx_a, cam_a)
            st.image(
                cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB),
                caption=f"{cam_a} — {_format_absolute_time(ts, st.session_state.get(f'start_time_{cam_a}'))}",  # noqa: E501
                use_container_width=True,
            )

    with col_right:
        fidx_b = min(int(ts * fps_b), total_b - 1)
        frame_b = _read_frame(video_b, fidx_b)
        if frame_b is not None:
            frame_b = _overlay_tracked_boxes(frame_b, fidx_b, cam_b)
            st.image(
                cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB),
                caption=f"{cam_b} — {_format_absolute_time(ts, st.session_state.get(f'start_time_{cam_b}'))}",  # noqa: E501
                use_container_width=True,
            )

    # Show matched entities summary below
    st.divider()
    _show_cross_matches_summary(cam_a, cam_b)


def _format_absolute_time(video_ts: float, start_time: Optional[str]) -> str:
    """Convert video timestamp to HH:MM:SS.

    Returns absolute wall-clock time when start_time (from OCR) is known,
    otherwise returns a relative HH:MM:SS offset from video start.
    """
    from datetime import datetime, timedelta

    if start_time:
        try:
            base = datetime.strptime(start_time, "%H:%M:%S")
            return (base + timedelta(seconds=video_ts)).strftime("%H:%M:%S")
        except Exception:
            pass
    # Fallback: relative time — always HH:MM:SS, never raw seconds
    h = int(video_ts // 3600)
    m = int((video_ts % 3600) // 60)
    s = int(video_ts % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _to_absolute_datetime(
    video_ts: float,
    start_time: Optional[str],
    start_date: Optional[str] = None,
):
    """Convert video timestamp to a full datetime object."""
    from datetime import datetime, timedelta

    if not start_time:
        return None
    try:
        date_str = start_date or "2000-01-01"
        base = datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %H:%M:%S")
        return base + timedelta(seconds=video_ts)
    except Exception:
        return None


def _calc_real_gap_standalone(s1, s2, get_start_time):
    """Calculate real-world time gap between two sightings.

    get_start_time is callable(camera) -> Optional[str].
    Returns seconds (positive = gap, negative = overlap).
    """
    from datetime import datetime, timedelta

    st1 = get_start_time(s1["camera"])
    st2 = get_start_time(s2["camera"])
    if not st1 or not st2:
        return None
    try:
        base1 = datetime.strptime(st1, "%H:%M:%S")
        base2 = datetime.strptime(st2, "%H:%M:%S")
        end1 = base1 + timedelta(seconds=s1["last_ts"])
        start2 = base2 + timedelta(seconds=s2["first_ts"])
        return (start2 - end1).total_seconds()
    except Exception:
        return None


def page_cross_identity(cam_a: str, cam_b: str) -> None:
    """Identity deep-dive — unified IDs across cameras with timelines."""
    _ensure_start_time(cam_a)
    _ensure_start_time(cam_b)
    st.header("Identity Tracking")
    st.caption(
        "Unified person/vehicle IDs across all cameras. "
        "Same person = same ID regardless of which camera."
    )

    # ── Video Start Times ──
    # Show auto-detected times; allow manual override
    auto_a = st.session_state.get(f"start_time_{cam_a}", "")
    auto_b = st.session_state.get(f"start_time_{cam_b}", "")
    date_a = st.session_state.get(f"start_date_{cam_a}", "")
    date_b = st.session_state.get(f"start_date_{cam_b}", "")

    if auto_a or auto_b:
        info_parts = []
        if auto_a:
            info_parts.append(f"**{cam_a}**: {date_a} {auto_a}")
        if auto_b:
            info_parts.append(f"**{cam_b}**: {date_b} {auto_b}")
        st.info("Timestamps detected from video: " + " | ".join(info_parts))

    with st.expander("Override start times", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            start_a = st.text_input(
                f"{cam_a} start (HH:MM:SS)",
                value=auto_a,
                key=f"input_start_{cam_a}",
            )
            if start_a:
                st.session_state[f"start_time_{cam_a}"] = start_a
        with col2:
            start_b = st.text_input(
                f"{cam_b} start (HH:MM:SS)",
                value=auto_b,
                key=f"input_start_{cam_b}",
            )
            if start_b:
                st.session_state[f"start_time_{cam_b}"] = start_b

    start_time_a = st.session_state.get(f"start_time_{cam_a}")
    start_time_b = st.session_state.get(f"start_time_{cam_b}")

    from cctv_dissertation.tracker import generate_cross_camera_clip

    identities = _get_unified_identities(cam_a, cam_b)

    persons = identities["persons"]
    vehicles = identities["vehicles"]

    if not persons and not vehicles:
        st.info("No tracked entities found.")
        return

    # ── PDF Download ──
    date_a = st.session_state.get(f"start_date_{cam_a}")
    date_b = st.session_state.get(f"start_date_{cam_b}")

    # Get video paths from database
    conn = sqlite3.connect(str(TRACKER_DB))
    video_a = conn.execute(
        "SELECT DISTINCT video_path FROM tracked_entities WHERE camera_label = ? LIMIT 1",  # noqa: E501
        (cam_a,),
    ).fetchone()
    video_b = conn.execute(
        "SELECT DISTINCT video_path FROM tracked_entities WHERE camera_label = ? LIMIT 1",  # noqa: E501
        (cam_b,),
    ).fetchone()
    conn.close()

    video_a = video_a[0] if video_a else None
    video_b = video_b[0] if video_b else None

    pdf_bytes = _generate_forensic_pdf(
        identities,
        cam_a,
        cam_b,
        start_time_a,
        start_time_b,
        date_a,
        date_b,
        video_a,
        video_b,
    )
    st.download_button(
        label="Download Forensic Report (PDF)",
        data=pdf_bytes,
        file_name="forensic_report.pdf",
        mime="application/pdf",
        type="primary",
    )

    def _get_start_time(camera: str) -> Optional[str]:
        if camera == cam_a:
            return start_time_a
        elif camera == cam_b:
            return start_time_b
        return None

    def _calc_real_gap(s1: dict, s2: dict) -> Optional[float]:
        """Calculate real-world time gap between two sightings."""
        st1 = _get_start_time(s1["camera"])
        st2 = _get_start_time(s2["camera"])
        if not st1 or not st2:
            return None
        try:
            from datetime import datetime, timedelta

            base1 = datetime.strptime(st1, "%H:%M:%S")
            base2 = datetime.strptime(st2, "%H:%M:%S")
            actual1_end = base1 + timedelta(seconds=s1["last_ts"])
            actual2_start = base2 + timedelta(seconds=s2["first_ts"])
            gap = (actual2_start - actual1_end).total_seconds()
            return gap
        except Exception:
            return None

    # ── PERSONS ──
    if persons:
        st.subheader(f"Persons ({len(persons)})")

        for identity in persons:
            uid = identity["unified_id"]
            sightings = identity["sightings"]
            raw = identity.get("raw_sightings", sightings)
            matched = identity["matched"]
            n_appearances = len(raw)
            cameras_seen = sorted(set(s["camera"] for s in raw))

            # Create an expander for each person
            if len(cameras_seen) > 1:
                title = (
                    f"**Person {uid}** — {n_appearances} appearances "
                    f"across {len(cameras_seen)} cameras"
                )
                if identity["similarity"]:
                    title += f" ({identity['similarity'] * 100:.0f}% match)"
            else:
                title = (
                    f"**Person {uid}** — {n_appearances} appearance"
                    f"{'s' if n_appearances > 1 else ''} on {cameras_seen[0]}"
                )

            with st.expander(title, expanded=matched):
                # Show each raw sighting as a separate crop
                max_cols = min(n_appearances, 6)
                for row_start in range(0, n_appearances, max_cols):
                    row_items = raw[row_start : row_start + max_cols]
                    cols = st.columns(len(row_items))
                    for i, s in enumerate(row_items):
                        with cols[i]:
                            img = _load_image(s["crop_path"])
                            if img is not None:
                                st.image(img, use_container_width=True)
                            cam_start = _get_start_time(s["camera"])
                            time_str = _format_absolute_time(s["first_ts"], cam_start)
                            time_end_str = _format_absolute_time(
                                s["last_ts"], cam_start
                            )
                            st.markdown(
                                f"**{s['camera']}**  \n"
                                f"{s['description']}  \n"
                                f"**{time_str}** - {time_end_str}"
                            )

                # Movement timeline with gaps
                if n_appearances > 1:
                    st.markdown("**Movement Timeline:**")
                    timeline_parts = []
                    for s in raw:
                        cam_start = _get_start_time(s["camera"])
                        time_str = _format_absolute_time(s["first_ts"], cam_start)
                        timeline_parts.append(f"{s['camera']} @ {time_str}")
                    st.markdown(" \u2192 ".join(timeline_parts))

                    # Show gaps between consecutive appearances
                    for idx in range(len(raw) - 1):
                        real_gap = _calc_real_gap(raw[idx], raw[idx + 1])
                        lbl = (
                            f"{raw[idx]['camera']} \u2192 " f"{raw[idx + 1]['camera']}"
                        )
                        if real_gap is not None:
                            if real_gap >= 0:
                                st.caption(f"{lbl}: off camera {real_gap:.0f}s")
                            else:
                                st.caption(f"{lbl}: overlap {abs(real_gap):.0f}s")
                        else:
                            gap = raw[idx + 1]["first_ts"] - raw[idx]["last_ts"]
                            st.caption(f"{lbl}: gap {gap:.0f}s")

                    # Cross-camera clip (first + last from different cams)
                    if len(cameras_seen) > 1:
                        clip_name = f"person_{uid}_cross.mp4"
                        clip_path = CLIPS_DIR / clip_name
                        if clip_path.exists():
                            st.video(str(clip_path))
                        else:
                            first_s = sightings[0]
                            last_s = sightings[-1]
                            if first_s["camera"] != last_s["camera"] and st.button(
                                f"Generate tracking clip for Person {uid}",
                                key=f"gen_person_{uid}",
                            ):
                                CLIPS_DIR.mkdir(parents=True, exist_ok=True)
                                with st.spinner("Generating clip..."):
                                    generate_cross_camera_clip(
                                        str(TRACKER_DB),
                                        first_s["entity_id"],
                                        last_s["entity_id"],
                                        str(clip_path),
                                        unified_id=uid,
                                    )
                                st.rerun()

    # ── VEHICLES ──
    if vehicles:
        st.divider()
        st.subheader(f"Vehicles ({len(vehicles)})")

        for identity in vehicles:
            uid = identity["unified_id"]
            sightings = identity["sightings"]
            matched = identity["matched"]

            if matched:
                title = f"**Vehicle {uid}** — Seen on {len(sightings)} cameras"
                if identity["similarity"]:
                    title += f" ({identity['similarity'] * 100:.0f}% match)"
            else:
                title = f"**Vehicle {uid}** — {sightings[0]['camera']} only"

            with st.expander(title, expanded=matched):
                cols = st.columns(len(sightings))
                for i, s in enumerate(sightings):
                    with cols[i]:
                        img = _load_image(s["crop_path"])
                        if img is not None:
                            st.image(img, use_container_width=True)
                        cam_start = _get_start_time(s["camera"])
                        time_str = _format_absolute_time(s["first_ts"], cam_start)
                        time_end_str = _format_absolute_time(s["last_ts"], cam_start)
                        st.markdown(
                            f"**{s['camera']}**  \n"
                            f"{s['description']}  \n"
                            f"**{time_str}** - {time_end_str}"
                        )

                if len(sightings) > 1:
                    st.markdown("**Movement Timeline:**")
                    timeline_parts = []
                    for s in sightings:
                        cam_start = _get_start_time(s["camera"])
                        time_str = _format_absolute_time(s["first_ts"], cam_start)
                        timeline_parts.append(f"{s['camera']} @ {time_str}")
                    st.markdown(" → ".join(timeline_parts))

                    # Time gap
                    real_gap = _calc_real_gap(sightings[0], sightings[1])
                    if real_gap is not None:
                        if real_gap >= 0:
                            mins = real_gap / 60
                            st.success(f"Off camera: {real_gap:.1f}s ({mins:.1f}min)")
                        else:
                            st.warning(f"Overlap: {abs(real_gap):.1f}s")
                    else:
                        gap = sightings[1]["first_ts"] - sightings[0]["last_ts"]
                        st.caption(f"Gap: {gap:.1f}s (set start times)")

                    clip_name = f"vehicle_{uid}_cross.mp4"
                    clip_path = CLIPS_DIR / clip_name

                    if clip_path.exists():
                        st.video(str(clip_path))
                    else:
                        if st.button(
                            f"Generate tracking clip for Vehicle {uid}",
                            key=f"gen_vehicle_{uid}",
                        ):
                            CLIPS_DIR.mkdir(parents=True, exist_ok=True)
                            with st.spinner("Generating clip..."):
                                generate_cross_camera_clip(
                                    str(TRACKER_DB),
                                    sightings[0]["entity_id"],
                                    sightings[1]["entity_id"],
                                    str(clip_path),
                                    unified_id=uid,
                                )
                            st.rerun()

    # ── MOVEMENT EVENTS (Person→Vehicle linkages) ──
    _journey_section(identities, cam_a, cam_b)


def _journey_section(identities: dict, cam_a: str, cam_b: str) -> None:
    """Render person-vehicle departure/arrival linkages as a
    horizontal cross-camera timeline: car - garage - garden."""
    journeys = identities.get("journeys", [])
    if not journeys:
        return

    st.divider()
    st.subheader("Movement Events")
    st.caption(
        "Cross-camera journey reconstruction — "
        "vehicle and person movements linked by temporal proximity and direction."
    )

    person_map = {p["unified_id"]: p for p in identities.get("persons", [])}
    vehicle_map = {v["unified_id"]: v for v in identities.get("vehicles", [])}

    def _st(camera: str):
        return st.session_state.get(f"start_time_{camera}")

    for j in journeys:
        person = person_map.get(j["person_id"])
        vehicle = vehicle_map.get(j["vehicle_id"])
        if not person:
            continue

        event_label = "departed in" if j["event"] == "departure" else "arrived in"
        title = (
            f"Person {j['person_id']} {event_label} "
            f"Vehicle {j['vehicle_id']} ({j['vehicle_desc']})"
        )

        # Build chronological timeline steps across cameras
        steps: list = []

        # Vehicle event
        v_crop = j.get("vehicle_crop")
        if not v_crop and vehicle:
            v_crop = vehicle["sightings"][0].get("crop_path")
        steps.append(
            {
                "type": "vehicle",
                "label": f"Vehicle {j['vehicle_id']}",
                "camera": j["camera"],
                "ts": j["timestamp"],
                "desc": j["vehicle_desc"],
                "crop": v_crop,
            }
        )

        # Person sightings on each camera
        for s in person["sightings"]:
            steps.append(
                {
                    "type": "person",
                    "label": f"Person {j['person_id']}",
                    "camera": s["camera"],
                    "ts": s["first_ts"],
                    "ts_end": s["last_ts"],
                    "desc": s["description"],
                    "crop": s.get("crop_path"),
                }
            )

        # Sort chronologically
        steps.sort(key=lambda s: s["ts"])

        with st.expander(f"**{title}**", expanded=True):
            # Horizontal timeline with one column per step + arrows
            n = len(steps)
            # Interleave step columns with arrow columns
            col_spec = []
            for i in range(n):
                col_spec.append(2)
                if i < n - 1:
                    col_spec.append(1)  # arrow column

            cols = st.columns(col_spec)
            col_idx = 0
            for i, step in enumerate(steps):
                with cols[col_idx]:
                    # Crop image
                    if step.get("crop"):
                        img = _load_image(step["crop"])
                        if img is not None:
                            st.image(img, use_container_width=True)

                    # Camera label
                    cam_label = step["camera"]
                    time_str = _format_absolute_time(step["ts"], _st(cam_label))

                    if step["type"] == "vehicle":
                        icon = "**Vehicle**"
                    else:
                        icon = "**Person**"

                    st.markdown(f"{icon}  \n{cam_label}")
                    st.caption(f"{step['desc']}  \n{time_str}")

                col_idx += 1

                # Arrow between steps
                if i < n - 1:
                    with cols[col_idx]:
                        st.markdown("")
                        st.markdown("")
                        # Calculate gap between this step and next
                        next_step = steps[i + 1]
                        gap = abs(next_step["ts"] - step["ts"])
                        if gap < 60:
                            gap_text = f"{gap:.0f}s"
                        else:
                            gap_text = f"{gap / 60:.1f}m"
                        st.markdown(
                            f"<div style='text-align:center; "
                            f"padding-top:30px; font-size:24px;'>"
                            f"&rarr;</div>"
                            f"<div style='text-align:center; "
                            f"font-size:12px; color:grey;'>"
                            f"{gap_text}</div>",
                            unsafe_allow_html=True,
                        )
                    col_idx += 1

            # Confidence
            conf_pct = j["confidence"] * 100
            if conf_pct >= 60:
                st.success(f"Confidence: {conf_pct:.0f}%")
            elif conf_pct >= 30:
                st.warning(f"Confidence: {conf_pct:.0f}%")
            else:
                st.info(f"Confidence: {conf_pct:.0f}%")

    # ── Vehicle Return Matching ──
    round_trips = identities.get("round_trips", [])
    if round_trips:
        st.divider()
        st.subheader("Vehicle Return Matching")
        st.caption("Vehicles that departed and later returned, matched by Re-ID.")

        for rt in round_trips:
            dep = rt["departure"]
            arr = rt["arrival"]
            away = rt["away_seconds"]

            if away < 60:
                away_str = f"{away:.0f}s"
            elif away < 3600:
                away_str = f"{away / 60:.0f}min"
            else:
                h = int(away) // 3600
                m = (int(away) % 3600) // 60
                away_str = f"{h}h {m}m"

            dep_time = _format_absolute_time(dep["timestamp"], _st(dep["camera"]))
            arr_time = _format_absolute_time(arr["timestamp"], _st(arr["camera"]))

            title = (
                f"Vehicle {rt['vehicle_id']} ({rt['vehicle_desc']}) "
                f"- Left {dep_time}, Returned {arr_time} "
                f"(away {away_str})"
            )

            with st.expander(f"**{title}**", expanded=True):
                col_out, col_arrow, col_in = st.columns([3, 1, 3])

                with col_out:
                    st.markdown("**Departure**")
                    if rt.get("person_crop_out"):
                        img = _load_image(rt["person_crop_out"])
                        if img is not None:
                            st.image(img, width=120)
                    st.write(f"Person {dep['person_id']} " f"({dep['person_desc']})")
                    st.caption(
                        f"{dep['camera']} at {dep_time}  \n"
                        f"Confidence: {dep['confidence'] * 100:.0f}%"
                    )

                with col_arrow:
                    st.markdown("")
                    st.markdown(
                        f"<div style='text-align:center; "
                        f"padding-top:40px; font-size:28px;'>"
                        f"&rarr;</div>"
                        f"<div style='text-align:center; "
                        f"font-size:14px; color:grey; "
                        f"font-weight:bold;'>"
                        f"{away_str}</div>",
                        unsafe_allow_html=True,
                    )

                with col_in:
                    st.markdown("**Return**")
                    if rt.get("person_crop_in"):
                        img = _load_image(rt["person_crop_in"])
                        if img is not None:
                            st.image(img, width=120)
                    st.write(f"Person {arr['person_id']} " f"({arr['person_desc']})")
                    st.caption(
                        f"{arr['camera']} at {arr_time}  \n"
                        f"Confidence: {arr['confidence'] * 100:.0f}%"
                    )

                if rt["same_person"]:
                    st.success(
                        f"Same person (P{dep['person_id']}) " f"departed and returned"
                    )
                else:
                    st.info(
                        f"Different persons: P{dep['person_id']} "
                        f"departed, P{arr['person_id']} returned"
                    )


def page_cross_timeline(cam_a: str, cam_b: str) -> None:
    """Interactive Gantt-style timeline of person/vehicle movements."""
    _ensure_start_time(cam_a)
    _ensure_start_time(cam_b)
    st.header("Movement Timeline")

    identities = _get_unified_identities(cam_a, cam_b)

    def _st(cam):
        return st.session_state.get(f"start_time_{cam}")

    def _dt(cam):
        return st.session_state.get(f"start_date_{cam}")

    has_times = bool(_st(cam_a) or _st(cam_b))

    def _fmt_duration(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f}s"
        m = int(secs) // 60
        s = int(secs) % 60
        if m < 60:
            return f"{m}m {s}s"
        h = m // 60
        m = m % 60
        return f"{h}h {m}m"

    def _fmt_time(ts: float, cam: str) -> str:
        return _format_absolute_time(ts, _st(cam))

    for label, items in [
        ("Persons", identities["persons"]),
        ("Vehicles", identities["vehicles"]),
    ]:
        if not items:
            continue

        # Build rows for Gantt (on-camera bars) and gap rows (off-camera)
        on_rows = []
        gap_rows = []
        summary_rows = []

        for identity in items:
            uid = identity["unified_id"]
            id_label = f"{label[:-1]} {uid}"
            if identity["entity_type"] == "person":
                timeline_sightings = identity.get(
                    "raw_sightings", identity["sightings"]
                )
            else:
                timeline_sightings = identity["sightings"]

            total_on = 0.0
            total_off = 0.0
            n_gaps = 0

            for idx, s in enumerate(timeline_sightings):
                duration = s["last_ts"] - s["first_ts"]
                total_on += duration

                if has_times:
                    t0 = _to_absolute_datetime(
                        s["first_ts"], _st(s["camera"]), _dt(s["camera"])
                    )
                    t1 = _to_absolute_datetime(
                        s["last_ts"], _st(s["camera"]), _dt(s["camera"])
                    )
                    if not t0 or not t1:
                        continue
                else:
                    t0 = s["first_ts"]
                    t1 = s["last_ts"]

                on_rows.append(
                    {
                        "Identity": id_label,
                        "Camera": s["camera"],
                        "Start": t0,
                        "End": t1,
                        "Type": "On Camera",
                        "Description": s["description"],
                        "Duration": _fmt_duration(duration),
                    }
                )

                # Off-camera gap between consecutive sightings
                if idx > 0:
                    prev = timeline_sightings[idx - 1]
                    if has_times:
                        g0 = _to_absolute_datetime(
                            prev["last_ts"],
                            _st(prev["camera"]),
                            _dt(prev["camera"]),
                        )
                        g1 = t0
                    else:
                        g0 = prev["last_ts"]
                        g1 = s["first_ts"]

                    gap_secs = s["first_ts"] - prev["last_ts"]
                    if gap_secs > 0:
                        total_off += gap_secs
                        n_gaps += 1
                        transit = prev["camera"] != s["camera"]
                        gap_rows.append(
                            {
                                "Identity": id_label,
                                "Camera": "Off Camera",
                                "Start": g0,
                                "End": g1,
                                "Type": ("Transit" if transit else "Off Camera"),
                                "Description": (
                                    f"{prev['camera']} -> {s['camera']}"
                                    if transit
                                    else f"Not visible on {s['camera']}"
                                ),
                                "Duration": _fmt_duration(gap_secs),
                            }
                        )

            # Build summary row
            if timeline_sightings:
                first_s = timeline_sightings[0]
                last_s = timeline_sightings[-1]
                first_time = _fmt_time(first_s["first_ts"], first_s["camera"])
                last_time = _fmt_time(last_s["last_ts"], last_s["camera"])
                cameras_seen = sorted(set(s["camera"] for s in timeline_sightings))
                summary_rows.append(
                    {
                        "Identity": id_label,
                        "Cameras": ", ".join(cameras_seen),
                        "First Seen": first_time,
                        "Last Seen": last_time,
                        "Sightings": len(timeline_sightings),
                        "On Camera": _fmt_duration(total_on),
                        "Off Camera": _fmt_duration(total_off) if total_off else "-",
                        "Gaps": n_gaps,
                    }
                )

        if not on_rows:
            continue

        st.subheader(label)

        # ── Gantt chart with on-camera bars + off-camera gaps ──
        all_rows = on_rows + gap_rows
        df = pd.DataFrame(all_rows)

        if has_times:
            x_enc = alt.X("Start:T", title="Time")
            x2_enc = alt.X2("End:T")
        else:
            x_enc = alt.X("Start:Q", title="Video Seconds")
            x2_enc = alt.X2("End:Q")

        # On-camera bars coloured by camera
        bar_on = (
            alt.Chart(df[df["Type"] != "Off Camera"])
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=x_enc,
                x2=x2_enc,
                y=alt.Y(
                    "Identity:N",
                    sort=alt.EncodingSortField(field="Start", order="ascending"),
                    title=None,
                ),
                color=alt.Color(
                    "Camera:N",
                    scale=alt.Scale(scheme="category10"),
                    legend=alt.Legend(title="Camera"),
                ),
                tooltip=["Identity", "Camera", "Description", "Duration"],
            )
        )

        # Off-camera gaps as thin grey striped bars
        df_gaps = df[df["Type"].isin(["Off Camera", "Transit"])]
        if not df_gaps.empty:
            bar_off = (
                alt.Chart(df_gaps)
                .mark_bar(
                    cornerRadiusEnd=2,
                    color="#e0e0e0",
                    height=8,
                    stroke="#999",
                    strokeDash=[4, 2],
                )
                .encode(
                    x=x_enc,
                    x2=x2_enc,
                    y=alt.Y(
                        "Identity:N",
                        sort=alt.EncodingSortField(field="Start", order="ascending"),
                        title=None,
                    ),
                    tooltip=[
                        "Identity",
                        "Type",
                        "Description",
                        "Duration",
                    ],
                )
            )
            chart = (bar_on + bar_off).properties(
                height=max(len(summary_rows) * 50, 150)
            )
        else:
            chart = bar_on.properties(height=max(len(summary_rows) * 50, 150))

        st.altair_chart(chart.interactive(), use_container_width=True)

        # ── On/Off Camera Summary Table ──
        if summary_rows:
            st.markdown("**On/Off Camera Summary**")
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
            )

    # ── Journey event markers ──
    journeys = identities.get("journeys", [])
    if journeys:
        st.subheader("Movement Events")
        event_rows = []
        for j in journeys:
            ev_ts = j["timestamp"]
            cam = j["camera"]
            ev_time = _fmt_time(ev_ts, cam)

            arrow = "departed in" if j["event"] == "departure" else "arrived in"
            event_rows.append(
                {
                    "Time": ev_time,
                    "Event": (
                        f"P{j['person_id']} {arrow} "
                        f"V{j['vehicle_id']} ({j['vehicle_desc']})"
                    ),
                    "Confidence": f"{j['confidence'] * 100:.0f}%",
                    "Gap": _fmt_duration(j["gap_seconds"]),
                }
            )

        if event_rows:
            st.dataframe(
                pd.DataFrame(event_rows),
                use_container_width=True,
                hide_index=True,
            )

    # ── Vehicle return matching ──
    round_trips = identities.get("round_trips", [])
    if round_trips:
        st.subheader("Vehicle Returns")
        rt_rows = []
        for rt in round_trips:
            dep = rt["departure"]
            arr = rt["arrival"]
            dep_time = _fmt_time(dep["timestamp"], dep["camera"])
            arr_time = _fmt_time(arr["timestamp"], arr["camera"])
            rt_rows.append(
                {
                    "Vehicle": f"V{rt['vehicle_id']} ({rt['vehicle_desc']})",
                    "Departed": dep_time,
                    "Departed By": f"P{dep['person_id']}",
                    "Returned": arr_time,
                    "Returned By": f"P{arr['person_id']}",
                    "Away": _fmt_duration(rt["away_seconds"]),
                    "Same Person": "Yes" if rt["same_person"] else "No",
                }
            )
        st.dataframe(
            pd.DataFrame(rt_rows),
            use_container_width=True,
            hide_index=True,
        )

    if not has_times:
        st.caption(
            "Showing video-relative seconds. "
            "Set start times in Identity Deep Dive for "
            "real-world timestamps."
        )


def page_search(cameras: List[str]) -> None:
    """Appearance-based entity search across cameras."""
    st.header("Appearance Search")

    ALL_COLORS = [
        "",
        "beige",
        "black",
        "blue",
        "brown",
        "dark grey",
        "green",
        "grey",
        "olive",
        "orange",
        "pink",
        "purple",
        "red",
        "white",
        "yellow",
    ]

    col_type, col_text = st.columns([1, 3])
    with col_type:
        etype = st.selectbox(
            "Entity type",
            ["All", "person", "vehicle"],
            key="search_etype",
        )
    with col_text:
        desc_q = st.text_input(
            "Description search",
            placeholder="e.g. red top, white car",
            key="search_desc",
        )

    search_etype = None if etype == "All" else etype
    upper_c = lower_c = veh_c = veh_t = None

    if etype != "vehicle":
        c1, c2 = st.columns(2)
        with c1:
            upper_c = (
                st.selectbox(
                    "Upper body color",
                    ALL_COLORS,
                    format_func=lambda x: x or "Any",
                    key="search_upper",
                )
                or None
            )
        with c2:
            lower_c = (
                st.selectbox(
                    "Lower body color",
                    ALL_COLORS,
                    format_func=lambda x: x or "Any",
                    key="search_lower",
                )
                or None
            )

    if etype != "person":
        c1, c2 = st.columns(2)
        with c1:
            veh_c = (
                st.selectbox(
                    "Vehicle color",
                    ALL_COLORS,
                    format_func=lambda x: x or "Any",
                    key="search_vcolor",
                )
                or None
            )
        with c2:
            veh_t = (
                st.selectbox(
                    "Vehicle type",
                    ["", "car", "motorcycle", "bus", "truck"],
                    format_func=lambda x: x or "Any",
                    key="search_vtype",
                )
                or None
            )

    results = _search_entities(
        TRACKER_DB,
        cameras=cameras,
        entity_type=search_etype,
        description_query=desc_q or None,
        upper_color=upper_c,
        lower_color=lower_c,
        vehicle_color=veh_c,
        vehicle_type=veh_t,
        run_id=st.session_state.get("current_run_id"),
    )

    st.caption(f"{len(results)} result(s)")
    if not results:
        st.info("No entities match the current filters.")
        return

    n_cols = min(len(results), 4)
    cols = st.columns(n_cols)
    for i, entity in enumerate(results):
        with cols[i % n_cols]:
            img = _load_image(entity["crop_path"])
            if img is not None:
                st.image(img, use_container_width=True)
            cam_start = st.session_state.get(f"start_time_{entity['camera_label']}")
            t0 = _format_absolute_time(entity["first_ts"], cam_start)
            t1 = _format_absolute_time(entity["last_ts"], cam_start)
            etype_label = (
                "Person"
                if entity["entity_type"] == "person"
                else (entity.get("vehicle_type") or "Vehicle")
            )
            st.markdown(
                f"**{etype_label} T{entity['track_id']}** "
                f"({entity['camera_label']})  \n"
                f"{entity['description']}  \n"
                f"{t0} - {t1}"
            )


def page_person_of_interest(cameras: Optional[List[str]] = None) -> None:
    """Upload a reference photo to find matching persons across cameras."""
    st.header("Person of Interest")
    st.caption(
        "Upload a photo of a person to search for them "
        "across all processed camera footage."
    )

    uploaded = st.file_uploader(
        "Upload reference photo",
        type=["jpg", "jpeg", "png"],
        key="poi_upload",
    )
    threshold = st.slider(
        "Match sensitivity",
        min_value=0.30,
        max_value=0.80,
        value=0.45,
        step=0.05,
        help="Lower = more results (looser match). "
        "Higher = fewer but more confident matches.",
    )

    if not uploaded:
        st.info("Upload a photo to begin searching.")
        return

    # Display the reference image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    ref_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if ref_img is None:
        st.error("Could not read the uploaded image.")
        return

    col_ref, col_results = st.columns([1, 3])
    with col_ref:
        st.image(
            cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB),
            caption="Reference photo",
            use_container_width=True,
        )

    # Extract embedding from reference photo
    with st.spinner("Extracting features..."):
        from cctv_dissertation.person_reid import PersonReID

        reid = PersonReID(
            db_path=str(PROJECT_ROOT / "data" / "person_reid.db"),
            device="cpu",
        )
        query_emb = reid.extract_features(ref_img)

    # Compare against all person entities in DB
    if not TRACKER_DB.exists():
        st.warning("No tracking data available.")
        return

    conn = sqlite3.connect(str(TRACKER_DB))
    conn.row_factory = sqlite3.Row
    if cameras:
        ph = ",".join("?" for _ in cameras)
        rows = conn.execute(
            "SELECT * FROM tracked_entities "
            f"WHERE entity_type = 'person' AND embedding IS NOT NULL "
            f"AND camera_label IN ({ph}) "
            "ORDER BY camera_label, track_id",
            cameras,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tracked_entities "
            "WHERE entity_type = 'person' AND embedding IS NOT NULL "
            "ORDER BY camera_label, track_id"
        ).fetchall()
    conn.close()

    matches = []
    for row in rows:
        entity = dict(row)
        db_emb = np.frombuffer(entity["embedding"], dtype=np.float32)
        norm = np.linalg.norm(db_emb)
        if norm > 0:
            db_emb = db_emb / norm
        sim = float(np.dot(query_emb, db_emb))
        if sim >= threshold:
            entity["similarity"] = sim
            matches.append(entity)

    matches.sort(key=lambda x: x["similarity"], reverse=True)

    with col_results:
        if not matches:
            st.warning("No matches found. Try lowering sensitivity.")
            return

        st.success(f"Found {len(matches)} match(es)")

        for entity in matches:
            sim_pct = entity["similarity"] * 100
            cam = entity["camera_label"]
            cam_start = st.session_state.get(f"start_time_{cam}")
            t0 = _format_absolute_time(entity["first_ts"], cam_start)
            t1 = _format_absolute_time(entity["last_ts"], cam_start)

            cols = st.columns([1, 3])
            with cols[0]:
                img = _load_image(entity["crop_path"])
                if img is not None:
                    st.image(img, use_container_width=True)
            with cols[1]:
                st.markdown(
                    f"**{sim_pct:.0f}% match** — "
                    f"Person T{entity['track_id']} "
                    f"({cam})  \n"
                    f"{entity['description']}  \n"
                    f"Visible: {t0} - {t1}"
                )
            st.divider()


def _generate_forensic_pdf(
    identities: dict,
    cam_a: str,
    cam_b: str,
    start_time_a: Optional[str],
    start_time_b: Optional[str],
    date_a: Optional[str],
    date_b: Optional[str],
    video_path_a: Optional[str] = None,
    video_path_b: Optional[str] = None,
) -> bytes:
    """Generate comprehensive forensic PDF with full audit trail and metadata."""
    import hashlib
    from datetime import datetime

    from fpdf import FPDF

    GREY = (100, 100, 100)
    DARK = (30, 30, 30)
    ACCENT = (0, 90, 160)
    LIGHT_BG = (245, 245, 245)
    RED = (180, 0, 0)

    def _dur(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f}s"
        m = int(secs) // 60
        s = int(secs) % 60
        if m < 60:
            return f"{m}m {s}s"
        h = m // 60
        m = m % 60
        return f"{h}h {m}m"

    # Extract video metadata
    def get_video_metadata(video_path):
        if not video_path or not Path(video_path).exists():
            return None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        metadata = {
            "filename": Path(video_path).name,
            "filesize_mb": Path(video_path).stat().st_size / (1024 * 1024),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_sec": (
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
                if cap.get(cv2.CAP_PROP_FPS) > 0
                else 0
            ),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        cap.release()

        sha256_hash = hashlib.sha256()
        with open(video_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        metadata["sha256"] = sha256_hash.hexdigest()

        return metadata

    meta_a = get_video_metadata(video_path_a)
    meta_b = get_video_metadata(video_path_b)

    class ForensicPDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*GREY)
            self.cell(0, 8, "Forensic Video Analysis Report", ln=True)
            self.set_draw_color(*ACCENT)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*GREY)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = ForensicPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    def _st(camera):
        return start_time_a if camera == cam_a else start_time_b

    persons = identities["persons"]
    vehicles = identities["vehicles"]
    journeys = identities.get("journeys", [])
    n_matched = sum(1 for p in persons if p["matched"])

    def _resolve_crop(crop_path):
        p = Path(crop_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return str(p) if p.exists() else None

    def _section_header(title):
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(*ACCENT)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_draw_color(*ACCENT)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

    def _table_row(cells, widths, bold=False, bg=False):
        if bg:
            pdf.set_fill_color(*LIGHT_BG)
        pdf.set_font("Helvetica", "B" if bold else "", 8)
        pdf.set_text_color(*DARK)
        for i, (txt, w) in enumerate(zip(cells, widths)):
            pdf.cell(w, 6, str(txt), border=0, fill=bg, ln=(i == len(cells) - 1))

    # ── PAGE 1: Title ──
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 15, "Forensic Video Analysis", ln=True, align="C")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 10, "Cross-Camera Tracking Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_draw_color(*ACCENT)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)

    report_time = datetime.now()
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*GREY)
    pdf.cell(
        0,
        8,
        f"Generated: {report_time:%Y-%m-%d %H:%M:%S}",
        ln=True,
        align="C",
    )
    pdf.ln(15)

    # Case overview box
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*DARK)
    pdf.cell(0, 8, "Case Summary", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    for cam, d, t in [(cam_a, date_a, start_time_a), (cam_b, date_b, start_time_b)]:
        ts_info = f"{d} {t}" if d and t else "not recorded"
        pdf.cell(0, 6, f"  Camera: {cam}  |  Start: {ts_info}", ln=True)
    pdf.ln(4)

    # Key figures in a row
    n_returns = len(identities.get("round_trips", []))
    pdf.set_font("Helvetica", "B", 11)
    col_w = 47
    for lbl, val in [
        ("Persons", str(len(persons))),
        ("Vehicles", str(len(vehicles))),
        ("Matches", str(n_matched)),
        ("Returns", str(n_returns)),
    ]:
        pdf.cell(col_w, 12, f"  {lbl}: {val}", border=1, align="L")
    pdf.ln(12)

    # ── PAGE 2: Executive Summary - On/Off Camera ──
    pdf.add_page()
    _section_header("Activity Summary")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GREY)
    pdf.cell(
        0,
        5,
        "On/off camera durations for each identified individual.",
        ln=True,
    )
    pdf.ln(4)

    # Table header
    hdr = [
        "ID",
        "Cameras",
        "First Seen",
        "Last Seen",
        "Sightings",
        "On Camera",
        "Off Camera",
    ]
    widths = [18, 40, 28, 28, 20, 27, 27]
    _table_row(hdr, widths, bold=True, bg=True)

    for identity in persons:
        uid = identity["unified_id"]
        raw = identity.get("raw_sightings", identity["sightings"])
        cameras = sorted(set(s["camera"] for s in raw))
        total_on = sum(s["last_ts"] - s["first_ts"] for s in raw)
        total_off = 0.0
        for i in range(1, len(raw)):
            g = raw[i]["first_ts"] - raw[i - 1]["last_ts"]
            if g > 0:
                total_off += g
        first_s = raw[0]
        last_s = raw[-1]
        _table_row(
            [
                f"P{uid}",
                ", ".join(cameras),
                _format_absolute_time(first_s["first_ts"], _st(first_s["camera"])),
                _format_absolute_time(last_s["last_ts"], _st(last_s["camera"])),
                str(len(raw)),
                _dur(total_on),
                _dur(total_off) if total_off > 0 else "-",
            ],
            widths,
        )

    for identity in vehicles:
        uid = identity["unified_id"]
        sightings = identity["sightings"]
        cameras = sorted(set(s["camera"] for s in sightings))
        total_on = sum(s["last_ts"] - s["first_ts"] for s in sightings)
        first_s = sightings[0]
        last_s = sightings[-1]
        _table_row(
            [
                f"V{uid}",
                ", ".join(cameras),
                _format_absolute_time(first_s["first_ts"], _st(first_s["camera"])),
                _format_absolute_time(last_s["last_ts"], _st(last_s["camera"])),
                str(len(sightings)),
                _dur(total_on),
                "-",
            ],
            widths,
        )

    # ── Journey summary (if any) ──
    if journeys:
        pdf.ln(10)
        _section_header("Movement Events")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(
            0,
            5,
            "Person-vehicle linkages based on temporal proximity.",
            ln=True,
        )
        pdf.ln(4)

        j_hdr = ["Event", "Person", "Vehicle", "Time", "Gap", "Confidence"]
        j_widths = [28, 18, 45, 28, 24, 22]
        _table_row(j_hdr, j_widths, bold=True, bg=True)

        for j in journeys:
            cam_st = _st(j["camera"])
            evt = "Departure" if j["event"] == "departure" else "Arrival"
            _table_row(
                [
                    evt,
                    f"P{j['person_id']}",
                    f"V{j['vehicle_id']} ({j['vehicle_desc']})",
                    _format_absolute_time(j["timestamp"], cam_st),
                    _dur(j["gap_seconds"]),
                    f"{j['confidence'] * 100:.0f}%",
                ],
                j_widths,
            )

    # ── Video Source Metadata ──
    pdf.add_page()
    _section_header("Video Source Metadata")

    for cam_name, meta in [(cam_a, meta_a), (cam_b, meta_b)]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 7, f"Camera: {cam_name}", ln=True)
        pdf.set_font("Helvetica", "", 9)

        if meta:
            details = [
                ("Filename", meta["filename"]),
                ("File Size", f"{meta['filesize_mb']:.2f} MB"),
                ("Resolution", f"{meta['width']} x {meta['height']}"),
                ("Frame Rate", f"{meta['fps']:.2f} fps"),
                ("Total Frames", f"{meta['frame_count']:,}"),
                ("Duration", _dur(meta["duration_sec"])),
                ("SHA256", meta["sha256"]),
            ]
            for lbl, value in details:
                pdf.cell(35, 5, f"  {lbl}:", ln=False)
                pdf.cell(0, 5, str(value), ln=True)
        else:
            pdf.cell(0, 5, "  Metadata unavailable", ln=True)
        pdf.ln(5)

    # Processing info
    pdf.ln(2)
    _section_header("Processing Information")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK)

    processing_info = [
        ("Detection Model", "YOLOv8n (COCO pretrained)"),
        ("Tracking", "ByteTrack"),
        ("Person Re-ID", "OSNet x1_0 (torchreid)"),
        ("Vehicle Re-ID", "OSNet x0_25 (torchreid)"),
        ("Person Threshold", "0.45 (cross-camera)"),
        ("Vehicle Threshold", "0.80 (cross-camera)"),
        ("Color Extraction", "K-means (k=5) with background filtering"),
    ]
    for lbl, value in processing_info:
        pdf.cell(40, 5, f"  {lbl}:", ln=False)
        pdf.cell(0, 5, str(value), ln=True)

    # Chain of custody
    pdf.ln(6)
    _section_header("Chain of Custody")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK)
    for lbl, value in [
        ("Examiner", "System User"),
        ("Examination Date", f"{report_time:%Y-%m-%d}"),
        ("Processing Method", "Automated video analysis pipeline"),
        ("Evidence Integrity", "SHA256 hashes recorded for source files"),
        ("Report Status", "PRELIMINARY - For investigative use"),
    ]:
        pdf.cell(40, 5, f"  {lbl}:", ln=False)
        pdf.cell(0, 5, str(value), ln=True)

    # ── Persons Detail ──
    pdf.add_page()
    _section_header(f"Person Detail ({len(persons)})")

    for identity in persons:
        uid = identity["unified_id"]
        raw = identity.get("raw_sightings", identity["sightings"])
        matched = identity["matched"]
        sim = identity["similarity"]
        cameras = sorted(set(s["camera"] for s in raw))
        n_appearances = len(raw)

        if pdf.get_y() > 190:
            pdf.add_page()

        # Person header with background bar
        pdf.set_fill_color(*LIGHT_BG)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*DARK)
        title = f"Person {uid}"
        if len(cameras) > 1:
            title += f" - {n_appearances} appearances across {len(cameras)} cameras"
        else:
            cam_name = cameras[0] if cameras else "unknown"
            sfx = "s" if n_appearances > 1 else ""
            title += f" - {n_appearances} appearance{sfx} on {cam_name}"
        if matched and sim:
            title += f"  ({sim * 100:.0f}% match)"
        pdf.cell(0, 9, f"  {title}", ln=True, fill=True)
        pdf.ln(4)

        # Sighting rows
        img_w = 22
        for idx, s in enumerate(raw):
            if pdf.get_y() > 235:
                pdf.add_page()

            cam_st = _st(s["camera"])
            t0 = _format_absolute_time(s["first_ts"], cam_st)
            t1 = _format_absolute_time(s["last_ts"], cam_st)
            on_dur = s["last_ts"] - s["first_ts"]

            x_start = pdf.get_x()
            y_start = pdf.get_y()

            # Crop image
            crop = _resolve_crop(s["crop_path"])
            if crop:
                try:
                    pdf.image(crop, x=x_start, y=y_start, w=img_w)
                except Exception:
                    pass

            # Text next to crop
            text_x = x_start + img_w + 4
            pdf.set_xy(text_x, y_start)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*ACCENT)
            pdf.cell(0, 5, f"Sighting {idx + 1}: {s['camera']}", ln=True)
            pdf.set_x(text_x)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*DARK)
            pdf.cell(0, 4, s["description"], ln=True)
            pdf.set_x(text_x)
            pdf.cell(0, 4, f"Visible: {t0} - {t1}  ({_dur(on_dur)})", ln=True)

            # Off-camera gap from previous sighting
            if idx > 0:
                prev = raw[idx - 1]
                gap = _calc_real_gap_standalone(prev, s, _st)
                if gap is not None and gap >= 0:
                    pdf.set_x(text_x)
                    pdf.set_font("Helvetica", "B", 8)
                    pdf.set_text_color(*RED)
                    transit = prev["camera"] != s["camera"]
                    lbl = "Transit" if transit else "Off camera"
                    pdf.cell(0, 4, f"{lbl}: {_dur(gap)}", ln=True)
                    pdf.set_text_color(*DARK)

            pdf.set_y(max(pdf.get_y(), y_start + 30))

        # Movement path
        if n_appearances > 1:
            pdf.ln(1)
            parts = []
            for s in raw:
                cam_st = _st(s["camera"])
                t = _format_absolute_time(s["first_ts"], cam_st)
                parts.append(f"{s['camera']} @ {t}")
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*ACCENT)
            pdf.multi_cell(0, 4, "Movement: " + " -> ".join(parts))
            pdf.set_text_color(*DARK)

        pdf.ln(4)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

    # ── Vehicles Detail ──
    if vehicles:
        pdf.add_page()
        _section_header(f"Vehicle Detail ({len(vehicles)})")

        for identity in vehicles:
            uid = identity["unified_id"]
            sightings = identity["sightings"]
            sim = identity["similarity"]

            if pdf.get_y() > 200:
                pdf.add_page()

            pdf.set_fill_color(*LIGHT_BG)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(*DARK)
            title = f"Vehicle {uid}"
            if identity["matched"] and sim:
                title += f"  ({sim * 100:.0f}% match)"
            pdf.cell(0, 9, f"  {title}", ln=True, fill=True)
            pdf.ln(4)

            x_start = pdf.get_x()
            y_start = pdf.get_y()
            img_w = 28
            imgs_shown = 0
            for s in sightings[:2]:
                crop = _resolve_crop(s["crop_path"])
                if crop:
                    try:
                        pdf.image(
                            crop,
                            x=x_start + imgs_shown * (img_w + 4),
                            y=y_start,
                            w=img_w,
                        )
                        imgs_shown += 1
                    except Exception:
                        pass

            text_x = x_start + imgs_shown * (img_w + 4) + 6
            pdf.set_xy(text_x, y_start)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*DARK)
            for s in sightings:
                cam_st = _st(s["camera"])
                t0 = _format_absolute_time(s["first_ts"], cam_st)
                t1 = _format_absolute_time(s["last_ts"], cam_st)
                on_dur = s["last_ts"] - s["first_ts"]
                pdf.set_x(text_x)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(0, 4, s["camera"], ln=True)
                pdf.set_x(text_x)
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(0, 4, s["description"], ln=True)
                pdf.set_x(text_x)
                pdf.cell(0, 4, f"Visible: {t0} - {t1}  ({_dur(on_dur)})", ln=True)
                pdf.ln(2)

            pdf.set_y(max(pdf.get_y(), y_start + 50))
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(6)

    # ── Movement Events Detail ──
    if journeys:
        pdf.add_page()
        _section_header("Movement Event Detail")

        for j in journeys:
            if pdf.get_y() > 230:
                pdf.add_page()

            cam_st = _st(j["camera"])
            event_time = _format_absolute_time(j["timestamp"], cam_st)
            person_time = _format_absolute_time(j["person_ts"], cam_st)

            arrow = "departed in" if j["event"] == "departure" else "arrived in"

            pdf.set_fill_color(*LIGHT_BG)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*DARK)
            pdf.cell(
                0,
                8,
                f"  Person {j['person_id']} {arrow} "
                f"Vehicle {j['vehicle_id']} ({j['vehicle_desc']})",
                ln=True,
                fill=True,
            )
            pdf.ln(2)

            pdf.set_font("Helvetica", "", 9)
            if j["event"] == "departure":
                pdf.cell(0, 5, f"  Person last seen: {person_time}", ln=True)
                pdf.cell(0, 5, f"  Vehicle departed: {event_time}", ln=True)
            else:
                first_t = min(person_time, event_time)
                second_t = max(person_time, event_time)
                pdf.cell(0, 5, f"  Vehicle arrived: {first_t}", ln=True)
                pdf.cell(0, 5, f"  Person seen: {second_t}", ln=True)

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*RED)
            pdf.cell(
                0,
                5,
                f"  Gap: {_dur(j['gap_seconds'])}  |  "
                f"Confidence: {j['confidence'] * 100:.0f}%",
                ln=True,
            )
            pdf.set_text_color(*DARK)

            pdf.ln(4)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)

    # ── Vehicle Returns ──
    round_trips = identities.get("round_trips", [])
    if round_trips:
        pdf.add_page()
        _section_header("Vehicle Returns")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(
            0,
            5,
            "Vehicles matched departing and returning via Re-ID.",
            ln=True,
        )
        pdf.ln(4)

        rt_hdr = ["Vehicle", "Departed", "By", "Returned", "By", "Away"]
        rt_widths = [45, 28, 18, 28, 18, 25]
        _table_row(rt_hdr, rt_widths, bold=True, bg=True)

        for rt in round_trips:
            dep = rt["departure"]
            arr = rt["arrival"]
            dep_time = _format_absolute_time(dep["timestamp"], _st(dep["camera"]))
            arr_time = _format_absolute_time(arr["timestamp"], _st(arr["camera"]))
            _table_row(
                [
                    f"V{rt['vehicle_id']} ({rt['vehicle_desc']})",
                    dep_time,
                    f"P{dep['person_id']}",
                    arr_time,
                    f"P{arr['person_id']}",
                    _dur(rt["away_seconds"]),
                ],
                rt_widths,
            )

        pdf.ln(6)
        for rt in round_trips:
            if pdf.get_y() > 230:
                pdf.add_page()

            dep = rt["departure"]
            arr = rt["arrival"]
            dep_time = _format_absolute_time(dep["timestamp"], _st(dep["camera"]))
            arr_time = _format_absolute_time(arr["timestamp"], _st(arr["camera"]))

            pdf.set_fill_color(*LIGHT_BG)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*DARK)
            pdf.cell(
                0,
                8,
                f"  Vehicle {rt['vehicle_id']} ({rt['vehicle_desc']})"
                f" - Away {_dur(rt['away_seconds'])}",
                ln=True,
                fill=True,
            )
            pdf.ln(2)

            pdf.set_font("Helvetica", "", 9)
            pdf.cell(
                0,
                5,
                f"  Departed: {dep_time} on {dep['camera']}"
                f" (Person {dep['person_id']})",
                ln=True,
            )
            pdf.cell(
                0,
                5,
                f"  Returned: {arr_time} on {arr['camera']}"
                f" (Person {arr['person_id']})",
                ln=True,
            )

            if rt["same_person"]:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(*ACCENT)
                pdf.cell(
                    0,
                    5,
                    f"  Same person (P{dep['person_id']}) " f"departed and returned",
                    ln=True,
                )
            else:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(*GREY)
                pdf.cell(
                    0,
                    5,
                    f"  Different persons: P{dep['person_id']} out, "
                    f"P{arr['person_id']} in",
                    ln=True,
                )
            pdf.set_text_color(*DARK)

            pdf.ln(4)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(4)

    return bytes(pdf.output())


def _show_cross_matches_summary(cam_a: str, cam_b: str) -> None:
    """Show unified identities summary below dual view."""
    identities = _get_unified_identities(cam_a, cam_b)
    persons = identities["persons"]
    vehicles = identities["vehicles"]

    def _get_start(cam: str) -> Optional[str]:
        return st.session_state.get(f"start_time_{cam}")

    if persons:
        st.subheader(f"Persons ({len(persons)})")
        for p in persons:
            uid = p["unified_id"]
            raw = p.get("raw_sightings", p["sightings"])
            cameras = sorted(set(s["camera"] for s in raw))
            n = len(raw)
            label = f"Person {uid}"
            if len(cameras) > 1:
                label += f" — {n} appearances across " f"{len(cameras)} cameras"
            else:
                label += f" — {n} appearance" f"{'s' if n > 1 else ''}"

            with st.expander(label, expanded=p["matched"]):
                max_cols = min(n, 5)
                for row_start in range(0, n, max_cols):
                    row = raw[row_start : row_start + max_cols]
                    cols = st.columns(len(row))
                    for i, s in enumerate(row):
                        with cols[i]:
                            img = _load_image(s["crop_path"])
                            if img is not None:
                                st.image(
                                    img,
                                    use_container_width=True,
                                )
                            cam_start = _get_start(s["camera"])
                            t0 = _format_absolute_time(s["first_ts"], cam_start)
                            t1 = _format_absolute_time(s["last_ts"], cam_start)
                            st.caption(
                                f"**{s['camera']}**  \n"
                                f"{s['description']}  \n"
                                f"{t0} - {t1}"
                            )

    if vehicles:
        st.subheader(f"Vehicles ({len(vehicles)})")
        for v in vehicles:
            uid = v["unified_id"]
            sightings = v["sightings"]
            label = f"Vehicle {uid}"
            if v["matched"]:
                label += f" — {len(sightings)} cameras"
            with st.expander(label, expanded=v["matched"]):
                cols = st.columns(len(sightings))
                for i, s in enumerate(sightings):
                    with cols[i]:
                        img = _load_image(s["crop_path"])
                        if img is not None:
                            st.image(
                                img,
                                use_container_width=True,
                            )
                        cam_start = _get_start(s["camera"])
                        t0 = _format_absolute_time(s["first_ts"], cam_start)
                        t1 = _format_absolute_time(s["last_ts"], cam_start)
                        st.caption(
                            f"**{s['camera']}**  \n"
                            f"{s['description']}  \n"
                            f"{t0} - {t1}"
                        )

    if not persons and not vehicles:
        st.info("No tracked entities found.")


# ── Utility helpers ───────────────────────────────────────────────


def _read_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _overlay_tracked_boxes(
    frame: np.ndarray, frame_idx: int, camera_label: str
) -> np.ndarray:
    """Draw tracked bounding boxes from the DB onto a frame."""
    if not TRACKER_DB.exists():
        return frame

    conn = sqlite3.connect(str(TRACKER_DB))
    rows = conn.execute(
        "SELECT te.entity_type, te.track_id, te.description, "
        "tf.bbox_x1, tf.bbox_y1, tf.bbox_x2, tf.bbox_y2 "
        "FROM track_frames tf "
        "JOIN tracked_entities te ON tf.entity_id = te.id "
        "WHERE te.camera_label = ? AND tf.frame_idx = ?",
        (camera_label, frame_idx),
    ).fetchall()
    conn.close()

    for etype, tid, desc, x1, y1, x2, y2 in rows:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if etype == "person":
            color = (0, 255, 0)
            label = f"Person {tid}: {desc}"
        else:
            color = (255, 100, 0)
            label = f"V{tid}: {desc}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


# ── Legacy detection pages (kept for compatibility) ───────────────


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


def class_color(name: str) -> Tuple[int, int, int]:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(digest[0:2], 16), int(digest[2:4], 16), int(digest[4:6], 16)


def overlay_detections_on_frame(
    frame: np.ndarray, detections: List[dict]
) -> np.ndarray:
    overlay = frame.copy()
    for det in detections:
        bbox = det.get("bbox_xyxy") or [0, 0, 0, 0]
        class_name = det.get("class_name") or str(det.get("class_id"))
        r, g, b = class_color(class_name)
        label = f"{class_name} {det.get('confidence', 0):.2f}"
        cv2.rectangle(
            overlay,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (int(b), int(g), int(r)),
            2,
        )
        cv2.putText(
            overlay,
            label,
            (int(bbox[0]), max(0, int(bbox[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (int(b), int(g), int(r)),
            2,
            cv2.LINE_AA,
        )
    return overlay


def page_tamper_detection(cameras: List[str]) -> None:
    """Video tamper detection analysis page."""
    from cctv_dissertation.tamper_detection import analyze_video

    st.header("Video Tamper Detection")
    st.caption(
        "Automatic analysis for signs of tampering: frame drops, "
        "splices, re-encoding artefacts, and timestamp anomalies. "
        "Results are generated when videos are processed."
    )

    # Get video paths for each camera from DB
    video_paths: dict[str, Optional[str]] = {}
    if TRACKER_DB.exists():
        conn = sqlite3.connect(str(TRACKER_DB))
        for cam in cameras:
            row = conn.execute(
                "SELECT DISTINCT video_path FROM tracked_entities "
                "WHERE camera_label = ? LIMIT 1",
                (cam,),
            ).fetchone()
            video_paths[cam] = row[0] if row else None
        conn.close()

    if not any(video_paths.values()):
        st.info("No videos found. Process videos first.")
        return

    # Initialise session state for tamper reports
    if "tamper_reports" not in st.session_state:
        st.session_state["tamper_reports"] = {}

    # Analysis controls
    cols = st.columns(len(cameras) + 1)
    for i, cam in enumerate(cameras):
        vp = video_paths.get(cam)
        with cols[i]:
            if vp and Path(vp).exists():
                already_run = cam in st.session_state["tamper_reports"]
                btn_label = f"Re-analyse {cam}" if already_run else f"Analyse {cam}"
                if st.button(btn_label, key=f"tamper_btn_{cam}"):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def _progress(pct, msg, _bar=progress_bar, _txt=status_text):
                        _bar.progress(min(pct, 1.0))
                        _txt.text(msg)

                    report = analyze_video(vp, progress_callback=_progress)
                    st.session_state["tamper_reports"][cam] = report.to_dict()
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()
            else:
                st.write(f"{cam}: video not found")

    with cols[-1]:
        if st.button("Analyse All", key="tamper_all"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            cams_to_run = [
                c
                for c in cameras
                if video_paths.get(c) and Path(video_paths[c]).exists()
            ]
            for ci, cam in enumerate(cams_to_run):
                base_pct = ci / len(cams_to_run)
                step = 1.0 / len(cams_to_run)

                def _progress(
                    pct,
                    msg,
                    _b=base_pct,
                    _s=step,
                    _bar=progress_bar,
                    _txt=status_text,
                    _c=cam,
                ):
                    _bar.progress(min(_b + pct * _s, 1.0))
                    _txt.text(f"[{_c}] {msg}")

                report = analyze_video(video_paths[cam], progress_callback=_progress)
                st.session_state["tamper_reports"][cam] = report.to_dict()
            progress_bar.empty()
            status_text.empty()
            st.rerun()

    # Display results
    for cam in cameras:
        report_dict = st.session_state.get("tamper_reports", {}).get(cam)
        if not report_dict:
            continue

        st.divider()
        risk = report_dict["overall_risk"]
        risk_colors = {
            "clean": "green",
            "low": "orange",
            "medium": "orange",
            "high": "red",
        }
        risk_icons = {
            "clean": "**CLEAN**",
            "low": "LOW RISK",
            "medium": "**MEDIUM RISK**",
            "high": "**HIGH RISK**",
        }

        col_title, col_badge = st.columns([3, 1])
        with col_title:
            st.subheader(f"{cam}")
            st.caption(f"SHA-256: {report_dict['sha256'][:32]}...")
        with col_badge:
            colour = risk_colors.get(risk, "grey")
            st.markdown(f":{colour}[{risk_icons.get(risk, risk.upper())}]")

        # Flag summary
        flags = report_dict.get("flags", [])
        n_crit = sum(1 for f in flags if f["severity"] == "critical")
        n_warn = sum(1 for f in flags if f["severity"] == "warning")
        n_info = sum(1 for f in flags if f["severity"] == "info")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Critical", n_crit)
        mc2.metric("Warnings", n_warn)
        mc3.metric("Info", n_info)

        # Structural summary
        struct = report_dict.get("structural_summary", {})
        if struct and "error" not in struct:
            with st.expander("Structural Analysis", expanded=False):
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Packets", f"{struct.get('packet_count', 0):,}")
                sc2.metric(
                    "Frame Duration",
                    f"{struct.get('median_frame_duration', 0) * 1000:.1f}ms",
                )
                sc3.metric(
                    "GOP Length",
                    f"{struct.get('gop_mean_length', 0):.0f} frames",
                )
                if struct.get("large_gaps_found", 0) > 0:
                    st.warning(f"{struct['large_gaps_found']} frame gap(s) detected")

        # Quality summary + SSIM chart
        qual = report_dict.get("quality_summary", {})
        ssim_data = qual.get("ssim_timeline", [])
        if qual and "error" not in qual:
            with st.expander("Frame Quality Analysis", expanded=bool(ssim_data)):
                qc1, qc2, qc3 = st.columns(3)
                qc1.metric("Mean SSIM", f"{qual.get('mean_ssim', 0):.4f}")
                qc2.metric("Min SSIM", f"{qual.get('min_ssim', 0):.4f}")
                qc3.metric(
                    "Frames Sampled",
                    f"{qual.get('frames_sampled', 0):,}",
                )

                if ssim_data:
                    df_ssim = pd.DataFrame(ssim_data)
                    chart = (
                        alt.Chart(df_ssim)
                        .mark_line(strokeWidth=1)
                        .encode(
                            x=alt.X(
                                "timestamp:Q",
                                title="Time (seconds)",
                            ),
                            y=alt.Y(
                                "ssim:Q",
                                title="SSIM",
                                scale=alt.Scale(domain=[0, 1]),
                            ),
                            tooltip=["timestamp", "ssim", "frame"],
                        )
                        .properties(height=200)
                        .interactive()
                    )
                    st.altair_chart(chart, use_container_width=True)

        # Metadata summary
        meta = report_dict.get("metadata_summary", {})
        if meta and "error" not in meta:
            with st.expander("Metadata Consistency", expanded=False):
                mc1, mc2 = st.columns(2)
                mc1.write(f"**Format:** {meta.get('container_format', '?')}")
                mc1.write(f"**Codec:** {meta.get('codec', '?')}")
                mc2.write(f"**Duration:** {meta.get('container_duration', 0):.1f}s")
                mc2.write(f"**Size:** {meta.get('file_size_mb', 0):.1f} MB")

        # Compression summary
        comp = report_dict.get("compression_summary", {})
        if comp and "error" not in comp:
            with st.expander("Compression Analysis (ELA)", expanded=False):
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Mean ELA", f"{comp.get('mean_ela', 0):.3f}")
                cc2.metric("Std ELA", f"{comp.get('std_ela', 0):.3f}")
                cc3.metric("Anomalies", comp.get("anomalies_found", 0))

        # Segment hashes
        seg_hashes = report_dict.get("segment_hashes", [])
        if seg_hashes:
            with st.expander(
                f"Segment Hashes ({len(seg_hashes)} segments)", expanded=False
            ):
                df_seg = pd.DataFrame(seg_hashes)
                df_seg = df_seg[["segment", "start_sec", "end_sec", "sha256"]]
                st.dataframe(df_seg, use_container_width=True, hide_index=True)

        # Detailed flags table
        if flags:
            with st.expander(f"All Flags ({len(flags)})", expanded=False):
                flag_rows = []
                for f in flags:
                    ts = (
                        f"{f['timestamp_sec']:.1f}s"
                        if f["timestamp_sec"] is not None
                        else "File-level"
                    )
                    flag_rows.append(
                        {
                            "Severity": f["severity"].upper(),
                            "Category": f["category"],
                            "Time": ts,
                            "Confidence": f"{f['confidence']:.0%}",
                            "Description": f["description"],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(flag_rows),
                    use_container_width=True,
                    hide_index=True,
                )


def page_evidence_export(cam_a: str, cam_b: str) -> None:
    """Evidence export package builder page."""
    from cctv_dissertation.evidence_export import ExportConfig, build_evidence_package

    st.header("Evidence Export Package")
    st.caption(
        "Generate a court-ready ZIP file containing the forensic report, "
        "crop images, hash manifest, and optional extras."
    )

    identities = _get_unified_identities(cam_a, cam_b)
    if not identities["persons"] and not identities["vehicles"]:
        st.info("No tracked entities found. Process videos first.")
        return

    # Options
    st.subheader("Package Contents")
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Forensic PDF Report", value=True, disabled=True, key="exp_pdf")
        st.checkbox("Crop Images", value=True, key="exp_crops")
        include_clips = st.checkbox("Video Clips", value=False, key="exp_clips")
    with col2:
        include_db = st.checkbox("Tracking Database", value=False, key="exp_db")
        include_tamper = st.checkbox(
            "Tamper Detection Reports",
            value=bool(st.session_state.get("tamper_reports")),
            key="exp_tamper",
            disabled=not st.session_state.get("tamper_reports"),
        )

    st.subheader("Case Information")
    examiner = st.text_input("Examiner Name", value="System User", key="exp_examiner")
    case_notes = st.text_area("Case Notes", value="", key="exp_notes", height=80)

    # Get video paths
    video_paths = []
    if TRACKER_DB.exists():
        conn = sqlite3.connect(str(TRACKER_DB))
        for cam in [cam_a, cam_b]:
            row = conn.execute(
                "SELECT DISTINCT video_path FROM tracked_entities "
                "WHERE camera_label = ? LIMIT 1",
                (cam,),
            ).fetchone()
            if row:
                video_paths.append(row[0])
        conn.close()

    # Generate PDF bytes (same as identity deep dive)
    start_time_a = st.session_state.get(f"start_time_{cam_a}")
    start_time_b = st.session_state.get(f"start_time_{cam_b}")
    date_a = st.session_state.get(f"start_date_{cam_a}")
    date_b = st.session_state.get(f"start_date_{cam_b}")

    if st.button("Generate Evidence Package", type="primary", key="gen_evidence"):
        with st.spinner("Building evidence package..."):
            # Generate fresh PDF
            pdf_bytes = _generate_forensic_pdf(
                identities,
                cam_a,
                cam_b,
                start_time_a,
                start_time_b,
                date_a,
                date_b,
                video_paths[0] if len(video_paths) > 0 else None,
                video_paths[1] if len(video_paths) > 1 else None,
            )

            # Collect tamper reports if selected
            tamper_reports = None
            if include_tamper and st.session_state.get("tamper_reports"):
                tamper_reports = list(st.session_state["tamper_reports"].values())

            config = ExportConfig(
                pdf_bytes=pdf_bytes,
                identities=identities,
                cameras=[cam_a, cam_b],
                db_path=str(TRACKER_DB),
                tracked_output_dir=str(PROJECT_ROOT / "data" / "tracked_output"),
                project_root=str(PROJECT_ROOT),
                video_paths=video_paths or None,
                tamper_reports=tamper_reports,
                include_db=include_db,
                include_clips=include_clips,
                clips_dir=str(PROJECT_ROOT / "data" / "clips"),
                examiner_name=examiner,
                case_notes=case_notes,
            )

            zip_bytes = build_evidence_package(config)

        st.success(f"Package ready ({len(zip_bytes) / 1024:.0f} KB)")
        timestamp = st.session_state.get(f"start_date_{cam_a}", "").replace("-", "")
        st.download_button(
            label="Download Evidence Package (ZIP)",
            data=zip_bytes,
            file_name=f"evidence_package_{timestamp}.zip",
            mime="application/zip",
            type="primary",
        )


def page_detections(selected_file, db_path: str) -> None:
    """Legacy detections page."""
    if selected_file is None:
        st.info("No detection reports loaded. Upload a video using the sidebar.")
        return

    summary = cached_summary(str(selected_file))
    report_data = load_detection_report(str(selected_file))

    # Hash verification
    source_path = report_data.get("source_path")
    if source_path and Path(source_path).exists():
        try:
            verification = verify_video_integrity(source_path)
            if verification["match"] == "MATCH":
                st.success(
                    "**Video Integrity Verified** — "
                    f"Hash matches manifest (ingested "
                    f"{verification.get('ingested_at', 'N/A')})"
                )
            elif verification["match"] == "MISMATCH":
                st.error("**HASH MISMATCH** — Video may be tampered!")
        except Exception:
            pass

    # Summary metrics
    st.subheader("Detection Summary")
    cols = st.columns(4)
    cols[0].metric("Frames analyzed", summary["frames_in_report"])
    cols[1].metric("Frames w/ detections", summary["frames_with_detections"])
    cols[2].metric("Total detections", summary["detections_total"])
    dur = summary["time_bounds"]
    span = (
        f"{dur['start_seconds']:.2f}s - {dur['end_seconds']:.2f}s"
        if dur["start_seconds"] is not None
        else "N/A"
    )
    cols[3].metric("Time span", span)

    # Class stats
    class_stats = summary.get("class_stats", {})
    if class_stats:
        records = [
            {
                "Class": name,
                "Count": s.get("count", 0),
                "First seen (s)": s.get("first_seen_time"),
                "Last seen (s)": s.get("last_seen_time"),
            }
            for name, s in class_stats.items()
        ]
        df = pd.DataFrame(records).sort_values("Count", ascending=False)
        st.dataframe(df, hide_index=True)

    # Visual preview
    video_path = report_data.get("source_path")
    frames = [f for f in report_data.get("detections", []) if f.get("detections")]
    if video_path and Path(video_path).exists() and frames:
        st.subheader("Visual Preview")
        metadata = report_data.get("metadata") or {}
        fps = metadata.get("frame_rate") or 25.0
        last_ts = frames[-1].get("timestamp_seconds")
        max_dur = (
            metadata.get("duration_seconds")
            or last_ts
            or max(f.get("frame_index", 0) for f in frames) / max(fps, 1e-6)
        )

        ts = st.slider(
            "Frame scrubber (seconds)",
            0.0,
            max(max_dur, 0.1),
            0.0,
            step=max(1.0 / fps, 0.05),
            key="det_scrub",
        )

        def _frame_time(f):
            if f.get("timestamp_seconds") is not None:
                return float(f["timestamp_seconds"])
            return (f.get("frame_index") or 0) / max(fps, 1e-6)

        best = min(frames, key=lambda f: abs(_frame_time(f) - ts))
        try:
            raw = _read_frame(video_path, best.get("frame_index", 0))
            if raw is not None:
                annotated = overlay_detections_on_frame(raw, best["detections"])
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(
                    annotated,
                    caption=f"Frame {best.get('frame_index')} "
                    f"@ {best.get('timestamp_seconds', 0):.2f}s",
                    use_container_width=True,
                )
        except Exception as exc:
            st.error(f"Unable to render: {exc}")

    # Query interface
    st.divider()
    st.subheader("Investigator Query")
    query_mode = st.radio("Mode", ["Detections", "Tracks"], horizontal=True)
    class_filter = st.text_input("Class filter", placeholder="e.g. car")
    c1, c2, c3 = st.columns(3)
    with c1:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.3, 0.05)
    with c2:
        time_start = st.number_input("Start time (s)", 0.0, value=0.0)
    with c3:
        end_val = float(summary["time_bounds"]["end_seconds"] or 0.0)
        time_end = st.number_input("End time (s)", 0.0, value=end_val)

    if st.button("Run query", type="primary"):
        try:
            if query_mode == "Tracks":
                data = query_tracks(
                    summary["sha256"],
                    db_path=db_path,
                    class_name=class_filter or None,
                )
                if data:
                    rows = [
                        {
                            "Track": d["track_label"],
                            "Class": d["class_name"],
                            "Detections": d["detections_count"],
                            "Start (s)": d["start_time"],
                            "End (s)": d["end_time"],
                        }
                        for d in data
                    ]
                    st.dataframe(pd.DataFrame(rows), hide_index=True)
                else:
                    st.warning("No tracks found.")
            else:
                tr = (
                    time_start if time_start > 0 else None,
                    time_end if time_end > 0 else None,
                )
                data = query_detections(
                    summary["sha256"],
                    db_path=db_path,
                    class_name=class_filter or None,
                    min_conf=min_conf,
                    time_range=tr,
                )
                if data:
                    st.dataframe(pd.DataFrame(data), hide_index=True)
                else:
                    st.warning("No detections found.")
        except Exception as exc:
            st.error(f"Query failed: {exc}")


# ── Upload handlers ───────────────────────────────────────────────


def handle_single_upload(uploaded_file) -> None:
    """Process a single video with ByteTrack tracking."""
    import uuid

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOADS_DIR / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())

    camera_label = target.stem
    run_id = str(uuid.uuid4())[:12]

    # Ingest for chain-of-custody
    ingest_video(target)

    from cctv_dissertation.tracker import SingleCameraTracker

    # Clear previous data for this camera + stale clips
    shutil.rmtree(CLIPS_DIR, ignore_errors=True)
    if TRACKER_DB.exists():
        conn = sqlite3.connect(str(TRACKER_DB))
        conn.execute(
            "DELETE FROM track_frames WHERE entity_id IN "
            "(SELECT id FROM tracked_entities WHERE camera_label = ?)",
            (camera_label,),
        )
        conn.execute(
            "DELETE FROM tracked_entities WHERE camera_label = ?",
            (camera_label,),
        )
        conn.commit()
        conn.close()

    output_dir = TRACKED_OUTPUT / camera_label
    tracker = SingleCameraTracker(
        plate_model_path=str(PROJECT_ROOT / "models" / "license_plate_detector.pt"),
        db_path=str(TRACKER_DB),
    )

    # Progress UI
    progress_bar = st.progress(0, text="Initializing tracker...")
    status_text = st.empty()
    progress_state = {"last_update": 0}

    def update_progress(info):
        pct = info["frame"] / info["total"] if info["total"] > 0 else 0
        eta_min = info["eta_seconds"] / 60
        progress_bar.progress(
            pct,
            text=f"Frame {info['frame']:,}/{info['total']:,} "
            f"({pct * 100:.1f}%) — ETA: {eta_min:.1f}min",
        )
        status_text.caption(
            f"Persons: {info['persons']} | "
            f"Vehicles: {info['vehicles']} | "
            f"Skipped (static): {info['skipped']:,} | "
            f"Speed: {info['fps_processing']:.1f} fps"
        )
        progress_state["last_update"] = info["frame"]

    # Auto-calculate optimal stride based on video duration
    from cctv_dissertation.tracker import calc_auto_stride

    stride = calc_auto_stride(str(target))
    if stride > 1:
        cap = cv2.VideoCapture(str(target))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        effective_fps = fps / stride
        st.info(
            f"Auto stride: processing every {stride}th frame "
            f"({effective_fps:.1f} effective fps) for faster processing"
        )

    result = tracker.process_video(
        video_path=str(target),
        output_dir=str(output_dir),
        camera_label=camera_label,
        progress_callback=update_progress,
        frame_stride=stride,
        run_id=run_id,
    )
    progress_bar.empty()
    status_text.empty()

    # Auto-generate annotated video
    ann_video = TRACKED_OUTPUT / camera_label / "annotated.mp4"
    with st.spinner("Rendering annotated video..."):
        from cctv_dissertation.tracker import generate_annotated_video

        generate_annotated_video(
            str(target),
            str(TRACKER_DB),
            str(ann_video),
            camera_label=camera_label,
        )

    # Auto-extract timestamp from video overlay
    from cctv_dissertation.tracker import extract_video_timestamp

    with st.spinner("Reading video timestamp..."):
        ts = extract_video_timestamp(str(target))
    if ts:
        # ts = "YYYY-MM-DD HH:MM:SS" -> store just the time
        st.session_state[f"start_time_{camera_label}"] = ts.split(" ")[1]
        st.session_state[f"start_date_{camera_label}"] = ts.split(" ")[0]

    st.success(
        f"Tracked {len(result['persons'])} people and "
        f"{len(result['vehicles'])} vehicles."
    )

    # Auto-run tamper detection
    from cctv_dissertation.tamper_detection import analyze_video as _tamper

    with st.spinner("Running tamper detection..."):
        tamper = _tamper(str(target))
        if "tamper_reports" not in st.session_state:
            st.session_state["tamper_reports"] = {}
        st.session_state["tamper_reports"][camera_label] = tamper.to_dict()

    st.session_state["active_camera"] = camera_label
    st.session_state["current_run_id"] = run_id
    st.session_state["upload_mode"] = "single"
    st.rerun()


def handle_cross_upload(file_a, file_b) -> None:
    """Process two videos for cross-camera tracking."""
    import uuid

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    target_a = UPLOADS_DIR / file_a.name
    target_b = UPLOADS_DIR / file_b.name
    target_a.write_bytes(file_a.getbuffer())
    target_b.write_bytes(file_b.getbuffer())

    cam_a = target_a.stem
    cam_b = target_b.stem

    # Generate a unique run_id for this processing session
    run_id = str(uuid.uuid4())[:12]

    ingest_video(target_a)
    ingest_video(target_b)

    from cctv_dissertation.tracker import SingleCameraTracker

    # Clear previous data for these cameras + stale clips
    shutil.rmtree(CLIPS_DIR, ignore_errors=True)
    if TRACKER_DB.exists():
        conn = sqlite3.connect(str(TRACKER_DB))
        for cam in [cam_a, cam_b]:
            conn.execute(
                "DELETE FROM track_frames WHERE entity_id IN "
                "(SELECT id FROM tracked_entities WHERE camera_label = ?)",
                (cam,),
            )
            conn.execute(
                "DELETE FROM tracked_entities WHERE camera_label = ?",
                (cam,),
            )
        conn.commit()
        conn.close()

    import threading
    from cctv_dissertation.tracker import calc_auto_stride

    stride_a = calc_auto_stride(str(target_a))
    stride_b = calc_auto_stride(str(target_b))

    # ── Dual side-by-side progress bars ───────────────────────────────────
    st.subheader("Processing cameras in parallel")
    col_a, col_b = st.columns(2)

    with col_a:
        st.caption(f"**Camera A** — {file_a.name}")
        cap = cv2.VideoCapture(str(target_a))
        fps_a = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        st.info(f"Stride {stride_a} ({fps_a / stride_a:.1f} fps effective)")
        prog_a = st.progress(0, text="Initializing...")
        stat_a = st.empty()

    with col_b:
        st.caption(f"**Camera B** — {file_b.name}")
        cap = cv2.VideoCapture(str(target_b))
        fps_b = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        st.info(f"Stride {stride_b} ({fps_b / stride_b:.1f} fps effective)")
        prog_b = st.progress(0, text="Initializing...")
        stat_b = st.empty()

    # Shared progress state — written by threads, read by main thread
    shared = {
        "a": {
            "frame": 0,
            "total": 1,
            "persons": 0,
            "vehicles": 0,
            "eta_seconds": 0,
            "skipped": 0,
        },
        "b": {
            "frame": 0,
            "total": 1,
            "persons": 0,
            "vehicles": 0,
            "eta_seconds": 0,
            "skipped": 0,
        },
        "done_a": False,
        "done_b": False,
        "result_a": None,
        "result_b": None,
        "error_a": None,
        "error_b": None,
    }

    def run_camera_a():
        tracker_a = SingleCameraTracker(
            plate_model_path=str(PROJECT_ROOT / "models" / "license_plate_detector.pt"),
            db_path=str(TRACKER_DB),
        )
        try:
            shared["result_a"] = tracker_a.process_video(
                video_path=str(target_a),
                output_dir=str(TRACKED_OUTPUT / cam_a),
                camera_label=cam_a,
                progress_callback=lambda info: shared["a"].update(info),
                frame_stride=stride_a,
                run_id=run_id,
            )
        except Exception as e:
            shared["error_a"] = str(e)
        finally:
            shared["done_a"] = True

    def run_camera_b():
        tracker_b = SingleCameraTracker(
            plate_model_path=str(PROJECT_ROOT / "models" / "license_plate_detector.pt"),
            db_path=str(TRACKER_DB),
        )
        try:
            shared["result_b"] = tracker_b.process_video(
                video_path=str(target_b),
                output_dir=str(TRACKED_OUTPUT / cam_b),
                camera_label=cam_b,
                progress_callback=lambda info: shared["b"].update(info),
                frame_stride=stride_b,
                run_id=run_id,
            )
        except Exception as e:
            shared["error_b"] = str(e)
        finally:
            shared["done_b"] = True

    t_a = threading.Thread(target=run_camera_a, daemon=True)
    t_b = threading.Thread(target=run_camera_b, daemon=True)
    t_a.start()
    t_b.start()

    # Poll and update UI until both threads finish
    import time as _time

    while not (shared["done_a"] and shared["done_b"]):
        for key, prog, stat, label in [
            ("a", prog_a, stat_a, "Camera A"),
            ("b", prog_b, stat_b, "Camera B"),
        ]:
            info = shared[key]
            pct = info["frame"] / info["total"] if info["total"] > 0 else 0
            eta = info["eta_seconds"] / 60
            done = shared[f"done_{key}"]
            prog.progress(
                min(pct, 1.0),
                text=(
                    "Done ✓"
                    if done
                    else f"{info['frame']:,}/{info['total']:,} — ETA {eta:.1f} min"
                ),
            )
            stat.caption(
                f"Persons: {info['persons']} | "
                f"Vehicles: {info['vehicles']} | "
                f"Skipped: {info['skipped']:,}"
            )
        _time.sleep(0.4)

    t_a.join()
    t_b.join()

    # Final 100% update
    for key, prog, stat in [("a", prog_a, stat_a), ("b", prog_b, stat_b)]:
        info = shared[key]
        prog.progress(1.0, text="Done ✓")
        stat.caption(f"Persons: {info['persons']} | Vehicles: {info['vehicles']}")

    if shared["error_a"] or shared["error_b"]:
        if shared["error_a"]:
            st.error(f"Camera A error: {shared['error_a']}")
        if shared["error_b"]:
            st.error(f"Camera B error: {shared['error_b']}")
        return

    result_a = shared["result_a"]
    result_b = shared["result_b"]

    # Auto-generate annotated videos for both cameras
    from cctv_dissertation.tracker import generate_annotated_video

    with st.spinner("Rendering annotated video for Camera A..."):
        generate_annotated_video(
            str(target_a),
            str(TRACKER_DB),
            str(TRACKED_OUTPUT / cam_a / "annotated.mp4"),
            camera_label=cam_a,
        )
    with st.spinner("Rendering annotated video for Camera B..."):
        generate_annotated_video(
            str(target_b),
            str(TRACKER_DB),
            str(TRACKED_OUTPUT / cam_b / "annotated.mp4"),
            camera_label=cam_b,
        )

    # Auto-extract timestamps from video overlays
    from cctv_dissertation.tracker import extract_video_timestamp

    with st.spinner("Reading video timestamps..."):
        for cam, tgt in [(cam_a, target_a), (cam_b, target_b)]:
            ts = extract_video_timestamp(str(tgt))
            if ts:
                parts = ts.split(" ")
                st.session_state[f"start_time_{cam}"] = parts[1]
                st.session_state[f"start_date_{cam}"] = parts[0]

    st.success(
        f"Camera A: {len(result_a['persons'])} people, "
        f"{len(result_a['vehicles'])} vehicles  \n"
        f"Camera B: {len(result_b['persons'])} people, "
        f"{len(result_b['vehicles'])} vehicles"
    )

    # Auto-run tamper detection on both videos
    from cctv_dissertation.tamper_detection import analyze_video as _tamper

    if "tamper_reports" not in st.session_state:
        st.session_state["tamper_reports"] = {}
    with st.spinner("Running tamper detection on both videos..."):
        for cam, tgt in [(cam_a, target_a), (cam_b, target_b)]:
            report = _tamper(str(tgt))
            st.session_state["tamper_reports"][cam] = report.to_dict()

    st.session_state["cross_cam_a"] = cam_a
    st.session_state["cross_cam_b"] = cam_b
    st.session_state["current_run_id"] = run_id
    st.session_state["upload_mode"] = "cross"
    st.rerun()


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Forensic Video Toolkit",
        layout="wide",
    )
    st.title("Forensic Video Analysis Dashboard")

    # ── Sidebar: upload mode ──
    with st.sidebar:
        st.header("Upload Mode")
        mode = st.radio(
            "Select mode",
            ["Single Camera", "Cross-Camera (2 videos)"],
            key="mode_radio",
            horizontal=False,
        )

        st.divider()

        if mode == "Single Camera":
            st.subheader("Upload Video")
            uploaded = st.file_uploader(
                "Select video file",
                type=["mp4", "avi", "mov", "mkv"],
                key="single_upload",
            )
            if st.button(
                "Process video",
                disabled=uploaded is None,
                use_container_width=True,
                type="primary",
            ):
                if uploaded:
                    handle_single_upload(uploaded)

        else:
            st.subheader("Upload Two Videos")
            file_a = st.file_uploader(
                "Camera A video",
                type=["mp4", "avi", "mov", "mkv"],
                key="cross_upload_a",
            )
            file_b = st.file_uploader(
                "Camera B video",
                type=["mp4", "avi", "mov", "mkv"],
                key="cross_upload_b",
            )
            if st.button(
                "Process both videos",
                disabled=(file_a is None or file_b is None),
                use_container_width=True,
                type="primary",
            ):
                if file_a and file_b:
                    handle_cross_upload(file_a, file_b)

        # Show existing cameras from DB
        st.divider()
        cameras = _query_cameras(TRACKER_DB)
        if cameras:
            st.subheader("Processed Cameras")
            for cam in cameras:
                entities = _query_tracked_entities(TRACKER_DB, cam)
                persons = sum(1 for e in entities if e["entity_type"] == "person")
                vehicles = sum(1 for e in entities if e["entity_type"] == "vehicle")
                st.caption(f"{cam}: {persons}P / {vehicles}V")

            st.subheader("Quick Switch")
            if len(cameras) >= 2:
                sel_a = st.selectbox("Camera A", cameras, index=0, key="qs_cam_a")
                sel_b = st.selectbox(
                    "Camera B", cameras, index=min(1, len(cameras) - 1), key="qs_cam_b"
                )
                if st.button(
                    "Cross-camera view",
                    use_container_width=True,
                    key="switch_cross",
                ):
                    st.session_state["cross_cam_a"] = sel_a
                    st.session_state["cross_cam_b"] = sel_b
                    st.session_state["upload_mode"] = "cross"
                    st.rerun()
            for cam in cameras:
                if st.button(
                    f"Single: {cam}",
                    use_container_width=True,
                    key=f"switch_{cam}",
                ):
                    st.session_state["active_camera"] = cam
                    st.session_state["upload_mode"] = "single"
                    st.rerun()

        # Legacy detection report selector
        st.divider()
        detection_files = list_detection_files()
        if detection_files:
            st.subheader("Detection Reports")
            selected_det = st.selectbox(
                "Report",
                detection_files,
                format_func=lambda p: p.name,
                key="det_report_select",
            )
        else:
            selected_det = None

    # ── Main content area ──
    upload_mode = st.session_state.get("upload_mode", "")

    if upload_mode == "single":
        camera = st.session_state.get("active_camera", "")
        if camera:
            (
                tab_track,
                tab_summary,
                tab_search,
                tab_poi,
                tab_tamper,
                tab_detect,
            ) = st.tabs(
                [
                    "Tracking",
                    "Summary",
                    "Search",
                    "Person of Interest",
                    "Tamper Detection",
                    "Detections",
                ]
            )
            with tab_track:
                page_single_tracking(camera)
            with tab_summary:
                page_single_summary(camera)
            with tab_search:
                page_search([camera])
            with tab_poi:
                page_person_of_interest([camera])
            with tab_tamper:
                page_tamper_detection([camera])
            with tab_detect:
                page_detections(selected_det, str(DEFAULT_DB_PATH))
        else:
            st.info("Upload a video to get started.")

    elif upload_mode == "cross":
        cam_a = st.session_state.get("cross_cam_a", "")
        cam_b = st.session_state.get("cross_cam_b", "")
        if cam_a and cam_b:
            (
                tab_dual,
                tab_id,
                tab_timeline,
                tab_search,
                tab_poi,
                tab_tamper,
                tab_export,
                tab_sum_a,
                tab_sum_b,
            ) = st.tabs(
                [
                    "Dual View",
                    "Identity Deep Dive",
                    "Timeline",
                    "Search",
                    "Person of Interest",
                    "Tamper Detection",
                    "Evidence Export",
                    f"Summary: {cam_a}",
                    f"Summary: {cam_b}",
                ]
            )
            with tab_dual:
                page_cross_dual_view(cam_a, cam_b)
            with tab_id:
                page_cross_identity(cam_a, cam_b)
            with tab_timeline:
                page_cross_timeline(cam_a, cam_b)
            with tab_search:
                page_search([cam_a, cam_b])
            with tab_poi:
                page_person_of_interest([cam_a, cam_b])
            with tab_tamper:
                page_tamper_detection([cam_a, cam_b])
            with tab_export:
                page_evidence_export(cam_a, cam_b)
            with tab_sum_a:
                page_single_summary(cam_a)
            with tab_sum_b:
                page_single_summary(cam_b)
        else:
            st.info("Upload two videos to enable cross-camera analysis.")
    else:
        # Default view — show existing data or prompt to upload
        _rid = st.session_state.get("current_run_id")
        cameras = _query_cameras(TRACKER_DB, run_id=_rid)
        if not cameras:
            # Fallback: load most recent run_id from DB
            if TRACKER_DB.exists():
                _conn = sqlite3.connect(str(TRACKER_DB))
                _row = _conn.execute(
                    "SELECT run_id FROM tracked_entities "
                    "WHERE run_id IS NOT NULL "
                    "ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                _conn.close()
                if _row:
                    st.session_state["current_run_id"] = _row[0]
                    cameras = _query_cameras(TRACKER_DB, run_id=_row[0])
            if not cameras:
                cameras = _query_cameras(TRACKER_DB)
        if cameras:
            if len(cameras) >= 2:
                # Auto-load cross-camera view
                st.session_state["cross_cam_a"] = cameras[0]
                st.session_state["cross_cam_b"] = cameras[1]
                st.session_state["upload_mode"] = "cross"
                st.rerun()
            elif len(cameras) == 1:
                st.session_state["active_camera"] = cameras[0]
                st.session_state["upload_mode"] = "single"
                st.rerun()
        elif detection_files:
            tab_detect = st.tabs(["Detections"])[0]
            with tab_detect:
                page_detections(selected_det, str(DEFAULT_DB_PATH))
        else:
            st.info("Upload a video using the sidebar to begin analysis.")


if __name__ == "__main__":
    main()
