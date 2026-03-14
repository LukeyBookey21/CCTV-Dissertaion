"""
Standalone script to process the merged DVR videos through the cross-camera pipeline.
Mirrors the logic in streamlit_app.py run_cross_camera_processing().
"""

import sys
import sqlite3
from pathlib import Path
import time

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
TRACKER_DB = PROJECT_ROOT / "data" / "tracker.db"
TRACKED_OUTPUT = PROJECT_ROOT / "data" / "tracked_output"

VIDEO_A = UPLOADS_DIR / "20260217_garage_merged.mp4"
VIDEO_B = UPLOADS_DIR / "20260217_garden_merged.mp4"

CAM_A = VIDEO_A.stem  # "20260217_garage_merged"
CAM_B = VIDEO_B.stem  # "20260217_garden_merged"

# ── imports ────────────────────────────────────────────────────────────────
from cctv_dissertation.ingest import ingest_video  # noqa: E402
from cctv_dissertation.tracker import (  # noqa: E402
    SingleCameraTracker,
    match_across_cameras,
)


# ── helpers ────────────────────────────────────────────────────────────────
def make_progress_callback(label: str):
    start = time.time()

    def cb(info):
        pct = info["frame"] / info["total"] * 100 if info["total"] > 0 else 0
        eta = info.get("eta_seconds", 0)
        elapsed = time.time() - start
        print(
            f"\r[{label}] {pct:5.1f}%  frame {info['frame']:,}/{info['total']:,}"
            f"  persons={info['persons']}  vehicles={info['vehicles']}"
            f"  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min",
            end="",
            flush=True,
        )

    return cb


def clear_camera(cam_label: str):
    if not TRACKER_DB.exists():
        return
    conn = sqlite3.connect(str(TRACKER_DB))
    conn.execute(
        "DELETE FROM track_frames WHERE entity_id IN "
        "(SELECT id FROM tracked_entities WHERE camera_label = ?)",
        (cam_label,),
    )
    conn.execute("DELETE FROM tracked_entities WHERE camera_label = ?", (cam_label,))
    conn.commit()
    conn.close()
    print(f"[setup] Cleared existing data for '{cam_label}'")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    for v, label in [(VIDEO_A, CAM_A), (VIDEO_B, CAM_B)]:
        if not v.exists():
            print(f"ERROR: {v} not found — aborting.")
            sys.exit(1)
        print(f"[setup] Found {v.name}  ({v.stat().st_size / 1e9:.2f} GB)")

    # ── ingest ─────────────────────────────────────────────────────────────
    print("\n[ingest] Ingesting videos for chain-of-custody...")
    ingest_video(VIDEO_A)
    ingest_video(VIDEO_B)
    print("[ingest] Done.")

    # ── clear old tracking data ────────────────────────────────────────────
    clear_camera(CAM_A)
    clear_camera(CAM_B)

    # ── stride ────────────────────────────────────────────────────────────
    # Force stride=20 (1 effective fps at 20fps source) — 4x faster than
    # the auto-calculated stride=5 with no meaningful accuracy loss for
    # security camera footage where events last several seconds.
    stride_a = 15
    stride_b = 15
    print(
        f"\n[stride] Camera A stride={stride_a},"
        f" Camera B stride={stride_b} (1.3 effective fps)"
    )

    # ── process Camera A ──────────────────────────────────────────────────
    print(f"\n[tracking] Processing Camera A: {CAM_A}")
    (TRACKED_OUTPUT / CAM_A).mkdir(parents=True, exist_ok=True)
    tracker_a = SingleCameraTracker(
        plate_model_path=str(PROJECT_ROOT / "models" / "license_plate_detector.pt"),
        db_path=str(TRACKER_DB),
    )
    t0 = time.time()
    result_a = tracker_a.process_video(
        video_path=str(VIDEO_A),
        output_dir=str(TRACKED_OUTPUT / CAM_A),
        camera_label=CAM_A,
        progress_callback=make_progress_callback("garage"),
        frame_stride=stride_a,
    )
    print(f"\n[tracking] Camera A done in {(time.time()-t0)/60:.1f} min — {result_a}")

    # ── process Camera B ──────────────────────────────────────────────────
    print(f"\n[tracking] Processing Camera B: {CAM_B}")
    (TRACKED_OUTPUT / CAM_B).mkdir(parents=True, exist_ok=True)
    tracker_b = SingleCameraTracker(
        plate_model_path=str(PROJECT_ROOT / "models" / "license_plate_detector.pt"),
        db_path=str(TRACKER_DB),
    )
    t0 = time.time()
    result_b = tracker_b.process_video(
        video_path=str(VIDEO_B),
        output_dir=str(TRACKED_OUTPUT / CAM_B),
        camera_label=CAM_B,
        progress_callback=make_progress_callback("garden"),
        frame_stride=stride_b,
    )
    print(f"\n[tracking] Camera B done in {(time.time()-t0)/60:.1f} min — {result_b}")

    # ── cross-camera matching ─────────────────────────────────────────────
    print("\n[cross-camera] Running cross-camera matching...")
    t0 = time.time()
    match_across_cameras(
        db_path=str(TRACKER_DB),
        camera_a=CAM_A,
        camera_b=CAM_B,
    )
    print(f"[cross-camera] Done in {(time.time()-t0)/60:.1f} min")

    print("\n=== ALL DONE ===")
    print(f"Camera A: {CAM_A}")
    print(f"Camera B: {CAM_B}")
    print("Open the Streamlit UI → Cross-Camera mode to review results.")


if __name__ == "__main__":
    main()
