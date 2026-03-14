"""Run garage and garden cameras in parallel, then cross-camera match."""

import sys
import sqlite3
import time
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cctv_dissertation.ingest import ingest_video  # noqa: E402
from cctv_dissertation.tracker import (  # noqa: E402
    SingleCameraTracker,
    match_across_cameras,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB = str(PROJECT_ROOT / "data" / "tracker.db")
UPLOADS = PROJECT_ROOT / "data" / "uploads"
OUTPUT = PROJECT_ROOT / "data" / "tracked_output"

VIDEO_A = UPLOADS / "test_garage_short.mp4"
VIDEO_B = UPLOADS / "test_garden_short.mp4"
CAM_A = "test_garage_short"
CAM_B = "test_garden_short"
STRIDE = 10


def clear_camera(label):
    conn = sqlite3.connect(DB)
    conn.execute(
        "DELETE FROM track_frames WHERE entity_id IN "
        "(SELECT id FROM tracked_entities WHERE camera_label = ?)",
        (label,),
    )
    conn.execute("DELETE FROM tracked_entities WHERE camera_label = ?", (label,))
    conn.commit()
    conn.close()


RUN_ID = str(uuid.uuid4())[:12]


def run_camera(video, cam_label, log_prefix):
    output_dir = OUTPUT / cam_label
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = SingleCameraTracker(
        plate_model_path=str(PROJECT_ROOT / "models" / "license_plate_detector.pt"),
        db_path=DB,
    )

    start = time.time()

    def progress(info):
        pct = info["frame"] / info["total"] * 100 if info["total"] > 0 else 0
        print(
            f"\r[{log_prefix}] {pct:5.1f}%  "
            f"{info['frame']:,}/{info['total']:,}  "
            f"P={info['persons']} V={info['vehicles']}  "
            f"ETA={info['eta_seconds']/60:.1f}min   ",
            end="",
            flush=True,
        )

    result = tracker.process_video(
        video_path=str(video),
        output_dir=str(output_dir),
        camera_label=cam_label,
        progress_callback=progress,
        frame_stride=STRIDE,
        run_id=RUN_ID,
    )
    elapsed = (time.time() - start) / 60
    print(f"\n[{log_prefix}] Done in {elapsed:.1f} min | {result}")
    return result


# ── Setup ──────────────────────────────────────────────────────────────────
print("[setup] Ingesting videos...")
ingest_video(VIDEO_A)
ingest_video(VIDEO_B)
print("[setup] Clearing old data...")
clear_camera(CAM_A)
clear_camera(CAM_B)
fps = 20 / STRIDE
print(f"[setup] Stride={STRIDE} ({fps:.1f} fps) | Running both cameras in parallel\n")

# ── Parallel processing ────────────────────────────────────────────────────
results = {}
errors = {}


def run_a():
    try:
        results["a"] = run_camera(VIDEO_A, CAM_A, "garage")
    except Exception as e:
        errors["a"] = e
        print(f"\n[garage] ERROR: {e}")


def run_b():
    try:
        results["b"] = run_camera(VIDEO_B, CAM_B, "garden")
    except Exception as e:
        errors["b"] = e
        print(f"\n[garden] ERROR: {e}")


t_a = threading.Thread(target=run_a)
t_b = threading.Thread(target=run_b)

wall_start = time.time()
t_a.start()
t_b.start()
t_a.join()
t_b.join()
print(
    f"\n[parallel] Both cameras done in {(time.time()-wall_start)/60:.1f} min wall time"
)

if errors:
    print(f"ERRORS: {errors}")
    sys.exit(1)

# ── Cross-camera matching ──────────────────────────────────────────────────
print("\n[cross-camera] Matching...")
matches = match_across_cameras(
    db_path=DB, camera_a=CAM_A, camera_b=CAM_B, person_threshold=0.45
)
pm = matches.get("person_matches", [])
vm = matches.get("vehicle_matches", [])
print(f"[cross-camera] {len(pm)} person matches, {len(vm)} vehicle matches")
for m in pm:
    desc_a = m["entity_a"]["description"]
    desc_b = m["entity_b"]["description"]
    print(f"  sim={m['similarity']} | {desc_a} <-> {desc_b}")

print("\n=== ALL DONE ===")
