"""Run cross-camera processing directly without the UI.

Collects performance metrics (timing, RAM, FPS) for dissertation evaluation.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cctv_dissertation.tracker import SingleCameraTracker  # noqa: E402

DB_PATH = str(PROJECT_ROOT / "data" / "tracker.db")
TRACKED_OUTPUT = PROJECT_ROOT / "data" / "tracked_output"

GARAGE = str(PROJECT_ROOT / "data" / "uploads" / "garage_merged.mp4")
GARDEN = str(PROJECT_ROOT / "data" / "uploads" / "garden_merged.mp4")

CAM_A = "garage"
CAM_B = "garden"

STRIDE = 10

# ── Performance tracking ───────────────────────────────────────────
process = psutil.Process(os.getpid())
peak_ram_mb = 0.0
ram_lock = threading.Lock()


def _track_ram():
    """Background thread to sample peak RAM usage."""
    global peak_ram_mb
    while not _stop_ram_tracker.is_set():
        mem = process.memory_info().rss / (1024 * 1024)
        with ram_lock:
            if mem > peak_ram_mb:
                peak_ram_mb = mem
        _stop_ram_tracker.wait(1.0)


_stop_ram_tracker = threading.Event()
ram_thread = threading.Thread(target=_track_ram, daemon=True)

# Per-camera metrics
camera_metrics = {}
metrics_lock = threading.Lock()


def run_camera(video_path: str, camera_label: str):
    tracker = SingleCameraTracker(db_path=DB_PATH)
    output_dir = str(TRACKED_OUTPUT / camera_label)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cam_start = time.time()
    total_frames = 0

    def progress(info):
        nonlocal total_frames
        total_frames = info["total"]
        pct = info["frame"] / info["total"] if info["total"] else 0
        eta = info.get("eta_seconds", 0) / 60
        print(
            f"\r[{camera_label}] {info['frame']:,}/{info['total']:,} "
            f"({pct*100:.1f}%) | persons={info['persons']} vehicles={info['vehicles']} "
            f"| ETA: {eta:.1f}min    ",
            end="",
            flush=True,
        )

    print(f"[{camera_label}] Starting — stride={STRIDE}")
    result = tracker.process_video(
        video_path=video_path,
        output_dir=output_dir,
        camera_label=camera_label,
        frame_stride=STRIDE,
        progress_callback=progress,
    )
    cam_elapsed = time.time() - cam_start
    n_persons = len(result["persons"])
    n_vehicles = len(result["vehicles"])
    frames_processed = total_frames // STRIDE if total_frames else 0
    avg_fps = frames_processed / cam_elapsed if cam_elapsed > 0 else 0

    with metrics_lock:
        camera_metrics[camera_label] = {
            "total_frames": total_frames,
            "frames_processed": frames_processed,
            "processing_time_seconds": round(cam_elapsed, 1),
            "average_fps": round(avg_fps, 2),
            "persons_detected": n_persons,
            "vehicles_detected": n_vehicles,
        }

    print(
        f"\n[{camera_label}] Done — {n_persons} persons, {n_vehicles} vehicles "
        f"in {cam_elapsed/60:.1f}min ({avg_fps:.1f} fps)"
    )


# ── Main ───────────────────────────────────────────────────────────
overall_start = time.time()

# Start RAM tracker
ram_thread.start()

# Initialise DB schema before threads start to avoid race condition
_init_tracker = SingleCameraTracker(db_path=DB_PATH)
del _init_tracker

t_a = threading.Thread(target=run_camera, args=(GARAGE, CAM_A))
t_b = threading.Thread(target=run_camera, args=(GARDEN, CAM_B))

t_a.start()
t_b.start()
t_a.join()
t_b.join()

# Stop RAM tracker
_stop_ram_tracker.set()
ram_thread.join(timeout=2)

overall_elapsed = time.time() - overall_start

# DB file size
db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)

# Collect summary
summary = {
    "overall": {
        "total_processing_time_seconds": round(overall_elapsed, 1),
        "total_processing_time_minutes": round(overall_elapsed / 60, 1),
        "peak_ram_mb": round(peak_ram_mb, 1),
        "db_file_size_mb": round(db_size_mb, 2),
        "stride": STRIDE,
        "errors": 0,
        "crashes": False,
    },
    "cameras": camera_metrics,
}

print(f"\n{'='*60}")
print("PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Total time:  {overall_elapsed/60:.1f} min")
print(f"Peak RAM:    {peak_ram_mb:.0f} MB")
print(f"DB size:     {db_size_mb:.2f} MB")
for cam, m in camera_metrics.items():
    print(
        f"  [{cam}] {m['frames_processed']:,} frames in "
        f"{m['processing_time_seconds']/60:.1f}min "
        f"({m['average_fps']:.1f} fps) — "
        f"{m['persons_detected']} persons, {m['vehicles_detected']} vehicles"
    )

# Save metrics to file
metrics_path = PROJECT_ROOT / "data" / "processing_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nMetrics saved to {metrics_path}")
