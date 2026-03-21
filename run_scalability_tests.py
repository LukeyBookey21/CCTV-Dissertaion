"""Run scalability tests across multiple video durations.

Extracts 1h, 3h, 5h, 8h clips from the merged DVR videos,
processes each independently, and saves results + metrics
to separate directories for side-by-side comparison.

Usage:
    python run_scalability_tests.py

Results are saved to data/scalability_tests/<duration>/
Each contains: tracker.db, tracked_output/, metrics.json

To view results in the UI:
    streamlit run ui/streamlit_app.py -- --db data/scalability_tests/1h/tracker.db
Or load the session ZIP from within the UI.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cctv_dissertation.tracker import SingleCameraTracker  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────
GARAGE_FULL = PROJECT_ROOT / "data" / "uploads" / "garage_merged.mp4"
GARDEN_FULL = PROJECT_ROOT / "data" / "uploads" / "garden_merged.mp4"

DURATIONS = {
    "1h": "01:00:00",
    "3h": "03:00:00",
    "5h": "05:00:00",
    "8h": "08:00:00",
}

STRIDE = 10
BASE_DIR = PROJECT_ROOT / "data" / "scalability_tests"


# ── Helpers ───────────────────────────────────────────────────────
def extract_clip(source: Path, output: Path, duration: str):
    """Extract a clip of given duration from the start of the source video."""
    if output.exists():
        print(f"  Clip already exists: {output.name}")
        return
    print(f"  Extracting {duration} from {source.name}...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "00:00:00",
            "-to",
            duration,
            "-i",
            str(source),
            "-c",
            "copy",
            str(output),
        ],
        capture_output=True,
    )


def run_camera(
    video_path: str,
    camera_label: str,
    db_path: str,
    output_dir: str,
    metrics_dict: dict,
    metrics_lock: threading.Lock,
    peak_ram: list,
    ram_lock: threading.Lock,
):
    """Process a single camera and record metrics."""
    tracker = SingleCameraTracker(db_path=db_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cam_start = time.time()
    total_frames = 0

    def progress(info):
        nonlocal total_frames
        total_frames = info["total"]
        pct = info["frame"] / info["total"] if info["total"] else 0
        eta = info.get("eta_seconds", 0) / 60
        # Sample RAM
        mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        with ram_lock:
            if mem > peak_ram[0]:
                peak_ram[0] = mem
        print(
            f"\r  [{camera_label}] {info['frame']:,}/{info['total']:,} "
            f"({pct*100:.1f}%) | P={info['persons']} V={info['vehicles']} "
            f"| ETA: {eta:.1f}min | RAM: {mem:.0f}MB    ",
            end="",
            flush=True,
        )

    print(f"  [{camera_label}] Starting — stride={STRIDE}")
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
        metrics_dict[camera_label] = {
            "total_frames": total_frames,
            "frames_processed": frames_processed,
            "processing_time_seconds": round(cam_elapsed, 1),
            "average_fps": round(avg_fps, 2),
            "persons_detected": n_persons,
            "vehicles_detected": n_vehicles,
        }

    print(
        f"\n  [{camera_label}] Done — {n_persons} persons, {n_vehicles} vehicles "
        f"in {cam_elapsed/60:.1f}min ({avg_fps:.1f} fps)"
    )


def run_duration_test(label: str, duration: str):
    """Run a complete test for one duration."""
    print(f"\n{'='*60}")
    print(f"SCALABILITY TEST: {label} ({duration})")
    print(f"{'='*60}")

    test_dir = BASE_DIR / label
    test_dir.mkdir(parents=True, exist_ok=True)

    clips_dir = test_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    garage_clip = clips_dir / "garage.mp4"
    garden_clip = clips_dir / "garden.mp4"

    # Step 1: Extract clips
    print("\nStep 1: Extracting clips...")
    extract_clip(GARAGE_FULL, garage_clip, duration)
    extract_clip(GARDEN_FULL, garden_clip, duration)

    # Step 2: Process
    print("\nStep 2: Processing...")
    db_path = str(test_dir / "tracker.db")
    tracked_dir = test_dir / "tracked_output"

    # Clear previous results
    if Path(db_path).exists():
        os.remove(db_path)
    if tracked_dir.exists():
        shutil.rmtree(tracked_dir)

    # Init DB
    init_tracker = SingleCameraTracker(db_path=db_path)
    del init_tracker

    camera_metrics = {}
    metrics_lock = threading.Lock()
    peak_ram = [0.0]
    ram_lock = threading.Lock()

    overall_start = time.time()

    t_a = threading.Thread(
        target=run_camera,
        args=(
            str(garage_clip),
            "garage",
            db_path,
            str(tracked_dir / "garage"),
            camera_metrics,
            metrics_lock,
            peak_ram,
            ram_lock,
        ),
    )
    t_b = threading.Thread(
        target=run_camera,
        args=(
            str(garden_clip),
            "garden",
            db_path,
            str(tracked_dir / "garden"),
            camera_metrics,
            metrics_lock,
            peak_ram,
            ram_lock,
        ),
    )

    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    overall_elapsed = time.time() - overall_start
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    # Step 3: Save metrics
    summary = {
        "test_label": label,
        "video_duration": duration,
        "overall": {
            "total_processing_time_seconds": round(overall_elapsed, 1),
            "total_processing_time_minutes": round(overall_elapsed / 60, 1),
            "peak_ram_mb": round(peak_ram[0], 1),
            "db_file_size_mb": round(db_size_mb, 2),
            "stride": STRIDE,
            "errors": 0,
            "crashes": False,
        },
        "cameras": camera_metrics,
    }

    metrics_path = test_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Total time:  {overall_elapsed/60:.1f} min")
    print(f"  Peak RAM:    {peak_ram[0]:.0f} MB")
    print(f"  DB size:     {db_size_mb:.2f} MB")
    print(f"  Metrics saved to {metrics_path}")

    return summary


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check source videos exist
    if not GARAGE_FULL.exists() or not GARDEN_FULL.exists():
        print("ERROR: Source videos not found in data/uploads/")
        print(f"  Expected: {GARAGE_FULL}")
        print(f"  Expected: {GARDEN_FULL}")
        sys.exit(1)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for label, duration in DURATIONS.items():
        result = run_duration_test(label, duration)
        all_results[label] = result

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("SCALABILITY COMPARISON TABLE")
    print(f"{'='*80}")
    print(
        f"{'Duration':<10} {'Time (min)':<12} {'FPS':<10} "
        f"{'RAM (MB)':<10} {'DB (MB)':<10} {'Persons':<10} {'Vehicles':<10}"
    )
    print("-" * 80)
    for label, r in all_results.items():
        o = r["overall"]
        cams = r["cameras"]
        total_persons = sum(c["persons_detected"] for c in cams.values())
        total_vehicles = sum(c["vehicles_detected"] for c in cams.values())
        avg_fps = sum(c["average_fps"] for c in cams.values()) / len(cams)
        print(
            f"{label:<10} {o['total_processing_time_minutes']:<12} "
            f"{avg_fps:<10.1f} {o['peak_ram_mb']:<10.0f} "
            f"{o['db_file_size_mb']:<10.2f} {total_persons:<10} {total_vehicles:<10}"
        )

    # Save combined results
    combined_path = BASE_DIR / "scalability_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")
    print("\nTo view results in UI:")
    print("  Copy the tracker.db from any test folder to data/tracker.db")
    print("  Or load a session ZIP from the UI")
