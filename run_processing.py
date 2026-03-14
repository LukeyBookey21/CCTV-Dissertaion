"""Run cross-camera processing directly without the UI."""

import sys
import threading
from pathlib import Path

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


def run_camera(video_path: str, camera_label: str):
    tracker = SingleCameraTracker(db_path=DB_PATH)
    output_dir = str(TRACKED_OUTPUT / camera_label)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def progress(info):
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
    n_persons = len(result["persons"])
    n_vehicles = len(result["vehicles"])
    print(f"\n[{camera_label}] Done — {n_persons} persons, {n_vehicles} vehicles")


t_a = threading.Thread(target=run_camera, args=(GARAGE, CAM_A))
t_b = threading.Thread(target=run_camera, args=(GARDEN, CAM_B))

t_a.start()
t_b.start()
t_a.join()
t_b.join()

print("\nBoth cameras done. Loading UI to view results...")
