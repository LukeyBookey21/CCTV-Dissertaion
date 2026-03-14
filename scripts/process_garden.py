"""Process garden camera and run cross-camera matching."""

import sys
import sqlite3
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cctv_dissertation.tracker import (  # noqa: E402
    SingleCameraTracker,
    match_across_cameras,
)
from cctv_dissertation.ingest import ingest_video  # noqa: E402

DB = "data/tracker.db"
VIDEO_B = Path("data/uploads/20260217_garden_merged.mp4")
CAM_A = "20260217_garage_merged"
CAM_B = "20260217_garden_merged"
OUTPUT = Path("data/tracked_output") / CAM_B

# Clear old (wrong) garden data
conn = sqlite3.connect(DB)
conn.execute(
    "DELETE FROM track_frames WHERE entity_id IN "
    "(SELECT id FROM tracked_entities WHERE camera_label = ?)",
    (CAM_B,),
)
conn.execute("DELETE FROM tracked_entities WHERE camera_label = ?", (CAM_B,))
conn.commit()
conn.close()
print(f"Cleared old data for {CAM_B}")

ingest_video(VIDEO_B)
print("Ingest done.")

OUTPUT.mkdir(parents=True, exist_ok=True)

tracker = SingleCameraTracker(
    plate_model_path="models/license_plate_detector.pt",
    db_path=DB,
)

start = time.time()


def progress(info):
    pct = info["frame"] / info["total"] * 100 if info["total"] > 0 else 0
    print(
        f"\r[garden] {pct:5.1f}%  frame {info['frame']:,}/{info['total']:,}"
        f"  persons={info['persons']}  vehicles={info['vehicles']}"
        f"  ETA={info['eta_seconds']/60:.1f}min",
        end="",
        flush=True,
    )


result = tracker.process_video(
    video_path=str(VIDEO_B),
    output_dir=str(OUTPUT),
    camera_label=CAM_B,
    progress_callback=progress,
    frame_stride=10,
)
print(f"\n[garden] Done in {(time.time()-start)/60:.1f} min — {result}")

# Cross-camera matching
print("\n[cross-camera] Matching...")
matches = match_across_cameras(db_path=DB, camera_a=CAM_A, camera_b=CAM_B)
pm = matches.get("person_matches", [])
vm = matches.get("vehicle_matches", [])
print(f"[cross-camera] Done — {len(pm)} person matches, {len(vm)} vehicle matches")
for m in pm:
    print(
        f"  sim={m['similarity']} | {m['entity_a']['description']}"
        f" <-> {m['entity_b']['description']}"
    )

print("\n=== ALL DONE ===")
