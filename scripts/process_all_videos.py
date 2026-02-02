#!/usr/bin/env python3
"""Process all uploaded videos for vehicle and plate detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detect_vehicles_plates import detect_vehicles_and_plates  # noqa: E402

UPLOADS_DIR = Path("/workspaces/CCTV-Dissertaion/data/uploads")
OUTPUT_BASE = Path("/workspaces/CCTV-Dissertaion/data/debug_plates")
PLATE_MODEL = "/workspaces/CCTV-Dissertaion/models/license_plate_detector.pt"

# Videos to process (user's uploads)
VIDEOS = [
    "video_15cbab23451e052e2208e6ca0dda962a.mp4",
    "video_26db9890ea017e5370be63d7be0f5ae2.mp4",
    "video_62f88a18b468b37395ec2b9bacba6db0.mp4",
    "video_c0075590b4d2949a08a6aa1c8fa810bf.mp4",
    # ec4d already processed
]


def main():
    print("=" * 60)
    print("BATCH VEHICLE/PLATE DETECTION")
    print("=" * 60)

    for video_name in VIDEOS:
        video_path = UPLOADS_DIR / video_name
        if not video_path.exists():
            print(f"\nSkipping {video_name} - not found")
            continue

        # Create output directory based on video hash
        video_id = video_name.replace("video_", "").replace(".mp4", "")[:8]
        output_dir = OUTPUT_BASE / f"{video_id}_results"

        print(f"\n{'=' * 60}")
        print(f"Processing: {video_name}")
        print(f"Output: {output_dir}")
        print("=" * 60)

        try:
            detect_vehicles_and_plates(
                video_path=str(video_path),
                plate_model_path=PLATE_MODEL,
                output_dir=str(output_dir),
                frame_stride=3,
                vehicle_conf=0.25,
                plate_conf=0.15,
            )
        except Exception as e:
            print(f"ERROR processing {video_name}: {e}")

    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
