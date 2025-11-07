from pathlib import Path

import cv2
import numpy as np

from cctv_dissertation.ingest import ingest_video, load_manifest


def _create_dummy_video(path: Path, frame_count: int = 5) -> None:
    height, width = 32, 32
    codecs = ("mp4v", "MJPG", "XVID")
    writer = None
    for codec in codecs:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            5.0,
            (width, height),
        )
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        raise RuntimeError("Unable to create video writer for test fixture")

    for i in range(frame_count):
        frame = np.full((height, width, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_ingest_video_writes_manifest(tmp_path):
    video_path = tmp_path / "sample.avi"
    manifest_path = tmp_path / "manifest.json"
    _create_dummy_video(video_path)

    entry = ingest_video(video_path, manifest_path)

    assert entry["source_path"] == str(video_path.resolve())
    assert len(entry["sha256"]) == 64
    metadata = entry["metadata"]
    assert metadata["width"] == 32
    assert metadata["height"] == 32
    assert metadata["frame_count"] is not None

    manifest_data = load_manifest(manifest_path)
    assert manifest_data[-1]["sha256"] == entry["sha256"]
