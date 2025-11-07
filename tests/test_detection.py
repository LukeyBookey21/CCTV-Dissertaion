from pathlib import Path

import cv2
import numpy as np
import pytest

import cctv_dissertation.detection as detection


def _create_dummy_video(path: Path, frame_count: int = 5) -> None:
    height, width = 32, 32
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create video writer for test fixture")

    for i in range(frame_count):
        writer.write(np.full((height, width, 3), i * 10, dtype=np.uint8))
    writer.release()


class FakeArray:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return list(self._data)


class FakeBoxes:
    def __init__(self):
        self.cls = FakeArray([0])
        self.conf = FakeArray([0.9])
        self.xyxy = FakeArray([[1.0, 2.0, 3.0, 4.0]])

    def __len__(self):
        return len(self.cls._data)


class FakeResult:
    def __init__(self, source):
        self.path = source
        self.frame = 10
        self.names = {0: "person"}
        self.boxes = FakeBoxes()


class FakeYOLO:
    def __init__(self, model_path):
        self.model_path = model_path

    def predict(self, **kwargs):
        yield FakeResult(kwargs["source"])


def test_run_yolo_detection_with_stub(monkeypatch, tmp_path):
    video_path = tmp_path / "det.avi"
    _create_dummy_video(video_path)

    monkeypatch.setattr(detection, "YOLO", FakeYOLO)

    report = detection.run_yolo_detection(
        video_path,
        model_path="fake.pt",
        frame_stride=2,
        conf=0.4,
        max_frames=1,
    )

    assert report["model_path"] == "fake.pt"
    assert report["detections"][0]["detections"][0]["class_name"] == "person"
    assert report["detections"][0]["frame_index"] == 10

    output_file = tmp_path / "report.json"
    written_path = detection.write_detection_report(report, output_file)
    assert written_path == output_file
    assert output_file.exists()
