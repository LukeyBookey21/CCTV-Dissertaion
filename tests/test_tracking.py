import json
from pathlib import Path

from cctv_dissertation.tracking import generate_tracks, bbox_iou


def _fake_report(path: Path):
    data = {
        "sha256": "abc",
        "detections": [
            {
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "detections": [
                    {
                        "class_id": 2,
                        "class_name": "car",
                        "confidence": 0.8,
                        "bbox_xyxy": [0, 0, 1, 1],
                    }
                ],
            },
            {
                "frame_index": 1,
                "timestamp_seconds": 0.1,
                "detections": [
                    {
                        "class_id": 2,
                        "class_name": "car",
                        "confidence": 0.85,
                        "bbox_xyxy": [0, 0, 1.1, 1.1],
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(data))


def test_generate_tracks(tmp_path):
    report_path = tmp_path / "report.json"
    _fake_report(report_path)
    result = generate_tracks(report_path, iou_threshold=0.1)
    assert result["sha256"] == "abc"
    assert len(result["tracks"]) == 1
    track = result["tracks"][0]
    assert track["class_name"] == "car"
    assert track["start_frame"] == 0
    assert len(track["detections"]) == 2


def test_bbox_iou():
    assert bbox_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert bbox_iou([0, 0, 1, 1], [1, 1, 2, 2]) == 0.0
