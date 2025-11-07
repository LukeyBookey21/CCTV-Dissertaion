import json
from pathlib import Path

from cctv_dissertation.analysis import (
    format_summary,
    summarize_detection_report,
)


def _fake_detection_report(path: Path) -> None:
    data = {
        "source_path": "example.mp4",
        "sha256": "abc",
        "model_path": "yolov8n.pt",
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
                "frame_index": 5,
                "timestamp_seconds": 0.5,
                "detections": [
                    {
                        "class_id": 10,
                        "class_name": "traffic light",
                        "confidence": 0.7,
                        "bbox_xyxy": [0, 0, 1, 1],
                    },
                    {
                        "class_id": 2,
                        "class_name": "car",
                        "confidence": 0.9,
                        "bbox_xyxy": [0, 0, 1, 1],
                    },
                ],
            },
        ],
    }
    path.write_text(json.dumps(data))


def test_summarize_detection_report(tmp_path):
    report_path = tmp_path / "report.json"
    _fake_detection_report(report_path)

    summary = summarize_detection_report(report_path)

    assert summary["detections_total"] == 3
    assert summary["frames_with_detections"] == 2
    assert summary["class_stats"]["car"]["count"] == 2
    assert summary["class_stats"]["traffic light"]["first_seen_time"] == 0.5

    rendered = format_summary(summary)
    assert "car" in rendered
    assert "traffic light" in rendered
