import json
from pathlib import Path

from cctv_dissertation.storage import (
    import_detection_report,
    query_detections,
    query_tracks,
    store_tracks,
)


def _fake_report(path: Path):
    data = {
        "source_path": "clip.mp4",
        "sha256": "hash123",
        "model_path": "yolo.pt",
        "confidence_threshold": 0.4,
        "frame_stride": 2,
        "metadata": {"frame_rate": 10},
        "detections": [
            {
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "detections": [
                    {
                        "class_id": 2,
                        "class_name": "car",
                        "confidence": 0.8,
                        "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                    }
                ],
            },
            {
                "frame_index": 2,
                "timestamp_seconds": 0.2,
                "detections": [
                    {
                        "class_id": 10,
                        "class_name": "traffic light",
                        "confidence": 0.9,
                        "bbox_xyxy": [0.1, 0.1, 1.0, 1.0],
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(data))


def test_import_and_query(tmp_path):
    report_path = tmp_path / "report.json"
    db_path = tmp_path / "detections.db"
    _fake_report(report_path)

    result = import_detection_report(report_path, db_path)

    assert result["sha256"] == "hash123"
    assert result["frames_imported"] == 2
    assert result["detections_imported"] == 2

    cars = query_detections("hash123", db_path=db_path, class_name="car", min_conf=0.5)
    assert len(cars) == 1
    assert cars[0]["frame_index"] == 0

    lights = query_detections(
        "hash123",
        db_path=db_path,
        class_name="traffic light",
        time_range=(0.15, None),
    )
    assert len(lights) == 1
    assert lights[0]["timestamp_seconds"] == 0.2

    tracks = [
        {
            "track_id": 1,
            "class_name": "car",
            "start_frame": 0,
            "end_frame": 2,
            "start_time": 0.0,
            "end_time": 0.2,
            "detections": [
                {"frame_index": 0, "timestamp_seconds": 0.0, "confidence": 0.8, "bbox_xyxy": [0, 0, 1, 1]},
                {"frame_index": 2, "timestamp_seconds": 0.2, "confidence": 0.7, "bbox_xyxy": [0, 0, 1, 1]},
            ],
        }
    ]
    store_tracks("hash123", tracks, db_path=db_path)
    stored_tracks = query_tracks("hash123", db_path=db_path)
    assert stored_tracks[0]["class_name"] == "car"
    assert stored_tracks[0]["points"][0]["frame_index"] == 0
