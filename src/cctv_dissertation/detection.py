"""YOLOv8 detection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ultralytics import YOLO

from .ingest import extract_video_metadata
from .utils.hashing import sha256_hash_file

DEFAULT_DETECTIONS_DIR = Path("data/detections")


def run_yolo_detection(
    video_path: str | Path,
    model_path: str = "yolov8n.pt",
    conf: float = 0.3,
    frame_stride: int = 5,
    device: str = "cpu",
    imgsz: int = 640,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Run YOLOv8 on `video_path` and return structured detections."""
    source = Path(video_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    metadata = extract_video_metadata(source)
    fps = metadata.get("frame_rate") or None

    model = YOLO(model_path)
    detections: List[Dict[str, Any]] = []
    for idx, result in enumerate(
        model.predict(
            source=str(source),
            stream=True,
            conf=conf,
            device=device,
            imgsz=imgsz,
            vid_stride=max(frame_stride, 1),
        )
    ):
        frame_index = _infer_frame_index(result, idx, frame_stride)
        frame_entry = {
            "frame_index": frame_index,
            "timestamp_seconds": (
                frame_index / fps if fps and frame_index is not None else None
            ),
            "detections": _format_boxes(result),
        }
        detections.append(frame_entry)
        if max_frames is not None and len(detections) >= max_frames:
            break

    report = {
        "source_path": str(source),
        "sha256": sha256_hash_file(str(source)),
        "model_path": model_path,
        "confidence_threshold": conf,
        "frame_stride": frame_stride,
        "device": device,
        "imgsz": imgsz,
        "metadata": metadata,
        "detections": detections,
    }
    return report


def write_detection_report(
    report: Dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Persist detection `report` as JSON and return the path."""
    output_dir = DEFAULT_DETECTIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    target = (
        Path(output_path).expanduser().resolve()
        if output_path
        else output_dir / f"{report['sha256']}.json"
    )
    target.write_text(json.dumps(report, indent=2))
    return target


def _format_boxes(result: Any) -> List[Dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    names: Dict[int, str] = getattr(result, "names", {})
    if boxes is None or len(boxes) == 0:
        return []

    cls_values = _to_list(getattr(boxes, "cls", []))
    conf_values = _to_list(getattr(boxes, "conf", []))
    xyxy_values = _to_list(getattr(boxes, "xyxy", []))

    formatted = []
    for cls_id, conf, bbox in zip(cls_values, conf_values, xyxy_values):
        int_cls = int(cls_id)
        formatted.append(
            {
                "class_id": int_cls,
                "class_name": names.get(int_cls, str(int_cls)),
                "confidence": float(conf),
                "bbox_xyxy": [float(v) for v in bbox],
            }
        )
    return formatted


def _infer_frame_index(result: Any, idx: int, frame_stride: int) -> Optional[int]:
    frame = getattr(result, "frame", None)
    if frame is not None:
        try:
            return int(frame)
        except (TypeError, ValueError):
            return None
    return idx * max(frame_stride, 1)


def _to_list(data: Any) -> List[Any]:
    if hasattr(data, "tolist"):
        return list(data.tolist())
    if isinstance(data, Iterable):
        return list(data)
    return []


__all__ = [
    "DEFAULT_DETECTIONS_DIR",
    "run_yolo_detection",
    "write_detection_report",
]
