"""Simple IoU-based tracking over detection reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_tracks(
    report_path: str | Path,
    iou_threshold: float = 0.3,
    max_frame_gap: int = 2,
) -> Dict[str, Any]:
    """Assign track IDs to detections in a report and return track summaries."""
    report = json.loads(Path(report_path).read_text())
    frames = sorted(
        report.get("detections", []),
        key=lambda f: f.get("frame_index") or 0,
    )
    tracks: List[Dict[str, Any]] = []
    active: List[Dict[str, Any]] = []
    next_id = 1

    for frame in frames:
        frame_index = frame.get("frame_index")
        timestamp = frame.get("timestamp_seconds")
        detections = frame.get("detections", [])

        matched_track_ids = set()
        for det in detections:
            cls = det.get("class_name") or str(det.get("class_id"))
            bbox = det.get("bbox_xyxy")
            best_track = None
            best_iou = 0.0
            for track in active:
                if track["class_name"] != cls:
                    continue
                gap = (
                    0
                    if track["last_frame"] is None or frame_index is None
                    else frame_index - track["last_frame"]
                )
                if gap is not None and gap > max_frame_gap:
                    continue
                iou = bbox_iou(track["last_bbox"], bbox)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track = track
            if best_track:
                best_track["detections"].append(
                    _track_point(frame_index, timestamp, det)
                )
                best_track["last_frame"] = frame_index
                best_track["last_time"] = timestamp
                best_track["last_bbox"] = bbox
                matched_track_ids.add(best_track["track_id"])
            else:
                track_entry = {
                    "track_id": next_id,
                    "class_name": cls,
                    "detections": [
                        _track_point(frame_index, timestamp, det),
                    ],
                    "last_frame": frame_index,
                    "last_time": timestamp,
                    "last_bbox": bbox,
                }
                active.append(track_entry)
                tracks.append(track_entry)
                matched_track_ids.add(next_id)
                next_id += 1

        # retire tracks that have not been matched recently
        active = [
            track
            for track in active
            if track["last_frame"] is None
            or frame_index is None
            or frame_index - track["last_frame"] <= max_frame_gap
        ]

    summarized_tracks = [
        {
            "track_id": track["track_id"],
            "class_name": track["class_name"],
            "start_frame": track["detections"][0]["frame_index"],
            "end_frame": track["detections"][-1]["frame_index"],
            "start_time": track["detections"][0]["timestamp_seconds"],
            "end_time": track["detections"][-1]["timestamp_seconds"],
            "detections": track["detections"],
        }
        for track in tracks
    ]

    return {
        "sha256": report.get("sha256"),
        "tracks": summarized_tracks,
    }


def bbox_iou(bbox_a: Optional[List[float]], bbox_b: Optional[List[float]]) -> float:
    if not bbox_a or not bbox_b:
        return 0.0
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _track_point(frame_index, timestamp, det):
    return {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp,
        "confidence": det.get("confidence"),
        "bbox_xyxy": det.get("bbox_xyxy"),
    }


__all__ = [
    "generate_tracks",
    "bbox_iou",
]
