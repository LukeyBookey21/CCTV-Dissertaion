"""Utilities to summarize detection reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def summarize_detection_report(path: str | Path) -> Dict[str, Any]:
    """Load a detection JSON report and compute aggregate statistics."""
    report_path = Path(path).expanduser().resolve()
    data = json.loads(report_path.read_text())
    frames = data.get("detections", [])

    total_detections = 0
    class_stats: Dict[str, Dict[str, Optional[float] | int]] = {}
    earliest_ts: Optional[float] = None
    latest_ts: Optional[float] = None
    frames_with_detections = 0

    for frame in frames:
        timestamp = frame.get("timestamp_seconds")
        if timestamp is not None:
            earliest_ts = (
                timestamp
                if earliest_ts is None
                else min(earliest_ts, timestamp)
            )
            latest_ts = (
                timestamp
                if latest_ts is None
                else max(latest_ts, timestamp)
            )
        detections = frame.get("detections", [])
        if detections:
            frames_with_detections += 1
        frame_index = frame.get("frame_index")
        for det in detections:
            total_detections += 1
            class_name = det.get("class_name") or str(det.get("class_id"))
            entry = class_stats.setdefault(
                class_name,
                {
                    "count": 0,
                    "first_seen_frame": None,
                    "last_seen_frame": None,
                    "first_seen_time": None,
                    "last_seen_time": None,
                },
            )
            entry["count"] += 1
            if entry["first_seen_frame"] is None or (
                isinstance(entry["first_seen_frame"], int)
                and frame_index is not None
                and frame_index < entry["first_seen_frame"]
            ):
                entry["first_seen_frame"] = frame_index
                entry["first_seen_time"] = timestamp
            if entry["last_seen_frame"] is None or (
                isinstance(entry["last_seen_frame"], int)
                and frame_index is not None
                and frame_index > entry["last_seen_frame"]
            ):
                entry["last_seen_frame"] = frame_index
                entry["last_seen_time"] = timestamp

    summary = {
        "source_path": data.get("source_path"),
        "sha256": data.get("sha256"),
        "model_path": data.get("model_path"),
        "frames_in_report": len(frames),
        "frames_with_detections": frames_with_detections,
        "detections_total": total_detections,
        "class_stats": class_stats,
        "time_bounds": {
            "start_seconds": earliest_ts,
            "end_seconds": latest_ts,
        },
    }
    return summary


def format_summary(summary: Dict[str, Any]) -> str:
    """Return a readable multi-line summary."""
    lines = [
        f"Source: {summary.get('source_path')}",
        f"SHA-256: {summary.get('sha256')}",
        f"Model: {summary.get('model_path')}",
        f"Frames analyzed: {summary.get('frames_in_report')} "
        f"(with detections: {summary.get('frames_with_detections')})",
        f"Total detections: {summary.get('detections_total')}",
    ]
    start = summary.get("time_bounds", {}).get("start_seconds")
    end = summary.get("time_bounds", {}).get("end_seconds")
    if start is not None and end is not None:
        lines.append(f"Time span covered: {start:.2f}s – {end:.2f}s")

    lines.append("Per-class stats:")
    class_stats = summary.get("class_stats", {})
    if not class_stats:
        lines.append("  (no detections)")
    else:
        for class_name, stats in sorted(class_stats.items()):
            count = stats.get("count")
            first_time = stats.get("first_seen_time")
            last_time = stats.get("last_seen_time")
            lines.append(
                f"  - {class_name}: {count} detections"
                + (
                    f" (first @ {first_time:.2f}s, last @ {last_time:.2f}s)"
                    if first_time is not None and last_time is not None
                    else ""
                )
            )
    return "\n".join(lines)


__all__ = [
    "summarize_detection_report",
    "format_summary",
]
