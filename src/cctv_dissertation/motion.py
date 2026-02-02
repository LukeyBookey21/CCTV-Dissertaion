"""Motion detection utilities for filtering forensic footage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np

# Motion detection presets optimized for forensic analysis
MOTION_PRESETS = {
    "fast": {
        "frame_stride": 30,
        "motion_threshold": 0.03,
        "min_area": 800,
        "history": 300,
        "var_threshold": 20,
        "description": "Fast screening - ~3x faster, good for initial review",
    },
    "balanced": {
        "frame_stride": 10,
        "motion_threshold": 0.02,
        "min_area": 500,
        "history": 500,
        "var_threshold": 16,
        "description": "Balanced - good speed/accuracy trade-off (recommended)",
    },
    "accurate": {
        "frame_stride": 5,
        "motion_threshold": 0.015,
        "min_area": 300,
        "history": 800,
        "var_threshold": 12,
        "description": "Accurate - detailed analysis, slower but catches subtle motion",
    },
}

MotionPreset = Literal["fast", "balanced", "accurate"]


def detect_motion_in_video(
    video_path: str | Path,
    frame_stride: int = 5,
    motion_threshold: float = 0.02,
    min_area: int = 500,
    preset: Optional[MotionPreset] = None,
    history: int = 500,
    var_threshold: int = 16,
) -> List[Dict[str, Any]]:
    """
    Detect motion in video frames using background subtraction.

    Args:
        video_path: Path to video file
        frame_stride: Analyze every Nth frame (matches detection pipeline)
        motion_threshold: Percentage of frame that must change (0.0-1.0)
        min_area: Minimum contour area to consider as motion (pixels)
        preset: Use preset configuration ("fast", "balanced", or "accurate")
                Overrides other parameters if specified
        history: MOG2 history length for background modeling
        var_threshold: MOG2 variance threshold for foreground detection

    Returns:
        List of dicts with frame_index, has_motion, motion_percentage, motion_area
    """
    # Apply preset if specified
    if preset:
        if preset not in MOTION_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset}. Choose from: {list(MOTION_PRESETS.keys())}"
            )
        config = MOTION_PRESETS[preset]
        frame_stride = config["frame_stride"]
        motion_threshold = config["motion_threshold"]
        min_area = config["min_area"]
        history = config["history"]
        var_threshold = config["var_threshold"]

    source = Path(video_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    # Use MOG2 background subtractor (works well for CCTV)
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=True
    )

    results = []
    frame_idx = 0
    processed_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames matching the stride
        if frame_idx % frame_stride == 0:
            # Apply background subtraction
            fg_mask = back_sub.apply(frame)

            # Remove shadows (MOG2 marks shadows as 127)
            fg_mask[fg_mask == 127] = 0

            # Morphological operations to reduce noise
            kernel = np.ones((5, 5), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours of moving objects
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Calculate motion metrics
            total_area = frame.shape[0] * frame.shape[1]
            motion_area = sum(
                cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= min_area
            )
            motion_percentage = motion_area / total_area

            has_motion = motion_percentage >= motion_threshold

            results.append(
                {
                    "frame_index": frame_idx,
                    "has_motion": has_motion,
                    "motion_percentage": float(motion_percentage),
                    "motion_area_pixels": int(motion_area),
                    "num_moving_objects": len(
                        [c for c in contours if cv2.contourArea(c) >= min_area]
                    ),
                }
            )

            processed_idx += 1

        frame_idx += 1

    cap.release()
    return results


def add_motion_to_detection_report(
    detection_report: Dict[str, Any],
    motion_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge motion detection results into an existing detection report.

    Args:
        detection_report: Original YOLO detection report
        motion_results: Motion detection results from detect_motion_in_video()

    Returns:
        Updated detection report with motion flags added to each frame
    """
    # Create lookup by frame_index
    motion_lookup = {m["frame_index"]: m for m in motion_results}

    # Add motion data to each detection frame
    for detection in detection_report.get("detections", []):
        frame_idx = detection.get("frame_index")
        motion_data = motion_lookup.get(frame_idx, {})

        detection["motion"] = {
            "has_motion": motion_data.get("has_motion", False),
            "motion_percentage": motion_data.get("motion_percentage", 0.0),
            "motion_area_pixels": motion_data.get("motion_area_pixels", 0),
            "num_moving_objects": motion_data.get("num_moving_objects", 0),
        }

    return detection_report


__all__ = [
    "detect_motion_in_video",
    "add_motion_to_detection_report",
    "MOTION_PRESETS",
    "MotionPreset",
]
