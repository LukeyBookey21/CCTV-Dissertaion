"""Minimal video I/O helpers (skeleton).

This file provides simple wrappers that can be extended later. We use
`opencv-python-headless` for frame reading in server/dev environments.
"""

import cv2


def open_video(path: str):
    """Open a video capture and return the cv2.VideoCapture object."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def release_video(cap):
    cap.release()
