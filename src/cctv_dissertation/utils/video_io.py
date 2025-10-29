"""Minimal video I/O helpers inside package namespace."""

import cv2


def open_video(path: str):
    """Open a video capture and return the cv2.VideoCapture object."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap


def release_video(cap):
    cap.release()
