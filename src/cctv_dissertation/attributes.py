"""Color-based attribute extraction for people and vehicles."""

import cv2
import numpy as np
from typing import Dict, List, Tuple

# Named color palette — HSV ranges mapped to human-readable names
# Format: (name, lower_hsv, upper_hsv)
COLOR_RANGES_HSV: List[Tuple[str, np.ndarray, np.ndarray]] = [
    ("red", np.array([0, 70, 50]), np.array([10, 255, 255])),
    ("red", np.array([170, 70, 50]), np.array([180, 255, 255])),
    ("orange", np.array([11, 70, 50]), np.array([25, 255, 255])),
    ("yellow", np.array([26, 70, 50]), np.array([34, 255, 255])),
    ("green", np.array([35, 70, 50]), np.array([85, 255, 255])),
    ("blue", np.array([86, 70, 50]), np.array([130, 255, 255])),
    ("purple", np.array([131, 70, 50]), np.array([160, 255, 255])),
    ("pink", np.array([161, 30, 50]), np.array([169, 255, 255])),
]

# Achromatic colors detected via saturation/value thresholds
ACHROMATIC_NAMES = ["black", "dark grey", "grey", "white"]


def _dominant_color_kmeans(image: np.ndarray, k: int = 3) -> np.ndarray:
    """Return the dominant BGR color of an image using k-means."""
    pixels = image.reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        return (
            pixels.mean(axis=0).astype(np.uint8)
            if len(pixels)
            else np.array([0, 0, 0], dtype=np.uint8)
        )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # Pick the cluster with most pixels
    counts = np.bincount(labels.flatten(), minlength=k)
    dominant_idx = counts.argmax()
    return centers[dominant_idx].astype(np.uint8)


def bgr_to_color_name(bgr: np.ndarray) -> str:
    """Map a single BGR color to the closest human-readable name."""
    hsv = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Achromatic detection
    if v < 40:
        return "black"
    if v < 90 and s < 50:
        return "dark grey"
    if s < 40 and v < 180:
        return "grey"
    if s < 30 and v >= 180:
        return "white"

    # Chromatic — match HSV ranges
    hsv_pixel = np.array([h, s, v])
    for name, lo, hi in COLOR_RANGES_HSV:
        if np.all(hsv_pixel >= lo) and np.all(hsv_pixel <= hi):
            return name

    # Fallback: olive / khaki region
    if 25 <= h <= 45 and 30 <= s < 70 and 50 <= v <= 180:
        return "olive"

    return "unknown"


def extract_region_color(image: np.ndarray, k: int = 3) -> str:
    """Extract the dominant color name from an image region."""
    if image.size == 0:
        return "unknown"
    dominant_bgr = _dominant_color_kmeans(image, k=k)
    return bgr_to_color_name(dominant_bgr)


# ── Person attributes ──────────────────────────────────────────────


def describe_person(crop: np.ndarray) -> Dict[str, str]:
    """
    Describe a person crop by splitting into upper and lower body
    and extracting dominant colors.

    Args:
        crop: BGR image of a person (full body preferred)

    Returns:
        Dict with keys: upper_color, lower_color, description
    """
    if crop.size == 0:
        return {
            "upper_color": "unknown",
            "lower_color": "unknown",
            "description": "unknown",
        }

    h, w = crop.shape[:2]

    # Split roughly at 45% from top (upper body) and 45-90% (lower body)
    # Avoids head and feet which add noise
    upper = crop[int(h * 0.15) : int(h * 0.45), :]
    lower = crop[int(h * 0.45) : int(h * 0.85), :]

    upper_color = extract_region_color(upper)
    lower_color = extract_region_color(lower)

    description = f"{upper_color} top, {lower_color} bottom"
    return {
        "upper_color": upper_color,
        "lower_color": lower_color,
        "description": description,
    }


# ── Vehicle attributes ─────────────────────────────────────────────


def describe_vehicle(crop: np.ndarray, vehicle_type: str = "car") -> Dict[str, str]:
    """
    Describe a vehicle crop by extracting its dominant body color.

    Args:
        crop: BGR image of a vehicle
        vehicle_type: COCO class name (car, truck, bus, motorcycle)

    Returns:
        Dict with keys: color, vehicle_type, description
    """
    if crop.size == 0:
        return {
            "color": "unknown",
            "vehicle_type": vehicle_type,
            "description": f"unknown {vehicle_type}",
        }

    h, w = crop.shape[:2]

    # Focus on the centre of the vehicle body to avoid road / background
    body = crop[int(h * 0.15) : int(h * 0.75), int(w * 0.1) : int(w * 0.9)]

    color = extract_region_color(body, k=3)
    description = f"{color} {vehicle_type}"

    return {
        "color": color,
        "vehicle_type": vehicle_type,
        "description": description,
    }
