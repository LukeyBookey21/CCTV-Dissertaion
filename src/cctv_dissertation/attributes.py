"""Color-based attribute extraction for people and vehicles."""

import cv2
import numpy as np
from typing import Dict, List, Tuple

# Named color palette — HSV ranges mapped to human-readable names
# Format: (name, lower_hsv, upper_hsv)
COLOR_RANGES_HSV: List[Tuple[str, np.ndarray, np.ndarray]] = [
    ("red", np.array([0, 70, 40]), np.array([10, 255, 255])),
    ("red", np.array([170, 70, 40]), np.array([180, 255, 255])),
    ("orange", np.array([11, 70, 50]), np.array([25, 255, 255])),
    ("yellow", np.array([26, 70, 50]), np.array([34, 255, 255])),
    ("green", np.array([35, 60, 40]), np.array([85, 255, 255])),
    # Blue extended: lower sat/val thresholds to catch dark navy & CCTV-compressed blues
    ("blue", np.array([86, 35, 25]), np.array([130, 255, 255])),
    ("purple", np.array([131, 50, 40]), np.array([160, 255, 255])),
    ("pink", np.array([161, 30, 50]), np.array([169, 255, 255])),
]

# Achromatic colors detected via saturation/value thresholds
ACHROMATIC_NAMES = ["black", "dark grey", "grey", "white"]


def _kmeans_clusters(image: np.ndarray, k: int = 5) -> List[Dict]:
    """
    Run k-means on image pixels and return clusters sorted by size.

    Each cluster dict has: bgr, count, fraction, hsv
    """
    pixels = image.reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        mean = (
            pixels.mean(axis=0).astype(np.uint8)
            if len(pixels)
            else np.array([0, 0, 0], dtype=np.uint8)
        )
        return [{"bgr": mean, "count": len(pixels), "fraction": 1.0}]

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0,
    )
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    total = len(labels)
    counts = np.bincount(labels.flatten(), minlength=k)
    clusters = []
    for i in range(k):
        bgr = centers[i].astype(np.uint8)
        hsv = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        clusters.append(
            {
                "bgr": bgr,
                "count": int(counts[i]),
                "fraction": counts[i] / total,
                "hsv": hsv,
            }
        )
    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters


def bgr_to_color_name(bgr: np.ndarray) -> str:
    """Map a single BGR color to the closest human-readable name.

    Order of checks:
      1. Black (very dark)
      2. Achromatic (low saturation) — catches dark grey, grey, white
         BEFORE any chromatic range can claim them
      3. Earth tones (olive, brown, beige) — need s >= 50
      4. Chromatic HSV ranges (red, blue, green, etc.)
    """
    hsv = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # 1. Very dark = black
    if v < 55:
        return "black"

    # 2. Achromatic — check FIRST so desaturated pixels with hue noise
    #    (e.g. dark grey HSV 113,51,40 wrongly matching blue) get caught.
    #    s < 50 is the gate: true coloured fabrics have s > 80-100.
    if s < 50:
        if v < 100:
            return "dark grey"
        if v < 200:
            return "grey"
        return "white"

    # 2b. Grey with camera-induced colour cast — outdoor CCTV often
    #     adds a blue/teal tint to grey cars.  These are perceptually
    #     grey, not blue.  True blue paint has S > 150 at any V.
    if s < 140 and 85 <= h <= 130 and v < 140:
        return "grey"

    # 2c. Bright near-white with slight colour cast — e.g. white vehicle
    #     body reflecting sky.  True blue paint has S > 100 at high V.
    if v > 200 and s < 75:
        return "white"

    # 3. Earth tones (s >= 50 guaranteed at this point)
    if 20 <= h <= 65 and 40 <= v <= 200:
        return "green" if h >= 35 else "olive"

    if 8 <= h <= 22 and 40 <= v <= 160:
        return "brown"

    if 15 <= h <= 30 and s < 80 and 150 <= v <= 235:
        return "beige"

    # 4. Chromatic — match HSV ranges
    hsv_pixel = np.array([h, s, v])
    for name, lo, hi in COLOR_RANGES_HSV:
        if np.all(hsv_pixel >= lo) and np.all(hsv_pixel <= hi):
            return name

    return "unknown"


def _is_background_color(hsv: np.ndarray) -> bool:
    """Check if an HSV color is likely brick/pavement/road background."""
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    # Brick pavement: low-sat brownish-grey, hue 5-25, low sat
    if 5 <= h <= 25 and s < 80 and 60 <= v <= 170:
        return True
    # Grey road/concrete
    if s < 30 and 80 <= v <= 180:
        return True
    # White paving tiles / bright concrete
    if s < 20 and v >= 160:
        return True
    return False


def _pick_clothing_color(clusters: List[Dict]) -> np.ndarray:
    """
    Pick the most likely clothing color from k-means clusters.
    Prefers chromatic (colourful) clusters over achromatic ones,
    then skips background-looking clusters.
    Falls back to the largest cluster if all look like background.
    """
    viable = [
        c
        for c in clusters
        if c["fraction"] >= 0.05 and not _is_background_color(c["hsv"])
    ]
    if not viable:
        viable = [c for c in clusters if c["fraction"] >= 0.05]
    if not viable:
        return clusters[0]["bgr"]

    # Prefer chromatic clusters (high saturation) — catches blue, red, green clothing
    # before falling back to grey/dark achromatic clusters.
    # s >= 50 matches bgr_to_color_name's achromatic gate.
    chromatic = [c for c in viable if int(c["hsv"][1]) >= 50 and int(c["hsv"][2]) >= 45]
    if chromatic:
        # Among chromatic, pick highest saturation × coverage score
        return max(chromatic, key=lambda c: int(c["hsv"][1]) * c["fraction"])["bgr"]

    # No chromatic clusters — return largest non-background
    return viable[0]["bgr"]


def _pick_vehicle_color(clusters: List[Dict]) -> np.ndarray:
    """
    Pick the most likely vehicle body color from k-means clusters.

    Uses the largest cluster that isn't very dark (windows/shadows/
    tyres).  The vehicle body dominates the cropped image area, so
    the biggest non-dark cluster is almost always the paint colour.
    """
    # Filter out very dark clusters (windows, shadows, tyres)
    viable = [c for c in clusters if c["fraction"] >= 0.05 and int(c["hsv"][2]) >= 50]
    if not viable:
        viable = clusters
    # Return the largest remaining cluster — the car body
    return max(viable, key=lambda c: c["fraction"])["bgr"]


def extract_region_color(image: np.ndarray, k: int = 5) -> str:
    """Extract the dominant color name from an image region."""
    if image.size == 0:
        return "unknown"
    clusters = _kmeans_clusters(image, k=k)
    return bgr_to_color_name(clusters[0]["bgr"])


# ── Person attributes ──────────────────────────────────────────────


def _extract_hair_color(crop: np.ndarray) -> str:
    """Extract hair color from the top portion of a person crop.

    Looks at the top 15% (head region), centre 40% width strip
    to avoid background.  Picks the dominant non-skin, non-background
    cluster as hair.
    """
    h, w = crop.shape[:2]
    head = crop[0 : int(h * 0.15), int(w * 0.30) : int(w * 0.70)]
    if head.size < 30:
        return "unknown"
    clusters = _kmeans_clusters(head, k=3)
    for c in clusters:
        hsv = c["hsv"]
        hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])
        # Skip skin tones (orange-ish, low-med sat)
        if 5 <= hue <= 25 and 40 <= sat <= 170 and val > 100:
            continue
        # Skip background
        if _is_background_color(hsv):
            continue
        name = bgr_to_color_name(c["bgr"])
        # Map to hair-specific names
        if name in ("black", "dark grey"):
            return "dark"
        if name in ("brown", "olive"):
            return "brown"
        if name in ("orange", "red"):
            return "red"
        if name in ("yellow", "beige"):
            return "blonde"
        if name in ("grey", "white"):
            return "light"
        return name
    return "unknown"


def _estimate_build(crop: np.ndarray) -> str:
    """Estimate body build from crop aspect ratio.

    A tall narrow crop (ratio < 0.40) suggests slim build.
    A wider crop (ratio > 0.55) suggests stocky build.
    """
    h, w = crop.shape[:2]
    if h < 10:
        return "unknown"
    ratio = w / h
    if ratio < 0.40:
        return "slim"
    if ratio > 0.55:
        return "stocky"
    return "medium"


def describe_person(crop: np.ndarray) -> Dict[str, str]:
    """
    Describe a person crop by splitting into upper and lower body
    and extracting dominant clothing colors, plus hair color and build.

    Uses the centre strip of the crop to avoid side background,
    and skips clusters that look like brick/pavement.
    """
    if crop.size == 0:
        return {
            "upper_color": "unknown",
            "lower_color": "unknown",
            "hair_color": "unknown",
            "build": "unknown",
            "description": "unknown",
        }

    h, w = crop.shape[:2]

    # Use centre 50% width strip to reduce background pixels
    x_start = int(w * 0.25)
    x_end = int(w * 0.75)

    # Upper body: 15%-45% height, centre strip
    upper = crop[int(h * 0.15) : int(h * 0.45), x_start:x_end]
    # Lower body: 50%-80% height, centre strip
    lower = crop[int(h * 0.50) : int(h * 0.80), x_start:x_end]

    upper_clusters = _kmeans_clusters(upper, k=5)
    lower_clusters = _kmeans_clusters(lower, k=5)

    upper_bgr = _pick_clothing_color(upper_clusters)
    lower_bgr = _pick_clothing_color(lower_clusters)

    upper_color = bgr_to_color_name(upper_bgr)
    lower_color = bgr_to_color_name(lower_bgr)

    hair_color = _extract_hair_color(crop)
    build = _estimate_build(crop)

    description = f"{upper_color} top, {lower_color} bottom"
    return {
        "upper_color": upper_color,
        "lower_color": lower_color,
        "hair_color": hair_color,
        "build": build,
        "description": description,
    }


# ── Vehicle attributes ─────────────────────────────────────────────


def describe_vehicle(crop: np.ndarray, vehicle_type: str = "car") -> Dict[str, str]:
    """
    Describe a vehicle crop by extracting its dominant body color.

    Uses k-means clustering and picks the cluster most likely to be
    vehicle body paint (brightest/most saturated, not windows/shadow).
    """
    if crop.size == 0:
        return {
            "color": "unknown",
            "vehicle_type": vehicle_type,
            "description": f"unknown {vehicle_type}",
        }

    h, w = crop.shape[:2]

    # Use the centre band of the crop — avoids roof/sky at top,
    # road/ground at bottom, and edge background.
    body = crop[
        int(h * 0.25) : int(h * 0.75),
        int(w * 0.1) : int(w * 0.9),
    ]

    clusters = _kmeans_clusters(body, k=5)
    body_bgr = _pick_vehicle_color(clusters)
    color = bgr_to_color_name(body_bgr)
    description = f"{color} {vehicle_type}"

    return {
        "color": color,
        "vehicle_type": vehicle_type,
        "description": description,
    }
