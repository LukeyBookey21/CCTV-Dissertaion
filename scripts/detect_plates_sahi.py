#!/usr/bin/env python3
"""SAHI-based license plate detection for small/distant plates in video."""

import sys
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import easyocr

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: easyocr not available, OCR will be skipped")


def detect_plates_sahi(
    video_path: str,
    model_path: str,
    output_dir: str,
    frame_stride: int = 5,
    slice_size: int = 512,
    overlap_ratio: float = 0.3,
    conf_threshold: float = 0.15,  # Lower threshold to catch more plates
):
    """
    Detect license plates using SAHI sliced inference.

    Args:
        video_path: Path to video file
        model_path: Path to YOLOv8 plate detection model
        output_dir: Directory for output images
        frame_stride: Process every Nth frame
        slice_size: Size of inference slices (smaller = better for tiny objects)
        overlap_ratio: Overlap between slices
        conf_threshold: Confidence threshold (lower catches more)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device="cpu",  # Use CPU for stability
    )

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames at {fps:.1f} fps")

    # Collect all detections with frame info
    all_detections = []
    frame_idx = 0
    frames_cache = {}  # Cache frames for later retrieval

    print(f"\nProcessing frames (stride={frame_stride})...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            # Run SAHI sliced prediction
            result = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                verbose=0,
            )

            # Store detections
            for obj in result.object_prediction_list:
                bbox = obj.bbox.to_xyxy()
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0

                # Plate filtering - UK plates are ~4.4:1 aspect ratio
                # Stricter filtering to reduce false positives
                if (
                    width >= 30
                    and height >= 8
                    and aspect_ratio >= 2.0
                    and aspect_ratio <= 7.0
                ):
                    detection = {
                        "frame_idx": frame_idx,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": obj.score.value,
                        "center_x": (x1 + x2) / 2,
                        "center_y": (y1 + y2) / 2,
                        "width": width,
                        "height": height,
                    }
                    all_detections.append(detection)

                    # Cache this frame
                    if frame_idx not in frames_cache:
                        frames_cache[frame_idx] = frame.copy()

            if frame_idx % 50 == 0:
                print(
                    f"  Frame {frame_idx}/{total_frames}, "
                    f"detections so far: {len(all_detections)}"
                )

        frame_idx += 1

    cap.release()
    print(f"\nTotal detections: {len(all_detections)}")

    if not all_detections:
        print("No plates detected!")
        return

    # Cluster detections by spatial proximity to find unique plates
    # Using simple grid-based clustering
    print("\nClustering detections to find unique plates...")
    unique_plates = cluster_detections(all_detections, frames_cache)
    print(f"Found {len(unique_plates)} unique plates")

    # Initialize OCR if available
    ocr_reader = None
    if OCR_AVAILABLE:
        print("\nInitializing EasyOCR...")
        ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # Generate output images for each unique plate
    print("\nGenerating output images...")
    ocr_results = []

    for i, plate_info in enumerate(unique_plates, 1):
        frame = plate_info["frame"]
        bbox = plate_info["bbox"]
        conf = plate_info["confidence"]
        frame_idx = plate_info["frame_idx"]

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # 1. Full frame with detection box
        frame_with_box = frame.copy()
        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame_with_box,
            f"Plate {i} ({conf:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        full_path = output_dir / f"plate_{i}_full.jpg"
        cv2.imwrite(str(full_path), frame_with_box)
        print(f"  Saved: {full_path.name}")

        # 2. Cropped and zoomed plate
        # Add padding around plate
        pad = 10
        h, w = frame.shape[:2]
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(w, x2 + pad)
        crop_y2 = min(h, y2 + pad)

        plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Upscale for better visibility
        scale = max(1, 200 // plate_crop.shape[0])  # Target ~200px height
        if scale > 1:
            plate_crop = cv2.resize(
                plate_crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        crop_path = output_dir / f"plate_{i}_crop.jpg"
        cv2.imwrite(str(crop_path), plate_crop)
        print(f"  Saved: {crop_path.name}")

        # 3. Run OCR on cropped plate
        if ocr_reader:
            try:
                # Preprocess for OCR
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                results = ocr_reader.readtext(gray)
                if results:
                    texts = []
                    for _, text, text_conf in results:
                        cleaned = "".join(c for c in text if c.isalnum()).upper()
                        if cleaned and text_conf > 0.3:
                            texts.append((cleaned, text_conf))

                    if texts:
                        best_text = " ".join(t[0] for t in texts)
                        avg_conf = np.mean([t[1] for t in texts])
                        ocr_results.append(
                            {
                                "plate": i,
                                "text": best_text,
                                "confidence": avg_conf,
                            }
                        )
                        print(f"  OCR Plate {i}: '{best_text}' (conf: {avg_conf:.2f})")
            except Exception as e:
                print(f"  OCR error for plate {i}: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Images generated: {len(unique_plates) * 2}")
    print(f"  - {len(unique_plates)} full frame images")
    print(f"  - {len(unique_plates)} cropped plate images")

    if ocr_results:
        print("\nOCR Results:")
        for r in ocr_results:
            print(
                f"  Plate {r['plate']}: {r['text']} (confidence: {r['confidence']:.2f})"
            )
    else:
        print("\nNo readable text detected (plates may be too low resolution)")

    return unique_plates, ocr_results


def cluster_detections(detections, frames_cache, min_distance=150):
    """
    Cluster detections to find unique plates.
    Uses the best detection (highest confidence) for each cluster.
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    clusters = []
    used = set()

    for det in sorted_dets:
        if id(det) in used:
            continue

        # Start new cluster with this detection
        cluster = [det]
        used.add(id(det))

        # Find all detections close to this one (across all frames)
        for other in sorted_dets:
            if id(other) in used:
                continue

            # Check spatial proximity
            dist = np.sqrt(
                (det["center_x"] - other["center_x"]) ** 2
                + (det["center_y"] - other["center_y"]) ** 2
            )

            if dist < min_distance:
                cluster.append(other)
                used.add(id(other))

        clusters.append(cluster)

    # For each cluster, get the best detection
    unique_plates = []
    for cluster in clusters:
        # Pick detection with highest confidence
        best = max(cluster, key=lambda x: x["confidence"])

        if best["frame_idx"] in frames_cache:
            unique_plates.append(
                {
                    "bbox": best["bbox"],
                    "confidence": best["confidence"],
                    "frame_idx": best["frame_idx"],
                    "frame": frames_cache[best["frame_idx"]],
                    "num_detections": len(cluster),
                }
            )

    # Sort by position in frame (left to right, top to bottom)
    unique_plates.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

    return unique_plates


if __name__ == "__main__":
    # Video and model paths
    VIDEO_PATH = (
        "/workspaces/CCTV-Dissertaion/data/uploads/"
        "video_ec4d66401c0eedaf248a6c612533b3d1.mp4"
    )
    MODEL_PATH = "/workspaces/CCTV-Dissertaion/models/license_plate_detector.pt"
    OUTPUT_DIR = "/workspaces/CCTV-Dissertaion/data/debug_plates/ec4d_sahi"

    detect_plates_sahi(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        frame_stride=3,  # Process every 3rd frame for better coverage
        slice_size=512,  # 512x512 slices
        overlap_ratio=0.3,  # 30% overlap
        conf_threshold=0.25,  # Higher threshold to reduce false positives
    )
