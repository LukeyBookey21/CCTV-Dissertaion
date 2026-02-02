#!/usr/bin/env python3
"""Two-stage license plate detection: vehicles first, then plates within vehicles."""

import sys
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import easyocr

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: easyocr not available, OCR will be skipped")

# COCO vehicle class IDs
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def detect_plates_twostage(
    video_path: str,
    plate_model_path: str,
    output_dir: str,
    frame_stride: int = 3,
    vehicle_conf: float = 0.3,
    plate_conf: float = 0.2,
):
    """
    Two-stage plate detection:
    1. Detect vehicles using YOLOv8n (COCO pretrained)
    2. For each vehicle, detect plates within that region

    Args:
        video_path: Path to video file
        plate_model_path: Path to license plate model
        output_dir: Directory for output images
        frame_stride: Process every Nth frame
        vehicle_conf: Confidence threshold for vehicle detection
        plate_conf: Confidence threshold for plate detection
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vehicle detector (YOLOv8n)...")
    vehicle_model = YOLO("yolov8n.pt")

    print(f"Loading plate detector from {plate_model_path}...")
    plate_model = YOLO(plate_model_path)

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames at {fps:.1f} fps")

    # Collect all plate detections
    all_detections = []
    frames_cache = {}
    frame_idx = 0

    print(f"\nProcessing frames (stride={frame_stride})...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            h, w = frame.shape[:2]

            # Stage 1: Detect vehicles
            vehicle_results = vehicle_model.predict(
                frame,
                conf=vehicle_conf,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
            )

            vehicles_found = 0
            for result in vehicle_results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in VEHICLE_CLASSES:
                        continue

                    vx1, vy1, vx2, vy2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    vehicles_found += 1

                    # Add padding around vehicle
                    pad = 20
                    vx1 = max(0, vx1 - pad)
                    vy1 = max(0, vy1 - pad)
                    vx2 = min(w, vx2 + pad)
                    vy2 = min(h, vy2 + pad)

                    # Crop vehicle region
                    vehicle_crop = frame[vy1:vy2, vx1:vx2]

                    if vehicle_crop.size == 0:
                        continue

                    # Stage 2: Detect plates within vehicle
                    plate_results = plate_model.predict(
                        vehicle_crop,
                        conf=plate_conf,
                        imgsz=640,
                        verbose=False,
                    )

                    for plate_result in plate_results:
                        for plate_box in plate_result.boxes:
                            # Plate coordinates relative to vehicle crop
                            px1, py1, px2, py2 = plate_box.xyxy[0].cpu().numpy()
                            plate_conf_score = float(plate_box.conf[0])

                            # Convert to full frame coordinates
                            abs_x1 = vx1 + px1
                            abs_y1 = vy1 + py1
                            abs_x2 = vx1 + px2
                            abs_y2 = vy1 + py2

                            plate_width = abs_x2 - abs_x1
                            plate_height = abs_y2 - abs_y1
                            aspect_ratio = (
                                plate_width / plate_height if plate_height > 0 else 0
                            )

                            # Filter by aspect ratio (UK plates ~4.4:1)
                            if (
                                plate_width >= 20
                                and plate_height >= 5
                                and aspect_ratio >= 1.5
                                and aspect_ratio <= 8.0
                            ):

                                detection = {
                                    "frame_idx": frame_idx,
                                    "bbox": [abs_x1, abs_y1, abs_x2, abs_y2],
                                    "vehicle_bbox": [vx1, vy1, vx2, vy2],
                                    "confidence": plate_conf_score,
                                    "vehicle_type": VEHICLE_CLASSES[cls_id],
                                    "center_x": (abs_x1 + abs_x2) / 2,
                                    "center_y": (abs_y1 + abs_y2) / 2,
                                }
                                all_detections.append(detection)

                                if frame_idx not in frames_cache:
                                    frames_cache[frame_idx] = frame.copy()

            if frame_idx % 50 == 0:
                print(
                    f"  Frame {frame_idx}/{total_frames}, "
                    f"vehicles: {vehicles_found}, "
                    f"plates: {len(all_detections)}"
                )

        frame_idx += 1

    cap.release()
    print(f"\nTotal plate detections: {len(all_detections)}")

    # Debug: show detection positions
    if all_detections:
        print("\nDetection positions (center x,y):")
        positions = set()
        for d in all_detections:
            # Round to nearest 50 pixels to see clusters
            pos = (int(d["center_x"] / 50) * 50, int(d["center_y"] / 50) * 50)
            positions.add(pos)
        for pos in sorted(positions):
            count = sum(
                1
                for d in all_detections
                if abs(d["center_x"] - pos[0]) < 50 and abs(d["center_y"] - pos[1]) < 50
            )
            print(f"  Position ~({pos[0]}, {pos[1]}): {count} detections")

    if not all_detections:
        print("No plates detected! Try lowering confidence thresholds.")
        return

    # Cluster detections to find unique plates
    print("\nClustering to find unique plates...")
    unique_plates = cluster_detections(all_detections, frames_cache, min_distance=50)
    print(f"Found {len(unique_plates)} unique plates")

    # Initialize OCR
    ocr_reader = None
    if OCR_AVAILABLE:
        print("\nInitializing EasyOCR...")
        ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # Generate output images
    print("\nGenerating output images...")
    ocr_results = []

    for i, plate_info in enumerate(unique_plates, 1):
        frame = plate_info["frame"]
        bbox = plate_info["bbox"]
        conf = plate_info["confidence"]
        vehicle_type = plate_info.get("vehicle_type", "vehicle")

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # 1. Full frame with detection box
        frame_with_box = frame.copy()
        cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Plate {i} on {vehicle_type} ({conf:.2f})"
        cv2.putText(
            frame_with_box,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        full_path = output_dir / f"plate_{i}_full.jpg"
        cv2.imwrite(str(full_path), frame_with_box)
        print(f"  Saved: {full_path.name}")

        # 2. Cropped plate with padding and upscaling
        pad = 15
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(w, x2 + pad)
        crop_y2 = min(h, y2 + pad)

        plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Upscale for visibility
        scale = max(1, 150 // max(plate_crop.shape[0], 1))
        if scale > 1:
            plate_crop = cv2.resize(
                plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )

        crop_path = output_dir / f"plate_{i}_crop.jpg"
        cv2.imwrite(str(crop_path), plate_crop)
        print(f"  Saved: {crop_path.name}")

        # 3. OCR
        if ocr_reader:
            try:
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
                print(f"  OCR error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Output: {output_dir}")
    print(
        f"Images: {len(unique_plates) * 2} "
        f"({len(unique_plates)} full + "
        f"{len(unique_plates)} crop)"
    )

    if ocr_results:
        print("\nOCR Results:")
        for r in ocr_results:
            print(f"  Plate {r['plate']}: {r['text']} (conf: {r['confidence']:.2f})")
    else:
        print("\nNo readable text (plates may be too low resolution)")

    return unique_plates, ocr_results


def cluster_detections(detections, frames_cache, min_distance=100):
    """Cluster detections by spatial proximity."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    clusters = []
    used = set()

    for det in sorted_dets:
        if id(det) in used:
            continue

        cluster = [det]
        used.add(id(det))

        for other in sorted_dets:
            if id(other) in used:
                continue

            dist = np.sqrt(
                (det["center_x"] - other["center_x"]) ** 2
                + (det["center_y"] - other["center_y"]) ** 2
            )

            if dist < min_distance:
                cluster.append(other)
                used.add(id(other))

        clusters.append(cluster)

    unique_plates = []
    for cluster in clusters:
        best = max(cluster, key=lambda x: x["confidence"])
        if best["frame_idx"] in frames_cache:
            unique_plates.append(
                {
                    "bbox": best["bbox"],
                    "confidence": best["confidence"],
                    "frame_idx": best["frame_idx"],
                    "frame": frames_cache[best["frame_idx"]],
                    "vehicle_type": best.get("vehicle_type", "vehicle"),
                    "num_detections": len(cluster),
                }
            )

    unique_plates.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return unique_plates


if __name__ == "__main__":
    VIDEO_PATH = (
        "/workspaces/CCTV-Dissertaion/data/uploads/"
        "video_ec4d66401c0eedaf248a6c612533b3d1.mp4"
    )
    PLATE_MODEL = "/workspaces/CCTV-Dissertaion/models/license_plate_detector.pt"
    OUTPUT_DIR = "/workspaces/CCTV-Dissertaion/data/debug_plates/ec4d_twostage"

    detect_plates_twostage(
        video_path=VIDEO_PATH,
        plate_model_path=PLATE_MODEL,
        output_dir=OUTPUT_DIR,
        frame_stride=2,  # Every 2nd frame for better coverage
        vehicle_conf=0.25,  # Lower threshold to catch more vehicles
        plate_conf=0.15,  # Lower threshold for plates within vehicles
    )
