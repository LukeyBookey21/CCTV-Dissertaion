#!/usr/bin/env python3
"""Vehicle detection with optional plate detection - forensic analysis tool."""

from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

try:
    import easyocr

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# COCO vehicle class IDs
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def detect_vehicles_and_plates(
    video_path: str,
    plate_model_path: str,
    output_dir: str,
    frame_stride: int = 5,
    vehicle_conf: float = 0.3,
    plate_conf: float = 0.15,
):
    """
    Detect vehicles and attempt plate detection.
    Falls back to vehicle crop if no plate found.

    For each unique vehicle:
    - Full frame with bounding box
    - Cropped region (plate if detected, otherwise vehicle)
    - OCR text if plate is readable
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

    # Track unique vehicles by position
    vehicle_detections = []  # All vehicle detections
    frames_cache = {}
    frame_idx = 0

    print(f"\nProcessing frames (stride={frame_stride})...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            h, w = frame.shape[:2]

            # Detect vehicles
            results = vehicle_model.predict(
                frame,
                conf=vehicle_conf,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
            )

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in VEHICLE_CLASSES:
                        continue

                    vx1, vy1, vx2, vy2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    vehicle_conf_score = float(box.conf[0])

                    # Try to detect plate within vehicle region
                    pad = 10
                    crop_x1 = max(0, vx1 - pad)
                    crop_y1 = max(0, vy1 - pad)
                    crop_x2 = min(w, vx2 + pad)
                    crop_y2 = min(h, vy2 + pad)
                    vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    plate_bbox = None
                    plate_conf_score = 0

                    if vehicle_crop.size > 0:
                        plate_results = plate_model.predict(
                            vehicle_crop, conf=plate_conf, imgsz=640, verbose=False
                        )
                        for pr in plate_results:
                            for pb in pr.boxes:
                                px1, py1, px2, py2 = pb.xyxy[0].cpu().numpy()
                                pc = float(pb.conf[0])
                                pw = px2 - px1
                                ph = py2 - py1
                                ar = pw / ph if ph > 0 else 0

                                # Filter valid plates
                                if (
                                    pw >= 15
                                    and ph >= 5
                                    and 1.5 <= ar <= 8.0
                                    and pc > plate_conf_score
                                ):
                                    plate_bbox = [
                                        crop_x1 + px1,
                                        crop_y1 + py1,
                                        crop_x1 + px2,
                                        crop_y1 + py2,
                                    ]
                                    plate_conf_score = pc

                    detection = {
                        "frame_idx": frame_idx,
                        "vehicle_bbox": [vx1, vy1, vx2, vy2],
                        "vehicle_conf": vehicle_conf_score,
                        "vehicle_type": VEHICLE_CLASSES[cls_id],
                        "plate_bbox": plate_bbox,
                        "plate_conf": plate_conf_score,
                        "center_x": (vx1 + vx2) / 2,
                        "center_y": (vy1 + vy2) / 2,
                    }
                    vehicle_detections.append(detection)

                    if frame_idx not in frames_cache:
                        frames_cache[frame_idx] = frame.copy()

            if frame_idx % 50 == 0:
                print(
                    f"  Frame {frame_idx}/{total_frames}, "
                    f"vehicles found: {len(vehicle_detections)}"
                )

        frame_idx += 1

    cap.release()
    print(f"\nTotal vehicle detections: {len(vehicle_detections)}")

    if not vehicle_detections:
        print("No vehicles detected!")
        return

    # Cluster to find unique vehicles
    print("\nFinding unique vehicles...")
    unique_vehicles = cluster_vehicles(
        vehicle_detections, frames_cache, min_distance=80
    )
    print(f"Found {len(unique_vehicles)} unique vehicles")

    # Initialize OCR
    ocr_reader = None
    if OCR_AVAILABLE:
        print("\nInitializing EasyOCR...")
        ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # Generate output
    print("\nGenerating output images...")
    results_summary = []

    for i, vehicle in enumerate(unique_vehicles, 1):
        frame = vehicle["frame"]
        vbbox = vehicle["vehicle_bbox"]
        pbbox = vehicle["plate_bbox"]
        vtype = vehicle["vehicle_type"]
        h, w = frame.shape[:2]

        vx1, vy1, vx2, vy2 = [int(v) for v in vbbox]
        has_plate = pbbox is not None

        # 1. Full frame with detection boxes
        frame_annotated = frame.copy()

        # Draw vehicle box (blue)
        cv2.rectangle(frame_annotated, (vx1, vy1), (vx2, vy2), (255, 100, 0), 2)

        # Draw plate box if found (green)
        if has_plate:
            px1, py1, px2, py2 = [int(v) for v in pbbox]
            cv2.rectangle(frame_annotated, (px1, py1), (px2, py2), (0, 255, 0), 3)
            label = f"Vehicle {i} ({vtype}) - PLATE DETECTED"
        else:
            label = f"Vehicle {i} ({vtype}) - no plate"

        cv2.putText(
            frame_annotated,
            label,
            (vx1, vy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        full_path = output_dir / f"vehicle_{i}_full.jpg"
        cv2.imwrite(str(full_path), frame_annotated)
        print(f"  Saved: {full_path.name}")

        # 2. Cropped image - plate if available, otherwise vehicle
        if has_plate:
            px1, py1, px2, py2 = [int(v) for v in pbbox]
            pad = 15
            cx1, cy1 = max(0, px1 - pad), max(0, py1 - pad)
            cx2, cy2 = min(w, px2 + pad), min(h, py2 + pad)
            crop_label = "plate"
        else:
            pad = 20
            cx1, cy1 = max(0, vx1 - pad), max(0, vy1 - pad)
            cx2, cy2 = min(w, vx2 + pad), min(h, vy2 + pad)
            crop_label = "vehicle"

        crop_img = frame[cy1:cy2, cx1:cx2]

        # Upscale for visibility
        if crop_img.size > 0:
            target_height = 200 if has_plate else 300
            scale = max(1, target_height // max(crop_img.shape[0], 1))
            if scale > 1:
                crop_img = cv2.resize(
                    crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )

        crop_path = output_dir / f"vehicle_{i}_crop.jpg"
        cv2.imwrite(str(crop_path), crop_img)
        print(f"  Saved: {crop_path.name} ({crop_label})")

        # 3. OCR if plate detected
        ocr_text = None
        if has_plate and ocr_reader and crop_img.size > 0:
            try:
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                ocr_results = ocr_reader.readtext(gray)

                texts = []
                for _, text, conf in ocr_results:
                    cleaned = "".join(c for c in text if c.isalnum()).upper()
                    if cleaned and conf > 0.3:
                        texts.append(cleaned)

                if texts:
                    ocr_text = " ".join(texts)
                    print(f"  OCR: '{ocr_text}'")
            except Exception:
                pass

        results_summary.append(
            {
                "vehicle": i,
                "type": vtype,
                "plate_detected": has_plate,
                "ocr_text": ocr_text,
            }
        )

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Unique vehicles: {len(unique_vehicles)}")
    print(f"Images generated: {len(unique_vehicles) * 2}")

    plates_found = sum(1 for r in results_summary if r["plate_detected"])
    print(f"\nPlates detected: {plates_found}/{len(unique_vehicles)}")

    print("\nVehicle details:")
    for r in results_summary:
        status = "PLATE" if r["plate_detected"] else "vehicle only"
        ocr = f" -> '{r['ocr_text']}'" if r["ocr_text"] else ""
        print(f"  {r['vehicle']}. {r['type']}: {status}{ocr}")

    return results_summary


def cluster_vehicles(detections, frames_cache, min_distance=80):
    """Cluster vehicle detections to find unique vehicles."""
    if not detections:
        return []

    # Sort by vehicle confidence, prefer those with plates
    sorted_dets = sorted(
        detections,
        key=lambda x: (x["plate_bbox"] is not None, x["vehicle_conf"]),
        reverse=True,
    )

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

    # For each cluster, pick the best detection (prefer one with plate)
    unique = []
    for cluster in clusters:
        # Sort: plate detected first, then by confidence
        best = max(
            cluster, key=lambda x: (x["plate_bbox"] is not None, x["vehicle_conf"])
        )

        if best["frame_idx"] in frames_cache:
            unique.append(
                {
                    "vehicle_bbox": best["vehicle_bbox"],
                    "plate_bbox": best["plate_bbox"],
                    "vehicle_conf": best["vehicle_conf"],
                    "plate_conf": best["plate_conf"],
                    "vehicle_type": best["vehicle_type"],
                    "frame_idx": best["frame_idx"],
                    "frame": frames_cache[best["frame_idx"]],
                }
            )

    # Sort by position
    unique.sort(key=lambda x: (x["vehicle_bbox"][1], x["vehicle_bbox"][0]))
    return unique


if __name__ == "__main__":
    VIDEO = (
        "/workspaces/CCTV-Dissertaion/data/uploads/"
        "video_ec4d66401c0eedaf248a6c612533b3d1.mp4"
    )
    PLATE_MODEL = "/workspaces/CCTV-Dissertaion/models/license_plate_detector.pt"
    OUTPUT = "/workspaces/CCTV-Dissertaion/data/debug_plates/ec4d_final"

    detect_vehicles_and_plates(
        video_path=VIDEO,
        plate_model_path=PLATE_MODEL,
        output_dir=OUTPUT,
        frame_stride=3,
        vehicle_conf=0.25,
        plate_conf=0.15,
    )
