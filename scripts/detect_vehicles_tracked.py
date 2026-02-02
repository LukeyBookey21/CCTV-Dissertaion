#!/usr/bin/env python3
"""Vehicle detection with tracking - each unique vehicle appears only once."""

from pathlib import Path
from collections import defaultdict

import cv2

from ultralytics import YOLO

try:
    import easyocr

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# COCO vehicle class IDs
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def detect_vehicles_tracked(
    video_path: str,
    plate_model_path: str,
    output_dir: str,
    vehicle_conf: float = 0.3,
    plate_conf: float = 0.15,
):
    """
    Detect and track vehicles using ByteTrack.
    Each unique vehicle (by track ID) produces only one output.

    Args:
        video_path: Path to video file
        plate_model_path: Path to license plate model
        output_dir: Directory for output images
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

    # Track data per vehicle ID
    # For each track_id, store the best detection (highest conf or largest bbox)
    tracks = defaultdict(list)  # track_id -> list of detections
    frames_cache = {}
    frame_idx = 0

    print("\nTracking vehicles with ByteTrack...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Run tracking (ByteTrack is default)
        results = vehicle_model.track(
            frame,
            conf=vehicle_conf,
            classes=list(VEHICLE_CLASSES.keys()),
            persist=True,  # Persist tracks across frames
            verbose=False,
        )

        for result in results:
            if result.boxes.id is None:
                continue

            boxes = result.boxes
            for i, box in enumerate(boxes):
                track_id = int(boxes.id[i])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                bbox_area = (x2 - x1) * (y2 - y1)

                # Store this detection for this track
                tracks[track_id].append(
                    {
                        "frame_idx": frame_idx,
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "area": bbox_area,
                        "cls_id": cls_id,
                    }
                )

                # Cache frame if not already
                if frame_idx not in frames_cache:
                    frames_cache[frame_idx] = frame.copy()

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames}, unique tracks: {len(tracks)}")

        frame_idx += 1

    cap.release()
    print(f"\nTotal unique vehicle tracks: {len(tracks)}")

    if not tracks:
        print("No vehicles detected!")
        return

    # For each track, select the best frame (largest bounding box = closest/clearest)
    print("\nSelecting best frame for each vehicle...")
    best_detections = []

    for track_id, detections in tracks.items():
        # Pick detection with largest area (vehicle is closest/clearest)
        best = max(detections, key=lambda x: x["area"])
        best["track_id"] = track_id
        best_detections.append(best)

    print(f"Processing {len(best_detections)} unique vehicles...")

    # Initialize OCR
    ocr_reader = None
    if OCR_AVAILABLE:
        print("\nInitializing EasyOCR...")
        ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # Generate output for each unique vehicle
    print("\nGenerating output images...")
    results_summary = []

    for i, det in enumerate(best_detections, 1):
        frame = frames_cache[det["frame_idx"]]
        vbbox = det["bbox"]
        vtype = VEHICLE_CLASSES[det["cls_id"]]
        track_id = det["track_id"]
        h, w = frame.shape[:2]

        vx1, vy1, vx2, vy2 = vbbox

        # Try to detect plate within vehicle region
        pad = 10
        crop_x1 = max(0, vx1 - pad)
        crop_y1 = max(0, vy1 - pad)
        crop_x2 = min(w, vx2 + pad)
        crop_y2 = min(h, vy2 + pad)
        vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        plate_bbox = None
        if vehicle_crop.size > 0:
            plate_results = plate_model.predict(
                vehicle_crop, conf=plate_conf, imgsz=640, verbose=False
            )
            best_plate_conf = 0
            for pr in plate_results:
                for pb in pr.boxes:
                    px1, py1, px2, py2 = pb.xyxy[0].cpu().numpy()
                    pc = float(pb.conf[0])
                    pw, ph = px2 - px1, py2 - py1
                    ar = pw / ph if ph > 0 else 0

                    if (
                        pw >= 15
                        and ph >= 5
                        and 1.5 <= ar <= 8.0
                        and pc > best_plate_conf
                    ):
                        plate_bbox = [
                            crop_x1 + px1,
                            crop_y1 + py1,
                            crop_x1 + px2,
                            crop_y1 + py2,
                        ]
                        best_plate_conf = pc

        has_plate = plate_bbox is not None

        # 1. Full frame with detection box
        frame_annotated = frame.copy()
        cv2.rectangle(frame_annotated, (vx1, vy1), (vx2, vy2), (255, 100, 0), 2)

        if has_plate:
            px1, py1, px2, py2 = [int(v) for v in plate_bbox]
            cv2.rectangle(frame_annotated, (px1, py1), (px2, py2), (0, 255, 0), 3)
            label = f"#{track_id} {vtype} - PLATE"
        else:
            label = f"#{track_id} {vtype}"

        cv2.putText(
            frame_annotated,
            label,
            (vx1, vy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        full_path = output_dir / f"track_{track_id}_full.jpg"
        cv2.imwrite(str(full_path), frame_annotated)
        print(f"  Saved: {full_path.name}")

        # 2. Cropped image
        if has_plate:
            px1, py1, px2, py2 = [int(v) for v in plate_bbox]
            pad = 15
            cx1, cy1 = max(0, px1 - pad), max(0, py1 - pad)
            cx2, cy2 = min(w, px2 + pad), min(h, py2 + pad)
            crop_type = "plate"
        else:
            pad = 20
            cx1, cy1 = max(0, vx1 - pad), max(0, vy1 - pad)
            cx2, cy2 = min(w, vx2 + pad), min(h, vy2 + pad)
            crop_type = "vehicle"

        crop_img = frame[cy1:cy2, cx1:cx2]

        if crop_img.size > 0:
            target_h = 200 if has_plate else 300
            scale = max(1, target_h // max(crop_img.shape[0], 1))
            if scale > 1:
                crop_img = cv2.resize(
                    crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )

        crop_path = output_dir / f"track_{track_id}_crop.jpg"
        cv2.imwrite(str(crop_path), crop_img)
        print(f"  Saved: {crop_path.name} ({crop_type})")

        # 3. OCR
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
                "track_id": track_id,
                "type": vtype,
                "plate_detected": has_plate,
                "ocr_text": ocr_text,
            }
        )

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Output: {output_dir}")
    print(f"Unique vehicles: {len(best_detections)}")
    print(f"Images: {len(best_detections) * 2}")

    plates_found = sum(1 for r in results_summary if r["plate_detected"])
    print(f"Plates detected: {plates_found}/{len(best_detections)}")

    print("\nVehicles:")
    for r in results_summary:
        status = "PLATE" if r["plate_detected"] else "vehicle"
        ocr = f" -> '{r['ocr_text']}'" if r["ocr_text"] else ""
        print(f"  Track #{r['track_id']}: {r['type']} ({status}){ocr}")

    return results_summary


if __name__ == "__main__":
    # Default test on ec4d
    VIDEO = (
        "/workspaces/CCTV-Dissertaion/data/uploads/"
        "video_ec4d66401c0eedaf248a6c612533b3d1.mp4"
    )
    PLATE_MODEL = "/workspaces/CCTV-Dissertaion/models/license_plate_detector.pt"
    OUTPUT = "/workspaces/CCTV-Dissertaion/data/debug_plates/tracked_test"

    detect_vehicles_tracked(
        video_path=VIDEO,
        plate_model_path=PLATE_MODEL,
        output_dir=OUTPUT,
        vehicle_conf=0.25,
        plate_conf=0.15,
    )
