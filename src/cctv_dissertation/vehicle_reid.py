"""Vehicle Re-ID feature extraction and matching for cross-camera tracking."""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

from cctv_dissertation.attributes import describe_vehicle

# Try to import torchreid for Re-ID model
try:
    import torchreid

    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False

try:
    import easyocr

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# COCO vehicle class IDs
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleReID:
    """Vehicle Re-ID feature extractor for cross-camera matching."""

    def __init__(
        self,
        plate_model_path: str,
        db_path: str = "data/vehicle_reid.db",
        device: str = "cpu",
    ):
        """
        Initialize the Vehicle Re-ID system.

        Args:
            plate_model_path: Path to license plate detection model
            db_path: Path to SQLite database for storing embeddings
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.device = device
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        print("Loading vehicle detector (YOLOv8n)...")
        self.vehicle_model = YOLO("yolov8n.pt")

        print(f"Loading plate detector from {plate_model_path}...")
        self.plate_model = YOLO(plate_model_path)

        print("Loading Re-ID feature extractor...")
        self._init_reid_model()

        print("Initializing database...")
        self._init_database()

        self.ocr_reader = None
        if OCR_AVAILABLE:
            print("Initializing EasyOCR...")
            self.ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        # Image preprocessing for Re-ID
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),  # Standard Re-ID input size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _init_reid_model(self):
        """Initialize the Re-ID feature extraction model."""
        if TORCHREID_AVAILABLE:
            # Use OSNet - lightweight and effective for Re-ID
            self.reid_model = torchreid.models.build_model(
                name="osnet_x0_25",  # Smallest OSNet variant
                num_classes=1,  # We only need features, not classification
                pretrained=True,
            )
            self.reid_model.eval()
            self.reid_model.to(self.device)
            self.feature_dim = 512
        else:
            # Fallback to torchvision ResNet18 for feature extraction
            from torchvision.models import resnet18, ResNet18_Weights

            self.reid_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            # Remove classification layer to get features
            self.reid_model = torch.nn.Sequential(
                *list(self.reid_model.children())[:-1]
            )
            self.reid_model.eval()
            self.reid_model.to(self.device)
            self.feature_dim = 512

    def _init_database(self):
        """Initialize SQLite database for storing vehicle embeddings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_hash TEXT NOT NULL,
                track_id INTEGER,
                vehicle_type TEXT,
                frame_idx INTEGER,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                plate_text TEXT,
                plate_confidence REAL,
                embedding BLOB NOT NULL,
                image_path TEXT,
                color TEXT,
                description TEXT,
                timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_video_hash ON vehicles(video_hash)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_plate_text ON vehicles(plate_text)
        """
        )

        conn.commit()
        conn.close()

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Re-ID features from a vehicle image.

        Args:
            image: BGR image of vehicle crop

        Returns:
            512-dimensional feature vector (normalized)
        """
        if image.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            if TORCHREID_AVAILABLE:
                features = self.reid_model(input_tensor)
            else:
                features = self.reid_model(input_tensor)
                features = features.squeeze(-1).squeeze(-1)

        # Normalize features (L2 normalization)
        features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy().flatten()

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors."""
        return float(np.dot(feat1, feat2))

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        video_hash: Optional[str] = None,
        vehicle_conf: float = 0.3,
        plate_conf: float = 0.15,
        similarity_threshold: float = 0.7,
        frame_stride: int = 10,  # Sample every Nth frame for speed
    ) -> List[Dict[str, Any]]:
        """
        Process a video: detect vehicles, extract features, deduplicate, save to DB.

        Args:
            video_path: Path to video file
            output_dir: Directory for output images
            video_hash: Unique identifier for this video
            vehicle_conf: Vehicle detection confidence threshold
            plate_conf: Plate detection confidence threshold
            similarity_threshold: Threshold for considering vehicles as same (0-1)

        Returns:
            List of unique vehicle detections with their features
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if video_hash is None:
            video_hash = video_path.stem

        print(f"\nProcessing: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video: {total_frames} frames at {fps:.1f} fps")

        # Collect all vehicle detections with features
        all_detections = []
        frame_idx = 0

        print(
            f"\nDetecting vehicles and extracting features (stride={frame_stride})..."
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for speed
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]

            # Detect vehicles
            results = self.vehicle_model.predict(
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
                    conf = float(box.conf[0])

                    # Crop vehicle with padding
                    pad = 10
                    cx1 = max(0, vx1 - pad)
                    cy1 = max(0, vy1 - pad)
                    cx2 = min(w, vx2 + pad)
                    cy2 = min(h, vy2 + pad)
                    vehicle_crop = frame[cy1:cy2, cx1:cx2]

                    if vehicle_crop.size == 0:
                        continue

                    # Extract Re-ID features
                    features = self.extract_features(vehicle_crop)

                    # Extract color description
                    vtype = VEHICLE_CLASSES[cls_id]
                    vehicle_desc = describe_vehicle(vehicle_crop, vehicle_type=vtype)

                    # Try plate detection
                    plate_bbox, plate_conf_score = self._detect_plate(
                        vehicle_crop, plate_conf
                    )

                    detection = {
                        "frame_idx": frame_idx,
                        "frame": frame.copy(),
                        "vehicle_bbox": [vx1, vy1, vx2, vy2],
                        "vehicle_crop": vehicle_crop.copy(),
                        "vehicle_conf": conf,
                        "vehicle_type": vtype,
                        "vehicle_color": vehicle_desc["color"],
                        "vehicle_description": vehicle_desc["description"],
                        "features": features,
                        "plate_bbox": (
                            [
                                cx1 + plate_bbox[0],
                                cy1 + plate_bbox[1],
                                cx1 + plate_bbox[2],
                                cy1 + plate_bbox[3],
                            ]
                            if plate_bbox
                            else None
                        ),
                        "plate_conf": plate_conf_score,
                        "bbox_area": (vx2 - vx1) * (vy2 - vy1),
                    }
                    all_detections.append(detection)

            if frame_idx % 50 == 0:
                print(
                    f"  Frame {frame_idx}/{total_frames}, "
                    f"detections: {len(all_detections)}"
                )

            frame_idx += 1

        cap.release()
        print(f"\nTotal detections: {len(all_detections)}")

        if not all_detections:
            print("No vehicles detected!")
            return []

        # Deduplicate using feature similarity
        print(f"\nDeduplicating with similarity threshold {similarity_threshold}...")
        unique_vehicles = self._deduplicate_by_features(
            all_detections, similarity_threshold
        )
        print(f"Unique vehicles: {len(unique_vehicles)}")

        # Generate outputs and save to database
        print("\nGenerating outputs and saving to database...")
        results = self._save_results(unique_vehicles, video_hash, output_dir)

        return results

    def _detect_plate(
        self, vehicle_crop: np.ndarray, conf_threshold: float
    ) -> Tuple[Optional[List], float]:
        """Detect license plate within vehicle crop."""
        plate_results = self.plate_model.predict(
            vehicle_crop, conf=conf_threshold, imgsz=640, verbose=False
        )

        best_plate = None
        best_conf = 0

        for pr in plate_results:
            for pb in pr.boxes:
                px1, py1, px2, py2 = pb.xyxy[0].cpu().numpy()
                pc = float(pb.conf[0])
                pw, ph = px2 - px1, py2 - py1
                ar = pw / ph if ph > 0 else 0

                if pw >= 15 and ph >= 5 and 1.5 <= ar <= 8.0 and pc > best_conf:
                    best_plate = [px1, py1, px2, py2]
                    best_conf = pc

        return best_plate, best_conf

    def _deduplicate_by_features(
        self, detections: List[Dict], threshold: float
    ) -> List[Dict]:
        """
        Deduplicate detections using feature similarity.
        Groups similar vehicles and keeps the best detection from each group.
        """
        if not detections:
            return []

        # Sort by bbox area (larger = closer = clearer)
        sorted_dets = sorted(detections, key=lambda x: x["bbox_area"], reverse=True)

        clusters = []
        used = set()

        for i, det in enumerate(sorted_dets):
            if i in used:
                continue

            cluster = [det]
            used.add(i)

            # Find all similar detections
            for j, other in enumerate(sorted_dets):
                if j in used:
                    continue

                similarity = self.compute_similarity(det["features"], other["features"])
                if similarity >= threshold:
                    cluster.append(other)
                    used.add(j)

            clusters.append(cluster)

        # For each cluster, pick the best detection (largest area or plate)
        unique = []
        for cluster in clusters:
            # Prefer detection with plate, then largest area
            best = max(
                cluster, key=lambda x: (x["plate_bbox"] is not None, x["bbox_area"])
            )
            unique.append(best)

        return unique

    def _save_results(
        self, vehicles: List[Dict], video_hash: str, output_dir: Path
    ) -> List[Dict[str, Any]]:
        """Save vehicle images and embeddings to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        results = []
        for i, vehicle in enumerate(vehicles, 1):
            frame = vehicle["frame"]
            vbbox = vehicle["vehicle_bbox"]
            features = vehicle["features"]
            h, w = frame.shape[:2]

            vx1, vy1, vx2, vy2 = vbbox
            has_plate = vehicle["plate_bbox"] is not None

            # 1. Full frame with box
            frame_annotated = frame.copy()
            cv2.rectangle(frame_annotated, (vx1, vy1), (vx2, vy2), (255, 100, 0), 2)

            v_desc = vehicle.get("vehicle_description", vehicle["vehicle_type"])

            if has_plate:
                px1, py1, px2, py2 = [int(v) for v in vehicle["plate_bbox"]]
                cv2.rectangle(frame_annotated, (px1, py1), (px2, py2), (0, 255, 0), 3)
                label = f"V{i} {v_desc} - PLATE"
            else:
                label = f"V{i} {v_desc}"

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

            # 2. Cropped image
            if has_plate:
                px1, py1, px2, py2 = [int(v) for v in vehicle["plate_bbox"]]
                pad = 15
                cx1, cy1 = max(0, px1 - pad), max(0, py1 - pad)
                cx2, cy2 = min(w, px2 + pad), min(h, py2 + pad)
            else:
                pad = 20
                cx1, cy1 = max(0, vx1 - pad), max(0, vy1 - pad)
                cx2, cy2 = min(w, vx2 + pad), min(h, vy2 + pad)

            crop_img = frame[cy1:cy2, cx1:cx2]
            if crop_img.size > 0:
                scale = max(1, 200 // max(crop_img.shape[0], 1))
                if scale > 1:
                    crop_img = cv2.resize(
                        crop_img,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )

            crop_path = output_dir / f"vehicle_{i}_crop.jpg"
            cv2.imwrite(str(crop_path), crop_img)

            # 3. OCR if plate detected
            plate_text = None
            if has_plate and self.ocr_reader:
                try:
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    ocr_results = self.ocr_reader.readtext(gray)

                    texts = []
                    for _, text, conf in ocr_results:
                        cleaned = "".join(c for c in text if c.isalnum()).upper()
                        if cleaned and conf > 0.3:
                            texts.append(cleaned)
                    if texts:
                        plate_text = " ".join(texts)
                        print(f"  OCR: '{plate_text}'")
                except Exception:
                    pass

            print(f"  Vehicle {i}: {v_desc}")

            # Save to database
            cursor.execute(
                """
                INSERT INTO vehicles (
                    video_hash, vehicle_type, frame_idx,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    plate_text, plate_confidence, embedding, image_path,
                    color, description, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    video_hash,
                    vehicle["vehicle_type"],
                    vehicle["frame_idx"],
                    vx1,
                    vy1,
                    vx2,
                    vy2,
                    plate_text,
                    vehicle["plate_conf"] if has_plate else None,
                    features.tobytes(),
                    str(crop_path),
                    vehicle.get("vehicle_color", ""),
                    v_desc,
                    datetime.now().isoformat(),
                ),
            )

            results.append(
                {
                    "id": i,
                    "type": vehicle["vehicle_type"],
                    "color": vehicle.get("vehicle_color", ""),
                    "description": v_desc,
                    "plate_detected": has_plate,
                    "plate_text": plate_text,
                    "image_path": str(crop_path),
                }
            )

        conn.commit()
        conn.close()

        # Print summary
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Output: {output_dir}")
        print(f"Unique vehicles: {len(vehicles)}")
        print(f"Images: {len(vehicles) * 2}")
        plates = sum(1 for r in results if r["plate_detected"])
        print(f"Plates detected: {plates}/{len(vehicles)}")
        print(f"Embeddings saved to: {self.db_path}")

        return results

    def find_matches(
        self, query_embedding: np.ndarray, threshold: float = 0.7, limit: int = 10
    ) -> List[Dict]:
        """
        Find matching vehicles across all videos in the database.

        Args:
            query_embedding: 512-dim feature vector to match
            threshold: Minimum similarity threshold
            limit: Maximum number of matches to return

        Returns:
            List of matching vehicle records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM vehicles")
        rows = cursor.fetchall()
        conn.close()

        matches = []
        for row in rows:
            db_embedding = np.frombuffer(row[11], dtype=np.float32)  # embedding column
            similarity = self.compute_similarity(query_embedding, db_embedding)

            if similarity >= threshold:
                matches.append(
                    {
                        "id": row[0],
                        "video_hash": row[1],
                        "vehicle_type": row[3],
                        "plate_text": row[8],
                        "image_path": row[12],
                        "similarity": similarity,
                    }
                )

        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:limit]


def process_video_with_reid(
    video_path: str,
    output_dir: str,
    plate_model_path: str = "models/license_plate_detector.pt",
    db_path: str = "data/vehicle_reid.db",
    similarity_threshold: float = 0.7,
    frame_stride: int = 10,
):
    """Convenience function to process a single video."""
    reid = VehicleReID(
        plate_model_path=plate_model_path,
        db_path=db_path,
    )
    return reid.process_video(
        video_path=video_path,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold,
        frame_stride=frame_stride,
    )


if __name__ == "__main__":
    # Test on ec4d video
    _vid = (
        "/workspaces/CCTV-Dissertaion/data/uploads/"
        "video_ec4d66401c0eedaf248a6c612533b3d1.mp4"
    )
    _plate = "/workspaces/CCTV-Dissertaion/models/" "license_plate_detector.pt"
    process_video_with_reid(
        video_path=_vid,
        output_dir="/workspaces/CCTV-Dissertaion/data/debug_plates/reid_test",
        plate_model_path=_plate,
        db_path="/workspaces/CCTV-Dissertaion/data/vehicle_reid.db",
    )
