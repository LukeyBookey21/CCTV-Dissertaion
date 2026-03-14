"""Person Re-ID feature extraction and matching for cross-camera tracking."""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

from cctv_dissertation.attributes import describe_person

# Try to import torchreid for Re-ID model
try:
    import torchreid

    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False


class PersonReID:
    """Person Re-ID feature extractor for cross-camera tracking."""

    def __init__(
        self,
        db_path: str = "data/person_reid.db",
        device: str = "cpu",
    ):
        """
        Initialize the Person Re-ID system.

        Args:
            db_path: Path to SQLite database for storing embeddings
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.device = device
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        print("Loading person detector (YOLOv8n)...")
        self.detector = YOLO("yolov8n.pt")

        print("Loading Person Re-ID model...")
        self._init_reid_model()

        print("Initializing database...")
        self._init_database()

        # Image preprocessing for Re-ID (person images are typically 256x128)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _init_reid_model(self):
        """Initialize the Person Re-ID model."""
        try:
            # Try OSNet - excellent for person Re-ID
            self.reid_model = torchreid.models.build_model(
                name="osnet_x1_0",
                num_classes=1,
                pretrained=True,
            )
            self.reid_model.eval()
            self.reid_model.to(self.device)
            self.feature_dim = 512
            self.use_torchreid = True
            print("  Using OSNet for person Re-ID")
        except Exception as e:
            print(f"  OSNet failed: {e}")
            # Fallback to ResNet18
            from torchvision.models import resnet18, ResNet18_Weights

            self.reid_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.reid_model = torch.nn.Sequential(
                *list(self.reid_model.children())[:-1]
            )
            self.reid_model.eval()
            self.reid_model.to(self.device)
            self.feature_dim = 512
            self.use_torchreid = False
            print("  Using ResNet18 fallback")

    def _init_database(self):
        """Initialize SQLite database for storing person embeddings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                camera_id TEXT,
                frame_idx INTEGER,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                confidence REAL,
                embedding BLOB NOT NULL,
                image_path TEXT,
                upper_color TEXT,
                lower_color TEXT,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_video ON persons(video_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_camera ON persons(camera_id)")

        conn.commit()
        conn.close()

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Re-ID features from a person crop."""
        if image.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.reid_model(input_tensor)
            if not getattr(self, "use_torchreid", False):
                features = features.squeeze(-1).squeeze(-1)

        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy().flatten()

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors."""
        return float(np.dot(feat1, feat2))

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        camera_id: Optional[str] = None,
        person_conf: float = 0.25,  # Lower to catch more people
        similarity_threshold: float = 0.47,
        frame_stride: int = 5,  # Sample every 5th frame
    ) -> List[Dict[str, Any]]:
        """
        Process a video: detect people, extract features, deduplicate, save to DB.

        Args:
            video_path: Path to video file
            output_dir: Directory for output images
            camera_id: Identifier for this camera/location
            person_conf: Person detection confidence threshold
            similarity_threshold: Threshold for deduplication
            frame_stride: Sample every Nth frame

        Returns:
            List of unique person detections
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_name = video_path.stem
        if camera_id is None:
            camera_id = video_name

        print(f"\nProcessing: {video_path.name}")
        print(f"Camera ID: {camera_id}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video: {total_frames} frames at {fps:.1f} fps")

        all_detections = []
        frame_idx = 0

        print(f"\nDetecting people (stride={frame_stride})...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            h, w = frame.shape[:2]

            # Detect people (class 0 in COCO)
            results = self.detector.predict(
                frame,
                conf=person_conf,
                classes=[0],  # Person class only
                verbose=False,
            )

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    conf = float(box.conf[0])

                    # Filter small detections
                    box_h = y2 - y1
                    box_w = x2 - x1
                    if box_h < 50 or box_w < 20:
                        continue

                    # Crop person with padding
                    pad = 5
                    cx1 = max(0, x1 - pad)
                    cy1 = max(0, y1 - pad)
                    cx2 = min(w, x2 + pad)
                    cy2 = min(h, y2 + pad)
                    person_crop = frame[cy1:cy2, cx1:cx2]

                    if person_crop.size == 0:
                        continue

                    # Extract Re-ID features
                    features = self.extract_features(person_crop)

                    # Extract color description
                    person_desc = describe_person(person_crop)

                    detection = {
                        "frame_idx": frame_idx,
                        "frame": frame.copy(),
                        "bbox": [x1, y1, x2, y2],
                        "crop": person_crop.copy(),
                        "confidence": conf,
                        "features": features,
                        "bbox_area": box_h * box_w,
                        "upper_color": person_desc["upper_color"],
                        "lower_color": person_desc["lower_color"],
                        "description": person_desc["description"],
                    }
                    all_detections.append(detection)

            if frame_idx % 30 == 0:
                print(
                    f"  Frame {frame_idx}/{total_frames}, "
                    f"detections: {len(all_detections)}"
                )

            frame_idx += 1

        cap.release()
        print(f"\nTotal detections: {len(all_detections)}")

        if not all_detections:
            print("No people detected!")
            return []

        # Deduplicate using feature similarity
        print(f"\nDeduplicating (threshold={similarity_threshold})...")
        unique_persons = self._deduplicate(all_detections, similarity_threshold)
        print(f"Unique persons: {len(unique_persons)}")

        # Save results
        print("\nSaving results...")
        results = self._save_results(unique_persons, video_name, camera_id, output_dir)

        return results

    def _deduplicate(self, detections: List[Dict], threshold: float) -> List[Dict]:
        """Deduplicate detections using feature similarity.

        Greedy clustering: sorts by bbox area (largest first),
        then merges any later detection whose similarity to
        the cluster leader exceeds the threshold.
        """
        if not detections:
            return []

        # Filter very small / partial detections
        min_area = 2500
        dets = [d for d in detections if d["bbox_area"] >= min_area]
        if not dets:
            dets = detections

        sorted_dets = sorted(dets, key=lambda x: x["bbox_area"], reverse=True)

        clusters = []
        used = set()

        for i, det in enumerate(sorted_dets):
            if i in used:
                continue

            cluster = [det]
            used.add(i)

            for j, other in enumerate(sorted_dets):
                if j in used:
                    continue

                # Same frame = must be different people
                if abs(det["frame_idx"] - other["frame_idx"]) <= 1:
                    continue

                sim = self.compute_similarity(det["features"], other["features"])
                if sim >= threshold:
                    cluster.append(other)
                    used.add(j)

            clusters.append(cluster)

        # Pick best (largest, clearest) from each cluster
        unique = []
        for cluster in clusters:
            best = max(cluster, key=lambda x: x["bbox_area"])
            unique.append(best)

        return unique

    def _save_results(
        self, persons: List[Dict], video_name: str, camera_id: str, output_dir: Path
    ) -> List[Dict]:
        """Save person images and embeddings to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        results = []
        for i, person in enumerate(persons, 1):
            frame = person["frame"]
            bbox = person["bbox"]
            features = person["features"]
            h, w = frame.shape[:2]

            x1, y1, x2, y2 = bbox

            # 1. Full frame with box
            frame_annotated = frame.copy()
            cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_annotated,
                f"P{i}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            full_path = output_dir / f"person_{i}_full.jpg"
            cv2.imwrite(str(full_path), frame_annotated)

            # 2. Cropped person (upscaled for visibility)
            crop = person["crop"]
            scale = max(1, 200 // max(crop.shape[0], 1))
            if scale > 1:
                crop = cv2.resize(
                    crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )

            crop_path = output_dir / f"person_{i}_crop.jpg"
            cv2.imwrite(str(crop_path), crop)

            desc = person.get("description", "unknown")
            print(f"  Saved: person_{i}_full.jpg, person_{i}_crop.jpg  ->  {desc}")

            # Save to database
            cursor.execute(
                """
                INSERT INTO persons (
                    video_name, camera_id, frame_idx,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    confidence, embedding, image_path,
                    upper_color, lower_color, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    video_name,
                    camera_id,
                    person["frame_idx"],
                    x1,
                    y1,
                    x2,
                    y2,
                    person["confidence"],
                    features.tobytes(),
                    str(crop_path),
                    person.get("upper_color", ""),
                    person.get("lower_color", ""),
                    desc,
                ),
            )

            results.append(
                {
                    "id": i,
                    "camera": camera_id,
                    "image_path": str(crop_path),
                    "description": desc,
                }
            )

        conn.commit()
        conn.close()

        print(f"\n{'=' * 50}")
        print("SUMMARY")
        print("=" * 50)
        print(f"Camera: {camera_id}")
        print(f"Unique persons: {len(persons)}")
        print(f"Images saved to: {output_dir}")
        print(f"Embeddings saved to: {self.db_path}")

        return results

    def find_cross_camera_matches(self, threshold: float = 0.6) -> List[Dict]:
        """
        Find matching persons across different cameras.

        Returns list of matches with similarity scores.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, video_name, camera_id, embedding, image_path FROM persons"
        )
        rows = cursor.fetchall()
        conn.close()

        if len(rows) < 2:
            return []

        # Compare persons from different cameras
        matches = []
        for i, row1 in enumerate(rows):
            emb1 = np.frombuffer(row1[3], dtype=np.float32)
            cam1 = row1[2]

            for j, row2 in enumerate(rows):
                if j <= i:
                    continue

                cam2 = row2[2]
                if cam1 == cam2:  # Skip same camera
                    continue

                emb2 = np.frombuffer(row2[3], dtype=np.float32)
                similarity = self.compute_similarity(emb1, emb2)

                if similarity >= threshold:
                    matches.append(
                        {
                            "person1_id": row1[0],
                            "person1_camera": cam1,
                            "person1_image": row1[4],
                            "person2_id": row2[0],
                            "person2_camera": cam2,
                            "person2_image": row2[4],
                            "similarity": similarity,
                        }
                    )

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches


def process_cross_camera_videos(video_paths: List[str], output_base: str, db_path: str):
    """Process multiple videos and find cross-camera matches."""
    reid = PersonReID(db_path=db_path)
    output_base = Path(output_base)

    # Process each video independently
    for video_path in video_paths:
        video_name = Path(video_path).stem
        output_dir = output_base / video_name

        print(f"\n{'='*60}")
        print(f"PROCESSING: {video_name}")
        print("=" * 60)

        reid.process_video(
            video_path=video_path,
            output_dir=str(output_dir),
            camera_id=video_name,
        )

    # Find cross-camera matches
    print("\n" + "=" * 60)
    print("CROSS-CAMERA MATCHING")
    print("=" * 60)

    matches = reid.find_cross_camera_matches(threshold=0.5)

    if matches:
        print(f"\nFound {len(matches)} cross-camera matches:\n")
        for m in matches:
            print(
                f"  {m['person1_camera']} P#{m['person1_id']} <-> "
                f"{m['person2_camera']} P#{m['person2_id']} "
                f"(similarity: {m['similarity']:.3f})"
            )
    else:
        print("\nNo cross-camera matches found.")

    return matches


if __name__ == "__main__":
    videos = [
        "data/uploads/Cross-camera-garage.mp4",
        "data/uploads/Cross-camera-garden.mp4",
    ]

    process_cross_camera_videos(
        video_paths=videos,
        output_base="data/debug_plates/person_reid",
        db_path="data/person_reid.db",
    )
