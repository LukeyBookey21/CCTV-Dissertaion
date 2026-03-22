"""Single-camera tracking with ByteTrack — persistent IDs for people and vehicles."""

import shutil
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

from cctv_dissertation.attributes import (
    describe_person,
    describe_vehicle,
)


def _reencode_h264(src: str) -> str:
    """Re-encode an mp4v video to H.264 so browsers can play it.

    If ffmpeg is unavailable the original file is returned as-is.
    """
    if not shutil.which("ffmpeg"):
        return src
    tmp = str(src) + ".tmp.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                "-an",
                tmp,
            ],
            check=True,
            capture_output=True,
        )
        Path(src).unlink(missing_ok=True)
        Path(tmp).rename(src)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
    return src


def calc_auto_stride(video_path: str, target_minutes: int = 15) -> int:
    """Calculate optimal frame stride based on video duration.

    Args:
        video_path: Path to the video file.
        target_minutes: Target processing time in minutes (default 15).

    Returns:
        Recommended frame stride (1 = all frames, 5 = every 5th, etc).
        Ensures at least 1 effective fps for accuracy.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # Assume ~150 frames/sec processing speed on CPU (conservative estimate)
    processing_speed = 150
    target_frames = processing_speed * 60 * target_minutes

    if total_frames <= target_frames:
        return 1  # Short video, process all frames

    # Calculate stride needed to hit target
    stride = int(total_frames / target_frames)

    # Ensure effective fps is at least 1 (catches anyone visible for 1+ second)
    max_stride = max(1, int(fps))
    stride = min(stride, max_stride)

    return max(1, stride)


def extract_video_timestamp(video_path: str) -> Optional[str]:
    """Read the first frame of a video and OCR the burned-in timestamp.

    Returns an ISO-style string ``"YYYY-MM-DD HH:MM:SS"`` if a
    timestamp is found in the bottom strip of the frame, or *None*.
    """
    import re

    try:
        import easyocr
    except ImportError:
        return None

    vid = cv2.VideoCapture(video_path)
    ret, frame = vid.read()
    vid.release()
    if not ret:
        return None

    h = frame.shape[0]
    strip = frame[h - 60 :, :]

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(strip)
    full_text = " ".join(t for _, t, _ in results)

    # Normalise common OCR noise
    full_text = re.sub(r"(\d)\s*:\s*(\d)", r"\1:\2", full_text)
    full_text = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", full_text)
    full_text = full_text.replace("_", "-")  # underscore → dash
    full_text = re.sub(r"(\d),(\d)", r"\1 \2", full_text)  # comma → space

    # Strip one spurious inserted digit from 5-char year lookalikes
    # e.g. "20026" → "2026": keep first digit + last 3, drop the noise digit
    full_text = re.sub(r"\b([12])\d(\d{3})\b", r"\1\2", full_text)

    m = re.search(
        r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})",
        full_text,
    )
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return None


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

# COCO class IDs
PERSON_CLASS = 0
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class SingleCameraTracker:
    """Track people and vehicles through a single video using ByteTrack.

    Each entity gets a persistent ID that stays consistent across frames.
    Collects the best crop per track and extracts colour/plate descriptions.
    """

    def __init__(
        self,
        plate_model_path: str = "models/license_plate_detector.pt",
        db_path: str = "data/tracker.db",
        device: str = "cpu",
    ):
        self.device = device
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Detection model (shared for people + vehicles)
        self.model = YOLO("yolov8s.pt")

        # Plate detector
        plate_path = Path(plate_model_path)
        self.plate_model = YOLO(str(plate_path)) if plate_path.exists() else None

        # Re-ID feature extractor for cross-camera matching later
        self._init_reid_model()

        # OCR
        self.ocr_reader = None
        if OCR_AVAILABLE:
            self.ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        # Preprocessing for Re-ID
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._init_database()

    def _init_reid_model(self):
        """Load OSNet (or ResNet18 fallback) for embedding extraction."""
        if TORCHREID_AVAILABLE:
            try:
                self.reid_model = torchreid.models.build_model(
                    name="osnet_x1_0",
                    num_classes=1,
                    pretrained=True,
                )
                self.reid_model.eval().to(self.device)
                self.feature_dim = 512
                self._use_torchreid = True
                return
            except Exception:
                pass

        from torchvision.models import resnet18, ResNet18_Weights

        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.reid_model = torch.nn.Sequential(*list(base.children())[:-1])
        self.reid_model.eval().to(self.device)
        self.feature_dim = 512
        self._use_torchreid = False

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS tracked_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                video_path TEXT NOT NULL,
                camera_label TEXT,
                entity_type TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                first_frame INTEGER,
                last_frame INTEGER,
                first_ts REAL,
                last_ts REAL,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                confidence REAL,
                color TEXT,
                upper_color TEXT,
                lower_color TEXT,
                description TEXT,
                vehicle_type TEXT,
                plate_text TEXT,
                plate_confidence REAL,
                embedding BLOB,
                crop_path TEXT,
                full_frame_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS track_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER REFERENCES tracked_entities(id),
                frame_idx INTEGER,
                timestamp REAL,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL
            )
        """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_te_video " "ON tracked_entities(video_path)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_te_camera "
            "ON tracked_entities(camera_label)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_tf_entity " "ON track_frames(entity_id)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_te_run " "ON tracked_entities(run_id)"
        )
        # Migrate existing DBs: add run_id column if missing
        cols = {row[1] for row in c.execute("PRAGMA table_info(tracked_entities)")}
        if "run_id" not in cols:
            c.execute("ALTER TABLE tracked_entities ADD COLUMN run_id TEXT")
        conn.commit()
        conn.close()

    # ── Feature extraction ────────────────────────────────────────

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.reid_model(tensor)
            if not self._use_torchreid:
                feat = feat.squeeze(-1).squeeze(-1)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().flatten()

    # ── Plate helpers ─────────────────────────────────────────────

    def _detect_plate(
        self, crop: np.ndarray, conf: float = 0.15
    ) -> Tuple[Optional[List[int]], float, Optional[str]]:
        """Detect plate in crop, return (bbox, conf, ocr_text)."""
        if self.plate_model is None or crop.size == 0:
            return None, 0.0, None

        results = self.plate_model.predict(crop, conf=conf, imgsz=640, verbose=False)
        best_box, best_conf = None, 0.0
        for r in results:
            for b in r.boxes:
                px1, py1, px2, py2 = b.xyxy[0].cpu().numpy()
                pc = float(b.conf[0])
                pw, ph = px2 - px1, py2 - py1
                ar = pw / ph if ph > 0 else 0
                if pw >= 15 and ph >= 5 and 1.5 <= ar <= 8.0 and pc > best_conf:
                    best_box = [int(px1), int(py1), int(px2), int(py2)]
                    best_conf = pc

        ocr_text = None
        if best_box and self.ocr_reader:
            px1, py1, px2, py2 = best_box
            plate_crop = crop[
                max(0, py1 - 5) : min(crop.shape[0], py2 + 5),
                max(0, px1 - 5) : min(crop.shape[1], px2 + 5),
            ]
            if plate_crop.size > 0:
                try:
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    ocr_results = self.ocr_reader.readtext(gray)
                    texts = []
                    for _, text, c in ocr_results:
                        cleaned = "".join(ch for ch in text if ch.isalnum()).upper()
                        if cleaned and c > 0.3:
                            texts.append(cleaned)
                    if texts:
                        ocr_text = " ".join(texts)
                except Exception:
                    pass

        return best_box, best_conf, ocr_text

    # ── Main processing ───────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        camera_label: Optional[str] = None,
        person_conf: float = 0.25,
        vehicle_conf: float = 0.30,
        plate_conf: float = 0.15,
        progress_callback: Optional[Callable[[dict], None]] = None,
        motion_threshold: float = 0.0005,
        frame_cache_limit: int = 500,
        frame_stride: int = 1,
        run_id: Optional[str] = None,
    ) -> Dict:
        """Track all people and vehicles through a video.

        Args:
            progress_callback: Optional function called with progress dict:
                {frame, total, persons, vehicles, fps, eta_seconds, skipped}
            motion_threshold: Min fraction of changed pixels to process frame.
                Set to 0 to disable motion detection (process all frames).
            frame_cache_limit: Max frames to keep in memory for crop extraction.
            frame_stride: Process every Nth frame (1=all, 5=every 5th, etc).
                Higher values = faster but may miss brief appearances.

        Returns dict with 'persons' and 'vehicles' lists, each containing
        per-track summaries with descriptions, crops, and frame histories.
        """
        import time

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if camera_label is None:
            camera_label = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # We run two separate trackers (persist=True uses ByteTrack internally).
        # One for people, one for vehicles — so track IDs don't collide.
        person_tracks: Dict[int, List[dict]] = defaultdict(list)
        vehicle_tracks: Dict[int, List[dict]] = defaultdict(list)
        frame_cache: Dict[int, np.ndarray] = {}

        # Best crop per track (highest conf) — stored as small crops so we can
        # compute Re-ID embeddings during merge even after frame_cache evicts
        # early frames (fixes cross-frame-walking identity splits).
        track_best_crop: Dict[int, np.ndarray] = {}
        track_best_conf: Dict[int, float] = {}

        all_classes = [PERSON_CLASS] + list(VEHICLE_CLASSES.keys())
        min_conf = min(person_conf, vehicle_conf)

        # Reset ByteTrack state between videos (persist=True keeps state on
        # the model object; clear it so each video starts fresh).
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor = None

        # Choose ByteTrack config: stride>1 needs lower match_thresh so a
        # walking person whose bbox doesn't overlap between sampled frames
        # still gets assigned to the same track.
        _stride_cfg = Path(__file__).parents[3] / "models" / "bytetrack_stride.yaml"
        if frame_stride > 1 and _stride_cfg.exists():
            self._tracker_cfg = str(_stride_cfg)
        else:
            # Fall back to YOLO's bundled bytetrack.yaml
            import ultralytics

            self._tracker_cfg = str(
                Path(ultralytics.__file__).parent
                / "cfg"
                / "trackers"
                / "bytetrack.yaml"
            )

        # Motion detection setup
        prev_gray = None
        frames_skipped = 0
        use_motion = motion_threshold > 0

        frame_idx = 0
        start_time = time.time()
        stride_info = f" (stride={frame_stride})" if frame_stride > 1 else ""
        print(
            f"Tracking {video_path.name}: {total_frames} frames "
            f"@ {fps:.1f} fps{stride_info}"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame stride: skip frames not on the stride interval
            if frame_stride > 1 and frame_idx % frame_stride != 0:
                frame_idx += 1
                frames_skipped += 1
                continue

            h, w = frame.shape[:2]

            # Motion detection: skip frames with no significant change.
            # Always process the first 100 strided frames to let ByteTrack
            # establish tracks for stationary objects (parked vehicles).
            frames_processed = frame_idx // max(frame_stride, 1)
            process_frame = True
            if use_motion and prev_gray is not None and frames_processed > 100:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_ratio = np.count_nonzero(thresh) / thresh.size
                if motion_ratio < motion_threshold:
                    process_frame = False
                    frames_skipped += 1
                prev_gray = gray
            elif use_motion:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

            if process_frame:
                results = self.model.track(
                    frame,
                    conf=min_conf,
                    classes=all_classes,
                    persist=True,
                    verbose=False,
                    tracker=self._tracker_cfg,
                )

                for result in results:
                    if result.boxes.id is None:
                        continue
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        track_id = int(boxes.id[i])
                        cls_id = int(boxes.cls[i])
                        conf = float(boxes.conf[i])
                        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].cpu().numpy()]
                        area = (x2 - x1) * (y2 - y1)
                        ts = frame_idx / fps

                        det = {
                            "frame_idx": frame_idx,
                            "timestamp": ts,
                            "bbox": [x1, y1, x2, y2],
                            "conf": conf,
                            "area": area,
                            "cls_id": cls_id,
                        }

                        if cls_id == PERSON_CLASS and conf >= person_conf:
                            person_tracks[track_id].append(det)
                            # Cache best crop per person track for Re-ID merge
                            pw, ph = x2 - x1, y2 - y1
                            if (
                                conf > track_best_conf.get(track_id, 0.0)
                                and pw >= 20
                                and ph >= 30
                            ):
                                crop = frame[
                                    max(0, y1) : min(h, y2),
                                    max(0, x1) : min(w, x2),
                                ]
                                if crop.size > 0:
                                    track_best_crop[track_id] = crop.copy()
                                    track_best_conf[track_id] = conf
                        elif cls_id in VEHICLE_CLASSES and conf >= vehicle_conf:
                            vehicle_tracks[track_id].append(det)
                        else:
                            continue

                        # Cache frame for best-crop selection later
                        if frame_idx not in frame_cache:
                            frame_cache[frame_idx] = frame.copy()

                # Limit frame cache size (keep most recent)
                if len(frame_cache) > frame_cache_limit:
                    oldest = min(frame_cache.keys())
                    del frame_cache[oldest]

            # Progress callback
            if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
                elapsed = time.time() - start_time
                fps_actual = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0

                if progress_callback:
                    progress_callback(
                        {
                            "frame": frame_idx,
                            "total": total_frames,
                            "persons": len(person_tracks),
                            "vehicles": len(vehicle_tracks),
                            "fps_processing": fps_actual,
                            "eta_seconds": eta,
                            "skipped": frames_skipped,
                        }
                    )
                else:
                    print(
                        f"  Frame {frame_idx}/{total_frames} — "
                        f"persons: {len(person_tracks)}, "
                        f"vehicles: {len(vehicle_tracks)}, "
                        f"skipped: {frames_skipped}, "
                        f"ETA: {eta:.0f}s"
                    )

            frame_idx += 1

        cap.release()
        print(
            f"Tracking complete. "
            f"Person tracks: {len(person_tracks)}, "
            f"Vehicle tracks: {len(vehicle_tracks)}"
        )
        # Post-processing thresholds
        min_vehicle_frames = 2

        # Step 0: minimal pre-filter — remove only zero-detection entries and
        # vehicle noise (vehicles are reliably tracked at stride=10; persons
        # moving quickly may produce many 1-det ByteTrack tracks that we want
        # to chain via Re-ID in the merge step below, so we let them through).
        person_tracks = {tid: dets for tid, dets in person_tracks.items() if dets}
        vehicle_tracks = {
            tid: dets
            for tid, dets in vehicle_tracks.items()
            if len(dets) >= min_vehicle_frames
        }
        print(
            f"After pre-filter: "
            f"persons={len(person_tracks)}, "
            f"vehicles={len(vehicle_tracks)}"
        )

        # Step 1a: Split vehicle tracks with large internal gaps.
        # (Person gap split is deferred to after fragment merge — Step 2b.)
        vehicle_tracks = self._split_gapped_tracks(
            vehicle_tracks, fps=fps, max_gap_secs=300.0
        )

        # Step 1a2: Split vehicle tracks where bbox area changes drastically
        # (different vehicle replacing one that left the same parking spot).
        pre_split = len(vehicle_tracks)
        vehicle_tracks = self._split_bbox_discontinuity(
            vehicle_tracks, area_change_ratio=0.5
        )
        if len(vehicle_tracks) != pre_split:
            print(
                f"Bbox discontinuity split: {pre_split} -> "
                f"{len(vehicle_tracks)} vehicle tracks"
            )

        # Step 1b: dedup concurrent tracks — ByteTrack sometimes creates two IDs
        # for the same person/vehicle in the same frame (noisy detections).
        # Pure bbox IoU check; no frame cache needed.
        person_tracks = self._dedup_concurrent_tracks(person_tracks, iou_threshold=0.25)
        vehicle_tracks = self._dedup_concurrent_tracks(
            vehicle_tracks, iou_threshold=0.25
        )

        # Step 2: Merge fragmented tracks (same entity, different ByteTrack IDs).
        # This includes chaining the many 1-detection tentative tracks that
        # ByteTrack creates for fast-moving people (IoU=0 between stride frames)
        # into a single coherent track using Re-ID embedding similarity.
        person_tracks = self._merge_fragmented_tracks(
            person_tracks,
            frame_cache,
            entity_type="person",
            iou_threshold=0.2,
            sim_threshold=0.65,
            track_best_crop=track_best_crop,
            fps=fps,
            frame_stride=frame_stride,
        )
        vehicle_tracks = self._merge_fragmented_tracks(
            vehicle_tracks,
            frame_cache,
            entity_type="vehicle",
            iou_threshold=0.3,
            sim_threshold=0.72,
            fps=fps,
            frame_stride=frame_stride,
        )
        print(
            f"After merging fragments: "
            f"persons={len(person_tracks)}, "
            f"vehicles={len(vehicle_tracks)}"
        )

        # Step 2b: Split person tracks with large internal gaps.
        # Done AFTER fragment merge so that 1-detection tentative tracks
        # are chained into coherent tracks first, then real absence gaps
        # (>15s off-camera) produce separate sighting records.
        person_tracks = self._split_gapped_tracks(
            person_tracks, fps=fps, max_gap_secs=15.0
        )

        # Step 3: Final noise gate — applied AFTER merge.
        # Require at least 2 detections for persons to filter out single-frame
        # false positives (plant pots, shadows, etc.).
        person_tracks = {
            tid: dets for tid, dets in person_tracks.items() if len(dets) >= 2
        }
        vehicle_tracks = {
            tid: dets
            for tid, dets in vehicle_tracks.items()
            if len(dets) >= min_vehicle_frames
        }
        print(
            f"After final noise gate: "
            f"persons={len(person_tracks)}, "
            f"vehicles={len(vehicle_tracks)}"
        )

        # Build summaries for each track — pass video_path so frames
        # evicted from cache can be re-read from disk (critical for
        # long videos where cache_limit << total frames).
        persons = self._build_person_summaries(
            person_tracks,
            frame_cache,
            fps,
            output_dir,
            video_path=str(video_path),
        )
        vehicles = self._build_vehicle_summaries(
            vehicle_tracks,
            frame_cache,
            fps,
            output_dir,
            plate_conf,
            video_path=str(video_path),
        )

        # Persist to database
        self._save_to_db(persons, vehicles, str(video_path), camera_label, run_id)

        # Print summary
        print(f"\n{'=' * 50}")
        print("TRACKING SUMMARY")
        print("=" * 50)
        print(f"Camera: {camera_label}")
        print(f"People tracked: {len(persons)}")
        for p in persons:
            print(
                f"  Person {p['track_id']}: {p['description']} "
                f"(frames {p['first_frame']}-{p['last_frame']})"
            )
        print(f"Vehicles tracked: {len(vehicles)}")
        for v in vehicles:
            plate = f", plate: {v['plate_text']}" if v.get("plate_text") else ""
            print(
                f"  Vehicle {v['track_id']}: {v['description']}{plate} "
                f"(frames {v['first_frame']}-{v['last_frame']})"
            )

        return {
            "camera_label": camera_label,
            "video_path": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "persons": persons,
            "vehicles": vehicles,
        }

    def _split_gapped_tracks(
        self,
        tracks: Dict[int, List[dict]],
        fps: float,
        max_gap_secs: float = 30.0,
    ) -> Dict[int, List[dict]]:
        """Split ByteTrack tracks that have large internal detection gaps.

        ByteTrack can keep the same track ID alive across long absences,
        causing two different people who appear sequentially in the same
        location to share one ID. We split any track whose consecutive
        detections are more than max_gap_secs apart into separate tracks.
        New IDs use a large offset to avoid colliding with real IDs.
        """
        result: Dict[int, List[dict]] = {}
        next_synthetic_id = max(tracks.keys(), default=0) + 10000

        for tid, dets in tracks.items():
            sorted_dets = sorted(dets, key=lambda d: d["frame_idx"])
            segments: List[List[dict]] = []
            current_seg = [sorted_dets[0]]

            for det in sorted_dets[1:]:
                gap_secs = (det["frame_idx"] - current_seg[-1]["frame_idx"]) / fps
                if gap_secs > max_gap_secs:
                    segments.append(current_seg)
                    current_seg = [det]
                else:
                    current_seg.append(det)
            segments.append(current_seg)

            if len(segments) == 1:
                result[tid] = sorted_dets
            else:
                # First segment keeps the original ID
                result[tid] = segments[0]
                for seg in segments[1:]:
                    result[next_synthetic_id] = seg
                    next_synthetic_id += 1

        return result

    def _split_bbox_discontinuity(
        self,
        tracks: Dict[int, List[dict]],
        area_change_ratio: float = 0.4,
    ) -> Dict[int, List[dict]]:
        """Split vehicle tracks where the bbox area changes drastically.

        This catches ByteTrack linking two different vehicles that occupy
        a similar position sequentially (e.g. a car drives away and a
        different car parks in roughly the same spot).  A sustained drop
        or jump in bbox area indicates a different physical object.

        Uses the largest bbox per frame to avoid noise from multiple
        detections sharing the same track ID in a single frame.
        """
        result: Dict[int, List[dict]] = {}
        next_id = max(tracks.keys(), default=0) + 20000

        for tid, dets in tracks.items():
            sorted_dets = sorted(dets, key=lambda d: d["frame_idx"])

            # Compute the largest bbox area per frame
            frame_max_area: Dict[int, float] = {}
            for d in sorted_dets:
                fi = d["frame_idx"]
                area = (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                frame_max_area[fi] = max(frame_max_area.get(fi, 0), area)

            sorted_frames = sorted(frame_max_area.keys())
            if len(sorted_frames) < 10:
                result[tid] = sorted_dets
                continue

            # Find where rolling median area permanently drops/jumps.
            # Use window=15 to avoid false splits from brief fluctuations
            # (person walking past causes temporary bbox changes).
            split_frame = None
            window = 15
            for i in range(window, len(sorted_frames) - window):
                before = sorted(
                    [frame_max_area[sorted_frames[j]] for j in range(i - window, i)]
                )
                after = sorted(
                    [frame_max_area[sorted_frames[j]] for j in range(i, i + window)]
                )
                med_before = before[len(before) // 2]
                med_after = after[len(after) // 2]
                if med_before > 0 and med_after > 0:
                    ratio = min(med_before, med_after) / max(med_before, med_after)
                    if ratio < area_change_ratio:
                        split_frame = sorted_frames[i]
                        break

            if split_frame is None:
                result[tid] = sorted_dets
            else:
                seg_a = [d for d in sorted_dets if d["frame_idx"] < split_frame]
                seg_b = [d for d in sorted_dets if d["frame_idx"] >= split_frame]
                if seg_a and seg_b:
                    result[tid] = seg_a
                    result[next_id] = seg_b
                    next_id += 1
                else:
                    result[tid] = sorted_dets

        return result

    def _dedup_concurrent_tracks(
        self,
        tracks: Dict[int, List[dict]],
        iou_threshold: float = 0.4,
    ) -> Dict[int, List[dict]]:
        """Merge tracks that are active at the same time and overlap spatially.

        ByteTrack occasionally creates two IDs for the same entity in the same
        frames (e.g., when a detection is briefly lost and immediately re-found).
        Uses bounding box IoU only — no frame cache or embeddings needed.
        """
        if len(tracks) <= 1:
            return tracks

        track_info = {}
        for tid, dets in tracks.items():
            if not dets:
                continue
            frames = [d["frame_idx"] for d in dets]
            # Mean bbox across all dets: handles moving persons whose
            # best-frame positions differ between concurrent tracks
            mean_bbox = [
                int(sum(d["bbox"][i] for d in dets) / len(dets)) for i in range(4)
            ]
            track_info[tid] = {
                "first_frame": min(frames),
                "last_frame": max(frames),
                "avg_bbox": mean_bbox,
            }

        sorted_tids = sorted(
            track_info.keys(), key=lambda t: track_info[t]["first_frame"]
        )
        merged: Dict[int, List[dict]] = {}
        used: set = set()

        for i, tid_a in enumerate(sorted_tids):
            if tid_a in used:
                continue
            merged_dets = list(tracks[tid_a])
            used.add(tid_a)
            info_a = track_info[tid_a]

            for tid_b in sorted_tids[i + 1 :]:
                if tid_b in used:
                    continue
                info_b = track_info[tid_b]

                # Must overlap in time
                if info_b["first_frame"] > info_a["last_frame"]:
                    continue
                if info_a["first_frame"] > info_b["last_frame"]:
                    continue

                # Check spatial overlap
                ba = info_a["avg_bbox"]
                bb = info_b["avg_bbox"]
                ix1 = max(ba[0], bb[0])
                iy1 = max(ba[1], bb[1])
                ix2 = min(ba[2], bb[2])
                iy2 = min(ba[3], bb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (ba[2] - ba[0]) * (ba[3] - ba[1])
                area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0

                if iou >= iou_threshold:
                    merged_dets.extend(tracks[tid_b])
                    used.add(tid_b)
                    info_a["last_frame"] = max(
                        info_a["last_frame"], info_b["last_frame"]
                    )

            merged[tid_a] = merged_dets

        return merged

    def _merge_fragmented_tracks(
        self,
        tracks: Dict[int, List[dict]],
        frame_cache: Dict[int, np.ndarray],
        entity_type: str = "person",
        iou_threshold: float = 0.3,
        sim_threshold: float = 0.55,
        track_best_crop: Optional[Dict[int, np.ndarray]] = None,
        fps: float = 25.0,
        frame_stride: int = 1,
    ) -> Dict[int, List[dict]]:
        """Merge ByteTrack fragments that are the same entity.

        Two tracks are merged if their appearance embeddings are similar
        enough (Re-ID match) OR their bounding boxes sufficiently overlap.
        Uses cached per-track crops (track_best_crop) so early-video tracks
        are not skipped due to frame_cache eviction.

        Temporal gap constraint: the embedding-only similarity threshold
        scales up with the time gap between tracks, preventing different
        people who look similar (e.g. both wearing blue) from being merged
        across large time gaps.
          gap < 30s  → sim >= 0.62  (brief occlusion / walk across frame)
          gap 30-90s → sim >= 0.75  (conservative)
          gap > 90s  → embedding-only merge disabled; IoU required
        """
        if len(tracks) <= 1:
            return tracks

        # Build per-track summaries for comparison
        track_info = {}
        for tid, dets in tracks.items():
            best = max(dets, key=lambda d: d["area"])

            # Try to get a crop: first from track_best_crop cache,
            # then fall back to frame_cache reconstruction
            crop = None
            if track_best_crop and tid in track_best_crop:
                crop = track_best_crop[tid]
            else:
                frame = frame_cache.get(best["frame_idx"])
                if frame is not None:
                    x1, y1, x2, y2 = best["bbox"]
                    fh, fw = frame.shape[:2]
                    pad = 5 if entity_type == "person" else 10
                    crop = frame[
                        max(0, y1 - pad) : min(fh, y2 + pad),
                        max(0, x1 - pad) : min(fw, x2 + pad),
                    ]

            emb = (
                self.extract_features(crop)
                if (crop is not None and crop.size > 0)
                else None
            )

            track_info[tid] = {
                "first_frame": min(d["frame_idx"] for d in dets),
                "last_frame": max(d["frame_idx"] for d in dets),
                "first_ts": min(d["timestamp"] for d in dets),
                "last_ts": max(d["timestamp"] for d in dets),
                "avg_bbox": best["bbox"],
                "embedding": emb,
            }

        # Sort tracks by first appearance
        sorted_tids = sorted(
            track_info.keys(), key=lambda t: track_info[t]["first_frame"]
        )

        # Greedy merge
        merged = {}
        used = set()

        for i, tid_a in enumerate(sorted_tids):
            if tid_a in used:
                continue

            merged_dets = list(tracks[tid_a])
            used.add(tid_a)
            info_a = track_info[tid_a]

            for tid_b in sorted_tids[i + 1 :]:
                if tid_b in used:
                    continue
                info_b = track_info[tid_b]

                # Compute embedding similarity when both are available
                sim: float = 0.0
                has_sim = (
                    info_a["embedding"] is not None and info_b["embedding"] is not None
                )
                if has_sim:
                    sim = float(np.dot(info_a["embedding"], info_b["embedding"]))
                    # Early exit only if below the minimum possible threshold
                    # (0.62 for short gaps).  Using sim_threshold (0.65) here
                    # would incorrectly skip pairs whose gap-adjusted threshold
                    # is 0.62 and whose sim falls between 0.62-0.65.
                    if sim < 0.62:
                        continue

                # Check spatial overlap (avg bboxes)
                ba = info_a["avg_bbox"]
                bb = info_b["avg_bbox"]
                ix1 = max(ba[0], bb[0])
                iy1 = max(ba[1], bb[1])
                ix2 = min(ba[2], bb[2])
                iy2 = min(ba[3], bb[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (ba[2] - ba[0]) * (ba[3] - ba[1])
                area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0

                # Temporal gap using actual timestamps (immune to
                # motion-detection frame skipping).
                gap_secs = info_b["first_ts"] - info_a["last_ts"]

                # Hard cap: vehicles get a longer window because stationary
                # cars are visible the whole clip but motion detection
                # creates artificial gaps.  Persons stay conservative.
                max_gap = 600 if entity_type == "vehicle" else 90
                if gap_secs > max_gap:
                    continue

                # Scale threshold with gap: short → 0.62, medium → 0.75
                if gap_secs < 30:
                    emb_threshold = 0.62
                else:
                    emb_threshold = 0.75

                iou_needed = iou_threshold if has_sim else 0.5

                # Prevent merging vehicles with very different bbox sizes
                # (e.g. a large BMW and a small Mini in the same parking spot).
                area_ratio = (
                    min(area_a, area_b) / max(area_a, area_b)
                    if max(area_a, area_b) > 0
                    else 1.0
                )
                if entity_type == "vehicle" and area_ratio < 0.4:
                    continue

                # Persons: merge if IoU OR embedding match (brief occlusion
                # can change bbox but appearance stays the same).
                # Vehicles: require BOTH IoU AND embedding match — two
                # different vehicles can share the same parking spot
                # sequentially, so spatial overlap alone isn't enough.
                if entity_type == "vehicle":
                    if has_sim:
                        do_merge = iou >= iou_needed and sim >= emb_threshold
                    else:
                        do_merge = iou >= 0.7  # very high IoU if no embedding
                else:
                    do_merge = iou >= iou_needed or (has_sim and sim >= emb_threshold)
                if do_merge:
                    # Merge B into A
                    merged_dets.extend(tracks[tid_b])
                    used.add(tid_b)
                    # Update info_a to reflect merged track
                    info_a["last_frame"] = max(
                        info_a["last_frame"], info_b["last_frame"]
                    )
                    info_a["last_ts"] = max(info_a["last_ts"], info_b["last_ts"])

            merged[tid_a] = merged_dets

        return merged

    @staticmethod
    def _read_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """Read a single frame from video file (fallback for cache miss)."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def _get_frame(
        self,
        frame_idx: int,
        frame_cache: Dict[int, np.ndarray],
        video_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Get frame from cache, falling back to video file if needed."""
        frame = frame_cache.get(frame_idx)
        if frame is None and video_path:
            frame = self._read_frame(video_path, frame_idx)
        return frame

    def _build_person_summaries(
        self,
        tracks: Dict[int, List[dict]],
        frame_cache: Dict[int, np.ndarray],
        fps: float,
        output_dir: Path,
        video_path: Optional[str] = None,
    ) -> List[dict]:
        """For each person track, pick the best crop and describe them."""
        summaries = []
        for track_id, dets in sorted(tracks.items()):
            best = max(dets, key=lambda d: d["area"])
            frame = self._get_frame(best["frame_idx"], frame_cache, video_path)
            if frame is None:
                continue

            x1, y1, x2, y2 = best["bbox"]
            h, w = frame.shape[:2]
            pad = 5
            crop = frame[
                max(0, y1 - pad) : min(h, y2 + pad),
                max(0, x1 - pad) : min(w, x2 + pad),
            ]
            if crop.size == 0:
                continue

            # Multi-frame colour voting: sample up to 5 largest crops and majority-vote
            # on upper/lower colour to reduce single-frame noise
            top_dets = sorted(dets, key=lambda d: d["area"], reverse=True)[:5]
            upper_votes: List[str] = []
            lower_votes: List[str] = []
            for det in top_dets:
                f = self._get_frame(det["frame_idx"], frame_cache, video_path)
                if f is None:
                    continue
                bx1, by1, bx2, by2 = det["bbox"]
                fh, fw = f.shape[:2]
                c = f[
                    max(0, by1 - pad) : min(fh, by2 + pad),
                    max(0, bx1 - pad) : min(fw, bx2 + pad),
                ]
                if c.size == 0:
                    continue
                d = describe_person(c)
                if d["upper_color"] != "unknown":
                    upper_votes.append(d["upper_color"])
                if d["lower_color"] != "unknown":
                    lower_votes.append(d["lower_color"])

            def _majority(votes: List[str], fallback: str) -> str:
                if not votes:
                    return fallback
                from collections import Counter  # noqa: PLC0415

                return Counter(votes).most_common(1)[0][0]

            desc = describe_person(crop)
            desc["upper_color"] = _majority(upper_votes, desc["upper_color"])
            desc["lower_color"] = _majority(lower_votes, desc["lower_color"])
            desc["description"] = (
                f"{desc['upper_color']} top, {desc['lower_color']} bottom"
            )

            embedding = self.extract_features(crop)

            # Save images
            crop_path = output_dir / f"person_T{track_id}_crop.jpg"
            full_path = output_dir / f"person_T{track_id}_full.jpg"

            # Upscale small crops
            save_crop = crop.copy()
            scale = max(1, 200 // max(save_crop.shape[0], 1))
            if scale > 1:
                save_crop = cv2.resize(
                    save_crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            cv2.imwrite(str(crop_path), save_crop)

            annotated = frame.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"Person {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imwrite(str(full_path), annotated)

            first_det = min(dets, key=lambda d: d["frame_idx"])
            last_det = max(dets, key=lambda d: d["frame_idx"])

            summaries.append(
                {
                    "track_id": track_id,
                    "entity_type": "person",
                    "description": desc["description"],
                    "upper_color": desc["upper_color"],
                    "lower_color": desc["lower_color"],
                    "color": None,
                    "vehicle_type": None,
                    "plate_text": None,
                    "plate_confidence": None,
                    "confidence": best["conf"],
                    "bbox": best["bbox"],
                    "first_frame": first_det["frame_idx"],
                    "last_frame": last_det["frame_idx"],
                    "first_ts": first_det["timestamp"],
                    "last_ts": last_det["timestamp"],
                    "num_detections": len(dets),
                    "embedding": embedding,
                    "crop_path": str(crop_path),
                    "full_frame_path": str(full_path),
                    "frame_history": [
                        {
                            "frame_idx": d["frame_idx"],
                            "timestamp": d["timestamp"],
                            "bbox": d["bbox"],
                        }
                        for d in dets
                    ],
                }
            )
        return summaries

    def _build_vehicle_summaries(
        self,
        tracks: Dict[int, List[dict]],
        frame_cache: Dict[int, np.ndarray],
        fps: float,
        output_dir: Path,
        plate_conf: float,
        video_path: Optional[str] = None,
    ) -> List[dict]:
        """For each vehicle track, pick the best crop and describe colour/plate."""
        summaries = []
        for track_id, dets in sorted(tracks.items()):
            # Use the detection at the median frame of the track
            # rather than max area — avoids transition frames where
            # two vehicles overlap (e.g. one leaving, another arriving).
            sorted_dets = sorted(dets, key=lambda d: d["frame_idx"])
            median_idx = len(sorted_dets) // 2
            best = None
            for offset in range(len(sorted_dets)):
                for candidate_idx in [median_idx + offset, median_idx - offset]:
                    if 0 <= candidate_idx < len(sorted_dets):
                        if sorted_dets[candidate_idx]["frame_idx"] in frame_cache:
                            best = sorted_dets[candidate_idx]
                            break
                if best is not None:
                    break
            if best is None:
                best = max(dets, key=lambda d: d["area"])
            frame = self._get_frame(best["frame_idx"], frame_cache, video_path)
            if frame is None:
                continue

            x1, y1, x2, y2 = best["bbox"]
            h, w = frame.shape[:2]
            pad = 10
            crop = frame[
                max(0, y1 - pad) : min(h, y2 + pad),
                max(0, x1 - pad) : min(w, x2 + pad),
            ]
            if crop.size == 0:
                continue

            vtype = VEHICLE_CLASSES.get(best["cls_id"], "car")
            vdesc = describe_vehicle(crop, vehicle_type=vtype)

            # Plate detection on the best crop
            plate_box, plate_c, plate_text = self._detect_plate(crop, conf=plate_conf)

            embedding = self.extract_features(crop)

            # Save images
            crop_path = output_dir / f"vehicle_T{track_id}_crop.jpg"
            full_path = output_dir / f"vehicle_T{track_id}_full.jpg"

            save_crop = crop.copy()
            scale = max(1, 200 // max(save_crop.shape[0], 1))
            if scale > 1:
                save_crop = cv2.resize(
                    save_crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            cv2.imwrite(str(crop_path), save_crop)

            annotated = frame.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)
            label = f"Vehicle {track_id}: {vdesc['description']}"
            if plate_text:
                label += f" [{plate_text}]"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            if plate_box:
                # Offset plate box to frame coordinates
                px1 = max(0, x1 - pad) + plate_box[0]
                py1 = max(0, y1 - pad) + plate_box[1]
                px2 = max(0, x1 - pad) + plate_box[2]
                py2 = max(0, y1 - pad) + plate_box[3]
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.imwrite(str(full_path), annotated)

            first_det = min(dets, key=lambda d: d["frame_idx"])
            last_det = max(dets, key=lambda d: d["frame_idx"])

            summaries.append(
                {
                    "track_id": track_id,
                    "entity_type": "vehicle",
                    "description": vdesc["description"],
                    "upper_color": None,
                    "lower_color": None,
                    "color": vdesc["color"],
                    "vehicle_type": vtype,
                    "plate_text": plate_text,
                    "plate_confidence": plate_c if plate_box else None,
                    "confidence": best["conf"],
                    "bbox": best["bbox"],
                    "first_frame": first_det["frame_idx"],
                    "last_frame": last_det["frame_idx"],
                    "first_ts": first_det["timestamp"],
                    "last_ts": last_det["timestamp"],
                    "num_detections": len(dets),
                    "embedding": embedding,
                    "crop_path": str(crop_path),
                    "full_frame_path": str(full_path),
                    "frame_history": [
                        {
                            "frame_idx": d["frame_idx"],
                            "timestamp": d["timestamp"],
                            "bbox": d["bbox"],
                        }
                        for d in dets
                    ],
                }
            )
        return summaries

    def _save_to_db(
        self,
        persons: List[dict],
        vehicles: List[dict],
        video_path: str,
        camera_label: str,
        run_id: Optional[str] = None,
    ):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        for entity in persons + vehicles:
            c.execute(
                """
                INSERT INTO tracked_entities (
                    run_id, video_path, camera_label, entity_type, track_id,
                    first_frame, last_frame, first_ts, last_ts,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    confidence, color, upper_color, lower_color,
                    description, vehicle_type, plate_text,
                    plate_confidence, embedding, crop_path,
                    full_frame_path
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?
                )
            """,
                (
                    run_id,
                    video_path,
                    camera_label,
                    entity["entity_type"],
                    entity["track_id"],
                    entity["first_frame"],
                    entity["last_frame"],
                    entity["first_ts"],
                    entity["last_ts"],
                    entity["bbox"][0],
                    entity["bbox"][1],
                    entity["bbox"][2],
                    entity["bbox"][3],
                    entity["confidence"],
                    entity.get("color"),
                    entity.get("upper_color"),
                    entity.get("lower_color"),
                    entity["description"],
                    entity.get("vehicle_type"),
                    entity.get("plate_text"),
                    entity.get("plate_confidence"),
                    (
                        entity["embedding"].tobytes()
                        if entity.get("embedding") is not None
                        else None
                    ),
                    entity["crop_path"],
                    entity["full_frame_path"],
                ),
            )

            entity_id = c.lastrowid

            # Save per-frame bbox history
            for fh in entity.get("frame_history", []):
                c.execute(
                    """
                    INSERT INTO track_frames (
                        entity_id, frame_idx, timestamp,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entity_id,
                        fh["frame_idx"],
                        fh["timestamp"],
                        fh["bbox"][0],
                        fh["bbox"][1],
                        fh["bbox"][2],
                        fh["bbox"][3],
                    ),
                )

        conn.commit()
        conn.close()


def match_across_cameras(
    db_path: str,
    camera_a: str,
    camera_b: str,
    person_threshold: float = 0.55,
    vehicle_threshold: float = 0.65,
) -> Dict[str, List[dict]]:
    """Match tracked entities across two cameras using embeddings + descriptions.

    Returns dict with 'person_matches' and 'vehicle_matches'.
    Each match has entity info from both cameras plus a similarity score.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows_a = conn.execute(
        "SELECT * FROM tracked_entities WHERE camera_label = ?",
        (camera_a,),
    ).fetchall()
    rows_b = conn.execute(
        "SELECT * FROM tracked_entities WHERE camera_label = ?",
        (camera_b,),
    ).fetchall()
    conn.close()

    entities_a = [dict(r) for r in rows_a]
    entities_b = [dict(r) for r in rows_b]

    # Parse embeddings
    for e in entities_a + entities_b:
        if e["embedding"]:
            e["_emb"] = np.frombuffer(e["embedding"], dtype=np.float32)
            norm = np.linalg.norm(e["_emb"])
            if norm > 0:
                e["_emb"] = e["_emb"] / norm
        else:
            e["_emb"] = None

    person_matches = _match_entities(
        [e for e in entities_a if e["entity_type"] == "person"],
        [e for e in entities_b if e["entity_type"] == "person"],
        person_threshold,
    )
    vehicle_matches = _match_entities(
        [e for e in entities_a if e["entity_type"] == "vehicle"],
        [e for e in entities_b if e["entity_type"] == "vehicle"],
        vehicle_threshold,
    )

    return {
        "person_matches": person_matches,
        "vehicle_matches": vehicle_matches,
    }


def _match_entities(
    list_a: List[dict], list_b: List[dict], threshold: float
) -> List[dict]:
    """Find best matches between two lists using embedding + temporal proximity.

    Temporal bonus: sightings close in time get a score boost (up to +0.10),
    which helps distinguish between people wearing similar clothes. The bonus
    decays over 5 minutes and vanishes entirely beyond that.
    """
    # Build all candidate pairs with scores, then assign greedily
    # from highest score to lowest (prevents suboptimal greedy order).
    all_pairs: list = []
    for i, ea in enumerate(list_a):
        if ea["_emb"] is None:
            continue
        for j, eb in enumerate(list_b):
            if eb["_emb"] is None:
                continue
            sim = float(np.dot(ea["_emb"], eb["_emb"]))

            time_gap = min(
                abs(ea["first_ts"] - eb["first_ts"]),
                abs(ea["last_ts"] - eb["first_ts"]),
                abs(ea["first_ts"] - eb["last_ts"]),
                abs(ea["last_ts"] - eb["last_ts"]),
            )
            if time_gap < 300:
                temporal_bonus = 0.15 * (1.0 - time_gap / 300.0)
            else:
                temporal_bonus = 0.0

            score = sim + temporal_bonus
            # Short cross-camera gaps provide strong temporal evidence
            # that two sightings are the same person.  Reduce the
            # effective threshold proportionally — larger reductions
            # for higher base thresholds because the gap between
            # embedding similarity and threshold is wider at 0.90.
            effective_threshold = threshold
            if time_gap < 60:
                if threshold <= 0.80:
                    effective_threshold = threshold - 0.05
                else:
                    effective_threshold = threshold - 0.15
            elif time_gap < 120:
                if threshold <= 0.80:
                    effective_threshold = threshold - 0.03
                else:
                    effective_threshold = threshold - 0.10
            if score < effective_threshold:
                continue
            all_pairs.append((score, sim, i, j, ea, eb))

    # Sort by score descending — best pairs assigned first
    all_pairs.sort(key=lambda p: p[0], reverse=True)

    matches = []
    used_a: set = set()
    used_b: set = set()

    for score, sim, i, j, ea, eb in all_pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)

        def _entity_dict(e: dict) -> dict:
            return {
                "id": e["id"],
                "track_id": e["track_id"],
                "camera": e["camera_label"],
                "entity_type": e["entity_type"],
                "description": e["description"],
                "color": e.get("color"),
                "upper_color": e.get("upper_color"),
                "lower_color": e.get("lower_color"),
                "vehicle_type": e.get("vehicle_type"),
                "plate_text": e.get("plate_text"),
                "crop_path": e["crop_path"],
                "first_ts": e["first_ts"],
                "last_ts": e["last_ts"],
                "first_frame": e["first_frame"],
                "last_frame": e["last_frame"],
            }

        matches.append(
            {
                "entity_a": _entity_dict(ea),
                "entity_b": _entity_dict(eb),
                "similarity": round(sim, 3),
            }
        )

    matches.sort(key=lambda m: m["similarity"], reverse=True)
    return matches


class UnionFind:
    """Union-Find data structure for clustering entities via transitive relationships.

    WARNING: Creates transitive merges (A→B, B→C implies A=C) which can incorrectly
    group different entities. Use with caution and higher thresholds.
    """

    def __init__(self):
        self.parent = {}

    def find(self, x):
        """Find root of x with path compression."""
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Merge sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

    def get_clusters(self) -> Dict[int, List]:
        """Return all clusters as {root: [members]}."""
        clusters = {}
        for node in self.parent:
            root = self.find(node)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(node)
        return clusters


def _adaptive_identity_thresholds(db_path: str) -> Tuple[float, float]:
    """Return (person_threshold, vehicle_threshold) tuned for scalability sets.

    Falls back to the historical defaults when the DB path does not map to a
    known scalability duration folder.
    """
    tuned = {
        "1h": (0.80, 0.80),
        "3h": (0.80, 0.80),
        "5h": (0.80, 0.80),
        "8h": (0.90, 0.80),
        "11h": (0.90, 0.80),
    }
    def _detect_duration(path_like: str) -> Optional[str]:
        parts = {p.lower() for p in Path(path_like).parts}
        for duration in tuned:
            if duration in parts:
                return duration
        return None

    duration = _detect_duration(db_path)
    if duration:
        return tuned[duration]

    # If db_path is generic (for example data/tracker.db), infer duration
    # from tracked entity paths persisted in the DB.
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            """
            SELECT video_path, crop_path
            FROM tracked_entities
            WHERE (video_path IS NOT NULL AND video_path != '')
               OR (crop_path IS NOT NULL AND crop_path != '')
            LIMIT 1
            """
        ).fetchone()
        conn.close()

        if row:
            for candidate in row:
                if candidate:
                    duration = _detect_duration(candidate)
                    if duration:
                        return tuned[duration]
    except Exception:
        pass

    return (0.50, 0.60)


def build_unified_identities(
    db_path: str,
    camera_a: str,
    camera_b: str,
    person_threshold: Optional[float] = None,
    vehicle_threshold: Optional[float] = None,
) -> Dict[str, List[dict]]:
    """Build unified identities across cameras using Union-Find clustering.

    Uses transitive matching: if Track A matches Track B, and Track B matches Track C,
    then all three are merged into one identity even if A and C don't directly match.

    WARNING: This can cause false merges (grouping different people together) if
    similarity threshold is too low. Use higher thresholds than 1-to-1 matching.

    Thresholds:
      - If provided, explicit values are used.
      - If None, thresholds are selected from db_path duration folders
        (1h/3h/5h/8h/11h), with fallback to 0.50/0.60.

    Returns a dict with 'persons' and 'vehicles', each a list of unified identities.
    Each identity has:
      - unified_id: Sequential ID (1, 2, 3...)
      - sightings: List of dicts (camera, entity_id, track_id, etc.)
      - similarity: Average match confidence (if cross-camera)
      - matched: True if entity appears on multiple cameras
    """
    adaptive_person, adaptive_vehicle = _adaptive_identity_thresholds(db_path)
    if person_threshold is None:
        person_threshold = adaptive_person
    if vehicle_threshold is None:
        vehicle_threshold = adaptive_vehicle

    # Get all pairwise matches across cameras
    matches = match_across_cameras(
        db_path, camera_a, camera_b, person_threshold, vehicle_threshold
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get entities from the two cameras only (not stale data from other runs)
    all_entities = conn.execute(
        "SELECT * FROM tracked_entities WHERE camera_label IN (?, ?)",
        (camera_a, camera_b),
    ).fetchall()
    conn.close()

    # Group by entity type
    all_entities_dict = {r["id"]: dict(r) for r in all_entities}

    # Compute per-camera clip duration from max timestamp across all entities.
    # Used to extend stationary vehicle bars to clip boundaries.
    cam_clip_end: Dict[str, float] = {}
    cam_clip_start: Dict[str, float] = {}
    for e in all_entities_dict.values():
        cam = e["camera_label"]
        cam_clip_end[cam] = max(cam_clip_end.get(cam, 0), e.get("last_ts") or 0)
        ts = e.get("first_ts")
        if ts is not None:
            cam_clip_start[cam] = min(cam_clip_start.get(cam, ts), ts)

    def _build_identities_union_find(
        match_list: List[dict], entity_type: str
    ) -> List[dict]:
        """Build identities using Union-Find clustering with transitive merging."""
        # Create Union-Find structure
        uf = UnionFind()

        # Pre-compute best within-camera similarity for each entity
        # so we can reject cross-camera matches that are weaker than
        # an entity's same-camera partner (prevents pairing two
        # different people who happen to be temporally close across
        # cameras when one of them has a strong same-camera match).
        _wc_embs: Dict[int, Optional[np.ndarray]] = {}
        _wc_cam: Dict[int, str] = {}
        for eid, e in all_entities_dict.items():
            if e["entity_type"] != entity_type:
                continue
            _wc_cam[eid] = e["camera_label"]
            raw = e.get("embedding")
            if raw is not None:
                arr = np.frombuffer(raw, dtype=np.float32).copy()
                n = np.linalg.norm(arr)
                _wc_embs[eid] = arr / n if n > 0 else None
            else:
                _wc_embs[eid] = None

        best_wc_sim: Dict[int, float] = {}
        _wc_ts: Dict[int, float] = {}
        for eid, e in all_entities_dict.items():
            if e["entity_type"] == entity_type:
                _wc_ts[eid] = e.get("first_ts") or 0
        eids_by_cam: Dict[str, list] = {}
        for eid in _wc_embs:
            eids_by_cam.setdefault(_wc_cam[eid], []).append(eid)
        for cam_eids in eids_by_cam.values():
            for i, ea in enumerate(cam_eids):
                for eb in cam_eids[i + 1 :]:
                    if _wc_embs[ea] is None or _wc_embs[eb] is None:
                        continue
                    # Only count temporally close pairs (<120s)
                    # as priority blockers.  Long-gap within-camera
                    # matches may be false (different people in
                    # similar clothes) and shouldn't block valid
                    # cross-camera links.
                    tg = abs(_wc_ts.get(ea, 0) - _wc_ts.get(eb, 0))
                    if tg > 120:
                        continue
                    wt = 0.55
                    s = float(np.dot(_wc_embs[ea], _wc_embs[eb]))
                    if s < wt:
                        continue  # wouldn't merge, don't count
                    if s > best_wc_sim.get(ea, 0):
                        best_wc_sim[ea] = s
                    if s > best_wc_sim.get(eb, 0):
                        best_wc_sim[eb] = s

        # Union cross-camera matched pairs, skipping any match
        # where either entity has a stronger within-camera partner
        # that belongs to a DIFFERENT identity group.  If the strong
        # partner is already in the same UF group as the candidate,
        # it shouldn't block — they're the same person.
        _wc_best_partner: Dict[int, int] = {}
        for cam_eids in eids_by_cam.values():
            for i, ea in enumerate(cam_eids):
                for eb in cam_eids[i + 1 :]:
                    if _wc_embs[ea] is None or _wc_embs[eb] is None:
                        continue
                    tg = abs(_wc_ts.get(ea, 0) - _wc_ts.get(eb, 0))
                    if tg > 120:
                        continue
                    s = float(np.dot(_wc_embs[ea], _wc_embs[eb]))
                    if s > best_wc_sim.get(ea, 0):
                        _wc_best_partner[ea] = eb
                    if s > best_wc_sim.get(eb, 0):
                        _wc_best_partner[eb] = ea

        for m in match_list:
            eid_a = m["entity_a"]["id"]
            eid_b = m["entity_b"]["id"]
            cross_sim = m["similarity"]
            wc_a = best_wc_sim.get(eid_a, 0)
            wc_b = best_wc_sim.get(eid_b, 0)
            margin = 0.08
            # Only block if the strong within-camera partner is in
            # a different group — if partner is already merged with
            # the candidate, it shouldn't prevent cross-camera linking.
            block = False
            if wc_a > cross_sim + margin:
                partner = _wc_best_partner.get(eid_a)
                if partner is not None and uf.find(partner) != uf.find(eid_a):
                    # Don't block if the partner would also match
                    # the cross-camera entity (they're all the same
                    # person, just not yet merged).
                    partner_sim = 0.0
                    if (
                        _wc_embs.get(partner) is not None
                        and _wc_embs.get(eid_b) is not None
                    ):
                        partner_sim = float(np.dot(_wc_embs[partner], _wc_embs[eid_b]))
                    if partner_sim < 0.45:
                        block = True
            if wc_b > cross_sim + margin:
                partner = _wc_best_partner.get(eid_b)
                if partner is not None and uf.find(partner) != uf.find(eid_b):
                    partner_sim = 0.0
                    if (
                        _wc_embs.get(partner) is not None
                        and _wc_embs.get(eid_a) is not None
                    ):
                        partner_sim = float(np.dot(_wc_embs[partner], _wc_embs[eid_a]))
                    if partner_sim < 0.45:
                        block = True
            if block:
                continue
            # Time-proximity gate: cross-camera matches over long
            # gaps need higher similarity.  A person walking between
            # cameras takes < 2 minutes.
            if entity_type == "person":
                ts_a = _wc_ts.get(eid_a, 0)
                ts_b = _wc_ts.get(eid_b, 0)
                cross_gap = abs(ts_a - ts_b)
                if cross_gap > 600 and cross_sim < 0.70:
                    continue
                if cross_gap > 120 and cross_sim < 0.55:
                    continue
            uf.union(eid_a, eid_b)

        # Get all entities of this type, filtering out noise (0-duration singletons)
        vehicle_median_area_by_cam: Dict[str, float] = {}
        if entity_type == "vehicle":
            area_by_cam: Dict[str, List[float]] = {}
            for e in all_entities_dict.values():
                if e["entity_type"] != "vehicle":
                    continue
                dur = (e.get("last_ts") or 0) - (e.get("first_ts") or 0)
                if dur < 300:
                    continue
                w = max(0.0, (e.get("bbox_x2") or 0) - (e.get("bbox_x1") or 0))
                h = max(0.0, (e.get("bbox_y2") or 0) - (e.get("bbox_y1") or 0))
                area_by_cam.setdefault(e["camera_label"], []).append(w * h)
            for cam, vals in area_by_cam.items():
                if vals:
                    vehicle_median_area_by_cam[cam] = float(np.median(vals))

        type_entities = {}
        for eid, e in all_entities_dict.items():
            if e["entity_type"] != entity_type:
                continue
            # Skip noise tracks:
            # - Person: 0-duration (single-frame false positives)
            # - Vehicle: short fragments (<120s) are usually detection
            #   noise during transitions (someone walking past a parked
            #   car), not real vehicles.
            dur = (e["last_ts"] or 0) - (e["first_ts"] or 0)
            if dur < 0.1 and entity_type == "person":
                continue
            if entity_type == "vehicle" and dur < 180:
                continue
            if entity_type == "vehicle":
                cam = e["camera_label"]
                clip_start = cam_clip_start.get(cam, 0)
                clip_dur = cam_clip_end.get(cam, 0) - clip_start
                w = max(0.0, (e.get("bbox_x2") or 0) - (e.get("bbox_x1") or 0))
                h = max(0.0, (e.get("bbox_y2") or 0) - (e.get("bbox_y1") or 0))
                area = w * h
                median_area = vehicle_median_area_by_cam.get(cam, 0.0)
                # Drop tiny late-start fragments that are usually false positives
                # caused by foreground motion over parked vehicles.
                # Frame-edge detections (bbox near y=0 or y=frame_max) use a
                # stricter area threshold because they are often partial/blurry.
                if clip_dur > 0 and median_area > 0:
                    late_start = (e.get("first_ts") or 0) > (clip_start + clip_dur * 0.20)
                    shortish = dur < (clip_dur * 0.80)
                    y1 = e.get("bbox_y1") or 0
                    y2 = e.get("bbox_y2") or 0
                    near_edge = y1 < 50 or y2 > 1030
                    tiny = area < (median_area * (0.35 if near_edge else 0.10))
                    if late_start and shortish and tiny:
                        continue
            if dur < 0.1 and e.get("description", "").startswith("unknown"):
                continue
            type_entities[eid] = e

        # Deduplicate exact-timestamp duplicates (DB double-insert bug).
        # Include bbox to avoid merging different objects at the same time.
        seen_keys: Dict[tuple, int] = {}
        for eid, e in type_entities.items():
            # Round bbox to 1px to handle float noise
            bx = (
                round(e.get("bbox_x1") or 0),
                round(e.get("bbox_y1") or 0),
                round(e.get("bbox_x2") or 0),
                round(e.get("bbox_y2") or 0),
            )
            key = (e["camera_label"], e["first_ts"], e["last_ts"], bx)
            if key in seen_keys:
                uf.union(seen_keys[key], eid)
            else:
                seen_keys[key] = eid

        # Within-camera identity merging — unify fragments of the same
        # entity that the tracker pipeline kept separate.
        by_camera: Dict[str, list] = {}
        for eid, e in type_entities.items():
            by_camera.setdefault(e["camera_label"], []).append((eid, e))

        # Track entities that have cross-camera matches so
        # within-camera merges don't accidentally collapse two
        # different cross-camera groups together.
        cross_matched: set = set()
        for m in match_list:
            ea, eb = m["entity_a"]["id"], m["entity_b"]["id"]
            if uf.find(ea) == uf.find(eb):
                # This pair was actually merged (not blocked)
                cross_matched.add(ea)
                cross_matched.add(eb)

        for _cam, cam_entities in by_camera.items():
            # Pre-parse embeddings once per camera
            embs: Dict[int, Optional[np.ndarray]] = {}
            for eid, e in cam_entities:
                raw = e.get("embedding")
                if raw is not None:
                    arr = np.frombuffer(raw, dtype=np.float32).copy()
                    n = np.linalg.norm(arr)
                    embs[eid] = arr / n if n > 0 else None
                else:
                    embs[eid] = None

            for i in range(len(cam_entities)):
                eid_i, ei = cam_entities[i]
                for j in range(i + 1, len(cam_entities)):
                    eid_j, ej = cam_entities[j]

                    if entity_type == "vehicle":
                        # Vehicles: merge by bbox IoU (stationary)
                        # OR by high embedding similarity + same color
                        # (handles vehicles that move between sightings)
                        bx_i = (
                            ei.get("bbox_x1", 0),
                            ei.get("bbox_y1", 0),
                            ei.get("bbox_x2", 0),
                            ei.get("bbox_y2", 0),
                        )
                        bx_j = (
                            ej.get("bbox_x1", 0),
                            ej.get("bbox_y1", 0),
                            ej.get("bbox_x2", 0),
                            ej.get("bbox_y2", 0),
                        )
                        ix1 = max(bx_i[0], bx_j[0])
                        iy1 = max(bx_i[1], bx_j[1])
                        ix2 = min(bx_i[2], bx_j[2])
                        iy2 = min(bx_i[3], bx_j[3])
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        a_i = (bx_i[2] - bx_i[0]) * (bx_i[3] - bx_i[1])
                        a_j = (bx_j[2] - bx_j[0]) * (bx_j[3] - bx_j[1])
                        union = a_i + a_j - inter
                        iou = inter / union if union > 0 else 0
                        do_merge = False
                        # Containment: if the smaller bbox sits
                        # almost entirely inside the larger one,
                        # they overlap heavily in time, and share
                        # the same color — duplicate detection.
                        min_area = min(a_i, a_j)
                        containment = inter / min_area if min_area > 0 else 0
                        c_i_early = (ei.get("color") or "").lower()
                        c_j_early = (ej.get("color") or "").lower()
                        # Temporal overlap between the two tracks
                        t_overlap_start = max(
                            ei.get("first_ts") or 0, ej.get("first_ts") or 0
                        )
                        t_overlap_end = min(
                            ei.get("last_ts") or 0, ej.get("last_ts") or 0
                        )
                        t_overlap = max(0.0, t_overlap_end - t_overlap_start)
                        shorter_dur = min(
                            (ei.get("last_ts") or 0) - (ei.get("first_ts") or 0),
                            (ej.get("last_ts") or 0) - (ej.get("first_ts") or 0),
                        )
                        t_overlap_frac = (
                            t_overlap / shorter_dur if shorter_dur > 0 else 0
                        )
                        # For non-overlapping tracks with a large
                        # temporal gap, skip spatial merges — the car
                        # may have left and returned.  Scale the
                        # threshold with clip duration so short clips
                        # catch departures while long clips allow
                        # natural tracking gaps.
                        _large_gap = False
                        if t_overlap == 0:
                            _g_start = max(
                                ei.get("first_ts") or 0,
                                ej.get("first_ts") or 0,
                            )
                            _g_end = min(
                                ei.get("last_ts") or 0,
                                ej.get("last_ts") or 0,
                            )
                            _cam_dur = (
                                cam_clip_end.get(_cam, 0)
                                - cam_clip_start.get(_cam, 0)
                            )
                            _gap_limit = max(3000, _cam_dur * 0.3)
                            if _g_start - _g_end > _gap_limit:
                                _large_gap = True
                        if iou > 0.5 and not _large_gap:
                            do_merge = True
                        elif (
                            containment > 0.95
                            and t_overlap_frac > 0.90
                            and c_i_early == c_j_early
                            and c_i_early != ""
                            and max(a_i, a_j) < 500_000
                        ):
                            do_merge = True
                        else:
                            # Fallback: embedding similarity + color
                            # Skip if non-overlapping tracks have a large
                            # temporal gap — the car may have left and
                            # returned (separate identity events).
                            _do_emb_merge = True
                            if t_overlap == 0:
                                gap_start = max(
                                    ei.get("first_ts") or 0,
                                    ej.get("first_ts") or 0,
                                )
                                gap_end = min(
                                    ei.get("last_ts") or 0,
                                    ej.get("last_ts") or 0,
                                )
                                temporal_gap = gap_start - gap_end
                                if temporal_gap > _gap_limit:
                                    _do_emb_merge = False
                            if _do_emb_merge:
                                e_i = embs.get(eid_i)
                                e_j = embs.get(eid_j)
                                if e_i is not None and e_j is not None:
                                    v_sim = float(np.dot(e_i, e_j))
                                    c_i = (ei.get("color") or "").lower()
                                    c_j = (ej.get("color") or "").lower()
                                    same_color = c_i == c_j and c_i != ""
                                    cx_i = (bx_i[0] + bx_i[2]) / 2
                                    cy_i = (bx_i[1] + bx_i[3]) / 2
                                    cx_j = (bx_j[0] + bx_j[2]) / 2
                                    cy_j = (bx_j[1] + bx_j[3]) / 2
                                    cdist = ((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2) ** 0.5
                                    area_ratio = (
                                        min(a_i, a_j) / max(a_i, a_j)
                                        if max(a_i, a_j) > 0
                                        else 0
                                    )
                                    if (
                                        same_color
                                        and v_sim >= 0.75
                                        and cdist < 200
                                        and area_ratio > 0.5
                                    ):
                                        do_merge = True
                                    elif (
                                        v_sim >= 0.70
                                        and cdist < 150
                                        and area_ratio > 0.8
                                    ):
                                        do_merge = True
                                    elif v_sim >= 0.90 and (same_color or cdist < 100):
                                        do_merge = True
                        if do_merge:
                            uf.union(eid_i, eid_j)
                    else:
                        # Persons: merge by embedding similarity,
                        # scaled by time gap to avoid merging different
                        # people in similar clothes seen far apart.
                        e_i = embs.get(eid_i)
                        e_j = embs.get(eid_j)
                        if e_i is not None and e_j is not None:
                            sim = float(np.dot(e_i, e_j))
                            time_gap = abs(
                                (ei.get("first_ts") or 0) - (ej.get("first_ts") or 0)
                            )

                            # Check clothing description similarity
                            # to allow lower thresholds for long gaps
                            # when the person looks the same.
                            def _color_group(c: str) -> str:
                                c = c.lower().strip()
                                if c in ("dark grey", "dark gray"):
                                    return "dark_grey"
                                if c in ("grey", "gray", "light grey"):
                                    return "grey"
                                if c in ("white", "light"):
                                    return "light"
                                return c

                            desc_i = (
                                ei.get("upper_color") or "",
                                ei.get("lower_color") or "",
                            )
                            desc_j = (
                                ej.get("upper_color") or "",
                                ej.get("lower_color") or "",
                            )
                            same_clothes = (
                                _color_group(desc_i[0]) == _color_group(desc_j[0])
                                and _color_group(desc_i[1]) == _color_group(desc_j[1])
                                and desc_i[0] != ""
                                and desc_i[1] != ""
                            )
                            # Time-gap scaled thresholds:
                            #   <=30s:  0.55 (brief occlusion)
                            #   <=120s: 0.55 (walked off and back)
                            #   120-600s same clothes: 0.62
                            #   120-600s diff clothes: 0.65
                            #   >600s same clothes: 0.68
                            #   >600s diff clothes: 0.78
                            if time_gap <= 120:
                                thresh = 0.55
                            elif time_gap <= 600:
                                thresh = 0.62 if same_clothes else 0.78
                            else:
                                base = 0.735 if same_clothes else (
                                    0.79 if time_gap < 300 else 0.80
                                )
                                if person_threshold > 0.80:
                                    # High adaptive threshold (8h/11h).
                                    # Allow a lower bar for tracks with
                                    # matching upper colour in a moderate
                                    # time window — catches residents
                                    # going in/out while blocking people
                                    # seen hours apart.
                                    same_upper = (
                                        _color_group(desc_i[0])
                                        == _color_group(desc_j[0])
                                        and desc_i[0] != ""
                                    )
                                    if time_gap < 3000 and same_upper:
                                        thresh = 0.75
                                    else:
                                        thresh = person_threshold
                                else:
                                    thresh = base
                            # Cross-camera anchor protection:
                            # if either entity already has a cross-camera
                            # match and they sit in different UF groups,
                            # raise the within-camera merge threshold to
                            # prevent pulling a cross-matched entity into
                            # the wrong identity cluster.
                            # Exempt temporally close pairs (<=120s) —
                            # tracks overlapping or adjacent on the same
                            # camera are clearly the same person.
                            if (
                                (eid_i in cross_matched or eid_j in cross_matched)
                                and uf.find(eid_i) != uf.find(eid_j)
                                and time_gap > 180
                            ):
                                thresh = max(thresh, 0.86)
                            if sim >= thresh:
                                uf.union(eid_i, eid_j)

        # Initialize all entities in Union-Find
        for eid in type_entities:
            uf.find(eid)

        # Get clusters
        clusters = uf.get_clusters()

        # Build identities from clusters, consolidating same-camera
        # sightings into single entries with the full time span.
        identities = []
        for root, entity_ids in clusters.items():
            # Collect every individual track as a raw sighting
            raw_sightings = []
            cam_sightings: Dict[str, list] = {}
            for eid in entity_ids:
                e = type_entities.get(eid)
                if not e:
                    continue
                cam = e["camera_label"]
                entry = {
                    "camera": cam,
                    "entity_id": eid,
                    "track_id": e["track_id"],
                    "description": e["description"],
                    "first_ts": e["first_ts"],
                    "last_ts": e["last_ts"],
                    "crop_path": e["crop_path"],
                }
                raw_sightings.append(entry)
                cam_sightings.setdefault(cam, []).append(entry)

            raw_sightings.sort(key=lambda s: s["first_ts"])

            # Consolidate: one sighting per camera with earliest/latest ts
            sightings = []
            for cam, cam_list in cam_sightings.items():
                cam_list.sort(key=lambda s: s["first_ts"])
                best = max(cam_list, key=lambda s: s["last_ts"] - s["first_ts"])
                first = cam_list[0]["first_ts"]
                last = max(s["last_ts"] for s in cam_list)

                # Extend stationary vehicle bars to clip boundaries.
                # If detected near clip start/end, assume present the
                # whole time (motion detection causes intermittent gaps).
                if entity_type == "vehicle":
                    c_start = cam_clip_start.get(cam, 0)
                    c_end = cam_clip_end.get(cam, 0)
                    clip_dur = c_end - c_start
                    if clip_dur > 0:
                        margin = clip_dur * 0.05  # 5% margin
                        if first - c_start < margin:
                            first = c_start
                        if c_end - last < margin:
                            last = c_end

                sightings.append(
                    {
                        "camera": cam,
                        "entity_id": best["entity_id"],
                        "track_id": best["track_id"],
                        "description": best["description"],
                        "first_ts": first,
                        "last_ts": last,
                        "crop_path": best["crop_path"],
                    }
                )

            sightings.sort(key=lambda s: s["first_ts"])

            cameras_in_cluster = set(s["camera"] for s in sightings)
            matched = len(cameras_in_cluster) > 1

            identities.append(
                {
                    "unified_id": None,
                    "entity_type": entity_type,
                    "sightings": sightings,
                    "raw_sightings": raw_sightings,
                    "similarity": None,
                    "matched": matched,
                }
            )

        # Sort by first appearance timestamp
        identities.sort(key=lambda i: i["sightings"][0]["first_ts"])

        # Assign unified IDs after sorting
        for i, identity in enumerate(identities, 1):
            identity["unified_id"] = i

        return identities

    persons = _build_identities_union_find(matches["person_matches"], "person")

    # Post-processing: merge "returning person" identities.
    # If a cross-camera person departs early in the clip and another
    # cross-camera person arrives late, they may be the same individual
    # returning after a clothing change.  Merge when the best
    # sighting-pair embedding similarity exceeds a threshold.
    _clip_s = min(cam_clip_start.values()) if cam_clip_start else 0
    _clip_e = max(cam_clip_end.values()) if cam_clip_end else 0
    _clip_dur = _clip_e - _clip_s
    if _clip_dur > 3600:  # only for clips > 1h
        _early_cut = _clip_s + _clip_dur * 0.20
        # Use the tighter of percentage-based and absolute cap so
        # the window works for both 8h and 11h clips.  The absolute
        # cap (7h from clip start) lets 11h evening returns qualify
        # without widening the window so much that 8h false-merges.
        _late_cut = min(
            _clip_e - _clip_dur * 0.20,
            _clip_s + 7 * 3600,
        )
        _ret_thresh = 0.70
        _merged_idx: set = set()
        for i, pi in enumerate(persons):
            if i in _merged_idx:
                continue
            pi_cams = set(s["camera"] for s in pi["sightings"])
            if len(pi_cams) < 2:
                continue
            pi_latest = max(s["last_ts"] for s in pi["sightings"])
            if pi_latest > _early_cut:
                continue
            for j, pj in enumerate(persons):
                if j <= i or j in _merged_idx:
                    continue
                pj_cams = set(s["camera"] for s in pj["sightings"])
                if len(pj_cams) < 2:
                    continue
                pj_earliest = min(s["first_ts"] for s in pj["sightings"])
                if pj_earliest < _late_cut:
                    continue
                # Best embedding sim between any raw sighting pair
                _best = 0.0
                for si in pi.get("raw_sightings", []):
                    ei = all_entities_dict.get(si["entity_id"])
                    if not ei or not ei.get("embedding"):
                        continue
                    _ei = np.frombuffer(ei["embedding"], dtype=np.float32)
                    _ni = np.linalg.norm(_ei)
                    if _ni == 0:
                        continue
                    _ei = _ei / _ni
                    for sj in pj.get("raw_sightings", []):
                        ej = all_entities_dict.get(sj["entity_id"])
                        if not ej or not ej.get("embedding"):
                            continue
                        _ej = np.frombuffer(ej["embedding"], dtype=np.float32)
                        _nj = np.linalg.norm(_ej)
                        if _nj == 0:
                            continue
                        _ej = _ej / _nj
                        s = float(np.dot(_ei, _ej))
                        if s > _best:
                            _best = s
                if _best >= _ret_thresh:
                    pi["sightings"].extend(pj["sightings"])
                    pi["raw_sightings"].extend(
                        pj.get("raw_sightings", [])
                    )
                    pi["sightings"].sort(key=lambda x: x["first_ts"])
                    pi["raw_sightings"].sort(
                        key=lambda x: x["first_ts"]
                    )
                    pi["matched"] = True
                    # Store returning-person re-ID similarity
                    existing_sim = pi.get("similarity")
                    if existing_sim:
                        pi["similarity"] = max(existing_sim, _best)
                    else:
                        pi["similarity"] = round(_best, 3)
                    _merged_idx.add(j)
                    break  # one returning match per person
        if _merged_idx:
            persons = [
                p for idx, p in enumerate(persons)
                if idx not in _merged_idx
            ]
            # Re-number unified IDs
            for idx, p in enumerate(persons, 1):
                p["unified_id"] = idx

    vehicles = _build_identities_union_find(matches["vehicle_matches"], "vehicle")

    # Collect short-duration vehicle tracks (filtered by the 300s noise
    # gate) — these may be brief arrival/departure detections where a
    # car is only visible for a few seconds as it pulls up or drives off.
    brief_vehicles: List[dict] = []
    for eid, e in all_entities_dict.items():
        if e["entity_type"] != "vehicle":
            continue
        dur = (e["last_ts"] or 0) - (e["first_ts"] or 0)
        if 0.1 <= dur < 300:
            brief_vehicles.append(
                {
                    "entity_id": eid,
                    "camera": e["camera_label"],
                    "first_ts": e["first_ts"],
                    "last_ts": e["last_ts"],
                    "description": e.get("description", ""),
                    "crop_path": e.get("crop_path"),
                    "bbox": (
                        e.get("bbox_x1", 0),
                        e.get("bbox_y1", 0),
                        e.get("bbox_x2", 0),
                        e.get("bbox_y2", 0),
                    ),
                }
            )

    journeys = link_persons_to_vehicles(
        persons,
        vehicles,
        cam_clip_start,
        cam_clip_end,
        brief_vehicles=brief_vehicles,
        all_entities=all_entities_dict,
    )

    round_trips = match_vehicle_returns(journeys, vehicles)

    return {
        "persons": persons,
        "vehicles": vehicles,
        "journeys": journeys,
        "round_trips": round_trips,
    }


def link_persons_to_vehicles(
    persons: List[dict],
    vehicles: List[dict],
    cam_clip_start: Dict[str, float],
    cam_clip_end: Dict[str, float],
    max_gap_seconds: float = 600.0,
    brief_vehicles: Optional[List[dict]] = None,
    all_entities: Optional[Dict[int, dict]] = None,
) -> List[dict]:
    """Link persons to vehicle departures and arrivals.

    A vehicle *departs* if it was present from near clip start but
    disappears before clip end.  A vehicle *arrives* if it first
    appears well after clip start and stays until near clip end.

    brief_vehicles are short-duration vehicle tracks (filtered by the
    noise gate) that may represent a car briefly appearing as it pulls
    up or drives off.  These are also checked as arrival/departure events.

    Returns a list of journey event dicts.
    """
    if not persons:
        return []
    if not vehicles and not brief_vehicles:
        return []

    # Determine clip boundaries (use max across cameras)
    clip_start = min(cam_clip_start.values()) if cam_clip_start else 0
    clip_end = max(cam_clip_end.values()) if cam_clip_end else 0
    clip_dur = clip_end - clip_start
    if clip_dur <= 0:
        return []

    # Classify each vehicle as departure, arrival, or stationary
    vehicle_events: List[dict] = []
    for v in vehicles:
        for s in v["sightings"]:
            cam = s["camera"]
            c_start = cam_clip_start.get(cam, clip_start)
            c_end = cam_clip_end.get(cam, clip_end)
            c_dur = c_end - c_start
            if c_dur <= 0:
                continue
            c_margin = c_dur * 0.05

            near_start = (s["first_ts"] - c_start) < c_margin
            near_end = (c_end - s["last_ts"]) < c_margin

            if near_start and not near_end:
                # Present from start, leaves before end → departure
                vehicle_events.append(
                    {
                        "vehicle": v,
                        "sighting": s,
                        "event": "departure",
                        "event_ts": s["last_ts"],
                        "camera": cam,
                    }
                )
            elif not near_start and near_end:
                # Appears after start, stays until end → arrival
                vehicle_events.append(
                    {
                        "vehicle": v,
                        "sighting": s,
                        "event": "arrival",
                        "event_ts": s["first_ts"],
                        "camera": cam,
                    }
                )
            # If near_start AND near_end → stationary whole clip, skip
            elif not near_start and not near_end:
                # Appeared and disappeared mid-clip → visiting vehicle
                vehicle_events.append(
                    {
                        "vehicle": v,
                        "sighting": s,
                        "event": "arrival",
                        "event_ts": s["first_ts"],
                        "camera": cam,
                    }
                )
                vehicle_events.append(
                    {
                        "vehicle": v,
                        "sighting": s,
                        "event": "departure",
                        "event_ts": s["last_ts"],
                        "camera": cam,
                    }
                )

    # Brief vehicle tracks (filtered by noise gate) may represent a car
    # briefly appearing as it arrives or departs.  Classify by position
    # within the clip: near start = departure, near end = arrival,
    # mid-clip = could be either — try to match with a main vehicle.
    for bv in brief_vehicles or []:
        cam = bv["camera"]
        c_start = cam_clip_start.get(cam, clip_start)
        c_end = cam_clip_end.get(cam, clip_end)
        c_dur = c_end - c_start
        if c_dur <= 0:
            continue

        # Match brief vehicle to a main vehicle by bbox IoU + temporal
        # proximity.  A brief detection must be within 1800s of the
        # main vehicle's sighting to avoid matching cars that parked
        # in the same spot hours apart.
        bv_bbox = bv.get("bbox", (0, 0, 0, 0))
        matched_vehicle = None
        best_iou = 0.0
        for v in vehicles:
            for s in v["sightings"]:
                if s["camera"] != cam:
                    continue
                # Temporal proximity: brief detection must be within
                # 1800s of the main vehicle's time range
                bv_mid = (bv["first_ts"] + bv["last_ts"]) / 2
                if (bv_mid < s["first_ts"] - 1800
                        or bv_mid > s["last_ts"] + 1800):
                    continue
                # Skip stationary vehicles (present whole clip)
                _sv_start = cam_clip_start.get(cam, clip_start)
                _sv_end = cam_clip_end.get(cam, clip_end)
                _sv_dur = _sv_end - _sv_start
                if _sv_dur > 0:
                    _sv_margin = _sv_dur * 0.05
                    if ((s["first_ts"] - _sv_start) < _sv_margin
                            and (_sv_end - s["last_ts"]) < _sv_margin):
                        continue
                # Get main vehicle bbox from DB entity
                eid = s.get("entity_id")
                if eid and all_entities and eid in all_entities:
                    e = all_entities[eid]
                    mv_bbox = (
                        e.get("bbox_x1", 0),
                        e.get("bbox_y1", 0),
                        e.get("bbox_x2", 0),
                        e.get("bbox_y2", 0),
                    )
                    # Calculate IoU
                    ix1 = max(bv_bbox[0], mv_bbox[0])
                    iy1 = max(bv_bbox[1], mv_bbox[1])
                    ix2 = min(bv_bbox[2], mv_bbox[2])
                    iy2 = min(bv_bbox[3], mv_bbox[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    a1 = (bv_bbox[2] - bv_bbox[0]) * (bv_bbox[3] - bv_bbox[1])
                    a2 = (mv_bbox[2] - mv_bbox[0]) * (mv_bbox[3] - mv_bbox[1])
                    union = a1 + a2 - inter
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        matched_vehicle = v

        if matched_vehicle is None or best_iou < 0.15:
            continue

        if matched_vehicle is None:
            continue

        # Skip arrival events from vehicles that have been parked
        # for a long time — a car present for hours is not "arriving".
        mv_earliest = min(
            s["first_ts"]
            for s in matched_vehicle["sightings"]
            if s["camera"] == cam
        )
        if bv["first_ts"] - mv_earliest > 1800:
            continue

        # Brief detection mid-clip = arrival event
        c_margin = c_dur * 0.1
        near_start = (bv["first_ts"] - c_start) < c_margin
        if not near_start:
            vehicle_events.append(
                {
                    "vehicle": matched_vehicle,
                    "sighting": {
                        "camera": cam,
                        "description": bv.get("description", ""),
                        "first_ts": bv["first_ts"],
                        "last_ts": bv["last_ts"],
                        "crop_path": bv.get("crop_path"),
                    },
                    "event": "arrival",
                    "event_ts": bv["first_ts"],
                    "camera": cam,
                }
            )

    if not vehicle_events:
        return []

    # Build candidate linkages scored by direction + temporal proximity.
    # Direction (the order of cameras a person traverses) is the
    # strongest signal: garden→garage = leaving house (departure),
    # garage→garden = arriving home (arrival).
    #
    # The person may appear on ANY camera near the vehicle event,
    # not just the vehicle's camera (e.g. car arrives on garage,
    # person walks to garden — first sighting is on garden cam).
    candidates: List[dict] = []

    for ve in vehicle_events:
        ev_cam = ve["camera"]
        ev_ts = ve["event_ts"]
        ev_type = ve["event"]

        for p in persons:
            all_sightings = p["sightings"]
            if not all_sightings:
                continue

            # Use raw_sightings (chronologically ordered individual
            # appearances) for direction detection rather than
            # consolidated sightings, which lose temporal ordering
            # when a person goes camera A → B → A.
            raw = p.get("raw_sightings", all_sightings)

            # Determine person direction across cameras
            direction_matches = False
            has_direction = False
            if len(raw) >= 2:
                first_cam = raw[0]["camera"]
                last_cam = raw[-1]["camera"]
                if first_cam != last_cam:
                    has_direction = True
                    if ev_type == "departure":
                        # Departure: person ends on vehicle camera
                        # (walks TO car then car leaves)
                        direction_matches = last_cam == ev_cam
                    else:
                        # Arrival: person starts on vehicle camera
                        # (gets out of car at garage, walks to garden)
                        direction_matches = first_cam == ev_cam

            # Find the person sighting closest to the vehicle event.
            # Use absolute gap — allow either temporal order because
            # the vehicle's detected arrival/departure time may not
            # match the physical event exactly (e.g. bbox split delay).
            best_ps = None
            best_gap = float("inf")
            for ps in all_sightings:
                if ev_type == "departure":
                    # Prefer person BEFORE vehicle departs
                    gap = ev_ts - ps["last_ts"]
                else:
                    # Use absolute gap — person may appear before
                    # or after the detected vehicle arrival
                    gap = ps["first_ts"] - ev_ts

                abs_gap = abs(gap)
                if abs_gap < best_gap:
                    best_gap = abs_gap
                    best_ps = ps

            if best_ps is None or best_gap > max_gap_seconds:
                continue

            if ev_type == "departure":
                person_ts = best_ps["last_ts"]
            else:
                person_ts = best_ps["first_ts"]

            # Base score from temporal proximity (sqrt decay)
            score = max(0.0, 1.0 - (best_gap / max_gap_seconds) ** 0.5)

            # Direction is the primary signal — large multiplier.
            if direction_matches:
                score *= 3.0
            elif has_direction and not direction_matches:
                score *= 0.15

            # Bonus if person sighting is on the same camera as vehicle
            if best_ps["camera"] == ev_cam:
                score *= 1.2

            # Temporal ordering: for arrivals the person should appear
            # AFTER the vehicle; for departures the person should be
            # seen BEFORE the vehicle leaves.
            if ev_type == "departure":
                order_gap = ev_ts - person_ts  # positive = correct
            else:
                order_gap = person_ts - ev_ts  # positive = correct

            if order_gap < -120:
                # Wrong order by >2 min — almost certainly not real
                score *= 0.05
            elif order_gap < -30:
                score *= 0.2

            # "Already present" / "Still around" checks — gap-aware.
            # A returning person (leaves morning, returns evening)
            # has sightings spanning the whole day.  Check for
            # NEARBY sightings that contradict the journey, not the
            # overall time span.
            if ev_type == "arrival":
                # Find latest sighting BEFORE the vehicle event
                prev_sightings = [
                    s for s in all_sightings
                    if s["last_ts"] < ev_ts - 60
                ]
                if prev_sightings:
                    prev_latest = max(
                        s["last_ts"] for s in prev_sightings
                    )
                    if ev_ts - prev_latest < 3600:
                        # Person was seen <1h before arrival —
                        # they were already on-site
                        score *= 0.1
                    # else: >1h gap means they left and returned

                # Stronger: if person's NEAREST pre-event sighting
                # is on a different camera and very close in time,
                # they were already walking around on-site.
                near_before = [
                    s for s in all_sightings
                    if s["first_ts"] < ev_ts
                    and ev_ts - s["first_ts"] < 300
                ]
                if near_before:
                    first_near = min(
                        near_before, key=lambda s: s["first_ts"]
                    )
                    if (first_near["camera"] != ev_cam
                            and first_near["first_ts"] < ev_ts - 10):
                        score *= 0.05

            if ev_type == "departure":
                # Find earliest sighting AFTER the vehicle event
                post_sightings = [
                    s for s in all_sightings
                    if s["first_ts"] > ev_ts + 60
                ]
                if post_sightings:
                    post_earliest = min(
                        s["first_ts"] for s in post_sightings
                    )
                    if post_earliest - ev_ts < 3600:
                        # Person reappears <1h after departure —
                        # they didn't leave in the vehicle
                        score *= 0.1
                    # else: >1h gap means they left and came back

            if score >= 0.25:
                candidates.append(
                    {
                        "person_id": p["unified_id"],
                        "vehicle_id": ve["vehicle"]["unified_id"],
                        "event": ev_type,
                        "vehicle_desc": ve["sighting"]["description"],
                        "person_desc": best_ps["description"],
                        "timestamp": ev_ts,
                        "person_ts": person_ts,
                        "gap_seconds": best_gap,
                        # Keep raw score for ranking; cap for display
                        "_raw_score": score,
                        "confidence": round(min(score, 1.0), 3),
                        "camera": ev_cam,
                        "person_crop": best_ps.get("crop_path"),
                        "vehicle_crop": ve["sighting"].get("crop_path"),
                    }
                )

    # Greedy assignment: best score first, each person/vehicle used once per event type
    candidates.sort(key=lambda c: c["_raw_score"], reverse=True)
    used_persons: Dict[tuple, bool] = {}  # (person_id, event_type)
    used_vehicles: Dict[tuple, bool] = {}  # (vehicle_id, event_type)
    journeys: List[dict] = []

    for c in candidates:
        p_key = (c["person_id"], c["event"])
        v_key = (c["vehicle_id"], c["event"])
        if p_key in used_persons or v_key in used_vehicles:
            continue
        used_persons[p_key] = True
        used_vehicles[v_key] = True
        journeys.append(c)

    journeys.sort(key=lambda j: j["timestamp"])

    # Merge same-person departure + arrival into round-trip entries.
    by_person: Dict[int, List[dict]] = {}
    for j in journeys:
        by_person.setdefault(j["person_id"], []).append(j)

    merged: List[dict] = []
    merged_ids: set = set()
    for pid, pj_list in by_person.items():
        deps = [j for j in pj_list if j["event"] == "departure"]
        arrs = [j for j in pj_list if j["event"] == "arrival"]
        if deps and arrs:
            dep = deps[0]
            arr = arrs[0]
            merged.append({
                "person_id": pid,
                "vehicle_id": arr["vehicle_id"],
                "event": "round_trip",
                "vehicle_desc": arr["vehicle_desc"],
                "person_desc": dep["person_desc"],
                "timestamp": dep["timestamp"],
                "person_ts": dep["person_ts"],
                "gap_seconds": dep["gap_seconds"],
                "confidence": round(
                    (dep["confidence"] + arr["confidence"]) / 2, 3
                ),
                "camera": dep["camera"],
                "person_crop": dep.get("person_crop"),
                "vehicle_crop": arr.get("vehicle_crop"),
                "departure": dep,
                "arrival": arr,
            })
            merged_ids.add(id(dep))
            merged_ids.add(id(arr))

    # Keep non-merged journeys, replace merged pairs with round-trip
    result = [j for j in journeys if id(j) not in merged_ids]
    result.extend(merged)
    result.sort(key=lambda j: j["timestamp"])
    return result


def match_vehicle_returns(
    journeys: List[dict],
    vehicles: List[dict],
    max_away_hours: float = 24.0,
) -> List[dict]:
    """Match vehicle departures with subsequent returns of the same vehicle.

    Pairs a departure journey with a later arrival journey for the same
    vehicle (by Re-ID match — same unified_id).  Produces round-trip
    records showing who left in which car and when they came back.

    This is a read-only post-processing step — it does NOT modify the
    input journeys list.

    Args:
        journeys: Output of link_persons_to_vehicles().
        vehicles: Unified vehicle identities (with sightings).
        max_away_hours: Maximum time between departure and return.

    Returns:
        List of round-trip dicts, each containing departure + return
        journey data and computed away-time.
    """
    if not journeys:
        return []

    max_away_sec = max_away_hours * 3600

    # Group journeys by vehicle
    by_vehicle: Dict[int, List[dict]] = {}
    for j in journeys:
        by_vehicle.setdefault(j["vehicle_id"], []).append(j)

    # Also look for vehicles seen on multiple cameras (cross-camera match)
    # where the same vehicle appears, leaves, and reappears — even if
    # no person was linked to the return.
    vehicle_map = {v["unified_id"]: v for v in vehicles}

    round_trips: List[dict] = []

    for vid, vj_list in by_vehicle.items():
        departures = [j for j in vj_list if j["event"] == "departure"]
        arrivals = [j for j in vj_list if j["event"] == "arrival"]

        # Sort both by timestamp
        departures.sort(key=lambda j: j["timestamp"])
        arrivals.sort(key=lambda j: j["timestamp"])

        # Greedy pairing: each departure matched to next arrival
        used_arrivals: set = set()
        for dep in departures:
            for i, arr in enumerate(arrivals):
                if i in used_arrivals:
                    continue
                # Arrival must come AFTER departure
                if arr["timestamp"] <= dep["timestamp"]:
                    continue
                away_time = arr["timestamp"] - dep["timestamp"]
                if away_time > max_away_sec:
                    continue

                used_arrivals.add(i)

                # Build round-trip record
                v = vehicle_map.get(vid)
                v_desc = dep["vehicle_desc"]
                if v and v.get("sightings"):
                    v_desc = v["sightings"][0]["description"]

                round_trips.append(
                    {
                        "vehicle_id": vid,
                        "vehicle_desc": v_desc,
                        "departure": {
                            "person_id": dep["person_id"],
                            "person_desc": dep["person_desc"],
                            "timestamp": dep["timestamp"],
                            "person_ts": dep["person_ts"],
                            "camera": dep["camera"],
                            "confidence": dep["confidence"],
                            "gap_seconds": dep["gap_seconds"],
                        },
                        "arrival": {
                            "person_id": arr["person_id"],
                            "person_desc": arr["person_desc"],
                            "timestamp": arr["timestamp"],
                            "person_ts": arr["person_ts"],
                            "camera": arr["camera"],
                            "confidence": arr["confidence"],
                            "gap_seconds": arr["gap_seconds"],
                        },
                        "away_seconds": away_time,
                        "same_person": dep["person_id"] == arr["person_id"],
                        "person_crop_out": dep.get("person_crop"),
                        "person_crop_in": arr.get("person_crop"),
                        "vehicle_crop": dep.get("vehicle_crop")
                        or arr.get("vehicle_crop"),
                    }
                )
                break  # Move to next departure

    round_trips.sort(key=lambda r: r["departure"]["timestamp"])
    return round_trips


def generate_entity_clip(
    db_path: str,
    entity_id: int,
    output_path: str,
    padding_frames: int = 10,
) -> str:
    """Generate a video clip showing just one tracked entity.

    Crops to follow the entity through the video, with some padding.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    entity = dict(
        conn.execute(
            "SELECT * FROM tracked_entities WHERE id = ?", (entity_id,)
        ).fetchone()
    )

    frames = conn.execute(
        "SELECT frame_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2 "
        "FROM track_frames WHERE entity_id = ? ORDER BY frame_idx",
        (entity_id,),
    ).fetchall()
    conn.close()

    frames = [dict(f) for f in frames]
    if not frames:
        return ""

    video_path = entity["video_path"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Build a frame_idx -> bbox map
    bbox_map = {}
    for f in frames:
        bbox_map[f["frame_idx"]] = [
            int(f["bbox_x1"]),
            int(f["bbox_y1"]),
            int(f["bbox_x2"]),
            int(f["bbox_y2"]),
        ]

    start_frame = max(0, frames[0]["frame_idx"] - padding_frames)
    end_frame = min(
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1,
        frames[-1]["frame_idx"] + padding_frames,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (vid_w, vid_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for fidx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        bbox = bbox_map.get(fidx)
        if bbox:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if entity["entity_type"] == "person" else (255, 100, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                entity["description"],
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

    cap.release()
    writer.release()
    _reencode_h264(str(output_path))
    return str(output_path)


def generate_cross_camera_clip(
    db_path: str,
    entity_a_id: int,
    entity_b_id: int,
    output_path: str,
    padding_frames: int = 10,
    unified_id: Optional[int] = None,
) -> str:
    """Generate a merged video that follows an entity across two cameras.

    Orders segments chronologically by first_ts — whichever camera saw
    the entity first plays first.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    ea = dict(
        conn.execute(
            "SELECT * FROM tracked_entities WHERE id = ?", (entity_a_id,)
        ).fetchone()
    )
    eb = dict(
        conn.execute(
            "SELECT * FROM tracked_entities WHERE id = ?", (entity_b_id,)
        ).fetchone()
    )

    frames_a = conn.execute(
        "SELECT frame_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2 "
        "FROM track_frames WHERE entity_id = ? ORDER BY frame_idx",
        (entity_a_id,),
    ).fetchall()
    frames_b = conn.execute(
        "SELECT frame_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2 "
        "FROM track_frames WHERE entity_id = ? ORDER BY frame_idx",
        (entity_b_id,),
    ).fetchall()
    conn.close()

    frames_a = [dict(f) for f in frames_a]
    frames_b = [dict(f) for f in frames_b]

    if not frames_a or not frames_b:
        return ""

    # Order by timestamp — whichever entity was seen first plays first
    if ea["first_ts"] <= eb["first_ts"]:
        first_entity, second_entity = ea, eb
        first_frames, second_frames = frames_a, frames_b
    else:
        first_entity, second_entity = eb, ea
        first_frames, second_frames = frames_b, frames_a

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap_first = cv2.VideoCapture(first_entity["video_path"])
    cap_second = cv2.VideoCapture(second_entity["video_path"])
    fps_first = cap_first.get(cv2.CAP_PROP_FPS) or 25.0
    fps_second = cap_second.get(cv2.CAP_PROP_FPS) or 25.0
    w_first = int(cap_first.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_first = int(cap_first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_second = int(cap_second.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_second = int(cap_second.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use the dimensions of the larger video
    out_w = max(w_first, w_second)
    out_h = max(h_first, h_second)
    out_fps = max(fps_first, fps_second)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (out_w, out_h))

    def _build_bbox_map(frames_list):
        m = {}
        for f in frames_list:
            m[f["frame_idx"]] = [
                int(f["bbox_x1"]),
                int(f["bbox_y1"]),
                int(f["bbox_x2"]),
                int(f["bbox_y2"]),
            ]
        return m

    bbox_first = _build_bbox_map(first_frames)
    bbox_second = _build_bbox_map(second_frames)

    def _draw_entity(frame, bbox, entity, cam_label, uid):
        etype = entity["entity_type"]
        desc = entity["description"]
        if bbox:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if etype == "person" else (255, 100, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Show unified ID if provided
            if uid is not None:
                label = f"Person {uid} @ {cam_label}"
            else:
                label = f"{cam_label}: {desc}"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    # Part 1: first camera (chronologically)
    start_1 = max(0, first_frames[0]["frame_idx"] - padding_frames)
    end_1 = first_frames[-1]["frame_idx"] + padding_frames
    cap_first.set(cv2.CAP_PROP_POS_FRAMES, start_1)
    for fidx in range(start_1, end_1 + 1):
        ret, frame = cap_first.read()
        if not ret:
            break
        if frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h))
        bb = bbox_first.get(fidx)
        frame = _draw_entity(
            frame, bb, first_entity, first_entity["camera_label"], unified_id
        )
        writer.write(frame)

    # Part 2: second camera (chronologically)
    start_2 = max(0, second_frames[0]["frame_idx"] - padding_frames)
    end_2 = second_frames[-1]["frame_idx"] + padding_frames
    cap_second.set(cv2.CAP_PROP_POS_FRAMES, start_2)
    for fidx in range(start_2, end_2 + 1):
        ret, frame = cap_second.read()
        if not ret:
            break
        if frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h))
        bb = bbox_second.get(fidx)
        frame = _draw_entity(
            frame, bb, second_entity, second_entity["camera_label"], unified_id
        )
        writer.write(frame)

    cap_first.release()
    cap_second.release()
    writer.release()
    _reencode_h264(str(output_path))
    return str(output_path)


def generate_annotated_video(
    video_path: str,
    db_path: str,
    output_path: str,
    camera_label: Optional[str] = None,
) -> str:
    """Render an annotated video with bounding boxes and track IDs overlaid.

    Reads the per-frame bbox data from track_frames and draws them
    on each frame of the source video.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get all entities for this video
    if camera_label:
        rows = c.execute(
            "SELECT id, entity_type, track_id, description, color "
            "FROM tracked_entities WHERE video_path = ? AND camera_label = ?",
            (str(video_path), camera_label),
        ).fetchall()
    else:
        rows = c.execute(
            "SELECT id, entity_type, track_id, description, color "
            "FROM tracked_entities WHERE video_path = ?",
            (str(video_path),),
        ).fetchall()

    # Build frame -> [(bbox, label, color)] mapping
    frame_annotations: Dict[int, list] = defaultdict(list)

    for entity_id, etype, tid, desc, color in rows:
        frames = c.execute(
            "SELECT frame_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2 "
            "FROM track_frames WHERE entity_id = ?",
            (entity_id,),
        ).fetchall()

        if etype == "person":
            box_color = (0, 255, 0)
            label = f"Person {tid}: {desc}"
        else:
            box_color = (255, 100, 0)
            label = f"V{tid}: {desc}"

        for fidx, bx1, by1, bx2, by2 in frames:
            frame_annotations[fidx].append(
                {
                    "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                    "label": label,
                    "color": box_color,
                }
            )

    conn.close()

    # Render
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for ann in frame_annotations.get(fidx, []):
            x1, y1, x2, y2 = ann["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), ann["color"], 2)
            cv2.putText(
                frame,
                ann["label"],
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ann["color"],
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()
    _reencode_h264(str(output_path))
    return str(output_path)
