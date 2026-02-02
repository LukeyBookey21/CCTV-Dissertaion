"""License plate detection and OCR for forensic video analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import easyocr
except ImportError:
    easyocr = None


class LicensePlateDetector:
    """
    License plate detector using YOLOv8 and EasyOCR.

    Two-stage pipeline:
    1. Detect license plate regions with YOLO
    2. Extract text from plates with OCR
    """

    def __init__(
        self,
        plate_model_path: str = "yolov8n.pt",
        ocr_languages: List[str] = ["en"],
        ocr_gpu: bool = True,
    ):
        """
        Initialize plate detector.

        Args:
            plate_model_path: Path to YOLOv8 license plate detection model
            ocr_languages: Languages for OCR (default: English)
            ocr_gpu: Use GPU for OCR if available
        """
        if YOLO is None:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        self.plate_model = YOLO(plate_model_path)

        # Initialize EasyOCR reader
        if easyocr is None:
            raise ImportError("easyocr not installed. Run: pip install easyocr")

        self.ocr_reader = easyocr.Reader(ocr_languages, gpu=ocr_gpu)

    def detect_plates_in_frame(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.25,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 12.0,
        min_width: int = 30,
        min_height: int = 10,
        imgsz: int = 640,
    ) -> List[Dict[str, Any]]:
        """
        Detect license plates in a single frame with aspect ratio filtering.

        Args:
            frame: Video frame (BGR format)
            conf_threshold: Confidence threshold for detections
            min_aspect_ratio: Minimum width/height ratio (UK plates: ~4.4:1)
            max_aspect_ratio: Maximum width/height ratio
            min_width: Minimum plate width in pixels
            min_height: Minimum plate height in pixels
            imgsz: Image size for YOLO inference (default: 640,
                use 1280 for distant/small plates)

        Returns:
            List of detected plates with bbox and confidence (filtered)
        """
        results = self.plate_model.predict(
            frame,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        plates = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                # Calculate dimensions
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0

                # Filter by aspect ratio and minimum size
                # UK plates are typically 4.4:1 to 5.5:1 ratio
                if (
                    aspect_ratio >= min_aspect_ratio
                    and aspect_ratio <= max_aspect_ratio
                    and width >= min_width
                    and height >= min_height
                ):

                    plates.append(
                        {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": conf,
                            "aspect_ratio": float(aspect_ratio),
                            "width": float(width),
                            "height": float(height),
                        }
                    )

        return plates

    def extract_text_from_plate(
        self,
        frame: np.ndarray,
        bbox: List[float],
        preprocess: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract text from a license plate region using OCR.

        Args:
            frame: Full video frame
            bbox: Bounding box [x1, y1, x2, y2] of plate
            preprocess: Apply preprocessing to improve OCR

        Returns:
            Dict with text and confidence, or None if no text found
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Crop plate region
        plate_img = frame[y1:y2, x1:x2]

        if plate_img.size == 0:
            return None

        # Preprocessing for better OCR
        if preprocess:
            # Convert to grayscale
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

            # Increase contrast
            gray = cv2.equalizeHist(gray)

            # Denoise
            gray = cv2.fastNlMeansDenoising(gray, h=10)

            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            plate_img = thresh

        # Run OCR
        results = self.ocr_reader.readtext(plate_img)

        if not results:
            return None

        # Combine all detected text (in case plate is split)
        texts = []
        confidences = []

        for bbox_ocr, text, conf in results:
            # Clean up text (remove spaces, special chars)
            cleaned = "".join(c for c in text if c.isalnum()).upper()
            if cleaned:
                texts.append(cleaned)
                confidences.append(conf)

        if not texts:
            return None

        # Return combined text with average confidence
        return {
            "text": "".join(texts),
            "confidence": float(np.mean(confidences)),
            "raw_text": " ".join(texts),
        }

    def detect_and_read_plates(
        self,
        frame: np.ndarray,
        frame_index: int,
        detect_conf: float = 0.4,
        ocr_min_conf: float = 0.5,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 12.0,
        imgsz: int = 640,
    ) -> List[Dict[str, Any]]:
        """
        Complete pipeline: detect plates and extract text with filtering.

        Args:
            frame: Video frame
            frame_index: Frame number in video
            detect_conf: Detection confidence threshold
            ocr_min_conf: Minimum OCR confidence to include
            min_aspect_ratio: Minimum width/height ratio for valid plates
            max_aspect_ratio: Maximum width/height ratio for valid plates
            imgsz: Image size for YOLO inference
                (640=default, 1280=better for small/distant plates)

        Returns:
            List of plates with bbox, text, and confidences (filtered)
        """
        plates = self.detect_plates_in_frame(
            frame,
            detect_conf,
            min_aspect_ratio,
            max_aspect_ratio,
            imgsz=imgsz,
        )

        results = []
        for plate in plates:
            ocr_result = self.extract_text_from_plate(frame, plate["bbox"])

            result = {
                "frame_index": frame_index,
                "bbox": plate["bbox"],
                "detection_confidence": plate["confidence"],
                "text": None,
                "ocr_confidence": None,
                "raw_text": None,
            }

            if ocr_result and ocr_result["confidence"] >= ocr_min_conf:
                result["text"] = ocr_result["text"]
                result["ocr_confidence"] = ocr_result["confidence"]
                result["raw_text"] = ocr_result["raw_text"]

            results.append(result)

        return results


def detect_plates_in_video(
    video_path: str | Path,
    plate_model_path: str = "yolov8n.pt",
    frame_stride: int = 5,
    detect_conf: float = 0.4,
    ocr_min_conf: float = 0.5,
    max_frames: Optional[int] = None,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 12.0,
    imgsz: int = 640,
) -> List[Dict[str, Any]]:
    """
    Detect license plates in a video file with aspect ratio filtering.

    Args:
        video_path: Path to video file
        plate_model_path: Path to YOLOv8 license plate model
        frame_stride: Process every Nth frame
        detect_conf: Detection confidence threshold
            (default: 0.4, increased to reduce false positives)
        ocr_min_conf: Minimum OCR confidence
        max_frames: Maximum frames to process (None = all)
        min_aspect_ratio: Minimum width/height ratio for valid plates (default: 0.5)
        max_aspect_ratio: Maximum width/height ratio for valid plates (default: 12.0)
        imgsz: Image size for YOLO inference
            (640=default, 1280=better for distant/small plates)

    Returns:
        List of all detected plates across all frames
        (filtered by aspect ratio)
    """
    source = Path(video_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    detector = LicensePlateDetector(plate_model_path)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    all_plates = []
    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_idx % frame_stride == 0:
            plates = detector.detect_and_read_plates(
                frame,
                frame_idx,
                detect_conf,
                ocr_min_conf,
                min_aspect_ratio,
                max_aspect_ratio,
                imgsz,
            )
            all_plates.extend(plates)

            processed_frames += 1
            if max_frames and processed_frames >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return all_plates


__all__ = [
    "LicensePlateDetector",
    "detect_plates_in_video",
]
