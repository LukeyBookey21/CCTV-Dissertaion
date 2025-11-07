"""Video ingestion helpers: hashing, metadata, and manifest storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

try:  # ffmpeg-python is optional at runtime; fallback to OpenCV metadata
    import ffmpeg  # type: ignore
except ImportError:  # pragma: no cover - guarded import for optional dep
    ffmpeg = None  # type: ignore

from .utils.hashing import sha256_hash_file

DEFAULT_MANIFEST_PATH = Path("data/manifests/ingest_manifest.json")


def ingest_video(
    video_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Compute hash + metadata for `video_path` and append to manifest."""
    source = Path(video_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")
    manifest = Path(manifest_path or DEFAULT_MANIFEST_PATH).expanduser().resolve()

    metadata = extract_video_metadata(source)
    entry = {
        "source_path": str(source),
        "sha256": sha256_hash_file(str(source)),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
    }
    append_manifest_entry(manifest, entry)
    return entry


def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Try ffprobe first, fall back to OpenCV-derived metadata."""
    if ffmpeg is not None:
        try:
            return _metadata_via_ffprobe(video_path)
        except RuntimeError:
            pass  # fallback below

    return _metadata_via_opencv(video_path)


def append_manifest_entry(manifest_path: Path, entry: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(manifest_path)
    entries.append(entry)
    manifest_path.write_text(json.dumps(entries, indent=2))


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text())
    if isinstance(data, list):
        return data
    raise ValueError(f"Manifest at {manifest_path} is not a JSON list.")


def _metadata_via_ffprobe(video_path: Path) -> Dict[str, Any]:
    if ffmpeg is None:
        raise RuntimeError("ffprobe unavailable")
    try:
        probe = ffmpeg.probe(str(video_path))
    except (ffmpeg.Error, FileNotFoundError) as exc:  # type: ignore[attr-defined]
        raise RuntimeError("ffprobe metadata extraction failed") from exc

    format_info = probe.get("format", {})
    streams = probe.get("streams", [])
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), {}
    )

    return {
        "metadata_backend": "ffprobe",
        "duration_seconds": _safe_float(format_info.get("duration")),
        "size_bytes": _safe_int(format_info.get("size")),
        "bit_rate": _safe_int(format_info.get("bit_rate")),
        "codec": video_stream.get("codec_name"),
        "width": _safe_int(video_stream.get("width")),
        "height": _safe_int(video_stream.get("height")),
        "frame_rate": _parse_frame_rate(
            video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
        ),
        "frame_count": _safe_int(video_stream.get("nb_frames")),
        "creation_time": _find_creation_time(format_info, video_stream),
    }


def _metadata_via_opencv(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for metadata extraction: {video_path}")

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = None
    if frame_rate and frame_rate > 0 and frame_count and frame_count > 0:
        duration = frame_count / frame_rate
    cap.release()

    return {
        "metadata_backend": "opencv",
        "duration_seconds": duration,
        "size_bytes": video_path.stat().st_size,
        "codec": None,
        "bit_rate": None,
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "frame_rate": frame_rate or None,
        "frame_count": int(frame_count) if frame_count else None,
        "creation_time": None,
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_frame_rate(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    if "/" in value:
        num, _, denom = value.partition("/")
        try:
            denom_value = float(denom)
            if denom_value == 0:
                return None
            return float(num) / denom_value
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _find_creation_time(
    format_info: Dict[str, Any], video_stream: Dict[str, Any]
) -> Optional[str]:
    format_tags = format_info.get("tags", {}) or {}
    stream_tags = video_stream.get("tags", {}) or {}
    return (
        stream_tags.get("creation_time")
        or format_tags.get("creation_time")
        or format_tags.get("com.apple.quicktime.creationdate")
    )


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "append_manifest_entry",
    "extract_video_metadata",
    "ingest_video",
    "load_manifest",
]
