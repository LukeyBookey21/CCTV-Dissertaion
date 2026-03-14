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


def merge_dvr_segments(
    dvr_folder: str | Path,
    output_path: str | Path,
    start_hour: int = 0,
    end_hour: int = 23,
    start_minute: int = 0,
    end_minute: int = 59,
    callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Merge DVR minute-segment files into a single MP4 for processing.

    DVR systems typically record as YYYYMMDD_camera/HH/MM.mp4.
    This function scans a camera folder, validates readable files in
    the requested time range, writes an ffmpeg concat list, and merges
    them with stream-copy (no re-encode, very fast).

    Args:
        dvr_folder:    Path to the camera folder (e.g. 20260217_garage)
        output_path:   Destination MP4 path for the merged file
        start_hour:    First hour to include (inclusive, 0-23)
        end_hour:      Last hour to include (inclusive, 0-23)
        start_minute:  Minute within start_hour to begin from (0-59)
        end_minute:    Minute within end_hour to stop at (0-59)
        callback:      Optional callable(progress_pct, message) for UI updates

    Returns:
        Dict with keys: output_path, total_files, skipped_files,
        duration_seconds, start_hour, end_hour
    """
    import subprocess
    import tempfile

    dvr_folder = Path(dvr_folder).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the full list of (hour, minute) tuples in range
    time_slots = []
    for hour in range(start_hour, end_hour + 1):
        for minute in range(60):
            if hour == start_hour and minute < start_minute:
                continue
            if hour == end_hour and minute > end_minute:
                continue
            time_slots.append((hour, minute))

    # Scan and validate all segment files in the time range
    valid_files: List[Path] = []
    skipped: List[str] = []
    total_expected = len(time_slots)

    for checked, (hour, minute) in enumerate(time_slots):
        seg = dvr_folder / f"{hour:02d}" / f"{minute:02d}.mp4"

        if not seg.exists():
            skipped.append(f"{hour:02d}:{minute:02d} - missing")
            continue

        cap = cv2.VideoCapture(str(seg))
        readable = cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) > 0
        cap.release()

        if readable:
            valid_files.append(seg)
        else:
            skipped.append(f"{hour:02d}:{minute:02d} - unreadable")

        if callback and checked % 60 == 0:
            pct = int(((checked + 1) / total_expected) * 40)
            callback(pct, f"Scanning {hour:02d}:{minute:02d}...")

    if not valid_files:
        raise RuntimeError(f"No readable video segments found in {dvr_folder}")

    if callback:
        callback(40, f"Found {len(valid_files)} valid segments, merging...")

    # Write ffmpeg concat list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        for f in valid_files:
            tmp.write(f"file '{f}'\n")
        concat_list = tmp.name

    # Merge with ffmpeg stream-copy (fast, no re-encode)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list,
        "-c:v",
        "copy",
        "-an",  # drop audio - DVR audio codecs often unsupported in MP4
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    Path(concat_list).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg merge failed: {result.stderr[-500:]}")

    if callback:
        callback(100, "Merge complete!")

    # Get output metadata
    cap = cv2.VideoCapture(str(output_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    return {
        "output_path": str(output_path),
        "total_files": len(valid_files),
        "skipped_files": len(skipped),
        "skipped_details": skipped[:10],  # first 10 only
        "duration_seconds": frames / fps if frames else None,
        "start_hour": start_hour,
        "end_hour": end_hour,
    }


def verify_video_integrity(
    video_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Verify video integrity by comparing current hash against manifest.

    Returns a dict with:
        - verified: bool (True if hash matches)
        - current_hash: str (newly computed)
        - stored_hash: str | None (from manifest)
        - match: str (one of "MATCH", "MISMATCH", "NOT_FOUND")
        - message: str (human-readable status)
    """
    source = Path(video_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    manifest = Path(manifest_path or DEFAULT_MANIFEST_PATH).expanduser().resolve()

    # Compute current hash
    current_hash = sha256_hash_file(str(source))

    # Search manifest for matching entry by path or hash
    entries = load_manifest(manifest)
    stored_entry = None

    for entry in entries:
        entry_path = Path(entry.get("source_path", "")).resolve()
        entry_hash = entry.get("sha256")

        if entry_path == source or entry_hash == current_hash:
            stored_entry = entry
            break

    if stored_entry is None:
        return {
            "verified": False,
            "current_hash": current_hash,
            "stored_hash": None,
            "match": "NOT_FOUND",
            "message": f"Video not found in manifest: {source}",
        }

    stored_hash = stored_entry.get("sha256")
    matches = current_hash == stored_hash

    return {
        "verified": matches,
        "current_hash": current_hash,
        "stored_hash": stored_hash,
        "match": "MATCH" if matches else "MISMATCH",
        "message": (
            f"✓ Integrity verified: {source}"
            if matches
            else f"✗ Hash mismatch! File may be corrupted or tampered: {source}"
        ),
        "ingested_at": stored_entry.get("ingested_at"),
    }


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "append_manifest_entry",
    "extract_video_metadata",
    "ingest_video",
    "load_manifest",
    "verify_video_integrity",
]
