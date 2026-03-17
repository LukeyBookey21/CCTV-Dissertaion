"""Video tamper detection module.

Analyses video files for signs of tampering: frame drops, splices,
re-encoding artefacts, timestamp discontinuities, and compression
anomalies. Produces a structured report with confidence scores.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Data Structures ──────────────────────────────────────────────


@dataclass
class TamperFlag:
    """A single detected anomaly."""

    category: str  # pts_gap, quality_shift, gop_anomaly, etc.
    severity: str  # info, warning, critical
    confidence: float  # 0.0 - 1.0
    timestamp_sec: Optional[float]  # Where in video (None = file-level)
    frame_index: Optional[int]
    description: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "confidence": self.confidence,
            "timestamp_sec": self.timestamp_sec,
            "frame_index": self.frame_index,
            "description": self.description,
            "details": self.details,
        }


@dataclass
class TamperReport:
    """Complete tamper detection report for one video."""

    video_path: str
    sha256: str
    analysis_date: str
    duration_seconds: float
    total_frames: int
    flags: List[TamperFlag] = field(default_factory=list)
    structural_summary: dict = field(default_factory=dict)
    quality_summary: dict = field(default_factory=dict)
    metadata_summary: dict = field(default_factory=dict)
    compression_summary: dict = field(default_factory=dict)
    segment_hashes: List[dict] = field(default_factory=list)
    overall_risk: str = "clean"  # clean, low, medium, high
    overall_confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "sha256": self.sha256,
            "analysis_date": self.analysis_date,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "flags": [f.to_dict() for f in self.flags],
            "structural_summary": self.structural_summary,
            "quality_summary": self.quality_summary,
            "metadata_summary": self.metadata_summary,
            "compression_summary": self.compression_summary,
            "segment_hashes": self.segment_hashes,
            "overall_risk": self.overall_risk,
            "overall_confidence": self.overall_confidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "info")


# ── Main Orchestrator ────────────────────────────────────────────


def analyze_video(
    video_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> TamperReport:
    """Run full tamper detection analysis on a video file.

    Args:
        video_path: Path to video file.
        progress_callback: Optional (progress_0_to_1, status_text) callback.

    Returns:
        TamperReport with all findings.
    """
    from cctv_dissertation.utils.hashing import sha256_hash_file

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    def _prog(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    _prog(0.0, "Computing file hash...")
    sha = sha256_hash_file(str(path))

    # Get basic video info
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    report = TamperReport(
        video_path=str(path),
        sha256=sha,
        analysis_date=datetime.now().isoformat(),
        duration_seconds=duration,
        total_frames=total_frames,
    )

    # 1. Structural analysis (ffprobe-based)
    _prog(0.10, "Analysing packet structure...")
    struct_summary, struct_flags = _analyze_structure(str(path), fps)
    report.structural_summary = struct_summary
    report.flags.extend(struct_flags)

    # 2. Frame quality analysis (SSIM sampling)
    _prog(0.30, "Analysing frame quality...")
    qual_summary, qual_flags, ssim_data = _analyze_frame_quality(
        str(path),
        fps,
        total_frames,
        progress_callback=lambda p, m: _prog(0.30 + p * 0.35, m),
    )
    report.quality_summary = qual_summary
    report.quality_summary["ssim_timeline"] = ssim_data
    report.flags.extend(qual_flags)

    # 3. Metadata consistency
    _prog(0.65, "Checking metadata consistency...")
    meta_summary, meta_flags = _analyze_metadata(str(path), fps, total_frames)
    report.metadata_summary = meta_summary
    report.flags.extend(meta_flags)

    # 4. Compression analysis (ELA)
    _prog(0.75, "Running compression analysis...")
    comp_summary, comp_flags = _analyze_compression(str(path), total_frames)
    report.compression_summary = comp_summary
    report.flags.extend(comp_flags)

    # 5. Segment hashes
    _prog(0.85, "Computing segment hashes...")
    report.segment_hashes = _compute_segment_hashes(str(path), duration)

    # 6. Overall risk
    _prog(0.95, "Classifying overall risk...")
    report.overall_risk, report.overall_confidence = _classify_risk(report.flags)

    _prog(1.0, "Analysis complete.")
    return report


# ── Structural Analysis ──────────────────────────────────────────


def _analyze_structure(video_path: str, fps: float) -> Tuple[dict, List[TamperFlag]]:
    """Analyse PTS/DTS timestamps and GOP patterns via ffprobe."""
    flags: List[TamperFlag] = []
    summary: Dict = {}

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,duration_time,flags",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            summary["error"] = "ffprobe failed"
            return summary, flags
    except (subprocess.TimeoutExpired, FileNotFoundError):
        summary["error"] = "ffprobe unavailable or timed out"
        return summary, flags

    pts_values = []
    durations = []
    keyframe_positions = []

    for i, line in enumerate(result.stdout.strip().split("\n")):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            pts = float(parts[0]) if parts[0] != "N/A" else None
            dur = float(parts[1]) if parts[1] != "N/A" else None
            is_key = "K" in parts[2]
        except (ValueError, IndexError):
            continue

        if pts is not None:
            pts_values.append(pts)
        if dur is not None:
            durations.append(dur)
        if is_key:
            keyframe_positions.append(i)

    if len(pts_values) < 2:
        summary["packet_count"] = len(pts_values)
        summary["error"] = "insufficient packets"
        return summary, flags

    # PTS gap analysis
    expected_dur = 1.0 / fps if fps > 0 else 0.05
    pts_deltas = np.diff(pts_values)
    median_delta = float(np.median(pts_deltas))
    gap_threshold = max(expected_dur * 3.0, 0.15)  # 3x expected or 150ms

    large_gaps = []
    for i, delta in enumerate(pts_deltas):
        if delta > gap_threshold:
            large_gaps.append(
                {
                    "index": i,
                    "pts": pts_values[i],
                    "gap_sec": float(delta),
                    "expected": expected_dur,
                }
            )

    for g in large_gaps:
        severity = "critical" if g["gap_sec"] > 1.0 else "warning"
        flags.append(
            TamperFlag(
                category="pts_gap",
                severity=severity,
                confidence=min(g["gap_sec"] / gap_threshold, 1.0),
                timestamp_sec=g["pts"],
                frame_index=g["index"],
                description=(
                    f"Frame gap of {g['gap_sec']:.3f}s at "
                    f"{g['pts']:.2f}s (expected ~{expected_dur:.3f}s)"
                ),
                details=g,
            )
        )

    # Duration consistency
    if durations:
        dur_std = float(np.std(durations))
        dur_mean = float(np.mean(durations))
        if dur_std > expected_dur * 0.5 and dur_mean > 0:
            flags.append(
                TamperFlag(
                    category="duration_variance",
                    severity="warning",
                    confidence=min(dur_std / expected_dur, 1.0),
                    timestamp_sec=None,
                    frame_index=None,
                    description=(
                        f"High frame duration variance: "
                        f"mean={dur_mean:.4f}s, std={dur_std:.4f}s"
                    ),
                    details={"mean": dur_mean, "std": dur_std},
                )
            )

    # GOP pattern analysis
    if len(keyframe_positions) >= 3:
        gop_lengths = np.diff(keyframe_positions)
        gop_std = float(np.std(gop_lengths))
        gop_mean = float(np.mean(gop_lengths))
        gop_cv = gop_std / gop_mean if gop_mean > 0 else 0

        if gop_cv > 0.5:  # High variation in GOP length
            flags.append(
                TamperFlag(
                    category="gop_anomaly",
                    severity="warning",
                    confidence=min(gop_cv, 1.0),
                    timestamp_sec=None,
                    frame_index=None,
                    description=(
                        f"Irregular GOP pattern: "
                        f"mean={gop_mean:.1f} frames, CV={gop_cv:.2f}"
                    ),
                    details={
                        "gop_mean": gop_mean,
                        "gop_std": gop_std,
                        "gop_cv": gop_cv,
                    },
                )
            )

    summary = {
        "packet_count": len(pts_values),
        "duration_range": (
            (float(pts_values[0]), float(pts_values[-1])) if pts_values else (0, 0)
        ),
        "median_frame_duration": median_delta,
        "expected_frame_duration": expected_dur,
        "large_gaps_found": len(large_gaps),
        "keyframe_count": len(keyframe_positions),
        "gop_mean_length": (
            float(np.mean(np.diff(keyframe_positions)))
            if len(keyframe_positions) >= 2
            else 0
        ),
    }

    return summary, flags


# ── Frame Quality Analysis ───────────────────────────────────────


def _ssim_fast(a: np.ndarray, b: np.ndarray) -> float:
    """Compute simplified SSIM between two grayscale frames."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    mu_a = np.mean(a_f)
    mu_b = np.mean(b_f)
    sigma_a_sq = np.var(a_f)
    sigma_b_sq = np.var(b_f)
    sigma_ab = np.mean((a_f - mu_a) * (b_f - mu_b))
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a_sq + sigma_b_sq + C2)
    return float(num / den) if den > 0 else 0.0


def _analyze_frame_quality(
    video_path: str,
    fps: float,
    total_frames: int,
    sample_rate: int = 100,
    progress_callback: Optional[Callable] = None,
) -> Tuple[dict, List[TamperFlag], List[dict]]:
    """Sample frames and compute SSIM to detect quality discontinuities."""
    flags: List[TamperFlag] = []
    ssim_timeline: List[dict] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "cannot open video"}, flags, ssim_timeline

    prev_gray = None
    ssim_values = []
    frame_indices = list(range(0, total_frames, sample_rate))
    n_samples = len(frame_indices)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downscale for speed
        gray = cv2.resize(gray, (320, 180))

        if prev_gray is not None:
            ssim_val = _ssim_fast(prev_gray, gray)
            ts = frame_idx / fps if fps > 0 else frame_idx
            ssim_values.append(ssim_val)
            ssim_timeline.append(
                {
                    "frame": frame_idx,
                    "timestamp": round(ts, 2),
                    "ssim": round(ssim_val, 4),
                }
            )

        prev_gray = gray

        if progress_callback and i % 50 == 0:
            progress_callback(i / max(n_samples, 1), "Sampling frames...")

    cap.release()

    if not ssim_values:
        return {"error": "no frames sampled"}, flags, ssim_timeline

    ssim_arr = np.array(ssim_values)
    mean_ssim = float(np.mean(ssim_arr))
    std_ssim = float(np.std(ssim_arr))

    # Detect sudden drops using two methods:
    # 1. Absolute: SSIM < mean - 3*std (catches large drops)
    # 2. Relative: SSIM deviates > 5*std from mean (catches
    #    subtle splices in stable CCTV where std is tiny)
    abs_threshold = max(mean_ssim - 3 * std_ssim, 0.3)
    # For very stable video (std < 0.005), use a tighter check
    rel_threshold = mean_ssim - max(5 * std_ssim, 0.01)
    for entry in ssim_timeline:
        ssim = entry["ssim"]
        is_abs_drop = ssim < abs_threshold and ssim < 0.5
        is_rel_drop = ssim < rel_threshold and std_ssim < 0.005
        if is_abs_drop or is_rel_drop:
            deviation = (mean_ssim - ssim) / max(std_ssim, 1e-6)
            flags.append(
                TamperFlag(
                    category="quality_shift",
                    severity="critical" if ssim < 0.5 or deviation > 10 else "warning",
                    confidence=min(deviation / 10.0, 1.0),
                    timestamp_sec=entry["timestamp"],
                    frame_index=entry["frame"],
                    description=(
                        f"Sudden quality drop at {entry['timestamp']:.1f}s "
                        f"(SSIM={ssim:.4f}, expected ~{mean_ssim:.4f}, "
                        f"{deviation:.1f} std deviations)"
                    ),
                    details={
                        "ssim": ssim,
                        "threshold": rel_threshold if is_rel_drop else abs_threshold,
                        "deviation_sigma": round(deviation, 1),
                    },
                )
            )

    summary = {
        "frames_sampled": len(ssim_values),
        "sample_rate": sample_rate,
        "mean_ssim": round(mean_ssim, 4),
        "std_ssim": round(std_ssim, 4),
        "min_ssim": round(float(np.min(ssim_arr)), 4),
        "max_ssim": round(float(np.max(ssim_arr)), 4),
        "quality_drops_found": sum(1 for f in flags if f.category == "quality_shift"),
    }

    return summary, flags, ssim_timeline


# ── Metadata Consistency ─────────────────────────────────────────


def _analyze_metadata(
    video_path: str, fps: float, total_frames: int
) -> Tuple[dict, List[TamperFlag]]:
    """Cross-check container metadata against stream properties."""
    flags: List[TamperFlag] = []
    summary: Dict = {}

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_format",
            "-show_streams",
            "-select_streams",
            "v:0",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"error": "ffprobe failed"}, flags
        probe = json.loads(result.stdout)
    except Exception:
        return {"error": "ffprobe unavailable"}, flags

    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    vs = streams[0] if streams else {}

    # Container duration vs computed duration
    container_dur = float(fmt.get("duration", 0))
    computed_dur = total_frames / fps if fps > 0 else 0
    if container_dur > 0 and computed_dur > 0:
        dur_diff = abs(container_dur - computed_dur)
        if dur_diff > max(container_dur * 0.02, 1.0):
            flags.append(
                TamperFlag(
                    category="metadata_mismatch",
                    severity="warning",
                    confidence=min(dur_diff / container_dur, 1.0),
                    timestamp_sec=None,
                    frame_index=None,
                    description=(
                        f"Duration mismatch: container={container_dur:.1f}s, "
                        f"computed={computed_dur:.1f}s (diff={dur_diff:.1f}s)"
                    ),
                    details={
                        "container_duration": container_dur,
                        "computed_duration": computed_dur,
                    },
                )
            )

    # Bitrate consistency
    file_size = float(fmt.get("size", 0))
    declared_bitrate = float(fmt.get("bit_rate", 0))
    if file_size > 0 and container_dur > 0:
        actual_bitrate = (file_size * 8) / container_dur
        if declared_bitrate > 0:
            br_diff = abs(actual_bitrate - declared_bitrate) / declared_bitrate
            if br_diff > 0.15:
                flags.append(
                    TamperFlag(
                        category="metadata_mismatch",
                        severity="info",
                        confidence=min(br_diff, 1.0),
                        timestamp_sec=None,
                        frame_index=None,
                        description=(
                            f"Bitrate mismatch: declared="
                            f"{declared_bitrate / 1000:.0f}kbps, "
                            f"actual={actual_bitrate / 1000:.0f}kbps"
                        ),
                        details={
                            "declared_bps": declared_bitrate,
                            "actual_bps": actual_bitrate,
                        },
                    )
                )

    # Frame count check
    stream_frames = int(vs.get("nb_frames", 0) or 0)
    if stream_frames > 0 and total_frames > 0:
        fc_diff = abs(stream_frames - total_frames)
        if fc_diff > max(total_frames * 0.01, 5):
            flags.append(
                TamperFlag(
                    category="metadata_mismatch",
                    severity="warning",
                    confidence=min(fc_diff / total_frames, 1.0),
                    timestamp_sec=None,
                    frame_index=None,
                    description=(
                        f"Frame count mismatch: metadata={stream_frames}, "
                        f"decoded={total_frames}"
                    ),
                    details={
                        "metadata_frames": stream_frames,
                        "decoded_frames": total_frames,
                    },
                )
            )

    summary = {
        "container_format": fmt.get("format_name", "unknown"),
        "codec": vs.get("codec_name", "unknown"),
        "container_duration": container_dur,
        "computed_duration": computed_dur,
        "file_size_mb": file_size / (1024 * 1024) if file_size else 0,
        "declared_bitrate_kbps": declared_bitrate / 1000 if declared_bitrate else 0,
        "stream_frame_count": stream_frames,
    }

    return summary, flags


# ── Compression Analysis (ELA) ───────────────────────────────────


def _analyze_compression(
    video_path: str, total_frames: int, n_samples: int = 20
) -> Tuple[dict, List[TamperFlag]]:
    """Error Level Analysis on sampled frames to detect double compression."""
    flags: List[TamperFlag] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "cannot open video"}, flags

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    step = max(total_frames // n_samples, 1)
    ela_scores = []

    for i in range(n_samples):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Encode at known quality, decode, compute difference
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        _, encoded = cv2.imencode(".jpg", frame, encode_params)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        diff = cv2.absdiff(frame, recompressed)
        ela_score = float(np.mean(diff))
        ts = frame_idx / fps
        ela_scores.append(
            {
                "frame": frame_idx,
                "timestamp": round(ts, 2),
                "ela_score": round(ela_score, 3),
            }
        )

    cap.release()

    if not ela_scores:
        return {"error": "no frames analysed"}, flags

    scores = np.array([e["ela_score"] for e in ela_scores])
    mean_ela = float(np.mean(scores))
    std_ela = float(np.std(scores))

    # Flag frames with significantly different ELA (potential re-encoding)
    for entry in ela_scores:
        if std_ela > 0:
            z_score = abs(entry["ela_score"] - mean_ela) / std_ela
            if z_score > 3.0:
                flags.append(
                    TamperFlag(
                        category="compression_anomaly",
                        severity="warning",
                        confidence=min(z_score / 5.0, 1.0),
                        timestamp_sec=entry["timestamp"],
                        frame_index=entry["frame"],
                        description=(
                            f"Unusual compression level at "
                            f"{entry['timestamp']:.1f}s "
                            f"(ELA={entry['ela_score']:.2f}, "
                            f"mean={mean_ela:.2f})"
                        ),
                        details={
                            "ela_score": entry["ela_score"],
                            "z_score": round(z_score, 2),
                        },
                    )
                )

    summary = {
        "frames_analysed": len(ela_scores),
        "mean_ela": round(mean_ela, 3),
        "std_ela": round(std_ela, 3),
        "min_ela": round(float(np.min(scores)), 3),
        "max_ela": round(float(np.max(scores)), 3),
        "anomalies_found": len(flags),
    }

    return summary, flags


# ── Segment Hashes ───────────────────────────────────────────────


def _compute_segment_hashes(
    video_path: str, duration: float, segment_duration: float = 60.0
) -> List[dict]:
    """Compute SHA-256 hashes of fixed byte-range segments."""
    path = Path(video_path)
    file_size = path.stat().st_size
    n_segments = max(int(duration / segment_duration), 1)
    segment_bytes = file_size // n_segments

    segments = []
    with open(video_path, "rb") as f:
        for i in range(n_segments):
            start_byte = i * segment_bytes
            end_byte = min((i + 1) * segment_bytes, file_size)
            f.seek(start_byte)
            data = f.read(end_byte - start_byte)

            h = hashlib.sha256(data).hexdigest()
            segments.append(
                {
                    "segment": i,
                    "start_sec": round(i * segment_duration, 1),
                    "end_sec": round(min((i + 1) * segment_duration, duration), 1),
                    "start_byte": start_byte,
                    "end_byte": end_byte,
                    "sha256": h,
                }
            )

    return segments


# ── Risk Classification ──────────────────────────────────────────


def _classify_risk(flags: List[TamperFlag]) -> Tuple[str, float]:
    """Aggregate flags into overall risk level."""
    if not flags:
        return "clean", 1.0

    critical = sum(1 for f in flags if f.severity == "critical")
    warnings = sum(1 for f in flags if f.severity == "warning")

    if critical >= 2:
        return "high", min(0.5 + critical * 0.1, 1.0)
    elif critical == 1:
        return "medium", 0.7
    elif warnings >= 3:
        return "medium", 0.6
    elif warnings >= 1:
        return "low", 0.5
    else:
        return "clean", 0.9  # Info-only flags
