"""Evidence export package builder.

Generates a court-ready ZIP file containing the forensic PDF report,
crop images, optional video clips, hash manifest, tamper detection
results, and a README with verification instructions.
"""

import io
import json
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from cctv_dissertation.utils.hashing import sha256_hash_file


@dataclass
class ExportConfig:
    """Configuration for evidence package generation."""

    pdf_bytes: bytes
    identities: dict  # From build_unified_identities()
    cameras: List[str]
    db_path: str
    tracked_output_dir: str
    project_root: str
    video_paths: Optional[List[str]] = None
    tamper_reports: Optional[List[dict]] = None
    include_db: bool = False
    include_clips: bool = False
    clips_dir: Optional[str] = None
    examiner_name: str = "System User"
    case_notes: str = ""


@dataclass
class ManifestEntry:
    """Single file in the evidence manifest."""

    path: str
    sha256: str
    size_bytes: int
    description: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "description": self.description,
        }


@dataclass
class EvidenceManifest:
    """Manifest of all files in the evidence package."""

    package_id: str
    created_at: str
    examiner: str
    case_notes: str
    files: List[ManifestEntry] = field(default_factory=list)
    source_video_hashes: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "package_id": self.package_id,
            "created_at": self.created_at,
            "examiner": self.examiner,
            "case_notes": self.case_notes,
            "files": [f.to_dict() for f in self.files],
            "source_video_hashes": self.source_video_hashes,
            "total_files": len(self.files),
            "total_size_bytes": sum(f.size_bytes for f in self.files),
        }


def build_evidence_package(config: ExportConfig) -> bytes:
    """Build a ZIP evidence package and return as bytes.

    Args:
        config: Export configuration with all required data.

    Returns:
        ZIP file as bytes, ready for st.download_button.
    """
    manifest = EvidenceManifest(
        package_id=str(uuid.uuid4()),
        created_at=datetime.now().isoformat(),
        examiner=config.examiner_name,
        case_notes=config.case_notes,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Forensic PDF report
        zf.writestr("forensic_report.pdf", config.pdf_bytes)
        manifest.files.append(
            ManifestEntry(
                path="forensic_report.pdf",
                sha256=_hash_bytes(config.pdf_bytes),
                size_bytes=len(config.pdf_bytes),
                description="Forensic analysis report with person/vehicle tracking",
            )
        )

        # 2. Crop images
        _add_crops(zf, manifest, config)

        # 3. Video clips (optional)
        if config.include_clips and config.clips_dir:
            _add_clips(zf, manifest, config.clips_dir)

        # 4. Tamper detection reports (if available)
        if config.tamper_reports:
            for i, tr in enumerate(config.tamper_reports):
                name = f"tamper_report_{i + 1}.json"
                data = json.dumps(tr, indent=2).encode("utf-8")
                zf.writestr(name, data)
                manifest.files.append(
                    ManifestEntry(
                        path=name,
                        sha256=_hash_bytes(data),
                        size_bytes=len(data),
                        description=f"Tamper detection results for video {i + 1}",
                    )
                )

        # 5. Database copy (optional)
        if config.include_db:
            db_path = Path(config.db_path)
            if db_path.exists():
                db_data = db_path.read_bytes()
                zf.writestr("tracker.db", db_data)
                manifest.files.append(
                    ManifestEntry(
                        path="tracker.db",
                        sha256=_hash_bytes(db_data),
                        size_bytes=len(db_data),
                        description="SQLite database with all tracking data",
                    )
                )

        # 6. Source video hashes (not included, just referenced)
        if config.video_paths:
            for vp in config.video_paths:
                if vp and Path(vp).exists():
                    manifest.source_video_hashes.append(
                        {
                            "filename": Path(vp).name,
                            "sha256": sha256_hash_file(vp),
                            "size_bytes": Path(vp).stat().st_size,
                            "note": "Source video not included (too large). "
                            "Hash provided for integrity verification.",
                        }
                    )

        # 7. Manifest (added near-last so it covers everything above)
        manifest_json = json.dumps(manifest.to_dict(), indent=2).encode("utf-8")
        zf.writestr("manifest.json", manifest_json)

        # 8. README
        readme = _generate_readme(config, manifest)
        zf.writestr("README.txt", readme.encode("utf-8"))

    buf.seek(0)
    return buf.getvalue()


def _hash_bytes(data: bytes) -> str:
    """Compute SHA-256 of in-memory bytes."""
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _resolve_crop(crop_path: str, project_root: str) -> Optional[Path]:
    """Resolve a crop path to an absolute path."""
    p = Path(crop_path)
    if not p.is_absolute():
        p = Path(project_root) / p
    return p if p.exists() else None


def _add_crops(
    zf: zipfile.ZipFile,
    manifest: EvidenceManifest,
    config: ExportConfig,
) -> None:
    """Add all crop images referenced in identities to the ZIP."""
    persons = config.identities.get("persons", [])
    vehicles = config.identities.get("vehicles", [])

    for identity in persons:
        uid = identity["unified_id"]
        raw = identity.get("raw_sightings", identity.get("sightings", []))
        for idx, s in enumerate(raw):
            crop = _resolve_crop(s["crop_path"], config.project_root)
            if crop:
                archive_name = (
                    f"crops/persons/P{uid}_sighting_{idx + 1}"
                    f"_{s['camera']}{crop.suffix}"
                )
                data = crop.read_bytes()
                zf.writestr(archive_name, data)
                manifest.files.append(
                    ManifestEntry(
                        path=archive_name,
                        sha256=_hash_bytes(data),
                        size_bytes=len(data),
                        description=(
                            f"Person {uid} sighting {idx + 1} on {s['camera']}"
                        ),
                    )
                )

    for identity in vehicles:
        uid = identity["unified_id"]
        sightings = identity.get("sightings", [])
        for idx, s in enumerate(sightings):
            crop = _resolve_crop(s["crop_path"], config.project_root)
            if crop:
                archive_name = (
                    f"crops/vehicles/V{uid}_sighting_{idx + 1}"
                    f"_{s['camera']}{crop.suffix}"
                )
                data = crop.read_bytes()
                zf.writestr(archive_name, data)
                manifest.files.append(
                    ManifestEntry(
                        path=archive_name,
                        sha256=_hash_bytes(data),
                        size_bytes=len(data),
                        description=(
                            f"Vehicle {uid} sighting {idx + 1} on {s['camera']}"
                        ),
                    )
                )


def _add_clips(
    zf: zipfile.ZipFile,
    manifest: EvidenceManifest,
    clips_dir: str,
) -> None:
    """Add generated video clips to the ZIP."""
    clips_path = Path(clips_dir)
    if not clips_path.exists():
        return
    for clip in sorted(clips_path.glob("*.mp4")):
        archive_name = f"clips/{clip.name}"
        data = clip.read_bytes()
        zf.writestr(archive_name, data)
        manifest.files.append(
            ManifestEntry(
                path=archive_name,
                sha256=_hash_bytes(data),
                size_bytes=len(data),
                description=f"Tracking video clip: {clip.stem}",
            )
        )


def _generate_readme(config: ExportConfig, manifest: EvidenceManifest) -> str:
    """Generate plain-text README for the evidence package."""
    now = datetime.now()
    n_files = len(manifest.files)
    total_mb = sum(f.size_bytes for f in manifest.files) / (1024 * 1024)

    lines = [
        "=" * 60,
        "FORENSIC VIDEO ANALYSIS - EVIDENCE PACKAGE",
        "=" * 60,
        "",
        f"Package ID:  {manifest.package_id}",
        f"Created:     {now:%Y-%m-%d %H:%M:%S}",
        f"Examiner:    {config.examiner_name}",
        f"Files:       {n_files}",
        f"Total Size:  {total_mb:.2f} MB",
        "",
    ]

    if config.case_notes:
        lines += [
            "CASE NOTES",
            "-" * 40,
            config.case_notes,
            "",
        ]

    lines += [
        "PACKAGE CONTENTS",
        "-" * 40,
        "",
        "forensic_report.pdf",
        "  Complete forensic analysis report including person and",
        "  vehicle identification, cross-camera matching, movement",
        "  timelines, and person-vehicle linkages.",
        "",
        "crops/",
        "  Cropped images of each identified person and vehicle",
        "  from every sighting, organised by type and identity.",
        "",
        "manifest.json",
        "  SHA-256 hashes of every file in this package for",
        "  integrity verification.",
        "",
    ]

    if config.tamper_reports:
        lines += [
            "tamper_report_*.json",
            "  Video tamper detection results including structural",
            "  analysis, frame quality checks, metadata verification,",
            "  and compression analysis.",
            "",
        ]

    if config.include_clips:
        lines += [
            "clips/",
            "  Generated video clips showing tracked entities",
            "  with bounding box annotations.",
            "",
        ]

    if config.include_db:
        lines += [
            "tracker.db",
            "  SQLite database containing all raw tracking data,",
            "  embeddings, and entity metadata.",
            "",
        ]

    lines += [
        "SOURCE VIDEO INTEGRITY",
        "-" * 40,
        "",
        "Source videos are NOT included in this package due to",
        "file size. Their SHA-256 hashes are recorded in",
        "manifest.json under 'source_video_hashes' for",
        "independent verification.",
        "",
    ]

    if manifest.source_video_hashes:
        for vh in manifest.source_video_hashes:
            lines += [
                f"  {vh['filename']}",
                f"  SHA-256: {vh['sha256']}",
                "",
            ]

    lines += [
        "VERIFICATION INSTRUCTIONS",
        "-" * 40,
        "",
        "To verify package integrity:",
        "",
        "1. Extract the ZIP archive",
        "2. Open manifest.json",
        "3. For each file listed, compute its SHA-256 hash:",
        "   sha256sum <filename>",
        "4. Compare against the hash in manifest.json",
        "5. All hashes should match exactly",
        "",
        "Note: manifest.json itself is not self-hashing.",
        "The manifest serves as the integrity anchor for all",
        "other files in the package.",
        "",
        "PROCESSING PIPELINE",
        "-" * 40,
        "",
        "Detection:     YOLOv8n (COCO pretrained)",
        "Tracking:      ByteTrack (per-camera)",
        "Person Re-ID:  OSNet x1_0 (torchreid)",
        "Vehicle Re-ID: OSNet x0_25 (torchreid)",
        "Color Analysis: K-means (k=5) with background filtering",
        "Cross-Camera:  Cosine similarity on Re-ID embeddings",
        "",
        "DISCLAIMER",
        "-" * 40,
        "",
        "This report is generated by an automated analysis",
        "pipeline. All findings should be verified by a",
        "qualified examiner before use as evidence.",
        "",
        "Report Status: PRELIMINARY - For investigative use.",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)
