"""FastAPI server exposing ingestion/detection/query endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .analysis import summarize_detection_report
from .detection import run_yolo_detection, write_detection_report
from .ingest import ingest_video
from .storage import (
    DEFAULT_DB_PATH,
    import_detection_report,
    query_detections,
    query_tracks,
    store_tracks,
)
from .tracking import generate_tracks

DETECTIONS_DIR = Path("data/detections")
UPLOADS_DIR = Path("data/uploads")

app = FastAPI(
    title="Forensic Video Analysis API",
    description="Upload footage, run detections, and query stored results.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _detection_path_from_sha(sha: str) -> Path:
    path = DETECTIONS_DIR / f"{sha}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Detection report not found")
    return path


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/detections")
def list_detections():
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted(DETECTIONS_DIR.glob("*.json")):
        try:
            summary = summarize_detection_report(path)
            reports.append(
                {
                    "sha256": summary["sha256"],
                    "source_path": summary["source_path"],
                    "frames": summary["frames_in_report"],
                    "detections": summary["detections_total"],
                    "model": summary["model_path"],
                }
            )
        except Exception:
            continue
    return {"count": len(reports), "reports": reports}


@app.get("/summary/{sha}")
def get_summary(sha: str):
    path = _detection_path_from_sha(sha)
    return summarize_detection_report(path)


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    store: bool = Query(False),
    track: bool = Query(False),
    db_path: str = Query(str(DEFAULT_DB_PATH)),
):
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOADS_DIR / file.filename
    data = await file.read()
    target.write_bytes(data)

    ingest_video(target)
    report = run_yolo_detection(str(target))
    report_path = write_detection_report(report)

    response = {
        "sha256": report["sha256"],
        "detection_report": str(report_path),
        "frames": len(report["detections"]),
    }

    if store:
        store_result = import_detection_report(report_path, db_path)
        response["store"] = store_result
    if track:
        tracks = generate_tracks(report_path)
        store_tracks(report["sha256"], tracks["tracks"], db_path)
        response["tracks_stored"] = len(tracks["tracks"])

    return response


@app.get("/query/detections")
def api_query_detections(
    sha: str,
    class_name: Optional[str] = None,
    min_conf: float = 0.0,
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    db_path: str = str(DEFAULT_DB_PATH),
):
    time_range = None
    if time_start is not None or time_end is not None:
        time_range = (time_start, time_end)
    data = query_detections(
        sha,
        db_path=db_path,
        class_name=class_name,
        min_conf=min_conf,
        time_range=time_range,
    )
    return {"count": len(data), "results": data}


@app.get("/query/tracks")
def api_query_tracks(
    sha: str,
    class_name: Optional[str] = None,
    db_path: str = str(DEFAULT_DB_PATH),
):
    data = query_tracks(
        sha,
        db_path=db_path,
        class_name=class_name,
    )
    return {"count": len(data), "results": data}
