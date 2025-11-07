"""Persistence helpers for detections and investigator queries."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_DB_PATH = Path("data/analysis.db")


CREATE_VIDEOS = """
CREATE TABLE IF NOT EXISTS videos (
    sha256 TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    model_path TEXT,
    confidence_threshold REAL,
    frame_stride INTEGER,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_FRAMES = """
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_sha TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp_seconds REAL,
    detections_count INTEGER,
    UNIQUE(video_sha, frame_index),
    FOREIGN KEY (video_sha) REFERENCES videos(sha256) ON DELETE CASCADE
);
"""

CREATE_DETECTIONS = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_sha TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    class_id INTEGER,
    class_name TEXT,
    confidence REAL,
    x1 REAL,
    y1 REAL,
    x2 REAL,
    y2 REAL,
    FOREIGN KEY (video_sha) REFERENCES videos(sha256) ON DELETE CASCADE
);
"""

CREATE_TRACKS = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_sha TEXT NOT NULL,
    track_label TEXT,
    class_name TEXT,
    start_frame INTEGER,
    end_frame INTEGER,
    start_time REAL,
    end_time REAL,
    detections_count INTEGER,
    UNIQUE(video_sha, track_label),
    FOREIGN KEY (video_sha) REFERENCES videos(sha256) ON DELETE CASCADE
);
"""

CREATE_TRACK_POINTS = """
CREATE TABLE IF NOT EXISTS track_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    frame_index INTEGER,
    timestamp_seconds REAL,
    confidence REAL,
    x1 REAL,
    y1 REAL,
    x2 REAL,
    y2 REAL,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(CREATE_VIDEOS)
    conn.execute(CREATE_FRAMES)
    conn.execute(CREATE_DETECTIONS)
    conn.execute(CREATE_TRACKS)
    conn.execute(CREATE_TRACK_POINTS)


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or DEFAULT_DB_PATH).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    _ensure_schema(conn)
    return conn


def import_detection_report(
    report_path: str | Path,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Persist detection JSON into SQLite for later querying."""
    report = json.loads(Path(report_path).read_text())
    sha256 = report.get("sha256")
    if not sha256:
        raise ValueError("Detection report missing 'sha256'")

    frames = report.get("detections", [])
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO videos (sha256, source_path, model_path, confidence_threshold,
                                frame_stride, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET
                source_path=excluded.source_path,
                model_path=excluded.model_path,
                confidence_threshold=excluded.confidence_threshold,
                frame_stride=excluded.frame_stride,
                metadata_json=excluded.metadata_json
            """,
            (
                sha256,
                report.get("source_path"),
                report.get("model_path"),
                report.get("confidence_threshold"),
                report.get("frame_stride"),
                json.dumps(report.get("metadata")),
            ),
        )
        conn.execute("DELETE FROM frames WHERE video_sha = ?", (sha256,))
        conn.execute("DELETE FROM detections WHERE video_sha = ?", (sha256,))

        frame_rows = []
        detection_rows = []
        for frame in frames:
            frame_index = frame.get("frame_index")
            timestamp = frame.get("timestamp_seconds")
            detections = frame.get("detections", [])
            frame_rows.append(
                (sha256, frame_index, timestamp, len(detections)),
            )
            for det in detections:
                detection_rows.append(
                    (
                        sha256,
                        frame_index,
                        det.get("class_id"),
                        det.get("class_name"),
                        det.get("confidence"),
                        _bbox_value(det, 0),
                        _bbox_value(det, 1),
                        _bbox_value(det, 2),
                        _bbox_value(det, 3),
                    )
                )

        conn.executemany(
            """
            INSERT INTO frames (video_sha, frame_index, timestamp_seconds, detections_count)
            VALUES (?, ?, ?, ?)
            """,
            frame_rows,
        )
        conn.executemany(
            """
            INSERT INTO detections (
                video_sha, frame_index, class_id, class_name,
                confidence, x1, y1, x2, y2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            detection_rows,
        )

    return {
        "sha256": sha256,
        "frames_imported": len(frames),
        "detections_imported": sum(row[3] for row in frame_rows),
    }


def query_detections(
    sha256: str,
    db_path: str | Path | None = None,
    class_name: Optional[str] = None,
    min_conf: float = 0.0,
    time_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> List[Dict[str, Any]]:
    """Return detections filtered by class, confidence, and time bounds."""
    sql = [
        """
        SELECT d.frame_index, f.timestamp_seconds, d.class_name, d.class_id,
               d.confidence, d.x1, d.y1, d.x2, d.y2
        FROM detections d
        JOIN frames f
            ON d.video_sha = f.video_sha AND d.frame_index = f.frame_index
        WHERE d.video_sha = ?
        """
    ]
    params: List[Any] = [sha256]
    if class_name:
        sql.append("AND d.class_name = ?")
        params.append(class_name)
    if min_conf:
        sql.append("AND d.confidence >= ?")
        params.append(min_conf)
    if time_range:
        start, end = time_range
        if start is not None:
            sql.append("AND f.timestamp_seconds >= ?")
            params.append(start)
        if end is not None:
            sql.append("AND f.timestamp_seconds <= ?")
            params.append(end)
    sql.append("ORDER BY f.timestamp_seconds ASC")

    with get_connection(db_path) as conn:
        rows = conn.execute(" ".join(sql), params).fetchall()

    return [
        {
            "frame_index": row[0],
            "timestamp_seconds": row[1],
            "class_name": row[2],
            "class_id": row[3],
            "confidence": row[4],
            "bbox_xyxy": [row[5], row[6], row[7], row[8]],
        }
        for row in rows
    ]


def store_tracks(
    sha256: str,
    tracks: Sequence[Dict[str, Any]],
    db_path: str | Path | None = None,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM track_points WHERE track_id IN (SELECT id FROM tracks WHERE video_sha = ?)", (sha256,))
        conn.execute("DELETE FROM tracks WHERE video_sha = ?", (sha256,))
        for track in tracks:
            cursor = conn.execute(
                """
                INSERT INTO tracks (
                    video_sha, track_label, class_name,
                    start_frame, end_frame, start_time, end_time, detections_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sha256,
                    f"track-{track.get('track_id')}",
                    track.get("class_name"),
                    track.get("start_frame"),
                    track.get("end_frame"),
                    track.get("start_time"),
                    track.get("end_time"),
                    len(track.get("detections", [])),
                ),
            )
            track_db_id = cursor.lastrowid
            points = [
                (
                    track_db_id,
                    point.get("frame_index"),
                    point.get("timestamp_seconds"),
                    point.get("confidence"),
                    _bbox_value(point, 0),
                    _bbox_value(point, 1),
                    _bbox_value(point, 2),
                    _bbox_value(point, 3),
                )
                for point in track.get("detections", [])
            ]
            conn.executemany(
                """
                INSERT INTO track_points (
                    track_id, frame_index, timestamp_seconds,
                    confidence, x1, y1, x2, y2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                points,
            )


def query_tracks(
    sha256: str,
    db_path: str | Path | None = None,
    class_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sql = ["SELECT id, track_label, class_name, start_frame, end_frame, start_time, end_time, detections_count FROM tracks WHERE video_sha = ?"]
    params: List[Any] = [sha256]
    if class_name:
        sql.append("AND class_name = ?")
        params.append(class_name)
    sql.append("ORDER BY start_time ASC")
    with get_connection(db_path) as conn:
        rows = conn.execute(" ".join(sql), params).fetchall()
        track_ids = [row[0] for row in rows]
        points_map = {}
        if track_ids:
            placeholders = ",".join(["?"] * len(track_ids))
            point_rows = conn.execute(
                f"""
                SELECT track_id, frame_index, timestamp_seconds, confidence, x1, y1, x2, y2
                FROM track_points
                WHERE track_id IN ({placeholders})
                ORDER BY timestamp_seconds ASC
                """,
                track_ids,
            ).fetchall()
            for row in point_rows:
                points_map.setdefault(row[0], []).append(
                    {
                        "frame_index": row[1],
                        "timestamp_seconds": row[2],
                        "confidence": row[3],
                        "bbox_xyxy": [row[4], row[5], row[6], row[7]],
                    }
                )

    results = []
    for row in rows:
        results.append(
            {
                "track_label": row[1],
                "class_name": row[2],
                "start_frame": row[3],
                "end_frame": row[4],
                "start_time": row[5],
                "end_time": row[6],
                "detections_count": row[7],
                "points": points_map.get(row[0], []),
            }
        )
    return results


def _bbox_value(det: Dict[str, Any], idx: int) -> Optional[float]:
    bbox = det.get("bbox_xyxy") or []
    try:
        return float(bbox[idx])
    except (IndexError, TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_DB_PATH",
    "get_connection",
    "import_detection_report",
    "query_detections",
    "store_tracks",
    "query_tracks",
]
