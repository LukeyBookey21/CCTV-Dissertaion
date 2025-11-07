"""Minimal CLI entrypoint (package namespace)."""

import argparse
import json

from cctv_dissertation.analysis import format_summary, summarize_detection_report
from cctv_dissertation.ingest import DEFAULT_MANIFEST_PATH, ingest_video
from cctv_dissertation.storage import DEFAULT_DB_PATH
from cctv_dissertation.utils.hashing import sha256_hash_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forensics video analysis toolbox (skeleton)",
    )
    subparsers = parser.add_subparsers(dest="command")

    hash_parser = subparsers.add_parser(
        "hash",
        help="Compute a SHA-256 hash for an arbitrary file",
    )
    hash_parser.add_argument("file", help="Path to the file to hash")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a video: hash + metadata + manifest update",
    )
    ingest_parser.add_argument("file", help="Path to the video file to ingest")
    ingest_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help=f"Manifest output path (default: {DEFAULT_MANIFEST_PATH})",
    )

    detect_parser = subparsers.add_parser(
        "detect",
        help="Run YOLOv8 detections against a video",
    )
    detect_parser.add_argument("file", help="Path to the video file to analyze")
    detect_parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    detect_parser.add_argument(
        "--conf", type=float, default=0.3, help="Confidence threshold"
    )
    detect_parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Analyze every Nth frame (vid_stride)",
    )
    detect_parser.add_argument("--device", default="cpu", help="Computation device")
    detect_parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size"
    )
    detect_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames processed (for quick tests)",
    )
    detect_parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to data/detections/<sha256>.json",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Summarize an existing detection report",
    )
    analyze_parser.add_argument(
        "report",
        help="Path to detection JSON produced by the detect command",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text",
    )

    store_parser = subparsers.add_parser(
        "store",
        help="Persist detection JSON into SQLite for future queries",
    )
    store_parser.add_argument("report", help="Path to detection JSON file")
    store_parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite destination (default: {DEFAULT_DB_PATH})",
    )

    track_parser = subparsers.add_parser(
        "track",
        help="Assign track IDs to detections and optionally store them",
    )
    track_parser.add_argument("report", help="Path to detection JSON file")
    track_parser.add_argument("--iou", type=float, default=0.3, help="IoU match threshold")
    track_parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Allow linking detections separated by up to N frames",
    )
    track_parser.add_argument(
        "--store",
        action="store_true",
        help="Persist tracks into SQLite (requires --db)",
    )
    track_parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite destination (default: {DEFAULT_DB_PATH})",
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Query stored detections or tracks for investigator workflows",
    )
    query_parser.add_argument("sha256", help="Video hash to query")
    query_parser.add_argument(
        "--class",
        dest="class_name",
        help="Filter by class name (e.g., car, person)",
    )
    query_parser.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="Minimum confidence threshold",
    )
    query_parser.add_argument(
        "--time-start",
        type=float,
        default=None,
        help="Earliest timestamp (seconds)",
    )
    query_parser.add_argument(
        "--time-end",
        type=float,
        default=None,
        help="Latest timestamp (seconds)",
    )
    query_parser.add_argument(
        "--tracks",
        action="store_true",
        help="Return track summaries instead of raw detections",
    )
    query_parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite source (default: {DEFAULT_DB_PATH})",
    )

    args = parser.parse_args()

    if args.command == "hash":
        h = sha256_hash_file(args.file)
        print(f"SHA-256: {h}")
    elif args.command == "ingest":
        entry = ingest_video(args.file, args.manifest)
        print(json.dumps(entry, indent=2))
    elif args.command == "detect":
        from cctv_dissertation.detection import (
            run_yolo_detection,
            write_detection_report,
        )

        report = run_yolo_detection(
            args.file,
            model_path=args.model,
            conf=args.conf,
            frame_stride=args.frame_stride,
            device=args.device,
            imgsz=args.imgsz,
            max_frames=args.max_frames,
        )
        output_path = write_detection_report(report, args.output)
        summary = {
            "output_path": str(output_path),
            "frames_processed": len(report["detections"]),
            "sha256": report["sha256"],
        }
        print(json.dumps(summary, indent=2))
    elif args.command == "analyze":
        summary = summarize_detection_report(args.report)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(format_summary(summary))
    elif args.command == "store":
        from cctv_dissertation.storage import import_detection_report

        result = import_detection_report(args.report, args.db)
        print(json.dumps(result, indent=2))
    elif args.command == "track":
        from cctv_dissertation.tracking import generate_tracks

        track_result = generate_tracks(
            args.report,
            iou_threshold=args.iou,
            max_frame_gap=args.max_gap,
        )
        summary = {
            "sha256": track_result["sha256"],
            "tracks_found": len(track_result["tracks"]),
        }
        if args.store:
            from cctv_dissertation.storage import store_tracks

            store_tracks(track_result["sha256"], track_result["tracks"], args.db)
            summary["stored"] = True
            summary["db_path"] = args.db
        print(json.dumps(summary, indent=2))
    elif args.command == "query":
        from cctv_dissertation.storage import query_detections, query_tracks

        time_range = None
        if args.time_start is not None or args.time_end is not None:
            time_range = (args.time_start, args.time_end)
        if args.tracks:
            results = query_tracks(
                args.sha256,
                db_path=args.db,
                class_name=args.class_name,
            )
            payload = {
                "mode": "tracks",
                "sha256": args.sha256,
                "count": len(results),
                "results": results,
            }
        else:
            results = query_detections(
                args.sha256,
                db_path=args.db,
                class_name=args.class_name,
                min_conf=args.min_conf,
                time_range=time_range,
            )
            payload = {
                "mode": "detections",
                "sha256": args.sha256,
                "count": len(results),
                "results": results,
            }
        print(json.dumps(payload, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
