"""Minimal CLI entrypoint (package namespace)."""

import argparse
import json

from cctv_dissertation.analysis import format_summary, summarize_detection_report
from cctv_dissertation.ingest import (
    DEFAULT_MANIFEST_PATH,
    ingest_video,
    verify_video_integrity,
)
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

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify video integrity against stored hash in manifest",
    )
    verify_parser.add_argument("file", help="Path to the video file to verify")
    verify_parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help=f"Manifest source path (default: {DEFAULT_MANIFEST_PATH})",
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
    detect_parser.add_argument(
        "--motion",
        action="store_true",
        help="Include motion detection in the report",
    )
    detect_parser.add_argument(
        "--motion-preset",
        choices=["fast", "balanced", "accurate"],
        help="Motion detection preset (fast/balanced/accurate). "
        "Overrides other motion settings.",
    )
    detect_parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.02,
        help="Motion detection threshold (percentage of frame, "
        "0.0-1.0). Ignored if preset is used.",
    )

    motion_parser = subparsers.add_parser(
        "motion",
        help="Detect motion in video frames (for filtering)",
    )
    motion_parser.add_argument("file", help="Path to the video file")
    motion_parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "accurate"],
        help="Use preset configuration (fast/balanced/accurate). "
        "Overrides other settings.",
    )
    motion_parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Analyze every Nth frame (ignored if preset is used)",
    )
    motion_parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Motion threshold (percentage of frame, "
        "0.0-1.0, ignored if preset is used)",
    )
    motion_parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum contour area in pixels (ignored if preset is used)",
    )
    motion_parser.add_argument(
        "--output",
        help="Optional output JSON path for motion results",
    )

    plates_parser = subparsers.add_parser(
        "plates",
        help="Detect and read license plates in video",
    )
    plates_parser.add_argument("file", help="Path to the video file")
    plates_parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path to YOLOv8 license plate detection model (default: yolov8n.pt)",
    )
    plates_parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Analyze every Nth frame",
    )
    plates_parser.add_argument(
        "--detect-conf",
        type=float,
        default=0.4,
        help="Detection confidence threshold "
        "(0.0-1.0, increased default to reduce false positives)",
    )
    plates_parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.5,
        help="Minimum OCR confidence to include result (0.0-1.0)",
    )
    plates_parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to process (optional)",
    )
    plates_parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for YOLO inference "
        "(640=default, 1280=better for small/distant plates)",
    )
    plates_parser.add_argument(
        "--output",
        help="Optional output JSON path for plate results",
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
    track_parser.add_argument(
        "--iou", type=float, default=0.3, help="IoU match threshold"
    )
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
    elif args.command == "verify":
        result = verify_video_integrity(args.file, args.manifest)
        print(json.dumps(result, indent=2))
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

        # Add motion detection if requested
        if args.motion:
            from cctv_dissertation.motion import (
                MOTION_PRESETS,
                add_motion_to_detection_report,
                detect_motion_in_video,
            )

            if args.motion_preset:
                preset_desc = MOTION_PRESETS[args.motion_preset]["description"]
                print(f"Using motion preset: " f"{args.motion_preset} - {preset_desc}")
                motion_results = detect_motion_in_video(
                    args.file,
                    preset=args.motion_preset,
                )
                motion_mode = f"preset:{args.motion_preset}"
            else:
                motion_results = detect_motion_in_video(
                    args.file,
                    frame_stride=args.frame_stride,
                    motion_threshold=args.motion_threshold,
                )
                motion_mode = "manual"

            report = add_motion_to_detection_report(report, motion_results)
            frames_with_motion = sum(1 for m in motion_results if m["has_motion"])
            motion_pct = frames_with_motion / len(motion_results) * 100
            print(
                f"Motion detected in "
                f"{frames_with_motion}/{len(motion_results)} "
                f"frames ({motion_pct:.1f}%)"
            )

        output_path = write_detection_report(report, args.output)
        summary = {
            "output_path": str(output_path),
            "frames_processed": len(report["detections"]),
            "sha256": report["sha256"],
            "motion_detection_enabled": args.motion,
        }
        if args.motion:
            summary["motion_mode"] = motion_mode
            summary["frames_with_motion"] = frames_with_motion
        print(json.dumps(summary, indent=2))
    elif args.command == "motion":
        from cctv_dissertation.motion import (
            MOTION_PRESETS,
            detect_motion_in_video,
        )

        if args.preset:
            p_desc = MOTION_PRESETS[args.preset]["description"]
            print(f"Using preset: {args.preset} - {p_desc}")
            results = detect_motion_in_video(
                args.file,
                preset=args.preset,
            )
        else:
            results = detect_motion_in_video(
                args.file,
                frame_stride=args.frame_stride,
                motion_threshold=args.threshold,
                min_area=args.min_area,
            )

        if args.output:
            import json as json_module
            from pathlib import Path

            output_path = Path(args.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json_module.dumps(results, indent=2))
            print(f"Motion results written to: {output_path}")
        else:
            frames_with_motion = sum(1 for r in results if r["has_motion"])
            print(
                json.dumps(
                    {
                        "total_frames_analyzed": len(results),
                        "frames_with_motion": frames_with_motion,
                        "motion_percentage": (
                            frames_with_motion / len(results) if results else 0
                        ),
                        "results": (
                            results[:10] if len(results) > 10 else results
                        ),  # Show first 10
                    },
                    indent=2,
                )
            )
    elif args.command == "plates":
        from cctv_dissertation.plates import detect_plates_in_video

        print(f"Detecting license plates using model: {args.model}")
        print(
            f"Frame stride: {args.frame_stride}, "
            f"Detection conf: {args.detect_conf}, "
            f"OCR conf: {args.ocr_conf}, "
            f"Image size: {args.imgsz}"
        )

        plates = detect_plates_in_video(
            args.file,
            plate_model_path=args.model,
            frame_stride=args.frame_stride,
            detect_conf=args.detect_conf,
            ocr_min_conf=args.ocr_conf,
            max_frames=args.max_frames,
            imgsz=args.imgsz,
        )

        # Count plates with successful OCR
        plates_with_text = [p for p in plates if p["text"] is not None]

        if args.output:
            from pathlib import Path

            output_path = Path(args.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(plates, indent=2))
            print(f"Plate results written to: {output_path}")
        else:
            print(
                json.dumps(
                    {
                        "total_plates_detected": len(plates),
                        "plates_with_text": len(plates_with_text),
                        "success_rate": (
                            len(plates_with_text) / len(plates) if plates else 0
                        ),
                        "plates": (
                            plates[:10] if len(plates) > 10 else plates
                        ),  # Show first 10
                    },
                    indent=2,
                )
            )
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
