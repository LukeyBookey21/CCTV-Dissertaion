"""Evaluation framework for tracking and re-identification accuracy."""

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class GroundTruthEntry:
    """A single ground truth annotation."""

    person_id: str  # Global ID (same person across cameras)
    camera: str
    track_id: int  # System's track ID for this appearance
    first_frame: int
    last_frame: int
    description: str = ""


@dataclass
class EvaluationResult:
    """Results from evaluating system output against ground truth."""

    # Detection/Tracking metrics
    true_positives: int = 0  # Correctly detected tracks
    false_positives: int = 0  # System tracks with no ground truth match
    false_negatives: int = 0  # Ground truth tracks the system missed
    id_switches: int = 0  # Times the system changed ID for same person

    # Re-ID metrics (cross-camera)
    reid_true_positives: int = 0  # Correctly matched across cameras
    reid_false_positives: int = 0  # Incorrectly matched (different people)
    reid_false_negatives: int = 0  # Same person not matched across cameras

    # Identity-level metrics
    gt_identities: int = 0  # Number of ground truth identities
    sys_identities: int = 0  # Number of system unified identities
    correct_identities: int = 0  # System IDs mapping to exactly 1 GT, sole owner
    over_fragmented: int = 0  # GT persons split across multiple system IDs
    over_merged: int = 0  # System IDs grouping different GT persons

    # Detailed breakdowns
    per_camera_results: Dict[str, dict] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    matched_pairs: List[Tuple[str, int, str]] = field(
        default_factory=list
    )  # (gt_id, sys_track, camera)

    @property
    def precision(self) -> float:
        """Detection precision: TP / (TP + FP)."""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """Detection recall: TP / (TP + FN)."""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 score: harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def reid_precision(self) -> float:
        """Re-ID precision: correct cross-camera matches / total matches."""
        total = self.reid_true_positives + self.reid_false_positives
        return self.reid_true_positives / total if total > 0 else 0.0

    @property
    def reid_recall(self) -> float:
        """Re-ID recall: correct matches / total same-person pairs."""
        total = self.reid_true_positives + self.reid_false_negatives
        return self.reid_true_positives / total if total > 0 else 0.0

    @property
    def reid_f1(self) -> float:
        """Re-ID F1 score."""
        p, r = self.reid_precision, self.reid_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mota(self) -> float:
        """Multiple Object Tracking Accuracy (simplified).

        MOTA = 1 - (FN + FP + ID_switches) / total_gt
        """
        total_gt = self.true_positives + self.false_negatives
        if total_gt == 0:
            return 0.0
        errors = self.false_negatives + self.false_positives + self.id_switches
        return 1.0 - (errors / total_gt)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tracking": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "id_switches": self.id_switches,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1_score": round(self.f1_score, 4),
                "mota": round(self.mota, 4),
            },
            "reid": {
                "true_positives": self.reid_true_positives,
                "false_positives": self.reid_false_positives,
                "false_negatives": self.reid_false_negatives,
                "precision": round(self.reid_precision, 4),
                "recall": round(self.reid_recall, 4),
                "f1_score": round(self.reid_f1, 4),
            },
            "identity": {
                "gt_identities": self.gt_identities,
                "sys_identities": self.sys_identities,
                "correct": self.correct_identities,
                "over_fragmented": self.over_fragmented,
                "over_merged": self.over_merged,
            },
            "per_camera": self.per_camera_results,
        }


# ── Ground Truth Format ────────────────────────────────────────────────


def create_ground_truth_template(cameras: List[str], output_path: str) -> Dict:
    """Create an empty ground truth template JSON.

    Ground truth format:
    {
        "metadata": {
            "created": "2024-01-15",
            "annotator": "Name",
            "notes": "..."
        },
        "persons": {
            "P1": {
                "description": "Male, red shirt, jeans",
                "appearances": [
                    {"camera": "garage", "frames": [100, 500], "notes": ""},
                    {"camera": "garden", "frames": [200, 400], "notes": ""}
                ]
            },
            "P2": { ... }
        },
        "vehicles": {
            "V1": {
                "description": "Blue sedan, plate ABC123",
                "appearances": [...]
            }
        }
    }
    """
    from datetime import datetime

    template = {
        "metadata": {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "annotator": "",
            "cameras": cameras,
            "notes": "Fill in person/vehicle appearances with frame ranges",
        },
        "persons": {
            "P1": {
                "description": "Describe appearance here",
                "appearances": [
                    {"camera": cameras[0] if cameras else "camera1", "frames": [0, 100]}
                ],
            }
        },
        "vehicles": {},
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)

    return template


def load_ground_truth(path: str) -> Dict:
    """Load ground truth annotations from JSON file."""
    with open(path) as f:
        return json.load(f)


# ── System Output Extraction ───────────────────────────────────────────


def extract_system_tracks(
    db_path: str, cameras: Optional[List[str]] = None
) -> Dict[str, List[dict]]:
    """Extract tracked entities from the database.

    Returns:
        Dict mapping camera -> list of track dicts with:
            {track_id, entity_type, first_frame, last_frame, description}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            te.camera_label,
            te.track_id,
            te.entity_type,
            te.description,
            MIN(tf.frame_idx) as first_frame,
            MAX(tf.frame_idx) as last_frame
        FROM tracked_entities te
        JOIN track_frames tf ON te.id = tf.entity_id
    """
    if cameras:
        placeholders = ",".join("?" * len(cameras))
        query += f" WHERE te.camera_label IN ({placeholders})"
        query += " GROUP BY te.id"
        rows = conn.execute(query, cameras).fetchall()
    else:
        query += " GROUP BY te.id"
        rows = conn.execute(query).fetchall()

    conn.close()

    result: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        result[row["camera_label"]].append(
            {
                "track_id": row["track_id"],
                "entity_type": row["entity_type"],
                "first_frame": row["first_frame"],
                "last_frame": row["last_frame"],
                "description": row["description"] or "",
            }
        )

    return dict(result)


def extract_unified_identities(
    db_path: str, cameras: Optional[List[str]] = None
) -> Dict[str, List[dict]]:
    """Extract unified identities from build_unified_identities output.

    Returns dict mapping unified_id -> list of appearances across cameras.
    """
    from cctv_dissertation.tracker import build_unified_identities

    # Determine cameras from DB if not provided
    if not cameras or len(cameras) < 2:
        import sqlite3

        conn = sqlite3.connect(db_path)
        cameras = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT camera_label FROM tracked_entities"
            ).fetchall()
        ]
        conn.close()
        if len(cameras) < 2:
            return {}

    identities = build_unified_identities(db_path, cameras[0], cameras[1])

    result = {}
    for entity_type in ("persons", "vehicles"):
        for entity in identities.get(entity_type, []):
            uid = f"{'P' if entity_type == 'persons' else 'V'}{entity['unified_id']}"
            result[uid] = []
            for sighting in entity.get("sightings", []):
                result[uid].append(
                    {
                        "camera": sighting.get("camera"),
                        "track_id": sighting.get("track_id"),
                        "description": sighting.get("description", ""),
                    }
                )

    return result


# ── Evaluation Logic ───────────────────────────────────────────────────


def compute_iou_frames(
    gt_frames: Tuple[int, int], sys_frames: Tuple[int, int]
) -> float:
    """Compute temporal IoU between two frame ranges."""
    gt_start, gt_end = gt_frames
    sys_start, sys_end = sys_frames

    intersection_start = max(gt_start, sys_start)
    intersection_end = min(gt_end, sys_end)
    intersection = max(0, intersection_end - intersection_start)

    union = (gt_end - gt_start) + (sys_end - sys_start) - intersection

    return intersection / union if union > 0 else 0.0


def match_tracks(
    gt_appearances: List[dict],
    sys_tracks: List[dict],
    iou_threshold: float = 0.3,
) -> Tuple[List[Tuple[dict, dict]], List[dict], List[dict]]:
    """Match ground truth appearances to system tracks.

    Returns:
        (matched_pairs, unmatched_gt, unmatched_sys)
    """
    matched = []
    used_sys = set()

    # Sort by overlap quality
    candidates = []
    for gt in gt_appearances:
        gt_frames = (gt["frames"][0], gt["frames"][1])
        for i, sys in enumerate(sys_tracks):
            sys_frames = (sys["first_frame"], sys["last_frame"])
            iou = compute_iou_frames(gt_frames, sys_frames)
            if iou >= iou_threshold:
                candidates.append((iou, gt, sys, i))

    # Greedy matching by best IoU
    candidates.sort(reverse=True, key=lambda x: x[0])
    used_gt = set()

    for iou, gt, sys, sys_idx in candidates:
        gt_key = (gt["camera"], gt["frames"][0], gt["frames"][1])
        if gt_key not in used_gt and sys_idx not in used_sys:
            matched.append((gt, sys))
            used_gt.add(gt_key)
            used_sys.add(sys_idx)

    unmatched_gt = [
        gt
        for gt in gt_appearances
        if (gt["camera"], gt["frames"][0], gt["frames"][1]) not in used_gt
    ]
    unmatched_sys = [sys for i, sys in enumerate(sys_tracks) if i not in used_sys]

    return matched, unmatched_gt, unmatched_sys


def evaluate_tracking(
    ground_truth: Dict,
    db_path: str,
    entity_type: str = "person",
    iou_threshold: float = 0.3,
) -> EvaluationResult:
    """Evaluate tracking accuracy against ground truth.

    Args:
        ground_truth: Loaded ground truth dict
        db_path: Path to tracker database
        entity_type: "person" or "vehicle"
        iou_threshold: Minimum temporal IoU to count as a match
    """
    result = EvaluationResult()

    # Get ground truth entries
    gt_key = "persons" if entity_type == "person" else "vehicles"
    gt_entities = ground_truth.get(gt_key, {})

    # Get system tracks
    cameras = ground_truth.get("metadata", {}).get("cameras", [])
    sys_tracks = extract_system_tracks(db_path, cameras)

    # Flatten ground truth to per-camera appearances
    gt_by_camera: Dict[str, List[dict]] = defaultdict(list)
    for entity_id, entity_data in gt_entities.items():
        for appearance in entity_data.get("appearances", []):
            camera = appearance["camera"]
            gt_by_camera[camera].append(
                {
                    "entity_id": entity_id,
                    "camera": camera,
                    "frames": appearance["frames"],
                }
            )

    # Evaluate per camera
    for camera in set(list(gt_by_camera.keys()) + list(sys_tracks.keys())):
        gt_list = gt_by_camera.get(camera, [])
        sys_list = [
            t for t in sys_tracks.get(camera, []) if t["entity_type"] == entity_type
        ]

        matched, unmatched_gt, unmatched_sys = match_tracks(
            gt_list, sys_list, iou_threshold
        )

        cam_tp = len(matched)
        cam_fp = len(unmatched_sys)
        cam_fn = len(unmatched_gt)

        result.true_positives += cam_tp
        result.false_positives += cam_fp
        result.false_negatives += cam_fn

        result.per_camera_results[camera] = {
            "true_positives": cam_tp,
            "false_positives": cam_fp,
            "false_negatives": cam_fn,
            "precision": cam_tp / (cam_tp + cam_fp) if (cam_tp + cam_fp) > 0 else 0,
            "recall": cam_tp / (cam_tp + cam_fn) if (cam_tp + cam_fn) > 0 else 0,
        }

        # Record matches for re-id evaluation
        for gt, sys in matched:
            result.matched_pairs.append((gt["entity_id"], sys["track_id"], camera))

    return result


def evaluate_reid(
    ground_truth: Dict,
    db_path: str,
    entity_type: str = "person",
) -> EvaluationResult:
    """Evaluate cross-camera re-identification accuracy.

    Compares the system's unified identities against ground truth
    cross-camera identity annotations.
    """
    result = EvaluationResult()

    gt_key = "persons" if entity_type == "person" else "vehicles"
    gt_entities = ground_truth.get(gt_key, {})

    # Build ground truth cross-camera pairs
    # (entity_id, camera1, camera2) for each entity appearing in multiple cameras
    gt_pairs = set()
    for entity_id, entity_data in gt_entities.items():
        cameras = [a["camera"] for a in entity_data.get("appearances", [])]
        cameras = list(set(cameras))
        for i, c1 in enumerate(cameras):
            for c2 in cameras[i + 1 :]:
                pair = tuple(sorted([c1, c2]))
                gt_pairs.add((entity_id, pair[0], pair[1]))

    # Get system unified identities
    cameras = ground_truth.get("metadata", {}).get("cameras", [])
    sys_identities = extract_unified_identities(db_path, cameras)

    # Match system pairs to ground truth pairs
    # A system pair is correct if it links the same ground truth entity
    # We need to trace back: which GT entity does each system track correspond to?

    # First, run tracking evaluation to get the mapping
    tracking_result = evaluate_tracking(ground_truth, db_path, entity_type)

    # Build mapping: (camera, sys_track_id) -> gt_entity_id
    sys_to_gt: Dict[Tuple[str, int], str] = {}
    for gt_id, sys_track, camera in tracking_result.matched_pairs:
        sys_to_gt[(camera, sys_track)] = gt_id

    # Build bidirectional mapping: sys_uid -> set(gt_ids), gt_id -> set(sys_uids)
    prefix = "P" if entity_type == "person" else "V"
    sys_uid_to_gt: Dict[str, set] = {}
    gt_to_sys_uids: Dict[str, set] = defaultdict(set)

    for uid, appearances in sys_identities.items():
        if not uid.startswith(prefix):
            continue
        gt_ids_in_unified = set()
        for app in appearances:
            key = (app["camera"], app["track_id"])
            if key in sys_to_gt:
                gt_ids_in_unified.add(sys_to_gt[key])
        sys_uid_to_gt[uid] = gt_ids_in_unified
        for gt_id in gt_ids_in_unified:
            gt_to_sys_uids[gt_id].add(uid)

    # Identity-level metrics
    result.gt_identities = len(gt_entities)
    result.sys_identities = len(sys_uid_to_gt)

    # Over-merged: system UID groups different GT persons
    for uid, gt_ids in sys_uid_to_gt.items():
        if len(gt_ids) > 1:
            result.over_merged += 1

    # Over-fragmented: GT person split across multiple system UIDs
    for gt_id, sys_uids in gt_to_sys_uids.items():
        if len(sys_uids) > 1:
            result.over_fragmented += 1
        elif len(sys_uids) == 1:
            uid = list(sys_uids)[0]
            if len(sys_uid_to_gt.get(uid, set())) == 1:
                result.correct_identities += 1

    # Re-ID: cross-camera matching evaluation
    # TP: GT cross-camera person fully covered by a SINGLE system UID
    # FN: GT cross-camera person NOT fully covered by any single system UID
    # FP: System UID groups appearances from different GT persons across cameras
    for gt_id, c1, c2 in gt_pairs:
        found_single_uid = False
        for uid in gt_to_sys_uids.get(gt_id, set()):
            app_cameras = set(a["camera"] for a in sys_identities.get(uid, []))
            # This UID must cover both cameras AND only map to this GT person
            if c1 in app_cameras and c2 in app_cameras:
                if len(sys_uid_to_gt.get(uid, set())) == 1:
                    found_single_uid = True
                    break
        if found_single_uid:
            result.reid_true_positives += 1
        else:
            result.reid_false_negatives += 1

    for uid, gt_ids in sys_uid_to_gt.items():
        if len(gt_ids) > 1:
            app_cameras = set(a["camera"] for a in sys_identities.get(uid, []))
            if len(app_cameras) > 1:
                result.reid_false_positives += 1

    # Copy tracking results
    result.true_positives = tracking_result.true_positives
    result.false_positives = tracking_result.false_positives
    result.false_negatives = tracking_result.false_negatives
    result.per_camera_results = tracking_result.per_camera_results
    result.matched_pairs = tracking_result.matched_pairs

    return result


def generate_evaluation_report(
    ground_truth_path: str,
    db_path: str,
    output_path: Optional[str] = None,
) -> Dict:
    """Generate a comprehensive evaluation report.

    Args:
        ground_truth_path: Path to ground truth JSON
        db_path: Path to tracker database
        output_path: Optional path to save JSON report

    Returns:
        Dict with full evaluation results
    """
    gt = load_ground_truth(ground_truth_path)

    # Evaluate persons
    person_result = evaluate_reid(gt, db_path, "person")

    # Evaluate vehicles
    vehicle_result = evaluate_reid(gt, db_path, "vehicle")

    report = {
        "ground_truth_file": ground_truth_path,
        "database_file": db_path,
        "persons": person_result.to_dict(),
        "vehicles": vehicle_result.to_dict(),
        "summary": {
            "overall_tracking_f1": round(
                (person_result.f1_score + vehicle_result.f1_score) / 2, 4
            ),
            "overall_reid_f1": round(
                (person_result.reid_f1 + vehicle_result.reid_f1) / 2, 4
            ),
            "person_mota": round(person_result.mota, 4),
            "vehicle_mota": round(vehicle_result.mota, 4),
        },
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report


# ── CLI Interface ──────────────────────────────────────────────────────


def main():
    """CLI for evaluation tools."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate tracking/re-id accuracy")
    subparsers = parser.add_subparsers(dest="command")

    # Create template
    create_parser = subparsers.add_parser(
        "create-template", help="Create ground truth template"
    )
    create_parser.add_argument(
        "--cameras", nargs="+", required=True, help="Camera names"
    )
    create_parser.add_argument(
        "--output", default="data/ground_truth.json", help="Output path"
    )

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--gt", required=True, help="Ground truth JSON path")
    eval_parser.add_argument(
        "--db", default="data/tracker.db", help="Tracker database path"
    )
    eval_parser.add_argument("--output", help="Output report path")

    args = parser.parse_args()

    if args.command == "create-template":
        create_ground_truth_template(args.cameras, args.output)
        print(f"Created template: {args.output}")

    elif args.command == "evaluate":
        report = generate_evaluation_report(args.gt, args.db, args.output)
        print("\n=== Evaluation Results ===\n")
        print(f"Person Tracking:  F1={report['persons']['tracking']['f1_score']:.2%}")
        print(f"Person Re-ID:     F1={report['persons']['reid']['f1_score']:.2%}")
        print(f"Vehicle Tracking: F1={report['vehicles']['tracking']['f1_score']:.2%}")
        print(f"Vehicle Re-ID:    F1={report['vehicles']['reid']['f1_score']:.2%}")
        print(f"\nMOTA (persons):   {report['summary']['person_mota']:.2%}")
        if args.output:
            print(f"\nFull report saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
