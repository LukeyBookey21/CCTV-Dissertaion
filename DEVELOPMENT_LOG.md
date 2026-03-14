# Development Log

Chronological record of features, issues, and solutions for the CCTV Forensic Analysis System.

---

## Phase 1: Core Infrastructure

### Initial Setup
- Created project structure with `src/cctv_dissertation/` package
- Set up Streamlit UI at `ui/streamlit_app.py`
- Configured pre-commit hooks (black + flake8, max-line-length 88)

### Video Ingestion Pipeline
- Built chain-of-custody system with SHA256 hashing
- Created SQLite database for storing analysis results
- Added video metadata extraction

---

## Phase 2: Detection & Tracking

### YOLO Detection
- Integrated YOLOv8n for person and vehicle detection
- **Issue**: Needed different confidence thresholds per class
- **Solution**: person_conf=0.25, vehicle_conf=0.30, plate_conf=0.15

### ByteTrack Integration
- Used `model.track(persist=True)` for persistent IDs within a video
- **Issue**: `lap` package not installed by default
- **Solution**: YOLO auto-installs it, but documented as a gotcha

- **Issue**: Track IDs reset when processing second video
- **Solution**: Create separate tracker instances for each video (ByteTrack maintains internal state)

### License Plate Detection
- Trained custom YOLO model for plate detection
- Integrated EasyOCR for plate text extraction
- **Issue**: OCR accuracy varies with image quality
- **Solution**: Accept partial reads, store best confidence result

---

## Phase 3: Re-Identification (Cross-Camera Matching)

### Person Re-ID
- Integrated OSNet x1_0 from torchreid for 512-dim embeddings
- L2 normalization for cosine similarity via dot product
- **Issue**: False matches between different people
- **Solution**: Tuned threshold to 0.47 after testing

- **Issue**: Same person in consecutive frames counted as multiple matches
- **Solution**: Added `same_moment` window (<=1 frame difference = same sighting)

### Vehicle Re-ID
- Used OSNet x0_25 (lighter model for vehicles)
- Combined with color matching for better accuracy
- **Issue**: Merge threshold too low caused false merges
- **Solution**: Set thresholds: person 0.78, vehicle 0.80

### Unified Identity Assignment
- `build_unified_identities()` assigns consistent P1, P2, V1, V2 IDs across cameras
- Orders by first appearance timestamp
- Groups matched entities under same unified ID
- **Issue**: 1-to-1 matching failed when same person had multiple tracks (e.g., leaves for 2hrs, returns)
- **Attempted**: Union-Find clustering to handle multiple appearances
- **Issue**: Union-Find caused transitive merges, incorrectly grouping different people
- **Solution**: Reverted to 1-to-1 greedy matching (preserves accuracy)
- Cross-camera threshold: 0.60 (empirically tuned for this dataset)

---

## Phase 4: Color & Appearance Description

### Color Extraction
- K-means clustering (k=5) on cropped images
- Background filtering to ignore scene colors
- **Issue**: Olive/earth tones misclassified as grey
- **Solution**: Check earth tones BEFORE achromatic in `bgr_to_color_name()`

### Vehicle Color
- **Issue**: White vehicles sometimes misidentified
- **Solution**: Added 3.0x multiplier for white paint (s<50, v>170) in body_score

### Description Generation
- `describe_person()`: upper/lower body colors
- `describe_vehicle()`: type + body color + plate if detected

---

## Phase 5: UI Features

### Single Camera Mode
- Tabs: Tracking, Summary, Search, Person of Interest, Detections
- Annotated video playback with bounding boxes
- Crop image gallery per tracked entity

### Cross-Camera Mode
- Dual View: side-by-side video comparison
- Identity Deep Dive: unified identities with time gaps
- **Issue**: Clips not sorted chronologically
- **Solution**: Order by `first_ts` (first timestamp)

### Timestamp Display
- **Issue**: Displayed relative seconds ("0s - 11s") not useful
- **Solution**: Added `_format_absolute_time()` to show actual clock times ("11:00:00 - 11:00:11")

### Auto-OCR Timestamp Extraction
- **Issue**: User had to manually enter video start times
- **Solution**: `extract_video_timestamp()` uses EasyOCR on bottom strip of first frame to read burned-in timestamps

### Timeline Visualisation
- Altair Gantt chart showing person appearances across cameras
- Horizontal bars per person, color-coded by camera
- Interactive pan/zoom with tooltips

### Appearance Search
- Filter by entity type, description text, colors
- `_search_entities()` builds parameterized SQL queries
- Color dropdown options from `bgr_to_color_name()` output

### PDF Forensic Report
- Professional layout with fpdf2
- Title page, person sections with crops, timestamps, movement timeline
- **Issue**: `pdf.output()` returns bytearray, Streamlit needs bytes
- **Solution**: Wrap with `bytes(pdf.output())`
- **Issue**: Report lacked audit trail and video metadata
- **Solution**: Added comprehensive metadata sections:
  - Video source metadata (filename, resolution, FPS, duration, SHA256 hash)
  - Processing information (models used, thresholds, algorithms)
  - Chain of custody section (examiner, date, evidence integrity)
- **Issue**: Vehicle images stacked vertically, cluttered layout
- **Solution**: Limited to 2 images max side-by-side per vehicle

### Person of Interest
- Upload reference photo
- Extract Re-ID embedding with PersonReID
- Cosine similarity search against all tracked persons
- Rank matches by similarity score

---

## Phase 6: Video Playback Fixes

### Browser Compatibility
- **Issue**: OpenCV's `mp4v` codec not playable in browsers/Streamlit
- **Solution**: `_reencode_h264()` converts to H.264 via ffmpeg after rendering

---

## Phase 7: Performance Optimisation

### Progress Feedback
- Added `progress_callback` parameter to `process_video()`
- Real-time UI updates: frame count, ETA, entities found, processing speed

### Motion Detection Gate
- Skip frames with no significant change
- Background subtraction + Gaussian blur + threshold
- `motion_threshold=0.002` (0.2% of pixels must change)
- **Rationale**: Doesn't affect accuracy for truly static frames

### Frame Cache Limit
- **Issue**: Long videos caused memory blowup
- **Solution**: `frame_cache_limit=500` evicts old frames from memory

### Auto Frame Stride
- `calc_auto_stride()` calculates optimal stride based on video duration
- Target: ~15 minutes processing time per video
- 5-hour video at 30fps → stride=4, 7.5 effective fps
- **Accuracy**: Catches anyone visible for 0.13+ seconds

### Database Indexes
- Added indexes on frequently queried columns
- Improves search and filtering performance

---

## Phase 8: Evaluation Framework

### Ground Truth Format
- JSON structure for manual annotations
- Persons/vehicles with frame ranges per camera
- Same ID = same entity across cameras

### Metrics
- Tracking: Precision, Recall, F1, MOTA
- Re-ID: Cross-camera matching accuracy
- `evaluation.py` module for command-line evaluation

### UI Integration
- Initially added Evaluation tab to UI
- **Decision**: Removed from UI (evaluation is a one-time test, not a user feature)
- Kept as command-line script for dissertation testing

---

## Ongoing Issues / Future Work

### Summary Page ID Mismatch
- Single-camera summary shows per-camera track IDs
- Cross-camera shows unified IDs
- Users may find this confusing (not yet addressed)

### Processing Resumability
- Currently must restart from beginning if interrupted
- Future: Save progress to DB incrementally

---

## Key Thresholds (Tuned Values)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Person detection conf | 0.25 | Catch partial/occluded persons |
| Vehicle detection conf | 0.30 | Reduce false positives |
| Plate detection conf | 0.15 | Plates are small, need low threshold |
| Person Re-ID match | 0.47 | Balanced precision/recall |
| Person merge threshold | 0.78 | Prevent false merges |
| Vehicle merge threshold | 0.80 | Vehicles more similar, need higher |
| Min track frames (person) | 10 | Filter noise tracks |
| Min track frames (vehicle) | 5 | Vehicles pass through faster |
| Motion threshold | 0.002 | Skip truly static frames |
| Frame cache limit | 500 | Prevent memory issues |

---

## Dependencies Added

- `ultralytics` — YOLOv8
- `torchreid` — Person Re-ID (OSNet)
- `easyocr` — License plate + timestamp OCR
- `fpdf2` — PDF report generation
- `altair` — Timeline visualisation (bundled with Streamlit)
