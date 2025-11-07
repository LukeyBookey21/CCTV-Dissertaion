# Forensics Video Analysis Tool: Automated Detection, Tracking and Integrity Verification System

**Student ID:** C3675541  
**First Supervisor:** Andrew Scholey  
**Second Supervisor:** Pip Trevorrow

---

## Project Specification

### Working Title

Forensics Video Analysis Tool: Automated Detection, Tracking and Integrity Verification System

### Project Aim (≈100–150 words)

This project aims to develop a forensic video analysis tool that helps investigators work through large amounts of video footage more efficiently by automating the tedious parts of video analysis. The tool will accept video files from any source (CCTV systems, dashcams, phones, body cameras) and use computer vision techniques to automatically detect, track, and filter specific objects, people, and events. Instead of reviewing footage at normal speed, investigators can upload videos and query for specifics (e.g., “red vehicles, between 2 pm and 4 pm”). The system will extract metadata to verify timestamps and support analyzing multiple video files together (e.g., different camera angles). A critical feature is automated hashing to preserve evidence integrity for court: hashing occurs on upload, enabling later proof of non-tampering. The project applies established machine learning and computer vision to real-world digital forensics challenges in a way that’s practically useful for law enforcement and security professionals.

---

## Project Expectations (≈600–850 words)

This project will investigate how existing computer vision and machine learning techniques can be practically applied to forensic video analysis. Research will focus on real-time object detection algorithms (e.g., YOLO, Faster R-CNN) and tracking systems (e.g., DeepSORT, Kalman filters). The central question is whether these methods remain reliable on real-world footage with poor lighting, low resolution, oblique camera angles, and compression artifacts. Academic datasets are clean; actual CCTV is messy. Development will be iterative: begin with single-video detection, ensure stability, then expand.

The system comprises five core components:

1. **Video Import**  
   Accept common formats (MP4, AVI, MOV, MKV) and handle codecs robustly. Ensure metadata extraction works across formats.

2. **Detection & Tracking**  
   Use pre-trained models (likely YOLO for speed/documentation) to detect and track objects. Training custom models is out of scope for the time frame.

3. **Search & Filter**  
   Query detections by object class, time ranges, and basic motion to reduce manual review.

4. **Metadata & Timestamps**  
   Use FFmpeg to extract recording time, device info, and other forensic details.

5. **Integrity (Hashing)**  
   Automatically compute SHA-256 on upload and support later verification for chain-of-custody.

**Good practice and engineering approach:**  
Adopt a modular architecture (detection, tracking, UI, hashing) to enable isolated testing and resilient integration. Use Git with regular, meaningful commits. Maintain documentation for maintainability and assessment. Since video processing is expensive, benchmark and optimize throughout—e.g., frame sampling rather than per-frame analysis when appropriate. The UI should be usable by non-technical investigators. Testing includes unit tests and end-to-end tests with real video files.

**Evaluation plan:**  

- **Quantitative:** detection precision/recall vs. labeled ground truth, FPS throughput, tracking success (ID switches, MOTA-style metrics if feasible), hash generation time, false positive/negative rates.  
- **Qualitative:** usability of the UI, robustness across video quality conditions, and supervisor/stakeholder feedback. If possible, compare tool-assisted review time vs. manual review.

**Ethics:**  
Video frequently includes identifiable people. Use public/consented footage. Apply privacy techniques (e.g., face blurring in demos). Comply with GDPR (storage, retention, access controls). Submit ethics application in Weeks 4–5; progress is blocked without approval.

**High-level timeline:**  

- **Semester A:**  
  - Weeks 1–3: Specification & literature review  
  - Weeks 4–6: Environment setup & proof-of-concept  
  - Weeks 7–9: Core detection & tracking  
  - Weeks 10–12: Basic UI + WIP presentation prep  
- **Semester B:**  
  - Weeks 13–16: Must-haves (hashing, metadata, single-video testing)  
  - Weeks 17–19: Bug fixes & optimization  
  - Weeks 19–21: (If ahead) multi-video tracking; otherwise skip  
  - Weeks 22–24: UI polish & documentation  
  - Weeks 25–28: Report finalization

**Key risks:** OpenCV learning curve, performance on large files, complexity of multi-video features, module deadline clashes. Mitigation: early testing, realistic scope, strict MoSCoW prioritization, and schedule discipline.

---

## Development Kick-off Tasks

1. **Video ingestion scaffold** – run `python -m cctv_dissertation.app ingest path/to/video.mp4` to hash a file, extract baseline metadata, and append it to `data/manifests/ingest_manifest.json`. This establishes the chain-of-custody layer the rest of the system will rely on.
2. **Manifest inspection** – the manifest grows as JSON list entries; open it in any editor to review stored hashes, timestamps, and metadata backends (ffprobe when available, OpenCV fallback otherwise).
3. **Hash-only utility** – `python -m cctv_dissertation.app hash some_file.bin` remains available for quick integrity checks outside the ingestion flow.
4. **YOLOv8 detections** – `python -m cctv_dissertation.app detect path/to/video.mp4 --frame-stride 10 --conf 0.35` samples frames, runs YOLOv8n, and writes results to `data/detections/<sha256>.json`.
5. **Detection summaries** – `python -m cctv_dissertation.app analyze data/detections/<sha256>.json` prints per-class counts, frame coverage, and timestamps; append `--json` for machine-readable output.
6. **Store detections** – `python -m cctv_dissertation.app store data/detections/<sha256>.json --db data/analysis.db` loads reports into SQLite so investigator queries can filter by class/time/confidence.
7. **Track objects** – `python -m cctv_dissertation.app track data/detections/<sha256>.json --store --db data/analysis.db` links detections over time (IoU-based) and saves tracks for timeline views.
8. **Query database** – `python -m cctv_dissertation.app query <sha256> --class car --time-start 5 --time-end 12 --db data/analysis.db` returns either per-frame detections or `--tracks` summaries to support investigator questions.

### Interactive UI

Run the Streamlit dashboard once at least one detection report exists:

```bash
streamlit run ui/streamlit_app.py
```

Use the sidebar to:

- Upload a new video (it is saved under `data/uploads/`, ingested, detected, and optionally auto-stored/tracked in SQLite).
- Select any detection JSON, view summaries, scrub through the annotated video with frame-by-frame overlays, and run class/time/confidence queries against the database.

### REST API

Prefer automation or remote access? Launch the FastAPI server:

```bash
uvicorn cctv_dissertation.api:app --reload --app-dir src
```

Key endpoints:

- `POST /upload` – multipart upload with optional `store`/`track` flags.
- `GET /detections` / `GET /summary/{sha}` – enumerate and inspect detection reports.
- `GET /query/detections` and `GET /query/tracks` – run the same investigator filters the CLI/UI expose.

### Future Extensions

- **AI deepfake detection:** once the core workflow is stable, integrate a deepfake classifier (e.g., FaceForensics, MesoNet) during ingest so uploaded clips carry authenticity scores alongside hashes.
- **Visual overlays & exports:** the Streamlit UI already previews frames; future work includes downloadable annotated clips and PDF evidence packets.

Once the ingest path feels reliable (ideally with a dedicated sample clip committed under `tests/`), detection/tracking experiments can hook into the verified files instead of ad-hoc paths.

---

## Search Term Template

### Topic

Forensic video analysis

### Prior Knowledge / Gaps

- **Known:** SHA-256 hashing; Python; digital evidence handling & chain of custody; YOLO family prominence; tracking (SORT/DeepSORT/Kalman); OpenCV and FFmpeg are likely core tools; Redmon’s YOLO papers; ACPO guidelines; Ultralytics as a practical YOLO route.  
- **Gaps:** handling codecs/formats; appropriate evaluation metrics for detection/tracking; techniques beyond hashing for tamper verification; challenges specific to real CCTV footage.

### Keywords

| Core Search Terms     | Broader Terms        | Narrower Terms         | Related Terms            |
|-----------------------|----------------------|------------------------|--------------------------|
| Video forensics       | Digital forensics    | CCTV analysis          | Digital evidence         |
| Object detection      | Computer vision      | YOLO detection         | Image processing         |
| Object tracking       | Video analysis       | DeepSORT tracking      | Surveillance systems     |
| Forensics video analysis | Artificial intelligence | Person re-identification | Evidence integrity     |
| Video metadata        | Machine learning     | Timestamp verification | Chain of custody         |
| Hash verification     | Data integrity       | SHA-256 hashing        | Security systems         |
| Surveillance analysis | Deep learning        | Multi-camera tracking  | Crime analysis           |

### Sources / Where to Search

- **Academic:** IEEE Xplore, ACM DL, Google Scholar, SpringerLink, ScienceDirect  
- **University:** Leeds Beckett library databases & subject guides  
- **Technical Docs:** OpenCV, TensorFlow/PyTorch, FFmpeg, Ultralytics YOLO  
- **Preprints/Repos:** arXiv, ResearchGate, GitHub  
- **Gov/Industry:** NIST CFTT, ACPO Good Practice Guide, vendors (Cellebrite, Magnet Forensics)  
- **Communities:** Stack Overflow, YOLO forums

### Example Search Strings

- **General:**  
  - `"video forensics" AND ("object detection" OR "object tracking")`  
  - `"CCTV forensics" AND "machine learning"`
- **Technical:**  
  - `YOLO AND (surveillance OR CCTV)`  
  - `"object tracking" AND (SORT OR DeepSORT)`  
  - `"pre-trained models" AND "video analysis"`
- **Multi-video:**  
  - `"cross-camera tracking" OR "multi-camera tracking"`  
  - `"person re-identification" AND surveillance`
- **Evidence integrity:**  
  - `"digital evidence" AND "video integrity"`  
  - `SHA-256 AND "evidence integrity"`  
  - `"timestamp verification" AND forensic`  
  - `FFmpeg AND metadata`
- **Applications:**  
  - `"license plate recognition" AND CCTV AND forensic`  
  - `"event detection" AND "surveillance video"`
- **Standards:**  
  - `ACPO AND "digital evidence"`  
  - `NIST AND "video forensics"`
- **Performance:**  
  - `"low quality video" AND "object detection"`  
  - `"poor lighting" AND "object detection"`

---

---

### Notes

- Keep searches broad at first, then narrow using datasets/terms found in relevant papers.  

- When experimenting, record search strings that returned useful references to save time later.

---

If you'd like, I can also run a markdown linter (e.g., markdownlint) on this file or open a preview to ensure it renders as you expect. Let me know which you'd prefer.
