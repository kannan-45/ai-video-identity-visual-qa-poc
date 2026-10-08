# AI Video Generation — Identity & Visual Consistency QA

Member 7 QA module for the AI Video Generation POC. It evaluates whether generated images and video frames remain consistent with a reference image and produces explainable QA results.

## Purpose

The QA engine evaluates:

- Face identity consistency
- Structural similarity (SSIM)
- Perceptual similarity (LPIPS)
- Semantic visual similarity (CLIP)
- Visual representation drift (DINOv3)
- Object consistency
- Environment consistency
- Keyframe consistency
- Deliberate identity and appearance failures

The system retains component-level evidence and explicit reason codes instead of collapsing everything into one opaque score.

## Reuse from Day 1

Day 2 reuses the Day 1 image-level QA components:

- AdaFace identity similarity
- PixelFace Prototype identity evidence
- SSIM
- LPIPS
- CLIP
- DINOv3
- Object/environment consistency
- Component-level scores and reason codes
- Image-level QA workflow
- Comparison UI

Day 2 extends this capability from image pairs to deterministic video-frame sampling and clip-level aggregation.

## What is New Today — Day 2

- Deterministic MP4 frame sampling
- Per-frame identity and visual QA
- Level 1 CPU-light video checks
- Level 2 selected-frame model checks
- Frame-selection merging
- Anomaly localization
- Dense resampling around flagged ranges
- Evidence aggregation
- PASS / AUTO_RETRY / HUMAN_REVIEW decision mapping
- Structured clip-level QAResult JSON
- ClipMetadata validation and SHA-256 verification
- Deterministic video and visual failure fixtures
- Day 2 automated tests

For a 25-frame video with 5 coarse samples, deterministic sampling selects:

```text
0, 6, 12, 18, 24
```

## Day 2 Dependency Contract

Member 7 consumes:

- ReferenceRecord images
- ClipMetadata
- MP4 clip

Member 7 provides:

- QAResult

## Day 2 QA Workflow

```text
Reference image + MP4 + ClipMetadata
                |
                v
          ingest_clip()
                |
                v
        Decode all frames
                |
                v
     LEVEL 1 — CHEAP CHECKS
     - frame difference
     - brightness delta
     - color delta
     - frozen-frame detection
     - motion energy
     - coarse deterministic sampling
                |
                v
        merge_frame_selection()
                |
                v
     Selected frame indexes
                |
                v
    LEVEL 2 — EXPENSIVE CHECKS
    - AdaFace
    - SSIM
    - LPIPS
    - CLIP
    - DINOv3
    - object/environment checks
                |
                v
       localize_anomalies()
                |
                v
         dense_resample()
                |
                v
       Re-run QA on flagged ranges
                |
                v
       aggregate_evidence()
                |
                v
          decision_engine()
                |
                v
     PASS / AUTO_RETRY / HUMAN_REVIEW
                |
                v
          QAReport JSON
```

RAFT motion analysis and LAB/CIEDE2000 color analysis are intentionally deferred.

## AdaFace Threshold Calibration — Frozen POC Operating Point

The AdaFace identity decision uses a frozen threshold of **0.239** for the current POC.

### Calibration basis

The threshold was selected using an identity-disjoint LFW benchmark:

| Dataset | Same-identity pairs | Different-identity pairs |
|---|---:|---:|
| Calibration | 3,993 | 3,998 |
| Held-out validation | 993 | 990 |
| Total | 4,986 | 4,988 |

All evaluated pairs were filtered for compatibility with the same MTCNN preprocessing used by the AdaFace checker.

Threshold selection was performed on the calibration split and then evaluated without re-fitting on the held-out validation split.

At threshold **0.239** on the held-out validation set:

- Accuracy: **96.02%**
- Precision: **100.00%**
- Recall: **92.04%**
- F1: **95.86%**
- False-positive rate (observed FAR): **0% (0/990 different-identity pairs)**
- False-negative rate (FRR): **7.96% (79/993 same-identity pairs)**

For comparison, the previous POC threshold of 0.400 produced validation accuracy of 95.36%, recall of 90.74%, F1 of 95.14%, with 0/990 observed false positives and 92/993 false negatives.

### Why the threshold is fixed

A QA threshold must remain stable while the current model/version is being evaluated. Otherwise, changing the threshold between generated videos would make QA decisions non-repeatable.

Therefore:

```text
AdaFace model:       ir_50 / MS1MV2 checkpoint
Face preprocessing:  MTCNN alignment
Frozen threshold:    0.239
QA operating point:  current POC v1
```

The threshold should **not** be changed per video or per test case.

### When to recalibrate

Recalibration is required only when there is a material change in the operating distribution or QA requirement, for example:

- the face-recognition model/checkpoint changes;
- face alignment/preprocessing changes;
- the AI video-generation model changes substantially;
- project-specific generated-face data shows a different score distribution;
- the team changes the required false-accept/false-reject trade-off.

A recalibration should use a new identity-disjoint calibration/validation split and preserve the same evaluation procedure.

### Production limitation

The current calibration is a strong benchmark-based POC operating point, but LFW consists of real photographs rather than the final project's generated Indian video faces. Before production deployment, the threshold should be revalidated on a sufficiently large, consented, project-specific dataset of generated video frames. Until then, **0.239 is the frozen threshold for this POC**, not a universal threshold for every face-generation system.

## Thresholds and Decision Mapping

| Component | Threshold | Meaning |
|---|---:|---|
| AdaFace | 0.239 | Below threshold indicates identity mismatch |
| Face identity checker | 0.90 | Below threshold indicates identity mismatch |
| SSIM | 0.90 | Below threshold indicates structural change |
| LPIPS | 0.30 | Above threshold indicates perceptual change |
| CLIP | 0.85 | Below threshold indicates low semantic similarity |
| DINOv3 | 0.80 | Below threshold indicates visual drift |
| Object consistency | 0.70 | Below threshold indicates object drift |
| Environment consistency | 0.70 | Below threshold indicates environment drift |

Clip-level decisions:

| Condition | Decision |
|---|---|
| No applicable failure reason codes | PASS |
| Identity mismatch/missing face, DINOv3 drift, object/environment drift, or fallback drift | AUTO_RETRY |
| Structural, perceptual, or semantic visual change requiring inspection | HUMAN_REVIEW |

Every analyzed frame retains its component scores and reason codes.

## QAResult Contract

The clip-level result contains:

- `qa_id`
- `clip_id`
- `qa_type` = `IDENTITY_VISUAL`
- `component_scores`
- `reason_codes`
- `decision`
- `evidence_files`
- `sampling`
- ClipMetadata and verification evidence when metadata is supplied

Example:

```json
{
  "qa_id": "qa_example",
  "clip_id": "test_clip",
  "qa_type": "IDENTITY_VISUAL",
  "component_scores": {
    "frames": []
  },
  "reason_codes": [],
  "decision": "PASS",
  "evidence_files": [],
  "sampling": {
    "sample_count": 5
  }
}
```

## ClipMetadata and Ingestion

`ingest_clip.py` validates the shared ClipMetadata contract and verifies that the metadata points to the supplied video.

Required metadata includes:

```text
clip_id
shot_id
mode
source_type
file
duration_s
fps
width
height
aspect_ratio
codec
model
settings
reference_ids
motion_controls
seed
status
failure_reason
checksum_sha256
```

The ingest step verifies:

- Video path
- SHA-256 checksum
- Frame count
- FPS
- Resolution

## Project Structure

```text
ai-video-identity-visual-qa-poc/
|
├── src/
│   ├── adaface_checker.py
│   ├── clip_checker.py
│   ├── dinov3_checker.py
│   ├── face_detector.py
│   ├── face_identity_checker.py
│   ├── lpips_checker.py
│   ├── object_environment_checker.py
│   ├── ssim_checker.py
│   ├── qa_engine.py
│   ├── comparison_ui.py
│   ├── keyframe_qa.py
│   │
│   ├── video_level1_checker.py
│   ├── frame_selection.py
│   ├── level2_frame_runner.py
│   ├── anomaly_localizer.py
│   ├── dense_resampler.py
│   ├── evidence_aggregator.py
│   ├── decision_engine.py
│   ├── qa_report_builder.py
│   └── ingest_clip.py
│
├── tests/
│   ├── test_visual_drift.py
│   └── test_day2_clip_qa.py
│
├── mock_data/
│   ├── inputs/
│   │   ├── adaface/
│   │   ├── coco/
│   │   └── member7/
│   │       ├── clip_pan_zoom_001.mp4
│   │       ├── clip_pan_zoom_001.json
│   │       └── pan_zoom_frames/
│   ├── expected/
│   ├── manifest.json
│   ├── generate_fixtures.py
│   ├── generate_coco_fixtures.py
│   └── generate_day2_fixtures.py
│
├── requirements.txt
├── TEST_REPORT.md
└── README.md
```

## Mock Data and Fixtures

The fixture manifest is:

```text
mock_data/manifest.json
```

The current manifest contains 20 labelled fixtures with:

```text
Missing files: 0
Metadata problems: 0
```

Day 2 deliberate transformations include:

- Brightness increase/decrease
- Colour shift
- Strong crop
- Blur
- Strong visual drift
- Zoom transformation
- Different identity
- Missing face
- Object replacement

The pan/zoom clip contains 25 deterministic frames and is encoded as MP4 with FFmpeg.

## Demo

### Clean video

The deterministic clean clip is evaluated against the reference image and produces a successful clip-level result when all sampled frames pass.

### Deliberate failure

The failure fixtures demonstrate identity/appearance changes and propagate explicit reason codes into the clip-level decision.

The comparison UI can be launched with:

```cmd
python -m src.comparison_ui
```

## Automated Tests

Day 2 clip suite:

```cmd
python -m pytest -q tests\test_day2_clip_qa.py
```

Latest result:

```text
6 passed, 4 warnings in 45.94s
```

Complete project suite:

```cmd
python -m pytest -q
```

Latest result:

```text
11 passed, 6 warnings in 54.00s
```

The warnings are non-blocking dependency/library warnings. There are no test failures.

## Reason Codes

### Identity

- `ID_REFERENCE_FACE_MISSING`
- `ID_FACE_MISSING`
- `ID_FACE_MISMATCH`

### Visual

- `VISUAL_STRUCTURAL_CHANGE`
- `VISUAL_PERCEPTUAL_CHANGE`
- `VISUAL_LOW_SIMILARITY`
- `VISUAL_DINOV3_DRIFT`
- `VISUAL_DRIFT_FALLBACK`

### Object / Environment

- `VISUAL_OBJECT_DRIFT`
- `VISUAL_ENVIRONMENT_DRIFT`

## How to Run

Activate the Windows virtual environment:

```cmd
venv\Scripts\activate.bat
```

Run image-level QA:

```cmd
python -m src.qa_engine
```

Run the comparison UI:

```cmd
python -m src.comparison_ui
```

Run keyframe QA:

```cmd
python -m src.keyframe_qa
```

Run Day 2 tests:

```cmd
python -m pytest -q tests\test_day2_clip_qa.py
```

Run the complete suite:

```cmd
python -m pytest -q
```

## Acceptance Criteria

- Known positive and negative fixtures are available.
- Deterministic frame sampling is used.
- Existing Day 1 scoring is reused for sampled frames.
- Component-level evidence is retained.
- Deliberate identity and appearance changes are detected.
- Clip-level decisions use PASS, AUTO_RETRY, and HUMAN_REVIEW.
- QAResult JSON is generated.
- Clip metadata and checksum are verified when supplied.
- Results are repeatable.
- Tests run without requiring a GPU.
- Tests run without requiring internet access.

## Assumptions and Limitations

- Reference images are available locally and suitable for comparison.
- Input videos are readable MP4 files.
- Sampling is deterministic for a fixed configuration.
- Component thresholds may require calibration for production data.
- Day 2 fixtures are development/QA fixtures, not production video data.
- Level 1 uses CPU-light deterministic heuristics.
- RAFT and LAB/CIEDE2000 are deferred.
- Object/environment checking is a deterministic POC and can later be strengthened with production object/segmentation models.

## Downstream Use

The QA layer provides a structured QAResult for later pipeline stages. In particular, `AUTO_RETRY` is retained explicitly so downstream retry logic can consume the decision.
