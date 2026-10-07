# Day 2 Test Report — Identity & Visual Consistency QA

## 1. Test Summary

Day 2 validates Member 7 Identity & Visual Consistency QA using deterministic local fixtures and mock video clips.

Validated areas:

- Deterministic video frame sampling
- Per-frame identity and visual QA
- Level 1 cheap video checks
- Level 2 selected-frame checks
- Anomaly localization
- Dense resampling
- Evidence aggregation
- PASS / AUTO_RETRY / HUMAN_REVIEW decision mapping
- QAResult JSON generation
- ClipMetadata ingestion and verification
- Clean and deliberate-failure cases
- Component evidence retention

## 2. Test Environment

- Python: 3.11.9
- Test framework: pytest
- GPU required: No
- Internet required: No
- Fixture type: local deterministic media
- Video processing: OpenCV / FFmpeg
- Identity model: AdaFace
- Visual model: DINOv3
- Additional metrics: SSIM, LPIPS, CLIP
- Object/environment checks: deterministic POC

## 3. Test Commands and Final Results

### Day 2 clip suite

```cmd
python -m pytest -q tests\test_day2_clip_qa.py
```

Result:

```text
6 passed, 4 warnings in 45.94s
```

### Complete project suite

```cmd
python -m pytest -q
```

Final result:

```text
11 passed, 6 warnings in 54.00s
```

There were no test failures. The warnings are non-blocking dependency/library warnings.

## 4. Day 2 Workflow Validation

```text
Clip + ClipMetadata
       |
       v
ingest_clip()
       |
       v
Decode all frames
       |
       v
Level 1 cheap checks
       |
       v
Merge coarse + anomaly candidate frames
       |
       v
Level 2 selected-frame QA
       |
       v
Anomaly localization
       |
       v
Dense resampling
       |
       v
Evidence aggregation
       |
       v
Decision engine
       |
       v
QAReport / QAResult JSON
```

Level 1 checks include frame difference, brightness delta, color delta, frozen-frame detection, motion energy, and deterministic coarse sampling.

Level 2 reuses the Day 1 QA engine for selected frames.

## 5. Deterministic Frame Sampling

For the 25-frame pan/zoom fixture with 5 coarse samples:

```text
0, 6, 12, 18, 24
```

The selection is deterministic and uses NumPy-based evenly spaced indexes.

## 6. Component Evidence Retention

Each analyzed frame retains:

- Identity / AdaFace result
- SSIM
- LPIPS
- CLIP
- DINOv3
- Object consistency
- Environment consistency
- Frame decision
- Frame reason codes

The evidence is aggregated without collapsing all metrics into one opaque score.

## 7. QAResult JSON Generation

The clip-level QAResult contains:

- `qa_id`
- `clip_id`
- `qa_type`
- `component_scores`
- `reason_codes`
- `decision`
- `evidence_files`
- `sampling`

When ClipMetadata is supplied, the result also retains metadata path and verification details including checksum, frame count, FPS, and resolution.

## 8. Fixture Validation

The manifest contains:

```text
20 fixtures
Missing files: 0
Metadata problems: 0
```

Fixtures include positive examples and deliberate transformations such as brightness changes, colour shift, crop, blur, strong visual drift, zoom, different identity, missing face, and object replacement.

## 9. Reproducibility

The Day 2 fixtures use deterministic local transformations.

The pan/zoom clip is generated from a fixed sequence of 25 frame images and encoded with FFmpeg.

No random operations are required for the deterministic fixture workflow.

## 10. Decision Mapping

The shared decision vocabulary is:

```text
PASS
AUTO_RETRY
HUMAN_REVIEW
```

Clear identity or severe visual drift conditions map to `AUTO_RETRY`. Structural, perceptual, or semantic visual changes requiring inspection map to `HUMAN_REVIEW`. No applicable failure reason codes map to `PASS`.

## 11. Deliberate Failure Validation

The implementation detects deliberate failures and propagates explicit reason codes.

Validated failure categories include:

- Identity mismatch
- Missing face
- Strong crop
- Blur
- Strong visual drift
- Object replacement
- Appearance changes

## 12. Acceptance Criteria

| Requirement | Result |
|---|---|
| Deterministic frame sampling | PASS |
| Clean video passes | PASS |
| Deliberate failure detected | PASS |
| Component scores retained | PASS |
| QAResult generated | PASS |
| ClipMetadata consumed and verified | PASS |
| Positive and negative fixtures available | PASS |
| Repeatable automated tests | PASS |
| GPU-free test execution | PASS |
| Internet-free test execution | PASS |
| Level 1 checks implemented | PASS |
| Level 2 selected-frame checks implemented | PASS |
| Anomaly localization implemented | PASS |
| Dense resampling implemented | PASS |
| Evidence aggregation implemented | PASS |
| Decision mapping implemented | PASS |

## 13. Final Test Status

```text
DAY 2 TEST STATUS: PASS

Day 2 clip suite: 6 passed, 4 warnings
Full project suite: 11 passed, 6 warnings
Failures: 0
```
