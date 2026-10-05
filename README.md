# AI Video Generation — Identity & Visual Consistency QA

QA module for the AI Video Generation POC that evaluates whether generated images and video frames remain consistent with a reference image.

The module combines independent identity, visual, object, and environment consistency checks and produces an explainable PASS/FAIL QA report.

---

## Purpose

The goal of this module is to detect identity and visual consistency problems in AI-generated content.

The QA engine evaluates:

- Face identity consistency
- Structural similarity
- Perceptual similarity
- Semantic visual similarity
- Object consistency
- Environment consistency
- Keyframe consistency
- Deliberate visual drift
- Object replacement
- Missing-face cases

The system does not rely on one opaque similarity score.

Instead, it retains component-level evidence and produces explicit reason codes explaining why a result passed or failed.

---

## QA Architecture

The QA checks are independent evidence-producing components.

They are conceptually evaluated in parallel and their results are combined by the central QA engine.

```text
                    Reference Image
                           +
                    Generated Image
                           |
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    Identity QA        Visual QA      Object / Environment
          │                │                │
     ┌────┴────┐      ┌────┼────┬────┐      │
     │         │      │    │    │    │      │
  AdaFace  PixelFace SSIM LPIPS CLIP DINOv3  │
     │         │      │    │    │    │       │
     └────┬────┘      └────┴────┴────┘       │
          │                                   │
          │                          Object consistency
          │                          Environment consistency
          │                                   │
          └────────────────┬──────────────────┘
                           ▼
                      QA Engine
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                  PASS           FAIL
                                  │
                           Reason Codes
                                  │
                           JSON QA Report

```

---


## Tools & Models

The module uses multiple complementary checks:

| Component | Purpose |
|---|---|
| AdaFace | Primary face identity similarity |
| PixelFace Prototype | Lightweight identity evidence / fallback |
| SSIM | Structural similarity |
| LPIPS | Perceptual visual similarity |
| CLIP | Semantic visual similarity |
| DINOv3 | Visual representation and drift detection |
| OpenCV | Face detection and image processing |
| Object/Environment Checker | Object and scene consistency |
| Gradio | Visual comparison UI |
| Pytest | Automated testing |

The system is designed so that individual component evidence can be inspected in the final QA report.

---

## Reason Codes

The QA engine produces explicit reason codes instead of returning only a similarity score.

### Identity

- `ID_REFERENCE_FACE_MISSING` — reference image does not contain a detectable face.
- `ID_FACE_MISSING` — generated image does not contain a detectable face.
- `ID_FACE_MISMATCH` — generated face does not sufficiently match the reference identity.

### Visual

- `VISUAL_STRUCTURAL_CHANGE` — structural similarity is below the configured SSIM threshold.
- `VISUAL_PERCEPTUAL_CHANGE` — perceptual difference is above the configured LPIPS threshold.
- `VISUAL_LOW_SIMILARITY` — CLIP semantic similarity is below the configured threshold.
- `VISUAL_DINOV3_DRIFT` — DINOv3 detects significant visual representation drift.
- `VISUAL_DRIFT_FALLBACK` — combined SSIM + LPIPS + CLIP fallback detects visual drift.

### Object / Environment

- `VISUAL_OBJECT_DRIFT` — object consistency check detects a significant object change.
- `VISUAL_ENVIRONMENT_DRIFT` — environment/scene consistency check detects a significant change.

A result is marked `FAIL` when one or more applicable failure reason codes are triggered.

---

## Project Structure

```text
ai-video-identity-visual-qa-poc/
│
├── src/
│   ├── adaface_checker.py
│   ├── clip_checker.py
│   ├── dinov3_checker.py
│   ├── face_detector.py
│   ├── face_identity_checker.py
│   ├── keyframe_qa.py
│   ├── lpips_checker.py
│   ├── object_environment_checker.py
│   ├── qa_engine.py
│   ├── ssim_checker.py
│   ├── comparison_ui.py
│   └── create_test_video.py
│
├── tests/
│   └── test_visual_drift.py
│
├── mock_data/
│   ├── inputs/
│   │   ├── adaface/
│   │   ├── coco/
│   │   ├── clean_test_video.mp4
│   │   └── test_video.mp4
│   │
│   ├── expected/
│   │   ├── keyframes/
│   │   └── visual_drift_qa_report.json
│   │
│   ├── metadata.json
│   ├── manifest.json
│   ├── generate_fixtures.py
│   └── generate_coco_fixtures.py
│
├── requirements.txt
└── README.md
```
---

## Output

The central QA engine produces a structured JSON report containing:

- Identity model results
- Identity similarity scores
- SSIM score
- LPIPS score
- CLIP similarity
- DINOv3 result
- Object consistency result
- Environment consistency result
- Fallback status
- Final PASS/FAIL decision
- Explicit reason codes

Example:

```json
{
  "decision": "FAIL",
  "reason_codes": [
    "VISUAL_STRUCTURAL_CHANGE",
    "VISUAL_PERCEPTUAL_CHANGE"
  ]
}

The full component-level evidence is retained in the report.
```
---

## How to Run

### 1. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```
### 2. Run the QA engine
```bash
python -m src.qa_engine
```
```markdown
```
### 3. Run the comparison UI
```bash
python -m src.comparison_ui
```
### 4. Run keyframe QA
```bash
python -m src.keyframe_qa
```
## Mock Fixture Workflow

The module is designed to run locally with deterministic mock data before using real AI-generated videos.

Fixtures include:

- Identity-positive examples
- Identity-negative examples
- Visual drift examples
- Object replacement examples
- Clean video examples
- Deliberately corrupted video frames
- COCO-based object/environment examples

The fixture manifest records the expected result and failure type for each test case.

This makes the QA pipeline reproducible and allows deliberate failures to be demonstrated during testing.
```
```
## Demo

### PASS Example

A reference image compared against a visually consistent generated image produces:

```text
Decision: PASS
```
### FAIL Example

A deliberately modified image or video frame produces:

```text
Decision: FAIL
```
## Acceptance Criteria

The module satisfies the following QA requirements:

- Known positive and negative fixtures produce separated results.
- Identity changes can be detected.
- Visual appearance changes can be detected.
- Object replacement can be detected.
- Environment changes can be detected.
- Deliberate video-frame corruption can be detected.
- Keyframes can be evaluated against a reference.
- Component-level evidence is retained.
- Explicit reason codes explain failures.
- QA results are available as structured JSON.
- Automated tests pass.
- A visual comparison UI is available.
- The module can run locally without depending on a live AI-video generation service.
```
```
## Deliverables

This module provides:

1. **Identity consistency scoring**
2. **Visual consistency scoring**
3. **Object consistency checking**
4. **Environment consistency checking**
5. **Keyframe QA**
6. **Explainable QA reason codes**
7. **Structured JSON QA reports**
8. **Deterministic mock fixtures**
9. **Automated tests**
10. **Gradio comparison UI**

The module is intended to serve as the QA layer for the broader AI Video Generation POC.
