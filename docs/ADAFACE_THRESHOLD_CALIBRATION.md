# AdaFace Identity Threshold Calibration Report

## Status

**Frozen POC operating threshold: 0.239**

This threshold is used by the Member 7 identity QA component for the current AI Video Generation POC.

## Objective

Select a repeatable AdaFace cosine-similarity threshold for deciding whether a generated face matches the reference identity.

The threshold must be selected once for the current QA model/version and then kept fixed during normal video evaluation.

## Evaluation design

The benchmark was divided by identity before pair evaluation:

- Calibration identities: 400
- Held-out validation identities: 100
- Random seed: 42
- Calibration and validation identities were disjoint.
- AdaFace used the same MTCNN alignment/preprocessing as the project checker.
- Pairs that could not be processed by the required face alignment pipeline were excluded.

## Pair counts

| Split | Same identity | Different identity | Total |
|---|---:|---:|---:|
| Calibration | 3,993 | 3,998 | 7,991 |
| Validation | 993 | 990 | 1,983 |
| Total | 4,986 | 4,988 | 9,974 |

## Calibration threshold selection

The threshold search was performed on the calibration split.

Selected operating point:

**0.239**

At calibration threshold 0.239:

- Accuracy: 97.62%
- Precision: 100.00%
- Recall: 95.20%
- F1: 97.62% approximately
- Observed FAR: 0%
- Observed FRR: 4.66%

The threshold was then frozen before evaluation on the held-out validation split.

## Held-out validation

At threshold 0.239:

- Accuracy: **96.02%**
- Precision: **100.00%**
- Recall: **92.04%**
- F1: **95.86%**
- False positives: **0 / 990**
- False negatives: **79 / 993**
- Observed FAR: **0%**
- Observed FRR: **7.96%**

This validation result supports using 0.239 as the current POC operating point.

## Comparison with previous threshold

The previous project threshold was 0.400.

| Threshold | Validation accuracy | Precision | Recall | F1 | False positives |
|---:|---:|---:|---:|---:|---:|
| 0.239 | 96.02% | 100.00% | 92.04% | 95.86% | 0 |
| 0.400 | 95.36% | 100.00% | 90.74% | 95.14% | 0 |

The calibrated threshold improves validation recall and F1 while retaining zero observed false positives in this held-out benchmark.

## Fixed-threshold policy

The value **0.239 must remain fixed** for the current POC.

For every generated frame:

- score >= 0.239 → SAME_IDENTITY
- score < 0.239 → DIFFERENT_IDENTITY

The threshold must not be adjusted based on an individual video.

## Recalibration triggers

Recalibrate only when there is a material change such as:

1. AdaFace model/checkpoint changes.
2. Face alignment or preprocessing changes.
3. The AI video-generation model changes substantially.
4. Project-specific generated-face data shows a different score distribution.
5. QA requirements change the acceptable false-accept/false-reject trade-off.

A recalibration must again use identity-disjoint calibration and held-out validation data.

## Production limitation

The benchmark is LFW-based and therefore does not fully represent the final target distribution of AI-generated Indian faces in video.

The current result is therefore a **calibrated and frozen POC operating point**, not a universal face-recognition threshold.

Before production deployment, the team should validate or recalibrate 0.239 using a sufficiently large, consented, project-specific dataset containing generated video frames and realistic variations such as pose, lighting, compression, motion blur, and generation artifacts.

## Implementation

The active project implementation is:

`src/adaface_checker.py`

The identity decision is frozen at:

```python
if similarity >= 0.239:
    label = "SAME_IDENTITY"
else:
    label = "DIFFERENT_IDENTITY"
```

## Versioning

Current QA operating point:

- QA component: Member 7 Identity & Visual Consistency
- Face model: AdaFace IR-50 / MS1MV2
- Preprocessing: MTCNN alignment
- Threshold: **0.239**
- Status: Frozen for current POC
