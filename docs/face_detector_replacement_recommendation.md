# Face detector replacement — options and tradeoffs

**UPDATE: a decision was made after this doc was written.** `fastmtcnn`
was picked and implemented in `face_service.py::detect_face()` — see
`TECH_DEBT.md`'s "Resolved: `detect_face()`'s Haar cascade sometimes
returned the WRONG PERSON'S face" entry for the fix, a subtlety found
during live-path re-verification (tied-confidence tie-breaking order
depending on `align`), and the final live-pipeline numbers. This doc's
comparison and tradeoffs below are kept as-is for the reasoning trail.

Follow-up to `TECH_DEBT.md`'s original bug entry. This doc itself was
diagnosis and options only at the time it was written — the analysis below
predates the implementation decision above. Reproduce with
`python scripts/compare_face_detectors.py`.

## Headline finding: confidence-thresholding doesn't work the way the ticket assumed

The original ticket's fix direction was "reject low-confidence detections
outright, rather than blindly taking the largest Haar blob." Tested this
directly across all four confirmed-available detectors — **for three of
the four, the confidence score is saturated near 1.0 regardless of whether
the detection is actually correct**, so there is no threshold that would
separate the bad cases from the good ones:

| Detector | confidence mean ± std | confidence range | Usable for thresholding? |
|---|---|---|---|
| mtcnn | 0.999 ± 0.004 | 0.980–1.000 | **No** — effectively constant |
| ssd | 1.000 ± 0.001 | 0.990–1.000 | **No** — effectively constant |
| fastmtcnn | 1.000 ± 0.000 | exactly 1.000, always | **No** — always identical |
| yunet | 0.923 ± 0.085 | 0.000–0.950 | Marginal — real variance exists, but its own worst bad-case in this test still scored 0.93 confidence, so it wouldn't have been caught by a threshold either |

This means the *actual* lever available isn't "add a confidence gate" —
none of these four give you one worth relying on. It's "pick a detector
whose underlying face-localization is inherently more accurate," which is
a different, less tunable kind of improvement.

## Does the wrong-face-crop bug reproduce with each detector?

Same methodology that found the original bug: per identity, find the crop
with the lowest mean similarity to that identity's *other* crops; flag it
as likely-wrong if that similarity falls at or below this dataset's
average impostor (different-person) similarity (~0.10–0.17, varies
slightly by detector run — see full script output).

| Detector | Bad identities (of 15) | Verdict |
|---|---|---|
| **fastmtcnn** | **0/15** | Bug did not reproduce in this test |
| yunet | 1/15 (Lula, sim=-0.207) | Reproduces, less often |
| mtcnn | 3/15 (Putin, Sharon, Chavez) | Reproduces |
| ssd | 3/15 (Rumsfeld, Bush, Koizumi) | Reproduces |
| *(reference: Haar, current)* | *9/15* | *Original bug* |

All four are a large improvement over Haar's 9/15. fastmtcnn is the only
one with zero reproductions in this specific test — but see the caveat
below before reading that as "fastmtcnn is definitively bug-free."

**Caveat on all of the above:** n=15 identities, 8 images each, one
dataset (LFW). "0/15 reproduced" is evidence fastmtcnn is more robust here,
not proof the failure mode is eliminated — a larger/different test set
could still find cases. Don't treat any of these four as guaranteed
immune; treat them as ranked by evidence collected so far.

## FAR/FRR with corrected crops (first real accuracy numbers on a working detector)

Same 420 genuine / 420 impostor pairs methodology as prior rounds, each
backend judged at its own DeepFace-documented threshold:

| Detector | Facenet512 FRR | Facenet512 FAR | SFace FRR | SFace FAR |
|---|---|---|---|---|
| *(Haar, current)* | *44.76%* | *0.00%* | *25.00%* | *1.19%* |
| mtcnn | 32.62% | 0.00% | 45.48% | 0.71% |
| yunet | 31.67% | 0.00% | 52.86% | 0.95% |
| ssd | 30.00% | 0.00% | 37.62% | 0.48% |
| fastmtcnn | 30.00% | 0.00% | 40.95% | 0.95% |

Two things worth being direct about:

1. **All four meaningfully improve Facenet512** (44.76% → 30-33% FRR) —
   roughly a third fewer false rejects. **All four make SFace worse**
   relative to its own Haar baseline (25.00% → 38-53%) — consistent with
   `docs/face_recognition_backend_evaluation.md`'s finding that SFace
   underperforms Facenet512 in this pipeline; this round reinforces that,
   it doesn't overturn it.
2. **Even the best combination here (Facenet512 + ssd/fastmtcnn, 30.00%
   FRR) is still far from LFW's published ~99% accuracy for Facenet512.**
   A detector swap alone gets roughly a third of the way there, not all
   the way. There is likely still a residual, unexplained factor beyond
   detection accuracy (candidate: this LFW subset's occlusion/pose
   difficulty, noted in the backend-evaluation doc — hands covering faces,
   non-frontal candid shots — which published benchmarks typically don't
   include as heavily). Not investigated further this round; flagging so
   nobody expects a detector swap alone to close the whole gap.

## Implementation complexity

No meaningful difference found between the four — all were used through
the exact same `DeepFace.represent(img_path=..., model_name=...,
detector_backend=<name>, align=True, enforce_detection=False)` call,
returning `embedding` and `face_confidence` together, with no
detector-specific handling needed in this testing. All four were already
available in the installed `deepface` package with no new dependencies
(confirmed in the prior round via `deepface.modules.modeling.build_model`).
One implementation note that *does* differ from today's code: all four
tested here bypass `detect_face()`'s YOLO person-detection pre-step
entirely (a dedicated face detector doesn't need it) — replacing Haar
alone, inside the existing YOLO→Haar chain, was not tested as a separate
option this round, since the direct full-frame approach is both simpler
and was what the confidence/accuracy numbers above actually measured.

## Summary for whoever makes the pick

- **If avoiding the wrong-face bug is the priority:** fastmtcnn looked
  best in this test (0/15 reproduced, tied-best FRR) — but its confidence
  score gives zero additional safety margin (always exactly 1.000), so
  this pick rests entirely on "better inherent accuracy," not "we can
  reject uncertain detections."
- **If a genuine (if imperfect) confidence signal matters** (e.g. wanting
  *some* per-detection gate even if imprecise): yunet is the only one that
  varies at all — but its own worst failure case in this test wasn't
  caught by it either, so treat this as a weak signal, not a safety net.
- **Latency is explicitly not decided here.** Dev-machine numbers were
  recorded (yunet fastest at ~33ms/image, mtcnn/fastmtcnn slowest at
  ~62-64ms/image) but are not reported as meaningful for the decision —
  same caution as every prior round: this is CPU-bound and needs real Pi 3
  hardware, and `docs/face_recognition_backend_evaluation.md` already
  demonstrated dev-machine timing isn't even stable run-to-run on this
  *same* machine, let alone predictive of Pi 3 behavior.
- **Whichever is picked, expect to still be short of ~99%-class accuracy**
  and plan for a follow-up investigation into the residual gap, not treat
  the detector swap as the final fix.
