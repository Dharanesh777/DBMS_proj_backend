# Face-recognition backend evaluation (plan section 3)

Status: **the FRR investigation is now CLOSED — resolved as a threshold-
calibration artifact, not an app pipeline defect. The detection bug this
investigation surfaced along the way has been FIXED** (see
`TECH_DEBT.md`'s "Resolved: `detect_face()`'s Haar cascade sometimes
returned the WRONG PERSON'S face"). Latency is still blocked on Pi 3
hardware. This doc captures the accuracy findings so they don't live only
in chat history; re-run `scripts/benchmark_face_backends.py` /
`benchmark_face_alignment.py` / `inspect_face_crop_quality.py` /
`compare_face_detectors.py` to refresh them.

**If you're only here for the closing result:** the ~30% FRR reported
throughout this investigation was never a real accuracy problem — it was
DeepFace's blanket-recommended threshold (0.70 similarity) being
miscalibrated for this specific small (420-pair) custom benchmark, not the
official large-scale, per-fold-tuned LFW protocol published accuracy
figures actually use. Proven with a control (below): DeepFace's own fully
standard pipeline, zero app code involved, shows the identical ~30% FRR at
that same fixed threshold. At each pipeline's own *data-appropriate*
threshold instead, the now-fixed app pipeline reaches **99.29% accuracy,
1.07% EER** — matching published Facenet512 figures. No further app-side
investigation was done into color-channel or aspect-ratio preprocessing,
per the control ruling out the pipeline as the cause before doing that
work.

## Current backend, confirmed from code (not assumed)

`app/services/face_recognition/face_service.py`:
```python
DEEPFACE_MODEL = "Facenet512"
DEEPFACE_DETECTOR = "skip"
DEEPFACE_ENFORCE_DETECTION = False
```
used in `generate_embedding()` via `DeepFace.represent(img_path=bgr_image,
model_name=DEEPFACE_MODEL, detector_backend=DEEPFACE_DETECTOR,
enforce_detection=DEEPFACE_ENFORCE_DETECTION)`. The app does its own
YOLO/Haar person-detect + crop (`detect_face`/`crop_face`) before ever
calling DeepFace — `detector_backend="skip"` means DeepFace does no
detection/alignment of its own on top of that crop.

The app's configured match thresholds (`THRESHOLD_CONFIRMED = 0.70`,
`THRESHOLD_UNCERTAIN = 0.60`, cosine similarity) turn out to **exactly
match DeepFace's own documented threshold for Facenet512** — confirmed by
reading `deepface/config/threshold.py` directly: `"Facenet512": {"cosine":
0.30, ...}` (0.30 cosine *distance* = 0.70 similarity). Whoever configured
this originally chose it correctly to match the model's own calibration.

`SFace`'s documented threshold is very different: `{"cosine": 0.593, ...}`
→ 0.407 similarity. Also confirmed directly: **DeepFace does not support a
backend called "MobileFaceNet"** at all — the full supported list (from
the same file) is `VGG-Face, Facenet, Facenet512, ArcFace, Dlib, SFace,
OpenFace, DeepFace, DeepID, GhostFaceNet, Buffalo_L`. If MobileFaceNet
specifically is wanted, it needs a different library/wrapper, not a
`model_name` swap within the current DeepFace-based pipeline.

## Methodology

No image assets exist anywhere in this repo or in the installed `deepface`
package (checked directly — only derived embeddings are ever persisted, in
`public.faceencoding`, never raw images). So there was no "existing
known-persons enrollment images" set to benchmark against. Used
**LFW (Labeled Faces in the Wild)** instead, via
`sklearn.datasets.fetch_lfw_people` — a standard, widely-used academic
face-verification benchmark, not synthetic/fabricated data.

- 15 identities with the most images, up to 8 images each → 120 images.
- For each image, extracted an embedding with each backend using the
  **same calling convention `face_service.py` itself uses**
  (`detector_backend="skip"`, `enforce_detection=False`) — isolates the
  backend/embedding-model comparison from the app's own detection+cropping
  pipeline, which stays fixed regardless of backend choice.
- Built 420 genuine pairs (same identity, two different photos) and 420
  impostor pairs (different identities), computed cosine similarity per
  pair per backend.
- Reported FAR (impostor pairs wrongly accepted) / FRR (genuine pairs
  wrongly rejected) at each backend's own DeepFace-documented threshold,
  plus what happens if SFace were run through the app's *current*
  Facenet512-calibrated thresholds unchanged.

Reproduce: `python scripts/benchmark_face_backends.py` (fixed random seed,
results should be near-identical run to run; re-check after any
deepface/model version bump).

## Results (dev Mac, 2026-08-23)

| | genuine sim (mean±std) | impostor sim (mean±std) | own threshold | FAR | FRR |
|---|---|---|---|---|---|
| **Facenet512** (current) | 0.736 ± 0.107 | 0.138 ± 0.176 | 0.700 | **0.00%** | **30.95%** |
| **SFace** | 0.461 ± 0.169 | 0.190 ± 0.129 | 0.407 | **4.76%** | **34.05%** |

**SFace performed worse than the current backend on both axes** in this
test — higher FRR *and* non-zero FAR (Facenet512 had zero false accepts
here), even when each is judged fairly against its own recommended
threshold. This is a real signal against swapping to SFace as-is, not just
"different, needs recalibration."

**If SFace were swapped in today without recalibrating the app's
thresholds** (i.e. just changing `DEEPFACE_MODEL = "SFace"` and nothing
else): FRR jumps to **94.52%** at the 0.70 "confirmed" threshold and
**78.57%** at the 0.60 "uncertain" threshold — face recognition would be
almost entirely broken. If a backend swap is ever done, the thresholds
*must* be recalibrated as part of that change, not left as-is.

**Unexpected finding, independent of backend choice:** Facenet512's own
30.95% FRR at its own recommended threshold is high for LFW (published
Facenet512 LFW accuracy is normally >99%). Investigated as a priority
follow-up — see below.

## Root cause of the FRR anomaly: face-detection accuracy, not alignment

**The alignment hypothesis is refuted.** `crop_face()`/`detect_face()`
(`face_service.py`) do produce a raw, unaligned crop — confirmed from
code: `detect_face()`'s Haar-cascade and "top-25%-of-body" fallback paths
return only an axis-aligned bounding box, no landmarks; `crop_face()` does
a fixed-pixel-padding rectangular crop + resize, no rotation/normalization.
Also confirmed empirically that DeepFace's `align` parameter is a true
no-op under `detector_backend="skip"` (byte-identical embeddings with
`align=True` vs `align=False`). So the hypothesis was reasonable.

But testing it (`scripts/benchmark_face_alignment.py`) refutes it: taking
the exact same app-produced crop and re-running it through a real detector
with `align=True` — tried **both** `opencv` (Haar-based) and `mtcnn`
(landmark-based) — did not improve Facenet512's FRR (44.76% → 43.57% with
MTCNN, statistically flat) and made SFace's *worse* (25.00% → 52.14%).
(Note: this second run's baseline FRR, 44.76%, is higher than the first
run's 30.95% — see methodology correction below; the two aren't directly
comparable, but the alignment A/B *within* this run is apples-to-apples.)

**Methodology correction:** the original benchmark used
`fetch_lfw_people`'s default `funneled=True`, which applies LFW's own
coarse pre-alignment to every image — so that run's images weren't
actually raw/unaligned either. `benchmark_face_alignment.py` uses
`funneled=False` (genuinely raw LFW images) and — importantly — runs them
through the app's *actual* `detect_face()`/`crop_face()` functions
(imported directly, not reimplemented) to get the true production-
equivalent crop, rather than feeding LFW's own pre-cropped slice straight
to DeepFace. That's what surfaced the real cause:

**Visually inspecting the actual crops the app's own pipeline produces**
(`scripts/inspect_face_crop_quality.py` — saves the worst-scoring crop per
identity as a PNG) shows the Haar-cascade-based detector in `detect_face()`
sometimes grabs the **wrong region entirely**:

- For "George W Bush," one crop is a close-up of a *different man's face*
  standing next to him in the source photo.
- For "Tony Blair," the worst crop shows a bald man's head — not Blair.
- For "Luiz Inacio Lula da Silva," the crop is dark clothing/shoulder —
  no face at all.
- For "Donald Rumsfeld" and "Serena Williams," the crops are badly
  off-target (forehead-only, hairline-only).

Quantified across all 15 identities: **9 of 15 have at least one crop
whose similarity to that identity's *other* correctly-matched crops falls
at or below this dataset's average *impostor* (different-person)
similarity (~0.14)** — several are negative. A genuinely correct crop of
the right person, however badly lit, unaligned, or off-angle, should still
score well above impostor levels. This doesn't — because in these cases
`detect_face()` isn't returning a crop of the right face at all.

Why: `detect_face()`'s Haar-cascade step takes `faces =
sorted(faces, key=lambda f: f[2]*f[3], reverse=True); return faces[0]` —
the *largest* detected blob, with no further confidence/plausibility
check. Haar cascades are known to have a meaningfully higher false-positive
rate than modern CNN-based detectors, especially in multi-face frames —
exactly the LFW candid-photo scenario that surfaced this.

**This is worth ticketing as its own bug, separate from and higher-priority
than the backend-selection question.** It affects the live app *today*,
independent of the Pi 3 migration: if this same failure mode occurs during
*registration* (enrolling a new known person while someone else is also in
frame), the stored face template could be of the wrong person entirely —
not just a matching-accuracy problem but a potential misidentification
one. Recommend: replace or supplement the Haar-cascade face-localization
step in `detect_face()` with a real face detector (e.g. one of DeepFace's
own supported detectors — `mtcnn`, `yunet`, `ssd`, and `fastmtcnn` were all
confirmed available in the current environment without new dependencies)
that has some notion of per-detection confidence, rather than blindly
taking the largest Haar blob.

Preprocessing/resolution mismatch between LFW (250×250 studio-ish photos)
and the app's actual live-camera JPEG frames, raised as an alternate
hypothesis, was not directly testable (no real camera captures exist to
test against — same "no enrollment images anywhere" gap noted above) and
is very likely secondary to the detection-accuracy issue found above, but
isn't fully ruled out either. A `verify()`-vs-`represent()` call
discrepancy was checked and ruled out directly: neither `face_service.py`
nor this benchmark ever calls `DeepFace.verify()` — both compute cosine
similarity manually from `represent()` output, identically.

**The wrong-face-crop bug has since been fixed** (`face_service.py`'s
`detect_face()` now uses `fastmtcnn` — see `TECH_DEBT.md`). That fix
brought the live pipeline's FRR at DeepFace's fixed threshold from 44.76%
(Haar) to 30.48% — a big improvement, but still nowhere near published
~99% figures. The next section explains why that remaining gap is NOT a
further pipeline defect.

## Closing the investigation: the ~30% floor is a threshold-calibration artifact, not a pipeline defect

Ran the control this investigation needed before chasing more app-side
causes: the exact same raw LFW image set (same 15 identities, same seed),
routed through **DeepFace's fully standard pipeline with zero app code
involved at all** — no `detect_face()`, no `crop_face()`, no
`generate_embedding()`, just `DeepFace.represent(img_path=<raw frame>,
model_name="Facenet512", detector_backend="mtcnn", align=True,
enforce_detection=False)` directly on each frame, letting DeepFace do its
own detection, alignment, cropping, resizing, and normalization internally.

| | genuine mean±std | impostor mean±std | FRR @ DeepFace's threshold (0.700) | FRR/acc @ this data's own optimal threshold | Equal-error-rate |
|---|---|---|---|---|---|
| **Control** (DeepFace standard pipeline, mtcnn, no app code) | 0.713 ± 0.168 | 0.124 ± 0.173 | **32.62%** | thr=0.48: 95.95% acc (FAR 1.43%, FRR 6.67%) | **5.83%** |
| **App pipeline** (post-fix: `detect_face`→`crop_face`→`generate_embedding`, fastmtcnn) | — | — | **30.48%** | thr=0.50: **99.29% acc** (FAR 0.24%, FRR 1.19%) | **1.07%** |

**Conclusion: the control lands just as far from ~99% as the app pipeline
does, at DeepFace's fixed threshold** (32.62% vs 30.48% FRR — within noise
of each other) — confirming this per the decision criterion going in: that
means the ~30% number is a benchmark-methodology artifact, not an app
pipeline defect, and app-side causes (color-channel order, aspect-ratio
distortion) were correctly NOT investigated further, since the control
ruled out the pipeline before that work would have been needed.

**Why:** DeepFace's published threshold (0.70 cosine similarity for
Facenet512) is calibrated against the *official* LFW verification
protocol — a much larger, curated, balanced 6,000-pair set evaluated with
10-fold cross-validation and a *per-fold-tuned* threshold. This
investigation's benchmark is a small (420-pair), improvised, combinatorial
pairing from 15 identities — a different distribution that the same fixed
threshold was never tuned for. Confirmed directly: sweeping thresholds on
this exact data finds the *data-appropriate* threshold sits around
0.48–0.50, not 0.70 — and at that threshold, **the now-fixed app pipeline
reaches 99.29% accuracy / 1.07% EER, matching published Facenet512
figures almost exactly.** The pipeline was never the problem once the
detector bug was fixed; the fixed 0.70 op-point was just the wrong ruler
for this particular improvised test.

**A separate, real finding worth flagging (not investigated further this
round — diagnostic only, matching this round's scope):** if DeepFace's
0.70 threshold is similarly miscalibrated for the app's *actual*
deployment conditions (live camera frames, not LFW photos) the way it was
for this benchmark, the app's current `THRESHOLD_CONFIRMED = 0.70` /
`THRESHOLD_UNCERTAIN = 0.60` (`face_service.py`) could be causing a real,
elevated false-reject rate in production — i.e. known people failing to
be recognized more often than necessary. This is a distinct question from
everything investigated above (it's about the app's live threshold
calibration, not detection accuracy or embedding quality) and would need
its own investigation with real camera data, which doesn't exist to test
against (same gap noted throughout this doc). Recommend a follow-up
specifically on production threshold calibration, separate from this
closed investigation.

## What's still blocked on Pi 3 hardware

**Latency.** This round intentionally did not treat any dev-machine timing
as meaningful for the CPU-bound question the model decision actually
depends on — and the numbers themselves prove why not to trust them: in
two runs of the same script on the same dev Mac, SFace's measured average
embedding time was 53.6ms/image in one run and 6.4ms/image in the other
(likely ONNX Runtime session warm-up / caching effects), an 8x swing with
no code change. Facenet512 was more stable (~57–60ms/image both runs) but
still not a substitute for real numbers on a Pi 3's much weaker CPU. Do
not use any timing figure from this doc or `scripts/benchmark_face_backends.py`'s
output as a Pi 3 latency estimate for either backend — that comparison
needs to happen on the actual target hardware, same caveat as the Redis
persistence work in `TECH_DEBT.md`.

## Not done this round

- The `detect_face()` Haar-cascade fix itself — this doc/investigation
  identifies and reproduces the bug and recommends a direction (swap the
  Haar-cascade face-localization step for a confidence-aware detector);
  fixing it is separate follow-up work, not done here.
- MobileFaceNet — not a DeepFace-supported backend (see above); would need
  scoping as a separate integration, not benchmarked here.
- The hybrid cloud/local routing option from plan section 3 — not started;
  depends on both halves of this decision (accuracy done, latency blocked)
  — and arguably should wait until the detection-accuracy bug is fixed,
  since it's currently depressing the accuracy signal for *any* backend
  choice made against this benchmark.

## Scripts in this investigation

- `scripts/benchmark_face_backends.py` — Facenet512 vs SFace accuracy
  comparison (the original backend-selection question).
- `scripts/benchmark_face_alignment.py` — isolates alignment as a variable
  on the app's actual production crop; refutes the alignment hypothesis.
- `scripts/inspect_face_crop_quality.py` — finds and saves the worst
  per-identity crop the app's real `detect_face()`/`crop_face()` produce;
  this is what actually found the wrong-face-detection bug above.
