#!/usr/bin/env python3
"""
scripts/compare_face_detectors.py — compares 4 confidence-aware detectors
(mtcnn, yunet, ssd, fastmtcnn) against the current Haar-cascade pipeline,
on the wrong-face-crop bug tracked in TECH_DEBT.md ("Haar cascade
sometimes returns the WRONG PERSON'S face"). Diagnosis/options only — does
NOT implement a swap. See docs/face_detector_replacement_recommendation.md
for the writeup of these results and the tradeoffs.

For each detector, tests each raw LFW frame DIRECTLY (bypassing YOLO+Haar
entirely — a real face detector doesn't need a person-detection pre-step,
and that's the realistic replacement: swap the whole detect_face() chain
for one confidence-aware call), captures the returned face_confidence
alongside the embedding, picks the HIGHEST-CONFIDENCE face when multiple
are detected (this is the actual behavior change under test — largest-
blob selection vs confidence-based selection), then runs the same
worst-crop-per-identity diagnostic that found the original bug
(scripts/inspect_face_crop_quality.py's approach — don't just trust
aggregate numbers, check whether the failure mode still reproduces), plus
a full FAR/FRR benchmark with each detector's corrected crops.

Any timing printed here is DEV-MACHINE ONLY — explicitly not meaningful
for detector selection, same caution as every prior round in this
investigation. This is a CPU-bound question that needs real Pi 3 hardware;
a dev-machine number would be actively misleading given how much detector
cost varies by architecture (see docs/face_recognition_backend_evaluation.md's
own dev-machine timing instability finding for why these numbers aren't
even stable run-to-run on the SAME machine).

HOW TO RUN
    source venv/bin/activate
    python scripts/compare_face_detectors.py
"""
import itertools
import random
import time
from collections import defaultdict

import numpy as np
from deepface import DeepFace
from deepface.config.threshold import thresholds as DEEPFACE_THRESHOLDS
from sklearn.datasets import fetch_lfw_people

random.seed(42)

DETECTORS = ["mtcnn", "yunet", "ssd", "fastmtcnn"]
BACKENDS = ["Facenet512", "SFace"]
MAX_IDENTITIES = 15
MAX_IMAGES_PER_IDENTITY = 8
BAD_CROP_THRESHOLD = 0.40  # same bar as inspect_face_crop_quality.py


def p(msg):
    print(msg, flush=True)


def cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


p("Loading RAW (non-funneled) LFW dataset — same 15 identities as prior rounds...")
data = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True, funneled=False,
                         slice_=(slice(0, 250), slice(0, 250)), download_if_missing=True)
by_identity = defaultdict(list)
for idx, label in enumerate(data.target):
    by_identity[label].append(idx)
top_identities = sorted(by_identity.keys(), key=lambda k: -len(by_identity[k]))[:MAX_IDENTITIES]
selected = {data.target_names[l]: by_identity[l][:MAX_IMAGES_PER_IDENTITY] for l in top_identities}
all_idxs = [idx for idxs in selected.values() for idx in idxs]
p(f"Using {len(selected)} identities, {len(all_idxs)} images.\n")

frames = {}
for idx in all_idxs:
    img_rgb_uint8 = (data.images[idx] * 255).astype(np.uint8)
    frames[idx] = img_rgb_uint8[:, :, ::-1].copy()  # BGR, matches app convention

# ── Per-detector: detect+embed (Facenet512, for the crop-quality diagnostic
# and confidence analysis), then also SFace for the FAR/FRR benchmark ──────
all_results = {}  # detector -> backend -> idx -> {"embedding":..., "confidence":..., "n_faces":...}
detect_timings = defaultdict(list)  # DEV-MACHINE ONLY, labeled as such throughout

for detector in DETECTORS:
    p(f"\n{'='*90}\nDETECTOR: {detector}\n{'='*90}")
    all_results[detector] = {b: {} for b in BACKENDS}

    for backend in BACKENDS:
        p(f"  Running {backend} + {detector} on {len(frames)} raw frames...")
        t_start = time.time()
        n_multi = 0
        n_fail = 0
        for idx, frame in frames.items():
            t0 = time.time()
            try:
                results = DeepFace.represent(
                    img_path=frame, model_name=backend, detector_backend=detector,
                    align=True, enforce_detection=False,
                )
            except Exception as e:
                n_fail += 1
                continue
            detect_timings[detector].append(time.time() - t0)
            if len(results) > 1:
                n_multi += 1
            # Pick HIGHEST CONFIDENCE face — the behavior change under test,
            # vs Haar's current "largest blob" selection.
            best = max(results, key=lambda r: r.get("face_confidence", 0.0))
            all_results[detector][backend][idx] = {
                "embedding": best["embedding"],
                "confidence": best.get("face_confidence", None),
                "n_faces_detected": len(results),
            }
        p(f"    {len(all_results[detector][backend])}/{len(frames)} succeeded, "
          f"{n_fail} failed outright, {n_multi} frames had >1 face detected "
          f"(confidence-based selection was actually exercised) "
          f"in {time.time()-t_start:.1f}s [DEV-MACHINE TIMING ONLY]")

# ── Confidence score behavior ────────────────────────────────────────────────
p(f"\n{'='*90}\nCONFIDENCE SCORE BEHAVIOR (Facenet512 pass, representative)\n{'='*90}")
for detector in DETECTORS:
    confs = [v["confidence"] for v in all_results[detector]["Facenet512"].values()
             if v["confidence"] is not None]
    if not confs:
        p(f"  {detector}: NO confidence values returned at all (None for every detection)")
        continue
    arr = np.array(confs)
    n_zero = int((arr == 0.0).sum())
    p(f"  {detector}: n={len(arr)} mean={arr.mean():.3f} std={arr.std():.3f} "
      f"min={arr.min():.3f} max={arr.max():.3f}  (exactly 0.0: {n_zero}/{len(arr)})")

# ── Worst-crop-per-identity diagnostic, per detector (same methodology as
# inspect_face_crop_quality.py — the thing that found the ORIGINAL bug) ────
p(f"\n{'='*90}\nWRONG-FACE-CROP CHECK (same methodology that found the Haar bug)\n{'='*90}")
for detector in DETECTORS:
    p(f"\n── {detector} ──")
    embs = all_results[detector]["Facenet512"]
    n_bad = 0
    worst_list = []
    for label, idxs in selected.items():
        idxs_present = [i for i in idxs if i in embs]
        if len(idxs_present) < 3:
            continue
        mean_sim = {}
        for i in idxs_present:
            others = [cosine_sim(embs[i]["embedding"], embs[j]["embedding"])
                      for j in idxs_present if j != i]
            mean_sim[i] = float(np.mean(others))
        worst_idx = min(mean_sim, key=mean_sim.get)
        worst_list.append((label, worst_idx, mean_sim[worst_idx], embs[worst_idx]["confidence"]))
        if mean_sim[worst_idx] < BAD_CROP_THRESHOLD:
            n_bad += 1
    worst_list.sort(key=lambda x: x[2])
    for label, idx, sim, conf in worst_list:
        flag = "  <-- BAD (below impostor-level similarity)" if sim < BAD_CROP_THRESHOLD else ""
        p(f"    {label:30s} idx={idx:6d} mean_sim={sim:.3f} confidence={conf}{flag}")
    p(f"  => {n_bad}/{len(worst_list)} identities have a worst-crop below "
      f"{BAD_CROP_THRESHOLD} — {'REPRODUCES the wrong-face bug' if n_bad else 'wrong-face bug NOT reproduced'}")

# ── Full FAR/FRR benchmark per detector per backend ──────────────────────────
p(f"\n{'='*90}\nFAR/FRR BENCHMARK — corrected crops per detector\n{'='*90}")

genuine_pairs_by_detector = {}
impostor_pairs_by_detector = {}
for detector in DETECTORS:
    present = set(all_results[detector]["Facenet512"].keys())
    genuine, impostor = [], []
    for label, idxs in selected.items():
        idxs_present = [i for i in idxs if i in present]
        genuine += list(itertools.combinations(idxs_present, 2))
    idlist = [(l, [i for i in idxs if i in present]) for l, idxs in selected.items()]
    idlist = [(l, i) for l, i in idlist if i]
    target = len(genuine)
    attempts = 0
    while len(impostor) < target and attempts < target * 30:
        attempts += 1
        (la, ia), (lb, ib) = random.sample(idlist, 2)
        impostor.append((random.choice(ia), random.choice(ib)))
    genuine_pairs_by_detector[detector] = genuine
    impostor_pairs_by_detector[detector] = impostor

for backend in BACKENDS:
    p(f"\n── {backend} ──")
    own_thresh = 1 - DEEPFACE_THRESHOLDS[backend]["cosine"]
    for detector in DETECTORS:
        embs = all_results[detector][backend]
        gp = genuine_pairs_by_detector[detector]
        ip = impostor_pairs_by_detector[detector]
        g_sims = [cosine_sim(embs[a]["embedding"], embs[b]["embedding"])
                  for a, b in gp if a in embs and b in embs]
        i_sims = [cosine_sim(embs[a]["embedding"], embs[b]["embedding"])
                  for a, b in ip if a in embs and b in embs]
        g_arr, i_arr = np.array(g_sims), np.array(i_sims)
        frr = float((g_arr < own_thresh).mean()) * 100
        far = float((i_arr >= own_thresh).mean()) * 100
        avg_ms = 1000 * np.mean(detect_timings[detector]) if detect_timings[detector] else float("nan")
        p(f"  {detector:12s} n_genuine={len(g_arr):4d} n_impostor={len(i_arr):4d} "
          f"genuine_mean={g_arr.mean():.3f} impostor_mean={i_arr.mean():.3f} "
          f"@thr={own_thresh:.3f}  FAR={far:5.2f}%  FRR={frr:5.2f}%  "
          f"[DEV-MACHINE-ONLY, not a Pi3 proxy: {avg_ms:.1f}ms/image]")

p(f"\n{'='*90}\nFor reference — current Haar pipeline (from prior rounds, same identities/images)\n{'='*90}")
p("  Facenet512 baseline (Haar, skip): genuine_mean=0.603 impostor_mean=0.090 FAR=0.00% FRR=44.76%")
p("  SFace      baseline (Haar, skip): genuine_mean=0.496 impostor_mean=0.172 FAR=1.19% FRR=25.00%")
