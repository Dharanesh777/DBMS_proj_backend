#!/usr/bin/env python3
"""
scripts/benchmark_face_alignment.py — tests whether inserting alignment
fixes the FRR gap found by scripts/benchmark_face_backends.py.

CONTEXT: benchmark_face_backends.py found Facenet512 (the app's current
backend) had a 30.95% FRR on LFW, far above published (~1%) benchmarks.
The alignment hypothesis (face_service.py's detector_backend="skip" means
no eye-landmark alignment happens) was a reasonable first suspect — but
this script's findings REFUTE it (see docs/face_recognition_backend_evaluation.md
for the full writeup; the real cause turned out to be face-detection
accuracy, not alignment — see scripts/inspect_face_crop_quality.py).

Kept here anyway since it's a real, reusable test that correctly isolates
alignment as a variable and is worth re-running if the detection-accuracy
issue is ever fixed and alignment becomes a live question again.

METHODOLOGY: unlike benchmark_face_backends.py (which fed whole/pre-cropped
LFW images straight to DeepFace with detector_backend='skip'), this script
runs the APP'S ACTUAL detect_face()/crop_face() functions (imported
directly from face_service.py, not reimplemented) on raw, non-funneled LFW
images to get the true production-equivalent crop, then compares:

  BASELINE : DeepFace.represent(detector_backend='skip', ...)  <- production
  ALIGNED  : DeepFace.represent(detector_backend=<real>, align=True, ...)

on that IDENTICAL crop — isolating alignment as the only variable. Also
separately verified (see docs) that `align` is a true no-op under
detector_backend='skip' (byte-identical embeddings with align=True/False).

HOW TO RUN
    source venv/bin/activate
    python scripts/benchmark_face_alignment.py [--detector opencv|mtcnn]
"""
import argparse
import itertools
import os
import random
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
from deepface import DeepFace
from deepface.config.threshold import thresholds as DEEPFACE_THRESHOLDS
from sklearn.datasets import fetch_lfw_people

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.face_recognition.face_service import detect_face, crop_face  # noqa: E402

random.seed(42)

MAX_IDENTITIES = 15
MAX_IMAGES_PER_IDENTITY = 8
APP_THRESHOLD_CONFIRMED = 0.70


def p(msg):
    print(msg, flush=True)


def cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def far_frr(genuine_sims, impostor_sims, threshold):
    g, i = np.array(genuine_sims), np.array(impostor_sims)
    frr = float((g < threshold).mean()) if len(g) else float("nan")
    far = float((i >= threshold).mean()) if len(i) else float("nan")
    return far, frr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="mtcnn", choices=["opencv", "mtcnn"],
                     help="Real detector to use for the ALIGNED condition (default: mtcnn — "
                          "a landmark-based detector; 'opencv' uses Haar-cascade-based "
                          "eye detection for alignment and was found to be weaker).")
    args = ap.parse_args()

    p("Loading RAW (non-funneled) LFW dataset...")
    data = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True,
                             funneled=False, slice_=(slice(0, 250), slice(0, 250)),
                             download_if_missing=True)
    p(f"Loaded {data.images.shape[0]} images, {len(data.target_names)} identities.")

    by_identity = defaultdict(list)
    for idx, label in enumerate(data.target):
        by_identity[label].append(idx)
    top_identities = sorted(by_identity.keys(), key=lambda k: -len(by_identity[k]))[:MAX_IDENTITIES]
    selected = {data.target_names[l]: by_identity[l][:MAX_IMAGES_PER_IDENTITY] for l in top_identities}
    all_idxs = [idx for idxs in selected.values() for idx in idxs]
    p(f"Using {len(selected)} identities, {len(all_idxs)} images.")

    p("\nRunning the app's actual detect_face()/crop_face() on each raw image...")
    crops = {}
    for idx in all_idxs:
        img_rgb_uint8 = (data.images[idx] * 255).astype(np.uint8)
        frame_bgr = img_rgb_uint8[:, :, ::-1].copy()
        detected, bbox = detect_face(frame_bgr)
        if not detected:
            continue
        crop_rgb = crop_face(frame_bgr, bbox)
        if crop_rgb is not None:
            crops[idx] = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    p(f"detect_face()/crop_face() succeeded on {len(crops)}/{len(all_idxs)} images.")

    conditions = {
        "BASELINE (skip, no alignment — matches production exactly)":
            dict(detector_backend="skip", enforce_detection=False),
        f"ALIGNED ({args.detector} detector + align=True, same crop)":
            dict(detector_backend=args.detector, align=True, enforce_detection=False),
    }

    genuine_pairs, impostor_pairs = [], []
    for label, idxs in selected.items():
        idxs_with_crop = [i for i in idxs if i in crops]
        genuine_pairs += list(itertools.combinations(idxs_with_crop, 2))
    identity_list = [(l, [i for i in idxs if i in crops]) for l, idxs in selected.items()]
    identity_list = [(l, i) for l, i in identity_list if i]
    target = len(genuine_pairs)
    attempts = 0
    while len(impostor_pairs) < target and attempts < target * 30:
        attempts += 1
        (la, ia), (lb, ib) = random.sample(identity_list, 2)
        impostor_pairs.append((random.choice(ia), random.choice(ib)))
    p(f"Built {len(genuine_pairs)} genuine pairs, {len(impostor_pairs)} impostor pairs.\n")

    p("=" * 90)
    p("RESULTS: BASELINE (production-equivalent) vs ALIGNED, same crop")
    p("=" * 90)
    for backend in ["Facenet512", "SFace"]:
        p(f"\n── {backend} ──")
        for cond_name, kwargs in conditions.items():
            t0 = time.time()
            emb = {}
            for idx, crop in crops.items():
                try:
                    result = DeepFace.represent(img_path=crop, model_name=backend, **kwargs)
                    emb[idx] = result[0]["embedding"]
                except Exception:
                    pass
            g_sims = [cosine_sim(emb[a], emb[b]) for a, b in genuine_pairs if a in emb and b in emb]
            i_sims = [cosine_sim(emb[a], emb[b]) for a, b in impostor_pairs if a in emb and b in emb]
            own_thresh = 1 - DEEPFACE_THRESHOLDS[backend]["cosine"]
            far, frr = far_frr(g_sims, i_sims, own_thresh)
            g_arr, i_arr = np.array(g_sims), np.array(i_sims)
            p(f"  {cond_name}  ({time.time()-t0:.1f}s)")
            p(f"    genuine  n={len(g_arr)} mean={g_arr.mean():.4f}   "
              f"impostor n={len(i_arr)} mean={i_arr.mean():.4f}")
            p(f"    at own threshold ({own_thresh:.3f}): FAR={far*100:.2f}%  FRR={frr*100:.2f}%")


if __name__ == "__main__":
    main()
