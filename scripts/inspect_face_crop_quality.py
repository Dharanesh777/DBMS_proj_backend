#!/usr/bin/env python3
"""
scripts/inspect_face_crop_quality.py — finds and saves the worst-quality
face crops the app's own detect_face()/crop_face() pipeline produces, per
identity, on a real (LFW) dataset.

WHY THIS EXISTS: this is the script that actually found the real cause of
the FRR anomaly in docs/face_recognition_backend_evaluation.md — not
missing alignment (ruled out by benchmark_face_alignment.py), but the Haar-
cascade-based detector in face_service.py's detect_face() sometimes
returning a face box for the WRONG PERSON (in a multi-face frame) or a
badly-off-target crop (e.g. just a forehead, or no face at all). Visually
inspecting the actual output — not just the numbers — is what surfaced
this; this script automates "find the worst outlier per identity and save
it as a PNG so a human (or Claude, via the Read tool) can look at it."

METRIC: for each identity's set of crops, computes the mean cosine
similarity of each crop's Facenet512 embedding to its OTHER same-identity
crops. A crop of the genuinely correct person, even badly lit or off-
angle, should still score well above typical impostor similarity
(~0.10-0.15 in this dataset). A crop scoring at or below that is a strong
signal the detector grabbed the wrong region entirely.

HOW TO RUN
    source venv/bin/activate
    python scripts/inspect_face_crop_quality.py [--out-dir DIR]
Then open the saved PNGs in DIR (default: ./crop_quality_outliers/) to see
what the detector actually produced for the worst case per identity.
"""
import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from deepface import DeepFace
from sklearn.datasets import fetch_lfw_people

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.face_recognition.face_service import detect_face, crop_face  # noqa: E402

MAX_IDENTITIES = 15
MAX_IMAGES_PER_IDENTITY = 8


def p(msg):
    print(msg, flush=True)


def cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="crop_quality_outliers")
    ap.add_argument("--bad-threshold", type=float, default=0.40,
                     help="Flag a crop as a likely bad detection if its mean "
                          "similarity to its own identity's other crops falls "
                          "below this (impostor pairs in this dataset average ~0.14).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    p("Loading RAW (non-funneled) LFW dataset...")
    data = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True,
                             funneled=False, slice_=(slice(0, 250), slice(0, 250)),
                             download_if_missing=True)
    by_identity = defaultdict(list)
    for idx, label in enumerate(data.target):
        by_identity[label].append(idx)
    top_identities = sorted(by_identity.keys(), key=lambda k: -len(by_identity[k]))[:MAX_IDENTITIES]
    selected = {data.target_names[l]: by_identity[l][:MAX_IMAGES_PER_IDENTITY] for l in top_identities}

    p(f"Using {len(selected)} identities.\n")
    worst_per_identity = []
    n_bad = 0

    for label, idxs in selected.items():
        crops_bgr, crops_rgb = {}, {}
        for idx in idxs:
            img_rgb_uint8 = (data.images[idx] * 255).astype(np.uint8)
            frame_bgr = img_rgb_uint8[:, :, ::-1].copy()
            detected, bbox = detect_face(frame_bgr)
            if not detected:
                continue
            crop = crop_face(frame_bgr, bbox)  # RGB
            if crop is not None:
                crops_rgb[idx] = crop
                crops_bgr[idx] = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        embeddings = {}
        for idx, crop in crops_bgr.items():
            result = DeepFace.represent(img_path=crop, model_name="Facenet512",
                                         detector_backend="skip", enforce_detection=False)
            embeddings[idx] = result[0]["embedding"]

        if len(embeddings) < 3:
            continue

        mean_sim = {}
        for idx in embeddings:
            others = [cosine_sim(embeddings[idx], embeddings[j]) for j in embeddings if j != idx]
            mean_sim[idx] = float(np.mean(others))

        worst_idx = min(mean_sim, key=mean_sim.get)
        worst_per_identity.append((label, worst_idx, mean_sim[worst_idx]))
        if mean_sim[worst_idx] < args.bad_threshold:
            n_bad += 1
            safe_label = label.replace(" ", "_")
            out_path = os.path.join(args.out_dir, f"{safe_label}_worst_sim{mean_sim[worst_idx]:.2f}.png")
            cv2.imwrite(out_path, cv2.cvtColor(crops_rgb[worst_idx], cv2.COLOR_RGB2BGR))

    worst_per_identity.sort(key=lambda x: x[2])
    p("Per-identity worst-outlier crop (lowest mean similarity to own identity-mates):")
    for label, idx, sim in worst_per_identity:
        flag = "  <-- BAD, saved for inspection" if sim < args.bad_threshold else ""
        p(f"  {label:30s} idx={idx:6d}  mean_sim={sim:.3f}{flag}")

    p(f"\n{n_bad}/{len(worst_per_identity)} identities have a worst-crop below "
      f"{args.bad_threshold} mean similarity to their own identity-mates.")
    p(f"Saved images written to: {args.out_dir}/")
    p("Open them — a genuinely correct-but-unaligned crop of the right person "
      "should never look this bad; expect to see wrong-face detections, "
      "badly-off-target crops, or non-face crops instead.")


if __name__ == "__main__":
    main()
