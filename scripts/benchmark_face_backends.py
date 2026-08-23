#!/usr/bin/env python3
"""
scripts/benchmark_face_backends.py — face-recognition backend accuracy
benchmark: current (Facenet512) vs SFace.

Part of the face-recognition model decision (plan section 3) — specifically
the ACCURACY half, which doesn't need Pi 3 hardware. The LATENCY half
(actual inference time on a Pi 3 CPU) is a separate, still-blocked
question — see the "DEV-MACHINE-ONLY TIMING" caveat in this script's
output and docs/face_recognition_backend_evaluation.md.

Dataset: LFW (Labeled Faces in the Wild) via sklearn.datasets.fetch_lfw_people
— a standard, widely-used academic face-verification benchmark, downloaded
on first run (~200MB, cached under ~/scikit_learn_data/ after that; needs
internet access once). Checked directly and confirmed: no image assets
exist anywhere in this repo or in the installed deepface package, so this
is a real, honestly-sourced dataset rather than synthetic or fabricated
data — there is no "existing known-persons enrollment images" set in this
repo to benchmark against (only derived embeddings are ever persisted, in
public.faceencoding — the raw images themselves are never stored).

Methodology: for each image, extract an embedding with each backend using
the SAME call pattern face_service.py itself uses (detector_backend='skip',
enforce_detection=False — this isolates the BACKEND/embedding-model
comparison from the app's own YOLO/Haar detection+cropping pipeline, which
stays the same regardless of which backend is chosen). Build genuine pairs
(same identity) and impostor pairs (different identities), compute cosine
similarity per pair per backend, report FAR/FRR at each backend's own
DeepFace-documented threshold (deepface/config/threshold.py, not guessed —
confirmed by reading the installed package), plus what happens if SFace's
output were pushed through the app's CURRENT Facenet512-calibrated
thresholds unchanged (i.e. a naive model_name swap with no recalibration).

Results as of the last run on a dev Mac are written up in
docs/face_recognition_backend_evaluation.md — re-run this script to
refresh them; numbers should be near-identical (fixed random seed) but
worth re-confirming after any deepface/model version bump.

HOW TO RUN
    source venv/bin/activate
    python scripts/benchmark_face_backends.py
"""
import time
import random
import itertools
from collections import defaultdict

import numpy as np
from deepface import DeepFace
from deepface.config.threshold import thresholds as DEEPFACE_THRESHOLDS
from sklearn.datasets import fetch_lfw_people

random.seed(42)

BACKENDS = ["Facenet512", "SFace"]
MAX_IDENTITIES = 15
MAX_IMAGES_PER_IDENTITY = 8

# The app's CURRENTLY CONFIGURED thresholds (face_service.py) — cosine
# SIMILARITY, not distance.
APP_THRESHOLD_CONFIRMED = 0.70
APP_THRESHOLD_UNCERTAIN = 0.60


def p(msg):
    print(msg, flush=True)


def cosine_sim(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


p("Loading LFW dataset (cached after first fetch)...")
data = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True, download_if_missing=True)
p(f"Loaded {data.images.shape[0]} images across {len(data.target_names)} identities.")

# Group image indices by identity, pick the identities with the most images
by_identity = defaultdict(list)
for idx, label in enumerate(data.target):
    by_identity[label].append(idx)

top_identities = sorted(by_identity.keys(), key=lambda k: -len(by_identity[k]))[:MAX_IDENTITIES]
selected = {}
for label in top_identities:
    idxs = by_identity[label][:MAX_IMAGES_PER_IDENTITY]
    selected[data.target_names[label]] = idxs

total_images = sum(len(v) for v in selected.values())
p(f"Using {len(selected)} identities, {total_images} images "
  f"(up to {MAX_IMAGES_PER_IDENTITY} each): {list(selected.keys())}")

# ── Extract embeddings for every image, every backend ───────────────────────
# images from sklearn are float [0,1] RGB; DeepFace expects uint8 BGR (same
# conversion convention face_service.py itself uses before calling represent()).
embeddings = {backend: {} for backend in BACKENDS}  # backend -> idx -> vector
timings = {backend: [] for backend in BACKENDS}

all_idxs = [idx for idxs in selected.values() for idx in idxs]
for backend in BACKENDS:
    p(f"\nExtracting {backend} embeddings for {len(all_idxs)} images...")
    t_backend_start = time.time()
    for i, idx in enumerate(all_idxs):
        img_rgb_uint8 = (data.images[idx] * 255).astype(np.uint8)
        img_bgr = img_rgb_uint8[:, :, ::-1]
        t0 = time.time()
        try:
            result = DeepFace.represent(
                img_path=img_bgr,
                model_name=backend,
                detector_backend="skip",
                enforce_detection=False,
            )
            embeddings[backend][idx] = result[0]["embedding"]
        except Exception as e:
            p(f"  [WARN] {backend} failed on image idx={idx}: {e}")
        timings[backend].append(time.time() - t0)
        if (i + 1) % 30 == 0:
            p(f"  {i+1}/{len(all_idxs)} done...")
    p(f"{backend}: {len(embeddings[backend])}/{len(all_idxs)} embeddings extracted "
      f"in {time.time() - t_backend_start:.1f}s total")

# ── Build genuine (same identity) and impostor (different identity) pairs ──
genuine_pairs = []
for label, idxs in selected.items():
    for a, b in itertools.combinations(idxs, 2):
        genuine_pairs.append((a, b))

identity_list = list(selected.items())
impostor_pairs = []
target_impostor_count = len(genuine_pairs)
attempts = 0
while len(impostor_pairs) < target_impostor_count and attempts < target_impostor_count * 20:
    attempts += 1
    (label_a, idxs_a), (label_b, idxs_b) = random.sample(identity_list, 2)
    a = random.choice(idxs_a)
    b = random.choice(idxs_b)
    impostor_pairs.append((a, b))

p(f"\nBuilt {len(genuine_pairs)} genuine pairs and {len(impostor_pairs)} impostor pairs.")

# ── Compute similarities per backend, per pair type ─────────────────────────
def pair_sims(backend, pairs):
    sims = []
    for a, b in pairs:
        if a in embeddings[backend] and b in embeddings[backend]:
            sims.append(cosine_sim(embeddings[backend][a], embeddings[backend][b]))
    return sims


def summarize(name, sims):
    arr = np.array(sims)
    return f"{name}: n={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}"


def far_frr(genuine_sims, impostor_sims, threshold):
    g = np.array(genuine_sims)
    i = np.array(impostor_sims)
    frr = float((g < threshold).mean()) if len(g) else float("nan")  # genuine wrongly rejected
    far = float((i >= threshold).mean()) if len(i) else float("nan")  # impostor wrongly accepted
    return far, frr


p("\n" + "=" * 78)
p("RESULTS")
p("=" * 78)

results = {}
for backend in BACKENDS:
    g_sims = pair_sims(backend, genuine_pairs)
    i_sims = pair_sims(backend, impostor_pairs)
    results[backend] = (g_sims, i_sims)

    deepface_cosine_distance_threshold = DEEPFACE_THRESHOLDS[backend]["cosine"]
    deepface_similarity_threshold = 1 - deepface_cosine_distance_threshold

    avg_ms = 1000 * np.mean(timings[backend]) if timings[backend] else float("nan")

    p(f"\n── {backend} ──")
    p(summarize("  genuine pairs (same person)", g_sims))
    p(summarize("  impostor pairs (different people)", i_sims))
    p(f"  DeepFace-documented threshold for {backend}: cosine distance "
      f"{deepface_cosine_distance_threshold} -> similarity {deepface_similarity_threshold:.3f}")
    far, frr = far_frr(g_sims, i_sims, deepface_similarity_threshold)
    p(f"  At that threshold: FAR={far:.4f} ({far*100:.2f}%)  FRR={frr:.4f} ({frr*100:.2f}%)")
    p(f"  [DEV-MACHINE-ONLY TIMING, not a Pi 3 proxy] avg embedding time: {avg_ms:.1f} ms/image "
      f"(n={len(timings[backend])})")

p(f"\n── What happens if SFace's output were pushed through the APP'S CURRENT")
p(f"   thresholds ({APP_THRESHOLD_CONFIRMED} confirmed / {APP_THRESHOLD_UNCERTAIN} uncertain)")
p(f"   WITHOUT recalibrating them — i.e. a naive model_name swap today ──")
g_sims, i_sims = results["SFace"]
far_c, frr_c = far_frr(g_sims, i_sims, APP_THRESHOLD_CONFIRMED)
p(f"  At {APP_THRESHOLD_CONFIRMED} (app's 'confirmed' threshold): "
  f"FAR={far_c:.4f} ({far_c*100:.2f}%)  FRR={frr_c:.4f} ({frr_c*100:.2f}%)")
far_u, frr_u = far_frr(g_sims, i_sims, APP_THRESHOLD_UNCERTAIN)
p(f"  At {APP_THRESHOLD_UNCERTAIN} (app's 'uncertain' threshold): "
  f"FAR={far_u:.4f} ({far_u*100:.2f}%)  FRR={frr_u:.4f} ({frr_u*100:.2f}%)")
