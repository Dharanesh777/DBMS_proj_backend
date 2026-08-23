"""
Face Recognition Service
========================
Unified service for face detection and identification.
Uses YOLOv8 for person detection and DeepFace for embeddings.
"""

import os
import json
import logging
import threading
from typing import Optional, Tuple, List

import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO

from app.database.db import get_db_connection

logger = logging.getLogger(__name__)

# --- Configuration ---
YOLO_MODEL_PATH = "yolov8n.pt"
DEEPFACE_MODEL = "Facenet512"
DEEPFACE_DETECTOR = "skip"
DEEPFACE_ENFORCE_DETECTION = False

# Face-localization detector for detect_face() — see TECH_DEBT.md's "Haar
# cascade sometimes returns the WRONG PERSON'S face" entry and
# docs/face_detector_replacement_recommendation.md for the comparison that
# picked this. Confidence-based selection among multiple detected faces
# (see detect_face() below) is the actual fix; fastmtcnn's own confidence
# score is NOT usable as a quality gate (saturates at ~1.0 regardless of
# correctness, confirmed in that doc) — its value here is better inherent
# localization accuracy than Haar's largest-blob selection, not a filter.
FACE_LOCALIZATION_DETECTOR = "fastmtcnn"

THRESHOLD_CONFIRMED = 0.70   # DeepFace's own pre-tuned Facenet512/cosine match threshold (1 - 0.30 distance)
THRESHOLD_UNCERTAIN = 0.60   # Below confirmed but worth asking the user to confirm


# --- State ---
_yolo_model: Optional[YOLO] = None
_face_cascade = None

def _get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        logger.info("Loading YOLOv8: %s", YOLO_MODEL_PATH)
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        # 1. Try OpenCV standard path
        try:
            path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            _face_cascade = cv2.CascadeClassifier(path)
        except Exception:
            pass

        # 2. Try project root fallback with temp file to bypass Windows Unicode path bug
        if _face_cascade is None or _face_cascade.empty():
            import tempfile
            root_path = os.path.abspath(os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml'))
            if os.path.exists(root_path):
                try:
                    temp_path = os.path.join(tempfile.gettempdir(), 'haarcascade_frontalface_default.xml')
                    with open(root_path, 'rb') as f_in:
                        with open(temp_path, 'wb') as f_out:
                            f_out.write(f_in.read())
                    _face_cascade = cv2.CascadeClassifier(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to copy Haar Cascade to temp path: {e}")
                    _face_cascade = cv2.CascadeClassifier(root_path)
            else:
                _face_cascade = cv2.CascadeClassifier(root_path)

        # 3. Last resort check
        if _face_cascade is None or _face_cascade.empty():
            logger.error("Failed to load Haar Cascade. Fallback detection disabled.")
    return _face_cascade

# --- Core Logic ---

def detect_person(frame: np.ndarray) -> tuple[bool, Optional[tuple[int, int, int, int]]]:
    """Detect person using YOLO or fallback to Haar Cascade."""
    model = _get_yolo_model()
    results = model(frame, verbose=False)
    
    best_box = None
    best_conf = 0.0
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf >= 0.50: # 0 is person
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_box = (x1, y1, x2, y2)
    
    if best_box:
        return True, best_box
        
    # Fallback — only if Haar Cascade loaded successfully
    cascade = get_face_cascade()
    if cascade is not None and not cascade.empty():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]
            return True, (x, y, x+w, y+h)

    return False, None

def detect_face(frame: np.ndarray) -> tuple[bool, Optional[tuple[int, int, int, int]]]:
    """Detect the actual face bounding box in the frame using a dedicated
    face detector (FACE_LOCALIZATION_DETECTOR), selecting the
    HIGHEST-CONFIDENCE face when multiple are detected.

    Replaces the previous YOLO-person-box + Haar-cascade-largest-blob
    approach, which sometimes selected the WRONG PERSON's face in
    multi-face frames (confirmed with real photos, not just theoretically —
    see TECH_DEBT.md's "Haar cascade sometimes returns the WRONG PERSON'S
    face" entry and docs/face_detector_replacement_recommendation.md for
    the comparison that picked this replacement). Runs directly on the
    full frame — a dedicated face detector doesn't need YOLO's person-box
    as a pre-filter the way the old Haar-cascade approach did.

    IMPORTANT: with enforce_detection=False, DeepFace.extract_faces() does
    NOT return an empty list when no face is present — it returns exactly
    one full-frame fallback box with confidence=0.0 (confirmed empirically
    against both random noise and a blank frame). Every genuine face
    detection observed in testing scored confidence ~1.0 for this
    detector, so confidence > 0 is used here purely as the "did it find
    anything at all" gate — NOT as a quality/correctness signal (that was
    tested and found unusable for this detector; see the recommendation
    doc — its confidence score doesn't distinguish a correct detection
    from a wrong one, only "something" from "nothing").

    align=True (not False) is deliberate and load-bearing, not cosmetic:
    in a multi-face frame, fastmtcnn frequently returns EVERY face at the
    exact same confidence (1.0000, tied) — confirmed directly on the
    photo that originally exposed this bug (two faces, both 1.0000). When
    tied, max() picks whichever face is FIRST in the returned list, and
    that list's ORDER changes depending on align — align=False returned
    the wrong (bystander's) face first for that photo; align=True
    returned the correct one first. This was caught by re-running
    inspect_face_crop_quality.py against this actual function (not just
    the benchmark script) after an initial align=False attempt regressed
    on exactly this photo. Flagging plainly: this makes the fix's
    reliability partly dependent on DeepFace's internal tie-breaking
    order for align=True, which is empirically better on the tested set,
    not a guaranteed-correct-by-design selection — see TECH_DEBT.md.
    """
    try:
        results = DeepFace.extract_faces(
            img_path=frame,
            detector_backend=FACE_LOCALIZATION_DETECTOR,
            align=True,  # see docstring — this ordering detail is load-bearing
            enforce_detection=False,
        )
    except Exception as e:
        logger.error(f"Face detector ({FACE_LOCALIZATION_DETECTOR}) error: {e}")
        return False, None

    if not results:
        return False, None

    best = max(results, key=lambda r: r.get("confidence") or 0.0)
    if not best.get("confidence"):
        return False, None  # the "nothing found" fallback case, see docstring

    area = best.get("facial_area") or {}
    x, y, bw, bh = area.get("x"), area.get("y"), area.get("w"), area.get("h")
    if x is None or y is None or not bw or not bh or bw <= 0 or bh <= 0:
        return False, None

    frame_h, frame_w = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + bw), min(frame_h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return False, None
    return True, (x1, y1, x2, y2)

def crop_face(frame: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 20) -> Optional[np.ndarray]:
    """Expand bbox and crop face, converting to RGB for DeepFace."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    if x2 <= x1 or y2 <= y1: return None
    
    crop = frame[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(crop_rgb, (224, 224))

def generate_embedding(face_image: np.ndarray) -> Optional[List[float]]:
    """Generate 512-d embedding using DeepFace."""
    try:
        # crop_face() returns RGB, but DeepFace's numpy-array input contract is BGR
        # (see deepface.commons.image_utils.load_image) — convert back before calling it.
        bgr_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        result = DeepFace.represent(
            img_path=bgr_image,
            model_name=DEEPFACE_MODEL,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=DEEPFACE_ENFORCE_DETECTION
        )
        return result[0]["embedding"] if result else None
    except Exception as e:
        logger.error(f"DeepFace error: {e}")
        return None

_face_encodings_cache = None
_face_encodings_cache_lock = threading.Lock()

# ── Redis L2 ──────────────────────────────────────────────────────────────────
# The in-process cache above stays the L1 (unchanged hot-path cost for
# /identify's frequent polling — no Redis round trip on the common case where
# the L1 is already warm). Redis is an L2 that survives a process restart:
# on this deployment's remote (Neon) Postgres, a cold DB connection has real,
# measured latency, so re-warming straight from Redis after e.g. a Pi 3
# reboot avoids waiting on that. Not durability-critical — a Redis miss just
# falls through to the DB exactly as it always has.
REDIS_CACHE_KEY = "agos:face:encodings_cache"
REDIS_CACHE_TTL_SECONDS = 6 * 3600  # safety net if a future clear_face_cache() DEL fails


def _cache_to_json(cache: list) -> str:
    return json.dumps([[pid, vec.tolist()] for pid, vec in cache])


def _cache_from_json(raw: str) -> list:
    return [(pid, np.array(vec, dtype=np.float32)) for pid, vec in json.loads(raw)]


def clear_face_cache():
    """Clear the cached face encodings (both tiers) to force a refresh on next compare."""
    global _face_encodings_cache
    with _face_encodings_cache_lock:
        _face_encodings_cache = None
    try:
        from app.services.redis_client import get_redis
        get_redis().delete(REDIS_CACHE_KEY)
    except Exception as e:
        # Best-effort — REDIS_CACHE_TTL_SECONDS is exactly the backstop for
        # this case, so a failed DEL here isn't a lasting correctness issue.
        logger.warning(f"Could not clear Redis face-cache mirror (will expire via TTL): {e}")
    logger.info("Face encodings cache cleared.")

def _load_face_encodings_cache() -> list:
    """Build the cache list: try the Redis L2 first, then fall back to
    Postgres. Caller must hold _face_encodings_cache_lock."""
    try:
        from app.services.redis_client import get_redis
        raw = get_redis().get(REDIS_CACHE_KEY)
        if raw is not None:
            cached = _cache_from_json(raw)
            logger.info(f"Loaded {len(cached)} face encodings from Redis L2 cache.")
            return cached
    except Exception as e:
        logger.warning(f"Redis L2 face-cache read failed, falling back to DB: {e}")

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT personid, encodingdata FROM public.faceencoding")
            rows = cur.fetchall()
        finally:
            cur.close()
            conn.close()

        cached = []
        for pid, data in rows:
            try:
                stored_vec = np.array(json.loads(data) if isinstance(data, str) else data, dtype=np.float32)
                cached.append((pid, stored_vec))
            except Exception as e:
                logger.error(f"Error parsing encoding for person {pid}: {e}")
        logger.info(f"Loaded {len(cached)} face encodings into cache.")

        try:
            from app.services.redis_client import get_redis
            get_redis().set(REDIS_CACHE_KEY, _cache_to_json(cached), ex=REDIS_CACHE_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Could not mirror face cache to Redis L2: {e}")

        return cached
    except Exception as db_err:
        logger.error(f"Database error loading face encodings: {db_err}")
        return []

def compare_embedding(embedding: List[float]) -> tuple[Optional[int], float, str]:
    """Compare embedding against database and return (person_id, confidence, status)."""
    global _face_encodings_cache

    with _face_encodings_cache_lock:
        if _face_encodings_cache is None:
            _face_encodings_cache = _load_face_encodings_cache()
        cache_snapshot = _face_encodings_cache

    if not cache_snapshot:
        return None, 0.0, "unknown"

    query_vec = np.array(embedding, dtype=np.float32)
    best_pid, best_sim = None, -1.0
    skipped = 0
    
    for pid, stored_vec in cache_snapshot:
        if stored_vec.shape != query_vec.shape:
            skipped += 1
            continue  # Skip dimension-mismatched rows gracefully
        norm_q = np.linalg.norm(query_vec)
        norm_s = np.linalg.norm(stored_vec)
        if norm_q == 0 or norm_s == 0: sim = 0.0
        else: sim = float(np.dot(query_vec, stored_vec) / (norm_q * norm_s))
        
        if sim > best_sim:
            best_sim, best_pid = sim, pid
    
    if skipped:
        logger.warning(f"Skipped {skipped} face encoding(s) with mismatched dimensions.")

    status = "unknown"
    if best_sim >= THRESHOLD_CONFIRMED: status = "confirmed"
    elif best_sim >= THRESHOLD_UNCERTAIN: status = "uncertain"
    
    return best_pid, round(best_sim, 4), status

def fetch_details(person_id: int) -> Optional[dict]:
    """Fetch person name, relationship and latest interaction."""
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name, relationshiptype, notes FROM public.knownperson WHERE personid = %s", (person_id,))
        p = cur.fetchone()
        if not p: return None
        
        name, relationship, notes = p
        
        # Check if they were newly registered via camera
        is_new_registration = notes == "Registered via live camera"
        
        cur.execute("""
            SELECT interactiondatetime, summarytext, emotiondetected, conversation
            FROM public.conversation WHERE personid = %s 
            AND summarytext NOT LIKE '[Face detected%%'
            AND summarytext NOT LIKE 'Person recognized by face%%'
            ORDER BY interactiondatetime DESC LIMIT 1
        """, (person_id,))
        c = cur.fetchone()
        
        # If no 'real' summary found, try to get the very last one anyway but maybe flag it
        if not c:
            cur.execute("""
                SELECT interactiondatetime, summarytext, emotiondetected, conversation
                FROM public.conversation WHERE personid = %s 
                ORDER BY interactiondatetime DESC LIMIT 1
            """, (person_id,))
            c = cur.fetchone()
            # If still has placeholder, mark as None for the UI to handle
            if c and ("[Face detected" in c[1] or "Person recognized" in c[1]):
                c = (c[0], None, c[2], c[3])

        return {
            "name": name,
            "relationship": relationship,
            "is_new_register": is_new_registration,
            "last_date": c[0].strftime("%Y-%m-%d") if c and c[0] else None,
            "last_summary": c[1] if c else None,
            "last_emotion": c[2] if c else None,
            "last_conversation": c[3] if c and len(c) > 3 else None
        }
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
