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
    """Detect the actual face bounding box in the frame.
    1. First tries to detect a person using YOLO.
    2. If a person is found, searches for a face inside the person's bounding box using Haar Cascade.
    3. If no face is found inside the person's box, searches the entire frame using Haar Cascade.
    4. If still no face is found, but a person was found, falls back to the top 25% of the person's body box.
    5. If no person was found, runs Haar Cascade on the entire frame.
    """
    person_detected, person_box = detect_person(frame)
    cascade = get_face_cascade()
    
    if person_detected and person_box:
        x1, y1, x2, y2 = person_box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 1. Search for face inside the person box
        if cascade is not None and not cascade.empty() and x2 > x1 and y2 > y1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[y1:y2, x1:x2]
            faces = cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(40, 40))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                xf, yf, wf, hf = faces[0]
                return True, (x1 + xf, y1 + yf, x1 + xf + wf, y1 + yf + hf)
        
        # 2. Fallback: estimate head region (top 25% of body box)
        width = x2 - x1
        height = y2 - y1
        head_height = int(height * 0.25)
        cx = (x1 + x2) // 2
        face_x1 = max(0, cx - head_height // 2)
        face_y1 = y1
        face_x2 = min(w, cx + head_height // 2)
        face_y2 = min(h, y1 + head_height)
        return True, (face_x1, face_y1, face_x2, face_y2)

    # 3. No person detected: run Haar Cascade on the entire frame
    if cascade is not None and not cascade.empty():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            xf, yf, wf, hf = faces[0]
            return True, (xf, yf, xf+wf, yf+hf)
            
    return False, None

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

def clear_face_cache():
    """Clear the cached face encodings to force a refresh on next compare."""
    global _face_encodings_cache
    with _face_encodings_cache_lock:
        _face_encodings_cache = None
    logger.info("Face encodings cache cleared.")

def _load_face_encodings_cache() -> list:
    """Query the DB and build the cache list. Caller must hold _face_encodings_cache_lock."""
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
