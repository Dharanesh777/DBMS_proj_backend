import os
import gc
from typing import Dict, Any, Optional
from pywhispercpp.model import Model

# Create a place to store temporary audio files
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Project root models directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS_DIR = os.getenv("WHISPER_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)

# Model Registry (whisper.cpp GGML quantized models)
AVAILABLE_MODELS: Dict[str, Dict[str, str]] = {
    "tiny.en-q5_1": {
        "name": "Tiny",
        "display_name": "Tiny.en Q5_1",
        "filename": "ggml-tiny.en-q5_1.bin",
        "path": os.path.join(MODELS_DIR, "ggml-tiny.en-q5_1.bin"),
    },
    "base.en-q5_1": {
        "name": "Base",
        "display_name": "Base.en Q5_1",
        "filename": "ggml-base.en-q5_1.bin",
        "path": os.path.join(MODELS_DIR, "ggml-base.en-q5_1.bin"),
    },
}

# Alias mapping for user convenience ("Tiny" -> "tiny.en-q5_1", "Base" -> "base.en-q5_1")
MODEL_ALIASES: Dict[str, str] = {
    "tiny": "tiny.en-q5_1",
    "tiny.en": "tiny.en-q5_1",
    "tiny.en-q5_1": "tiny.en-q5_1",
    "base": "base.en-q5_1",
    "base.en": "base.en-q5_1",
    "base.en-q5_1": "base.en-q5_1",
}

# Default configuration: tiny.en-q5_1
DEFAULT_MODEL_KEY = os.getenv("WHISPER_DEFAULT_MODEL", "tiny.en-q5_1")

# Internal single-model active state
_current_model_key: Optional[str] = None
_model_instance: Optional[Model] = None


def resolve_model_key(key_or_name: str) -> str:
    """Normalize user input selector (e.g. 'Tiny', 'base.en-q5_1') to standard model key."""
    normalized = key_or_name.strip().lower()
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized]
    for k in AVAILABLE_MODELS:
        if k.lower() == normalized:
            return k
    raise ValueError(
        f"Unknown Whisper model '{key_or_name}'. Available options: 'Tiny', 'Base' (keys: {list(AVAILABLE_MODELS.keys())})"
    )


def set_model(model_key: str) -> str:
    """
    Safely switches the active Whisper model.
    Unloads the currently loaded model before loading the new model.
    If the requested model is already active, avoids reloading.
    Raises FileNotFoundError if the corresponding model file is missing from models/.
    """
    global _current_model_key, _model_instance

    canonical_key = resolve_model_key(model_key)
    model_info = AVAILABLE_MODELS[canonical_key]
    model_path = model_info["path"]

    # Verify local file presence
    if not os.path.exists(model_path):
        filename = model_info["filename"]
        raise FileNotFoundError(
            f"Missing Whisper model file: '{filename}' (expected path: {model_path}). "
            f"Please download '{filename}' and place it inside the '{MODELS_DIR}' directory."
        )

    # Avoid redundant loading if already active
    if _current_model_key == canonical_key and _model_instance is not None:
        return canonical_key

    # Release existing model memory before allocating new model
    if _model_instance is not None:
        print(f"[Model] Unloading current Whisper model '{_current_model_key}'...", flush=True)
        _model_instance = None
        gc.collect()

    print(f"[Model] Loading Whisper model '{canonical_key}' from local path: {model_path}...", flush=True)
    _model_instance = Model(model_path, n_threads=min(4, os.cpu_count() or 2))
    _current_model_key = canonical_key
    print(f"[Model] Active Whisper model set to '{model_info['display_name']}'", flush=True)
    return canonical_key


def get_model(model_key: Optional[str] = None) -> Model:
    """
    Returns the active pywhispercpp Model instance.
    Loads requested model_key or default model if no model is currently loaded.
    """
    global _current_model_key, _model_instance

    if model_key is not None:
        canonical_key = resolve_model_key(model_key)
        if _current_model_key != canonical_key or _model_instance is None:
            set_model(canonical_key)
    elif _model_instance is None:
        set_model(DEFAULT_MODEL_KEY)

    return _model_instance  # type: ignore


def get_active_model_info() -> Dict[str, Any]:
    """Returns active model metadata including key, display name, and path."""
    key = _current_model_key or DEFAULT_MODEL_KEY
    info = AVAILABLE_MODELS.get(key, AVAILABLE_MODELS["tiny.en-q5_1"]).copy()
    info["key"] = key
    info["is_loaded"] = _model_instance is not None
    return info


def prepare_audio_16k(file_path: str) -> tuple[str, bool]:
    """
    Checks audio sample rate & channels.
    If already 16kHz mono WAV, returns (file_path, False).
    Otherwise converts to a temporary 16kHz mono WAV file and returns (temp_path, True).
    """
    try:
        import soundfile as sf
        import numpy as np

        data, sr = sf.read(file_path)
        if sr == 16000 and (data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1)):
            return file_path, False

        # Convert stereo to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16000 Hz if needed
        if sr != 16000:
            num_samples = int(round(len(data) * 16000 / sr))
            data = np.interp(
                np.linspace(0, len(data), num_samples, endpoint=False),
                np.arange(len(data)),
                data,
            )

        base_name = os.path.basename(file_path)
        temp_16k_path = os.path.join(UPLOAD_DIR, f"preprocessed_16k_{base_name}")
        sf.write(temp_16k_path, data.astype(np.float32), 16000)
        return temp_16k_path, True
    except Exception as e:
        print(f"[AudioPreprocess] Warning: Audio auto-resample fallback ({e})", flush=True)
        return file_path, False


def transcribe_audio(
    file_path: str, model_key: Optional[str] = None, auto_cleanup: bool = False
) -> Optional[str]:
    """
    Takes a path to an audio file and returns transcribed English text using whisper.cpp.
    Automatically handles 16kHz resampling for arbitrary WAV/audio formats.
    """
    temp_resampled_path = None
    try:
        model = get_model(model_key)
        active_info = get_active_model_info()

        target_path, is_temp = prepare_audio_16k(file_path)
        if is_temp:
            temp_resampled_path = target_path

        print(f"[Transcribe] Transcribing '{file_path}' with active model: {active_info['display_name']}...", flush=True)
        segments = model.transcribe(target_path, language="en")
        transcribed_text = " ".join([segment.text for segment in segments]).strip()
        return transcribed_text
    except Exception as e:
        print(f"Transcription Error: {e}", flush=True)
        return None
    finally:
        if temp_resampled_path and os.path.exists(temp_resampled_path):
            try:
                os.remove(temp_resampled_path)
            except OSError:
                pass
        if auto_cleanup and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


if __name__ == "__main__":
    import sys
    import time

    print("=== Audio Transcription Service (whisper.cpp) ===", flush=True)

    target_model = "tiny.en-q5_1"
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            target_model = sys.argv[idx + 1]

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--") and os.path.exists(sys.argv[1]):
        target_audio = sys.argv[1]
        print(f"Processing audio file: {target_audio}", flush=True)
    else:
        target_audio = os.path.join(UPLOAD_DIR, "live_test.wav")
        print("\nRecording audio or using sample...", flush=True)

    try:
        set_model(target_model)
        if os.path.exists(target_audio):
            start_time = time.time()
            result = transcribe_audio(target_audio, model_key=target_model)
            elapsed = time.time() - start_time
            print(f"\n🗣️ Output ({elapsed:.2f}s):\n{result}", flush=True)
        else:
            print(f"Audio file '{target_audio}' not found. Model initialization test passed.", flush=True)
    except FileNotFoundError as fnf:
        print(f"\n❌ Model File Error: {fnf}", flush=True)
