import os
from faster_whisper import WhisperModel

# Create a place to store temporary audio files
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Raspberry Pi Optimized: English-only 'tiny.en' (or 'base.en') with int8 quantization
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
_model_instance = None


def get_model():
    """
    Returns a singleton instance of the faster-whisper model configured for CPU with int8 quantization.
    """
    global _model_instance
    if _model_instance is None:
        # compute_type="int8" optimizes CPU memory footprint and execution speed on Raspberry Pi
        _model_instance = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model_instance


def transcribe_audio(file_path: str, auto_cleanup: bool = False) -> str | None:
    """
    Takes a path to an audio file and returns the transcribed English text.
    Optimized for low-resource environments like Raspberry Pi.
    """
    try:
        model = get_model()
        # language="en" enforces English processing; beam_size=1 and vad_filter=True minimize latency
        segments, info = model.transcribe(
            file_path,
            language="en",
            beam_size=1,
            vad_filter=True
        )
        transcribed_text = " ".join([segment.text for segment in segments]).strip()
        return transcribed_text
    except Exception as e:
        print(f"Transcription Error: {e}")
        return None
    finally:
        if auto_cleanup and os.path.exists(file_path):
            os.remove(file_path)

