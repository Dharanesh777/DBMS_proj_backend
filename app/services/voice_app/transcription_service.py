import numpy as np
import whisper
from whisper.audio import load_audio, SAMPLE_RATE
import os

# Create a place to store temporary audio files
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Load the whisper model ('small' is significantly better for Tamil)
model = whisper.load_model("small")

# Heuristic VAD gate — Whisper is known to hallucinate plausible-looking text
# from near-silent/noise-only audio rather than returning an empty string, so
# a post-hoc "is the output empty" check doesn't catch it. Reject clearly
# too-short or too-quiet audio BEFORE paying for a transcription pass at all;
# this also cuts the number of (eventually cloud-billed) STT calls made from
# high-frequency idle-audio polling. Thresholds are loose defaults — tune
# against real mic/room noise floor if false rejects/accepts show up.
MIN_SPEECH_DURATION_SECONDS = 0.3
SILENCE_RMS_THRESHOLD = 0.01


def has_speech(file_path: str) -> bool:
    """Cheap energy-based gate: decode to PCM (via ffmpeg, the same path
    Whisper itself uses internally) and reject audio too short or too quiet
    to plausibly contain speech. Fails open (returns True) if decoding itself
    fails, so a format Whisper could still handle isn't silently dropped."""
    try:
        audio = load_audio(file_path)
    except Exception as e:
        print(f"[VAD] Could not decode audio for speech-presence check: {e}")
        return True
    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_SPEECH_DURATION_SECONDS:
        return False
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return rms >= SILENCE_RMS_THRESHOLD


def transcribe_audio(file_path):
    """
    Takes a path to an audio file and returns the transcribed text, or None
    if transcription failed.
    """
    try:
        # Auto-detect language (works for both English and Tamil)
        result = model.transcribe(file_path)
        return result["text"].strip()
    except Exception as e:
        print(f"Transcription Error: {e}")
        return None
    finally:
        # Clean up the file after transcription if desired
        if os.path.exists(file_path):
            os.remove(file_path)
