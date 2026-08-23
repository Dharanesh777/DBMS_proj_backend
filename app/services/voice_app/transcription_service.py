import os
from faster_whisper import WhisperModel

# Create a place to store temporary audio files
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Raspberry Pi Optimized: English-only 'tiny.en' (or 'base.en') with int8 quantization
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
_model_instance: WhisperModel | None = None


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


if __name__ == "__main__":
    import sys
    import time

    # Ensure project root is in sys.path when script is executed directly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from app.services.voice_app.recorder_util import start_session_recording, stop_session_recording, recording_error


    print("=== Audio Transcription Service ===", flush=True)

    # Check if a custom audio file path was passed as argument
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        target_audio = sys.argv[1]
        print(f"Processing audio file: {target_audio}", flush=True)
    else:
        target_audio = os.path.join(UPLOAD_DIR, "live_test.wav")
        print("\nRecording 5 seconds from microphone. Speak now...", flush=True)
        started = start_session_recording(target_audio)
        if not started:
            print(f"Microphone error: {recording_error()}", flush=True)
            sys.exit(1)
        
        for count in range(5, 0, -1):
            print(f"Recording... {count}s", flush=True)
            time.sleep(1)
            
        stop_session_recording()
        print("Recording stopped.", flush=True)

    print("\n⚡ Transcribing audio with faster-whisper...", flush=True)
    start_time = time.time()
    result = transcribe_audio(target_audio, auto_cleanup=False)
    elapsed = time.time() - start_time

    print("\n" + "=" * 50, flush=True)
    print(f"🗣️ Transcribed Output ({elapsed:.2f}s):\n{result if result else '(No speech detected)'}", flush=True)
    print("=" * 50, flush=True)


