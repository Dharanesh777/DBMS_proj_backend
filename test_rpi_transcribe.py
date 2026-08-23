import os
import sys
import time
from app.services.voice_app.transcription_service import transcribe_audio, get_model


def main():
    print("=== Raspberry Pi Audio Transcription Module Test ===", flush=True)
    
    # Check if an audio file was passed as argument
    audio_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not audio_path or not os.path.exists(audio_path):
        print("No input audio file specified or file does not exist.", flush=True)
        print("Usage: python test_rpi_transcribe.py <path_to_audio.wav>", flush=True)
        print("\n[Test] Initializing model to test faster-whisper load performance...", flush=True)
        start_init = time.time()
        get_model()
        init_time = time.time() - start_init
        print(f"[Success] Model initialized in {init_time:.2f} seconds.", flush=True)
        return

    print(f"Transcribing audio file: {audio_path}", flush=True)
    start_time = time.time()
    transcription = transcribe_audio(audio_path)
    elapsed_time = time.time() - start_time
    
    print("\n---------------- RESULTS ----------------", flush=True)
    print(f"Transcribed Text : {transcription}", flush=True)
    print(f"Execution Time   : {elapsed_time:.2f} seconds", flush=True)
    print("-----------------------------------------", flush=True)


if __name__ == "__main__":
    main()

