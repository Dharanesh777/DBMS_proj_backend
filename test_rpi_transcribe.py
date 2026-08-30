import os
import sys
import time
from app.services.voice_app.transcription_service import (
    transcribe_audio,
    set_model,
    get_active_model_info,
    AVAILABLE_MODELS,
)


def main():
    print("=== Audio Transcription Service (whisper.cpp) Module Test ===", flush=True)

    # Parse arguments
    model_arg = None
    audio_path = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--model" and i + 1 < len(sys.argv):
            model_arg = sys.argv[i + 1]
            i += 2
        elif arg == "--test-switch":
            print("\n[Test] Running Model Switch Test...", flush=True)
            for key, info in AVAILABLE_MODELS.items():
                print(f"\n---> Testing model selection: {info['name']} ({key})", flush=True)
                try:
                    set_model(key)
                    active = get_active_model_info()
                    print(f"     Status: Active model is '{active['display_name']}'")
                except FileNotFoundError as fnf:
                    print(f"     [INFO] Expected file error for missing model:\n     {fnf}")
                except Exception as e:
                    print(f"     [ERROR] Unexpected error: {e}")
            return
        elif not arg.startswith("--") and audio_path is None:
            audio_path = arg
            i += 1
        else:
            i += 1

    selected_model = model_arg or "tiny.en-q5_1"

    print(f"\n[Target Model] {selected_model}", flush=True)
    try:
        start_init = time.time()
        set_model(selected_model)
        init_time = time.time() - start_init
        active_info = get_active_model_info()
        print(f"[OK] Loaded model '{active_info['display_name']}' in {init_time:.2f}s", flush=True)
    except FileNotFoundError as fnf:
        print(f"\n[ERROR] Model File Missing Error:\n{fnf}", flush=True)
        print("\nPlease download the required model binary files and place them in 'models/':")
        print("  - models/ggml-tiny.en-q5_1.bin")
        print("  - models/ggml-base.en-q5_1.bin")
        sys.exit(1)

    if not audio_path or not os.path.exists(audio_path):
        print("\nNo input audio file specified or file does not exist.", flush=True)
        print("Usage:")
        print("  python test_rpi_transcribe.py <path_to_audio.wav> [--model Tiny|Base]")
        print("  python test_rpi_transcribe.py --test-switch")
        return

    print(f"\n[Transcribing] Audio file: {audio_path}", flush=True)
    start_time = time.time()
    transcription = transcribe_audio(audio_path, model_key=selected_model)
    elapsed_time = time.time() - start_time

    print("\n---------------- RESULTS ----------------", flush=True)
    print(f"Active Model     : {get_active_model_info()['display_name']}", flush=True)
    print(f"Transcribed Text : {transcription}", flush=True)
    print(f"Execution Time   : {elapsed_time:.2f} seconds", flush=True)
    print("-----------------------------------------", flush=True)


if __name__ == "__main__":
    main()
