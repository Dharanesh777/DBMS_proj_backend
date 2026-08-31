import os
import sys
import time
from app.services.voice_app.transcription_service import (
    transcribe_audio,
    set_model,
    get_active_model_info,
    AVAILABLE_MODELS,
    UPLOAD_DIR,
)
from app.services.voice_app.recorder_util import (
    start_session_recording,
    stop_session_recording,
    recording_error,
)


def main():
    print("=== Audio Transcription Service (whisper.cpp) Module Test ===", flush=True)

    # Parse arguments
    model_arg = None
    audio_path = None
    is_record_mode = False

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--model" and i + 1 < len(sys.argv):
            model_arg = sys.argv[i + 1]
            i += 2
        elif arg == "--record":
            is_record_mode = True
            i += 1
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

    # Set up selected model using EXISTING set_model implementation
    try:
        set_model(selected_model)
        active_info = get_active_model_info()
    except FileNotFoundError as fnf:
        print(f"\n[ERROR] Missing Model File:\n{fnf}", flush=True)
        print("\nPlease ensure the required model file is placed in the 'models/' directory:")
        print("  - models/ggml-tiny.en-q5_1.bin")
        print("  - models/ggml-base.en-q5_1.bin")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to load model '{selected_model}': {e}", flush=True)
        sys.exit(1)

    # --- CLI RECORD MODE ---
    if is_record_mode:
        temp_recording_path = os.path.join(UPLOAD_DIR, "cli_live_recording.wav")
        if os.path.exists(temp_recording_path):
            try:
                os.remove(temp_recording_path)
            except OSError:
                pass

        print(f"\nStarting microphone recording...", flush=True)
        started = start_session_recording(temp_recording_path)
        if not started:
            err_msg = recording_error() or "Microphone device not found or access denied."
            print(f"\n[ERROR] Microphone Failure: {err_msg}", flush=True)
            sys.exit(1)

        print("\nRecording... Press Enter to stop.", flush=True)
        try:
            input()  # Wait for user to hit Enter
        except KeyboardInterrupt:
            print("\nRecording cancelled.", flush=True)

        print("Stopping recording...", flush=True)
        stop_session_recording()

        if not os.path.exists(temp_recording_path) or os.path.getsize(temp_recording_path) < 1000:
            print("\n[ERROR] Recording failure: Recorded audio is empty or silence was detected.", flush=True)
            if os.path.exists(temp_recording_path):
                try:
                    os.remove(temp_recording_path)
                except OSError:
                    pass
            sys.exit(1)

        print(f"Transcribing using {active_info['display_name']}...", flush=True)
        start_time = time.time()
        transcript = transcribe_audio(temp_recording_path, model_key=selected_model, auto_cleanup=True)
        elapsed_time = time.time() - start_time

        print("\nTranscript:", flush=True)
        print(transcript if transcript else "(No speech detected)", flush=True)
        print(f"\n[Execution Time: {elapsed_time:.2f}s | Active Model: {active_info['display_name']}]", flush=True)
        return

    # --- WAV FILE TRANSCRIBE MODE ---
    if not audio_path or not os.path.exists(audio_path):
        print("\nNo input audio file specified or file does not exist.", flush=True)
        print("Usage:")
        print("  python test_rpi_transcribe.py --record [--model Tiny|Base]")
        print("  python test_rpi_transcribe.py <path_to_audio.wav> [--model Tiny|Base]")
        print("  python test_rpi_transcribe.py --test-switch")
        return

    print(f"\n[Transcribing] Audio file: {audio_path}", flush=True)
    start_time = time.time()
    transcription = transcribe_audio(audio_path, model_key=selected_model)
    elapsed_time = time.time() - start_time

    print("\n---------------- RESULTS ----------------", flush=True)
    print(f"Active Model     : {get_active_model_info()['display_name']}", flush=True)
    print(f"Transcribed Text : {transcription if transcription else '(No speech detected)'}", flush=True)
    print(f"Execution Time   : {elapsed_time:.2f} seconds", flush=True)
    print("-----------------------------------------", flush=True)


if __name__ == "__main__":
    main()
