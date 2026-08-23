import os
import queue
import threading
import time
import sounddevice as sd
import soundfile as sf
import requests

# Global recorder instance
class BackgroundRecorder:
    def __init__(self):
        self.q = queue.Queue()
        self.recording = False
        self.thread = None
        self.filename = None
        self.last_error: Exception | None = None
        self._stream_ready = threading.Event()

    def callback(self, indata, frames, time, status):
        if status:
            print(f"[MIC STATUS] {status}", flush=True)
        self.q.put(indata.copy())

    def start(self, filename="session_recording.wav", samplerate=None, wait_for_ready: float = 1.0) -> bool:
        """Start recording in a background thread.

        Returns True if the mic stream is confirmed open within `wait_for_ready`
        seconds, False if it's still starting or failed to open — check
        self.last_error for the failure reason. Callers should not assume the
        mic is actually recording just because this method returned.
        """
        if self.recording:
            return True
        self.filename = filename
        self.recording = True
        self.last_error = None
        self.q = queue.Queue()
        self._stream_ready.clear()
        self.thread = threading.Thread(target=self._record_loop, args=(samplerate,), daemon=True)
        self.thread.start()
        ready = self._stream_ready.wait(timeout=wait_for_ready)
        return ready and self.last_error is None

    def _open_input_stream(self, samplerate, attempts=3, retry_delay=0.5):
        """Open the mic, retrying briefly — macOS CoreAudio can still be tearing
        down the previous session's stream when a new one starts right away,
        which surfaces as a transient PortAudio -9986 error."""
        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                return sd.InputStream(samplerate=samplerate, channels=1, callback=self.callback)
            except Exception as e:
                last_err = e
                if attempt < attempts:
                    print(f"[MIC] Retry {attempt}/{attempts} opening input stream after error: {e}", flush=True)
                    time.sleep(retry_delay)
        raise last_err

    def _record_loop(self, samplerate):
        try:
            # Whisper resamples internally regardless of input rate, so use the
            # device's native rate instead of forcing one CoreAudio may reject.
            if samplerate is None:
                samplerate = int(sd.query_devices(kind="input")["default_samplerate"])
            with self._open_input_stream(samplerate) as stream:
                with sf.SoundFile(self.filename, mode='w', samplerate=samplerate, channels=1) as f:
                    self._stream_ready.set()
                    while self.recording:
                        try:
                            data = self.q.get(timeout=0.1)
                            f.write(data)
                        except queue.Empty:
                            continue
        except Exception as e:
            print(f"❌ Microphone recording error: {e}")
            self.last_error = e
            self.recording = False
            self._stream_ready.set()

    def stop(self):
        if self.recording:
            self.recording = False
            if self.thread:
                self.thread.join(timeout=2.0)
            return self.filename
        return None

_recorder = BackgroundRecorder()
IS_RECORDING = False
IS_SUMMARIZING = False

def start_session_recording(filename="session_recording.wav") -> bool:
    """Returns True if the mic stream is confirmed recording, False if it failed
    to start — check recording_error() for why. IS_RECORDING now reflects actual
    stream state instead of being set unconditionally."""
    global IS_RECORDING
    started = _recorder.start(filename=filename)
    IS_RECORDING = started
    return started

def recording_error() -> str | None:
    """Message from the last recording failure, if any."""
    return str(_recorder.last_error) if _recorder.last_error else None

def stop_session_recording():
    global IS_RECORDING
    IS_RECORDING = False
    return _recorder.stop()

def process_recording_in_background(interaction_id: int, filepath: str):
    """Run Whisper transcription + GPT-4 summary in a separate thread and update DB."""
    global IS_SUMMARIZING

    def _process():
        global IS_SUMMARIZING
        IS_SUMMARIZING = True
        try:
            if not os.path.exists(filepath):
                print(f"❌ Audio file {filepath} not found for processing.")
                return

            # Check file size — if empty or extremely small, skip
            if os.path.getsize(filepath) < 1000:
                print("ℹ️ Audio file is too small (silence or error), skipping.")
                return

            print(f"📤 Transcribing audio from {filepath}...")
            try:
                from app.services.voice_app.transcription_service import transcribe_audio
            except ImportError:
                print("❌ Could not import transcription_service")
                return

            text = transcribe_audio(filepath)

            if text and text.strip():
                print(f"✅ Transcribed: {text}")

                try:
                    from app.services.conversation_summarizer import analyze_conversation
                    import json
                    analysis = analyze_conversation(text)
                    print("\n💡 COMBINED JSON RESULT:")
                    print(json.dumps(analysis, indent=2))

                    summary = analysis.get("summary", "Summarization complete.")
                    emotion = analysis.get("emotion", "Neutral")

                    if interaction_id:
                        try:
                            from app.database.db import update_conversation_results
                            update_conversation_results(interaction_id, text, summary, emotion)
                            print(f"✅ Database updated for Interaction {interaction_id}")
                        except Exception as db_err:
                            print(f"❌ Could not update DB: {db_err}")

                    # Calendar sync
                    events_list = analysis.get("events", [])
                    if isinstance(events_list, list) and len(events_list) > 0:
                        for event in events_list:
                            if not isinstance(event, dict):
                                continue
                            title = event.get("title")
                            date  = event.get("date")
                            time_str = event.get("time")
                            if title and date and time_str:
                                print(f"🚀 Pushing Reminder to Calendar: {title}")
                                _post_reminder_with_retry(event)
                    else:
                        print("ℹ️ No calendar events to sync.")

                except Exception as e:
                    print(f"❌ Summarization/Sync failed: {e}")
            else:
                print("ℹ️ Transcription was empty (likely silence).")
        finally:
            IS_SUMMARIZING = False
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"⚠️ Could not delete temp file: {e}")

    t = threading.Thread(target=_process, daemon=True)
    t.start()


def _post_reminder_with_retry(event: dict, attempts: int = 3, base_delay: float = 1.0) -> None:
    """POST a calendar reminder event, retrying with linear backoff on failure."""
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            requests.post(
                "http://localhost:8004/create-reminder",
                json=event,
                timeout=5,
            )
            return
        except Exception as e:
            last_err = e
            if attempt < attempts:
                time.sleep(base_delay * attempt)
    print(f"❌ Calendar sync failed after {attempts} attempts: {last_err}")
