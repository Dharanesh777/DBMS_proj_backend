import os
import queue
import threading
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

    def callback(self, indata, frames, time, status):
        if status:
            print(f"[MIC STATUS] {status}", flush=True)
        self.q.put(indata.copy())

    def start(self, filename="session_recording.wav", samplerate=16000):
        if self.recording:
            return
        self.filename = filename
        self.recording = True
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._record_loop, args=(samplerate,), daemon=True)
        self.thread.start()

    def _record_loop(self, samplerate):
        try:
            with sd.InputStream(samplerate=samplerate, channels=1, callback=self.callback):
                with sf.SoundFile(self.filename, mode='w', samplerate=samplerate, channels=1) as f:
                    while self.recording:
                        try:
                            data = self.q.get(timeout=0.1)
                            f.write(data)
                        except queue.Empty:
                            continue
        except Exception as e:
            print(f"❌ Microphone recording error: {e}")
            self.recording = False

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

def start_session_recording(filename="session_recording.wav"):
    global IS_RECORDING
    IS_RECORDING = True
    _recorder.start(filename=filename)

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
                                try:
                                    print(f"🚀 Pushing Reminder to Calendar: {title}")
                                    requests.post(
                                        "http://localhost:8004/create-reminder",
                                        json=event,
                                        timeout=5,
                                    )
                                except Exception as sync_err:
                                    print(f"❌ Calendar sync failed: {sync_err}")
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
