import React, { useRef, useEffect, useCallback, useState } from 'react';
import Webcam from 'react-webcam';
import { API_BASE } from '../config';

const FACE_API_URL    = `${API_BASE}/identify`;
const IDLE_AUDIO_URL  = `${API_BASE}/idle-audio`;
const SCAN_INTERVAL   = 1500;   // Poll interval while idle (ms)
const SESSION_POLL_MS = 500;    // Fast poll interval during active session (ms)
const IDLE_AUDIO_CHUNK_MS = 10000; // length of each idle-state audio chunk sent to the server
const MAX_CONSECUTIVE_FAILURES_LOGGED = 5; // stop spamming the console after this many
const IDLE_AUDIO_MIME_CANDIDATES = ['audio/webm', 'audio/mp4', 'audio/ogg'];
const IDLE_AUDIO_RETRY_DELAY_MS = 1000; // backoff before retrying a failed recorder.start()/construction

// audio/webm (the old hardcoded value) isn't supported on Safari/iOS — MediaRecorder
// construction throws every time there, which used to permanently stop idle-audio
// capture for the rest of the idle period with no retry. Pick whatever the browser
// actually supports, falling back to the browser's own default (undefined mimeType)
// rather than giving up.
function pickSupportedMimeType() {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
    return undefined;
  }
  return IDLE_AUDIO_MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type));
}

const CameraOverlay = ({ onSessionEvent, sysStatus }) => {
  const webcamRef    = useRef(null);
  const canvasRef    = useRef(null);   // reused across captureBlob() calls instead of allocating a new one every poll
  const intervalRef  = useRef(null);
  const abortRef     = useRef(null);
  const failureCountRef = useRef(0);

  // Actual reactive session state — sessionRef alone (read but never causing a
  // re-render) meant the polling interval below never restarted on transitions
  // between idle and active, since its effect only depended on `poll`'s identity.
  const [sessionState, setSessionState] = useState('idle');

  const [overlay, setOverlay] = useState({
    label: '🔍 Scanning...',
    color: '#00e5ff',
    showGrace: false,
    countdown: 0,
  });

  const [cameraError, setCameraError] = useState(null);

  const captureBlob = useCallback(() => {
    const webcam = webcamRef.current;
    if (!webcam || !webcam.video || webcam.video.readyState !== 4) return null;
    const video = webcam.video;
    if (!canvasRef.current) canvasRef.current = document.createElement('canvas');
    const canvas = canvasRef.current;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.85));
  }, []);

  const poll = useCallback(async () => {
    const blob = await captureBlob();
    if (!blob) return;

    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res  = await fetch(FACE_API_URL, { method: 'POST', body: formData, signal: controller.signal });
      if (!res.ok) return;
      const data = await res.json();
      failureCountRef.current = 0;

      const state = data.session_state ?? 'idle';
      setSessionState(state);

      // ── Grace period ────────────────────────────────────────────────
      if (state === 'grace_period') {
        setOverlay({
          label: `⚠️ Look at camera... ${data.grace_countdown}s`,
          color: '#ff9800',
          showGrace: true,
          countdown: data.grace_countdown,
        });
        onSessionEvent({ type: 'grace', countdown: data.grace_countdown });
        return;
      }

      // ── Session ended ───────────────────────────────────────────────
      if (state === 'ended') {
        setOverlay({ label: '⚙️ Processing...', color: '#f0c040', showGrace: false, countdown: 0 });
        onSessionEvent({
          type: 'session_ended',
          needsRegistration: data.needs_registration,
        });
        return;
      }

      // ── Active session ──────────────────────────────────────────────
      if (state === 'session_active' || state === 'session_started') {
        const name       = data.person_name || 'Unknown';
        const isRecording  = data.is_recording;
        const isSummarizing = data.is_summarizing;

        let label = `✅ ${name}`;
        let color = '#4caf50';
        if (isRecording)   { label = `🎤 Recording — ${name}`; color = '#ff4444'; }
        if (isSummarizing) { label = `⚙️ Summarizing — ${name}`; color = '#f0c040'; }

        setOverlay({ label, color, showGrace: false, countdown: 0 });

        if (state === 'session_started') {
          onSessionEvent({
            type: data.match_status === 'unknown' ? 'unknown_session' : 'known_session',
            name: data.person_name,
            relationship: data.relationship,
            confidence: data.confidence,
            lastVisit: data.last_visit,
            lastSummary: data.last_summary,
            lastEmotion: data.last_emotion,
            lastConversation: data.last_conversation,
          });
        }
        return;
      }

      // ── Idle ────────────────────────────────────────────────────────
      if (state === 'idle' || !state) {
        if (data.person_detected && data.match_status !== 'no_face') {
          // Session just kicked off
          setOverlay({ label: '🔎 Identifying...', color: '#00e5ff', showGrace: false, countdown: 0 });
        } else {
          setOverlay({ label: '🔍 Scanning...', color: '#00e5ff', showGrace: false, countdown: 0 });
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      failureCountRef.current += 1;
      if (failureCountRef.current <= MAX_CONSECUTIVE_FAILURES_LOGGED) {
        console.warn(`Camera identify poll failed (${failureCountRef.current}):`, err.message);
      }
    }
  }, [captureBlob, onSessionEvent]);

  // Restart the polling interval whenever the session actually transitions between
  // idle and active — previously this only re-ran when poll's identity changed,
  // reading session state from a ref instead of reactive state, so the interval
  // could lag a full cycle behind the real transition.
  useEffect(() => {
    const interval = sessionState === 'idle' ? SCAN_INTERVAL : SESSION_POLL_MS;
    clearInterval(intervalRef.current);
    intervalRef.current = setInterval(poll, interval);
    return () => clearInterval(intervalRef.current);
  }, [poll, sessionState]);

  // Stop the webcam's media stream tracks explicitly on unmount — react-webcam's
  // own cleanup isn't guaranteed to release the camera immediately otherwise.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      const stream = webcamRef.current?.video?.srcObject;
      stream?.getTracks?.().forEach((track) => track.stop());
    };
  }, []);

  // ── Idle-state client-mic audio capture ──────────────────────────────────
  // Replaces the old server-side mic recording (which captured the HOST
  // machine's own mic, not the caller's). Records fixed-length chunks from
  // the CLIENT's microphone only while sessionState is 'idle', and posts each
  // one to /idle-audio for transcription. Stops immediately once a session
  // starts, via this effect's cleanup running when sessionState changes.
  useEffect(() => {
    if (sessionState !== 'idle') return undefined;
    if (typeof MediaRecorder === 'undefined') return undefined;

    let cancelled = false;
    let stream = null;
    let currentRecorder = null;
    let nextChunkTimer = null;

    // Wraps recordAndSendChunk so every scheduling call site (initial kickoff,
    // onstop's self-re-invocation, and the retry-after-failure paths below) is
    // guaranteed to have its rejection handled — recordAndSendChunk is async
    // and was previously invoked bare from setTimeout, so any throw inside it
    // (e.g. recorder.start() below, before it had a try/catch) became an
    // unhandled promise rejection instead of a recoverable, logged failure.
    const safeRecordAndSendChunk = () => {
      recordAndSendChunk().catch((err) => {
        console.error('Idle audio: unexpected error in capture loop:', err);
      });
    };

    const recordAndSendChunk = async () => {
      if (cancelled) return;

      // Re-validate a cached stream before reusing it — if its track ended
      // (device unplugged/reclaimed, permission revoked mid-session) a stale
      // stream still constructs a MediaRecorder "successfully" but produces
      // zero dataavailable events, so chunks stay empty and nothing ever
      // uploads again for the rest of the idle period with no warning at all.
      if (stream && stream.getAudioTracks().every((t) => t.readyState === 'ended')) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
      }

      if (!stream) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (err) {
          console.warn('Idle audio: microphone access failed, giving up:', err.message);
          return;
        }
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
      }

      const chunks = [];
      let recorder;
      const mimeType = pickSupportedMimeType();
      try {
        recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      } catch (err) {
        console.warn('Idle audio: MediaRecorder unavailable, giving up for this idle period:', err.message);
        return;
      }
      currentRecorder = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };
      recorder.onstop = () => {
        if (chunks.length > 0) {
          const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
          const formData = new FormData();
          formData.append('file', blob, 'idle_chunk.webm');
          fetch(IDLE_AUDIO_URL, { method: 'POST', body: formData }).catch((err) => {
            console.warn('Idle audio upload failed:', err.message);
          });
        }
        if (!cancelled) {
          nextChunkTimer = setTimeout(safeRecordAndSendChunk, 0);
        }
      };

      try {
        recorder.start();
      } catch (err) {
        // Stream/device went bad between construction and start() (e.g. a
        // mid-session unplug/reclaim). Previously this threw with nothing
        // ever scheduling a retry, silently killing capture for the rest of
        // the idle period. Drop the stream so the next attempt re-acquires
        // getUserMedia instead of reusing a dead one, and retry after a
        // short backoff instead of a tight failure loop.
        console.warn('Idle audio: recorder.start() failed, retrying shortly:', err.message);
        stream?.getTracks().forEach((t) => t.stop());
        stream = null;
        if (!cancelled) {
          nextChunkTimer = setTimeout(safeRecordAndSendChunk, IDLE_AUDIO_RETRY_DELAY_MS);
        }
        return;
      }

      nextChunkTimer = setTimeout(() => {
        if (!cancelled && recorder.state !== 'inactive') recorder.stop();
      }, IDLE_AUDIO_CHUNK_MS);
    };

    safeRecordAndSendChunk();

    return () => {
      cancelled = true;
      clearTimeout(nextChunkTimer);
      if (currentRecorder && currentRecorder.state !== 'inactive') {
        currentRecorder.stop();
      }
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [sessionState]);

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', overflow: 'hidden', background: 'black' }}>
      <Webcam
        ref={webcamRef}
        muted={true}
        style={{ width: '100vw', height: '100vh', objectFit: 'cover' }}
        screenshotFormat="image/jpeg"
        onUserMediaError={(err) => {
          console.error('Webcam access failed:', err);
          setCameraError(
            err?.name === 'NotAllowedError'
              ? 'Camera permission denied — allow camera access and reload.'
              : 'Camera unavailable — check that it is connected and not in use elsewhere.'
          );
        }}
      />
      {cameraError && (
        <div style={{
          position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
          background: 'rgba(0,0,0,0.85)', color: '#ff6b6b', padding: '16px 24px',
          borderRadius: 8, fontFamily: 'monospace', fontSize: 14, textAlign: 'center', maxWidth: 360,
        }}>
          ⚠️ {cameraError}
        </div>
      )}

      {/* Scanning reticle overlay */}
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: 180,
        height: 180,
        border: `2px solid ${overlay.color}`,
        borderRadius: 4,
        boxShadow: `0 0 20px ${overlay.color}55`,
        pointerEvents: 'none',
      }}>
        {/* Corner marks */}
        {['top-left', 'top-right', 'bottom-left', 'bottom-right'].map((corner) => {
          const [v, h] = corner.split('-');
          return (
            <div key={corner} style={{
              position: 'absolute',
              [v]: -2, [h]: -2,
              width: 18, height: 18,
              borderTop: v === 'top' ? `3px solid ${overlay.color}` : 'none',
              borderBottom: v === 'bottom' ? `3px solid ${overlay.color}` : 'none',
              borderLeft: h === 'left' ? `3px solid ${overlay.color}` : 'none',
              borderRight: h === 'right' ? `3px solid ${overlay.color}` : 'none',
            }} />
          );
        })}
      </div>

      {/* Status label */}
      <div style={{
        position: 'absolute',
        bottom: 'calc(50% - 120px)',
        left: '50%',
        transform: 'translateX(-50%)',
        background: 'rgba(0,0,0,0.6)',
        color: overlay.color,
        padding: '4px 12px',
        borderRadius: 4,
        fontSize: 13,
        fontFamily: 'monospace',
        letterSpacing: 1,
        pointerEvents: 'none',
        whiteSpace: 'nowrap',
      }}>
        {overlay.label}
      </div>

      {/* Grace period full-screen flash */}
      {overlay.showGrace && (
        <div style={{
          position: 'absolute',
          inset: 0,
          border: `4px solid #ff9800`,
          pointerEvents: 'none',
          animation: 'pulse-border 1s infinite',
        }} />
      )}
    </div>
  );
};

export default CameraOverlay;
