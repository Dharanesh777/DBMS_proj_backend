import React, { useRef, useEffect, useCallback, useState } from 'react';
import Webcam from 'react-webcam';

const FACE_API_URL    = 'http://localhost:8004/identify';
const SCAN_INTERVAL   = 1500;   // Poll interval while idle (ms)
const SESSION_POLL_MS = 500;    // Fast poll interval during active session (ms)

const CameraOverlay = ({ onSessionEvent, sysStatus }) => {
  const webcamRef    = useRef(null);
  const intervalRef  = useRef(null);
  const sessionRef   = useRef({ state: 'idle' });

  const [overlay, setOverlay] = useState({
    label: '🔍 Scanning...',
    color: '#00e5ff',
    showGrace: false,
    countdown: 0,
  });

  const captureBlob = useCallback(() => {
    const webcam = webcamRef.current;
    if (!webcam || !webcam.video || webcam.video.readyState !== 4) return null;
    const canvas = document.createElement('canvas');
    const video  = webcam.video;
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

    try {
      const res  = await fetch(FACE_API_URL, { method: 'POST', body: formData });
      if (!res.ok) return;
      const data = await res.json();

      const state = data.session_state ?? 'idle';
      sessionRef.current = { state };

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
      // Silently ignore network errors
    }
  }, [captureBlob, onSessionEvent]);

  // Restart polling interval on state changes
  useEffect(() => {
    const interval = sessionRef.current.state === 'idle' ? SCAN_INTERVAL : SESSION_POLL_MS;
    clearInterval(intervalRef.current);
    intervalRef.current = setInterval(poll, interval);
    return () => clearInterval(intervalRef.current);
  }, [poll]);

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', overflow: 'hidden', background: 'black' }}>
      <Webcam
        ref={webcamRef}
        muted={true}
        style={{ width: '100vw', height: '100vh', objectFit: 'cover' }}
        screenshotFormat="image/jpeg"
      />

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
