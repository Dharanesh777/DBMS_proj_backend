import React, { useState, useEffect, useCallback, useRef } from 'react';
import CameraOverlay from './components/CameraOverlay';
import { API_BASE } from './config';
import './App.css';

const REMINDERS_URL   = `${API_BASE}/get-reminders`;
const REGISTER_URL    = `${API_BASE}/register-new`;
const LIVE_LOG_URL    = `${API_BASE}/live-log`;
const SESSION_URL     = `${API_BASE}/session-status`;
const PROVIDER_URL    = `${API_BASE}/config/provider`;

function App() {
  // ── State ──────────────────────────────────────────────────────────────────
  const [reminders, setReminders]   = useState([]);
  const [eventLog, setEventLog]     = useState([{ ts: '--:--:--', message: 'HUD Initialized — Scanning...', _key: 'init' }]);
  const [sessionInfo, setSessionInfo] = useState(null);   // current session data

  const [sysStatus, setSysStatus] = useState({
    state: 'idle',
    person_name: null,
    is_recording: false,
    is_summarizing: false,
    grace_countdown: 0,
    session_duration: 0,
  });

  const [llmProvider, setLlmProvider] = useState(
    () => localStorage.getItem('llmProvider') || 'groq'
  );

  // Tracks whether the last session-status poll succeeded — surfaced in the footer
  // so a dead backend connection is visible instead of silently doing nothing.
  const [connectionOk, setConnectionOk] = useState(true);

  // Registration modal (shown AFTER session ends for unknown persons)
  const [showModal, setShowModal]   = useState(false);
  const [regName, setRegName]       = useState('');
  const [regRelation, setRegRelation] = useState('');
  const [regStatus, setRegStatus]   = useState('');
  const pendingFrameRef = useRef(null);   // last frame blob captured during session

  // Read inside the polling effect without needing showModal in its dependency
  // array — otherwise toggling the modal tears down and recreates all three
  // polling intervals, including the unrelated 30s reminders poll.
  const showModalRef = useRef(showModal);
  useEffect(() => { showModalRef.current = showModal; }, [showModal]);

  // Reconcile the locally-cached LLM provider with actual server state once on
  // mount — a stale localStorage value could otherwise silently diverge from
  // what the backend is really using after a restart.
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(PROVIDER_URL);
        if (res.ok) {
          const data = await res.json();
          if (data.provider && data.provider !== llmProvider) {
            setLlmProvider(data.provider);
            localStorage.setItem('llmProvider', data.provider);
          }
        }
      } catch (_) {
        // Non-fatal — keep the cached value if the backend isn't reachable yet.
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Polling ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    let remindersAbort = null;
    let liveLogAbort = null;
    let sessionAbort = null;

    const fetchReminders = async () => {
      remindersAbort?.abort();
      remindersAbort = new AbortController();
      try {
        const res = await fetch(REMINDERS_URL, { signal: remindersAbort.signal });
        if (res.ok) {
          const data = await res.json();
          if (data.length > 0) setReminders(data.map(r => ({ id: r.id, text: r.summary })));
        }
      } catch (e) {
        if (e.name !== 'AbortError') console.warn('Reminders poll failed:', e.message);
      }
    };

    const fetchLiveLog = async () => {
      liveLogAbort?.abort();
      liveLogAbort = new AbortController();
      try {
        const res = await fetch(LIVE_LOG_URL, { signal: liveLogAbort.signal });
        if (res.ok) {
          const data = await res.json();
          if (data.logs && data.logs.length > 0) {
            const recent = data.logs.slice(-20).reverse();
            setEventLog(recent.map((entry, idx) => ({
              ...entry,
              _key: `${entry.ts}-${idx}-${entry.message.slice(0, 24)}`,
            })));
          }
        }
      } catch (e) {
        if (e.name !== 'AbortError') console.warn('Live log poll failed:', e.message);
      }
    };

    const fetchSession = async () => {
      sessionAbort?.abort();
      sessionAbort = new AbortController();
      try {
        const res = await fetch(SESSION_URL, { signal: sessionAbort.signal });
        if (res.ok) {
          const data = await res.json();
          setSysStatus(data);
          setConnectionOk(true);

          // If session just ended and needs registration, trigger modal
          if (data.state === 'idle' && data.needs_registration && !showModalRef.current) {
            setShowModal(true);
            setRegName('');
            setRegRelation('');
            setRegStatus('');
          }
        } else {
          setConnectionOk(false);
        }
      } catch (e) {
        if (e.name !== 'AbortError') {
          console.warn('Session poll failed:', e.message);
          setConnectionOk(false);
        }
      }
    };

    fetchReminders();
    fetchLiveLog();
    fetchSession();

    const rid  = setInterval(fetchReminders, 30000);
    const lid  = setInterval(fetchLiveLog, 800);
    const sid  = setInterval(fetchSession, 400);
    return () => {
      clearInterval(rid); clearInterval(lid); clearInterval(sid);
      remindersAbort?.abort(); liveLogAbort?.abort(); sessionAbort?.abort();
    };
  }, []);

  // ── Session event callback from CameraOverlay ────────────────────────────
  const handleSessionEvent = useCallback((event) => {
    if (event.type === 'known_session') {
      setSessionInfo(event);
    } else if (event.type === 'unknown_session') {
      setSessionInfo({ name: 'Unknown', relationship: '?', confidence: event.confidence });
    } else if (event.type === 'session_ended') {
      setSessionInfo(null);
      if (event.needsRegistration) {
        setShowModal(true);
        setRegName('');
        setRegRelation('');
        setRegStatus('');
      }
    }
  }, []);

  // ── Registration submit ──────────────────────────────────────────────────
  const handleRegister = async () => {
    if (!regName.trim() || !regRelation.trim()) {
      setRegStatus('⚠️ Please fill both fields.');
      return;
    }

    setRegStatus('⏳ Registering...');
    try {
      const formData = new FormData();
      formData.append('name', regName.trim());
      formData.append('relationship', regRelation.trim());

      const res = await fetch(REGISTER_URL, { method: 'POST', body: formData });

      let data = null;
      try {
        data = await res.json();
      } catch (_) {
        // Server returned a non-JSON body (e.g. an HTML error page) — fall through
        // to the generic status-based message below instead of throwing here.
      }

      if (res.ok && data) {
        setRegStatus(`✅ ${data.message}`);
        setTimeout(() => setShowModal(false), 1500);
      } else if (data) {
        setRegStatus(`❌ ${data.detail || 'Registration failed.'}`);
      } else {
        setRegStatus(`❌ Registration failed (server returned ${res.status}).`);
      }
    } catch (e) {
      setRegStatus(`❌ Network error: ${e.message}`);
    }
  };

  // ── Derived UI values ────────────────────────────────────────────────────
  const stateLabel = () => {
    if (sysStatus.is_recording)   return { text: '🎤 LISTENING', color: '#ff4444' };
    if (sysStatus.is_summarizing) return { text: '⚙️ PROCESSING', color: '#f0c040' };
    if (sysStatus.state === 'grace_period') return { text: `⚠️ LOST FACE — ${sysStatus.grace_countdown}s`, color: '#ff9800' };
    if (sysStatus.state === 'session_active') return { text: `🔗 SESSION — ${sysStatus.person_name || '?'}`, color: '#4caf50' };
    return { text: '🟢 SCANNING', color: '#00e5ff' };
  };

  const { text: stateText, color: stateColor } = stateLabel();

  return (
    <div className="ar-hud-container">
      <CameraOverlay onSessionEvent={handleSessionEvent} sysStatus={sysStatus} />

      {/* ── LEFT PANEL — Event Log ── */}
      <div className="floating-hud left-hud">
        <div className="hud-title">SYSTEM LOG</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 280, overflowY: 'auto' }}>
          {eventLog.map((entry, i) => (
            <div key={entry._key ?? `${entry.ts}-${i}`} style={{ fontSize: 11, color: i === 0 ? '#e6edf3' : '#6e7681', fontFamily: 'monospace', lineHeight: 1.4 }}>
              <span style={{ color: '#3fb950', marginRight: 6 }}>{entry.ts}</span>
              {entry.message}
            </div>
          ))}
        </div>
      </div>

      {/* ── RIGHT PANEL — Session Info + Reminders ── */}
      <div className="floating-hud right-hud">
        <div className="hud-title">SESSION</div>
        {sessionInfo ? (
          <div style={{ marginBottom: 12, fontSize: 12, lineHeight: 1.6 }}>
            <div style={{ color: '#3fb950', fontWeight: 700, fontSize: 14 }}>✅ {sessionInfo.name}</div>
            {sessionInfo.relationship && <div style={{ color: '#8b949e' }}>Relationship: {sessionInfo.relationship}</div>}
            {sessionInfo.confidence   && <div style={{ color: '#8b949e' }}>Confidence: {(sessionInfo.confidence * 100).toFixed(1)}%</div>}
            {sessionInfo.lastVisit    && <div style={{ color: '#8b949e' }}>Last visit: {sessionInfo.lastVisit}</div>}
            {sessionInfo.lastSummary  && (
              <div style={{ color: '#8b949e', marginTop: 4 }}>
                💬 {sessionInfo.lastSummary.substring(0, 80)}{sessionInfo.lastSummary.length > 80 ? '...' : ''}
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: '#3b434d', fontSize: 12, marginBottom: 12 }}>No active session</div>
        )}

        <div className="hud-title" style={{ marginTop: 8 }}>REMINDERS</div>
        {reminders.length === 0
          ? <div style={{ color: '#3b434d', fontSize: 12 }}>No upcoming reminders</div>
          : reminders.slice(0, 4).map(r => (
            <div key={r.id} style={{ fontSize: 11, color: '#8b949e', marginTop: 4 }}>📅 {r.text}</div>
          ))
        }
      </div>

      {/* ── FOOTER ── */}
      <div className="system-footer">
        <span>AG-OS v2.0</span>
        <span style={{ color: stateColor, fontWeight: 600, transition: 'color 0.3s' }}>
          {stateText}
        </span>
        {sysStatus.state === 'session_active' && sysStatus.session_duration > 0 && (
          <span style={{ color: '#8b949e' }}>Session: {sysStatus.session_duration}s</span>
        )}
        {!connectionOk && (
          <span style={{ color: '#ff4444', fontWeight: 600 }}>⚠️ Backend unreachable</span>
        )}
      </div>
{/* ── LLM PROVIDER SWITCHER ── */}
<div style={{
  position: 'absolute', bottom: 20, right: 20,
  background: 'rgba(10,15,25,0.85)',
  backdropFilter: 'blur(10px)',
  border: '1px solid rgba(0,229,255,0.2)',
  borderRadius: 10, padding: '10px 14px',
  color: '#e6edf3', fontFamily: 'monospace',
  display: 'flex', flexDirection: 'column', gap: 6, minWidth: 180,
  boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
}}>
  <div style={{ fontSize: 10, color: '#00e5ff', letterSpacing: 2, marginBottom: 2 }}>LLM PROVIDER</div>
  {['openai', 'groq', 'ollama'].map(p => (
    <button
      key={p}
      onClick={async () => {
        try {
          const res = await fetch(PROVIDER_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider: p })
          });
          if (res.ok) {
            setLlmProvider(p);
            localStorage.setItem('llmProvider', p);
          }
        } catch (_) {}
      }}
      style={{
        background: llmProvider === p ? 'rgba(0,229,255,0.15)' : 'transparent',
        border: llmProvider === p ? '1px solid #00e5ff' : '1px solid rgba(255,255,255,0.1)',
        borderRadius: 6, padding: '5px 10px',
        color: llmProvider === p ? '#00e5ff' : '#6e7681',
        fontFamily: 'monospace', fontSize: 12, cursor: 'pointer',
        textAlign: 'left', textTransform: 'uppercase', letterSpacing: 1,
        transition: 'all 0.2s',
      }}
    >
      {llmProvider === p ? '▶ ' : '  '}{p}
    </button>
  ))}
</div>

      {/* ── REGISTRATION MODAL (shown AFTER session ends) ── */}
      {showModal && (
        <div style={styles.overlay}>
          <div style={styles.modal}>
            <h2 style={styles.title}>🆕 Register New Person</h2>
            <p style={styles.sub}>
              The session with this unrecognized person has ended.<br />
              Enter their details to remember them next time.
            </p>
            <input
              style={styles.input}
              placeholder="Full Name"
              value={regName}
              onChange={e => setRegName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleRegister()}
              autoFocus
            />
            <input
              style={styles.input}
              placeholder="Relationship (e.g. Friend, Family)"
              value={regRelation}
              onChange={e => setRegRelation(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleRegister()}
            />
            {regStatus && <p style={styles.status}>{regStatus}</p>}
            <div style={styles.btnRow}>
              <button style={styles.btnPrimary} onClick={handleRegister}>Register</button>
              <button style={styles.btnSecondary} onClick={() => setShowModal(false)}>Skip</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  overlay: {
    position: 'fixed', inset: 0,
    background: 'rgba(0,0,0,0.8)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    zIndex: 9999,
    backdropFilter: 'blur(6px)',
  },
  modal: {
    background: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: 12,
    padding: '32px 28px',
    width: 380,
    display: 'flex', flexDirection: 'column', gap: 14,
    boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
  },
  title: { margin: 0, color: '#e6edf3', fontSize: 20, fontWeight: 700 },
  sub: { margin: 0, color: '#8b949e', fontSize: 13, lineHeight: 1.5 },
  input: {
    background: '#161b22', border: '1px solid #30363d',
    borderRadius: 8, padding: '10px 14px',
    color: '#e6edf3', fontSize: 14, outline: 'none',
  },
  status: { margin: 0, color: '#f0c040', fontSize: 13 },
  btnRow: { display: 'flex', gap: 10, marginTop: 4 },
  btnPrimary: {
    flex: 1, padding: '10px 0', borderRadius: 8, cursor: 'pointer',
    background: '#238636', color: '#fff', border: 'none', fontWeight: 600, fontSize: 14,
  },
  btnSecondary: {
    flex: 1, padding: '10px 0', borderRadius: 8, cursor: 'pointer',
    background: '#21262d', color: '#8b949e', border: '1px solid #30363d', fontSize: 14,
  },
};

export default App;
