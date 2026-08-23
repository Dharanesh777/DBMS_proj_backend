import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('Unhandled error in app render tree:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          position: 'fixed', inset: 0,
          background: '#0d1117', color: '#e6edf3',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          fontFamily: 'monospace', gap: 12, padding: 20, textAlign: 'center',
        }}>
          <div style={{ fontSize: 18, fontWeight: 700 }}>⚠️ Something went wrong</div>
          <div style={{ color: '#8b949e', fontSize: 13, maxWidth: 480 }}>
            The dashboard hit an unexpected error and stopped rendering. Reloading the page usually fixes this.
          </div>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: 8, padding: '8px 18px', borderRadius: 8, cursor: 'pointer',
              background: '#238636', color: '#fff', border: 'none', fontWeight: 600, fontSize: 14,
            }}
          >
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
