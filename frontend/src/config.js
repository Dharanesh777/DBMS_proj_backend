// Shared API base config — was previously hardcoded as http://localhost:8004
// independently in App.jsx and CameraOverlay.jsx. Override with VITE_API_BASE
// in a .env file for non-local deployments.
export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8004';
