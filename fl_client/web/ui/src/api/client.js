const BASE = '';

export async function api(method, path, body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(`${BASE}${path}`, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

export const getStatus   = () => api('GET', '/api/status');
export const getMetrics  = () => api('GET', '/api/metrics');
export const getLogs     = () => api('GET', '/api/logs');
export const createRoom  = (body) => api('POST', '/api/create-room', body);
export const joinRoom    = (body) => api('POST', '/api/join-room', body);
export const initialize  = (body) => api('POST', '/api/initialize', body);
export const validate    = (body) => api('POST', '/api/validate', body);
export const startTraining = () => api('POST', '/api/start-training');
export const stopTraining  = () => api('POST', '/api/stop-training');
export const resetState    = () => api('POST', '/api/reset');
