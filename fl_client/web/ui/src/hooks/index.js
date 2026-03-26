import { useState, useEffect, useRef, useCallback } from 'react';
import { getStatus, getMetrics, getLogs } from '../api/client';

/* ── useStatus ────────────────────────────────────────────────────────────── */
export function useStatus(interval = 3000) {
  const [status, setStatus] = useState(null);
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const data = await getStatus();
        if (active) setStatus(data);
      } catch {}
    };
    poll();
    const id = setInterval(poll, interval);
    return () => { active = false; clearInterval(id); };
  }, [interval]);
  return status;
}

/* ── useMetrics ───────────────────────────────────────────────────────────── */
export function useMetrics(interval = 2000) {
  const [metrics, setMetrics] = useState({ metrics: [], total_rounds: 0, latest_loss: null, latest_round: 0 });
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const data = await getMetrics();
        if (active) setMetrics(data);
      } catch {}
    };
    poll();
    const id = setInterval(poll, interval);
    return () => { active = false; clearInterval(id); };
  }, [interval]);
  return metrics;
}

/* ── useLogs ──────────────────────────────────────────────────────────────── */
export function useLogs(interval = 2000) {
  const [logs, setLogs] = useState([]);
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const data = await getLogs();
        if (active) setLogs(data.logs || []);
      } catch {}
    };
    poll();
    const id = setInterval(poll, interval);
    return () => { active = false; clearInterval(id); };
  }, [interval]);
  return logs;
}

/* ── useWebSocket ─────────────────────────────────────────────────────────── */
export function useWebSocket(callbacks = {}) {
  const wsRef = useRef(null);
  const cbRef = useRef(callbacks);
  cbRef.current = callbacks;

  useEffect(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws/live`;
    let reconnectTimer;

    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'log' && cbRef.current.onLog) cbRef.current.onLog(msg.data);
          if (msg.type === 'metrics' && cbRef.current.onMetrics) cbRef.current.onMetrics(msg.data);
          if (msg.type === 'status' && cbRef.current.onStatus) cbRef.current.onStatus(msg.data);
        } catch {}
      };
      ws.onclose = () => { reconnectTimer = setTimeout(connect, 3000); };
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);
}

/* ── useToast ─────────────────────────────────────────────────────────────── */
export function useToast() {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((message, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  return { toasts, addToast, removeToast };
}
