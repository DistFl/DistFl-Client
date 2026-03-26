import { useState } from 'react';
import { useStatus, useLogs } from '../hooks';
import { resetState, getLogs } from '../api/client';
import { useAppToast } from '../layouts/AppShell';
import PageHeader, { SectionCard, StatusBadge } from '../components/PageHeader';
import { Alert } from '../components/Toast';
import {
  Server, Sliders, Radio, Hash, User, RotateCcw, Trash2,
  Shield, Info, Zap
} from 'lucide-react';

export default function Settings() {
  const toast = useAppToast();
  const status = useStatus(5000);
  const [logLevel, setLogLevel] = useState('INFO');
  const [refreshRate, setRefreshRate] = useState('2000');

  const handleReset = async () => {
    if (!confirm('Reset all state? This will disconnect from any active room.')) return;
    try {
      await resetState();
      toast?.('State reset', 'success');
    } catch (err) {
      toast?.(err.message, 'error');
    }
  };

  const handleClearLogs = () => {
    toast?.('Logs cleared (client-side only)', 'info');
  };

  return (
    <div>
      <PageHeader
        title="Settings"
        description="Configure the Client SDK and UI preferences"
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 max-w-4xl">
        {/* SDK Configuration */}
        <SectionCard title="SDK Configuration" icon={Sliders}>
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Default Server URL</label>
              <input
                className="input-base"
                value={status?.server_url || 'ws://localhost:8080'}
                readOnly
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Log Level</label>
              <select
                className="input-base"
                value={logLevel}
                onChange={e => setLogLevel(e.target.value)}
              >
                <option value="DEBUG">DEBUG</option>
                <option value="INFO">INFO</option>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                Refresh Rate <span className="text-slate-600">(ms)</span>
              </label>
              <input
                className="input-base"
                type="number"
                value={refreshRate}
                onChange={e => setRefreshRate(e.target.value)}
                min="500"
                step="500"
              />
            </div>
          </div>
        </SectionCard>

        {/* Session State */}
        <SectionCard title="Session State" icon={Radio}>
          <div className="space-y-3">
            <div className="flex items-center justify-between py-1.5">
              <span className="text-xs text-slate-500 uppercase tracking-wider">Status</span>
              <StatusBadge status={status?.status || 'idle'} />
            </div>
            <div className="h-px bg-border" />
            <div className="flex items-center justify-between py-1.5">
              <span className="text-xs text-slate-500 uppercase tracking-wider">Client ID</span>
              <span className="text-xs font-mono text-slate-300">
                {status?.client_id || '—'}
              </span>
            </div>
            <div className="h-px bg-border" />
            <div className="flex items-center justify-between py-1.5">
              <span className="text-xs text-slate-500 uppercase tracking-wider">Room ID</span>
              <span className="text-xs font-mono text-slate-300">
                {status?.room_id ? status.room_id.slice(0, 16) + '…' : '—'}
              </span>
            </div>
          </div>
        </SectionCard>

        {/* Destructive Actions */}
        <SectionCard title="Danger Zone" icon={RotateCcw} className="border-red-500/10">
          <p className="text-xs text-slate-500 mb-4">
            These actions will affect your current session and cannot be undone.
          </p>
          <div className="flex items-center gap-3">
            <button
              onClick={handleReset}
              className="flex items-center gap-2 h-9 px-4 rounded-lg bg-red-500/10 text-red-400 text-xs font-medium border border-red-500/20 hover:bg-red-500/20 transition-colors"
            >
              <RotateCcw size={14} /> Reset All State
            </button>
            <button
              onClick={handleClearLogs}
              className="flex items-center gap-2 h-9 px-4 rounded-lg border border-border text-xs text-slate-400 hover:text-slate-200 hover:border-border-hover transition-colors"
            >
              <Trash2 size={14} /> Clear Logs
            </button>
          </div>
        </SectionCard>

        {/* About */}
        <SectionCard title="About" icon={Info}>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Zap size={16} className="text-accent" />
              <span className="text-sm font-semibold text-slate-200">DistFL Client SDK</span>
              <span className="text-xs text-slate-500">v1.0</span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              Production-grade Python Client SDK for room-based Federated Learning.
              Built with PyTorch, FastAPI, and WebSockets.
            </p>
          </div>

          <div className="mt-4 p-3 rounded-lg bg-surface border border-border">
            <div className="flex items-start gap-2">
              <Shield size={14} className="text-emerald-400 mt-0.5 shrink-0" />
              <div className="text-xs text-slate-400 leading-relaxed">
                <strong className="text-slate-300">Privacy First</strong> —
                UI runs locally inside the SDK. All training logic stays on your machine.
                The server only receives aggregated model updates — your data never leaves your device.
              </div>
            </div>
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
