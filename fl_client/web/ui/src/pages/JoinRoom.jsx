import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { joinRoom } from '../api/client';
import { useAppToast } from '../layouts/AppShell';
import { useLogs } from '../hooks';
import PageHeader, { SectionCard, StatusBadge } from '../components/PageHeader';
import Stepper from '../components/Stepper';
import LogTerminal from '../components/LogTerminal';
import { Alert } from '../components/Toast';
import {
  Server, Hash, Key, User, FolderOpen, CheckCircle,
  ArrowRight, ArrowLeft, Info, Rocket
} from 'lucide-react';

const STEPS = [
  { label: 'Connect', desc: 'Server & credentials' },
  { label: 'Dataset', desc: 'Select training data' },
  { label: 'Validate', desc: 'Confirm & join' },
];

export default function JoinRoom() {
  const toast = useAppToast();
  const navigate = useNavigate();
  const logs = useLogs(2000);
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const [serverUrl, setServerUrl] = useState('ws://localhost:8080');
  const [roomId, setRoomId] = useState('');
  const [inviteCode, setInviteCode] = useState('');
  const [clientId, setClientId] = useState('');
  const [dataPath, setDataPath] = useState('');

  const handleJoin = async () => {
    try {
      setLoading(true);
      const data = await joinRoom({
        server_url: serverUrl,
        room_id: roomId,
        invite_code: inviteCode,
        data_path: dataPath,
        client_id: clientId,
      });
      setResult(data);
      toast?.('Joined room successfully', 'success');
    } catch (err) {
      toast?.(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Success state
  if (result) {
    const meta = result.metadata || {};
    return (
      <div>
        <PageHeader title="Ready to Train" description="Room joined and dataset validated" />
        <SectionCard className="max-w-lg">
          <div className="flex items-center gap-3 mb-5">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400">
              <CheckCircle size={20} />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-200">Successfully Joined</p>
              <p className="text-xs text-slate-500">Room {roomId.slice(0, 12)}…</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-5">
            <div className="p-3 rounded-lg bg-surface border border-border">
              <p className="text-2xs text-slate-500 uppercase">Samples</p>
              <p className="text-lg font-semibold text-slate-200">{meta.num_samples || 0}</p>
            </div>
            <div className="p-3 rounded-lg bg-surface border border-border">
              <p className="text-2xs text-slate-500 uppercase">Classes</p>
              <p className="text-lg font-semibold text-slate-200">{meta.num_classes || '—'}</p>
            </div>
          </div>

          <button
            onClick={() => navigate('/room')}
            className="w-full h-10 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-hover transition-colors flex items-center justify-center gap-2"
          >
            <Rocket size={16} /> Start Training
          </button>
        </SectionCard>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Join Room"
        description="Connect to an existing room, validate your dataset, and prepare for training"
      />

      <div className="max-w-3xl">
        {/* Stepper */}
        <div className="mb-6 card">
          <Stepper steps={STEPS} currentStep={step} />
        </div>

        {/* Step 0: Connection */}
        {step === 0 && (
          <SectionCard title="Connection Details" icon={Server}>
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Server URL</label>
                <input className="input-base" value={serverUrl} onChange={e => setServerUrl(e.target.value)} />
                <p className="mt-1 text-2xs text-slate-600">WebSocket endpoint of the FL server</p>
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">Room ID</label>
                <input className="input-base font-mono text-xs" value={roomId} onChange={e => setRoomId(e.target.value)} placeholder="Paste room ID" />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5">
                    Invite Code <span className="text-slate-600">(optional)</span>
                  </label>
                  <input className="input-base" value={inviteCode} onChange={e => setInviteCode(e.target.value)} placeholder="invite code" />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1.5">
                    Client ID <span className="text-slate-600">(optional)</span>
                  </label>
                  <input className="input-base" value={clientId} onChange={e => setClientId(e.target.value)} placeholder="auto-generated if empty" />
                </div>
              </div>
            </div>

            <div className="flex justify-end mt-5">
              <button
                onClick={() => setStep(1)}
                disabled={!roomId}
                className="flex items-center gap-2 h-9 px-5 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-hover transition-colors disabled:opacity-40"
              >
                Next <ArrowRight size={14} />
              </button>
            </div>
          </SectionCard>
        )}

        {/* Step 1: Dataset */}
        {step === 1 && (
          <SectionCard title="Dataset Configuration" icon={FolderOpen}>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Dataset Path</label>
              <input
                className="input-base font-mono text-xs"
                value={dataPath}
                onChange={e => setDataPath(e.target.value)}
                placeholder="/absolute/path/to/data.csv"
              />
              <p className="mt-1 text-2xs text-slate-600">Absolute path to your local CSV dataset file</p>
            </div>

            <Alert type="info" className="mt-4">
              <strong className="text-slate-200">What happens next:</strong>
              <ul className="mt-1 text-xs text-slate-400 space-y-0.5 list-disc list-inside">
                <li>Your dataset will be validated against the room's schema</li>
                <li>Feature count and data types will be checked</li>
                <li>Your data <strong className="text-slate-300">never leaves</strong> your machine</li>
              </ul>
            </Alert>

            <div className="flex justify-between mt-5">
              <button
                onClick={() => setStep(0)}
                className="flex items-center gap-2 h-9 px-4 rounded-lg border border-border text-sm text-slate-400 hover:text-slate-200 transition-colors"
              >
                <ArrowLeft size={14} /> Back
              </button>
              <button
                onClick={() => setStep(2)}
                disabled={!dataPath}
                className="flex items-center gap-2 h-9 px-5 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-hover transition-colors disabled:opacity-40"
              >
                Next <ArrowRight size={14} />
              </button>
            </div>
          </SectionCard>
        )}

        {/* Step 2: Review & Join */}
        {step === 2 && (
          <div className="space-y-4">
            <SectionCard title="Review & Join" icon={CheckCircle}>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between py-1.5 border-b border-border">
                  <span className="text-slate-500">Server</span>
                  <span className="text-slate-300 font-mono text-xs">{serverUrl}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-border">
                  <span className="text-slate-500">Room ID</span>
                  <span className="text-slate-300 font-mono text-xs">{roomId.slice(0, 20)}{roomId.length > 20 ? '…' : ''}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-border">
                  <span className="text-slate-500">Dataset</span>
                  <span className="text-slate-300 font-mono text-xs">{dataPath.split('/').pop()}</span>
                </div>
                {inviteCode && (
                  <div className="flex justify-between py-1.5 border-b border-border">
                    <span className="text-slate-500">Invite Code</span>
                    <span className="text-slate-300">••••••</span>
                  </div>
                )}
              </div>

              <div className="flex justify-between mt-5">
                <button
                  onClick={() => setStep(1)}
                  className="flex items-center gap-2 h-9 px-4 rounded-lg border border-border text-sm text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <ArrowLeft size={14} /> Back
                </button>
                <button
                  onClick={handleJoin}
                  disabled={loading}
                  className="flex items-center gap-2 h-10 px-6 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-hover transition-colors disabled:opacity-50"
                >
                  <Rocket size={16} />
                  {loading ? 'Joining…' : 'Initialize & Validate'}
                </button>
              </div>
            </SectionCard>

            {/* Connection logs */}
            <SectionCard title="Connection Logs" icon={Info}>
              <LogTerminal logs={logs.slice(-20)} maxHeight="180px" />
            </SectionCard>
          </div>
        )}
      </div>
    </div>
  );
}
