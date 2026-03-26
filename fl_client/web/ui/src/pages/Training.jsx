import { useNavigate } from 'react-router-dom';
import { useStatus, useMetrics, useLogs } from '../hooks';
import { startTraining, stopTraining } from '../api/client';
import { useAppToast } from '../layouts/AppShell';
import PageHeader, { SectionCard, StatCard, StatusBadge, EmptyState } from '../components/PageHeader';
import ChartCard from '../components/ChartCard';
import LogTerminal from '../components/LogTerminal';
import {
  Activity, Play, Square, TrendingDown, Timer, Hash,
  Layers, ArrowDownRight, Database, Radio, Inbox, PlusCircle
} from 'lucide-react';

export default function Training() {
  const toast = useAppToast();
  const navigate = useNavigate();
  const status = useStatus(2000);
  const metrics = useMetrics(2000);
  const logs = useLogs(2000);

  const st = status?.status || 'idle';
  const isTraining = st === 'training';
  const canStart = st === 'validated' && !status?.training_active;
  const canStop = isTraining || status?.training_active;

  const metricsList = metrics?.metrics || [];
  const rounds = metricsList.map((_, i) => `R${i + 1}`);
  const losses = metricsList.map(m => m.loss);
  const deltas = metricsList.map(m => m.delta_w);
  const times = metricsList.map(m => m.train_time);
  const samples = metricsList.map(m => m.samples);

  const latestLoss = losses.length > 0 ? losses[losses.length - 1]?.toFixed(4) : '—';
  const latestDelta = deltas.length > 0 ? (deltas[deltas.length - 1] ?? 0).toFixed(4) : '—';
  const latestTime = times.length > 0 ? (times[times.length - 1] ?? 0).toFixed(2) + 's' : '—';
  const totalRounds = metrics?.total_rounds || 0;

  const handleStart = async () => {
    try {
      await startTraining();
      toast?.('Training started', 'success');
    } catch (err) {
      toast?.(err.message, 'error');
    }
  };

  const handleStop = async () => {
    try {
      await stopTraining();
      toast?.('Training stopped', 'info');
    } catch (err) {
      toast?.(err.message, 'error');
    }
  };

  // Empty state
  if (st === 'idle') {
    return (
      <div>
        <PageHeader title="Training" description="Federated Learning monitoring and control" />
        <EmptyState
          icon={Inbox}
          title="No active session"
          description="Create or join a room first to start training"
          action={
            <div className="flex gap-2">
              <button
                onClick={() => navigate('/create')}
                className="flex items-center gap-2 h-8 px-4 rounded-lg bg-accent/10 text-accent text-xs font-medium hover:bg-accent/20 transition-colors"
              >
                <PlusCircle size={14} /> Create Room
              </button>
            </div>
          }
        />
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Training"
        description="Monitor and control your federated learning session"
      />

      {/* Summary bar */}
      <div className="card flex flex-wrap items-center gap-x-6 gap-y-2 mb-5">
        <div className="flex items-center gap-2">
          <Radio size={14} className="text-slate-500" />
          <StatusBadge status={st} />
        </div>
        <div className="h-4 w-px bg-border hidden sm:block" />
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <span>
            <span className="text-slate-500">Room: </span>
            <span className="font-mono text-slate-300">{(status?.room_id || '').slice(0, 10)}…</span>
          </span>
          <span>
            <span className="text-slate-500">Model: </span>
            <span className="text-slate-300">{status?.room_info?.model_config?.model_type?.toUpperCase() || 'MLP'}</span>
          </span>
          <span>
            <span className="text-slate-500">Rounds: </span>
            <span className="text-slate-300 font-mono">{totalRounds}</span>
          </span>
        </div>
        <div className="flex-1" />
        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleStart}
            disabled={!canStart}
            className="flex items-center gap-1.5 h-8 px-4 rounded-lg bg-emerald-500/10 text-emerald-400 text-xs font-medium border border-emerald-500/20 hover:bg-emerald-500/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <Play size={12} /> Start
          </button>
          <button
            onClick={handleStop}
            disabled={!canStop}
            className="flex items-center gap-1.5 h-8 px-4 rounded-lg bg-red-500/10 text-red-400 text-xs font-medium border border-red-500/20 hover:bg-red-500/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <Square size={12} /> Stop
          </button>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        <StatCard label="Loss" value={latestLoss} color={losses.length ? 'accent' : 'default'} icon={TrendingDown} />
        <StatCard label="ΔW Norm" value={latestDelta} icon={ArrowDownRight} />
        <StatCard label="Train Time" value={latestTime} icon={Timer} />
        <StatCard label="Total Rounds" value={totalRounds} icon={Hash} />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-5">
        <ChartCard title="Loss" icon={TrendingDown} labels={rounds} data={losses} color="#6366f1" />
        <ChartCard title="ΔW Norm" icon={ArrowDownRight} labels={rounds} data={deltas} color="#10b981" />
        <ChartCard title="Train Time (s)" icon={Timer} labels={rounds} data={times} color="#f59e0b" />
        <ChartCard title="Samples / Round" icon={Database} labels={rounds} data={samples} color="#8b5cf6" />
      </div>

      {/* Logs */}
      <SectionCard title="Training Logs" icon={Activity}>
        <LogTerminal logs={logs} maxHeight="280px" />
      </SectionCard>
    </div>
  );
}
