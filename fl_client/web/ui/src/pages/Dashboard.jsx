import { Link } from 'react-router-dom';
import { useStatus, useLogs } from '../hooks';
import PageHeader, { SectionCard, StatCard, StatusBadge, EmptyState } from '../components/PageHeader';
import LogTerminal from '../components/LogTerminal';
import {
  PlusCircle, LogIn, Activity, Radio, Hash, Box, RotateCcw, TrendingDown,
  ArrowRight, Inbox
} from 'lucide-react';

function NextStepCard({ icon: Icon, title, desc, to, color }) {
  return (
    <Link
      to={to}
      className={`card flex items-center gap-4 hover:border-border-hover transition-all group cursor-pointer`}
    >
      <div className={`flex items-center justify-center w-10 h-10 rounded-lg ${color} shrink-0`}>
        <Icon size={18} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-200 group-hover:text-white">{title}</p>
        <p className="text-xs text-slate-500">{desc}</p>
      </div>
      <ArrowRight size={16} className="text-slate-600 group-hover:text-slate-400 transition-colors" />
    </Link>
  );
}

export default function Dashboard() {
  const status = useStatus(3000);
  const logs = useLogs(3000);
  const st = status?.status || 'idle';
  const isIdle = st === 'idle';

  return (
    <div>
      <PageHeader
        title="Dashboard"
        description="DistFL Client — Federated Learning Control Center"
      />

      {/* Summary strip */}
      {!isIdle && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          <StatCard
            label="Status"
            value={<StatusBadge status={st} />}
            icon={Radio}
          />
          <StatCard
            label="Room ID"
            value={
              <span className="text-sm font-mono truncate block">
                {status?.room_id ? status.room_id.slice(0, 12) + '…' : '—'}
              </span>
            }
            icon={Hash}
          />
          <StatCard
            label="Model"
            value={
              <span className="text-base">
                {status?.room_info?.model_config?.model_type || 'MLP'}
              </span>
            }
            icon={Box}
          />
          <StatCard
            label="Client"
            value={
              <span className="text-sm font-mono truncate block">
                {status?.client_id ? status.client_id.slice(0, 12) : '—'}
              </span>
            }
            icon={Activity}
          />
        </div>
      )}

      {/* Empty state or next steps */}
      {isIdle ? (
        <SectionCard title="Get Started" className="mb-6">
          <p className="text-sm text-slate-400 mb-4">
            No active session. Create a new room or join an existing one to begin federated training.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <NextStepCard
              icon={PlusCircle}
              title="Create Room"
              desc="Set up a new FL training room"
              to="/create"
              color="bg-accent/10 text-accent"
            />
            <NextStepCard
              icon={LogIn}
              title="Join Room"
              desc="Connect to an existing room"
              to="/join"
              color="bg-emerald-500/10 text-emerald-400"
            />
          </div>
        </SectionCard>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
          {st === 'validated' && (
            <NextStepCard
              icon={Activity}
              title="Start Training"
              desc="Begin federated learning"
              to="/room"
              color="bg-amber-500/10 text-amber-400"
            />
          )}
          {st === 'training' && (
            <NextStepCard
              icon={Activity}
              title="View Training"
              desc="Monitor current session"
              to="/room"
              color="bg-amber-500/10 text-amber-400"
            />
          )}
          {st === 'initialized' && (
            <NextStepCard
              icon={LogIn}
              title="Validate Dataset"
              desc="Complete setup to begin training"
              to="/join"
              color="bg-emerald-500/10 text-emerald-400"
            />
          )}
        </div>
      )}

      {/* Logs */}
      <SectionCard title="Recent Activity" icon={Activity}>
        {logs.length > 0 ? (
          <LogTerminal logs={logs.slice(-50)} maxHeight="280px" />
        ) : (
          <EmptyState
            icon={Inbox}
            title="No activity yet"
            description="Logs will appear here when you start working"
          />
        )}
      </SectionCard>
    </div>
  );
}
