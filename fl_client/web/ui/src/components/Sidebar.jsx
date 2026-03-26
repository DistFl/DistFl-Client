import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, PlusCircle, LogIn, Activity, Settings,
  Zap, ChevronLeft, ChevronRight
} from 'lucide-react';

const navItems = [
  { to: '/',         label: 'Dashboard',   icon: LayoutDashboard },
  { to: '/create',   label: 'Create Room', icon: PlusCircle },
  { to: '/join',     label: 'Join Room',   icon: LogIn },
  { to: '/room',     label: 'Training',    icon: Activity },
  { to: '/settings', label: 'Settings',    icon: Settings },
];

function StatusDot({ status }) {
  const color =
    status === 'training'  ? 'bg-amber-400 animate-pulse-slow' :
    status === 'validated'  ? 'bg-emerald-400' :
    status === 'initialized' ? 'bg-emerald-400' :
    status === 'stopped'   ? 'bg-slate-400' :
    'bg-slate-600';
  return <span className={`inline-block w-2 h-2 rounded-full ${color}`} />;
}

export default function Sidebar({ collapsed, onToggle, status }) {
  const st = status?.status || 'idle';

  return (
    <aside
      className={`fixed top-0 left-0 h-full z-40 flex flex-col bg-surface-1 border-r border-border transition-all duration-200 ${
        collapsed ? 'w-16' : 'w-60'
      }`}
    >
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-4 h-14 border-b border-border shrink-0">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent/10 text-accent shrink-0">
          <Zap size={16} />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <h1 className="text-sm font-semibold text-slate-100 truncate">DistFL</h1>
            <p className="text-2xs text-slate-500">Client SDK v1.0</p>
          </div>
        )}
        <button
          onClick={onToggle}
          className="ml-auto p-1 rounded-md text-slate-500 hover:text-slate-300 hover:bg-surface-3 transition-colors"
          aria-label="Toggle sidebar"
        >
          {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
        </button>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto">
        {!collapsed && (
          <p className="px-2 mb-2 text-2xs font-medium text-slate-500 uppercase tracking-wider">
            Navigation
          </p>
        )}
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-2.5 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-accent/10 text-accent'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-surface-3'
              } ${collapsed ? 'justify-center' : ''}`
            }
            title={collapsed ? label : undefined}
          >
            <Icon size={18} strokeWidth={1.75} />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-3 py-3 border-t border-border shrink-0">
        <div className={`flex items-center gap-2 ${collapsed ? 'justify-center' : ''}`}>
          <StatusDot status={st} />
          {!collapsed && (
            <span className="text-xs text-slate-500 capitalize">{st}</span>
          )}
        </div>
      </div>
    </aside>
  );
}
