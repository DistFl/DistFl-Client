import { CheckCircle, AlertCircle, Info, XCircle, X } from 'lucide-react';

const config = {
  success: { icon: CheckCircle, bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', text: 'text-emerald-400' },
  error:   { icon: XCircle,     bg: 'bg-red-500/10',     border: 'border-red-500/20',     text: 'text-red-400' },
  warning: { icon: AlertCircle, bg: 'bg-amber-500/10',   border: 'border-amber-500/20',   text: 'text-amber-400' },
  info:    { icon: Info,        bg: 'bg-blue-500/10',    border: 'border-blue-500/20',    text: 'text-blue-400' },
};

export default function Toast({ message, type = 'info', onClose }) {
  const c = config[type] || config.info;
  const Icon = c.icon;
  return (
    <div className={`flex items-center gap-2.5 px-4 py-2.5 rounded-lg border ${c.bg} ${c.border} shadow-lg animate-in slide-in-from-right min-w-[280px]`}>
      <Icon size={16} className={c.text} />
      <span className="text-sm text-slate-200 flex-1">{message}</span>
      <button onClick={onClose} className="p-0.5 rounded text-slate-500 hover:text-slate-300">
        <X size={14} />
      </button>
    </div>
  );
}

export function Alert({ type = 'info', children, className = '' }) {
  const c = config[type] || config.info;
  const Icon = c.icon;
  return (
    <div className={`flex items-start gap-2.5 px-4 py-3 rounded-lg border ${c.bg} ${c.border} ${className}`}>
      <Icon size={16} className={`${c.text} mt-0.5 shrink-0`} />
      <div className="text-sm text-slate-300">{children}</div>
    </div>
  );
}
