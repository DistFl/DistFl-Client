export default function PageHeader({ title, description, actions }) {
  return (
    <div className="flex items-start justify-between mb-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-100">{title}</h2>
        {description && (
          <p className="mt-1 text-sm text-slate-400">{description}</p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export function SectionCard({ title, icon: Icon, children, className = '' }) {
  return (
    <div className={`card ${className}`}>
      {title && (
        <div className="flex items-center gap-2 mb-4">
          {Icon && <Icon size={14} className="text-slate-500" strokeWidth={1.75} />}
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</h3>
        </div>
      )}
      {children}
    </div>
  );
}

export function StatCard({ label, value, color, icon: Icon }) {
  const colorMap = {
    green: 'text-emerald-400',
    yellow: 'text-amber-400',
    red: 'text-red-400',
    accent: 'text-accent',
    default: 'text-slate-100',
  };
  return (
    <div className="card flex flex-col justify-between min-h-[88px]">
      <div className="flex items-center justify-between">
        <span className="text-2xs font-medium text-slate-500 uppercase tracking-wider">{label}</span>
        {Icon && <Icon size={14} className="text-slate-600" />}
      </div>
      <div className={`stat-value mt-1 ${colorMap[color] || colorMap.default}`}>
        {value}
      </div>
    </div>
  );
}

export function StatusBadge({ status }) {
  const styles = {
    idle:        'bg-slate-700/60 text-slate-400',
    initialized: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    validated:   'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    training:    'bg-amber-500/10 text-amber-400 border-amber-500/20',
    stopped:     'bg-slate-500/10 text-slate-400 border-slate-500/20',
    error:       'bg-red-500/10 text-red-400 border-red-500/20',
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium border ${
      styles[status] || styles.idle
    }`}>
      {status}
    </span>
  );
}

export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {Icon && (
        <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-surface-3 text-slate-500 mb-4">
          <Icon size={24} strokeWidth={1.5} />
        </div>
      )}
      <h3 className="text-sm font-medium text-slate-300">{title}</h3>
      {description && <p className="mt-1 text-xs text-slate-500 max-w-xs">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
