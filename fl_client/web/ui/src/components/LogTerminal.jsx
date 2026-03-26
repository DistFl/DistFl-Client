import { useRef, useEffect } from 'react';

export default function LogTerminal({ logs = [], maxHeight = '320px' }) {
  const panelRef = useRef(null);

  useEffect(() => {
    if (panelRef.current) {
      panelRef.current.scrollTop = panelRef.current.scrollHeight;
    }
  }, [logs.length]);

  const levelColor = (level) => {
    const l = (level || '').toUpperCase();
    if (l === 'ERROR')   return 'text-red-400';
    if (l === 'WARNING') return 'text-amber-400';
    if (l === 'DEBUG')   return 'text-slate-500';
    return 'text-blue-400';
  };

  return (
    <div
      ref={panelRef}
      className="rounded-lg bg-surface border border-border overflow-y-auto font-mono text-xs"
      style={{ maxHeight }}
    >
      {logs.length === 0 ? (
        <div className="flex items-center justify-center h-24 text-slate-600">
          No logs yet
        </div>
      ) : (
        <div className="p-3 space-y-0.5">
          {logs.map((entry, i) => (
            <div key={i} className="flex gap-3 leading-5 hover:bg-surface-1/50 px-1 -mx-1 rounded">
              <span className="text-slate-600 shrink-0 w-[60px]">{entry.timestamp || ''}</span>
              <span className={`font-medium shrink-0 w-[52px] ${levelColor(entry.level)}`}>
                {entry.level || 'INFO'}
              </span>
              <span className="text-slate-400 break-all">{entry.message || ''}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
