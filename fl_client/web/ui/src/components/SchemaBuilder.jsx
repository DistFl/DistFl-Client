import { useState, useEffect } from 'react';
import { Plus, X } from 'lucide-react';

export default function SchemaBuilder({ value, onChange }) {
  const [mode, setMode] = useState('visual'); // visual | json
  const [jsonText, setJsonText] = useState('');
  const [jsonError, setJsonError] = useState('');

  // Parse value into columns
  const columns = value?.columns || [];
  const targetColumn = value?.target_column || '';

  // Sync visual → JSON text when switching to JSON mode
  useEffect(() => {
    if (mode === 'json') {
      setJsonText(JSON.stringify(value, null, 2));
      setJsonError('');
    }
  }, [mode]);

  const updateColumns = (newColumns, newTarget) => {
    onChange({
      columns: newColumns,
      target_column: newTarget ?? targetColumn,
    });
  };

  const addColumn = () => {
    updateColumns([...columns, { name: '', type: 'float' }], targetColumn);
  };

  const removeColumn = (i) => {
    if (columns.length <= 1) return;
    const next = columns.filter((_, idx) => idx !== i);
    const removedWasTarget = columns[i].name === targetColumn;
    updateColumns(next, removedWasTarget ? (next[0]?.name || '') : targetColumn);
  };

  const updateCol = (i, field, val) => {
    const next = columns.map((c, idx) => idx === i ? { ...c, [field]: val } : c);
    // If name changed and this was target, update target too
    if (field === 'name' && columns[i].name === targetColumn) {
      updateColumns(next, val);
    } else {
      updateColumns(next, targetColumn);
    }
  };

  const setTarget = (colName) => {
    onChange({ ...value, target_column: colName });
  };

  const applyJson = () => {
    try {
      const parsed = JSON.parse(jsonText);
      if (!parsed.columns || !Array.isArray(parsed.columns)) {
        setJsonError('Schema must have a "columns" array');
        return;
      }
      onChange(parsed);
      setJsonError('');
    } catch (e) {
      setJsonError('Invalid JSON: ' + e.message);
    }
  };

  return (
    <div>
      {/* Mode toggle */}
      <div className="flex items-center gap-1 mb-3 bg-surface rounded-lg p-0.5 w-fit">
        <button
          onClick={() => setMode('visual')}
          className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            mode === 'visual' ? 'bg-accent text-white' : 'text-slate-400 hover:text-slate-200'
          }`}
        >Visual</button>
        <button
          onClick={() => { setMode('json'); }}
          className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            mode === 'json' ? 'bg-accent text-white' : 'text-slate-400 hover:text-slate-200'
          }`}
        >JSON</button>
      </div>

      {mode === 'visual' ? (
        <div>
          {/* Header */}
          <div className="grid grid-cols-[1fr_120px_60px_32px] gap-2 mb-1.5 px-1">
            <span className="text-2xs text-slate-500 font-medium uppercase tracking-wider">Column Name</span>
            <span className="text-2xs text-slate-500 font-medium uppercase tracking-wider">Type</span>
            <span className="text-2xs text-slate-500 font-medium uppercase tracking-wider text-center">Target</span>
            <span />
          </div>

          {/* Rows */}
          <div className="space-y-1.5">
            {columns.map((col, i) => (
              <div key={i} className="grid grid-cols-[1fr_120px_60px_32px] gap-2 items-center">
                <input
                  className="input-base h-8 text-xs"
                  value={col.name}
                  onChange={(e) => updateCol(i, 'name', e.target.value)}
                  placeholder="column_name"
                />
                <select
                  className="input-base h-8 text-xs"
                  value={col.type}
                  onChange={(e) => updateCol(i, 'type', e.target.value)}
                >
                  <option value="float">float</option>
                  <option value="int">int</option>
                  <option value="string">string</option>
                </select>
                <div className="flex justify-center">
                  <input
                    type="radio"
                    name="target-col"
                    checked={col.name === targetColumn}
                    onChange={() => setTarget(col.name)}
                    className="w-3.5 h-3.5 accent-accent"
                  />
                </div>
                <button
                  onClick={() => removeColumn(i)}
                  className="p-1 rounded text-slate-600 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                  title="Remove"
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>

          <button
            onClick={addColumn}
            className="mt-3 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-slate-400 border border-dashed border-border hover:border-border-hover hover:text-slate-300 transition-colors"
          >
            <Plus size={12} /> Add Column
          </button>
        </div>
      ) : (
        <div>
          <textarea
            className="w-full rounded-lg bg-surface border border-border p-3 text-xs font-mono text-slate-300 focus:outline-none focus:border-accent resize-y min-h-[160px]"
            rows={8}
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
          />
          {jsonError && <p className="mt-1 text-xs text-red-400">{jsonError}</p>}
          <button
            onClick={applyJson}
            className="mt-2 px-3 py-1.5 rounded-lg text-xs bg-accent/10 text-accent hover:bg-accent/20 transition-colors font-medium"
          >
            Apply JSON
          </button>
        </div>
      )}
    </div>
  );
}
