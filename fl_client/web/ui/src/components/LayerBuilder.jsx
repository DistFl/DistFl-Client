import { Plus, X, GripVertical } from 'lucide-react';

export default function LayerBuilder({ layers, onChange }) {
  const addLayer = () => {
    onChange([...layers, { type: 'Dense', units: 64, activation: 'ReLU' }]);
  };

  const removeLayer = (i) => {
    if (layers.length <= 1) return;
    onChange(layers.filter((_, idx) => idx !== i));
  };

  const updateLayer = (i, field, val) => {
    const next = layers.map((l, idx) => {
      if (idx !== i) return l;
      const updated = { ...l, [field]: val };
      if (field === 'type') {
        if (val === 'Dropout') { updated.units = 0.5; updated.activation = 'None'; }
        if (val === 'BatchNorm') { updated.activation = 'None'; }
      }
      return updated;
    });
    onChange(next);
  };

  return (
    <div>
      <div className="space-y-2">
        {layers.map((layer, i) => (
          <div
            key={i}
            className="flex items-center gap-2 p-2.5 rounded-lg bg-surface border border-border group hover:border-border-hover transition-colors"
          >
            {/* Index */}
            <div className="flex items-center justify-center w-6 h-6 rounded-md bg-accent/10 text-accent text-2xs font-semibold shrink-0">
              {i + 1}
            </div>

            {/* Type */}
            <select
              className="input-base h-8 text-xs w-28"
              value={layer.type}
              onChange={(e) => updateLayer(i, 'type', e.target.value)}
            >
              <option value="Dense">Dense</option>
              <option value="Dropout">Dropout</option>
              <option value="BatchNorm">BatchNorm</option>
            </select>

            {/* Units / Rate */}
            <input
              className="input-base h-8 text-xs w-20"
              type="number"
              value={layer.units}
              onChange={(e) => updateLayer(i, 'units', layer.type === 'Dropout' ? parseFloat(e.target.value) : parseInt(e.target.value))}
              placeholder={layer.type === 'Dropout' ? 'Rate' : 'Units'}
              disabled={layer.type === 'BatchNorm'}
              min={layer.type === 'Dropout' ? 0 : 1}
              max={layer.type === 'Dropout' ? 1 : 8192}
              step={layer.type === 'Dropout' ? 0.1 : 1}
            />

            {/* Activation */}
            <select
              className="input-base h-8 text-xs w-24"
              value={layer.activation}
              onChange={(e) => updateLayer(i, 'activation', e.target.value)}
              disabled={layer.type !== 'Dense'}
            >
              <option value="ReLU">ReLU</option>
              <option value="Sigmoid">Sigmoid</option>
              <option value="Tanh">Tanh</option>
              <option value="None">None</option>
            </select>

            {/* Info labels */}
            <div className="flex-1" />

            {/* Remove */}
            <button
              onClick={() => removeLayer(i)}
              className="p-1 rounded opacity-0 group-hover:opacity-100 text-slate-600 hover:text-red-400 hover:bg-red-500/10 transition-all"
              title="Remove layer"
              disabled={layers.length <= 1}
            >
              <X size={14} />
            </button>
          </div>
        ))}
      </div>

      <button
        onClick={addLayer}
        className="mt-3 flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-slate-400 border border-dashed border-border hover:border-border-hover hover:text-slate-300 transition-colors"
      >
        <Plus size={12} /> Add Layer
      </button>
    </div>
  );
}
