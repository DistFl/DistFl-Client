import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createRoom } from '../api/client';
import { useAppToast } from '../layouts/AppShell';
import PageHeader, { SectionCard } from '../components/PageHeader';
import SchemaBuilder from '../components/SchemaBuilder';
import LayerBuilder from '../components/LayerBuilder';
import { Alert } from '../components/Toast';
import {
  Server, Tag, Layers, Cpu, Database, Table2, Settings2, Rocket,
  CheckCircle, Copy
} from 'lucide-react';

const DEFAULT_LAYERS = [
  { type: 'Dense', units: 128, activation: 'ReLU' },
  { type: 'Dense', units: 64, activation: 'ReLU' },
];

const DEFAULT_SCHEMA = {
  columns: [
    { name: 'feature_1', type: 'float' },
    { name: 'feature_2', type: 'float' },
    { name: 'label', type: 'int' },
  ],
  target_column: 'label',
};

export default function CreateRoom() {
  const toast = useAppToast();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // Form state
  const [serverUrl, setServerUrl] = useState('ws://localhost:8080');
  const [roomName, setRoomName] = useState('fl-room');
  const [modelType, setModelType] = useState('MLP');
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const [schema, setSchema] = useState(DEFAULT_SCHEMA);
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(32);
  const [lr, setLr] = useState(0.001);

  const handleCreate = async () => {
    try {
      setLoading(true);

      // Serialize layers to backend format
      const hiddenLayers = layers
        .filter(l => l.type === 'Dense')
        .map(l => l.units);
      const activations = layers
        .filter(l => l.type === 'Dense')
        .map(l => l.activation.toLowerCase());

      const body = {
        server_url: serverUrl,
        room_name: roomName,
        model_config_data: {
          model_type: modelType.toLowerCase(),
          input_dim: schema.columns.filter(c => c.name !== schema.target_column).length,
          output_dim: 2,
          hidden_layers: hiddenLayers,
          activations: activations,
        },
        data_schema: schema,
        training_config: {
          epochs: parseInt(epochs),
          batch_size: parseInt(batchSize),
          learning_rate: parseFloat(lr),
        },
      };

      const data = await createRoom(body);
      setResult(data);
      toast?.('Room created successfully', 'success');
    } catch (err) {
      toast?.(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Success panel
  if (result) {
    const room = result.room || {};
    return (
      <div>
        <PageHeader title="Room Created" description="Your federated learning room is ready" />
        <SectionCard className="max-w-lg">
          <div className="flex items-center gap-3 mb-5">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400">
              <CheckCircle size={20} />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-200">Room Created Successfully</p>
              <p className="text-xs text-slate-500">{roomName}</p>
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-2xs text-slate-500 uppercase tracking-wider">Room ID</label>
              <div className="flex items-center gap-2 mt-1">
                <code className="flex-1 px-3 py-2 rounded-lg bg-surface border border-border text-xs font-mono text-slate-300 select-all">
                  {room.id || '—'}
                </code>
                <button
                  onClick={() => { navigator.clipboard.writeText(room.id || ''); toast?.('Copied', 'info'); }}
                  className="p-2 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-surface-3"
                >
                  <Copy size={14} />
                </button>
              </div>
            </div>
            <div>
              <label className="text-2xs text-slate-500 uppercase tracking-wider">Invite Code</label>
              <div className="flex items-center gap-2 mt-1">
                <code className="flex-1 px-3 py-2 rounded-lg bg-surface border border-border text-xs font-mono text-slate-300 select-all">
                  {room.invite_code || '—'}
                </code>
                <button
                  onClick={() => { navigator.clipboard.writeText(room.invite_code || ''); toast?.('Copied', 'info'); }}
                  className="p-2 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-surface-3"
                >
                  <Copy size={14} />
                </button>
              </div>
            </div>
          </div>

          <div className="flex gap-3 mt-5">
            <button
              onClick={() => navigate('/room')}
              className="flex-1 h-9 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-hover transition-colors"
            >
              Go to Training
            </button>
            <button
              onClick={() => { setResult(null); }}
              className="h-9 px-4 rounded-lg border border-border text-sm text-slate-400 hover:text-slate-200 hover:border-border-hover transition-colors"
            >
              Create Another
            </button>
          </div>
        </SectionCard>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Create Room"
        description="Configure and launch a new federated learning room"
      />

      <div className="space-y-5 max-w-3xl">
        {/* 1. Room Basics */}
        <SectionCard title="Room Basics" icon={Server}>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Server URL</label>
              <input className="input-base" value={serverUrl} onChange={e => setServerUrl(e.target.value)} />
              <p className="mt-1 text-2xs text-slate-600">WebSocket endpoint of the FL server</p>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Room Name</label>
              <input className="input-base" value={roomName} onChange={e => setRoomName(e.target.value)} />
            </div>
          </div>
        </SectionCard>

        {/* 2. Model Architecture */}
        <SectionCard title="Model Architecture" icon={Layers}>
          <div className="mb-4">
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Model Type</label>
            <select className="input-base w-48" value={modelType} onChange={e => setModelType(e.target.value)}>
              <option value="MLP">MLP (Multilayer Perceptron)</option>
              <option value="CNN">CNN (Convolutional)</option>
              <option value="RNN">RNN (Recurrent)</option>
            </select>
          </div>

          {modelType === 'MLP' && (
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-2">
                Layer Configuration
              </label>
              <LayerBuilder layers={layers} onChange={setLayers} />
            </div>
          )}
        </SectionCard>

        {/* 3. Data Schema */}
        <SectionCard title="Data Schema" icon={Table2}>
          <SchemaBuilder value={schema} onChange={setSchema} />
        </SectionCard>

        {/* 4. Training Parameters */}
        <SectionCard title="Training Parameters" icon={Settings2}>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Epochs</label>
              <input className="input-base" type="number" value={epochs} onChange={e => setEpochs(e.target.value)} min={1} />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Batch Size</label>
              <input className="input-base" type="number" value={batchSize} onChange={e => setBatchSize(e.target.value)} min={1} />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">Learning Rate</label>
              <input className="input-base" type="number" value={lr} onChange={e => setLr(e.target.value)} step={0.0001} min={0} />
            </div>
          </div>
        </SectionCard>

        {/* Create Action */}
        <div className="flex items-center justify-between pt-2 pb-8">
          <p className="text-xs text-slate-600">
            All configuration will be sent to the server for room creation.
          </p>
          <button
            onClick={handleCreate}
            disabled={loading}
            className="flex items-center gap-2 h-10 px-6 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Rocket size={16} />
            {loading ? 'Creating…' : 'Create Room'}
          </button>
        </div>
      </div>
    </div>
  );
}
