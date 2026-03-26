# DistFL — Federated Learning Client SDK

A production-grade Python client SDK for room-based federated learning. Connects to the FL server, trains locally on private data, and communicates compressed model updates via WebSockets.

## Features

- **WebSocket communication** with gzip-compressed binary messages
- **Automatic reconnect** with exponential backoff
- **SQLite persistence** for crash recovery (no duplicate round submissions)
- **Config-driven model building** — MLP, CNN, LSTM out of the box, extensible via registry
- **Epoch-based local training** with PyTorch
- **Client-side validation** (NaN/Inf, shape, loss) before sending updates
- **CLI interface** — `fl-client run --config config.yaml`
- **pip-installable** package

## Installation

```bash
cd DistFL
pip install -e ".[dev]"
```

## Quick Start

### 1. Create a config file

```yaml
server_url: "ws://localhost:8080"
room_id: "R123"
data_path: "./data.csv"
batch_size: 32
local_epochs: 2
learning_rate: 0.001
label_column: "label"
```

### 2. Run via CLI

```bash
fl-client run --config config.yaml
```

### 3. Run programmatically

```python
from fl_client import FLClient

client = FLClient(
    server_url="ws://localhost:8080",
    room_id="R123",
    data_path="./data.csv",
)
client.run()
```

## Architecture

```
fl_client/
├── core/
│   ├── client.py          # Main orchestrator (lifecycle, event loop)
│   ├── connection.py      # WebSocket manager (reconnect, heartbeat)
│   └── state_manager.py   # Round tracking, crash recovery
├── model/
│   ├── registry.py        # Model registry (MLP, CNN, LSTM)
│   └── builder.py         # Config-driven model builder
├── training/
│   ├── dataset.py         # CSV loading, schema validation, metadata
│   └── trainer.py         # Epoch-based PyTorch training
├── communication/
│   ├── serializer.py      # tensor ↔ JSON (float16 optimization)
│   └── compressor.py      # gzip compress/decompress
├── validation/
│   └── checks.py          # Pre-send weight/loss validation
├── storage/
│   └── db.py              # SQLite state persistence
├── config/
│   └── config.py          # YAML/env-based configuration
└── cli/
    └── main.py            # CLI entry point
```

## Server Protocol

| Endpoint | Method | Purpose |
|---|---|---|
| `POST /join_room` | HTTP | Join a room with metadata |
| `GET /room_status?room_id=X` | HTTP | Get room state |
| `GET /ws/:room_id?client_id=X` | WS | WebSocket connection |

### WebSocket Messages

**Server → Client:**
- `global_model` — new global weights + round number (gzip binary)
- `update_result` — validation result for submitted update
- `error` — error message

**Client → Server:**
- `model_update` — trained weights + loss + metrics (gzip binary)

### Weight Format

3D nested list: `[[[float32]]]` — layers × rows × cols.

Pipeline: `tensor → numpy → float16 → list → JSON → gzip → WebSocket binary`

## Custom Models

Register custom model types without modifying SDK code:

```python
from fl_client.model.registry import ModelRegistry
import torch.nn as nn

@ModelRegistry.register("my_model")
def build_my_model(config: dict) -> nn.Module:
    return nn.Sequential(
        nn.Linear(config["input_size"], 256),
        nn.ReLU(),
        nn.Linear(256, config["output_size"]),
    )
```

## CLI Commands

```bash
fl-client run --config config.yaml          # Start training
fl-client status --client-id abc --db-path fl_state.db  # View state
fl-client clear --client-id abc --db-path fl_state.db   # Clear state
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## License

MIT
