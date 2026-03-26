<p align="center">
  <h1 align="center">DistFL</h1>
  <p align="center">
    <strong>Production-Grade Federated Learning Client SDK</strong>
  </p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-≥3.10-blue?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://pypi.org/project/distfl-client/"><img src="https://img.shields.io/pypi/v/distfl-client?color=green&label=PyPI" alt="PyPI"></a>
    <a href="#license"><img src="https://img.shields.io/badge/license-MIT-purple" alt="License"></a>
    <a href="#testing"><img src="https://img.shields.io/badge/tests-52%20passed-brightgreen" alt="Tests"></a>
  </p>
</p>

---

Bring your own model (PyTorch or Scikit-Learn), connect to a DistFL server, train locally on **private data**, and let the server aggregate updates — all via compressed WebSocket communication. No raw data ever leaves the client.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client A   │     │   Client B   │     │   Client C   │
│  (Hospital)  │     │  (Bank)      │     │  (Lab)       │
│  Local Data  │     │  Local Data  │     │  Local Data  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │   model updates    │   (gzip+WS)        │
       └────────────────────┼────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  DistFL Server │
                    │  (Go Backend)  │
                    │  FedAvg Agg.   │
                    └───────┬────────┘
                            │
                    aggregated global model
                    broadcast to all clients
```

---

## ✨ Features

| Category | Details |
|---|---|
| **BYOM** | Use any PyTorch `nn.Module` or Scikit-Learn estimator with `partial_fit` |
| **Simple Lifecycle** | `initialize()` → `validate()` → `start()` — 3 calls to go from zero to training |
| **Room-Based FL** | Create rooms, share invite codes, configure training params per room |
| **Compressed WebSocket** | GZIP-compressed binary messages over persistent WebSocket connections |
| **Auto Reconnect** | Exponential backoff with configurable delays and heartbeat pings |
| **Crash Recovery** | SQLite-backed state persistence — no duplicate round submissions after restart |
| **Live Dashboard** | Built-in web UI with real-time loss curves, ΔW tracking, and training logs |
| **Prediction** | Extract globally-aggregated weights and run inference locally |
| **CLI** | `distfl run`, `distfl create-room`, `distfl join-room`, `distfl ui`, `distfl status` |

---

## 📦 Installation

```bash
pip install distfl-client
```

**From source:**

```bash
git clone https://github.com/AbhaySingh002/new-repo-code.git
cd new-repo-code/DistFL
pip install -e ".[dev]"
```

---

## 🚀 Quick Start

### 1. Room Creator

The creator initializes the model, creates a room, waits for participants, and starts training:

```python
from sklearn.linear_model import SGDClassifier
from fl_client import FLClient
import pandas as pd
import numpy as np

# Prepare model (scikit-learn requires partial_fit to initialize weights)
model = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1,
                      learning_rate="constant", eta0=0.01)
df = pd.read_csv("./data.csv")
X = df.drop(columns=["label"]).values[:10].astype(np.float64)
y = df["label"].values[:10].astype(np.int64)
model.partial_fit(X, y, classes=[0, 1])

# Create room
client = FLClient(server_url="ws://localhost:8080")
room = client.create_room(
    model=model,
    data_path="./data.csv",
    target="label",
    training_config={"local_epochs": 1, "batch_size": 32, "learning_rate": 0.01},
    room_name="Phishing Detection",
)

room_id = room["id"]
print(f"✅ Room created: {room_id}")
print(f"   Invite code: {room['invite_code']}")

# Wait for participants, then start
client.wait_for_clients(min_clients=2, timeout=120)
client.start_training()
```

### 2. Room Joiner

Each participant joins an existing room, validates their local dataset, and trains:

```python
from sklearn.linear_model import SGDClassifier
from fl_client import FLClient

model = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1,
                      learning_rate="constant", eta0=0.01)
# ... partial_fit to initialize shape (same architecture as creator)

client = FLClient(server_url="ws://localhost:8080")
client.join(room_id, invite_code="abc123", model=model)
client.validate("./data.csv")
client.ready()
client.start(max_rounds=5)  # Blocks until training completes

print("✅ Training complete!")
```

### 3. PyTorch Models

```python
import torch.nn as nn
from fl_client import FLClient

class PhishingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        return self.net(x)

client = FLClient(server_url="ws://localhost:8080")
room = client.create_room(
    model=PhishingMLP(),
    data_path="./data.csv",
    target="label",
    training_config={"local_epochs": 2, "batch_size": 64, "learning_rate": 0.001},
    room_name="PyTorch FL Room",
)
```

### 4. Prediction After Training

```python
from fl_client.storage.db import StateDB
from fl_client.model.wrapper import wrap_model

db = StateDB("fl_client_state.db")
state = db.load_state("worker-1")

wrapper = wrap_model(model)
wrapper.set_weights(state.last_weights)

predictions = model.predict(X_test)
accuracy = (predictions == y_test).mean()
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
```

---

## 💻 CLI Reference

```bash
# Full lifecycle from a YAML config
distfl run --config config.yaml

# Create a room
distfl create-room --server-url ws://localhost:8080 --room-name "My Room"

# Join a room and train
distfl join-room ROOM_ID --data ./data.csv --server-url ws://localhost:8080

# Launch the real-time web dashboard
distfl ui --port 5050

# Inspect persisted client state
distfl status --client-id worker-1 --db-path fl_client_state.db

# Clear persisted state
distfl clear --client-id worker-1 --db-path fl_client_state.db
```

---

## ⚙️ Configuration

All options can be set via **YAML file**, **CLI flags**, or **environment variables** (`FL_` prefix):

```yaml
# Server connection
server_url: "ws://localhost:8080"
room_id: ""                          # Leave empty to create a new room
client_id: ""                        # Auto-generated if omitted

# Dataset
data_path: "./data.csv"
label_column: "label"

# Training hyperparameters
batch_size: 32
local_epochs: 2
learning_rate: 0.001

# State persistence
db_path: "fl_client_state.db"        # SQLite for crash recovery

# Networking
reconnect_max_delay: 60.0            # Max backoff delay (seconds)
reconnect_base_delay: 1.0            # Initial reconnect delay
heartbeat_interval: 30.0             # WebSocket ping interval

# Dashboard
dashboard_port: 5050                 # Real-time metrics UI (0 = disabled)

# Logging
log_level: "INFO"                    # DEBUG, INFO, WARNING, ERROR
```

---

## 🏗️ Architecture

```
fl_client/
├── core/
│   ├── client.py              # FLClient — main orchestrator (3-step lifecycle)
│   ├── connection.py          # WebSocket manager (reconnect, heartbeat)
│   └── state_manager.py       # Round tracking, crash recovery
├── model/
│   └── wrapper.py             # Unified BYOM wrapper (PyTorch + Scikit-Learn)
├── training/
│   ├── dataset.py             # FlexibleDataAdapter (CSV, DataLoader, NumPy)
│   └── trainer.py             # Framework-agnostic epoch training
├── communication/
│   ├── serializer.py          # Weights ↔ JSON (full float32 precision)
│   └── compressor.py          # GZIP compress / decompress
├── validation/
│   └── checks.py              # Pre-send weight & loss validation
├── storage/
│   └── db.py                  # SQLite state persistence
├── config/
│   └── config.py              # YAML / env-based configuration
├── web/
│   ├── bridge.py              # FastAPI backend for the dashboard
│   └── static/                # Pre-built React dashboard UI
├── dashboard/
│   └── dashboard.py           # Lightweight real-time metrics server
└── cli/
    └── main.py                # CLI entry point (distfl)
```

---

## 🔌 Server Protocol

### HTTP Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/create_room` | `POST` | Create a new FL room with model config and initial weights |
| `/join_room` | `POST` | Join a client to an existing room |
| `/room_info?room_id=X` | `GET` | Fetch room configuration and schema |
| `/room_status?room_id=X` | `GET` | Get current room state (round, clients, status) |
| `/health` | `GET` | Server health check |

### WebSocket (`/ws/:room_id?client_id=X`)

**Server → Client:**

| Message | Description |
|---|---|
| `room_init` | Room config + current round + initial weights |
| `global_model` | Aggregated global model after FedAvg (gzip binary) |
| `start_round` | Signal to begin a new training round |
| `sync` | Synchronization signal with latest weights |
| `update_result` | Validation result for a submitted update |

**Client → Server:**

| Message | Description |
|---|---|
| `model_update` | Trained weights + loss + metrics (gzip binary) |

### Weight Format

All weights use a **3D nested list** format: `[[[float32]]]` — layers × rows × cols.

```
Pipeline: model → numpy → float32 → list → JSON → gzip → WebSocket binary
```

---

## 🧪 Supported Frameworks

| Framework | Requirements | Weight Extraction |
|---|---|---|
| **PyTorch** | Any `nn.Module` | `state_dict()` → 3D float32 lists |
| **Scikit-Learn** | Estimator with `partial_fit` (e.g. `SGDClassifier`, `SGDRegressor`) | `coef_` + `intercept_` → 3D float32 lists |

---

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all 52 unit tests
python -m pytest tests/ -v
```

### Test Coverage

| Module | Tests | What's Covered |
|---|---|---|
| `test_compressor.py` | 7 | Compress/decompress round-trip, empty data, large payloads |
| `test_connection.py` | 8 | WS URL construction, connect/disconnect, send/receive |
| `test_serializer.py` | 9 | Serialize/deserialize, shape preservation, JSON round-trip |
| `test_storage.py` | 7 | SQLite save/load, upsert, clear, round logging |
| `test_trainer.py` | 4 | Train results, finite loss, accuracy metrics, multi-epoch |
| `test_validation.py` | 17 | NaN/Inf/shape/range checks, loss validation, weight shapes |

---

## 🔐 Privacy & Security

- **Data never leaves the client** — only model weight updates are transmitted
- **GZIP compression** — reduces bandwidth and adds a layer of obfuscation
- **Server-side validation** — NaN, Inf, out-of-range, shape mismatch, L2 norm, and duplicate submission checks
- **Invite codes** — rooms can be access-controlled via invite codes
- **Crash recovery** — SQLite persistence prevents duplicate round submissions

---

## 📄 License

MIT

---

<p align="center">
  Built with ❤️ for privacy-preserving machine learning
</p>
