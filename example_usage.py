#!/usr/bin/env python3
"""Example usage of the DistFL Client SDK (v1.0.1).

Demonstrates the two main workflows:
  1. Creator flow — create room, wait for clients, start training
  2. Joiner flow — join, validate, ready, start
"""

from sklearn.linear_model import SGDClassifier
from fl_client import FLClient


# ── Creator Flow ─────────────────────────────────────────────────────────────

def create_room_sklearn():
    """Creator: create room, wait for joiners, then trigger training."""
    import pandas as pd
    import numpy as np

    model = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1,
                          learning_rate="constant", eta0=0.01)

    # Scikit-Learn requires a partial_fit call to initialize coef_ shape
    df = pd.read_csv("./data.csv")
    X = df.drop(columns=["label"]).values[:10].astype(np.float64)
    y = df["label"].values[:10].astype(np.int64)
    model.partial_fit(X, y, classes=[0, 1])

    client = FLClient(server_url="wss://fedlearn-server.onrender.com")

    # 1. Create the room (infers schema from data_path)
    room = client.create_room(
        model=model,
        data_path="./data.csv",
        target="label",
        training_config={
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01,
        },
        room_name="My Sklearn Room",
    )

    room_id = room["id"]
    print(f"✅ Room created: {room_id}")
    print(f"   Invite code: {room['invite_code']}")

    # 2. Wait for at least 2 clients to join
    client.wait_for_clients(min_clients=2, timeout=120)

    # 3. Trigger training for everyone
    client.start_training()

    return room_id


# ── Creator Flow (PyTorch) ───────────────────────────────────────────────────

def create_room_pytorch():
    """Creator: create room with a PyTorch model."""
    import torch.nn as nn

    class PhishingMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(30, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            return self.net(x)

    client = FLClient(server_url="wss://fedlearn-server.onrender.com")

    room = client.create_room(
        model=PhishingMLP(),
        data_path="./data.csv",
        target="label",
        training_config={
            "local_epochs": 2,
            "batch_size": 64,
            "learning_rate": 0.001,
        },
        room_name="My PyTorch Room",
    )

    room_id = room["id"]
    print(f"✅ Room created: {room_id}")

    client.wait_for_clients(min_clients=2)
    client.start_training()

    return room_id


# ── Joiner Flow ──────────────────────────────────────────────────────────────

def join_and_train(room_id: str):
    """Joiner: join an existing room, validate, signal ready, train."""
    import pandas as pd
    import numpy as np

    # Build the same model architecture as the room creator
    model = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1,
                          learning_rate="constant", eta0=0.01)
    df = pd.read_csv("./data.csv")
    X = df.drop(columns=["label"]).values[:10].astype(np.float64)
    y = df["label"].values[:10].astype(np.int64)
    model.partial_fit(X, y, classes=[0, 1])

    client = FLClient(server_url="wss://fedlearn-server.onrender.com")

    # 1. Join the room
    client.join(room_id, invite_code="abc123", model=model)

    # 2. Validate local dataset
    client.validate("./data.csv")

    # 3. Signal readiness
    client.ready()

    # 4. Enter the training loop (blocks until training completes)
    client.start(max_rounds=5)

    print("✅ Training complete!")


# ── Predict with a trained model ─────────────────────────────────────────────

def predict_after_training():
    """Load weights from the local state DB and run predictions."""
    import numpy as np
    import pandas as pd
    from fl_client.storage.db import StateDB
    from fl_client.model.wrapper import wrap_model

    db = StateDB("fl_client_state.db")
    state = db.load_state("worker-1")

    if not state or not state.last_weights:
        print("No trained weights found.")
        return

    model = SGDClassifier(loss="log_loss", max_iter=1)
    df = pd.read_csv("./data.csv")
    X = df.drop(columns=["label"]).values.astype(np.float64)
    y = df["label"].values

    # Initialize architecture, then inject global weights
    model.partial_fit(X[:10], y[:10], classes=[0, 1])
    wrapper = wrap_model(model)
    wrapper.set_weights(state.last_weights)

    preds = model.predict(X)
    acc = (preds == y).mean()
    print(f"✅ Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    # Creator creates room, then joiner trains:
    rid = create_room_sklearn()
    join_and_train(rid)
    predict_after_training()
