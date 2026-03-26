#!/usr/bin/env python3
"""Example usage of the DistFL Client SDK.

This script demonstrates how to use the SDK both programmatically
and with configuration files.
"""

from fl_client import FLClient
from fl_client.config import FLConfig

# ── Option 1: Direct initialization ──────────────────────────────────────────

def run_direct():
    """Initialize and run the client directly with parameters."""
    client = FLClient(
        server_url="ws://localhost:8080",
        room_id="R123",
        data_path="./data.csv",
        batch_size=32,
        local_epochs=2,
        learning_rate=0.001,
        log_level="INFO",
    )
    client.run()


# ── Option 2: From YAML config file ─────────────────────────────────────────

def run_from_config():
    """Load config from YAML and run."""
    config = FLConfig.from_yaml("example_config.yaml")
    client = FLClient.from_config(config)
    client.run()


# ── Option 3: Custom model registration ─────────────────────────────────────

def run_with_custom_model():
    """Register a custom model type before running."""
    import torch.nn as nn
    from fl_client.model.registry import ModelRegistry

    @ModelRegistry.register("custom_mlp")
    def build_custom(config: dict) -> nn.Module:
        return nn.Sequential(
            nn.Linear(config["input_size"], 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config["output_size"]),
        )

    # Now the server can specify model_type: "custom_mlp" and it will work
    client = FLClient(
        server_url="ws://localhost:8080",
        room_id="R123",
        data_path="./data.csv",
    )
    client.run()


if __name__ == "__main__":
    # Choose one:
    run_from_config()
    # run_direct()
    # run_with_custom_model()
