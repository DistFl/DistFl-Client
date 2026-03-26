"""CLI for the DistFL federated learning client."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict

import yaml

from fl_client.config.config import FLConfig
from fl_client.core.client import FLClient


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="distfl",
        description="DistFL — Federated Learning Client SDK",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── run ───────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run the FL client (full lifecycle)")
    run_parser.add_argument("--config", "-c", help="Path to YAML config file")
    run_parser.add_argument("--server-url", help="Server WebSocket URL")
    run_parser.add_argument("--room-id", help="Room ID to join")
    run_parser.add_argument("--data", "-d", help="Path to CSV dataset")
    run_parser.add_argument("--client-id", help="Client ID (auto-generated if omitted)")
    run_parser.add_argument("--reconnect-max-retries", type=int, default=0, help="Max reconnect retries (0=unlimited)")
    run_parser.add_argument("--log-level", default="INFO", help="Logging level")
    run_parser.add_argument("--dashboard-port", type=int, default=5050, help="Dashboard port (0=disabled)")

    # ── create-room ───────────────────────────────────────────────────────
    create_parser = subparsers.add_parser("create-room", help="Create a new FL room")
    create_parser.add_argument("--config", "-c", help="Path to room YAML config file")
    create_parser.add_argument("--server-url", default="ws://localhost:8080", help="Server URL")
    create_parser.add_argument("--room-name", default="fl-room", help="Room name")
    create_parser.add_argument("--model-type", default="mlp", help="Model type (mlp, cnn, lstm)")
    create_parser.add_argument("--input-size", type=int, help="Model input size")
    create_parser.add_argument("--hidden-size", type=int, help="Model hidden size")
    create_parser.add_argument("--output-size", type=int, help="Model output size")
    create_parser.add_argument("--schema-file", help="Path to JSON schema file")
    create_parser.add_argument("--log-level", default="INFO", help="Logging level")

    # ── join-room ─────────────────────────────────────────────────────────
    join_parser = subparsers.add_parser("join-room", help="Join room and start training")
    join_parser.add_argument("room_id", help="Room ID to join")
    join_parser.add_argument("--data", "-d", required=True, help="Path to CSV dataset")
    join_parser.add_argument("--server-url", default="ws://localhost:8080", help="Server URL")
    join_parser.add_argument("--client-id", help="Client ID (auto-generated if omitted)")
    join_parser.add_argument("--log-level", default="INFO", help="Logging level")
    join_parser.add_argument("--dashboard-port", type=int, default=5050, help="Dashboard port (0=disabled)")

    # ── status ────────────────────────────────────────────────────────────
    status_parser = subparsers.add_parser("status", help="Show client state from the local database")
    status_parser.add_argument("--db-path", default="fl_client_state.db", help="SQLite DB path")
    status_parser.add_argument("--client-id", help="Client ID to inspect")

    # ── clear ─────────────────────────────────────────────────────────────
    clear_parser = subparsers.add_parser("clear", help="Clear persisted state")
    clear_parser.add_argument("--db-path", default="fl_client_state.db", help="SQLite DB path")
    clear_parser.add_argument("--client-id", required=True, help="Client ID to clear")

    # ── ui ────────────────────────────────────────────────────────────────
    ui_parser = subparsers.add_parser("ui", help="Launch the local web GUI")
    ui_parser.add_argument("--port", "-p", type=int, default=5050, help="Port for the UI server")
    ui_parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "create-room":
        cmd_create_room(args)
    elif args.command == "join-room":
        cmd_join_room(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "ui":
        cmd_ui(args)


def cmd_run(args: argparse.Namespace) -> None:
    """Run the FL client with a YAML config."""
    if args.config:
        config = FLConfig.from_yaml(args.config)
    else:
        config = FLConfig()

    # CLI overrides
    if args.server_url:
        config.server_url = args.server_url
    if args.room_id:
        config.room_id = args.room_id
    if args.data:
        config.data_path = args.data
    if args.client_id:
        config.client_id = args.client_id
    if hasattr(args, "reconnect_max_retries") and args.reconnect_max_retries is not None:
        config.reconnect_max_retries = args.reconnect_max_retries
    config.log_level = args.log_level
    config.dashboard_port = args.dashboard_port

    config.validate_for_training()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client = FLClient.from_config(config)
    client.run()


def cmd_create_room(args: argparse.Namespace) -> None:
    """Create a new FL room."""
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build model config
    model_config: Dict[str, Any] = {"model_type": args.model_type}
    if args.input_size:
        model_config["input_size"] = args.input_size
    if args.hidden_size:
        model_config["hidden_size"] = args.hidden_size
    if args.output_size:
        model_config["output_size"] = args.output_size

    # Load schema
    data_schema: Dict[str, Any] = {"columns": [], "target_column": "label"}
    if args.schema_file:
        with open(args.schema_file, "r") as f:
            data_schema = json.load(f)

    # Load from YAML config if provided
    training_config: Dict[str, Any] = {"local_epochs": 2, "batch_size": 32, "learning_rate": 0.001}
    if args.config:
        with open(args.config, "r") as f:
            cfg_data: Dict[str, Any] = yaml.safe_load(f) or {}
        model_config = cfg_data.get("model_config", model_config)
        data_schema = cfg_data.get("data_schema", data_schema)
        training_config = cfg_data.get("training_config", training_config)

    client = FLClient(server_url=args.server_url)
    result = client.create_room(
        model_config=model_config,
        data_schema=data_schema,
        training_config=training_config,
        room_name=args.room_name,
    )

    room = result.get("room", {})
    print(f"\n✅ Room created successfully!")
    print(f"   Room ID:     {room.get('id', 'N/A')}")
    print(f"   Invite Code: {room.get('invite_code', 'N/A')}")
    print(f"   Model Type:  {model_config.get('model_type', 'N/A')}")
    print(f"\n   Join with: fl-client join-room {room.get('id', 'ROOM_ID')} --data ./data.csv")


def cmd_join_room(args: argparse.Namespace) -> None:
    """Join a room and start training (3-step lifecycle)."""
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    client = FLClient(
        server_url=args.server_url,
        room_id=args.room_id,
        client_id=args.client_id or "",
        dashboard_port=args.dashboard_port,
    )

    # 3-step lifecycle
    print("Step 1/3: Initializing...")
    client.initialize()

    print("Step 2/3: Validating dataset...")
    metadata = client.validate(args.data)
    print(f"   ✅ Dataset valid — {metadata['num_samples']} samples, {metadata.get('num_classes', '?')} classes")

    print("Step 3/3: Starting training...")
    client.start()


def cmd_status(args: argparse.Namespace) -> None:
    """Display persisted client state."""
    from fl_client.storage.db import StateDB

    db = StateDB(args.db_path)

    if args.client_id:
        state = db.load_state(args.client_id)
        if state:
            print(f"Client ID:      {state.client_id}")
            print(f"Room ID:        {state.room_id}")
            print(f"Current Round:  {state.current_round}")
            print(f"Model Version:  {state.model_version}")
            print(f"Num Samples:    {state.num_samples}")
            print(f"Submitted:      {sorted(state.submitted_rounds)}")

            # Show round history
            history = db.get_round_history(state.client_id)
            if history:
                print(f"\nRound History ({len(history)} rounds):")
                for r in history[-10:]:  # Last 10
                    print(f"  R{r['round']}: loss={r['loss']:.4f} time={r['training_time']:.2f}s ΔW={r.get('delta_w', 0):.4f} [{r['status']}]")
        else:
            print(f"No state found for client_id={args.client_id}")
    else:
        print("Specify --client-id to view client state")

    db.close()


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear client state from the database."""
    from fl_client.storage.db import StateDB

    db = StateDB(args.db_path)
    db.clear_state(args.client_id)
    db.close()
    print(f"State cleared for client_id={args.client_id}")


def cmd_ui(args: argparse.Namespace) -> None:
    """Launch the local web GUI."""
    from fl_client.web.bridge import run_ui

    run_ui(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
