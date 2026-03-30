"""FastAPI bridge — connects the browser UI to the Python Client SDK.

All training, validation, and room management logic stays in the SDK.
The UI never talks directly to the FL server.

Usage:
    fl-client ui --port 5050
    # Open http://localhost:5050
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import torch.nn as nn

from fl_client.core.client import FLClient
from fl_client.training.dataset import DatasetValidationError

logger = logging.getLogger(__name__)

# ── Directory for static files ────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"


# ── Pydantic models ──────────────────────────────────────────────────────────

class CreateRoomRequest(BaseModel):
    server_url: str = "ws://localhost:8080"
    room_name: str = "fl-room"
    model_config_data: Dict[str, Any] = {}
    data_schema: Dict[str, Any] = {}
    training_config: Dict[str, Any] = {}


class JoinRoomRequest(BaseModel):
    server_url: str = "ws://localhost:8080"
    room_id: str = ""
    invite_code: str = ""
    data_path: str = ""
    client_id: str = ""


class InitializeRequest(BaseModel):
    server_url: str = "ws://localhost:8080"
    room_id: str = ""


class ValidateRequest(BaseModel):
    data_path: str = ""


class SettingsUpdate(BaseModel):
    server_url: str = ""
    client_id: str = ""
    log_level: str = "INFO"


# ── Shared state ─────────────────────────────────────────────────────────────

class AppState:
    """Shared mutable state for the bridge."""

    def __init__(self) -> None:
        self.client: Optional[FLClient] = None
        self.server_url: str = "ws://localhost:8080"
        self.room_id: str = ""
        self.room_info: Dict[str, Any] = {}
        self.dataset_metadata: Optional[Dict[str, Any]] = None
        self.data_path: str = ""
        # Unique db_path per bridge instance to avoid SQLite lock conflicts
        self.db_path: str = f"fl_client_state_{uuid.uuid4().hex[:8]}.db"
        self.training_thread: Optional[threading.Thread] = None
        self.training_active: bool = False
        self.initialized: bool = False
        self.validated: bool = False
        self.logs: deque = deque(maxlen=500)
        self.metrics: List[Dict[str, Any]] = []
        self.status: str = "idle"  # idle | initialized | validated | training | stopped
        self.error: str = ""
        self.connected_ws: List[WebSocket] = []

    def add_log(self, level: str, message: str) -> None:
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        self.logs.append(entry)
        # Push to WebSocket clients
        asyncio.get_event_loop().call_soon_threadsafe(
            lambda: asyncio.ensure_future(self._broadcast(entry))
        ) if self.connected_ws else None

    async def _broadcast(self, data: Dict) -> None:
        dead = []
        for ws in self.connected_ws:
            try:
                await ws.send_json({"type": "log", "data": data})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connected_ws.remove(ws)

    async def broadcast_metrics(self) -> None:
        dead = []
        for ws in self.connected_ws:
            try:
                await ws.send_json({
                    "type": "metrics",
                    "data": {
                        "metrics": self.metrics,
                        "status": self.status,
                        "room_id": self.room_id,
                        "room_info": self.room_info,
                        "dataset_metadata": self.dataset_metadata,
                    },
                })
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.connected_ws.remove(ws)


state = AppState()


# ── Custom log handler to capture SDK logs ────────────────────────────────────

class BridgeLogHandler(logging.Handler):
    """Captures all fl_client logs and pushes to the UI."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            state.add_log(record.levelname, msg)

            # Capture metrics from training updates
            if "Update sent" in msg and "loss=" in msg:
                try:
                    parts = msg.split("—")[1] if "—" in msg else ""
                    entry: Dict[str, Any] = {"timestamp": time.time()}
                    for kv in parts.split():
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            try:
                                entry[k] = float(v.rstrip("s"))
                            except ValueError:
                                entry[k] = v
                    if "round" in entry:
                        state.metrics.append(entry)
                except Exception:
                    pass
        except Exception:
            pass


# ── Legacy UI Model Builder ───────────────────────────────────────────────────

def _build_model_from_ui_config(config: Dict[str, Any]) -> "nn.Module":
    """Fallback builder for the React UI since the core SDK now requires BYOM."""
    model_type = config.get("model_type", "mlp").lower()
    in_dim = config.get("input_dim", 10)
    out_dim = config.get("output_dim", 2)
    hidden_layers = config.get("hidden_layers", [128, 64])

    layers = []
    
    if model_type == "mlp":
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)
        
    elif model_type == "cnn":
        ch = 1
        spatial = 28
        convs = []
        for i in range(2):
            out_ch = 32 * (2**i)
            convs.extend([
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            ch = out_ch
            spatial //= 2
        flat_size = ch * spatial * spatial
        
        class CNNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(*convs)
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flat_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, out_dim)
                )
            def forward(self, x): return self.classifier(self.features(x))
        return CNNModel()
        
    elif model_type == "rnn":
        hidden_size = hidden_layers[0] if hidden_layers else 64
        class RNNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(in_dim, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, out_dim)
            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(out[:, -1, :])
        return RNNModel()
        
    raise ValueError(f"Unknown model_type from UI: {model_type}")

# ── FastAPI app ───────────────────────────────────────────────────────────────

def _cleanup_db(db_path: str) -> None:
    """Helper to remove the SQLite database and its WAL/SHM companion files."""
    for ext in ["", "-shm", "-wal"]:
        path = f"{db_path}{ext}"
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logging.getLogger("fl_client").warning(f"Failed to delete {path}: {e}")

def create_app() -> FastAPI:
    app = FastAPI(title="DistFL Client UI", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up the SQLite database when the bridge server shuts down."""
        _cleanup_db(state.db_path)

    # Install log handler
    handler = BridgeLogHandler()
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger("fl_client").addHandler(handler)

    # ── Static assets (JS/CSS bundles) ──────────────────────────────────
    # Mount /assets for Vite build output before page routes
    assets_dir = STATIC_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # Legacy static mount for backwards compatibility
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


    # ── REST API ──────────────────────────────────────────────────────────

    @app.get("/api/status")
    async def get_status():
        return {
            "status": state.status,
            "server_url": state.server_url,
            "room_id": state.room_id,
            "room_info": state.room_info,
            "initialized": state.initialized,
            "validated": state.validated,
            "training_active": state.training_active,
            "dataset_metadata": state.dataset_metadata,
            "error": state.error,
            "client_id": state.client._config.client_id if state.client else "",
        }

    @app.get("/api/metrics")
    async def get_metrics():
        metrics = state.metrics
        return {
            "metrics": metrics,
            "total_rounds": len(metrics),
            "latest_loss": metrics[-1].get("loss") if metrics else None,
            "latest_round": metrics[-1].get("round") if metrics else 0,
        }

    @app.get("/api/logs")
    async def get_logs():
        return {"logs": list(state.logs)}

    @app.post("/api/create-room")
    async def create_room(req: CreateRoomRequest):
        try:
            state.server_url = req.server_url
            state.error = ""
            state.add_log("INFO", f"Creating room '{req.room_name}'...")

            client = FLClient(server_url=req.server_url, log_level="DEBUG", db_path=state.db_path)

            # Build Python model object from UI JSON config
            py_model = _build_model_from_ui_config(req.model_config_data)

            # Run in thread to avoid asyncio.run() conflict with uvicorn's loop
            result = await asyncio.to_thread(
                client.create_room,
                model=py_model,
                data_schema=req.data_schema,
                training_config=req.training_config,
                room_name=req.room_name,
                model_config=req.model_config_data,
            )

            room = result.get("room", {})
            state.client = client
            state.room_id = room.get("id", "")
            state.room_info = {
                "room_id": room.get("id"),
                "room_name": req.room_name,
                "invite_code": room.get("invite_code"),
                "model_config": req.model_config_data,
                "data_schema": req.data_schema,
                "training_config": req.training_config,
                "state": room.get("state", "waiting"),
            }
            state.initialized = True
            state.status = "initialized"

            state.add_log("INFO", f"✅ Room created — ID={state.room_id}")
            return {"success": True, "room": room}

        except Exception as e:
            state.error = str(e)
            state.add_log("ERROR", f"Create room failed: {e}")
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.post("/api/initialize")
    async def initialize(req: InitializeRequest):
        try:
            state.server_url = req.server_url
            state.room_id = req.room_id
            state.error = ""
            state.add_log("INFO", f"Initializing for room {req.room_id}...")

            client = FLClient(
                server_url=req.server_url,
                room_id=req.room_id,
                log_level="DEBUG",
                db_path=state.db_path,
            )

            # Pre-fetch the model config so we can instantiate the PyTorch model for the UI
            import httpx
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(f"{client._config.server_http_url}/room_info?room_id={req.room_id}")
                resp.raise_for_status()
                room_data = resp.json()
            model_config = room_data.get("model_config", {})
            try:
                py_model = _build_model_from_ui_config(model_config)
            except Exception as e:
                py_model = None
                state.add_log("WARNING", f"Could not restore UI model: {e}")

            # Run in thread to avoid asyncio.run() conflict with uvicorn's loop
            def _initialize():
                client.initialize(model=py_model)
            
            await asyncio.to_thread(_initialize)

            state.client = client
            state.initialized = True
            state.status = "initialized"

            # Extract room info
            state.room_info = {
                "room_id": req.room_id,
                "model_config": client._model_config or {},
                "data_schema": client._data_schema or {},
                "training_config": client._training_config or {},
            }

            state.add_log("INFO", "✅ Initialization complete")
            return {"success": True, "room_info": state.room_info}

        except Exception as e:
            state.error = str(e)
            state.add_log("ERROR", f"Initialize failed: {e}")
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.post("/api/validate")
    async def validate_dataset(req: ValidateRequest):
        try:
            if not state.client or not state.initialized:
                return JSONResponse(status_code=400, content={"error": "Not initialized. Call initialize first."})

            state.data_path = req.data_path
            state.error = ""
            state.add_log("INFO", f"Validating dataset: {req.data_path}")

            metadata = state.client.validate(req.data_path)
            state.dataset_metadata = metadata
            state.validated = True
            state.status = "validated"

            state.add_log("INFO", f"✅ Dataset valid — {metadata['num_samples']} samples")
            return {"success": True, "metadata": metadata}

        except DatasetValidationError as e:
            state.error = str(e)
            state.add_log("ERROR", f"Validation failed: {e.errors}")
            return JSONResponse(status_code=400, content={"error": str(e), "errors": e.errors})

        except Exception as e:
            state.error = str(e)
            state.add_log("ERROR", f"Validation failed: {e}")
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.post("/api/start-training")
    async def start_training():
        try:
            if not state.client:
                return JSONResponse(status_code=400, content={"error": "No client. Initialize first."})
            if not state.validated:
                return JSONResponse(status_code=400, content={"error": "Dataset not validated."})
            if state.training_active:
                return JSONResponse(status_code=400, content={"error": "Training already active."})

            state.error = ""
            state.training_active = True
            state.status = "training"
            state.add_log("INFO", "🚀 Starting training...")

            def _train():
                try:
                    state.client.start()
                except Exception as e:
                    state.add_log("ERROR", f"Training error: {e}")
                finally:
                    state.training_active = False
                    state.status = "stopped"
                    state.add_log("INFO", "Training stopped")

            t = threading.Thread(target=_train, daemon=True)
            t.start()
            state.training_thread = t

            return {"success": True, "message": "Training started"}

        except Exception as e:
            state.error = str(e)
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.post("/api/stop-training")
    async def stop_training():
        if state.client:
            state.client._running = False
            state.training_active = False
            state.status = "stopped"
            state.add_log("INFO", "⏹ Training stopped by user")
        return {"success": True}

    @app.post("/api/join-room")
    async def join_room(req: JoinRoomRequest):
        """Full join flow: initialize → validate → prepare for training."""
        try:
            state.server_url = req.server_url
            state.room_id = req.room_id
            state.data_path = req.data_path
            state.error = ""

            state.add_log("INFO", f"Step 1/2: Initializing room {req.room_id}...")
            client = FLClient(
                server_url=req.server_url,
                room_id=req.room_id,
                invite_code=req.invite_code,
                client_id=req.client_id or None,
                log_level="DEBUG",
                db_path=state.db_path,
            )

            import httpx
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(f"{client._config.server_http_url}/room_info?room_id={req.room_id}")
                resp.raise_for_status()
                room_data = resp.json()
            model_config = room_data.get("model_config", {})
            try:
                py_model = _build_model_from_ui_config(model_config)
            except Exception as e:
                py_model = None
                state.add_log("WARNING", f"Could not restore UI model: {e}")

            # Run in thread to avoid asyncio.run() conflict with uvicorn's loop
            def _initialize():
                client.initialize(model=py_model)
            
            await asyncio.to_thread(_initialize)
            state.client = client
            state.initialized = True

            state.room_info = {
                "room_id": req.room_id,
                "model_config": client._model_config or {},
                "data_schema": client._data_schema or {},
                "training_config": client._training_config or {},
            }

            state.add_log("INFO", f"Step 2/2: Validating dataset {req.data_path}...")
            metadata = client.validate(req.data_path)
            state.dataset_metadata = metadata
            state.validated = True
            state.status = "validated"

            state.add_log("INFO", f"✅ Ready to train — {metadata['num_samples']} samples, {metadata.get('num_classes', '?')} classes")
            return {
                "success": True,
                "room_info": state.room_info,
                "metadata": metadata,
            }

        except DatasetValidationError as e:
            state.error = str(e)
            state.add_log("ERROR", f"Dataset validation failed: {e.errors}")
            return JSONResponse(status_code=400, content={"error": str(e), "errors": e.errors})

        except Exception as e:
            state.error = str(e)
            state.add_log("ERROR", f"Join failed: {e}")
            return JSONResponse(status_code=400, content={"error": str(e)})

    @app.post("/api/reset")
    async def reset():
        """Reset client state."""
        if state.client:
            state.client._running = False
        state.client = None
        state.initialized = False
        state.validated = False
        state.training_active = False
        state.status = "idle"
        state.room_id = ""
        state.room_info = {}
        state.dataset_metadata = None
        state.metrics.clear()
        state.error = ""
        
        # Cleanup old database files and generate a new db_path for the next connection
        _cleanup_db(state.db_path)
        state.db_path = f"fl_client_state_{uuid.uuid4().hex[:8]}.db"
        
        state.add_log("INFO", "🔄 State reset and temporary files cleaned")
        return {"success": True}

    # ── WebSocket for live updates ────────────────────────────────────────

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket):
        await websocket.accept()
        state.connected_ws.append(websocket)
        state.add_log("INFO", "UI WebSocket connected")
        try:
            while True:
                # Send periodic status updates
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "status": state.status,
                        "room_id": state.room_id,
                        "training_active": state.training_active,
                        "initialized": state.initialized,
                        "validated": state.validated,
                        "metrics_count": len(state.metrics),
                    },
                })
                if state.metrics:
                    await websocket.send_json({
                        "type": "metrics",
                        "data": {"metrics": state.metrics},
                    })
                await asyncio.sleep(2)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            if websocket in state.connected_ws:
                state.connected_ws.remove(websocket)

    # ── SPA catch-all — serves React index.html for all page routes ───
    # MUST be registered AFTER all API and WebSocket routes
    @app.get("/{full_path:path}")
    async def spa_catch_all(full_path: str):
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        from fastapi.responses import HTMLResponse
        return HTMLResponse(
            "<h3>DistFL UI — frontend not built</h3>"
            "<p>Run <code>make frontend-build</code> then restart.</p>",
            status_code=503,
        )

    return app


def run_ui(port: int = 5050, open_browser: bool = True) -> None:
    """Start the local UI server."""
    import uvicorn
    import webbrowser

    app = create_app()

    if open_browser:
        def _open():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=_open, daemon=True).start()

    print(f"\n  DistFL Client UI running at http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
