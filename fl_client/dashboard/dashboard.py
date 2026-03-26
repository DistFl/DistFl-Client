"""Lightweight local web dashboard for FL training metrics.

Serves a self-contained HTML page with live-updating charts
(loss vs round, training time, participation, ΔW) using Chart.js via CDN.
Auto-refreshes via polling every 3 seconds.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricsDashboard:
    """In-process metrics dashboard served via http.server.

    Runs in a daemon thread so it doesn't block the training loop.

    Usage::

        dash = MetricsDashboard(port=5050, client_id="abc", room_id="R1")
        dash.start()
        dash.update(metrics_history)
        dash.stop()
    """

    def __init__(
        self,
        port: int = 5050,
        client_id: str = "",
        room_id: str = "",
    ) -> None:
        self._port = port
        self._client_id = client_id
        self._room_id = room_id
        self._metrics: List[Dict[str, Any]] = []
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def update(self, metrics: List[Dict[str, Any]]) -> None:
        """Update the metrics data (thread-safe)."""
        with self._lock:
            self._metrics = list(metrics)

    def get_metrics_json(self) -> str:
        """Get current metrics as JSON."""
        with self._lock:
            return json.dumps({
                "client_id": self._client_id,
                "room_id": self._room_id,
                "rounds": self._metrics,
                "total_rounds": len(self._metrics),
                "latest_loss": self._metrics[-1]["loss"] if self._metrics else None,
                "latest_round": self._metrics[-1]["round"] if self._metrics else 0,
            })

    def start(self) -> None:
        """Start the dashboard server in a daemon thread."""
        dashboard = self

        class DashHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/metrics":
                    data = dashboard.get_metrics_json().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.write(data)
                else:
                    html = dashboard._generate_html().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.write(html)

            def write(self, data):
                try:
                    self.wfile.write(data)
                except BrokenPipeError:
                    pass

            def log_message(self, fmt, *args):
                pass  # Suppress HTTP logs

        try:
            self._server = HTTPServer(("0.0.0.0", self._port), DashHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
        except Exception as e:
            logger.warning("Dashboard failed to start: %s", e)

    def stop(self) -> None:
        """Stop the dashboard server."""
        server = self._server
        if server is not None:
            self._server = None
            server.shutdown()

    def _generate_html(self) -> str:
        """Generate the self-contained dashboard HTML."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FL Dashboard — {self._client_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0f1117; color:#e8eaed; font-family:'Inter',sans-serif; padding:20px; }}
h1 {{ font-size:22px; background:linear-gradient(135deg,#6366f1,#a78bfa);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:6px; }}
.subtitle {{ color:#9ca3af; font-size:13px; margin-bottom:20px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.card {{ background:#222640; border:1px solid #2d3250; border-radius:12px; padding:16px; }}
.card-title {{ font-size:12px; text-transform:uppercase; letter-spacing:1px; color:#9ca3af; margin-bottom:12px; }}
.metrics-row {{ display:flex; gap:12px; margin-bottom:20px; }}
.metric {{ background:#1a1d2e; border-radius:8px; padding:14px; text-align:center; flex:1; }}
.metric-value {{ font-size:28px; font-weight:700; color:#6366f1; }}
.metric-label {{ font-size:11px; color:#9ca3af; margin-top:4px; text-transform:uppercase; }}
canvas {{ max-height:220px; }}
#logs {{ background:#1a1d2e; border-radius:8px; padding:12px; max-height:200px; overflow-y:auto;
         font[:12px; font-family:'SF Mono',monospace; line-height:1.6; }}
.log {{ padding:2px 0; color:#60a5fa; border-bottom:1px solid rgba(45,50,80,0.5); }}
.full {{ grid-column:1/-1; }}
</style>
</head>
<body>
<h1>📊 FL Training Dashboard</h1>
<p class="subtitle">Client: {self._client_id} | Room: {self._room_id}</p>

<div class="metrics-row">
  <div class="metric"><div class="metric-value" id="m-round">0</div><div class="metric-label">Round</div></div>
  <div class="metric"><div class="metric-value" id="m-loss">—</div><div class="metric-label">Loss</div></div>
  <div class="metric"><div class="metric-value" id="m-total">0</div><div class="metric-label">Participated</div></div>
  <div class="metric"><div class="metric-value" id="m-dw">—</div><div class="metric-label">ΔW</div></div>
</div>

<div class="grid">
  <div class="card">
    <div class="card-title">Loss vs Round</div>
    <canvas id="lossChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Training Time (s)</div>
    <canvas id="timeChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Weight Update Magnitude (ΔW)</div>
    <canvas id="dwChart"></canvas>
  </div>
  <div class="card">
    <div class="card-title">Samples per Round</div>
    <canvas id="samplesChart"></canvas>
  </div>
  <div class="card full">
    <div class="card-title">Training Logs</div>
    <div id="logs"></div>
  </div>
</div>

<script>
const chartOpts = {{
  responsive:true, animation:{{duration:300}},
  scales:{{ x:{{ticks:{{color:'#9ca3af'}}}}, y:{{ticks:{{color:'#9ca3af'}}}} }},
  plugins:{{ legend:{{display:false}} }}
}};

function makeChart(id, color) {{
  return new Chart(document.getElementById(id), {{
    type:'line', data:{{ labels:[], datasets:[{{ data:[], borderColor:color, backgroundColor:color+'33',
      fill:true, tension:0.3, pointRadius:3 }}] }}, options:chartOpts
  }});
}}

const lossChart = makeChart('lossChart','#ef4444');
const timeChart = makeChart('timeChart','#22c55e');
const dwChart = makeChart('dwChart','#f59e0b');
const samplesChart = makeChart('samplesChart','#6366f1');

async function refresh() {{
  try {{
    const r = await fetch('/api/metrics');
    const d = await r.json();
    const rounds = d.rounds || [];

    document.getElementById('m-round').textContent = d.latest_round || 0;
    document.getElementById('m-loss').textContent = d.latest_loss != null ? d.latest_loss.toFixed(4) : '—';
    document.getElementById('m-total').textContent = d.total_rounds;

    if (rounds.length > 0) {{
      const last = rounds[rounds.length-1];
      document.getElementById('m-dw').textContent = (last.delta_w||0).toFixed(4);
    }}

    const labels = rounds.map(r => r.round);
    lossChart.data.labels = labels;
    lossChart.data.datasets[0].data = rounds.map(r => r.loss);
    lossChart.update();

    timeChart.data.labels = labels;
    timeChart.data.datasets[0].data = rounds.map(r => r.training_time);
    timeChart.update();

    dwChart.data.labels = labels;
    dwChart.data.datasets[0].data = rounds.map(r => r.delta_w || 0);
    dwChart.update();

    samplesChart.data.labels = labels;
    samplesChart.data.datasets[0].data = rounds.map(r => r.num_samples);
    samplesChart.update();

    const logs = document.getElementById('logs');
    logs.innerHTML = rounds.map(r =>
      `<div class="log">Round ${{r.round}}: loss=${{(r.loss||0).toFixed(4)}} time=${{(r.training_time||0).toFixed(2)}}s ΔW=${{(r.delta_w||0).toFixed(4)}}</div>`
    ).join('');
    logs.scrollTop = logs.scrollHeight;
  }} catch(e) {{}}
}}

setInterval(refresh, 3000);
refresh();
</script>
</body>
</html>"""
