from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ETrainAgent
from simulation import build_demo_state


class ETrainHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "dashboard"), **kwargs)

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/run-loop":
            try:
                result = ETrainAgent(scenario_count=8, seed=42).run_once(build_demo_state()).to_dict()
                (ROOT / "dashboard" / "latest_result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self._json(result)
            except Exception as exc:
                self._json({"error": type(exc).__name__, "message": str(exc)}, status=500)
            return
        if path == "/api/state":
            self._json(build_demo_state().snapshot())
            return
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), ETrainHandler)
    print(f"E-TRAIN RE:LOOP dashboard: http://{host}:{port}")
    print("API: /api/run-loop, /api/state")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
