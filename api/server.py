from __future__ import annotations

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ETrainAgent
from agent.schedule import build_day_schedule
from agent.scenario import run_scenario
from simulation import build_demo_state


DEMAND_CSV_PATH = ROOT / "data" / "demand_shortage.csv"


def _build_state():
    """build_demo_state()를 만들고, 실제 수요 CSV가 있으면 metadata에 연결한다."""
    state = build_demo_state()
    if DEMAND_CSV_PATH.exists():
        state.metadata["demand_csv_path"] = str(DEMAND_CSV_PATH)
    return state


def _first_float(params: dict, key: str, default: float) -> float:
    values = params.get(key)
    if not values:
        return default
    try:
        return float(values[0])
    except (TypeError, ValueError):
        return default


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
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/api/run-loop":
            try:
                result = ETrainAgent(scenario_count=8, seed=42).run_once(_build_state()).to_dict()
                (ROOT / "dashboard" / "latest_result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self._json(result)
            except Exception as exc:
                self._json({"error": type(exc).__name__, "message": str(exc)}, status=500)
            return

        if path == "/api/state":
            self._json(_build_state().snapshot())
            return

        if path == "/api/day-schedule":
            # operator_v4 대시보드용: 하루 여러 시점의 실제 계산 결과를 상행/하행 편성 리스트로 반환
            try:
                plans = build_day_schedule(_build_state())
                self._json({"plans": plans})
            except Exception as exc:
                self._json({"error": type(exc).__name__, "message": str(exc)}, status=500)
            return

        if path == "/api/scenario":
            # operator_v4 대시보드의 what-if 시뮬레이터: 슬라이더 값을 실제 forecast/MILP에 반영
            try:
                proposal = run_scenario(
                    _build_state(),
                    curtailment_pct=_first_float(params, "curt", 0.0),
                    demand_pct=_first_float(params, "demand", 0.0),
                    cargo_ton=_first_float(params, "cargo_ton", 72.0),
                    delay_min=_first_float(params, "delay", 0.0),
                )
                self._json(proposal)
            except Exception as exc:
                self._json({"error": type(exc).__name__, "message": str(exc)}, status=500)
            return

        if path == "/":
            self.path = "/index.html"
        return super().do_GET()


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), ETrainHandler)
    print(f"E-TRAIN RE:LOOP dashboard: http://{host}:{port}")
    print("API: /api/run-loop, /api/state, /api/day-schedule, /api/scenario")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
