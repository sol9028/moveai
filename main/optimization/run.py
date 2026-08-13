"""CLI 진입점. 연속 7일 입력 생성(선택) → 최적화 → dashboard JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from forecast_adapter import build_week
from optimization import run
from verify import verify_result


def _first_existing(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{label} 결과를 찾지 못했습니다: {[str(x) for x in candidates]}")


def write_forecast_manifest(root: Path) -> Path:
    """예측 py가 만든 모든 CSV를 대시보드용 목록으로 등록한다."""
    search_dirs = [root.parent / "output", root.parent / "forecast",
                   root.parent / "forecast" / "output",
                   root.parent / "forecast" / "outputs"]
    files = []
    for folder in search_dirs:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            try:
                columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
                files.append({"name": path.name, "path": str(path.resolve()),
                              "size_bytes": path.stat().st_size, "columns": columns})
            except Exception as exc:
                files.append({"name": path.name, "path": str(path.resolve()),
                              "size_bytes": path.stat().st_size, "read_error": str(exc)})
    target = root / "outputs" / "forecast_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"forecast_csv_files": files}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    return target


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="E-Train 연속 7일 최적화")
    ap.add_argument("--start-date", required=True, help="계획 시작일 YYYY-MM-DD")
    ap.add_argument("--params", default=str(root / "params.yaml"))
    ap.add_argument("--output", default=str(root / "outputs" / "dashboard_result.json"))
    ap.add_argument("--initial-state", default=None, help="직전 계획의 ending_state JSON")
    ap.add_argument("--build-week", action="store_true",
                    help="main/forecast 산출물에서 보수적 연속 7일 입력 생성")
    ap.add_argument("--curtailment-source")
    ap.add_argument("--demand-source")
    ap.add_argument("--cargo-source")
    ap.add_argument("--smp-source")
    ap.add_argument("--fixed-smp", type=float, default=100.0,
                    help="SMP 파일이 없을 때 적용할 보수적 고정 SMP(원/kWh)")
    args = ap.parse_args()
    if args.build_week:
        forecast = root.parent / "forecast"
        cur = Path(args.curtailment_source) if args.curtailment_source else root.parent / "output" / "supply_forecast.csv"
        dem = Path(args.demand_source) if args.demand_source else _first_existing([
            forecast / "outputs" / "shortage_forecast.csv",
            root.parent / "output" / "shortage_forecast.csv",
            forecast / "outputs" / "top_shortage_days_hourly.csv",
        ], "수요")
        car = Path(args.cargo_source) if args.cargo_source else root.parent / "output" / "cargo_forecast.csv"
        build_week(cur, dem, car, Path(args.smp_source) if args.smp_source else None,
                   root / "data" / "forecast", args.start_date,
                   fixed_smp_value=args.fixed_smp)
    result = run(args.params, args.start_date, args.output, args.initial_state)
    verify_result(result, Path(args.params))
    s = result["summary"]
    print(f"[{result['status']}] {result['planning_start_date']} ~ {result['planning_end_date']}")
    print(f"왕복 {s['round_trips']}회 | 충전 {s['charged_mwh']:.1f}MWh | "
          f"공급 {s['discharged_mwh']:.1f}MWh | 화물 {s['cargo_cars']}화차/{s['cargo_ton']:.1f}톤")
    print(f"주간 순편익 {s['net_profit_krw']/1e6:.2f}백만원")
    print(f"대시보드 결과: {Path(args.output).resolve()}")
    print(f"예측 CSV 목록: {write_forecast_manifest(root).resolve()}")


if __name__ == "__main__":
    main()
