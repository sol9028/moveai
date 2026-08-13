"""연속 7일 예시 예측파일을 생성한다.

실제 운영에서는 이 파일 대신 demand.py, cargo.py, curtailment.py와
SMP 입력 모듈이 동일한 스키마의 CSV를 생성하면 된다.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


CHARGE = ["장성", "남원", "동산", "새만금"]
DISCHARGE = ["안중", "서화성"]
CARGO_NODES = ["안중", "서화성", "천안", "신례원"]
CARGO_TYPES = ["pack_ev", "pack_ess", "blackmass"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default="2026-08-17")
    ap.add_argument("--output-dir", default="data/forecast")
    args = ap.parse_args()
    start = date.fromisoformat(args.start_date)
    days = [(start + timedelta(days=i)).isoformat() for i in range(7)]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    curtail = []
    demand = []
    smp = []
    cargo = []
    for di, dy in enumerate(days):
        day_factor = [0.85, 1.00, 1.15, 0.95, 1.25, 0.80, 1.10][di]
        for h in range(24):
            solar_shape = max(0.0, 1 - abs(h - 13) / 5)
            price_shape = 96 + (38 if 17 <= h <= 20 else 12 if h < 9 else 0)
            smp.append({"date": dy, "hour": h,
                        "predicted_smp_krw_per_kwh": round(price_shape * day_factor, 2)})
            for ni, node in enumerate(CHARGE):
                curtail.append({"date": dy, "node": node, "hour": h,
                                "chargeable_mwh": round(solar_shape * day_factor * (28 + 4 * ni), 3)})
            for ni, node in enumerate(DISCHARGE):
                deficit_shape = 35 if 15 <= h <= 20 else 12 if 10 <= h < 15 else 0
                demand.append({"date": dy, "node": node, "hour": h,
                               "predicted_deficit_mwh": round(deficit_shape * day_factor * (1 + .1 * ni), 3)})
        for ni, node in enumerate(CARGO_NODES):
            for ki, kind in enumerate(CARGO_TYPES):
                base = {"pack_ev": 7.0, "pack_ess": 1.0, "blackmass": 16.0}[kind]
                cargo.append({"date": dy, "node": node, "cargo_type": kind,
                              "predicted_cargo_ton": round(base * (1 + .08 * ni) * (0.8 + .05 * di), 3)})

    pd.DataFrame(curtail).to_csv(out / "curtailment_forecast.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(demand).to_csv(out / "demand_forecast.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cargo).to_csv(out / "cargo_forecast.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(smp).to_csv(out / "smp_forecast.csv", index=False, encoding="utf-8-sig")
    print(f"연속 7일 예시 예측 생성: {days[0]} ~ {days[-1]} → {out}")


if __name__ == "__main__":
    main()

