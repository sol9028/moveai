"""예측모델과 최적화모델 사이의 데이터 계약과 검증."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ForecastBundle:
    days: list[str]
    curtailment: pd.DataFrame
    demand: pd.DataFrame
    cargo: pd.DataFrame
    smp: pd.DataFrame


SCHEMAS = {
    "curtailment": {"date", "node", "hour", "chargeable_mwh"},
    "demand": {"date", "node", "hour", "predicted_deficit_mwh"},
    "cargo": {"date", "node", "cargo_type", "predicted_cargo_ton"},
    "smp": {"date", "hour", "predicted_smp_krw_per_kwh"},
}


def consecutive_days(start_date: str, horizon: int = 7) -> list[str]:
    start = date.fromisoformat(start_date)
    return [(start + timedelta(days=i)).isoformat() for i in range(horizon)]


def _read(path: str | Path, name: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} 예측파일이 없습니다: {p}")
    df = pd.read_csv(p, encoding="utf-8-sig")
    missing = SCHEMAS[name] - set(df.columns)
    if missing:
        raise ValueError(f"{name} 필수 컬럼 누락: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.date.astype(str)
    return df


def _validate_dates(df: pd.DataFrame, name: str, expected: list[str]) -> None:
    actual = set(df["date"].unique())
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    if missing or extra:
        raise ValueError(f"{name} 날짜 불일치 — 누락={missing}, 계획기간 밖={extra}")


def _validate_numeric(df: pd.DataFrame, name: str, columns: list[str]) -> None:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if (df[col] < 0).any():
            raise ValueError(f"{name}.{col}에 음수가 있습니다.")


def load_forecasts(params: dict, start_date: str, base_dir: Path) -> ForecastBundle:
    horizon = int(params["planning"]["horizon_days"])
    if horizon != 7:
        raise ValueError("현재 모델은 연속 7일 계획 전용입니다.")
    expected = consecutive_days(start_date, horizon)
    cfg = params["data"]
    resolve = lambda key: base_dir / cfg[key]
    cur = _read(resolve("curtailment_csv"), "curtailment")
    dem = _read(resolve("demand_csv"), "demand")
    car = _read(resolve("cargo_csv"), "cargo")
    smp = _read(resolve("smp_csv"), "smp")
    for name, frame in (("curtailment", cur), ("demand", dem), ("cargo", car), ("smp", smp)):
        _validate_dates(frame, name, expected)
    _validate_numeric(cur, "curtailment", ["hour", "chargeable_mwh"])
    _validate_numeric(dem, "demand", ["hour", "predicted_deficit_mwh"])
    _validate_numeric(car, "cargo", ["predicted_cargo_ton"])
    _validate_numeric(smp, "smp", ["hour", "predicted_smp_krw_per_kwh"])
    for name, frame in (("curtailment", cur), ("demand", dem), ("smp", smp)):
        if not frame["hour"].between(0, 23).all():
            raise ValueError(f"{name}.hour는 0~23이어야 합니다.")
    if cur.duplicated(["date", "node", "hour"]).any():
        raise ValueError("curtailment에 date-node-hour 중복이 있습니다.")
    if dem.duplicated(["date", "node", "hour"]).any():
        raise ValueError("demand에 date-node-hour 중복이 있습니다.")
    if smp.duplicated(["date", "hour"]).any():
        raise ValueError("smp에 date-hour 중복이 있습니다.")
    if car.duplicated(["date", "node", "cargo_type"]).any():
        raise ValueError("cargo에 date-node-cargo_type 중복이 있습니다.")
    return ForecastBundle(expected, cur, dem, car, smp)

