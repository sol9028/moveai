"""main/forecast 예측 산출물을 연속 7일 최적화 입력으로 변환한다."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import pandas as pd


def expected_days(start_date: str) -> list[str]:
    start = date.fromisoformat(start_date)
    return [(start + timedelta(days=i)).isoformat() for i in range(7)]


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _date_hour(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out and "timestamp" in out:
        ts = pd.to_datetime(out["timestamp"], errors="raise")
        out["date"], out["hour"] = ts.dt.date.astype(str), ts.dt.hour
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.date.astype(str)
    return out


def _top7_remap(df: pd.DataFrame, score_col: str, start_date: str,
                output_cols: list[str], ascending: bool = False):
    """x*(d_i,n,h)=x(s_(i),n,h), s_(i)는 일별 점수 기준 i번째 원천일."""
    scores = df.groupby("date", as_index=False)[score_col].sum().sort_values(
        [score_col, "date"], ascending=[ascending, True]
    )
    if len(scores) < 7:
        raise ValueError(f"상위 7일 생성에 필요한 원천 날짜가 부족합니다: {len(scores)}일")
    chunks, mapping = [], []
    for rank, row in scores.head(7).reset_index(drop=True).iterrows():
        source, target = str(row["date"]), expected_days(start_date)[rank]
        chunk = df[df["date"] == source].copy()
        chunk["date"] = target
        chunks.append(chunk)
        mapping.append({"target_date": target, "source_date": source,
                        "scenario_rank": rank + 1, "daily_score": float(row[score_col])})
    return pd.concat(chunks, ignore_index=True)[output_cols], pd.DataFrame(mapping)


def normalize_curtailment(source: Path, start_date: str):
    df = _date_hour(_read(source)).rename(columns={
        "predicted_curtailment_mwh": "chargeable_mwh", "curtailment_mwh": "chargeable_mwh"})
    required = {"date", "node", "hour", "chargeable_mwh"}
    if missing := required - set(df):
        raise ValueError(f"curtailment 컬럼 누락: {sorted(missing)}")
    return _top7_remap(df, "chargeable_mwh", start_date,
                       ["date", "node", "hour", "chargeable_mwh"])


def normalize_demand(source: Path, start_date: str):
    df = _date_hour(_read(source))
    wide = [c for c in df if c.startswith("ess_need_") and c.endswith("_mw")]
    if wide and "node" not in df:
        df = df.melt(id_vars=["date", "hour"], value_vars=wide,
                     var_name="node", value_name="ess_need_mw")
        df["node"] = df["node"].str.removeprefix("ess_need_").str.removesuffix("_mw")
    for old in ("ess_need_mw", "predicted_shortage_mwh", "deficit_mwh", "shortage_mw"):
        if old in df and "predicted_deficit_mwh" not in df:
            df = df.rename(columns={old: "predicted_deficit_mwh"})
    required = {"date", "node", "hour", "predicted_deficit_mwh"}
    if missing := required - set(df):
        raise ValueError(f"demand 컬럼 누락: {sorted(missing)}")
    return _top7_remap(df, "predicted_deficit_mwh", start_date,
                       ["date", "node", "hour", "predicted_deficit_mwh"])


def normalize_cargo(source: Path, start_date: str, trips_per_week: int = 2):
    df = _date_hour(_read(source))
    if "predicted_cargo_ton" in df and "cargo_ton" not in df:
        df = df.rename(columns={"predicted_cargo_ton": "cargo_ton"})
    required = {"date", "node", "cargo_type", "cargo_ton"}
    if missing := required - set(df):
        raise ValueError(f"cargo 컬럼 누락: {sorted(missing)}")
    daily = df.groupby(["date", "node", "cargo_type"], as_index=False)["cargo_ton"].sum()
    daily["predicted_cargo_ton"] = daily["cargo_ton"] / (7.0 / trips_per_week)
    return _top7_remap(daily, "predicted_cargo_ton", start_date,
                       ["date", "node", "cargo_type", "predicted_cargo_ton"])


def normalize_smp(source: Path, start_date: str):
    df = _date_hour(_read(source)).rename(columns={
        "smp_krw_per_kwh": "predicted_smp_krw_per_kwh", "smp": "predicted_smp_krw_per_kwh"})
    required = {"date", "hour", "predicted_smp_krw_per_kwh"}
    if missing := required - set(df):
        raise ValueError(f"SMP 컬럼 누락: {sorted(missing)}")
    # 수익 과대평가 방지: 일평균 SMP가 낮은 7일을 선택한다.
    return _top7_remap(df, "predicted_smp_krw_per_kwh", start_date,
                       ["date", "hour", "predicted_smp_krw_per_kwh"], ascending=True)


def fixed_smp(start_date: str, value: float):
    if value < 0:
        raise ValueError("고정 SMP는 0 이상이어야 합니다.")
    rows = [{"date": d, "hour": h, "predicted_smp_krw_per_kwh": value}
            for d in expected_days(start_date) for h in range(24)]
    mapping = pd.DataFrame({
        "target_date": expected_days(start_date), "source_date": ["fixed"] * 7,
        "scenario_rank": list(range(1, 8)), "daily_score": [value] * 7,
    })
    return pd.DataFrame(rows), mapping


def build_week(curtailment: Path, demand: Path, cargo: Path, smp: Path | None,
               output_dir: Path, start_date: str, trips_per_week: int = 2,
               fixed_smp_value: float = 100.0) -> None:
    builders = {
        "curtailment": normalize_curtailment(curtailment, start_date),
        "demand": normalize_demand(demand, start_date),
        "cargo": normalize_cargo(cargo, start_date, trips_per_week),
        "smp": normalize_smp(smp, start_date) if smp else fixed_smp(start_date, fixed_smp_value),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = []
    for name, (frame, mapping) in builders.items():
        frame.to_csv(output_dir / f"{name}_forecast.csv", index=False, encoding="utf-8-sig")
        mapping.insert(0, "forecast", name)
        maps.append(mapping)
    pd.concat(maps, ignore_index=True).to_csv(
        output_dir / "scenario_mapping.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    root = Path(__file__).resolve().parent
    forecast = root.parent / "forecast"
    ap = argparse.ArgumentParser(description="상위 7일을 연속 스트레스 주간으로 변환")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--curtailment", default=str(root.parent / "output" / "supply_forecast.csv"))
    ap.add_argument("--demand", default=str(forecast / "outputs" / "shortage_forecast.csv"))
    ap.add_argument("--cargo", default=str(root.parent / "output" / "cargo_forecast.csv"))
    ap.add_argument("--smp", help="시간별 SMP CSV. 없으면 --fixed-smp 사용")
    ap.add_argument("--fixed-smp", type=float, default=100.0,
                    help="SMP 파일이 없을 때의 보수적 고정값(원/kWh)")
    ap.add_argument("--output-dir", default=str(root / "data" / "forecast"))
    ap.add_argument("--trips-per-week", type=int, default=2)
    args = ap.parse_args()
    build_week(Path(args.curtailment), Path(args.demand), Path(args.cargo),
               Path(args.smp) if args.smp else None, Path(args.output_dir),
               args.start_date, args.trips_per_week, args.fixed_smp)
    print(f"연속 7일 입력 생성 완료: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
