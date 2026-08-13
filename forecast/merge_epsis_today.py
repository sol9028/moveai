# -*- coding: utf-8 -*-
"""
merge_epsis_today.py
EPSIS(전력통계정보시스템) "실시간 전력수급" 페이지에서 다운받은
오늘자 CSV를 읽어서, 시간 단위로 집계한 뒤 capital_region_demand.csv 끝에 이어붙인다.

사용법:
    python -m forecast.merge_epsis_today "다운받은파일경로.csv"
"""

from __future__ import annotations
import sys
import pandas as pd

from . import config


def parse_epsis_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="euc-kr")
    df["timestamp"] = pd.to_datetime(df["일시"])
    df["demand_mw"] = pd.to_numeric(df["현재부하(MW)"], errors="coerce")
    return df[["timestamp", "demand_mw"]].dropna()


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        df.set_index("timestamp")["demand_mw"]
        .resample("h")
        .mean()
        .reset_index()
    )
    return hourly


def merge_into_capital_dataset(csv_path: str, save: bool = True) -> pd.DataFrame:
    raw = parse_epsis_csv(csv_path)
    hourly_national = aggregate_to_hourly(raw)
    hourly_national["capital_demand_mw"] = (
        hourly_national["demand_mw"] * config.CAPITAL_REGION_RATIO
    )

    print(f"[OK] {len(hourly_national)}개 시간대 집계 완료")
    print(hourly_national[["timestamp", "capital_demand_mw"]].to_string(index=False))

    existing = pd.read_csv(config.CAPITAL_DEMAND_PATH, parse_dates=["timestamp"])
    combined = pd.concat(
        [existing, hourly_national[["timestamp", "capital_demand_mw"]]],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    if save:
        combined.to_csv(config.CAPITAL_DEMAND_PATH, index=False, encoding="utf-8-sig")
        print(f"\n[OK] {config.CAPITAL_DEMAND_PATH} 갱신 완료")
        print(f"     마지막 시점: {combined['timestamp'].max()}")
        print(f"     전체 행 수: {len(combined)}")

    return combined


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python -m forecast.merge_epsis_today <CSV파일경로>")
        sys.exit(1)
    merge_into_capital_dataset(sys.argv[1])