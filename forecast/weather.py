# -*- coding: utf-8 -*-
"""
weather.py
기상청 ASOS 시간자료 API(kma_sfctm3.php)에서 수도권 대표 기온을 받아온다.

[중요] 이 스크립트는 apihub.kma.go.kr에 실제 네트워크 요청을 보냅니다.
Claude 샌드박스 환경에서는 이 도메인이 허용 목록에 없어 직접 실행이
안 됩니다 (robots.txt 및 네트워크 정책). 사용자 로컬 환경 / Jupyter에서
authKey를 넣어 직접 실행한 뒤, 결과 CSV(config.WEATHER_PATH)를
data/ 폴더에 넣어주면 이후 파이프라인이 자동으로 인식합니다.

사용법:
    export KMA_AUTH_KEY="발급받은키"
    python -m forecast.weather --start 2024-01-01 --end 2025-12-31

또는 코드에서 직접:
    from forecast.weather import fetch_capital_region_weather
    df = fetch_capital_region_weather("2024-01-01", "2025-12-31", auth_key="...")
"""

from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np

from . import config

# kma_sfctm3.php 응답 컬럼 정의 (고정 스펙)
COL_NAMES = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT", "PR",
    "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT", "SD_HR3", "SD_DAY",
    "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID", "CH_MIN", "CT", "CT_TOP",
    "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD", "TS", "TE_005", "TE_01", "TE_02",
    "TE_03", "ST_SEA", "WH", "BF", "IR", "IX",
]


def _month_ranges(start: str, end: str):
    """'2024-01-01' ~ '2025-12-31' -> 월별 (tm1, tm2) 튜플 리스트 (KMA API 포맷)."""
    months = pd.period_range(start=start, end=end, freq="M")
    ranges = []
    for p in months:
        tm1 = p.start_time.strftime("%Y%m%d0000")
        tm2 = p.end_time.strftime("%Y%m%d2300")
        ranges.append((tm1, tm2))
    return ranges


def fetch_station_temperature(
    stn: str,
    tm1: str,
    tm2: str,
    auth_key: str,
    timeout: int = 30,
) -> pd.DataFrame:
    """단일 지점의 기간 내 시간별 기온(TA)을 받아온다."""
    import requests  # 여기서만 import (샌드박스에 requests 없을 수 있어 지연 로딩)

    params = {"tm1": tm1, "tm2": tm2, "stn": stn, "help": "1", "authKey": auth_key}
    r = requests.get(config.KMA_WEATHER_URL, params=params, timeout=timeout)
    r.raise_for_status()

    lines = r.text.strip().split("\n")
    data_lines = [l for l in lines if l.strip() and not l.startswith("#")]
    if not data_lines:
        return pd.DataFrame(columns=["datetime", "TA", "stn"])

    rows = [l.split() for l in data_lines]
    df = pd.DataFrame(rows, columns=COL_NAMES)
    df = df[["TM", "TA"]].copy()
    df["TM"] = pd.to_datetime(df["TM"], format="%Y%m%d%H%M")
    df["TA"] = pd.to_numeric(df["TA"], errors="coerce").replace(-9, np.nan)
    df["stn"] = stn
    return df.rename(columns={"TM": "datetime"})


def fetch_capital_region_weather(
    start: str,
    end: str,
    auth_key: str | None = None,
    stations: dict | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    수도권 대표 기상(기온)을 여러 지점 평균으로 산출.
    stations 미지정 시 config.WEATHER_STATIONS(서울/인천/수원) 사용.
    """
    auth_key = auth_key or os.environ.get(config.KMA_AUTH_KEY_ENV)
    if not auth_key:
        raise ValueError(
            f"authKey가 필요합니다. 환경변수 {config.KMA_AUTH_KEY_ENV} 설정하거나 "
            "auth_key 인자로 직접 전달하세요."
        )
    stations = stations or config.WEATHER_STATIONS
    month_ranges = _month_ranges(start, end)

    all_station_dfs = []
    for stn, name in stations.items():
        print(f"[{name}({stn})] 수집 시작...")
        chunks = []
        for tm1, tm2 in month_ranges:
            chunk = fetch_station_temperature(stn, tm1, tm2, auth_key)
            chunks.append(chunk)
            print(f"  {tm1[:6]} 완료 ({len(chunk)}행)")
        station_df = pd.concat(chunks, ignore_index=True)
        all_station_dfs.append(station_df)

    merged = pd.concat(all_station_dfs, ignore_index=True)
    # 지점 평균 -> 수도권 대표 기온
    capital_weather = (
        merged.groupby("datetime")["TA"]
        .mean()
        .reset_index()
        .rename(columns={"TA": "TA_capital_avg", "datetime": "timestamp"})
    )

    if save:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        capital_weather.to_csv(config.WEATHER_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] 수도권 기온 데이터 저장: {config.WEATHER_PATH} ({len(capital_weather)}행)")

    return capital_weather


def load_weather_if_exists() -> pd.DataFrame | None:
    """이미 수집된 기상 CSV가 있으면 로드, 없으면 None."""
    if config.WEATHER_PATH.exists():
        df = pd.read_csv(config.WEATHER_PATH, parse_dates=["timestamp"])
        return df
    return None


def make_synthetic_weather_for_testing(capital_demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    [테스트 전용] 실제 authKey 없이 파이프라인 배관(merge/feature) 검증용
    합성 기온 데이터를 생성한다. 실제 예측 정확도와는 무관하며,
    코드 구조가 제대로 도는지만 확인하는 용도.
    실제 운영/제출 전에는 반드시 fetch_capital_region_weather()로 교체할 것.
    """
    ts = capital_demand_df["timestamp"]
    doy = ts.dt.dayofyear
    hour = ts.dt.hour
    # 연간 사인파(여름 고온/겨울 저온) + 일간 사인파(낮 고온/새벽 저온) + 잡음
    seasonal = 15 + 15 * np.sin(2 * np.pi * (doy - 105) / 365)
    diurnal = 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    noise = np.random.default_rng(config.RANDOM_STATE).normal(0, 1.5, len(ts))
    ta = seasonal + diurnal + noise
    return pd.DataFrame({"timestamp": ts, "TA_capital_avg": ta})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="수도권 기상(기온) 데이터 수집")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--auth-key", default=None, help="KMA authKey (미지정시 환경변수 사용)")
    args = parser.parse_args()

    fetch_capital_region_weather(args.start, args.end, auth_key=args.auth_key)
