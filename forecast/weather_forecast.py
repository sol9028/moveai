# -*- coding: utf-8 -*-
"""
weather_forecast.py
기상청 "단기예보"(getVilageFcst) API에서 앞으로 최대 3일치(글피까지) 시간별
기온 예보를 받아온다. weather.py(과거 관측 API, kma_sfctm3.php)와는 다른
API이니 헷갈리지 말 것.

[중요] 이 스크립트도 apihub.kma.go.kr에 실제 네트워크 요청을 보내므로,
Claude 샌드박스에서는 직접 실행이 안 됩니다. 사용자 로컬 환경에서
authKey 넣어 실행해야 합니다.

사용법:
    $env:KMA_AUTH_KEY = "발급받은키"        (PowerShell)
    python -m forecast.weather_forecast
"""

from __future__ import annotations
import os
import pandas as pd

from . import config


def _latest_base_datetime(now: pd.Timestamp | None = None):
    """
    단기예보는 하루 8회(02,05,08,11,14,17,20,23시) 발표됨.
    현재 시각 기준으로 가장 최근 발표시각을 찾는다.
    (발표 후 API 반영까지 시차가 있어 안전하게 10분 여유를 둠)
    """
    now = now or pd.Timestamp.now()
    now = now - pd.Timedelta(minutes=10)
    slots = [2, 5, 8, 11, 14, 17, 20, 23]

    base_date = now.strftime("%Y%m%d")
    base_hour = None
    for h in reversed(slots):
        if now.hour >= h:
            base_hour = h
            break

    if base_hour is None:
        # 자정~새벽 2시 사이면 전날 23시 발표분을 씀
        prev_day = now - pd.Timedelta(days=1)
        base_date = prev_day.strftime("%Y%m%d")
        base_hour = 23

    base_time = f"{base_hour:02d}00"
    return base_date, base_time


def fetch_station_forecast(
    nx: int,
    ny: int,
    auth_key: str,
    base_date: str | None = None,
    base_time: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """단일 격자지점의 시간별 기온(TMP) 예보를 받아온다."""
    import requests

    if base_date is None or base_time is None:
        base_date, base_time = _latest_base_datetime()

    params = {
        "serviceKey": auth_key,
        "pageNo": "1",
        "numOfRows": "1000",
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }
    r = requests.get(config.KMA_FORECAST_URL, params=params, timeout=timeout)
    r.raise_for_status()
    body = r.json()["response"]["body"]
    items = body["items"]["item"]

    df = pd.DataFrame(items)
    tmp = df[df["category"] == "TMP"].copy()
    tmp["timestamp"] = pd.to_datetime(
        tmp["fcstDate"] + tmp["fcstTime"], format="%Y%m%d%H%M"
    )
    tmp["TA"] = pd.to_numeric(tmp["fcstValue"], errors="coerce")
    return tmp[["timestamp", "TA"]].reset_index(drop=True)


def fetch_capital_region_forecast(
    auth_key: str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """서울/인천/수원 격자 평균으로 수도권 대표 미래 기온 예보 산출."""
    auth_key = auth_key or os.environ.get(config.KMA_AUTH_KEY_ENV)
    if not auth_key:
        raise ValueError(
            f"authKey가 필요합니다. 환경변수 {config.KMA_AUTH_KEY_ENV} 설정하거나 "
            "auth_key 인자로 직접 전달하세요."
        )

    all_dfs = []
    for stn, info in config.WEATHER_FORECAST_GRID.items():
        print(f"[{info['name']}] 예보 수집 중 (nx={info['nx']}, ny={info['ny']})...")
        df = fetch_station_forecast(info["nx"], info["ny"], auth_key)
        all_dfs.append(df)
        print(f"  -> {len(df)}개 시점 수신")

    merged = pd.concat(all_dfs, ignore_index=True)
    capital_forecast = (
        merged.groupby("timestamp")["TA"]
        .mean()
        .reset_index()
        .rename(columns={"TA": "TA_capital_avg"})
    )

    if save:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        capital_forecast.to_csv(
            config.FORECAST_WEATHER_PATH, index=False, encoding="utf-8-sig"
        )
        print(f"[OK] 수도권 예보기온 저장: {config.FORECAST_WEATHER_PATH} "
              f"({len(capital_forecast)}행)")

    return capital_forecast


if __name__ == "__main__":
    fetch_capital_region_forecast()