# -*- coding: utf-8 -*-
"""
latest_demand.py
data.go.kr의 "한국전력거래소_오늘전력수급현황조회" API에서 오늘자
5분 단위 전국 실측 수요를 받아, 시간 단위로 집계한 뒤
기존 capital_region_demand.csv 끝에 이어붙인다.

[중요] 이 API는 apis.data.go.kr(또는 openapi.kpx.or.kr)에 실제 네트워크
요청을 보내므로 Claude 샌드박스에서는 직접 실행이 안 됩니다.
사용자 로컬 환경에서 KPX_AUTH_KEY 넣어 실행해야 합니다.

이 API는 "오늘 하루치"만 줍니다. 매일 한 번씩(또는 몇 시간마다) 이
스크립트를 실행해서 데이터를 누적해야 "최근 데이터가 계속 이어지는"
상태가 유지됩니다.

사용법:
    $env:KPX_AUTH_KEY = "발급받은키"        (PowerShell)
    python -m forecast.latest_demand
"""

from __future__ import annotations
import os
import pandas as pd

from . import config


def fetch_today_demand(auth_key: str, num_of_rows: int = 300, timeout: int = 30) -> pd.DataFrame:
    """
    오늘자 5분 단위 전국 수요(현재수요, MW)를 받아온다.
    응답 필드: 기준일시, 공급능력, 현재수요, 최대예측수요,
              공급예비력, 공급예비율, 운영예비력, 운영예비율
    """
    import requests

    params = {
        "serviceKey": auth_key,
        "pageNo": "1",
        "numOfRows": str(num_of_rows),
        "dataType": "JSON",
    }
    r = requests.get(config.KPX_LATEST_DEMAND_URL, params=params, timeout=timeout)
    print("[상태코드]", r.status_code)
    print("[응답 원문 앞 500자]", r.text[:500])
    r.raise_for_status()
    data = r.json()

    # data.go.kr 응답 구조는 기관마다 조금씩 달라서, 두 가지 흔한 형태를 다 시도
    try:
        items = data["response"]["body"]["items"]["item"]
    except (KeyError, TypeError):
        items = data["items"]["item"] if "items" in data else data

    df = pd.DataFrame(items)
    print("[받은 컬럼]", df.columns.tolist())  # 실제 필드명 확인용 (처음 실행 시 꼭 확인)

    return df


def aggregate_to_hourly(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    5분 단위 원자료를 시간 단위(정시)로 집계.
    컬럼명은 실제 응답을 보고 필요시 조정할 것
    (예상: '기준일시', '현재수요').
    """
    df = raw_df.copy()

    # 컬럼명 후보 처리 (실제 응답 보고 아래 두 줄 수정 필요할 수 있음)
    ts_col = "기준일시" if "기준일시" in df.columns else df.columns[0]
    demand_col = "현재수요" if "현재수요" in df.columns else df.columns[2]

    df["timestamp"] = pd.to_datetime(df[ts_col], format="%Y%m%d%H%M%S", errors="coerce")
    df["demand_mw"] = pd.to_numeric(df[demand_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "demand_mw"])

    hourly = (
        df.set_index("timestamp")["demand_mw"]
        .resample("h")
        .mean()
        .reset_index()
    )
    return hourly


def update_capital_demand_dataset(auth_key: str | None = None):
    """오늘자 데이터를 받아서 capital_region_demand.csv 끝에 이어붙인다."""
    auth_key = auth_key or os.environ.get(config.KPX_AUTH_KEY_ENV)
    if not auth_key:
        raise ValueError(
            f"authKey가 필요합니다. 환경변수 {config.KPX_AUTH_KEY_ENV} 설정하거나 "
            "auth_key 인자로 직접 전달하세요."
        )

    raw = fetch_today_demand(auth_key)
    hourly_national = aggregate_to_hourly(raw)
    hourly_national["capital_demand_mw"] = (
        hourly_national["demand_mw"] * config.CAPITAL_REGION_RATIO
    )

    print(f"[OK] 오늘자 {len(hourly_national)}시간 수집 완료")
    print(hourly_national.tail(10))

    existing = pd.read_csv(config.CAPITAL_DEMAND_PATH, parse_dates=["timestamp"])
    
    combined = pd.concat(
        [existing, hourly_national[["timestamp", "capital_demand_mw"]]],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    print(f"[OK] {config.CAPITAL_DEMAND_PATH} 갱신 완료 "
          f"(마지막 시점: {combined['timestamp'].max()})")


if __name__ == "__main__":
    update_capital_demand_dataset()