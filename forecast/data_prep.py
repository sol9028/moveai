# -*- coding: utf-8 -*-
"""
data_prep.py
한국전력거래소 시간별 전국 전력수요량 CSV(2024, 2025)를 읽어서
- 날짜 포맷 통일 (2024: YYYY-MM-DD / 2025: YYYY.M.D)
- wide(1시~24시 컬럼) -> long(timestamp, demand_mw) 변환
- 두 연도 병합
- 전국 -> 수도권 비중 적용
까지 수행한다.

24시는 해당 날짜의 '자정 직전 시간대(00:00~01:00 구간의 마지막)'로 보고
익일 00:00 timestamp로 매핑한다. (한전 관례상 1시~24시 표기)
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

from . import config


def _parse_date_column(raw_date: str) -> pd.Timestamp:
    """'2024-01-01' 또는 '2025.1.1' 형태를 모두 처리."""
    s = str(raw_date).strip()
    if "-" in s:
        return pd.to_datetime(s, format="%Y-%m-%d")
    if "." in s:
        return pd.to_datetime(s, format="%Y.%m.%d")
    # 방어적 처리: pandas 추론에 맡김
    return pd.to_datetime(s)


def load_raw_csv(path: Path) -> pd.DataFrame:
    """EUC-KR 인코딩 CSV 한 개를 읽고 날짜 컬럼만 표준화해서 반환."""
    df = pd.read_csv(path, encoding="euc-kr")
    df["날짜"] = df["날짜"].apply(_parse_date_column)
    return df


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜, 1시..24시 wide 포맷 -> timestamp, demand_mw long 포맷.
    '24시'는 다음날 00:00으로 매핑 (예: 2024-01-01 24시 -> 2024-01-02 00:00).
    """
    records = []
    for _, row in df.iterrows():
        base_date = row["날짜"]
        for h in range(1, 25):
            col = f"{h}시"
            value = row[col]
            if h == 24:
                ts = base_date + pd.Timedelta(days=1)
            else:
                ts = base_date + pd.Timedelta(hours=h)
            records.append((ts, value))
    long_df = pd.DataFrame(records, columns=["timestamp", "demand_mw"])
    return long_df


def build_national_long_dataset(save: bool = True) -> pd.DataFrame:
    """2024+2025 CSV를 합쳐 timestamp 기준 정렬된 전국 수요 long 데이터셋 생성."""
    frames = []
    for path in config.RAW_FILES:
        raw = load_raw_csv(path)
        long_df = wide_to_long(raw)
        frames.append(long_df)

    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset="timestamp").sort_values("timestamp")
    full = full.reset_index(drop=True)

    # 결측/이상치 체크
    n_missing = full["demand_mw"].isna().sum()
    if n_missing > 0:
        print(f"[WARN] 결측치 {n_missing}건 발견 -> 선형보간 처리")
        full["demand_mw"] = full["demand_mw"].interpolate()

    if save:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        full.to_csv(config.PROCESSED_LONG_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] 전국 long 데이터 저장: {config.PROCESSED_LONG_PATH} ({len(full)}행)")

    return full


def apply_capital_region_ratio(
    national_long: pd.DataFrame,
    ratio: float = config.CAPITAL_REGION_RATIO,
    save: bool = True,
) -> pd.DataFrame:
    """
    전국 수요에 수도권 비중을 곱해 수도권 수요 프록시 생성.
    [가정치] 실제 지역별 통계 확보 시 이 함수를 대체할 것.
    """
    df = national_long.copy()
    df["capital_demand_mw"] = df["demand_mw"] * ratio

    if save:
        df.to_csv(config.CAPITAL_DEMAND_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] 수도권 수요(프록시) 저장: {config.CAPITAL_DEMAND_PATH}")

    return df


def load_or_build_capital_dataset() -> pd.DataFrame:
    """이미 만들어둔 처리 결과가 있으면 로드, 없으면 새로 생성."""
    if config.CAPITAL_DEMAND_PATH.exists():
        df = pd.read_csv(config.CAPITAL_DEMAND_PATH, parse_dates=["timestamp"])
        return df
    national = build_national_long_dataset()
    return apply_capital_region_ratio(national)


if __name__ == "__main__":
    national = build_national_long_dataset()
    capital = apply_capital_region_ratio(national)
    print(capital.head())
    print(capital.tail())
    print(capital.describe())
