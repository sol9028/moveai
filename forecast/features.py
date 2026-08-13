# -*- coding: utf-8 -*-
"""
features.py
전력수요 예측을 위한 피처 엔지니어링.

- 캘린더 피처 (요일, 월, 공휴일, 주말)
- 시간/계절 주기성 (sin/cos 인코딩)
- 지연(lag) 피처 (1시간 전, 24시간 전, 168시간(1주일) 전)
- 롤링 통계 (최근 24시간 평균/최대)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import holidays


def add_calendar_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    ts = df[ts_col]

    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek           # 0=월 ... 6=일
    df["month"] = ts.dt.month
    df["dayofyear"] = ts.dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    kr_holidays = holidays.KR(years=range(ts.dt.year.min(), ts.dt.year.max() + 1))
    df["is_holiday"] = ts.dt.date.astype(str).map(
        lambda d: 1 if pd.Timestamp(d) in kr_holidays else 0
    )
    # 실질적 '평일 부하 저하일' = 주말 OR 공휴일
    df["is_off_day"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "capital_demand_mw",
    ts_col: str = "timestamp",
    lags=(1, 24, 168),
) -> pd.DataFrame:
    """
    주의: lag/rolling 피처는 예측 시점에 '미래 정보 누수'가 없도록
    반드시 시간순 정렬된 데이터에 대해서만 적용해야 한다.

    데이터에 시간 공백(gap)이 있으면(예: 1월 데이터 이후 8월 데이터가
    바로 이어붙은 경우) 단순 shift()는 '몇 번째 이전 행'을 가져와서
    실제 경과시간과 안 맞을 수 있다. 그래서 shift 후 실제 시간차가
    의도한 시간(lag)과 정확히 일치하는지 검증하고, 안 맞으면 NaN 처리한다.
    """
    df = df.copy()
    ts = df[ts_col]

    for lag in lags:
        shifted_val = df[target_col].shift(lag)
        shifted_ts = ts.shift(lag)
        expected_ts = ts - pd.Timedelta(hours=lag)
        is_valid = shifted_ts == expected_ts
        df[f"lag_{lag}h"] = shifted_val.where(is_valid)

    prev_val = df[target_col].shift(1)
    prev_ts = ts.shift(1)
    is_prev_valid = (prev_ts == (ts - pd.Timedelta(hours=1)))

    roll_mean = df[target_col].shift(1).rolling(24).mean()
    roll_max = df[target_col].shift(1).rolling(24).max()
    roll_std = df[target_col].shift(1).rolling(24).std()

    df["roll_mean_24h"] = roll_mean.where(is_prev_valid)
    df["roll_max_24h"] = roll_max.where(is_prev_valid)
    df["roll_std_24h"] = roll_std.where(is_prev_valid)

    return df


def add_weather_features(
    df: pd.DataFrame,
    weather_df: pd.DataFrame | None,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """
    수도권 대표기온(TA_capital_avg)을 timestamp 기준으로 병합.
    weather_df가 None이면 아무것도 하지 않고 그대로 반환
    (날씨 데이터 없이도 파이프라인이 죽지 않게 하기 위함).
    """
    if weather_df is None:
        return df
    out = df.merge(weather_df[[ts_col, "TA_capital_avg"]], on=ts_col, how="left")
    n_missing = out["TA_capital_avg"].isna().sum()
    if n_missing > 0:
        print(f"[WARN] 기온 결측 {n_missing}건 -> 선형보간")
        out["TA_capital_avg"] = out["TA_capital_avg"].interpolate().bfill().ffill()
    return out


def build_feature_frame(
    df: pd.DataFrame,
    target_col: str = "capital_demand_mw",
    ts_col: str = "timestamp",
    weather_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    전체 피처 엔지니어링 파이프라인.
    weather_df를 넘기면 기온 피처가 자동으로 FEATURE_COLUMNS에 추가된다.
    반환값: (피처 데이터프레임, 실제 사용된 피처 컬럼 리스트)
    """
    out = df.sort_values(ts_col).reset_index(drop=True)
    out = add_calendar_features(out, ts_col)
    out = add_cyclical_features(out)
    out = add_weather_features(out, weather_df, ts_col)
    out = add_lag_features(out, target_col)

    feature_cols = list(BASE_FEATURE_COLUMNS)
    if weather_df is not None:
        feature_cols = feature_cols + ["TA_capital_avg"]

    # lag/rolling으로 인해 앞부분 168시간(1주일)은 NaN -> 학습에서 제외
    out = out.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    return out, feature_cols


BASE_FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday", "is_off_day",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "lag_1h", "lag_24h", "lag_168h",
    "roll_mean_24h", "roll_max_24h", "roll_std_24h",
]

# 하위호환: 기존에 FEATURE_COLUMNS를 직접 import하던 코드가 있으면
# (날씨 미포함) 기본 피처셋을 그대로 참조하도록 유지
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
