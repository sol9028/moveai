# -*- coding: utf-8 -*-
"""
capital_region_forecast_model.py
=====================================================================
수도권 전력수요 / 부족량 / ESS 방전필요량 예측 모델 (단일 파일 버전)
E-Train RE:LOOP 프로젝트 - forecast 모듈 통합본

원본은 forecast/ 패키지로 모듈화되어 있으며, 이 파일은 제출/공유
편의를 위해 하나로 합친 버전입니다.

실행 방법:
    python capital_region_forecast_model.py

필요 파일 (스크립트와 같은 폴더의 data/ 하위):
    data/한국전력거래소_시간별_전국_전력수요량_2024.csv
    data/한국전력거래소_시간별_전국_전력수요량_2025.csv
    data/capital_weather_hourly.csv   (선택, 없으면 합성데이터로 대체)

=====================================================================
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import holidays
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error



# ======================================================================
# --- py ---
# ======================================================================

# -*- coding: utf-8 -*-
"""
E-Train RE:LOOP - 수도권 전력수요/부족량/ESS 방전필요량 예측
설정값 모음

주의: 아래 수치들 중 '가정치'로 표시된 값은 실제 데이터가 확보되기 전까지
쓰는 임시값입니다. 실증/상용화 단계에서 실제 통계로 교체해야 합니다.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# 경로
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR 
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

RAW_FILES = [
    DATA_DIR / "한국전력거래소_시간별 전국 전력수요량_2024.csv",
    DATA_DIR / "한국전력거래소_시간별 전국 전력수요량_2025.csv",
]

PROCESSED_LONG_PATH = DATA_DIR / "national_demand_long.csv"
CAPITAL_DEMAND_PATH = DATA_DIR / "capital_region_demand.csv"
WEATHER_PATH = DATA_DIR / "capital_weather_hourly.csv"

# 최종 산출물 (수요예측 / 부족량은 별도 파일로 분리)
DEMAND_FORECAST_OUTPUT = OUTPUT_DIR / "capital_demand_forecast.csv"
SHORTAGE_FORECAST_OUTPUT = OUTPUT_DIR / "shortage_forecast.csv"
TOP_SHORTAGE_DAYS_OUTPUT = OUTPUT_DIR / "top_shortage_days_hourly.csv"
TOP_SHORTAGE_DAYS_OUTPUT_10TO18 = OUTPUT_DIR / "top_shortage_days_hourly_10to18.csv"
TOP_SHORTAGE_DAYS_N = 5
OPERATING_HOUR_RANGE = (10, 18)   # ESS 방전 운영 가능 시간대 (기존 노트북 기준)

# ----------------------------------------------------------------------
# 기상 데이터 (KMA API)
# ----------------------------------------------------------------------
# 수도권 커버용 ASOS 지점 (지점번호: 지점명)
# 서울/인천/수원 3개 평균으로 '수도권 대표기온' 근사.
# 필요시 지점 추가/조정 가능 (예: 동두천98, 파주99 등)
WEATHER_STATIONS = {
    "108": "서울",
    "112": "인천",
    "119": "수원",
}
KMA_WEATHER_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
# authKey는 코드에 하드코딩하지 않고 환경변수나 함수 인자로 전달할 것
KMA_AUTH_KEY_ENV = "KMA_AUTH_KEY"

# ----------------------------------------------------------------------
# [가정치] 전국 수요 -> 수도권 수요 변환 비중
# 근거: 한전 통계연보 기준 수도권(서울/인천/경기) 전력판매량 비중은
# 대략 35~40% 수준으로 알려져 있음. 정확한 최신 수치는
# 한전 데이터포털/전력통계정보시스템(EPSIS)에서 검증 필요.
# ----------------------------------------------------------------------
CAPITAL_REGION_RATIO = 0.40

# ----------------------------------------------------------------------
# 부족량(shortage/deficit) 산정 방식 — 두 가지 지원
# ----------------------------------------------------------------------
# "baseline": 일별 저부하 시간대 대비 초과분 방식 (기본값)
#   - 실제 발전설비/공급능력 데이터가 없어도 계산 가능
#   - baseline(t) = 최근 BASELINE_LOOKBACK_DAYS일 '일별 하위분위수'의 롤링 중앙값
#   - 하루짜리 노이즈에 덜 민감하도록 다일(多日) 롤링으로 개선
#     (특정 하루의 15%ile만 쓰면 그날 자체가 이상치일 때 baseline이 왜곡됨)
#   - shortage(t) = max(0, demand(t) - baseline(t))
# "capacity": 공급능력 가정 기반 방식 (구 버전, 실데이터 확보 시 유용)
#   - capacity(t) = 최근 30일 최대수요 * (1+예비율) * 가용률
#   - shortage(t) = max(0, demand(t) - capacity(t))
SHORTAGE_METHOD = "baseline"     # "baseline" | "capacity"

# --- baseline 방식 파라미터 ---
BASELINE_QUANTILE = 0.10          # 일별 하위 10% 지점을 그날의 '기저부하'로 봄
BASELINE_LOOKBACK_DAYS = 7        # 최근 7일 기저부하의 중앙값을 baseline으로 사용

# --- capacity 방식 파라미터 (레거시) ---
TARGET_RESERVE_MARGIN = 0.10     # 목표 예비율 10% 가정
SUPPLY_SAFETY_FACTOR = 0.95      # 실공급 가용률 95% 가정 (정비/고장 등 반영)

# ----------------------------------------------------------------------
# ESS 방전 관련 가정
# ----------------------------------------------------------------------
# 부족량 중 ESS가 커버하는 비율 (나머지는 다른 예비자원으로 대응한다고 가정)
ESS_COVERAGE_RATIO = 0.30

# 안중 / 서화성 방전거점 분배 비율 (현재는 50:50 고정, 추후 MILP에서
# 동적 배분으로 확장 가능)
DISCHARGE_SITE_SPLIT = {
    "안중": 0.5,
    "서화성": 0.5,
}

# ----------------------------------------------------------------------
# 학습/평가 관련
# ----------------------------------------------------------------------
TEST_HOLDOUT_DAYS = 60     # 마지막 60일을 테스트셋으로 분리 (시계열이므로 랜덤 분리 X)
RANDOM_STATE = 42

# 시간별 wide 컬럼명 (1시~24시)
HOUR_COLS = [f"{h}시" for h in range(1, 25)]


# ======================================================================
# --- data_prep.py ---
# ======================================================================

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

from pathlib import Path



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
    for path in RAW_FILES:
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
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        full.to_csv(PROCESSED_LONG_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] 전국 long 데이터 저장: {PROCESSED_LONG_PATH} ({len(full)}행)")

    return full


def apply_capital_region_ratio(
    national_long: pd.DataFrame,
    ratio: float = CAPITAL_REGION_RATIO,
    save: bool = True,
) -> pd.DataFrame:
    """
    전국 수요에 수도권 비중을 곱해 수도권 수요 프록시 생성.
    [가정치] 실제 지역별 통계 확보 시 이 함수를 대체할 것.
    """
    df = national_long.copy()
    df["capital_demand_mw"] = df["demand_mw"] * ratio

    if save:
        df.to_csv(CAPITAL_DEMAND_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] 수도권 수요(프록시) 저장: {CAPITAL_DEMAND_PATH}")

    return df


def load_or_build_capital_dataset() -> pd.DataFrame:
    """이미 만들어둔 처리 결과가 있으면 로드, 없으면 새로 생성."""
    if CAPITAL_DEMAND_PATH.exists():
        df = pd.read_csv(CAPITAL_DEMAND_PATH, parse_dates=["timestamp"])
        return df
    national = build_national_long_dataset()
    return apply_capital_region_ratio(national)


# ======================================================================
# --- weather.py ---
# ======================================================================

# -*- coding: utf-8 -*-
"""
weather.py
기상청 ASOS 시간자료 API(kma_sfctm3.php)에서 수도권 대표 기온을 받아온다.

[중요] 이 스크립트는 apihub.kma.go.kr에 실제 네트워크 요청을 보냅니다.
Claude 샌드박스 환경에서는 이 도메인이 허용 목록에 없어 직접 실행이
안 됩니다 (robots.txt 및 네트워크 정책). 사용자 로컬 환경 / Jupyter에서
authKey를 넣어 직접 실행한 뒤, 결과 CSV(WEATHER_PATH)를
data/ 폴더에 넣어주면 이후 파이프라인이 자동으로 인식합니다.

사용법:
    export KMA_AUTH_KEY="발급받은키"
    python -m forecast.weather --start 2024-01-01 --end 2025-12-31

또는 코드에서 직접:
    from forecast.weather import fetch_capital_region_weather
    df = fetch_capital_region_weather("2024-01-01", "2025-12-31", auth_key="...")
"""

import os
import argparse


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
    r = requests.get(KMA_WEATHER_URL, params=params, timeout=timeout)
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
    stations 미지정 시 WEATHER_STATIONS(서울/인천/수원) 사용.
    """
    auth_key = auth_key or os.environ.get(KMA_AUTH_KEY_ENV)
    if not auth_key:
        raise ValueError(
            f"authKey가 필요합니다. 환경변수 {KMA_AUTH_KEY_ENV} 설정하거나 "
            "auth_key 인자로 직접 전달하세요."
        )
    stations = stations or WEATHER_STATIONS
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
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        capital_weather.to_csv(WEATHER_PATH, index=False, encoding="utf-8-sig")
        print(f"[OK] 수도권 기온 데이터 저장: {WEATHER_PATH} ({len(capital_weather)}행)")

    return capital_weather


def load_weather_if_exists() -> pd.DataFrame | None:
    """이미 수집된 기상 CSV가 있으면 로드, 없으면 None."""
    if WEATHER_PATH.exists():
        df = pd.read_csv(WEATHER_PATH, parse_dates=["timestamp"])
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
    noise = np.random.default_rng(RANDOM_STATE).normal(0, 1.5, len(ts))
    ta = seasonal + diurnal + noise
    return pd.DataFrame({"timestamp": ts, "TA_capital_avg": ta})


# ======================================================================
# --- features.py ---
# ======================================================================

# -*- coding: utf-8 -*-
"""
features.py
전력수요 예측을 위한 피처 엔지니어링.

- 캘린더 피처 (요일, 월, 공휴일, 주말)
- 시간/계절 주기성 (sin/cos 인코딩)
- 지연(lag) 피처 (1시간 전, 24시간 전, 168시간(1주일) 전)
- 롤링 통계 (최근 24시간 평균/최대)
"""



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
    lags=(1, 24, 168),
) -> pd.DataFrame:
    """
    주의: lag/rolling 피처는 예측 시점에 '미래 정보 누수'가 없도록
    반드시 시간순 정렬된 데이터에 대해서만 적용해야 한다.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)

    df["roll_mean_24h"] = df[target_col].shift(1).rolling(24).mean()
    df["roll_max_24h"] = df[target_col].shift(1).rolling(24).max()
    df["roll_std_24h"] = df[target_col].shift(1).rolling(24).std()

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


# ======================================================================
# --- demand.py ---
# ======================================================================

# -*- coding: utf-8 -*-
"""
demand.py
수도권 시간대별 전력수요 예측 모델 (LightGBM 회귀).

- 시계열이므로 랜덤 K-fold가 아니라 '마지막 N일'을 테스트셋으로 분리
- lag/rolling 피처를 쓰기 때문에, 실제 운영 시 T+1시간 예측은 문제없으나
  T+24시간 이상 먼 미래를 예측하려면 재귀적(recursive) 예측이 필요함
  (recursive_forecast 함수 참고)
- 기온 피처(TA_capital_avg) 포함 여부는 feature_cols 인자로 결정되며,
  build_feature_frame()이 반환하는 feature_cols를 그대로 넘기면 된다.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error



def time_based_split(df: pd.DataFrame, test_days: int = TEST_HOLDOUT_DAYS):
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train = df[df["timestamp"] <= cutoff].reset_index(drop=True)
    test = df[df["timestamp"] > cutoff].reset_index(drop=True)
    return train, test


def train_demand_model(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "capital_demand_mw",
    save_model: bool = True,
    model_filename: str = "capital_demand_lgbm.joblib",
):
    train, test = time_based_split(feature_df)

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    pred = model.predict(X_test)
    metrics = evaluate(y_test.values, pred)

    print("[모델 성능 - 테스트셋 / 최근 {}일] 피처: {}".format(
        TEST_HOLDOUT_DAYS, feature_cols))
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / model_filename
        joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
        print(f"[OK] 모델 저장: {model_path}")

    result_df = test[["timestamp", target_col]].copy()
    result_df["predicted_mw"] = pred
    result_df["error_mw"] = result_df["predicted_mw"] - result_df[target_col]

    return model, metrics, result_df


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE(MW)": mae, "RMSE(MW)": rmse, "MAPE(%)": mape}


def feature_importance(model, feature_cols: list[str], top_n: int = 10) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp.head(top_n)


def recursive_forecast(
    model,
    history_df: pd.DataFrame,
    feature_cols: list[str],
    horizon_hours: int,
    target_col: str = "capital_demand_mw",
    weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    학습에 쓴 것과 동일한 피처 구조로, 마지막 시점 이후 horizon_hours 만큼
    한 시간씩 재귀적으로 예측한다.
    (lag_1h/24h/168h, rolling 값을 매 스텝 갱신)

    weather_df: 미래 시점의 기온 예보가 있으면 전달 (timestamp, TA_capital_avg).
    없으면 TA_capital_avg가 feature_cols에 포함된 경우 마지막 관측값으로 고정한다
    (단기 예측 시 근사로는 무방하나, 장기 예측일수록 오차가 커짐에 유의).
    """

    use_weather = "TA_capital_avg" in feature_cols
    work = history_df[["timestamp", target_col]].copy().sort_values("timestamp")
    work = work.reset_index(drop=True)

    last_known_ta = None
    if use_weather and weather_df is None:
        # fallback: 원 데이터에 기온이 있었다면 그 마지막 값을 그대로 사용
        if "TA_capital_avg" in history_df.columns:
            last_known_ta = history_df["TA_capital_avg"].iloc[-1]

    forecasts = []
    last_ts = work["timestamp"].iloc[-1]

    for step in range(1, horizon_hours + 1):
        next_ts = last_ts + pd.Timedelta(hours=step)
        row = pd.DataFrame({"timestamp": [next_ts]})
        row = add_calendar_features(row)
        row = add_cyclical_features(row)

        values = pd.concat(
            [work[target_col], pd.Series([np.nan])], ignore_index=True
        )

        def get_lag(h):
            idx = len(values) - 1 - h
            return values.iloc[idx] if idx >= 0 else np.nan

        row["lag_1h"] = get_lag(1)
        row["lag_24h"] = get_lag(24)
        row["lag_168h"] = get_lag(168)

        recent_24 = values.iloc[-24:].astype(float)
        row["roll_mean_24h"] = recent_24.mean()
        row["roll_max_24h"] = recent_24.max()
        row["roll_std_24h"] = recent_24.std()

        if use_weather:
            if weather_df is not None:
                match = weather_df.loc[weather_df["timestamp"] == next_ts, "TA_capital_avg"]
                row["TA_capital_avg"] = match.iloc[0] if len(match) else last_known_ta
            else:
                row["TA_capital_avg"] = last_known_ta

        X_next = row[feature_cols]
        pred = model.predict(X_next)[0]

        forecasts.append({"timestamp": next_ts, "predicted_mw": pred})
        work = pd.concat(
            [work, pd.DataFrame({"timestamp": [next_ts], target_col: [pred]})],
            ignore_index=True,
        )

    return pd.DataFrame(forecasts)


# ======================================================================
# --- shortage.py ---
# ======================================================================

# -*- coding: utf-8 -*-
"""
shortage.py
예측된 수도권 전력수요를 바탕으로
  1) 부족량(shortage_mw)
  2) ESS 방전 필요량 (ess_discharge_need_mw)
  3) 안중 / 서화성 방전거점별 분배량
을 계산한다.

두 가지 방식 지원 (SHORTAGE_METHOD로 선택):
  - "baseline" (기본값): 일별 저부하 구간 대비 초과분.
      실제 공급능력 데이터 없이도 계산 가능. 특정 하루의 노이즈에
      덜 민감하도록 여러 날의 롤링 중앙값을 사용.
  - "capacity": 공급능력(가정) 대비 초과분. 실제 발전설비 데이터
      확보 시 정확도가 올라가는 레거시 방식.
"""




# ----------------------------------------------------------------------
# 방식 1) baseline (기본값)
# ----------------------------------------------------------------------
def compute_rolling_baseline(
    df: pd.DataFrame,
    demand_col: str,
    ts_col: str = "timestamp",
    quantile: float = BASELINE_QUANTILE,
    lookback_days: int = BASELINE_LOOKBACK_DAYS,
) -> pd.Series:
    """
    각 날짜의 '하위 quantile 수요'를 구한 뒤, 그 값들의
    최근 lookback_days일 롤링 중앙값을 baseline으로 사용한다.

    예: quantile=0.10, lookback_days=7
        -> "최근 7일간, 하루의 하위 10% 지점 수요들의 중앙값"
        -> 특정 하루가 이상치여도 baseline이 크게 흔들리지 않음
    """
    daily_low = (
        df.groupby(df[ts_col].dt.date)[demand_col]
        .quantile(quantile)
        .rename("daily_low")
    )
    daily_low.index = pd.to_datetime(daily_low.index)
    rolling_baseline = daily_low.rolling(lookback_days, min_periods=1).median()

    date_map = rolling_baseline.to_dict()
    baseline_series = df[ts_col].dt.normalize().map(date_map)
    baseline_series.index = df.index
    return baseline_series


def estimate_shortage_baseline(
    df: pd.DataFrame,
    demand_col: str = "predicted_mw",
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    out = df.copy()
    out["baseline_mw"] = compute_rolling_baseline(out, demand_col, ts_col)
    out["shortage_mw"] = (out[demand_col] - out["baseline_mw"]).clip(lower=0)
    return out


# ----------------------------------------------------------------------
# 방식 2) capacity (레거시)
# ----------------------------------------------------------------------
def estimate_supply_capacity(demand_series: pd.Series) -> pd.Series:
    """[가정치] 공급능력 = 최근 30일 최대수요 * (1 + 목표예비율) * 가용률"""
    recent_peak = demand_series.rolling(24 * 30, min_periods=24).max()
    recent_peak = recent_peak.bfill()
    capacity = (
        recent_peak
        * (1 + TARGET_RESERVE_MARGIN)
        * SUPPLY_SAFETY_FACTOR
    )
    return capacity


def estimate_shortage_capacity(
    df: pd.DataFrame, demand_col: str = "predicted_mw"
) -> pd.DataFrame:
    out = df.copy()
    out["supply_capacity_mw"] = estimate_supply_capacity(out[demand_col])
    out["shortage_mw"] = (out[demand_col] - out["supply_capacity_mw"]).clip(lower=0)
    return out


# ----------------------------------------------------------------------
# 공통: ESS 방전필요량 + 거점 분배
# ----------------------------------------------------------------------
def estimate_ess_discharge_need(
    df: pd.DataFrame,
    coverage_ratio: float = ESS_COVERAGE_RATIO,
) -> pd.DataFrame:
    out = df.copy()
    out["ess_discharge_need_mw"] = out["shortage_mw"] * coverage_ratio
    return out


def split_by_discharge_site(
    df: pd.DataFrame,
    split: dict = DISCHARGE_SITE_SPLIT,
) -> pd.DataFrame:
    """ESS 방전필요량을 안중/서화성 거점으로 분배 (기본 50:50)."""
    out = df.copy()
    for site, ratio in split.items():
        out[f"ess_need_{site}_mw"] = out["ess_discharge_need_mw"] * ratio
    return out


def build_shortage_pipeline(
    demand_df: pd.DataFrame,
    demand_col: str = "predicted_mw",
    method: str = SHORTAGE_METHOD,
) -> pd.DataFrame:
    """예측 수요 df -> 부족량 -> ESS 방전필요량 -> 거점 분배까지 한 번에."""
    if method == "baseline":
        out = estimate_shortage_baseline(demand_df, demand_col)
    elif method == "capacity":
        out = estimate_shortage_capacity(demand_df, demand_col)
    else:
        raise ValueError(f"알 수 없는 method: {method} ('baseline' 또는 'capacity')")

    out = estimate_ess_discharge_need(out)
    out = split_by_discharge_site(out)
    return out


def top_shortage_days_hourly(
    full_df: pd.DataFrame,
    top_n: int = 5,
    rank_by: str = "daily_sum",
    ts_col: str = "timestamp",
    hour_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """
    부족량 상위 N일을 뽑아서, 그 날짜들의 시간별(0~23시, 또는 hour_range로
    제한한 구간) 초과량을 안중/서화성 노드별로 long format(위치 컬럼 포함)으로
    펼친다.

    입력: build_shortage_pipeline()을 거친 df (shortage_mw,
          ess_need_<site>_mw 컬럼들이 이미 있어야 함)

    rank_by:
      - "daily_sum": 일별 shortage_mw 총합 기준 (하루 전체 누적 초과량)
      - "daily_max": 일별 shortage_mw 최대값 기준 (그날의 순간 최대 초과)

    hour_range: (시작시, 끝시) 튜플, 예: (10, 18) -> 10~18시만 포함.
                None이면 0~23시 전체.

    반환 컬럼: date, node, hour, timestamp, shortage_mw, ess_need_mw
      (shortage_mw는 안중/서화성 공통 총량, ess_need_mw는 해당 노드 몫)
    """
    df = full_df.copy()
    df["date"] = df[ts_col].dt.date

    if rank_by == "daily_sum":
        daily_score = df.groupby("date")["shortage_mw"].sum()
    elif rank_by == "daily_max":
        daily_score = df.groupby("date")["shortage_mw"].max()
    else:
        raise ValueError("rank_by는 'daily_sum' 또는 'daily_max'만 가능합니다.")

    top_dates = daily_score.sort_values(ascending=False).head(top_n)
    print(f"[상위 {top_n}일 - 기준: {rank_by}]")
    print(top_dates.to_string())

    top_df = df[df["date"].isin(top_dates.index)].copy()
    if hour_range is not None:
        start_h, end_h = hour_range
        top_df = top_df[top_df[ts_col].dt.hour.between(start_h, end_h)]

    site_cols = [c for c in top_df.columns if c.startswith("ess_need_")]
    sites = [c.replace("ess_need_", "").replace("_mw", "") for c in site_cols]

    has_weather = "TA_capital_avg" in top_df.columns

    records = []
    for _, row in top_df.iterrows():
        for site in sites:
            record = {
                "date": row["date"],
                "node": site,
                "hour": row[ts_col].hour,
                "timestamp": row[ts_col],
                "shortage_mw": round(row["shortage_mw"], 1),
                "ess_need_mw": round(row[f"ess_need_{site}_mw"], 1),
            }
            if has_weather:
                record["TA_capital_avg"] = round(row["TA_capital_avg"], 1)
            records.append(record)

    result = pd.DataFrame(records).sort_values(["date", "hour", "node"]).reset_index(drop=True)
    return result




    capital_df = load_or_build_capital_dataset()
    weather_df = load_weather_if_exists()
    feat_df, feature_cols = build_feature_frame(capital_df, weather_df=weather_df)
    model, metrics, result_df = train_demand_model(feat_df, feature_cols, save_model=False)

    for method in ["baseline", "capacity"]:
        print(f"\n===== 방식: {method} =====")
        full = build_shortage_pipeline(result_df, demand_col="predicted_mw", method=method)
        shortage_hours = full[full["shortage_mw"] > 0]
        print(f"부족 발생: {len(shortage_hours)} / {len(full)} 시간 "
              f"({len(shortage_hours)/len(full)*100:.1f}%)")
        if len(shortage_hours) > 0:
            print(f"최대 부족량: {shortage_hours['shortage_mw'].max():.1f} MW")
            print(f"평균 안중 ESS 필요량: {shortage_hours['ess_need_안중_mw'].mean():.1f} MW")
            print(f"평균 서화성 ESS 필요량: {shortage_hours['ess_need_서화성_mw'].mean():.1f} MW")


# ======================================================================
# --- KPX 실시간 최신수요 API 연동 (선택, 키 있으면 자동 사용) ---
# ======================================================================

KPX_LATEST_DEMAND_URL = "https://openapi.kpx.or.kr/openapi/sukub5mToday/getSukub5mToday"
KPX_AUTH_KEY_ENV = "KPX_AUTH_KEY"


def fetch_today_demand_kpx(auth_key: str, num_of_rows: int = 300, timeout: int = 30):
    """KPX '오늘전력수급현황조회' API에서 오늘자 5분단위 전국 수요를 받아온다."""
    import requests

    params = {
        "serviceKey": auth_key,
        "pageNo": "1",
        "numOfRows": str(num_of_rows),
        "dataType": "JSON",
    }
    r = requests.get(KPX_LATEST_DEMAND_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    try:
        items = data["response"]["body"]["items"]["item"]
    except (KeyError, TypeError):
        items = data["items"]["item"] if "items" in data else data

    df = pd.DataFrame(items)
    ts_col = "기준일시" if "기준일시" in df.columns else df.columns[0]
    demand_col = "현재수요" if "현재수요" in df.columns else df.columns[2]

    df["timestamp"] = pd.to_datetime(df[ts_col], format="%Y%m%d%H%M%S", errors="coerce")
    df["demand_mw"] = pd.to_numeric(df[demand_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "demand_mw"])

    hourly = df.set_index("timestamp")["demand_mw"].resample("h").mean().reset_index()
    return hourly


def get_capital_history_up_to(target_date: str, kpx_auth_key: str | None = None) -> pd.DataFrame:
    """
    target_date까지의 수도권 수요 이력을 확보한다.
    보유한 과거 CSV로 부족하면(target_date가 더 미래이면), kpx_auth_key가
    주어질 경우 KPX 실시간 API로 오늘자 데이터를 받아 자동으로 이어붙인다.
    (KPX API는 '오늘' 데이터만 제공하므로, target_date가 오늘이 아니고
    보유 데이터에도 없는 미래/과거 날짜라면 이어붙이기가 안 될 수 있다.)
    """
    capital_df = load_or_build_capital_dataset()
    target_ts = pd.Timestamp(target_date)
    last_ts = capital_df["timestamp"].max()

    if target_ts > last_ts:
        kpx_auth_key = kpx_auth_key or os.environ.get(KPX_AUTH_KEY_ENV)
        if kpx_auth_key:
            try:
                print(f"[안내] 보유 데이터 마지막 시점({last_ts})이 목표일({target_ts})보다 "
                      f"이전 -> KPX API로 최신 데이터 보충 시도")
                hourly = fetch_today_demand_kpx(kpx_auth_key)
                hourly["capital_demand_mw"] = hourly["demand_mw"] * CAPITAL_REGION_RATIO
                capital_df = pd.concat(
                    [capital_df, hourly[["timestamp", "capital_demand_mw"]]],
                    ignore_index=True,
                )
                capital_df = (
                    capital_df.drop_duplicates(subset="timestamp", keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                print(f"[OK] KPX API로 최신 데이터 보충 완료 (마지막 시점: "
                      f"{capital_df['timestamp'].max()})")
            except Exception as e:
                print(f"[WARN] KPX API 호출 실패({e}) -> 보유 데이터까지만 사용")
        else:
            print(f"[안내] 보유 데이터 마지막 시점({last_ts})이 목표일({target_ts})보다 이전이고 "
                  f"KPX_AUTH_KEY가 없어 최신 데이터 보충 불가 -> 보유 데이터까지만 사용")

    history = capital_df[capital_df["timestamp"] <= target_ts].reset_index(drop=True)
    if len(history) == 0:
        raise ValueError(f"{target_date} 이전 데이터가 전혀 없습니다.")
    return history


def forecast_from_date(
    target_date: str,
    horizon_hours: int = 72,
    kpx_auth_key: str | None = None,
    weather_df: pd.DataFrame | None = None,
):
    """
    target_date까지의 데이터를 확보(부족하면 API로 자동 보충)한 뒤,
    그 시점부터 horizon_hours 시간만큼 미래를 재귀예측한다.

    예) forecast_from_date("2026-08-13", horizon_hours=72)
        -> 2026-08-13 마지막 보유 시점부터 72시간 이후까지 예측
    """
    print("=" * 70)
    print(f"[날짜지정 예측] 기준일: {target_date}, 예측 구간: {horizon_hours}시간")
    print("=" * 70)

    history = get_capital_history_up_to(target_date, kpx_auth_key)
    print(f"[OK] 확보된 이력 데이터: {len(history)}행 (마지막 시점: {history['timestamp'].max()})")

    if weather_df is None:
        weather_df = load_weather_if_exists()
        if weather_df is None:
            print("[안내] 기온 데이터 없음 -> 배관 검증용 합성 데이터 사용")
            weather_df = make_synthetic_weather_for_testing(history)

    feat_df, feature_cols = build_feature_frame(history, weather_df=weather_df)
    model, metrics, _ = train_demand_model(feat_df, feature_cols, save_model=False)
    print(f"[OK] 모델 학습 완료 (MAE={metrics['MAE(MW)']:.1f}MW, "
          f"MAPE={metrics['MAPE(%)']:.2f}%)")

    history_with_weather = history.merge(
        weather_df[["timestamp", "TA_capital_avg"]], on="timestamp", how="left"
    )
    history_with_weather["TA_capital_avg"] = (
        history_with_weather["TA_capital_avg"].ffill().bfill()
    )

    future = recursive_forecast(
        model, history_with_weather, feature_cols,
        horizon_hours=horizon_hours, weather_df=None,
    )
    future_full = build_shortage_pipeline(future, demand_col="predicted_mw", method="baseline")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"forecast_from_{pd.Timestamp(target_date).strftime('%Y%m%d')}_{horizon_hours}h.csv"
    future_full.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 예측 결과 저장: {out_path}")
    print(future_full[["timestamp", "predicted_mw", "shortage_mw",
                        "ess_need_안중_mw", "ess_need_서화성_mw"]].to_string(index=False))

    return future_full, metrics


# ======================================================================
# --- run_pipeline (전체 실행 로직) ---
# ======================================================================

def _setup_korean_font():
    import subprocess
    try:
        result = subprocess.run(
            ["fc-list", ":lang=ko"], capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            first_font = result.stdout.splitlines()[0].split(":")[0]
            fm.fontManager.addfont(first_font)
            plt.rcParams["font.family"] = fm.FontProperties(fname=first_font).get_name()
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def plot_prediction_vs_actual(result_df, out_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result_df["timestamp"], result_df["capital_demand_mw"], label="Actual", linewidth=1.2)
    ax.plot(result_df["timestamp"], result_df["predicted_mw"], label="Predicted", linewidth=1.2, alpha=0.8)
    ax.set_title("Capital Region Power Demand: Actual vs Predicted (Test Set)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_shortage_and_ess(full_df, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(full_df["timestamp"], full_df["predicted_mw"], label="Predicted Demand", color="tab:blue")
    axes[0].plot(full_df["timestamp"], full_df["baseline_mw"], label="Baseline (low-load ref.)",
                 color="tab:gray", linestyle="--")
    axes[0].set_ylabel("MW")
    axes[0].set_title("Demand vs Baseline")
    axes[0].legend()

    axes[1].fill_between(full_df["timestamp"], full_df["shortage_mw"], color="tab:red", alpha=0.6)
    axes[1].set_ylabel("MW")
    axes[1].set_title("Estimated Shortage (baseline method)")

    axes[2].plot(full_df["timestamp"], full_df["ess_need_안중_mw"], label="ESS Need - Anjung")
    axes[2].plot(full_df["timestamp"], full_df["ess_need_서화성_mw"], label="ESS Need - Seohwaseong")
    axes[2].set_ylabel("MW")
    axes[2].set_title("ESS Discharge Need by Site (50:50 split)")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_weather_or_synthetic(capital_df):
    """실제 기상 CSV가 있으면 로드, 없으면 배관 검증용 합성 데이터 사용."""
    weather_df = load_weather_if_exists()
    if weather_df is not None:
        print(f"[OK] 실제 기상 데이터 로드: {WEATHER_PATH} ({len(weather_df)}행)")
        return weather_df, False

    print("[안내] 실제 기상 데이터가 없습니다 (data/capital_weather_hourly.csv 없음).")
    print("       fetch_capital_region_weather()로 KMA API에서 직접 받아오면 자동으로 반영됩니다.")
    print("       지금은 배관(코드 구조) 검증용 합성 기온 데이터를 사용합니다.")
    synthetic = make_synthetic_weather_for_testing(capital_df)
    return synthetic, True


def run_full_pipeline():
    _setup_korean_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("[1/6] 데이터 로드 및 수도권 수요(프록시) 생성")
    print("=" * 70)
    capital_df = load_or_build_capital_dataset()

    print("\n" + "=" * 70)
    print("[2/6] 기상 데이터 로드")
    print("=" * 70)
    weather_df, is_synthetic = load_weather_or_synthetic(capital_df)

    print("\n" + "=" * 70)
    print("[3/6] 피처 엔지니어링")
    print("=" * 70)
    feat_df, feature_cols = build_feature_frame(capital_df, weather_df=weather_df)
    print(f"학습 가능 샘플 수: {len(feat_df)}")
    print(f"사용 피처: {feature_cols}")

    print("\n" + "=" * 70)
    print("[4/6] 수요예측 모델 학습")
    print("=" * 70)
    model, metrics, result_df = train_demand_model(feat_df, feature_cols)

    imp = feature_importance(model, feature_cols)
    print("\n[Feature Importance Top 10]")
    print(imp.to_string(index=False))

    plot_path_1 = OUTPUT_DIR / "demand_actual_vs_predicted.png"
    plot_prediction_vs_actual(result_df, plot_path_1)
    print(f"[OK] 그래프 저장: {plot_path_1}")

    demand_out = result_df.rename(columns={"capital_demand_mw": "actual_mw"})
    demand_out.to_csv(DEMAND_FORECAST_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[OK] 수요예측 결과 저장: {DEMAND_FORECAST_OUTPUT}")

    print("\n" + "=" * 70)
    print("[5/6] 부족량 / ESS 방전필요량 / 거점 분배 계산 (method=baseline)")
    print("=" * 70)
    full_df = build_shortage_pipeline(result_df, demand_col="predicted_mw", method="baseline")

    shortage_hours = full_df[full_df["shortage_mw"] > 0]
    print(f"부족 발생 시간대: {len(shortage_hours)} / {len(full_df)} "
          f"({len(shortage_hours) / len(full_df) * 100:.2f}%)")
    if len(shortage_hours) > 0:
        print(f"최대 부족량: {shortage_hours['shortage_mw'].max():.1f} MW")
        print(f"평균 ESS 방전필요량(부족 시간대): "
              f"{shortage_hours['ess_discharge_need_mw'].mean():.1f} MW")

    plot_path_2 = OUTPUT_DIR / "shortage_and_ess_need.png"
    plot_shortage_and_ess(full_df, plot_path_2)
    print(f"[OK] 그래프 저장: {plot_path_2}")

    shortage_out_cols = [
        "timestamp", "predicted_mw", "baseline_mw", "shortage_mw",
        "ess_discharge_need_mw", "ess_need_안중_mw", "ess_need_서화성_mw",
    ]
    full_df[shortage_out_cols].to_csv(SHORTAGE_FORECAST_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[OK] 부족량/ESS 방전필요량 결과 저장: {SHORTAGE_FORECAST_OUTPUT}")

    print("\n" + "=" * 70)
    print("[6/6] 향후 72시간 예측 (오늘 기준)")
    print("=" * 70)
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    forecast_from_date(
        target_date=today_str,
        horizon_hours=72,
        kpx_auth_key=os.environ.get(KPX_AUTH_KEY_ENV),
        weather_df=weather_df if not is_synthetic else None,
    )

    print("\n" + "=" * 70)
    print("파이프라인 완료")
    print("=" * 70)
    print(f"모델 성능: MAE={metrics['MAE(MW)']:.1f}MW, "
          f"RMSE={metrics['RMSE(MW)']:.1f}MW, MAPE={metrics['MAPE(%)']:.2f}%")
    if is_synthetic:
        print("\n[!] 주의: 위 성능은 '합성 기온 데이터' 기준입니다.")
        print("    실제 기상 데이터로 교체 후 반드시 재실행/재검증하세요.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        # 날짜 지정 실행: python capital_region_forecast_model.py 2026-08-13 [시간수]
        target_date_arg = sys.argv[1]
        horizon_arg = int(sys.argv[2]) if len(sys.argv) >= 3 else 72
        forecast_from_date(
            target_date=target_date_arg,
            horizon_hours=horizon_arg,
            kpx_auth_key=os.environ.get(KPX_AUTH_KEY_ENV),
        )
    else:
        # 인자 없이 실행: 전체 파이프라인(학습+평가+오늘기준 72시간 예측)
        run_full_pipeline()
