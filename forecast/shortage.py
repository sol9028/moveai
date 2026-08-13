# -*- coding: utf-8 -*-
"""
shortage.py
예측된 수도권 전력수요를 바탕으로
  1) 부족량(shortage_mw)
  2) ESS 방전 필요량 (ess_discharge_need_mw)
  3) 안중 / 서화성 방전거점별 분배량
을 계산한다.

두 가지 방식 지원 (config.SHORTAGE_METHOD로 선택):
  - "baseline" (기본값): 일별 저부하 구간 대비 초과분.
      실제 공급능력 데이터 없이도 계산 가능. 특정 하루의 노이즈에
      덜 민감하도록 여러 날의 롤링 중앙값을 사용.
  - "capacity": 공급능력(가정) 대비 초과분. 실제 발전설비 데이터
      확보 시 정확도가 올라가는 레거시 방식.
"""

from __future__ import annotations
import pandas as pd

from . import config


# ----------------------------------------------------------------------
# 방식 1) baseline (기본값)
# ----------------------------------------------------------------------
def compute_rolling_baseline(
    df: pd.DataFrame,
    demand_col: str,
    ts_col: str = "timestamp",
    quantile: float = config.BASELINE_QUANTILE,
    lookback_days: int = config.BASELINE_LOOKBACK_DAYS,
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
        * (1 + config.TARGET_RESERVE_MARGIN)
        * config.SUPPLY_SAFETY_FACTOR
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
    coverage_ratio: float = config.ESS_COVERAGE_RATIO,
) -> pd.DataFrame:
    out = df.copy()
    out["ess_discharge_need_mw"] = out["shortage_mw"] * coverage_ratio
    return out


def split_by_discharge_site(
    df: pd.DataFrame,
    split: dict = config.DISCHARGE_SITE_SPLIT,
) -> pd.DataFrame:
    """ESS 방전필요량을 안중/서화성 거점으로 분배 (기본 50:50)."""
    out = df.copy()
    for site, ratio in split.items():
        out[f"ess_need_{site}_mw"] = out["ess_discharge_need_mw"] * ratio
    return out


def build_shortage_pipeline(
    demand_df: pd.DataFrame,
    demand_col: str = "predicted_mw",
    method: str = config.SHORTAGE_METHOD,
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

if __name__ == "__main__":
    from .data_prep import load_or_build_capital_dataset
    from .features import build_feature_frame
    from .demand import train_demand_model
    from .weather import load_weather_if_exists

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
