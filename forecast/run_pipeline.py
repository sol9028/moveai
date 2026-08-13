# -*- coding: utf-8 -*-
"""
run_pipeline.py
전체 파이프라인 실행 스크립트.

  1. 원본 CSV 정제 -> 수도권 수요(프록시) 생성
  2. 기상 데이터 로드 (있으면 사용, 없으면 테스트용 합성 데이터로 배관만 검증)
  3. 피처 엔지니어링 (기온 포함 여부 자동 결정)
  4. LightGBM 수요예측 모델 학습 + 평가
  5. 부족량 / ESS 방전필요량 / 안중·서화성 분배 계산 (baseline 방식 기본)
  6. 결과를 두 개의 파일로 분리 저장:
       - outputs/capital_demand_forecast.csv  (전체 수요 예측)
       - outputs/shortage_forecast.csv        (부족량/ESS 방전필요량)

실행: python -m forecast.run_pipeline
실제 기상데이터 사용하려면 먼저:
  export KMA_AUTH_KEY="발급받은키"
  python -m forecast.weather --start 2024-01-01 --end 2025-12-31
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

from . import config
from .data_prep import load_or_build_capital_dataset
from .features import build_feature_frame
from .demand import train_demand_model, feature_importance, recursive_forecast
from .shortage import build_shortage_pipeline
from .weather import load_weather_if_exists, make_synthetic_weather_for_testing


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


def plot_prediction_vs_actual(result_df: pd.DataFrame, out_path):
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


def plot_shortage_and_ess(full_df: pd.DataFrame, out_path):
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


def load_weather_or_synthetic(capital_df: pd.DataFrame):
    """실제 기상 CSV가 있으면 로드, 없으면 배관 검증용 합성 데이터 사용."""
    weather_df = load_weather_if_exists()
    if weather_df is not None:
        print(f"[OK] 실제 기상 데이터 로드: {config.WEATHER_PATH} ({len(weather_df)}행)")
        return weather_df, False

    print("[안내] 실제 기상 데이터가 없습니다 (data/capital_weather_hourly.csv 없음).")
    print("       forecast/weather.py로 KMA API에서 직접 받아오면 자동으로 반영됩니다.")
    print("       지금은 배관(코드 구조) 검증용 합성 기온 데이터를 사용합니다.")
    synthetic = make_synthetic_weather_for_testing(capital_df)
    return synthetic, True


def main():
    _setup_korean_font()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    plot_path_1 = config.OUTPUT_DIR / "demand_actual_vs_predicted.png"
    plot_prediction_vs_actual(result_df, plot_path_1)
    print(f"[OK] 그래프 저장: {plot_path_1}")

    # --- 산출물 1: 전체 수요 예측 (부족량 컬럼 없이 별도 저장) ---
    demand_out = result_df.rename(columns={"capital_demand_mw": "actual_mw"})
    demand_out.to_csv(config.DEMAND_FORECAST_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[OK] 수요예측 결과 저장 (전체 수요, 별도 파일): {config.DEMAND_FORECAST_OUTPUT}")

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

    plot_path_2 = config.OUTPUT_DIR / "shortage_and_ess_need.png"
    plot_shortage_and_ess(full_df, plot_path_2)
    print(f"[OK] 그래프 저장: {plot_path_2}")

    # --- 산출물 2: 부족량/ESS 방전필요량 (수요 예측치와 별도 파일) ---
    shortage_out_cols = [
        "timestamp", "predicted_mw", "baseline_mw", "shortage_mw",
        "ess_discharge_need_mw", "ess_need_안중_mw", "ess_need_서화성_mw",
    ]
    full_df[shortage_out_cols].to_csv(
        config.SHORTAGE_FORECAST_OUTPUT, index=False, encoding="utf-8-sig"
    )
    print(f"[OK] 부족량/ESS 방전필요량 결과 저장 (별도 파일): {config.SHORTAGE_FORECAST_OUTPUT}")

    print("\n" + "=" * 70)
    print("[6/6] 향후 72시간 재귀 예측(recursive forecast) 데모")
    print("=" * 70)
    history_with_weather = capital_df.merge(
        weather_df[["timestamp", "TA_capital_avg"]], on="timestamp", how="left"
    )
    history_with_weather["TA_capital_avg"] = (
        history_with_weather["TA_capital_avg"].ffill().bfill()
    )
    from .weather_forecast import fetch_capital_region_forecast
    import os as _os

    forecast_weather_df = None
    if config.FORECAST_WEATHER_PATH.exists():
        forecast_weather_df = pd.read_csv(
            config.FORECAST_WEATHER_PATH, parse_dates=["timestamp"]
        )
        print(f"[OK] 미래 예보기온 로드: {config.FORECAST_WEATHER_PATH}")
    else:
        print("[안내] 예보기온 파일 없음 -> 마지막 관측기온 고정으로 진행")
        print("       python -m forecast.weather_forecast 먼저 실행하면 반영됩니다.")

    future = recursive_forecast(
        model, history_with_weather, feature_cols, horizon_hours=72,
        weather_df=forecast_weather_df,
    )
    future_shortage = build_shortage_pipeline(future, demand_col="predicted_mw", method="baseline")
    future_csv = config.OUTPUT_DIR / "future_72h_forecast.csv"
    future_shortage.to_csv(future_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] 향후 72시간 예측 CSV 저장: {future_csv}")
    print(future_shortage[["timestamp", "predicted_mw", "shortage_mw",
                            "ess_need_안중_mw", "ess_need_서화성_mw"]].head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("파이프라인 완료")
    print("=" * 70)
    print(f"모델 성능: MAE={metrics['MAE(MW)']:.1f}MW, "
          f"RMSE={metrics['RMSE(MW)']:.1f}MW, MAPE={metrics['MAPE(%)']:.2f}%")
    if is_synthetic:
        print("\n[!] 주의: 위 성능은 '합성 기온 데이터' 기준입니다.")
        print("    실제 기상 데이터로 교체 후 반드시 재실행/재검증하세요.")


if __name__ == "__main__":
    main()
