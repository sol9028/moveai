# -*- coding: utf-8 -*-
"""
top_shortage_days.py
과거 전체 기간(2024~2025 실측 수요) 기준으로 '초과 수요량이 컸던 상위 N일'을 뽑아,
그 날짜들의 시간별(0~23시) 초과량을 안중/서화성 노드별로 펼친 파일을 만든다.

[중요] 모델 예측치가 아니라 '실측 수요(capital_demand_mw)'를 기준으로 계산한다.
과거는 이미 정답(실측값)이 있으니 예측을 쓸 필요가 없고, 실측 기준이 더 정확하다.

실행: python -m forecast.top_shortage_days
출력:
  - outputs/top_shortage_days_hourly.csv        (0~23시 전체)
  - outputs/top_shortage_days_hourly_10to18.csv (10~18시만, 참고 노트북과 동일 포맷)
"""

from __future__ import annotations

from . import config
from .data_prep import load_or_build_capital_dataset
from .shortage import build_shortage_pipeline, top_shortage_days_hourly
from .weather import load_weather_if_exists


def main(top_n: int = config.TOP_SHORTAGE_DAYS_N, rank_by: str = "daily_sum"):
    print("=" * 70)
    print(f"과거 실측 수요 기준 초과량 상위 {top_n}일 추출 (기준: {rank_by})")
    print("=" * 70)

    capital_df = load_or_build_capital_dataset()

    weather_df = load_weather_if_exists()
    if weather_df is not None:
        capital_df = capital_df.merge(
            weather_df[["timestamp", "TA_capital_avg"]], on="timestamp", how="left"
        )
        print(f"[OK] 기온 데이터 병합 완료 ({len(weather_df)}행)")
    else:
        print("[WARN] 기온 데이터가 없습니다 (data/capital_weather_hourly.csv 없음) "
              "-> 기온 컬럼 없이 진행")

    # 실측 수요(capital_demand_mw)로 부족량 계산 (baseline 방식)
    full_df = build_shortage_pipeline(
        capital_df, demand_col="capital_demand_mw", method="baseline"
    )

    result = top_shortage_days_hourly(full_df, top_n=top_n, rank_by=rank_by)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.TOP_SHORTAGE_DAYS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 저장 완료: {config.TOP_SHORTAGE_DAYS_OUTPUT} ({len(result)}행)")
    print(f"     -> {top_n}일 x 24시간 x 2노드(안중/서화성) = {top_n * 24 * 2}행이 정상")
    print("\n미리보기:")
    print(result.head(10).to_string(index=False))

    # --- 운영시간대(10~18시)만 남긴, 참고 노트북과 동일한 4컬럼 포맷 ---
    print("\n" + "-" * 70)
    start_h, end_h = config.OPERATING_HOUR_RANGE
    print(f"운영시간대({start_h}~{end_h}시)만 필터링 + 참고 포맷으로 저장")
    print("-" * 70)
    result_op = top_shortage_days_hourly(
        full_df, top_n=top_n, rank_by=rank_by, hour_range=config.OPERATING_HOUR_RANGE
    )
    output_cols = ["date", "node", "hour", "ess_need_mw"]
    if "TA_capital_avg" in result_op.columns:
        output_cols.append("TA_capital_avg")
    result_op_formatted = result_op[output_cols].rename(
        columns={"ess_need_mw": "predicted_deficit_mwh", "TA_capital_avg": "TA"}
    )
    result_op_formatted.to_csv(
        config.TOP_SHORTAGE_DAYS_OUTPUT_10TO18, index=False, encoding="utf-8-sig"
    )
    n_hours = end_h - start_h + 1
    print(f"[OK] 저장 완료: {config.TOP_SHORTAGE_DAYS_OUTPUT_10TO18} ({len(result_op_formatted)}행)")
    print(f"     -> {top_n}일 x {n_hours}시간 x 2노드 = {top_n * n_hours * 2}행이 정상")
    print("\n미리보기:")
    print(result_op_formatted.head(10).to_string(index=False))


if __name__ == "__main__":
    main()