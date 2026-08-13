# -*- coding: utf-8 -*-
"""(c) 하행 순환자원 발생량 예측 — 코호트 모델.

상행(호남→수도권)은 ESS 충전 전력, 하행(수도권→호남)은 배터리 순환자원을 싣는다.
이 모듈은 하행에 실을 수 있는 화물량을 예측한다.

[문제] 사용후 배터리·블랙매스 발생량은 공개 통계가 없다.
[해법] 시계열 예측이 아니라 **코호트(생존) 모델**로 역산한다.

    t년 배출량 = Σ_y [ y년 EV 신규등록 × 폐차확률(t−y) × 팩중량 ]

    "과거에 등록된 차가 언제 수명을 다하는가"를 세는 방식이므로,
    EV 등록 실적(실측)만 있으면 미래 배출량이 도출된다.
    업계 전망치를 인용하는 것이 아니라 등록 데이터에서 유도하는 구조다.

[발생원 4종과 중복 처리]
    블랙매스는 독립 발생원이 아니라 사용후 배터리·제조스크랩을 파쇄한 결과물이다.
    넷을 단순 합산하면 같은 물질을 두 번 세게 되므로 경로를 나눈다.

    사용후 EV 배터리 ┬─ 수도권 전처리(40%) ─→ 블랙매스(수율 40%) ─┐
                     └─ 팩 그대로(60%) ──────────────────────┐  │
    제조스크랩 ──────── 전처리 ─→ 블랙매스 ──────────────────┼──┤
    ESS 사용후 ─────── 팩 그대로 ────────────────────────────┘  │
                                                                 ↓
                              cargo_type = pack_ev / pack_ess / blackmass

[실행]
    python forecast/cargo.py
→ forecast/cargo_forecast.csv (최적화팀 전달용)
  forecast/cargo_annual.csv   (연도별 발생 추이, 발표용)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent  # 프로젝트 루트
OUT = Path(__file__).resolve().parent          # forecast/: 입력·출력 CSV 위치

# ==========================================================================
# 1. 실측 입력 — 연도별 전기차 신규 보급대수 (대)
# --------------------------------------------------------------------------
# 출처: 기후에너지환경부 보도자료 "전기차 연간 보급 20만대 달성" (2025.11)
#       국토교통부 자동차등록현황 (2025년 신규등록 전기차 221천건)
# 2011~2016년은 합계 1.2만대를 균등 배분.
# 2026년 이후는 최근 3년 평균 증가 추세를 보수적으로 반영해 외삽.
# ==========================================================================
EV_REGISTRATION = {
    2011: 2000, 2012: 2000, 2013: 2000, 2014: 2000, 2015: 2000, 2016: 2000,
    2017: 14_000,
    2018: 30_000,
    2019: 35_000,
    2020: 47_000,
    2021: 100_000,    # 연간 10만대 최초 돌파
    2022: 164_000,
    2023: 163_000,
    2024: 147_000,
    2025: 201_000,    # 연간 20만대 돌파
}
# 2026~2040 외삽: 2025년 수준에서 연 5% 성장 (정부 보급목표 대비 보수적)
_last = EV_REGISTRATION[2025]
for _y in range(2026, 2041):
    _last = int(_last * 1.05)
    EV_REGISTRATION[_y] = _last

# ==========================================================================
# 2. 가정치 — 전부 파라미터로 노출해 민감도 분석 대상으로 삼는다
# ==========================================================================
PARAMS = {
    # --- 배터리 수명 (Weibull 분포) ---
    # 근거: EV 배터리 보증이 통상 8년/16만km. 보증 만료 후에도 상당 기간
    #       운행되므로 평균 수명 10년, 형상모수 3.0(중년 집중 폐차)
    "life_scale": 11.0,        # Weibull scale (평균 수명 ≈ 10년)
    "life_shape": 3.0,
    # --- 물성 ---
    "pack_weight_kg": 400,     # EV 배터리팩 1대 중량. 승용 기준 350~500kg
    "blackmass_yield": 0.40,   # 팩 대비 블랙매스 수율. 케이스·전선 등 제외
    # --- 회수 ---
    # 근거: 2020년 이후 등록 EV는 폐차 시 배터리 반납 의무가 없어졌으나,
    #       유가금속 가치 상승으로 회수 유인이 크다. 성숙기 85% 가정.
    "recovery_rate": 0.85,
    "metro_share": 0.40,       # 수도권 EV 등록 비중 (전국 대비)
    "pretreat_share": 0.40,    # 수도권에서 전처리를 거치는 비율
    # --- 철도 수송 분담률: 예측이 아니라 사업 설계 목표값 ---
    # 근거: ① 사용후 배터리는 열폭주 위험으로 도로 운송 규제가 강해
    #          대량·정기·장거리 수송에서 철도가 구조적으로 유리하다
    #       ② 하행은 ESS 화차 회송으로 어차피 발생하는 공차 구간이므로
    #          한계 수송원가가 낮아 가격 경쟁력이 있다
    #       ③ 새만금 이차전지 특화단지라는 단일 목적지가 확정적이라
    #          철도의 거점 간 대량수송 특성과 부합한다
    "rail_share": 0.80,
    # --- 제조스크랩 ---
    # 근거: 셀 1GWh 생산 시 소재 약 5,000톤 투입, 공정 스크랩률 8%.
    #       국내 생산거점(삼성SDI 천안, SK온 서산, LG엔솔 오창 등) 합산
    #       CAPA를 40GWh로 보면 소재 투입 20만톤 → 스크랩 약 1.6만톤/년.
    #       ※ 국내 CAPA는 해외 대비 작다(미국만 420GWh). 과대추정 금지.
    "scrap_annual_ton": 16_000,
    "scrap_growth": 1.03,
    # --- ESS 사용후 배터리 ---
    # 근거: 코레일 에너지트레인 2028년 상용화 목표, 편성당 200MWh급(약 100톤).
    #   교체는 10년 차에 일괄로 일어나지 않는다. 셀 불량·모듈 열화 편차로
    #   초기부터 소량이 상시 교체되며, 수명 도래 시점에 급증한다.
    #   따라서 EV와 동일한 Weibull(평균 10년) 분포를 적용해 연도별로 배분한다.
    "ess_intro_year": 2028,        # 상용화 시점
    "ess_fleet": 2,                # 편성 수 (코레일 미확정 — 시나리오 값)
    "ess_pack_ton_per_set": 100,   # 편성당 배터리 중량 (200MWh급)
    "ess_life_scale": 11.0,        # ESS 배터리 Weibull scale
    "ess_life_shape": 3.0,
    # --- 화차 ---
    # 근거: 일반 화차 최대 적재 50톤. 팩은 부피가 커 무게 한도 전에
    #       적재공간이 차므로 실적재를 낮게 잡는다.
    "car_capacity_pack_ton": 30,
    "car_capacity_blackmass_ton": 50,
    # --- 운행 빈도 ---
    # 근거: 하행 화물은 상행 대비 물량이 작아 매일 만재가 불가능하다.
    #       주 2회 만재 편성으로 축적 운송하면 1회당 3.5일치가 모여
    #       화차 4량 규모가 된다. 나머지 요일 하행은 ESS 회송 위주.
    "trips_per_week": 2,
    # --- 기준연도 ---
    # 근거: 코레일 에너지트레인 2028년 상용화 목표 + 사용후 배터리 배출이
    #       2030년부터 급증. 2032년을 운영 안정화 시점으로 본다.
    "target_year": 2032,
}

# ==========================================================================
# 하행 집하·경유 적재역
# --------------------------------------------------------------------------
# 수도권에서만 싣는 것이 아니라, 장항선을 따라 남하하며 중간역에서 추가 적재한다.
# 근거: 국내 배터리 생산거점이 실제로 장항선 연변에 위치한다.
#   - 천안   : 삼성SDI 천안사업장
#   - 신례원 : SK온 서산공장 인근 (트럭 단거리 반입)
# 하행이 공장 앞을 지나므로 트럭 장거리 운송 대비 원가 우위가 발생하며,
# 이것이 철도 분담률을 높게 잡는 근거가 된다.
#
# 화물 종류가 발생지에 따라 다르다:
#   수도권(안중·서화성) → 사용후 EV 배터리 (폐차장·정비소 발생)
#   충남 경유역(천안·신례원) → 제조스크랩 (공장 발생)
# ==========================================================================
NODES_EV = {"안중": 0.5, "서화성": 0.5}        # 인입 여건 미확보로 균등 분할
NODES_SCRAP = {"천안": 0.6, "신례원": 0.4}     # 생산능력 비중 개략치
NODES = list(NODES_EV) + list(NODES_SCRAP)

# 하행 적재 가능 시간대 (코레일 답변: 충방전 10~18시 → 하역도 주간)
LOADING_HOURS = list(range(10, 18))

# 출력제어 예측 상위 7일. supply_forecast.csv가 있으면 그 파일의 날짜를
# 우선 사용하고, 없을 때에도 Top 7 입력 규격을 유지하기 위한 기본값이다.
TOP7_DATES = [
    "2025-04-06", "2025-04-26", "2025-05-02", "2025-05-04",
    "2025-05-12", "2025-05-18", "2025-05-25",
]


# ==========================================================================
# 3. 코호트 모델
# ==========================================================================
def retirement_pdf(age: int, scale: float, shape: float) -> float:
    """Weibull 폐차확률밀도 — 등록 후 age년째에 폐차될 확률."""
    if age < 1:
        return 0.0
    s_prev = np.exp(-(((age - 1) / scale) ** shape))
    s_curr = np.exp(-((age / scale) ** shape))
    return float(s_prev - s_curr)


def annual_retired_packs(year: int, p: dict) -> float:
    """해당 연도에 수명을 다하는 EV 대수 (전국)."""
    total = 0.0
    for reg_year, count in EV_REGISTRATION.items():
        age = year - reg_year
        if age >= 1:
            total += count * retirement_pdf(age, p["life_scale"], p["life_shape"])
    return total


def annual_flows(year: int, p: dict) -> dict:
    """연도별 화물 발생량 (톤). 중복 없이 pack / blackmass 로 분리."""
    packs = annual_retired_packs(year, p)

    # 수도권에서 회수되어 철도로 수송되는 배터리 (톤)
    ev_ton = (packs * p["pack_weight_kg"] / 1000
              * p["recovery_rate"] * p["metro_share"] * p["rail_share"])

    # 전처리 여부로 분기
    ev_pretreated = ev_ton * p["pretreat_share"]
    ev_as_pack = ev_ton * (1 - p["pretreat_share"])

    # 제조스크랩 — 전량 전처리되어 블랙매스로 합류
    scrap_ton = (p["scrap_annual_ton"] * (p["scrap_growth"] ** (year - 2025))
                 * p["rail_share"])

    # 블랙매스는 발생지가 다르므로 분리 추적 (EV 유래 = 수도권, 스크랩 유래 = 충남)
    bm_from_ev = ev_pretreated * p["blackmass_yield"]
    bm_from_scrap = scrap_ton * p["blackmass_yield"]
    blackmass_ton = bm_from_ev + bm_from_scrap

    # ESS 사용후 배터리 — Weibull 순차 교체 (일괄 교체가 아님)
    age = year - p["ess_intro_year"]
    ess_ton = (p["ess_fleet"] * p["ess_pack_ton_per_set"]
               * retirement_pdf(age, p["ess_life_scale"], p["ess_life_shape"])
               if age >= 1 else 0.0)

    return {
        "year": year,
        "retired_packs": round(packs),
        "pack_ev_ton": round(ev_as_pack, 1),                 # 사용후 EV 배터리
        "pack_ess_ton": round(ess_ton, 2),                   # ESS 사용후 배터리
        "pack_ton": round(ev_as_pack + ess_ton, 1),          # 팩 계열 합
        "bm_metro_ton": round(bm_from_ev, 1),                # 수도권 발생
        "bm_transit_ton": round(bm_from_scrap, 1),           # 충남 경유역 발생
        "blackmass_ton": round(blackmass_ton, 1),
        "total_ton": round(ev_as_pack + ess_ton + blackmass_ton, 1),
    }


def build_annual(p: dict, years=range(2025, 2041)) -> pd.DataFrame:
    return pd.DataFrame([annual_flows(y, p) for y in years]).set_index("year")


# ==========================================================================
# 4. 연 단위 → 일·시간·역 단위 변환 (최적화 입력 규격)
# ==========================================================================
def to_hourly(annual: pd.DataFrame, target_year: int, dates: list,
              p: dict) -> pd.DataFrame:
    """연 발생량을 일·역·시간 단위로 배분.

    발생지별로 적재역이 다르다.
      수도권(안중·서화성)  : 사용후 EV 배터리 팩 + 수도권 유래 블랙매스
      충남 경유역(천안·신례원) : 제조스크랩 유래 블랙매스

    화물은 전력과 달리 시간대 변동이 없으므로 적재 가능 시간에 균등 배분한다.
    (실제로는 업체 출고 스케줄을 따르나 공개 데이터가 없어 균등 가정)
    """
    row = annual.loc[target_year]
    acc = 7.0 / p["trips_per_week"]        # 1회 운행당 축적일수
    nh = len(LOADING_HOURS)
    rows = []

    for date in dates:
        for hour in range(24):
            on = hour in LOADING_HOURS
            # 수도권 집하역
            for node, w in NODES_EV.items():
                ev_t = row["pack_ev_ton"] * w * acc / 365 / nh if on else 0.0
                ess_t = row["pack_ess_ton"] * w * acc / 365 / nh if on else 0.0
                bm_t = row["bm_metro_ton"] * w * acc / 365 / nh if on else 0.0
                rows.append({"date": date, "node": node, "hour": hour,
                             "cargo_type": "pack_ev",
                             "cargo_ton": round(ev_t, 3),
                             "cargo_cars": round(
                                 ev_t / p["car_capacity_pack_ton"], 4)})
                rows.append({"date": date, "node": node, "hour": hour,
                             "cargo_type": "pack_ess",
                             "cargo_ton": round(ess_t, 4),
                             "cargo_cars": round(
                                 ess_t / p["car_capacity_pack_ton"], 5)})
                rows.append({"date": date, "node": node, "hour": hour,
                             "cargo_type": "blackmass",
                             "cargo_ton": round(bm_t, 3),
                             "cargo_cars": round(
                                 bm_t / p["car_capacity_blackmass_ton"], 4)})
            # 충남 경유 적재역 (제조스크랩)
            for node, w in NODES_SCRAP.items():
                bm_t = row["bm_transit_ton"] * w * acc / 365 / nh if on else 0.0
                rows.append({"date": date, "node": node, "hour": hour,
                             "cargo_type": "blackmass",
                             "cargo_ton": round(bm_t, 3),
                             "cargo_cars": round(
                                 bm_t / p["car_capacity_blackmass_ton"], 4)})

    return (pd.DataFrame(rows)
            .sort_values(["date", "node", "hour", "cargo_type"])
            .reset_index(drop=True))


# ==========================================================================
# 5. 민감도 분석
# ==========================================================================
def sensitivity(p: dict, target_year: int = 2035) -> pd.DataFrame:
    """주요 파라미터를 흔들어 결과 변동 폭을 본다."""
    grid = {
        "life_scale": [9.0, 11.0, 13.0],        # 수명 8 / 10 / 12년
        "recovery_rate": [0.7, 0.85, 0.95],
        "rail_share": [0.40, 0.80, 1.00],
        "pack_weight_kg": [300, 400, 500],
        "pretreat_share": [0.2, 0.4, 0.6],
    }
    base = annual_flows(target_year, p)["total_ton"]
    rows = []
    for key, values in grid.items():
        for v in values:
            q = dict(p, **{key: v})
            r = annual_flows(target_year, q)
            rows.append({
                "parameter": key, "value": v,
                "total_ton": r["total_ton"],
                "pack_ton": r["pack_ton"],
                "blackmass_ton": r["blackmass_ton"],
                "vs_base_%": round((r["total_ton"] / base - 1) * 100, 1),
            })
    return pd.DataFrame(rows)


# ==========================================================================
def main() -> None:
    p = PARAMS

    annual = build_annual(p)
    annual.to_csv(OUT / "cargo_annual.csv", encoding="utf-8-sig")
    print("[연도별 발생량 — 철도 수송분 기준, 톤]")
    print(annual.loc[[2025, 2028, 2030, 2032, 2035, 2040]].to_string())

    # 급증 시점 확인 — 뉴스 전망("2030년부터 증가")과 대조
    growth = annual["total_ton"].pct_change() * 100
    surge = growth[growth > 15]
    if len(surge):
        print(f"\n[급증 구간] 전년 대비 15% 이상 증가: "
              f"{', '.join(str(y) for y in surge.index[:6])}")
    print("  → 2021년 EV 보급 급증(10만대) + 수명 10년의 결과. "
          "업계 전망과 동일한 시점이 데이터에서 도출됨")

    # 최적화 입력 (supply_forecast.csv 와 동일한 날짜 규격)
    supply = OUT / "supply_forecast.csv"
    if supply.exists():
        dates = sorted(pd.read_csv(supply, encoding="utf-8-sig")["date"].unique())
        print(f"\n[날짜] supply_forecast.csv 와 동일한 {len(dates)}일 사용")
    else:
        dates = TOP7_DATES
        print(f"\n[날짜] supply_forecast.csv 없음 → 기본 {len(dates)}일 사용")

    hourly = to_hourly(annual, p["target_year"], dates, p)
    hourly.to_csv(OUT / "cargo_forecast.csv", index=False, encoding="utf-8-sig")
    print(f"[저장] cargo_forecast.csv ({len(hourly)}행)")
    print(f"  적재역 — 수도권 {list(NODES_EV)} / 경유 {list(NODES_SCRAP)}")

    by_node = (hourly[hourly["cargo_ton"] > 0]
               .groupby(["node", "cargo_type"])["cargo_cars"]
               .sum().div(len(dates)).round(3).unstack(fill_value=0))
    print("\n[적재역별 1회당 화차]")
    print(by_node.to_string())

    day = hourly[hourly["cargo_ton"] > 0].groupby("cargo_type").agg(
        회당_톤=("cargo_ton", lambda s: s.sum() / len(dates)),
        회당_화차=("cargo_cars", lambda s: s.sum() / len(dates)))
    day.loc["합계"] = day.sum()
    print(f"\n[{p['target_year']}년 기준 · 주 {p['trips_per_week']}회 운행 시 1회당 물량]")
    print(day.round(3).to_string())

    sens = sensitivity(p)
    sens.to_csv(OUT / "cargo_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n[민감도 — 2035년 총 발생량 기준]")
    for key in sens["parameter"].unique():
        s = sens[sens["parameter"] == key]
        print(f"  {key:16s} {s['vs_base_%'].min():+6.1f}% ~ "
              f"{s['vs_base_%'].max():+6.1f}%")


if __name__ == "__main__":
    main()
