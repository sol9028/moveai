from __future__ import annotations


def build_briefing(selected: dict, candidates: list[dict], forecasts: dict) -> dict:
    plan = selected["plan"]
    opt = selected["optimization"]
    sim = selected["scenario_summary"]

    cargo_mix = ", ".join(
        f"{k} {v}량" for k, v in plan["cargo_wagons_by_type"].items() if v > 0
    ) or "순환자원 0량"
    extra = "추가 하행편을 투입" if plan["extra_return_train"] else "기본 하행편 1회로 처리"

    headline = (
        f"{plan['policy']} 안 선택: 하행 ESS {plan['return_ess_count']}량 + "
        f"순환자원 {sum(plan['cargo_wagons_by_type'].values())}량, {extra}"
    )
    cargo_forecast = forecasts.get("cargo", {})
    bm_assumption = cargo_forecast.get("assumptions", {}).get("black_mass", {})
    bm_limit = cargo_forecast.get("current_dispatch_limit_wagons", {}).get("black_mass")

    why = [
        f"호남 다음 시점 필요 ESS {opt['diagnostics']['required_honam_ess_next']}량을 고려해 회송량을 계산했습니다.",
        f"수도권 ESS 버퍼와 저SOC 후보 {opt['diagnostics']['expected_low_soc_return_candidates']}량을 동시에 반영했습니다.",
        f"상행 에너지는 예측 출력제어·수도권 수요·충방전 제약의 최소값인 {plan['energy_target_mwh']:.1f}MWh로 제한했습니다.",
        f"하행 순환자원 편성은 {cargo_mix}이며, 준비된 화물이 부족하면 20량을 억지로 채우지 않습니다.",
        (
            f"블랙매스는 주 {bm_assumption.get('dispatches_per_week', 2)}회 × 회당 최대 "
            f"{bm_assumption.get('max_wagons_per_dispatch', bm_limit or 4)}량 공급 가정을 적용해 "
            f"이번 cycle 최대 {bm_limit if bm_limit is not None else 4}량으로 제한했습니다."
        ),
        f"Monte Carlo 시뮬레이션에서 호남 ESS 부족확률 {sim['honam_ess_shortage_probability']:.1%}, 수도권 병목확률 {sim['capital_bottleneck_probability']:.1%}를 확인했습니다.",
    ]

    rejected = []
    for candidate in candidates:
        if candidate["policy"] == selected["policy"]:
            continue
        cplan = candidate["plan"]
        csim = candidate["scenario_summary"]
        reasons = []
        if csim["honam_ess_shortage_probability"] > sim["honam_ess_shortage_probability"] + 1e-9:
            reasons.append("호남 ESS 부족 위험이 더 큼")
        if csim["capital_bottleneck_probability"] > sim["capital_bottleneck_probability"] + 1e-9:
            reasons.append("수도권 ESS 병목 위험이 더 큼")
        if candidate["robust_score"] < selected["robust_score"]:
            reasons.append("위험조정 점수가 더 낮음")
        if cplan["extra_return_train"] and not plan["extra_return_train"]:
            reasons.append("추가 하행편 운행비가 발생")
        if not reasons:
            reasons.append("편익·재고안정·운행리스크의 종합점수가 선택안보다 낮음")
        rejected.append({
            "policy": candidate["policy"],
            "plan_mix": f"ESS {cplan['return_ess_count']} + Cargo {sum(cplan['cargo_wagons_by_type'].values())}",
            "reason": "; ".join(reasons),
        })

    return {
        "headline": headline,
        "why_selected": why,
        "rejected_alternatives": rejected,
        "forecast_context": forecasts,
        "explanation_mode": "deterministic MVP briefing; production can replace this renderer with an LLM",
    }
