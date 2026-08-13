from __future__ import annotations

from datetime import timedelta
from typing import Optional

from digital_twin.state import SystemState
from forecast import build_forecast_bundle
from optimization import solve_for_state


def run_scenario(
    base_state: SystemState,
    curtailment_pct: float = 0.0,
    demand_pct: float = 0.0,
    cargo_ton: Optional[float] = None,
    delay_min: float = 0.0,
    policy: str = "balanced",
    origin: str = "HONAM_MAIN",
    destination: str = "METRO_MAIN",
) -> dict:
    """operator_v4 대시보드의 what-if 슬라이더(출력제어%, 수요%, 순환자원 대기량t, 지연분)를
    실제 forecast/MILP 계산에 반영해 새 제안안을 만든다.

    프론트엔드의 로컬 수식(calcProposal) 대신 이 결과를 그대로 사용하면 된다.
    """
    state = base_state.clone()

    if delay_min:
        route = state.rail_network.find_route(origin, destination)
        if route:
            _, seg_ids = route
            per_segment = delay_min / max(len(seg_ids), 1)
            for seg_id in seg_ids:
                seg = state.rail_network.get_segment(seg_id)
                seg.set_delay(seg.delay_min + per_segment)

    forecasts = build_forecast_bundle(state)

    # 출력제어%/수요% 슬라이더는 이번 cycle의 '현재값' 자체를 흔드는 what-if로 취급한다.
    forecasts.curtailment.current_mwh = max(0.0, forecasts.curtailment.current_mwh * (1 + curtailment_pct / 100.0))
    forecasts.demand.current_mwh = max(0.0, forecasts.demand.current_mwh * (1 + demand_pct / 100.0))

    if cargo_ton is not None:
        wagon_capacity = float(state.metadata.get("cargo_wagon_capacity_ton", 20.0))
        total_allowed = max(0, int(cargo_ton // wagon_capacity))
        limits = dict(forecasts.cargo.current_dispatch_limit_wagons)
        current_sum = sum(limits.values())
        # 준비된 화물보다 많이 만들어내지는 않되(근거 없는 상향은 금지), 대기량이 줄면 비례 축소한다.
        if current_sum > 0 and total_allowed < current_sum:
            scale = total_allowed / current_sum
            limits = {k: int(v * scale) for k, v in limits.items()}
        forecasts.cargo.current_dispatch_limit_wagons = limits

    _, opt = solve_for_state(state, forecasts, origin, destination, policy=policy)

    departure = state.current_time + timedelta(minutes=60 + delay_min)
    low_soc_flag = opt.diagnostics.get("expected_low_soc_return_candidates", 0)
    risk = "중간" if (opt.return_ess_count <= 9 or (curtailment_pct >= 25 and opt.return_ess_count < 16)) else "낮음"

    return {
        "energy": round(opt.energy_mwh, 1),
        "ess": opt.return_ess_count,
        "cargo": opt.total_cargo_wagons,
        "cargo_wagons_by_type": dict(opt.cargo_wagons_by_type),
        "time": departure.strftime("%H:%M"),
        "delay": int(delay_min),
        "profit": f"₩{round(opt.objective_million_krw, 1)}M",
        "confidence": round((forecasts.curtailment.confidence + forecasts.demand.confidence) / 2 * 100),
        "risk": risk,
        "curt": curtailment_pct,
        "demand": demand_pct,
        "cargo_ton": cargo_ton,
        "low_soc_return_candidates": low_soc_flag,
        "source": "REAL_MILP",
    }
