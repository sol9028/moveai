from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

from digital_twin.state import SystemState

DEFAULT_TRAVEL_FALLBACK_MIN = 120.0


def _fmt(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def _pick_reason(why_selected: List[str], keywords: List[str], fallback: str) -> str:
    for sentence in why_selected:
        if any(k in sentence for k in keywords):
            return sentence
    return why_selected[0] if why_selected else fallback


def build_day_schedule(
    base_state: SystemState,
    cycle_times: Optional[List[datetime]] = None,
    origin: str = "HONAM_MAIN",
    destination: str = "METRO_MAIN",
) -> List[dict]:
    """하루 여러 시점에 실제 AI Loop(Predict→MILP→Simulate→Decide)를 반복 실행해
    operator_v4 대시보드의 plans[] 스키마(상행 U / 하행 D)로 변환한다.

    목업이 아니라 cycle_time마다 실제 optimize_dispatch 결과를 새로 계산한다.
    """
    from agent.orchestrator import ETrainAgent  # 순환 import 방지용 지연 import

    if cycle_times is None:
        base_day = base_state.current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        cycle_times = [
            base_day.replace(hour=9, minute=0),
            base_day.replace(hour=14, minute=0),
            base_day.replace(hour=20, minute=0),
        ]

    travel_min = base_state.rail_network.travel_time_min(origin, destination)
    if travel_min is None:
        travel_min = DEFAULT_TRAVEL_FALLBACK_MIN

    plans: List[dict] = []
    n = len(cycle_times)

    for idx, cycle_time in enumerate(cycle_times):
        state = base_state.clone()
        state.current_time = cycle_time
        # build_demo_state()에는 E01/E02 두 편성만 등록돼 있어 그 안에서만 순환시킨다.
        state.metadata["primary_train_id"] = "E01" if idx % 2 == 0 else "E02"

        result = ETrainAgent(scenario_count=6, seed=42 + idx).run_once(state, origin=origin, destination=destination)
        selected = result.selected  # dict
        plan = selected["plan"]  # dict (OperationPlan.to_dict())
        briefing = result.briefing  # dict
        forecasts = result.forecasts  # dict

        conf = round(
            (forecasts["curtailment"]["confidence"] + forecasts["demand"]["confidence"]) / 2 * 100
        )

        train_label = plan["train_id"]
        up_depart = datetime.fromisoformat(plan["departure_time"])
        up_arrive = up_depart + timedelta(minutes=travel_min)
        down_depart = up_arrive + timedelta(minutes=plan["discharge_duration_min"] + plan["turnaround_min"])
        down_arrive = down_depart + timedelta(minutes=travel_min)

        total_benefit = plan["expected_net_benefit_million_krw"] or 0.0
        up_benefit = round(total_benefit * 0.64, 1)
        down_benefit = round(total_benefit - up_benefit, 1)

        cargo_total = sum(plan["cargo_wagons_by_type"].values())

        is_last = idx == n - 1
        is_middle_review = n > 2 and idx == n // 2

        plans.append({
            "id": f"U{idx + 1}",
            "dir": "up",
            "train": train_label,
            "time": _fmt(up_depart),
            "end": _fmt(up_arrive),
            "origin": origin,
            "dest": destination,
            "energy": round(plan["energy_target_mwh"], 1),
            "ess": plan["outbound_ess_count"],
            "cargo": 0,
            "profit": f"₩{up_benefit}M",
            "confidence": conf,
            "state": "예정" if is_last else "확정",
            "note": f"ESS {plan['outbound_ess_count']}량 · {round(plan['energy_target_mwh'], 1)}MWh",
            "reason": _pick_reason(
                briefing["why_selected"], ["에너지", "출력제어", "수요"],
                "예측 출력제어·수도권 수요 시점을 고려해 편성했습니다.",
            ),
        })

        # 하루의 마지막 사이클은 하행 회차 없이 상행만(다음날로 이월)으로 표시한다.
        if not is_last:
            plans.append({
                "id": f"D{idx + 1}",
                "dir": "down",
                "train": train_label,
                "time": _fmt(down_depart),
                "end": _fmt(down_arrive),
                "origin": destination,
                "dest": origin,
                "energy": 0,
                "ess": plan["return_ess_count"],
                "cargo": cargo_total,
                "profit": f"₩{down_benefit}M",
                "confidence": conf,
                "state": "검토" if is_middle_review else "확정",
                "note": f"ESS {plan['return_ess_count']} + 순환자원 {cargo_total}",
                "reason": _pick_reason(
                    briefing["why_selected"], ["화종", "순환자원", "블랙매스", "회송"],
                    "호남 ESS 재고 하한과 순환자원 배정을 함께 반영했습니다.",
                ),
            })

    return plans
