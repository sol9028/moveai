from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from digital_twin.state import SystemState
from forecast import ForecastBundle, build_forecast_bundle
from optimization import solve_for_state
from simulation import OperationPlan, run_scenarios, simulate
from .briefing import build_briefing


@dataclass
class AILoopResult:
    state_snapshot: dict
    forecasts: dict
    reoptimization: dict
    candidates: list[dict]
    selected: dict
    final_run: dict
    briefing: dict

    def to_dict(self) -> dict:
        return {
            "state_snapshot": self.state_snapshot,
            "forecasts": self.forecasts,
            "reoptimization": self.reoptimization,
            "candidates": self.candidates,
            "selected": self.selected,
            "final_run": self.final_run,
            "briefing": self.briefing,
        }


class ETrainAgent:
    """Sense → Predict → Simulate/Optimize → Decide → Explain의 MVP 오케스트레이터."""

    def __init__(self, scenario_count: int = 12, seed: int = 42):
        self.scenario_count = scenario_count
        self.seed = seed

    def run_once(
        self,
        state: SystemState,
        origin: str = "HONAM_MAIN",
        destination: str = "METRO_MAIN",
        outbound_ess_count: int = 20,
    ) -> AILoopResult:
        snapshot = state.snapshot()  # Sense
        forecasts = build_forecast_bundle(state)  # Predict
        reopt = self._detect_reoptimization_need(state, forecasts, destination)

        candidates: list[dict] = []
        common_required_honam: int | None = None
        for idx, policy in enumerate(("balanced", "energy_priority", "cargo_priority")):
            milp_input, opt = solve_for_state(
                state,
                forecasts,
                origin_station=origin,
                destination_station=destination,
                policy=policy,
                outbound_ess_count=outbound_ess_count,
            )
            if common_required_honam is None:
                common_required_honam = int(opt.diagnostics["required_honam_ess_next"])
            plan = self._to_operation_plan(
                state, opt, origin, destination, outbound_ess_count, common_required_honam
            )
            batch = run_scenarios(
                state,
                plan,
                n=self.scenario_count,
                horizon_hours=18,
                seed=self.seed + idx * 101,
                step_minutes=10,
            )
            sim_summary = batch.summary()
            robust_score = self._robust_score(opt.objective_million_krw, sim_summary)
            candidates.append({
                "policy": policy,
                "milp_input": milp_input.__dict__,
                "optimization": opt.to_dict(),
                "plan": plan.to_dict(),
                "scenario_summary": sim_summary,
                "robust_score": round(robust_score, 3),
            })

        selected = max(candidates, key=lambda c: c["robust_score"])
        selected_plan = self._plan_from_dict(selected["plan"])
        final = simulate(state, selected_plan, horizon_hours=18, step_minutes=10, snapshot_interval_minutes=60)
        briefing = build_briefing(selected, candidates, forecasts.to_dict())

        return AILoopResult(
            state_snapshot=snapshot,
            forecasts=forecasts.to_dict(),
            reoptimization=reopt,
            candidates=candidates,
            selected=selected,
            final_run={
                "metrics": final.to_dict()["metrics"],
                "event_log": final.events_log,
                "snapshot_count": len(final.snapshots),
                "final_snapshot": final.final_state.snapshot(),
            },
            briefing=briefing,
        )

    def _detect_reoptimization_need(self, state: SystemState, forecasts: ForecastBundle, destination: str) -> dict:
        station = state.stations.get(destination)
        metro_count = state.ess_fleet.inventory_count(destination)
        reasons = []
        if forecasts.curtailment.current_mwh > 0:
            reasons.append("호남 출력제어 에너지 운송 기회 존재")
        if forecasts.demand.current_mwh > 0:
            reasons.append("수도권 ESS 방전 수요 존재")
        if station.ess_utilization(metro_count) >= 0.75:
            reasons.append("수도권 ESS 버퍼 사용률 상승")
        if sum(forecasts.cargo.current_ton.values()) > 0:
            reasons.append("하행 Battery Circular Cargo 대기량 존재")
        if any(seg.delay_min > 0 or seg.status.value != "open" for seg in state.rail_network.all_segments()):
            reasons.append("선로 상태 변화 감지")
        return {"needed": bool(reasons), "reasons": reasons or ["정기 Rolling-Horizon 재계산"]}

    def _to_operation_plan(self, state, opt, origin, destination, outbound_ess_count, common_required_honam) -> OperationPlan:
        departure = state.current_time + timedelta(minutes=60)
        return OperationPlan(
            train_id=state.metadata.get("primary_train_id", "E01"),
            additional_train_id=state.metadata.get("additional_train_id", "E02"),
            departure_time=departure,
            origin=origin,
            destination=destination,
            outbound_ess_count=outbound_ess_count,
            energy_target_mwh=opt.energy_mwh,
            return_ess_count=opt.return_ess_count,
            cargo_wagons_by_type=dict(opt.cargo_wagons_by_type),
            extra_return_train=opt.extra_return_train,
            discharge_duration_min=180,
            turnaround_min=30,
            required_honam_ess_next=int(common_required_honam),
            policy=opt.policy,
            expected_net_benefit_million_krw=opt.objective_million_krw,
        )

    def _plan_from_dict(self, data: dict) -> OperationPlan:
        from datetime import datetime
        payload = dict(data)
        payload.pop("total_return_wagons", None)
        payload["departure_time"] = datetime.fromisoformat(payload["departure_time"])
        return OperationPlan(**payload)

    @staticmethod
    def _robust_score(objective_million_krw: float, sim: dict) -> float:
        return (
            objective_million_krw
            - 12.0 * sim["honam_ess_shortage_probability"]
            - 4.5 * sim["capital_bottleneck_probability"]
            - 2.0 * sim["additional_return_probability"]
            - 3.0 * (1.0 - sim["completion_probability"])
            - 0.006 * sim["average_total_delay_min"]
            + 0.004 * sim["average_cargo_delivered_ton"]
            + 0.002 * sim["average_energy_delivered_mwh"]
        )
