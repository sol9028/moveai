from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil
from typing import Dict

import numpy as np

from digital_twin.cargo import CargoType
from digital_twin.state import SystemState
from digital_twin.station import Region
from forecast import ForecastBundle

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    _SCIPY_MILP = True
except Exception:  # pragma: no cover - fallback is tested conceptually
    _SCIPY_MILP = False


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    cargo_value_multiplier: float
    shortage_penalty_multiplier: float
    overflow_penalty_multiplier: float
    high_soc_return_penalty_multiplier: float
    reserve_bonus_wagons: int


POLICIES: Dict[str, PolicyProfile] = {
    "balanced": PolicyProfile("balanced", 1.00, 1.00, 1.00, 1.00, 2),
    "energy_priority": PolicyProfile("energy_priority", 0.80, 1.55, 1.30, 0.55, 7),
    "cargo_priority": PolicyProfile("cargo_priority", 1.45, 1.00, 0.75, 1.35, -2),
}


@dataclass
class MILPInput:
    origin_station: str
    destination_station: str
    outbound_ess_count: int
    max_wagons_per_train: int
    available_return_ess: int
    expected_low_soc_return_candidates: int
    empty_cargo_wagons: int
    cargo_ton_by_type: dict[str, float]
    cargo_wagon_limit_by_type: dict[str, int]
    cargo_wagon_capacity_ton: float
    honam_ess_before: int
    metro_ess_before: int
    metro_target_capacity: int
    required_honam_ess_next: int
    energy_upper_bound_mwh: float
    extra_train_available: bool


@dataclass
class MILPResult:
    success: bool
    policy: str
    return_ess_count: int
    cargo_wagons_by_type: dict[str, int]
    extra_return_train: bool
    energy_mwh: float
    honam_shortage_wagons: float
    metro_overflow_wagons: float
    high_soc_return_wagons: float
    objective_million_krw: float
    solver: str
    diagnostics: dict

    @property
    def total_cargo_wagons(self) -> int:
        return sum(self.cargo_wagons_by_type.values())

    @property
    def total_return_wagons(self) -> int:
        return self.return_ess_count + self.total_cargo_wagons

    def to_dict(self) -> dict:
        data = asdict(self)
        data["total_cargo_wagons"] = self.total_cargo_wagons
        data["total_return_wagons"] = self.total_return_wagons
        return data


def build_milp_input(
    state: SystemState,
    forecasts: ForecastBundle,
    origin_station: str,
    destination_station: str,
    outbound_ess_count: int = 20,
    max_wagons_per_train: int = 20,
    policy: str = "balanced",
) -> MILPInput:
    if policy not in POLICIES:
        raise KeyError(f"Unknown policy: {policy}")
    profile = POLICIES[policy]
    origin = state.stations.get(origin_station)
    destination = state.stations.get(destination_station)

    honam_before = state.ess_inventory_by_region(Region.HONAM)
    metro_before = state.ess_inventory_by_region(Region.METRO)
    available_return = metro_before + outbound_ess_count

    # 현재 저SOC + 이번 상행 에너지 중 방전될 것으로 예상되는 ESS를 1차 회송 후보로 본다.
    current_low_soc = state.ess_fleet.low_soc_count(destination_station)
    representative = next(iter(state.ess_fleet.all()))
    per_wagon_deliverable = representative.capacity_mwh * representative.discharge_efficiency
    predicted_discharged = min(
        outbound_ess_count,
        int(ceil(forecasts.demand.current_mwh / max(per_wagon_deliverable, 1e-9))),
    )
    expected_low_soc = min(available_return, current_low_soc + predicted_discharged)

    per_wagon_capture = representative.capacity_mwh * representative.charge_efficiency
    required_honam = int(ceil(forecasts.curtailment.next_period_mwh / max(per_wagon_capture, 1e-9)))
    required_honam = max(0, required_honam + profile.reserve_bonus_wagons)

    max_energy_by_wagons = outbound_ess_count * representative.capacity_mwh * representative.charge_efficiency * representative.discharge_efficiency
    charge_window_h = float(state.metadata.get("planning_charge_window_h", 2.0))
    discharge_window_h = float(state.metadata.get("planning_discharge_window_h", 3.0))
    max_energy_by_station = min(
        origin.charging_capacity_mw * charge_window_h * representative.charge_efficiency,
        destination.discharging_capacity_mw * discharge_window_h,
    )
    energy_upper = max(0.0, min(
        forecasts.curtailment.current_mwh,
        forecasts.demand.current_mwh,
        max_energy_by_wagons,
        max_energy_by_station,
    ))

    cargo_ton = dict(forecasts.cargo.current_ton)
    empty_cargo = len(state.cargo_inventory.wagons_at(destination_station, empty_only=True))
    cargo_capacity = float(state.metadata.get("cargo_wagon_capacity_ton", 20.0))

    extra_train_available = any(
        t.current_station == destination_station and t.id != state.metadata.get("primary_train_id", "E01")
        for t in state.trains.all()
    )

    return MILPInput(
        origin_station=origin_station,
        destination_station=destination_station,
        outbound_ess_count=outbound_ess_count,
        max_wagons_per_train=max_wagons_per_train,
        available_return_ess=available_return,
        expected_low_soc_return_candidates=expected_low_soc,
        empty_cargo_wagons=empty_cargo,
        cargo_ton_by_type=cargo_ton,
        cargo_wagon_limit_by_type=dict(forecasts.cargo.current_dispatch_limit_wagons),
        cargo_wagon_capacity_ton=cargo_capacity,
        honam_ess_before=honam_before,
        metro_ess_before=metro_before,
        metro_target_capacity=max(0, int(destination.ess_dock_capacity * 0.90)),
        required_honam_ess_next=required_honam,
        energy_upper_bound_mwh=energy_upper,
        extra_train_available=extra_train_available,
    )


def optimize_dispatch(data: MILPInput, policy: str = "balanced") -> MILPResult:
    if policy not in POLICIES:
        raise KeyError(f"Unknown policy: {policy}")
    profile = POLICIES[policy]

    cargo_types = [
        CargoType.BLACK_MASS.value,
        CargoType.USED_EV_BATTERY.value,
        CargoType.MANUFACTURING_SCRAP.value,
    ]
    # 단순 ton/화차용량뿐 아니라 실제 공급주기에서 이번 cycle에 출고 가능한 화차 수를 함께 제한한다.
    cargo_car_ub = []
    for ct in cargo_types:
        ton_based = int(ceil(data.cargo_ton_by_type.get(ct, 0.0) / data.cargo_wagon_capacity_ton))
        dispatch_limit = int(data.cargo_wagon_limit_by_type.get(ct, ton_based))
        cargo_car_ub.append(min(ton_based, max(0, dispatch_limit)))

    # 변수 순서:
    # 0 return_ess, 1 black_mass, 2 used_battery, 3 mfg_scrap,
    # 4 extra_train(binary), 5 energy(MWh), 6 honam_shortage,
    # 7 metro_overflow, 8 high_soc_return
    honam_base_after_outbound = data.honam_ess_before - data.outbound_ess_count
    required_return = max(0, data.required_honam_ess_next - honam_base_after_outbound)
    metro_excess = max(0, data.metro_ess_before + data.outbound_ess_count - data.metro_target_capacity)
    # 추가 하행편은 화물수익 확대용이 아니라, 한 편(20량)으로 ESS 재고 병목을 해소할 수 없을 때만 허용한다.
    inventory_need_exceeds_one_train = max(required_return, metro_excess) > data.max_wagons_per_train
    extra_ub = 1 if (data.extra_train_available and inventory_need_exceeds_one_train) else 0
    lower = np.zeros(9, dtype=float)
    upper = np.array([
        data.available_return_ess,
        cargo_car_ub[0], cargo_car_ub[1], cargo_car_ub[2],
        extra_ub,
        data.energy_upper_bound_mwh,
        np.inf, np.inf, np.inf,
    ], dtype=float)
    integrality = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int)

    # 백만원 기준 가치/비용. 실제 실증 시 운송단가·SMP·철도운영비 데이터로 교체.
    energy_value = 0.090
    cargo_value_per_ton = {
        CargoType.BLACK_MASS.value: 0.035,
        CargoType.USED_EV_BATTERY.value: 0.050,
        CargoType.MANUFACTURING_SCRAP.value: 0.030,
    }
    extra_train_cost = 3.5
    ess_return_handling_cost = 0.03
    shortage_penalty = 1.80 * profile.shortage_penalty_multiplier
    overflow_penalty = 1.25 * profile.overflow_penalty_multiplier
    high_soc_penalty = 0.45 * profile.high_soc_return_penalty_multiplier

    c = np.zeros(9, dtype=float)
    c[0] = ess_return_handling_cost
    for i, ct in enumerate(cargo_types, start=1):
        c[i] = -cargo_value_per_ton[ct] * data.cargo_wagon_capacity_ton * profile.cargo_value_multiplier
    c[4] = extra_train_cost
    c[5] = -energy_value
    c[6] = shortage_penalty
    c[7] = overflow_penalty
    c[8] = high_soc_penalty

    rows = []
    lbs = []
    ubs = []

    # 총 하행 편성: 기본 20량 + 추가편 20량
    row = np.zeros(9); row[0:4] = 1.0; row[4] = -data.max_wagons_per_train
    rows.append(row); lbs.append(-np.inf); ubs.append(data.max_wagons_per_train)

    # 보유한 빈 화물 화차 수 이상 사용 불가
    row = np.zeros(9); row[1:4] = 1.0
    rows.append(row); lbs.append(-np.inf); ubs.append(data.empty_cargo_wagons)

    # 호남 다음 시점 필요량: x_ess + shortage >= required - (현재 - 상행출발)
    row = np.zeros(9); row[0] = 1.0; row[6] = 1.0
    rows.append(row); lbs.append(required_return); ubs.append(np.inf)

    # 수도권 버퍼: x_ess + overflow >= (현재 + 상행유입 - 목표버퍼)
    row = np.zeros(9); row[0] = 1.0; row[7] = 1.0
    rows.append(row); lbs.append(metro_excess); ubs.append(np.inf)

    # 저SOC 후보를 넘겨 회송하면 high_soc_return slack으로 비용을 부과한다.
    row = np.zeros(9); row[0] = -1.0; row[8] = 1.0
    rows.append(row); lbs.append(-data.expected_low_soc_return_candidates); ubs.append(np.inf)

    constraints = LinearConstraint(np.vstack(rows), np.array(lbs), np.array(ubs)) if _SCIPY_MILP else None

    if _SCIPY_MILP:
        result = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={"time_limit": 5.0},
        )
        if result.x is None:
            return MILPResult(
                False, policy, 0, {ct: 0 for ct in cargo_types}, False, 0.0,
                0.0, 0.0, 0.0, 0.0, "scipy.optimize.milp",
                {"message": str(result.message)},
            )
        x = result.x
        fun = float(result.fun)
        solver = "scipy.optimize.milp"
        success = bool(result.success)
        message = str(result.message)
    else:  # pragma: no cover
        x, fun = _fallback_enumeration(c, lower, upper, data, cargo_car_ub)
        solver = "fallback_enumeration"
        success = True
        message = "SciPy unavailable; exact bounded enumeration used"

    return MILPResult(
        success=success,
        policy=policy,
        return_ess_count=int(round(x[0])),
        cargo_wagons_by_type={ct: int(round(x[i + 1])) for i, ct in enumerate(cargo_types)},
        extra_return_train=bool(round(x[4])),
        energy_mwh=round(float(x[5]), 3),
        honam_shortage_wagons=round(float(x[6]), 3),
        metro_overflow_wagons=round(float(x[7]), 3),
        high_soc_return_wagons=round(float(x[8]), 3),
        objective_million_krw=round(-fun, 3),
        solver=solver,
        diagnostics={
            "message": message,
            "required_honam_ess_next": data.required_honam_ess_next,
            "required_return_for_honam": required_return,
            "metro_excess_before_return": metro_excess,
            "expected_low_soc_return_candidates": data.expected_low_soc_return_candidates,
            "energy_upper_bound_mwh": round(data.energy_upper_bound_mwh, 3),
            "cargo_car_upper_bounds": dict(zip(cargo_types, cargo_car_ub)),
            "cargo_dispatch_limits": dict(data.cargo_wagon_limit_by_type),
        },
    )


def _fallback_enumeration(c, lower, upper, data: MILPInput, cargo_car_ub: list[int]):
    best_x = None
    best_fun = float("inf")
    required_return = max(0, data.required_honam_ess_next - (data.honam_ess_before - data.outbound_ess_count))
    metro_excess = max(0, data.metro_ess_before + data.outbound_ess_count - data.metro_target_capacity)
    for extra in range(int(upper[4]) + 1):
        cap = data.max_wagons_per_train * (1 + extra)
        for ess in range(min(int(upper[0]), cap) + 1):
            remaining = cap - ess
            for b in range(min(cargo_car_ub[0], remaining) + 1):
                for u in range(min(cargo_car_ub[1], remaining - b) + 1):
                    max_s = min(cargo_car_ub[2], remaining - b - u, data.empty_cargo_wagons - b - u)
                    if max_s < 0:
                        continue
                    for s in range(max_s + 1):
                        if b + u + s > data.empty_cargo_wagons:
                            continue
                        shortage = max(0.0, required_return - ess)
                        overflow = max(0.0, metro_excess - ess)
                        high_soc = max(0.0, ess - data.expected_low_soc_return_candidates)
                        x = np.array([ess, b, u, s, extra, data.energy_upper_bound_mwh, shortage, overflow, high_soc], dtype=float)
                        fun = float(c @ x)
                        if fun < best_fun:
                            best_fun, best_x = fun, x
    if best_x is None:
        raise RuntimeError("No feasible solution in fallback enumeration")
    return best_x, best_fun


def solve_for_state(
    state: SystemState,
    forecasts: ForecastBundle,
    origin_station: str,
    destination_station: str,
    policy: str = "balanced",
    outbound_ess_count: int = 20,
) -> tuple[MILPInput, MILPResult]:
    data = build_milp_input(
        state,
        forecasts,
        origin_station=origin_station,
        destination_station=destination_station,
        outbound_ess_count=outbound_ess_count,
        policy=policy,
    )
    return data, optimize_dispatch(data, policy=policy)
