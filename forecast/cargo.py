from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from digital_twin.cargo import CargoType
from digital_twin.state import SystemState
from digital_twin.station import Region


@dataclass
class CargoForecast:
    current_ton: dict[str, float]
    next_period_ton: dict[str, float]
    current_dispatch_limit_wagons: dict[str, int]
    next_dispatch_time: dict[str, str | None]
    weekly_nominal_capacity_ton: dict[str, float | None]
    assumptions: dict[str, dict]
    confidence: float
    source: str = "SIMULATED_FORECAST"

    def to_dict(self) -> dict:
        return {
            "current_ton": dict(self.current_ton),
            "next_period_ton": dict(self.next_period_ton),
            "current_dispatch_limit_wagons": dict(self.current_dispatch_limit_wagons),
            "next_dispatch_time": dict(self.next_dispatch_time),
            "weekly_nominal_capacity_ton": dict(self.weekly_nominal_capacity_ton),
            "assumptions": dict(self.assumptions),
            "confidence": self.confidence,
            "source": self.source,
        }


def predict_cargo(state: SystemState, growth_factors: dict[str, float] | None = None) -> CargoForecast:
    """현재 ready backlog와 반복 공급 스케줄을 분리해 예측한다.

    black_mass처럼 공급 스케줄이 등록된 화종은 단순 성장률로 부풀리지 않고
    다음 dispatch lot의 최대 용량을 다음 기간 공급량으로 사용한다.
    """
    factors = {
        CargoType.BLACK_MASS.value: 1.00,
        CargoType.USED_EV_BATTERY.value: 1.15,
        CargoType.MANUFACTURING_SCRAP.value: 1.10,
    }
    if growth_factors:
        factors.update(growth_factors)

    wagon_capacity = float(state.metadata.get("cargo_wagon_capacity_ton", 20.0))
    current: dict[str, float] = {}
    next_period: dict[str, float] = {}
    dispatch_limits: dict[str, int] = {}
    next_dispatch_time: dict[str, str | None] = {}
    weekly_capacity: dict[str, float | None] = {}
    assumptions: dict[str, dict] = {}

    for cargo_type in CargoType:
        key = cargo_type.value
        ready_ton = state.cargo_backlog_by_region(Region.METRO, cargo_type)
        current[key] = ready_ton
        schedule = state.cargo_inventory.get_supply_schedule(cargo_type)

        if schedule is not None:
            dispatch_limits[key] = state.cargo_inventory.current_dispatch_limit_wagons(
                "METRO_MAIN", state.current_time, cargo_type, schedule.wagon_capacity_ton
            )
            next_period[key] = schedule.max_ton_per_dispatch
            next_dispatch_time[key] = schedule.next_dispatch_time(state.current_time, include_now=False).isoformat()
            weekly_capacity[key] = schedule.weekly_nominal_capacity_ton
            assumptions[key] = schedule.to_dict(state.current_time)
        else:
            dispatch_limits[key] = int(ceil(ready_ton / wagon_capacity)) if ready_ton > 0 else 0
            next_period[key] = max(0.0, ready_ton * factors[key])
            next_dispatch_time[key] = None
            weekly_capacity[key] = None
            assumptions[key] = {
                "source": "SIMULATED_PLACEHOLDER",
                "note": "반복 공급주기 실데이터 미연결; 현재 backlog 기반 가상 예측",
            }

    return CargoForecast(
        current_ton=current,
        next_period_ton=next_period,
        current_dispatch_limit_wagons=dispatch_limits,
        next_dispatch_time=next_dispatch_time,
        weekly_nominal_capacity_ton=weekly_capacity,
        assumptions=assumptions,
        confidence=0.82,
    )
