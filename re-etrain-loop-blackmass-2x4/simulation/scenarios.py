from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from statistics import mean

from digital_twin import (
    CargoBatch,
    CargoInventoryManager,
    CargoType,
    CargoSupplySchedule,
    CargoWagon,
    ESSFleetManager,
    ESSWagon,
    RailNetwork,
    RailSegment,
    Region,
    Station,
    StationRegistry,
    StationRole,
    SystemState,
    Train,
    TrainRegistry,
)
from .events import Event, EventType
from .engine import OperationPlan, SimulationResult, simulate


@dataclass
class ScenarioSpec:
    name: str
    curtailment_factor: float = 1.0
    demand_factor: float = 1.0
    cargo_additional_ton: float = 0.0
    outbound_delay_min: float = 0.0
    return_delay_min: float = 0.0
    ess_failure_ids: tuple[str, ...] = ()


@dataclass
class ScenarioBatchResult:
    results: list[SimulationResult]
    scenario_specs: list[ScenarioSpec]
    capital_bottleneck_probability: float
    honam_ess_shortage_probability: float
    additional_return_probability: float
    completion_probability: float
    average_energy_delivered_mwh: float
    average_cargo_delivered_ton: float
    average_total_delay_min: float

    def summary(self) -> dict:
        return {
            "n": len(self.results),
            "capital_bottleneck_probability": round(self.capital_bottleneck_probability, 4),
            "honam_ess_shortage_probability": round(self.honam_ess_shortage_probability, 4),
            "additional_return_probability": round(self.additional_return_probability, 4),
            "completion_probability": round(self.completion_probability, 4),
            "average_energy_delivered_mwh": round(self.average_energy_delivered_mwh, 3),
            "average_cargo_delivered_ton": round(self.average_cargo_delivered_ton, 3),
            "average_total_delay_min": round(self.average_total_delay_min, 3),
        }


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def generate_scenarios(base_state: SystemState, plan: OperationPlan, n: int = 24, seed: int = 42) -> list[ScenarioSpec]:
    if n <= 0:
        raise ValueError("n must be > 0")
    rng = random.Random(seed)
    origin_available = [w.id for w in base_state.ess_fleet.at_location(plan.origin, available_only=True)]
    specs = []
    for i in range(n):
        failures: tuple[str, ...] = ()
        if origin_available and rng.random() < 0.08:
            failures = (rng.choice(origin_available),)
        specs.append(
            ScenarioSpec(
                name=f"scenario_{i + 1:03d}",
                curtailment_factor=_clip(rng.gauss(1.0, 0.12), 0.70, 1.35),
                demand_factor=_clip(rng.gauss(1.0, 0.10), 0.75, 1.30),
                # 블랙매스는 주 2회 × 회당 최대 4량 공급 제약을 사용하므로
                # 같은 dispatch window 안에서 임의의 추가 발생량을 만들지 않는다.
                cargo_additional_ton=0.0,
                outbound_delay_min=rng.choice([20, 30, 45, 60]) if rng.random() < 0.14 else 0.0,
                return_delay_min=rng.choice([20, 30, 45, 60]) if rng.random() < 0.12 else 0.0,
                ess_failure_ids=failures,
            )
        )
    return specs


def _events_for_scenario(state: SystemState, plan: OperationPlan, spec: ScenarioSpec) -> list[Event]:
    t0 = state.current_time
    events = [
        Event(
            t0 + timedelta(minutes=15),
            type=EventType.CURTAILMENT_UPDATED,
            payload={"station_id": plan.origin, "factor": spec.curtailment_factor},
        ),
        Event(
            t0 + timedelta(minutes=15),
            type=EventType.DEMAND_UPDATED,
            payload={"station_id": plan.destination, "factor": spec.demand_factor},
        ),
    ]
    black_mass = next(
        (
            b for b in state.cargo_inventory.all_batches()
            if b.cargo_type == CargoType.BLACK_MASS
            and b.origin_station == plan.destination
            and b.destination_station == plan.origin
        ),
        None,
    )
    if black_mass and spec.cargo_additional_ton > 0:
        events.append(
            Event(
                t0 + timedelta(hours=2),
                type=EventType.CARGO_READY,
                payload={"batch_id": black_mass.id, "additional_ton": spec.cargo_additional_ton},
            )
        )
    if spec.outbound_delay_min > 0:
        events.append(
            Event(
                t0 + timedelta(minutes=30),
                type=EventType.TRACK_DELAY,
                payload={"origin": plan.origin, "destination": plan.destination, "delay_min": spec.outbound_delay_min},
            )
        )
    if spec.return_delay_min > 0:
        events.append(
            Event(
                t0 + timedelta(hours=3),
                type=EventType.TRACK_DELAY,
                payload={"origin": plan.destination, "destination": plan.origin, "delay_min": spec.return_delay_min},
            )
        )
    for ess_id in spec.ess_failure_ids:
        events.append(
            Event(t0 + timedelta(minutes=20), type=EventType.ESS_FAILURE, payload={"ess_id": ess_id})
        )
    return events


def run_scenarios(
    base_state: SystemState,
    plan: OperationPlan,
    n: int = 24,
    horizon_hours: int = 24,
    seed: int = 42,
    step_minutes: int = 10,
) -> ScenarioBatchResult:
    specs = generate_scenarios(base_state, plan, n=n, seed=seed)
    results: list[SimulationResult] = []
    for spec in specs:
        result = simulate(
            base_state,
            plan,
            horizon_hours=horizon_hours,
            events=_events_for_scenario(base_state.clone(), plan, spec),
            step_minutes=step_minutes,
            snapshot_interval_minutes=120,
        )
        results.append(result)

    count = len(results)
    return ScenarioBatchResult(
        results=results,
        scenario_specs=specs,
        capital_bottleneck_probability=sum(r.metrics.capital_bottleneck for r in results) / count,
        honam_ess_shortage_probability=sum(r.metrics.honam_ess_shortage for r in results) / count,
        additional_return_probability=sum(r.metrics.additional_return_needed for r in results) / count,
        completion_probability=sum(r.metrics.train_completed_cycle for r in results) / count,
        average_energy_delivered_mwh=mean(r.metrics.delivered_energy_mwh for r in results),
        average_cargo_delivered_ton=mean(r.metrics.cargo_delivered_ton for r in results),
        average_total_delay_min=mean(r.metrics.total_route_delay_min for r in results),
    )


def build_demo_state(now: datetime | None = None) -> SystemState:
    """실제 데이터 연동 전 구조 검증용 가상 상태. 사용자 기획의 20량 상행/가변 하행 구조를 재현한다."""
    now = now or datetime(2026, 8, 13, 9, 0, 0)

    stations = StationRegistry()
    stations.register(
        Station(
            id="HONAM_MAIN",
            name="호남 재생에너지 거점",
            region=Region.HONAM,
            role=StationRole.SOURCE,
            charging_capacity_mw=120.0,
            discharging_capacity_mw=20.0,
            platform_capacity=3,
            ess_dock_capacity=70,
            cargo_dock_capacity=50,
            curtailment_available_mwh=190.0,
            demand_mwh=0.0,
        )
    )
    stations.register(
        Station(
            id="METRO_MAIN",
            name="수도권 전력수요 거점",
            region=Region.METRO,
            role=StationRole.SINK,
            charging_capacity_mw=20.0,
            discharging_capacity_mw=120.0,
            platform_capacity=3,
            ess_dock_capacity=60,
            cargo_dock_capacity=60,
            curtailment_available_mwh=0.0,
            demand_mwh=115.0,
        )
    )
    stations.register(
        Station(
            id="MID_HUB",
            name="중간 환적 허브",
            region=Region.HUB,
            role=StationRole.HUB,
            platform_capacity=4,
            ess_dock_capacity=30,
            cargo_dock_capacity=30,
        )
    )

    ess = ESSFleetManager()
    # 호남 44량: 상행 20량을 보내도 24량이 남는다.
    for i in range(1, 45):
        ess.register(
            ESSWagon(
                id=f"ESS_H_{i:03d}",
                capacity_mwh=10.0,
                soc_percent=76.0 if i <= 24 else 48.0,
                current_location="HONAM_MAIN",
                max_charge_rate_mw=5.0,
                max_discharge_rate_mw=5.0,
            )
        )
    # 수도권 30량 중 일부는 이미 저SOC 상태.
    for i in range(1, 31):
        soc = 12.0 if i <= 5 else (38.0 + (i % 5) * 6.0)
        ess.register(
            ESSWagon(
                id=f"ESS_M_{i:03d}",
                capacity_mwh=10.0,
                soc_percent=soc,
                current_location="METRO_MAIN",
                max_charge_rate_mw=5.0,
                max_discharge_rate_mw=5.0,
            )
        )

    cargo = CargoInventoryManager()
    # 현장 운영 가정 반영: 블랙매스는 주 2회, 회당 약 4량만 채워짐.
    # 화차 1량=20t 가정 → 80t/회, 주간 명목 최대 160t.
    # 실제 출고 요일은 아직 미확정이므로 데모에서는 월/목(0,3)을 사용하며 설정값으로 교체 가능.
    cargo.register_supply_schedule(
        CargoSupplySchedule(
            cargo_type=CargoType.BLACK_MASS,
            dispatches_per_week=2,
            max_wagons_per_dispatch=4,
            wagon_capacity_ton=20.0,
            dispatch_weekdays=(0, 3),
            dispatch_hour=9,
            source="FIELD_ASSUMPTION_WEEKLY_2X_4_WAGONS",
            note="주 2회 × 회당 약 4량은 반영 완료; 월/목 요일은 데모용 임시값",
        )
    )
    # 현재 스냅샷은 한 번의 출고 lot(4량=80t)이 준비된 시점으로 둔다.
    cargo.register_batch(CargoBatch("BM_20260813_A", CargoType.BLACK_MASS, 80.0, "METRO_MAIN", "HONAM_MAIN", now, priority=3))
    # 아래 두 화종은 아직 실물량 정보가 없으므로 기존 가상값을 유지한다.
    cargo.register_batch(CargoBatch("USED_001", CargoType.USED_EV_BATTERY, 40.0, "METRO_MAIN", "HONAM_MAIN", now, priority=2))
    cargo.register_batch(CargoBatch("SCRAP_001", CargoType.MANUFACTURING_SCRAP, 40.0, "METRO_MAIN", "HONAM_MAIN", now, priority=1))
    for i in range(1, 25):
        cargo.register_wagon(CargoWagon(f"CW_{i:03d}", capacity_ton=20.0, current_location="METRO_MAIN"))

    rail = RailNetwork()
    # 다중 구간으로 만들어 경로탐색/지연/slot 제약이 실제로 작동하도록 구성.
    rail.add_segment(RailSegment("SEG_HONAM_HUB", "HONAM_MAIN", "MID_HUB", 150.0, 150.0, capacity_per_hour=2))
    rail.add_segment(RailSegment("SEG_HUB_METRO", "MID_HUB", "METRO_MAIN", 150.0, 150.0, capacity_per_hour=2))

    trains = TrainRegistry()
    trains.register(Train("E01", "LOCO_01", current_station="HONAM_MAIN", max_wagons=20))
    trains.register(Train("E02", "LOCO_02", current_station="METRO_MAIN", max_wagons=20))

    state = SystemState(
        current_time=now,
        stations=stations,
        trains=trains,
        rail_network=rail,
        ess_fleet=ess,
        cargo_inventory=cargo,
        metadata={
            "data_source": "SIMULATED",
            "primary_train_id": "E01",
            "additional_train_id": "E02",
            "curtailment_growth_factor": 1.68,
            "demand_growth_factor": 1.08,
            "planning_charge_window_h": 2.0,
            "planning_discharge_window_h": 3.0,
            "cargo_wagon_capacity_ton": 20.0,
            "black_mass_supply_assumption": {
                "dispatches_per_week": 2,
                "max_wagons_per_dispatch": 4,
                "wagon_capacity_ton": 20.0,
                "max_ton_per_dispatch": 80.0,
                "weekly_nominal_capacity_ton": 160.0,
                "dispatch_weekdays_demo": [0, 3],
                "weekday_note": "월/목은 데모용 임시값; 실제 출고요일 확인 시 교체",
            },
        },
    )
    state.validate()
    return state
