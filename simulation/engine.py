from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Iterable

from digital_twin.cargo import CargoType
from digital_twin.ess import ESSStatus
from digital_twin.state import SystemState
from digital_twin.train import TrainDirection, TrainStatus, WagonType
from .events import Event, EventQueue, EventType


@dataclass
class OperationPlan:
    train_id: str
    additional_train_id: str | None
    departure_time: datetime
    origin: str
    destination: str
    outbound_ess_count: int
    energy_target_mwh: float
    return_ess_count: int
    cargo_wagons_by_type: dict[str, int]
    extra_return_train: bool = False
    discharge_duration_min: int = 180
    turnaround_min: int = 30
    required_honam_ess_next: int = 0
    policy: str = "balanced"
    expected_net_benefit_million_krw: float | None = None

    @property
    def total_return_wagons(self) -> int:
        return self.return_ess_count + sum(self.cargo_wagons_by_type.values())

    def validate(self, max_wagons: int = 20) -> None:
        if self.outbound_ess_count <= 0 or self.outbound_ess_count > max_wagons:
            raise ValueError("outbound_ess_count must be within train capacity")
        if self.return_ess_count < 0 or any(v < 0 for v in self.cargo_wagons_by_type.values()):
            raise ValueError("return consist counts must be >= 0")
        max_return = max_wagons * (2 if self.extra_return_train else 1)
        if self.total_return_wagons > max_return:
            raise ValueError(f"return consist exceeds capacity {max_return}")
        if self.extra_return_train and not self.additional_train_id:
            raise ValueError("additional_train_id is required when extra_return_train=True")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["departure_time"] = self.departure_time.isoformat()
        data["total_return_wagons"] = self.total_return_wagons
        return data


@dataclass
class SimulationMetrics:
    delivered_energy_mwh: float = 0.0
    cargo_delivered_ton: float = 0.0
    max_metro_ess: int = 0
    max_metro_utilization: float = 0.0
    min_honam_ess: int = 10**9
    capital_bottleneck: bool = False
    honam_ess_shortage: bool = False
    additional_return_needed: bool = False
    train_completed_cycle: bool = False
    total_route_delay_min: float = 0.0
    rail_slot_wait_min: float = 0.0
    constraint_violations: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    final_state: SystemState
    plan: OperationPlan
    snapshots: list[dict]
    events_log: list[dict]
    metrics: SimulationMetrics

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "final_state": self.final_state.to_dict(),
            "snapshots": self.snapshots,
            "events_log": self.events_log,
            "metrics": asdict(self.metrics),
        }


class SimulationEngine:
    def __init__(self, step_minutes: int = 10, snapshot_interval_minutes: int = 60):
        if step_minutes <= 0:
            raise ValueError("step_minutes must be > 0")
        self.step_minutes = int(step_minutes)
        self.snapshot_interval_minutes = int(snapshot_interval_minutes)

    def run(
        self,
        initial_state: SystemState,
        plan: OperationPlan,
        horizon_hours: int = 24,
        events: Iterable[Event] | None = None,
    ) -> SimulationResult:
        state = initial_state.clone()
        state.validate()
        primary = state.trains.get(plan.train_id)
        plan.validate(primary.max_wagons)
        if plan.origin not in {s.id for s in state.stations.all()} or plan.destination not in {s.id for s in state.stations.all()}:
            raise KeyError("Unknown station in plan")

        queue = EventQueue(events)
        event_log: list[dict] = []
        snapshots: list[dict] = []
        metrics = SimulationMetrics()
        end_time = state.current_time + timedelta(hours=horizon_hours)
        next_snapshot = state.current_time
        dt_h = self.step_minutes / 60.0

        # 상행에 사용할 ESS: SOC가 높은 가용 화차를 우선하되 충전 가능한 화차 전체를 허용.
        outbound_candidates = sorted(
            state.ess_fleet.at_location(plan.origin, available_only=True),
            key=lambda w: w.soc_percent,
            reverse=True,
        )
        if len(outbound_candidates) < plan.outbound_ess_count:
            raise RuntimeError(
                f"Not enough ESS at {plan.origin}: required={plan.outbound_ess_count}, available={len(outbound_candidates)}"
            )
        outbound_ids = [w.id for w in outbound_candidates[: plan.outbound_ess_count]]

        phase = "charging"
        energy_remaining = plan.energy_target_mwh
        discharge_started_at: datetime | None = None
        return_ready_time: datetime | None = None
        return_schedules: dict[str, dict] = {}
        return_composed = False
        outbound_reserved_departure: datetime | None = None
        outbound_route_segments: list[str] = []
        outbound_travel_time = 0.0

        while state.current_time <= end_time:
            for event in queue.pop_due(state.current_time):
                self._apply_event(state, event, event_log)

            if phase == "charging":
                self._charge_selected(state, outbound_ids, plan.origin, dt_h)
                if state.current_time >= plan.departure_time:
                    try:
                        if outbound_reserved_departure is None:
                            outbound_reserved_departure = state.rail_network.next_available_departure(
                                plan.origin, plan.destination, state.current_time
                            )
                            outbound_route_segments, outbound_travel_time = state.rail_network.reserve_route(
                                plan.origin, plan.destination, outbound_reserved_departure
                            )
                            metrics.rail_slot_wait_min += max(
                                0.0, (outbound_reserved_departure - state.current_time).total_seconds() / 60.0
                            )
                        if state.current_time >= outbound_reserved_departure:
                            available_selected = [eid for eid in outbound_ids if state.ess_fleet.get(eid).available]
                            if len(available_selected) < plan.outbound_ess_count:
                                replacements = [
                                    w.id for w in sorted(
                                        state.ess_fleet.at_location(plan.origin, available_only=True),
                                        key=lambda w: w.soc_percent, reverse=True
                                    )
                                    if w.id not in available_selected
                                ]
                                available_selected.extend(replacements[: plan.outbound_ess_count - len(available_selected)])
                            if len(available_selected) < plan.outbound_ess_count:
                                raise RuntimeError("ESS failure left insufficient outbound wagons")
                            outbound_ids = available_selected[: plan.outbound_ess_count]
                            primary.clear_wagons()
                            for eid in outbound_ids:
                                primary.attach_wagon(WagonType.ESS, eid)
                                state.ess_fleet.get(eid).move_to(primary.id, primary.id)
                            primary.depart(
                                plan.origin,
                                plan.destination,
                                outbound_reserved_departure,
                                outbound_travel_time,
                                outbound_route_segments,
                                TrainDirection.UPBOUND,
                            )
                            phase = "outbound"
                            event_log.append({"time": state.current_time.isoformat(), "type": "UPBOUND_DEPART", "train": primary.id})
                    except RuntimeError:
                        pass

            elif phase == "outbound":
                self._advance_train_with_dynamic_delay(state, primary, metrics)
                if primary.remaining_travel_min <= 1e-9:
                    primary.arrive(plan.destination)
                    for eid in list(primary.ess_wagon_ids()):
                        state.ess_fleet.get(eid).move_to(plan.destination, None)
                    primary.clear_wagons()
                    primary.status = TrainStatus.IDLE
                    discharge_started_at = state.current_time
                    energy_remaining = min(plan.energy_target_mwh, state.stations.get(plan.destination).demand_mwh)
                    phase = "discharging"
                    event_log.append({"time": state.current_time.isoformat(), "type": "UPBOUND_ARRIVE", "train": primary.id})

            elif phase == "discharging":
                delivered = self._discharge_sequential(
                    state, outbound_ids, plan.destination, energy_remaining, dt_h
                )
                energy_remaining = max(0.0, energy_remaining - delivered)
                metrics.delivered_energy_mwh += delivered
                destination = state.stations.get(plan.destination)
                destination.demand_mwh = max(0.0, destination.demand_mwh - delivered)

                elapsed = (state.current_time - discharge_started_at).total_seconds() / 60.0 if discharge_started_at else 0.0
                if energy_remaining <= 1e-6 or elapsed >= plan.discharge_duration_min:
                    return_ready_time = state.current_time + timedelta(minutes=plan.turnaround_min)
                    phase = "turnaround"

            elif phase == "turnaround":
                if state.current_time >= return_ready_time and not return_composed:
                    return_schedules = self._compose_and_schedule_return(state, plan, metrics)
                    return_composed = True
                    phase = "returning"

            elif phase == "returning":
                all_done = True
                for train_id, sched in return_schedules.items():
                    train = state.trains.get(train_id)
                    if sched["arrived"]:
                        continue
                    all_done = False
                    if not sched["departed"] and state.current_time >= sched["departure"]:
                        for eid in train.ess_wagon_ids():
                            state.ess_fleet.get(eid).move_to(train.id, train.id)
                        for cid in train.cargo_wagon_ids():
                            state.cargo_inventory.get_wagon(cid).move_to(train.id, train.id)
                        train.depart(
                            plan.destination,
                            plan.origin,
                            sched["departure"],
                            sched["travel_time"],
                            sched["segments"],
                            TrainDirection.DOWNBOUND,
                        )
                        sched["departed"] = True
                        event_log.append({"time": state.current_time.isoformat(), "type": "DOWNBOUND_DEPART", "train": train_id})

                    if sched["departed"] and not sched["arrived"]:
                        self._advance_train_with_dynamic_delay(state, train, metrics)
                        if train.remaining_travel_min <= 1e-9:
                            train.arrive(plan.origin)
                            for eid in list(train.ess_wagon_ids()):
                                state.ess_fleet.get(eid).move_to(plan.origin, None)
                            for cid in list(train.cargo_wagon_ids()):
                                wagon = state.cargo_inventory.get_wagon(cid)
                                wagon.move_to(plan.origin, None)
                                metrics.cargo_delivered_ton += state.cargo_inventory.unload_wagon(cid)
                            train.clear_wagons()
                            train.status = TrainStatus.IDLE
                            sched["arrived"] = True
                            event_log.append({"time": state.current_time.isoformat(), "type": "DOWNBOUND_ARRIVE", "train": train_id})

                if return_schedules and all(s["arrived"] for s in return_schedules.values()):
                    metrics.train_completed_cycle = True
                    phase = "complete"

            self._update_metrics(state, plan, metrics)

            if state.current_time >= next_snapshot:
                snap = state.snapshot()
                snap["phase"] = phase
                snap["metrics"] = {
                    "delivered_energy_mwh": round(metrics.delivered_energy_mwh, 3),
                    "cargo_delivered_ton": round(metrics.cargo_delivered_ton, 3),
                    "capital_bottleneck": metrics.capital_bottleneck,
                    "honam_ess_shortage": metrics.honam_ess_shortage,
                }
                snapshots.append(snap)
                next_snapshot += timedelta(minutes=self.snapshot_interval_minutes)

            if phase == "complete":
                break
            state.advance_time(self.step_minutes)

        self._update_metrics(state, plan, metrics)
        state.validate()
        return SimulationResult(state, plan, snapshots, event_log, metrics)

    def _charge_selected(self, state: SystemState, ess_ids: list[str], station_id: str, dt_h: float) -> float:
        station = state.stations.get(station_id)
        if not station.charging_available or station.charging_capacity_mw <= 0:
            return 0.0
        grid_budget = min(station.curtailment_available_mwh, station.charging_capacity_mw * dt_h)
        used_total = 0.0
        for eid in ess_ids:
            if grid_budget <= 1e-9:
                break
            wagon = state.ess_fleet.get(eid)
            used = wagon.charge(grid_budget, dt_h)
            used_total += used
            grid_budget -= used
        station.curtailment_available_mwh = max(0.0, station.curtailment_available_mwh - used_total)
        return used_total

    def _discharge_sequential(
        self,
        state: SystemState,
        ess_ids: list[str],
        station_id: str,
        remaining_demand_mwh: float,
        dt_h: float,
    ) -> float:
        station = state.stations.get(station_id)
        if not station.discharging_available or station.discharging_capacity_mw <= 0:
            return 0.0
        station_budget = min(remaining_demand_mwh, station.discharging_capacity_mw * dt_h)
        delivered_total = 0.0
        # 앞 화차부터 깊게 방전하여 일부는 저SOC 회송, 일부는 수도권 잔류가 가능하도록 한다.
        for eid in ess_ids:
            if station_budget <= 1e-9:
                break
            wagon = state.ess_fleet.get(eid)
            delivered = wagon.discharge(station_budget, dt_h)
            delivered_total += delivered
            station_budget -= delivered
        return delivered_total

    def _compose_and_schedule_return(self, state: SystemState, plan: OperationPlan, metrics: SimulationMetrics) -> dict[str, dict]:
        primary = state.trains.get(plan.train_id)
        primary.current_station = plan.destination
        primary.clear_wagons()
        trains = [primary]
        if plan.extra_return_train:
            secondary = state.trains.get(plan.additional_train_id)
            secondary.current_station = plan.destination
            secondary.clear_wagons()
            trains.append(secondary)

        selected_ess = state.select_return_ess(plan.destination, plan.return_ess_count)
        if len(selected_ess) < plan.return_ess_count:
            metrics.constraint_violations.append(
                f"return ESS shortage: planned={plan.return_ess_count}, selected={len(selected_ess)}"
            )

        cargo_ids: list[str] = []
        for cargo_type_value in (
            CargoType.BLACK_MASS.value,
            CargoType.USED_EV_BATTERY.value,
            CargoType.MANUFACTURING_SCRAP.value,
        ):
            requested = int(plan.cargo_wagons_by_type.get(cargo_type_value, 0))
            ids, _ = state.cargo_inventory.allocate_to_wagons(
                plan.destination,
                plan.origin,
                state.current_time,
                CargoType(cargo_type_value),
                requested,
            )
            cargo_ids.extend(ids)
            if len(ids) < requested:
                metrics.constraint_violations.append(
                    f"cargo wagon shortage {cargo_type_value}: planned={requested}, loaded={len(ids)}"
                )

        consist = [(WagonType.ESS, w.id) for w in selected_ess] + [(WagonType.CARGO, cid) for cid in cargo_ids]
        schedules: dict[str, dict] = {}
        offset = 0
        cursor = state.current_time
        for train in trains:
            slots = consist[offset: offset + train.max_wagons]
            offset += len(slots)
            if not slots:
                continue
            for wtype, wid in slots:
                train.attach_wagon(wtype, wid)
            earliest = cursor
            departure = state.rail_network.next_available_departure(plan.destination, plan.origin, earliest)
            segments, travel = state.rail_network.reserve_route(plan.destination, plan.origin, departure)
            metrics.rail_slot_wait_min += max(0.0, (departure - earliest).total_seconds() / 60.0)
            schedules[train.id] = {
                "departure": departure,
                "segments": segments,
                "travel_time": travel,
                "departed": False,
                "arrived": False,
            }
            cursor = departure + timedelta(minutes=5)

        if offset < len(consist):
            metrics.constraint_violations.append(f"unassigned return wagons: {len(consist) - offset}")
        return schedules

    def _advance_train_with_dynamic_delay(self, state: SystemState, train, metrics: SimulationMetrics) -> None:
        current_delay = state.rail_network.route_delay_min(train.route_segment_ids)
        if current_delay > train.applied_route_delay_min:
            extra = current_delay - train.applied_route_delay_min
            train.remaining_travel_min += extra
            train.applied_route_delay_min = current_delay
            metrics.total_route_delay_min += extra
        train.remaining_travel_min = max(0.0, train.remaining_travel_min - self.step_minutes)

    def _apply_event(self, state: SystemState, event: Event, log: list[dict]) -> None:
        p = event.payload
        if event.type in (EventType.TRACK_DELAY, EventType.TRACK_BLOCKED, EventType.TRACK_RECOVERED):
            segment_ids = []
            if "segment_id" in p:
                segment_ids = [p["segment_id"]]
            elif "origin" in p and "destination" in p:
                route = state.rail_network.find_route(p["origin"], p["destination"])
                if route:
                    _, segment_ids = route
            for sid in segment_ids:
                seg = state.rail_network.get_segment(sid)
                if event.type == EventType.TRACK_DELAY:
                    seg.set_delay(float(p.get("delay_min", 0.0)))
                elif event.type == EventType.TRACK_BLOCKED:
                    seg.block()
                else:
                    seg.recover()
        elif event.type == EventType.CURTAILMENT_UPDATED:
            st = state.stations.get(p["station_id"])
            st.curtailment_available_mwh = float(p.get("value_mwh", st.curtailment_available_mwh * float(p.get("factor", 1.0))))
        elif event.type == EventType.DEMAND_UPDATED:
            st = state.stations.get(p["station_id"])
            st.demand_mwh = float(p.get("value_mwh", st.demand_mwh * float(p.get("factor", 1.0))))
        elif event.type == EventType.CARGO_READY:
            batch = state.cargo_inventory.get_batch(p["batch_id"])
            batch.add_ready_amount(float(p.get("additional_ton", 0.0)))
        elif event.type == EventType.ESS_FAILURE:
            state.ess_fleet.get(p["ess_id"]).fail()
        elif event.type == EventType.ESS_RECOVERED:
            state.ess_fleet.get(p["ess_id"]).recover()
        log.append({"time": event.time.isoformat(), "type": event.type.value, "payload": p})

    def _update_metrics(self, state: SystemState, plan: OperationPlan, metrics: SimulationMetrics) -> None:
        honam_count = state.ess_fleet.inventory_count(plan.origin)
        metro_count = state.ess_fleet.inventory_count(plan.destination)
        metro_station = state.stations.get(plan.destination)
        util = metro_station.ess_utilization(metro_count)
        metrics.max_metro_ess = max(metrics.max_metro_ess, metro_count)
        metrics.max_metro_utilization = max(metrics.max_metro_utilization, util)
        metrics.min_honam_ess = min(metrics.min_honam_ess, honam_count)
        metrics.capital_bottleneck = metrics.capital_bottleneck or util >= 0.90
        if plan.required_honam_ess_next > 0:
            shortage = honam_count < plan.required_honam_ess_next
            # 다음 충전 시점 준비 여부는 최종/현재 재고 기준으로 판단한다.
            # 상행 출발 직후의 일시적 감소를 하루 전체의 실패로 누적하지 않는다.
            metrics.honam_ess_shortage = shortage
            metrics.additional_return_needed = shortage


def simulate(
    initial_state: SystemState,
    plan: OperationPlan,
    horizon_hours: int = 24,
    events: Iterable[Event] | None = None,
    step_minutes: int = 10,
    snapshot_interval_minutes: int = 60,
) -> SimulationResult:
    return SimulationEngine(step_minutes, snapshot_interval_minutes).run(
        initial_state, plan, horizon_hours=horizon_hours, events=events
    )
