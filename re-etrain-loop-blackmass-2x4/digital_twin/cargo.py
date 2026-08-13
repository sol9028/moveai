from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class CargoType(str, Enum):
    BLACK_MASS = "black_mass"
    USED_EV_BATTERY = "used_ev_battery"
    MANUFACTURING_SCRAP = "manufacturing_scrap"


class CargoBatchStatus(str, Enum):
    WAITING = "waiting"
    PARTIAL = "partial"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"


@dataclass(frozen=True)
class CargoSupplySchedule:
    """특정 화종의 반복 발생/출고 가능 패턴.

    dispatch_weekdays는 Python weekday 규칙(월=0 ... 일=6)을 사용한다.
    이 객체는 '열차가 그 요일에만 운행 가능'하다는 뜻이 아니라,
    해당 화물이 통상적으로 언제/얼마나 출고 가능한 lot으로 준비되는지를 나타낸다.
    """

    cargo_type: CargoType
    dispatches_per_week: int
    max_wagons_per_dispatch: int
    wagon_capacity_ton: float
    dispatch_weekdays: tuple[int, ...]
    dispatch_hour: int = 9
    source: str = "SIMULATED_OPERATION_ASSUMPTION"
    note: str = ""

    def __post_init__(self) -> None:
        if self.dispatches_per_week <= 0:
            raise ValueError("dispatches_per_week must be > 0")
        if self.max_wagons_per_dispatch <= 0:
            raise ValueError("max_wagons_per_dispatch must be > 0")
        if self.wagon_capacity_ton <= 0:
            raise ValueError("wagon_capacity_ton must be > 0")
        if not self.dispatch_weekdays:
            raise ValueError("dispatch_weekdays must not be empty")
        if len(set(self.dispatch_weekdays)) != len(self.dispatch_weekdays):
            raise ValueError("dispatch_weekdays must be unique")
        if any(day < 0 or day > 6 for day in self.dispatch_weekdays):
            raise ValueError("dispatch_weekdays must use 0(Mon)..6(Sun)")
        if not 0 <= self.dispatch_hour <= 23:
            raise ValueError("dispatch_hour must be 0..23")

    @property
    def max_ton_per_dispatch(self) -> float:
        return self.max_wagons_per_dispatch * self.wagon_capacity_ton

    @property
    def weekly_nominal_capacity_ton(self) -> float:
        return self.dispatches_per_week * self.max_ton_per_dispatch

    def is_dispatch_day(self, when: datetime) -> bool:
        return when.weekday() in self.dispatch_weekdays

    def next_dispatch_time(self, after: datetime, include_now: bool = False) -> datetime:
        """after 이후 가장 가까운 공급 lot 준비 시각을 반환."""
        for days_ahead in range(0, 8):
            candidate_date = (after + timedelta(days=days_ahead)).date()
            candidate = datetime.combine(candidate_date, datetime.min.time()).replace(hour=self.dispatch_hour)
            if candidate.weekday() not in self.dispatch_weekdays:
                continue
            if candidate > after or (include_now and candidate >= after):
                return candidate
        raise RuntimeError("could not resolve next dispatch within 7 days")

    def to_dict(self, now: Optional[datetime] = None) -> dict:
        data = {
            "cargo_type": self.cargo_type.value,
            "dispatches_per_week": self.dispatches_per_week,
            "max_wagons_per_dispatch": self.max_wagons_per_dispatch,
            "wagon_capacity_ton": self.wagon_capacity_ton,
            "max_ton_per_dispatch": self.max_ton_per_dispatch,
            "weekly_nominal_capacity_ton": self.weekly_nominal_capacity_ton,
            "dispatch_weekdays": list(self.dispatch_weekdays),
            "dispatch_hour": self.dispatch_hour,
            "source": self.source,
            "note": self.note,
        }
        if now is not None:
            data["is_dispatch_day"] = self.is_dispatch_day(now)
            data["next_dispatch_time"] = self.next_dispatch_time(now, include_now=False).isoformat()
        return data


@dataclass
class CargoBatch:
    id: str
    cargo_type: CargoType
    total_weight_ton: float
    origin_station: str
    destination_station: str
    ready_time: datetime
    priority: int = 0
    remaining_ton: Optional[float] = None
    in_transit_ton: float = 0.0
    delivered_ton: float = 0.0

    def __post_init__(self) -> None:
        if self.total_weight_ton < 0:
            raise ValueError("total_weight_ton must be >= 0")
        if self.remaining_ton is None:
            self.remaining_ton = float(self.total_weight_ton)

    @property
    def status(self) -> CargoBatchStatus:
        if self.delivered_ton >= self.total_weight_ton - 1e-9:
            return CargoBatchStatus.DELIVERED
        if self.in_transit_ton > 1e-9:
            return CargoBatchStatus.IN_TRANSIT
        if self.remaining_ton < self.total_weight_ton - 1e-9:
            return CargoBatchStatus.PARTIAL
        return CargoBatchStatus.WAITING

    def is_ready(self, now: datetime, station_id: str) -> bool:
        return self.remaining_ton > 1e-9 and now >= self.ready_time and self.origin_station == station_id

    def allocate(self, requested_ton: float) -> float:
        amount = max(0.0, min(float(requested_ton), self.remaining_ton))
        self.remaining_ton -= amount
        self.in_transit_ton += amount
        return amount

    def deliver(self, amount_ton: float) -> float:
        amount = max(0.0, min(float(amount_ton), self.in_transit_ton))
        self.in_transit_ton -= amount
        self.delivered_ton += amount
        return amount

    def add_ready_amount(self, amount_ton: float) -> None:
        if amount_ton > 0:
            self.total_weight_ton += amount_ton
            self.remaining_ton += amount_ton

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cargo_type"] = self.cargo_type.value
        data["ready_time"] = self.ready_time.isoformat()
        data["status"] = self.status.value
        return data


@dataclass
class CargoWagon:
    id: str
    capacity_ton: float = 20.0
    current_location: Optional[str] = None
    loads_ton: Dict[str, float] = field(default_factory=dict)
    train_id: Optional[str] = None

    def loaded_weight(self) -> float:
        return sum(self.loads_ton.values())

    def remaining_capacity(self) -> float:
        return max(0.0, self.capacity_ton - self.loaded_weight())

    def is_empty(self) -> bool:
        return self.loaded_weight() <= 1e-9

    def load(self, batch: CargoBatch, requested_ton: float) -> float:
        amount = min(self.remaining_capacity(), max(0.0, requested_ton))
        if amount <= 0:
            return 0.0
        allocated = batch.allocate(amount)
        if allocated > 0:
            self.loads_ton[batch.id] = self.loads_ton.get(batch.id, 0.0) + allocated
        return allocated

    def unload_all(self, batches: Dict[str, CargoBatch]) -> float:
        delivered = 0.0
        for batch_id, amount in list(self.loads_ton.items()):
            if batch_id in batches:
                delivered += batches[batch_id].deliver(amount)
        self.loads_ton.clear()
        return delivered

    def move_to(self, location: str, train_id: Optional[str] = None) -> None:
        self.current_location = location
        self.train_id = train_id

    def to_dict(self) -> dict:
        return asdict(self)


class CargoInventoryManager:
    def __init__(self) -> None:
        self._batches: Dict[str, CargoBatch] = {}
        self._wagons: Dict[str, CargoWagon] = {}
        self._supply_schedules: Dict[CargoType, CargoSupplySchedule] = {}

    def register_batch(self, batch: CargoBatch) -> None:
        self._batches[batch.id] = batch

    def register_wagon(self, wagon: CargoWagon) -> None:
        self._wagons[wagon.id] = wagon

    def register_supply_schedule(self, schedule: CargoSupplySchedule) -> None:
        self._supply_schedules[schedule.cargo_type] = schedule

    def get_supply_schedule(self, cargo_type: CargoType) -> Optional[CargoSupplySchedule]:
        return self._supply_schedules.get(cargo_type)

    def all_supply_schedules(self) -> List[CargoSupplySchedule]:
        return list(self._supply_schedules.values())

    def get_batch(self, batch_id: str) -> CargoBatch:
        return self._batches[batch_id]

    def get_wagon(self, wagon_id: str) -> CargoWagon:
        return self._wagons[wagon_id]

    def all_batches(self) -> List[CargoBatch]:
        return list(self._batches.values())

    def all_wagons(self) -> List[CargoWagon]:
        return list(self._wagons.values())

    def wagons_at(self, station_id: str, empty_only: bool = False) -> List[CargoWagon]:
        wagons = [w for w in self._wagons.values() if w.current_location == station_id]
        return [w for w in wagons if w.is_empty()] if empty_only else wagons

    def pending_backlog(
        self,
        station_id: str,
        now: Optional[datetime] = None,
        cargo_type: Optional[CargoType] = None,
        destination_station: Optional[str] = None,
    ) -> List[CargoBatch]:
        now = now or datetime.max
        batches = [
            b for b in self._batches.values()
            if b.origin_station == station_id
            and b.is_ready(now, station_id)
            and (cargo_type is None or b.cargo_type == cargo_type)
            and (destination_station is None or b.destination_station == destination_station)
        ]
        return sorted(batches, key=lambda b: (-b.priority, b.ready_time, b.id))

    def backlog_weight_ton(
        self,
        station_id: str,
        now: Optional[datetime] = None,
        cargo_type: Optional[CargoType] = None,
        destination_station: Optional[str] = None,
    ) -> float:
        return sum(
            b.remaining_ton
            for b in self.pending_backlog(station_id, now, cargo_type, destination_station)
        )

    def current_dispatch_limit_wagons(
        self,
        station_id: str,
        now: datetime,
        cargo_type: CargoType,
        wagon_capacity_ton: float = 20.0,
    ) -> int:
        """현재 준비된 물량과 공급 스케줄을 함께 고려한 이번 cycle의 최대 화차 수."""
        import math

        ready_ton = self.backlog_weight_ton(station_id, now, cargo_type)
        ton_based = int(math.ceil(ready_ton / max(wagon_capacity_ton, 1e-9))) if ready_ton > 0 else 0
        schedule = self.get_supply_schedule(cargo_type)
        if schedule is None:
            return ton_based
        return min(ton_based, schedule.max_wagons_per_dispatch)

    def supply_summary(self, now: datetime, station_id: Optional[str] = None) -> dict:
        result = {}
        for schedule in self.all_supply_schedules():
            payload = schedule.to_dict(now)
            if station_id is not None:
                payload["ready_backlog_ton"] = round(
                    self.backlog_weight_ton(station_id, now, schedule.cargo_type), 3
                )
                payload["current_dispatch_limit_wagons"] = self.current_dispatch_limit_wagons(
                    station_id, now, schedule.cargo_type, schedule.wagon_capacity_ton
                )
            result[schedule.cargo_type.value] = payload
        return result

    def allocate_to_wagons(
        self,
        station_id: str,
        destination_station: str,
        now: datetime,
        cargo_type: CargoType,
        max_wagons: int,
    ) -> tuple[list[str], float]:
        """우선순위가 높은 배치부터 빈 화차에 적재한다. 한 화차에는 한 화종만 적재한다."""
        if max_wagons <= 0:
            return [], 0.0
        batches = self.pending_backlog(station_id, now, cargo_type, destination_station)
        wagons = self.wagons_at(station_id, empty_only=True)[:max_wagons]
        used_ids: list[str] = []
        total_loaded = 0.0
        batch_idx = 0

        for wagon in wagons:
            while batch_idx < len(batches) and wagon.remaining_capacity() > 1e-9:
                batch = batches[batch_idx]
                if batch.remaining_ton <= 1e-9:
                    batch_idx += 1
                    continue
                loaded = wagon.load(batch, wagon.remaining_capacity())
                total_loaded += loaded
                if batch.remaining_ton <= 1e-9:
                    batch_idx += 1
                if loaded <= 1e-9:
                    break
            if not wagon.is_empty():
                used_ids.append(wagon.id)

        return used_ids, total_loaded

    def unload_wagon(self, wagon_id: str) -> float:
        return self.get_wagon(wagon_id).unload_all(self._batches)

    def to_dict(self) -> dict:
        return {
            "batches": {bid: batch.to_dict() for bid, batch in self._batches.items()},
            "wagons": {wid: wagon.to_dict() for wid, wagon in self._wagons.items()},
            "supply_schedules": {
                schedule.cargo_type.value: schedule.to_dict()
                for schedule in self._supply_schedules.values()
            },
        }
