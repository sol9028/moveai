from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TrainDirection(str, Enum):
    UPBOUND = "upbound"
    DOWNBOUND = "downbound"
    UNKNOWN = "unknown"


class TrainStatus(str, Enum):
    IDLE = "idle"
    CHARGING = "charging"
    LOADING = "loading"
    READY = "ready"
    EN_ROUTE = "en_route"
    UNLOADING = "unloading"
    MAINTENANCE = "maintenance"


class WagonType(str, Enum):
    ESS = "ess"
    CARGO = "cargo"


@dataclass
class Train:
    id: str
    locomotive_id: str
    current_station: Optional[str]
    max_wagons: int = 20
    direction: TrainDirection = TrainDirection.UNKNOWN
    status: TrainStatus = TrainStatus.IDLE
    destination_station: Optional[str] = None
    scheduled_departure: Optional[datetime] = None
    actual_departure: Optional[datetime] = None
    expected_arrival: Optional[datetime] = None
    remaining_travel_min: float = 0.0
    route_segment_ids: List[str] = field(default_factory=list)
    wagons: List[Tuple[WagonType, str]] = field(default_factory=list)
    applied_route_delay_min: float = 0.0

    def total_wagons(self) -> int:
        return len(self.wagons)

    def has_room(self, n: int = 1) -> bool:
        return self.total_wagons() + n <= self.max_wagons

    def attach_wagon(self, wagon_type: WagonType, wagon_id: str) -> None:
        if any(existing_id == wagon_id for _, existing_id in self.wagons):
            return
        if not self.has_room():
            raise ValueError(f"{self.id}: consist exceeds max_wagons={self.max_wagons}")
        self.wagons.append((wagon_type, wagon_id))

    def detach_wagon(self, wagon_id: str) -> bool:
        for i, (_, wid) in enumerate(self.wagons):
            if wid == wagon_id:
                del self.wagons[i]
                return True
        return False

    def clear_wagons(self) -> None:
        self.wagons.clear()

    def ess_wagon_ids(self) -> List[str]:
        return [wid for wtype, wid in self.wagons if wtype == WagonType.ESS]

    def cargo_wagon_ids(self) -> List[str]:
        return [wid for wtype, wid in self.wagons if wtype == WagonType.CARGO]

    def depart(
        self,
        origin: str,
        destination: str,
        departure_time: datetime,
        travel_time_min: float,
        route_segment_ids: List[str],
        direction: TrainDirection,
    ) -> None:
        if travel_time_min <= 0:
            raise ValueError("travel_time_min must be > 0")
        self.current_station = None
        self.destination_station = destination
        self.actual_departure = departure_time
        from datetime import timedelta
        self.expected_arrival = departure_time + timedelta(minutes=travel_time_min)
        self.remaining_travel_min = float(travel_time_min)
        self.route_segment_ids = list(route_segment_ids)
        self.direction = direction
        self.status = TrainStatus.EN_ROUTE
        self.applied_route_delay_min = 0.0

    def arrive(self, station_id: str) -> None:
        self.current_station = station_id
        self.destination_station = None
        self.remaining_travel_min = 0.0
        self.status = TrainStatus.UNLOADING

    def composition_summary(self) -> dict:
        return {
            "train_id": self.id,
            "direction": self.direction.value,
            "status": self.status.value,
            "total_wagons": self.total_wagons(),
            "ess_wagons": len(self.ess_wagon_ids()),
            "cargo_wagons": len(self.cargo_wagon_ids()),
        }

    def to_dict(self) -> dict:
        data = asdict(self)
        data["direction"] = self.direction.value
        data["status"] = self.status.value
        data["wagons"] = [[wtype.value, wid] for wtype, wid in self.wagons]
        for key in ("scheduled_departure", "actual_departure", "expected_arrival"):
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data


class TrainRegistry:
    def __init__(self) -> None:
        self._trains: Dict[str, Train] = {}

    def register(self, train: Train) -> None:
        self._trains[train.id] = train

    def get(self, train_id: str) -> Train:
        return self._trains[train_id]

    def all(self) -> List[Train]:
        return list(self._trains.values())

    def to_dict(self) -> dict:
        return {tid: train.to_dict() for tid, train in self._trains.items()}
