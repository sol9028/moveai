from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List


class Region(str, Enum):
    HONAM = "honam"
    METRO = "metro"
    HUB = "hub"


class StationRole(str, Enum):
    SOURCE = "source"
    SINK = "sink"
    HUB = "hub"


@dataclass
class Station:
    id: str
    name: str
    region: Region
    role: StationRole
    charging_capacity_mw: float = 0.0
    discharging_capacity_mw: float = 0.0
    platform_capacity: int = 1
    ess_dock_capacity: int = 40
    cargo_dock_capacity: int = 40
    curtailment_available_mwh: float = 0.0
    demand_mwh: float = 0.0
    charging_available: bool = True
    discharging_available: bool = True
    processing_available: bool = True

    def validate(self) -> None:
        if self.charging_capacity_mw < 0 or self.discharging_capacity_mw < 0:
            raise ValueError(f"{self.id}: charging/discharging capacity must be >= 0")
        if self.platform_capacity < 0 or self.ess_dock_capacity < 0 or self.cargo_dock_capacity < 0:
            raise ValueError(f"{self.id}: station capacities must be >= 0")

    def ess_utilization(self, count: int) -> float:
        return 1.0 if self.ess_dock_capacity <= 0 else count / self.ess_dock_capacity

    def is_ess_congested(self, count: int, threshold: float = 0.85) -> bool:
        return self.ess_utilization(count) >= threshold

    def available_charging_slots(self, per_wagon_rate_mw: float) -> int:
        if not self.charging_available or per_wagon_rate_mw <= 0:
            return 0
        return int(self.charging_capacity_mw // per_wagon_rate_mw)

    def available_discharging_slots(self, per_wagon_rate_mw: float) -> int:
        if not self.discharging_available or per_wagon_rate_mw <= 0:
            return 0
        return int(self.discharging_capacity_mw // per_wagon_rate_mw)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["region"] = self.region.value
        data["role"] = self.role.value
        return data


class StationRegistry:
    def __init__(self) -> None:
        self._stations: Dict[str, Station] = {}

    def register(self, station: Station) -> None:
        station.validate()
        self._stations[station.id] = station

    def get(self, station_id: str) -> Station:
        return self._stations[station_id]

    def all(self) -> List[Station]:
        return list(self._stations.values())

    def by_region(self, region: Region) -> List[Station]:
        return [s for s in self._stations.values() if s.region == region]

    def to_dict(self) -> dict:
        return {sid: station.to_dict() for sid, station in self._stations.items()}
