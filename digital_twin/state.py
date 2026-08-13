from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .cargo import CargoInventoryManager, CargoType
from .ess import ESSFleetManager, ESSWagon
from .rail import RailNetwork
from .station import Region, StationRegistry
from .train import TrainRegistry


@dataclass
class SystemState:
    current_time: datetime
    stations: StationRegistry = field(default_factory=StationRegistry)
    trains: TrainRegistry = field(default_factory=TrainRegistry)
    rail_network: RailNetwork = field(default_factory=RailNetwork)
    ess_fleet: ESSFleetManager = field(default_factory=ESSFleetManager)
    cargo_inventory: CargoInventoryManager = field(default_factory=CargoInventoryManager)
    metadata: Dict = field(default_factory=dict)

    def clone(self) -> "SystemState":
        return deepcopy(self)

    def ess_inventory_by_region(self, region: Region) -> int:
        return sum(self.ess_fleet.inventory_count(st.id) for st in self.stations.by_region(region))

    def ess_average_soc_by_region(self, region: Region) -> float:
        wagons: List[ESSWagon] = []
        for st in self.stations.by_region(region):
            wagons.extend(self.ess_fleet.at_location(st.id))
        return 0.0 if not wagons else sum(w.soc_percent for w in wagons) / len(wagons)

    def cargo_backlog_by_region(
        self,
        region: Region,
        cargo_type: Optional[CargoType] = None,
    ) -> float:
        total = 0.0
        for st in self.stations.by_region(region):
            total += self.cargo_inventory.backlog_weight_ton(st.id, self.current_time, cargo_type)
        return total

    def select_return_ess(self, station_id: str, count: int) -> List[ESSWagon]:
        candidates = self.ess_fleet.wagons_available_for_return(station_id)
        return candidates[: max(0, count)]

    def advance_time(self, minutes: float) -> None:
        from datetime import timedelta
        self.current_time += timedelta(minutes=minutes)

    def validate(self) -> None:
        station_ids = {s.id for s in self.stations.all()}
        train_ids = {t.id for t in self.trains.all()}
        for wagon in self.ess_fleet.all():
            wagon.validate()
            if wagon.train_id and wagon.train_id not in train_ids:
                raise ValueError(f"{wagon.id}: unknown train_id={wagon.train_id}")
            if not wagon.train_id and wagon.current_location and wagon.current_location not in station_ids:
                raise ValueError(f"{wagon.id}: unknown location={wagon.current_location}")
        seen = set()
        for train in self.trains.all():
            if train.total_wagons() > train.max_wagons:
                raise ValueError(f"{train.id}: consist exceeds max_wagons")
            for _, wid in train.wagons:
                if wid in seen:
                    raise ValueError(f"Wagon {wid} attached to multiple trains")
                seen.add(wid)

    def snapshot(self) -> dict:
        station_state = {}
        for station in self.stations.all():
            ess_count = self.ess_fleet.inventory_count(station.id)
            station_state[station.id] = {
                "name": station.name,
                "region": station.region.value,
                "role": station.role.value,
                "ess_count": ess_count,
                "ess_avg_soc": round(self.ess_fleet.average_soc(station.id), 1),
                "ess_capacity": station.ess_dock_capacity,
                "ess_utilization": round(station.ess_utilization(ess_count), 3),
                "cargo_backlog_ton": round(
                    self.cargo_inventory.backlog_weight_ton(station.id, self.current_time), 1
                ),
                "curtailment_available_mwh": round(station.curtailment_available_mwh, 1),
                "demand_mwh": round(station.demand_mwh, 1),
            }
        return {
            "time": self.current_time.isoformat(),
            "stations": station_state,
            "trains": {t.id: t.composition_summary() for t in self.trains.all()},
            "regions": {
                region.value: {
                    "ess_inventory": self.ess_inventory_by_region(region),
                    "ess_avg_soc": round(self.ess_average_soc_by_region(region), 1),
                    "cargo_backlog_ton": round(self.cargo_backlog_by_region(region), 1),
                }
                for region in (Region.HONAM, Region.METRO)
            },
            "cargo_supply": self.cargo_inventory.supply_summary(self.current_time, "METRO_MAIN"),
            "metadata": deepcopy(self.metadata),
        }

    def to_dict(self) -> dict:
        return {
            "current_time": self.current_time.isoformat(),
            "stations": self.stations.to_dict(),
            "trains": self.trains.to_dict(),
            "rail_network": self.rail_network.to_dict(),
            "ess_fleet": self.ess_fleet.to_dict(),
            "cargo_inventory": self.cargo_inventory.to_dict(),
            "metadata": deepcopy(self.metadata),
        }
