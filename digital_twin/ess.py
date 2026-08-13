from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional


class ESSStatus(str, Enum):
    IDLE = "idle"
    CHARGING = "charging"
    DISCHARGING = "discharging"
    IN_TRANSIT = "in_transit"
    FAILED = "failed"


@dataclass
class ESSWagon:
    """개별 ESS 화차의 SOC·충방전·위치·가용 상태를 표현한다."""

    id: str
    capacity_mwh: float = 10.0
    soc_percent: float = 0.0
    current_location: Optional[str] = None
    status: ESSStatus = ESSStatus.IDLE
    max_charge_rate_mw: float = 5.0
    max_discharge_rate_mw: float = 5.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    return_soc_threshold: float = 20.0
    available: bool = True
    train_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()
        self.soc_percent = min(100.0, max(0.0, float(self.soc_percent)))

    @property
    def soc_fraction(self) -> float:
        return self.soc_percent / 100.0

    @property
    def stored_energy_mwh(self) -> float:
        return self.capacity_mwh * self.soc_fraction

    @property
    def remaining_capacity_mwh(self) -> float:
        return max(0.0, self.capacity_mwh - self.stored_energy_mwh)

    @property
    def is_full(self) -> bool:
        return self.soc_percent >= 99.9

    @property
    def is_empty(self) -> bool:
        return self.soc_percent <= 0.1

    def needs_return(self) -> bool:
        return self.available and self.soc_percent <= self.return_soc_threshold

    def charge(self, grid_energy_mwh: float, duration_h: float) -> float:
        """주어진 시간 동안 충전한다. 반환값은 계통에서 실제로 소비한 MWh."""
        if grid_energy_mwh <= 0 or duration_h <= 0 or not self.available:
            return 0.0
        max_grid_by_rate = self.max_charge_rate_mw * duration_h
        max_grid_by_room = self.remaining_capacity_mwh / self.charge_efficiency
        used_grid = max(0.0, min(grid_energy_mwh, max_grid_by_rate, max_grid_by_room))
        stored = used_grid * self.charge_efficiency
        self.soc_percent = min(100.0, self.soc_percent + stored / self.capacity_mwh * 100.0)
        if used_grid > 0:
            self.status = ESSStatus.CHARGING
        return used_grid

    def discharge(self, requested_delivery_mwh: float, duration_h: float) -> float:
        """주어진 시간 동안 방전한다. 반환값은 수요측에 실제 전달한 MWh."""
        if requested_delivery_mwh <= 0 or duration_h <= 0 or not self.available:
            return 0.0
        max_by_rate = self.max_discharge_rate_mw * duration_h
        max_by_energy = self.stored_energy_mwh * self.discharge_efficiency
        delivered = max(0.0, min(requested_delivery_mwh, max_by_rate, max_by_energy))
        removed = delivered / self.discharge_efficiency
        self.soc_percent = max(0.0, self.soc_percent - removed / self.capacity_mwh * 100.0)
        if delivered > 0:
            self.status = ESSStatus.DISCHARGING
        return delivered

    def move_to(self, location: str, train_id: Optional[str] = None) -> None:
        self.current_location = location
        self.train_id = train_id
        self.status = ESSStatus.IN_TRANSIT if train_id else ESSStatus.IDLE

    def fail(self) -> None:
        self.available = False
        self.status = ESSStatus.FAILED

    def recover(self) -> None:
        self.available = True
        self.status = ESSStatus.IDLE

    def validate(self) -> None:
        if self.capacity_mwh <= 0:
            raise ValueError(f"{self.id}: capacity_mwh must be > 0")
        if not 0 <= self.soc_percent <= 100:
            raise ValueError(f"{self.id}: soc_percent must be in [0,100]")
        if self.max_charge_rate_mw < 0 or self.max_discharge_rate_mw < 0:
            raise ValueError(f"{self.id}: charge/discharge rate must be >= 0")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError(f"{self.id}: efficiency must be in (0,1]")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class ESSFleetManager:
    def __init__(self) -> None:
        self._wagons: Dict[str, ESSWagon] = {}

    def register(self, wagon: ESSWagon) -> None:
        self._wagons[wagon.id] = wagon

    def get(self, wagon_id: str) -> ESSWagon:
        return self._wagons[wagon_id]

    def all(self) -> List[ESSWagon]:
        return list(self._wagons.values())

    def at_location(self, location_id: str, available_only: bool = False) -> List[ESSWagon]:
        items = [w for w in self._wagons.values() if w.current_location == location_id]
        return [w for w in items if w.available] if available_only else items

    def inventory_count(self, location_id: str, available_only: bool = False) -> int:
        return len(self.at_location(location_id, available_only))

    def average_soc(self, location_id: str) -> float:
        wagons = self.at_location(location_id)
        return 0.0 if not wagons else sum(w.soc_percent for w in wagons) / len(wagons)

    def wagons_available_for_charge(self, location_id: str) -> List[ESSWagon]:
        return [w for w in self.at_location(location_id, True) if not w.is_full and w.status != ESSStatus.IN_TRANSIT]

    def wagons_available_for_return(self, location_id: str) -> List[ESSWagon]:
        return sorted(
            [w for w in self.at_location(location_id, True) if w.status != ESSStatus.IN_TRANSIT],
            key=lambda w: (not w.needs_return(), w.soc_percent),
        )

    def low_soc_count(self, location_id: str) -> int:
        return sum(1 for w in self.at_location(location_id, True) if w.needs_return())

    def to_dict(self) -> dict:
        return {wid: wagon.to_dict() for wid, wagon in self._wagons.items()}
