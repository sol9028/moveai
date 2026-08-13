from __future__ import annotations

from dataclasses import dataclass, asdict

from digital_twin.state import SystemState
from digital_twin.station import Region


@dataclass
class DemandForecast:
    current_mwh: float
    next_period_mwh: float
    confidence: float
    source: str = "SIMULATED_FORECAST"

    def to_dict(self) -> dict:
        return asdict(self)


def predict_demand(state: SystemState, growth_factor: float | None = None) -> DemandForecast:
    sinks = state.stations.by_region(Region.METRO)
    current = sum(s.demand_mwh for s in sinks)
    factor = growth_factor if growth_factor is not None else float(state.metadata.get("demand_growth_factor", 1.08))
    return DemandForecast(
        current_mwh=current,
        next_period_mwh=max(0.0, current * factor),
        confidence=0.86,
    )
