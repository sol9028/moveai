from __future__ import annotations

from dataclasses import dataclass, asdict

from digital_twin.state import SystemState
from digital_twin.station import Region


@dataclass
class CurtailmentForecast:
    current_mwh: float
    next_period_mwh: float
    confidence: float
    source: str = "SIMULATED_FORECAST"

    def to_dict(self) -> dict:
        return asdict(self)


def predict_curtailment(state: SystemState, growth_factor: float | None = None) -> CurtailmentForecast:
    """MVP용 가상 예측. 실제 상용화 시 발전/기상 예측 모델 출력으로 교체한다."""
    sources = state.stations.by_region(Region.HONAM)
    current = sum(s.curtailment_available_mwh for s in sources)
    factor = growth_factor if growth_factor is not None else float(state.metadata.get("curtailment_growth_factor", 1.68))
    next_period = max(0.0, current * factor)
    return CurtailmentForecast(current_mwh=current, next_period_mwh=next_period, confidence=0.82)
