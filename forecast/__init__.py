from dataclasses import dataclass

from .curtailment import CurtailmentForecast, predict_curtailment
from .demand import DemandForecast, predict_demand
from .cargo import CargoForecast, predict_cargo


@dataclass
class ForecastBundle:
    curtailment: CurtailmentForecast
    demand: DemandForecast
    cargo: CargoForecast

    def to_dict(self) -> dict:
        return {
            "curtailment": self.curtailment.to_dict(),
            "demand": self.demand.to_dict(),
            "cargo": self.cargo.to_dict(),
        }


def build_forecast_bundle(state, **overrides) -> ForecastBundle:
    return ForecastBundle(
        curtailment=predict_curtailment(state, overrides.get("curtailment_growth_factor")),
        demand=predict_demand(state, overrides.get("demand_growth_factor")),
        cargo=predict_cargo(state, overrides.get("cargo_growth_factors")),
    )


__all__ = [
    "CurtailmentForecast", "DemandForecast", "CargoForecast", "ForecastBundle",
    "predict_curtailment", "predict_demand", "predict_cargo", "build_forecast_bundle",
]
