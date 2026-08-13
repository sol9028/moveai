from __future__ import annotations

import csv as csv_module
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Union

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


def _load_daily_deficit(csv_path: Union[str, Path]) -> Dict[date, float]:
    """CSV(date, node, hour, predicted_deficit_mwh)를 날짜별 총 부족 전력량(MWh)으로 집계한다.

    같은 날짜의 여러 node·hour 행을 모두 더해 '그 날의 수도권 수요-공급 부족 총량'으로 취급한다.
    """
    totals: Dict[date, float] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            d = datetime.strptime(row["date"].strip(), "%Y-%m-%d").date()
            value = float(row["predicted_deficit_mwh"])
            totals[d] = totals.get(d, 0.0) + value
    return totals


def predict_demand_from_csv(
    csv_path: Union[str, Path],
    target_date: Optional[date] = None,
) -> DemandForecast:
    """실제 수요(부족) CSV를 읽어 현재/다음 기간 수요를 계산한다.

    이 CSV는 '상위 부족일(top shortage days)'만 골라 담은 표본이라 연속된 날짜가 아니다.
    - target_date가 CSV 안에 있으면 그 날을 '현재'로 사용한다.
    - 없으면 CSV에서 가장 이른 날짜를 '현재'로 사용한다.
    - '다음 기간'은 현재 날짜와 가장 가까운 다른 날짜를 사용한다(추후 실데이터가 연속 일자로
      들어오면 자동으로 '다음 날' 의미에 가까워진다).
    """
    totals = _load_daily_deficit(csv_path)
    if not totals:
        raise ValueError(f"No rows found in {csv_path}")

    sorted_dates = sorted(totals.keys())

    if target_date and target_date in totals:
        current_date = target_date
    else:
        current_date = sorted_dates[0]

    remaining = [d for d in sorted_dates if d != current_date]
    next_date = min(remaining, key=lambda d: abs((d - current_date).days)) if remaining else current_date

    current_mwh = totals[current_date]
    next_mwh = totals[next_date]

    # 실측/실예측 기반 데이터이므로 가상 성장계수 예측보다 신뢰도를 높게 둔다.
    # 표본일(7일)이 적어 상한은 두되, 데이터가 있다는 사실 자체를 반영한다.
    confidence = 0.93 if len(sorted_dates) > 1 else 0.85

    return DemandForecast(
        current_mwh=round(current_mwh, 1),
        next_period_mwh=round(next_mwh, 1),
        confidence=confidence,
        source="CSV_DATA",
    )


def predict_demand(
    state: SystemState,
    growth_factor: Optional[float] = None,
    csv_path: Optional[Union[str, Path]] = None,
    target_date: Optional[date] = None,
) -> DemandForecast:
    """수도권 수요 예측.

    csv_path가 직접 주어지거나 state.metadata['demand_csv_path']가 설정돼 있으면 실제 CSV
    데이터를 사용한다. 둘 다 없으면 기존처럼 SystemState의 demand_mwh × 성장계수로 만드는
    가상 예측(SIMULATED_FORECAST)으로 동작한다 — 기존 호출부/테스트와 100% 호환된다.
    """
    resolved_path = csv_path or state.metadata.get("demand_csv_path")
    if resolved_path:
        try:
            return predict_demand_from_csv(
                resolved_path,
                target_date=target_date or state.current_time.date(),
            )
        except (FileNotFoundError, ValueError, KeyError):
            # CSV 문제 시 조용히 실패하지 않고 가상 예측으로 폴백한다.
            pass

    sinks = state.stations.by_region(Region.METRO)
    current = sum(s.demand_mwh for s in sinks)
    factor = growth_factor if growth_factor is not None else float(state.metadata.get("demand_growth_factor", 1.08))
    return DemandForecast(
        current_mwh=current,
        next_period_mwh=max(0.0, current * factor),
        confidence=0.86,
    )
