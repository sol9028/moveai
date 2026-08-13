"""대시보드 출력 JSON을 최적화 모델과 독립적으로 점검한다."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import yaml


def verify_result(result: dict, params_path: Path) -> None:
    with params_path.open("r", encoding="utf-8") as f:
        p = yaml.safe_load(f)
    errors: list[str] = []
    days = result["days"]
    parsed = [date.fromisoformat(x) for x in days]
    if len(days) != 7:
        errors.append(f"계획일수 {len(days)} != 7")
    for i in range(1, len(parsed)):
        if parsed[i] != parsed[i - 1] + timedelta(days=1):
            errors.append(f"날짜가 연속되지 않음: {days[i-1]} → {days[i]}")
    cargo_days = [d for d in result["daily"] if d["cargo_day"]]
    if len(cargo_days) > int(p["weekly"]["max_cargo_days"]):
        errors.append("주간 화물일 상한 초과")
    if sum(d["trains"] for d in result["daily"]) > int(p["consist"]["total_trains"]) * \
            int(p["weekly"]["max_operating_days_per_train"]):
        errors.append("주간 편성-일 상한 초과")
    W_ESS = float(p["ess"]["capacity_mwh"]) * float(p["ess"]["ton_per_mwh"])
    S = int(p["consist"]["slots_per_train"])
    train_lim = float(p["weight"]["max_train_payload_ton"])
    cargo_weight = p["weight"]["cargo_car_weight_ton"]
    for trip in result["trips"]:
        capacity = S * trip["trains"]
        c_cars = sum(x["cars"] for x in trip["cargo"])
        if not all(isinstance(x["cars"], int) for x in trip["cargo"]):
            errors.append(f"{trip['date']} 소수 화차 발생")
        if trip["up_ess_containers"] > capacity:
            errors.append(f"{trip['date']} 상행 슬롯 초과")
        if trip["return_ess_containers"] + c_cars > capacity:
            errors.append(f"{trip['date']} 하행 슬롯 초과")
        up_weight = W_ESS * trip["up_ess_containers"]
        down_weight = W_ESS * trip["return_ess_containers"] + \
            sum(float(cargo_weight[x["cargo_type"]]) * x["cars"] for x in trip["cargo"])
        if up_weight > train_lim * trip["trains"] + 1e-5:
            errors.append(f"{trip['date']} 상행 중량 초과")
        if down_weight > train_lim * trip["trains"] + 1e-5:
            errors.append(f"{trip['date']} 하행 중량 초과")
        for item in trip["cargo"]:
            cap = float(p["weight"]["cargo_capacity_ton_per_car"][item["cargo_type"]])
            if item["ton"] > cap * item["cars"] + 1e-5:
                errors.append(f"{trip['date']} {item['cargo_type']} 화차 용량 초과")
    if any(v < -1e-6 for v in result["ending_state"]["cargo_inventory_ton"].values()):
        errors.append("종료 화물재고 음수")
    ess_count = sum(result["ending_state"]["home_containers"].values()) + \
        sum(result["ending_state"]["metro_containers"].values())
    if ess_count != int(p["ess"]["total_containers"]):
        errors.append(f"종료 ESS 보존 위반: {ess_count}")
    if errors:
        raise AssertionError("검증 실패:\n- " + "\n- ".join(errors))

