"""연속 7일 양방향 E-Train 운행·ESS·순환자원 조합 MILP."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pulp
import yaml

from forecast_contract import ForecastBundle, load_forecasts


def load_params(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _solver(cfg: dict) -> tuple[str, pulp.LpSolver]:
    limit = int(cfg.get("time_limit_seconds", 120))
    gap = float(cfg.get("relative_gap", 0.001))
    msg = bool(cfg.get("msg", False))
    candidates = [
        ("HiGHS", lambda: pulp.HiGHS(msg=msg, timeLimit=limit, gapRel=gap)),
        ("CBC", lambda: pulp.PULP_CBC_CMD(msg=msg, timeLimit=limit, gapRel=gap)),
    ]
    for name, factory in candidates:
        try:
            solver = factory()
            if solver.available():
                return name, solver
        except Exception:  # pragma: no cover - 환경별 솔버 탐색
            pass
    raise RuntimeError("사용 가능한 MILP 솔버가 없습니다. requirements.txt를 설치하세요.")


def _series(df: pd.DataFrame, keys: list[str], value: str) -> dict:
    return df.set_index(keys)[value].to_dict()


def _initial_state(p: dict, override: dict | None, CH: list[str], DI: list[str]) -> dict:
    ec = p["ess"]
    cap = float(ec["capacity_mwh"])
    low = (1 - float(ec["depth_of_discharge"])) * cap
    home = dict(ec["initial_home_containers"])
    metro = dict(ec["initial_metro_containers"])
    state = {
        "home_containers": home,
        "metro_containers": metro,
        "home_energy_mwh": {c: home[c] * float(ec["initial_soc_ratio"]) * cap for c in CH},
        "metro_energy_mwh": {m: metro[m] * low for m in DI},
        "cargo_inventory_ton": {},
    }
    if override:
        for key in state:
            if key in override:
                state[key] = override[key]
    n = sum(float(state["home_containers"].get(c, 0)) for c in CH) + \
        sum(float(state["metro_containers"].get(m, 0)) for m in DI)
    if abs(n - float(ec["total_containers"])) > 1e-6:
        raise ValueError(f"초기 ESS 컨테이너 합계 {n} != 보유량 {ec['total_containers']}")
    return state


def solve(p: dict, forecasts: ForecastBundle, initial_state: dict | None = None) -> dict[str, Any]:
    days = forecasts.days
    D = list(range(len(days)))
    hours = list(range(24))
    hh = list(range(25))
    CH = [s["id"] for s in p["stations"] if s["type"] == "charge"]
    DI = [s["id"] for s in p["stations"] if s["type"] == "discharge"]
    SS = {s["id"]: s for s in p["stations"]}
    TYPES = list(p["data"]["cargo_types"])
    arc_cfg = {(a["charge"], a["discharge"]): a for a in p["transport_arcs"]}
    ARCS = list(arc_cfg)
    T = {a: int(arc_cfg[a]["travel_h"]) for a in ARCS}
    cargo_nodes = sorted(forecasts.cargo["node"].unique().tolist())
    allowed_nodes = {v for a in ARCS for v in arc_cfg[a].get("cargo_load_nodes", [])}
    unknown_nodes = sorted(set(cargo_nodes) - allowed_nodes)
    if unknown_nodes:
        raise ValueError(f"운행경로와 연결되지 않은 화물 노드: {unknown_nodes}")
    unknown_types = sorted(set(forecasts.cargo["cargo_type"].unique()) - set(TYPES))
    if unknown_types:
        raise ValueError(f"params.yaml에 없는 화물 품목: {unknown_types}")

    cur = _series(forecasts.curtailment, ["date", "node", "hour"], "chargeable_mwh")
    dem = _series(forecasts.demand, ["date", "node", "hour"], "predicted_deficit_mwh")
    smp = _series(forecasts.smp, ["date", "hour"], "predicted_smp_krw_per_kwh")
    arrivals = _series(forecasts.cargo, ["date", "node", "cargo_type"], "predicted_cargo_ton")
    get_cur = lambda d, c, h: float(cur.get((days[d], c, h), 0.0))
    get_dem = lambda d, m, h: float(dem.get((days[d], m, h), 0.0))
    get_smp = lambda d, h: float(smp.get((days[d], h), 0.0))
    get_arr = lambda d, v, k: float(arrivals.get((days[d], v, k), 0.0))

    ec, cons, weight = p["ess"], p["consist"], p["weight"]
    econ, weekly, sched = p["economics"], p["weekly"], p["schedule"]
    NTR = int(cons["total_trains"])
    SLOTS = int(cons["slots_per_train"])
    NCAR = int(ec["total_containers"])
    CAP = float(ec["capacity_mwh"])
    RATE = float(ec["rate_mw"])
    SOC_LO = (1 - float(ec["depth_of_discharge"])) * CAP
    ETA_C = float(ec["charge_efficiency"])
    ETA_D = float(ec["discharge_efficiency"])
    W_ESS = CAP * float(ec["ton_per_mwh"])
    W_CARGO = weight["cargo_car_weight_ton"]
    Q_CARGO = weight["cargo_capacity_ton_per_car"]
    if W_ESS > float(weight["max_car_payload_ton"]) + 1e-9:
        raise ValueError("ESS 컨테이너 중량이 화차 적재한도를 초과합니다.")

    ow = p["operating_window"]
    CHG_H = list(range(int(ow["charge_start_hour"]), int(ow["charge_end_hour"])))
    RH = int(ow["discharge_end_hour"])
    DIS_END = RH - int(ow["handling_hours_before_return"])
    DIS_H = list(range(int(ow["discharge_start_hour"]), DIS_END))
    UPH = [(c, m, h) for c, m in ARCS for h in hours
           if h >= CHG_H[0] and h + T[(c, m)] < DIS_END and RH + T[(c, m)] <= 24]
    P = [(d, c, m, h) for d in D for c, m, h in UPH]
    CPK = [(d, c, m, h, v, k) for d, c, m, h in P
           for v in arc_cfg[(c, m)].get("cargo_load_nodes", []) for k in TYPES]

    model = pulp.LpProblem("etrain_consecutive_7day_v8", pulp.LpMaximize)
    iv = lambda name, keys, ub=None: pulp.LpVariable.dicts(name, keys, 0, ub, cat="Integer")
    cv = lambda name, keys: pulp.LpVariable.dicts(name, keys, 0)
    bv = lambda name, keys: pulp.LpVariable.dicts(name, keys, cat="Binary")

    n = iv("trains", P, NTR)
    up = iv("up_ess", P, SLOTS * NTR)
    ret = iv("return_ess", P, SLOTS * NTR)
    e_up = cv("up_energy", P)
    cargo_cars = iv("cargo_cars", CPK, SLOTS * NTR)
    cargo_ton = cv("cargo_ton", CPK)
    z = bv("cargo_day", D)
    chg = cv("charge", (D, CH, hours))
    dis = cv("discharge", (D, DI, hours))
    home_c = iv("home_containers", (D, CH, hh), NCAR)
    metro_c = iv("metro_containers", (D, DI, hh), NCAR)
    home_e = cv("home_energy", (D, CH, hh))
    metro_e = cv("metro_energy", (D, DI, hh))
    cargo_inv = cv("cargo_inventory", (list(range(8)), cargo_nodes, TYPES))

    state = _initial_state(p, initial_state, CH, DI)
    for c in CH:
        model += home_c[0][c][0] == float(state["home_containers"].get(c, 0))
        model += home_e[0][c][0] == float(state["home_energy_mwh"].get(c, 0))
    for m in DI:
        model += metro_c[0][m][0] == float(state["metro_containers"].get(m, 0))
        model += metro_e[0][m][0] == float(state["metro_energy_mwh"].get(m, 0))
    for v in cargo_nodes:
        for k in TYPES:
            key = f"{v}|{k}"
            model += cargo_inv[0][v][k] == float(state["cargo_inventory_ton"].get(key, 0))

    # 주간 운행·화물일 제약
    model += pulp.lpSum(z[d] for d in D) <= int(weekly["max_cargo_days"])
    model += pulp.lpSum(n[x] for x in P) <= NTR * int(weekly["max_operating_days_per_train"])
    big_cars = NTR * SLOTS
    for d in D:
        day_p = [(d, c, m, h) for c, m, h in UPH]
        day_cpk = [x for x in CPK if x[0] == d]
        model += pulp.lpSum(n[x] for x in day_p) <= NTR
        model += pulp.lpSum(n[x] for x in day_p) <= int(sched["max_round_trips_per_day"])
        model += pulp.lpSum(cargo_cars[x] for x in day_cpk) <= big_cars * z[d]
        model += pulp.lpSum(cargo_cars[x] for x in day_cpk) >= z[d]
        if bool(sched.get("same_day_return", True)):
            model += pulp.lpSum(ret[x] for x in day_p) == pulp.lpSum(up[x] for x in day_p)

    # 편성·화물 조합
    train_lim = float(weight["max_train_payload_ton"])
    for x in P:
        d, c, m, h = x
        ckeys = [q for q in CPK if q[:4] == x]
        model += up[x] <= SLOTS * n[x]
        model += ret[x] + pulp.lpSum(cargo_cars[q] for q in ckeys) <= SLOTS * n[x]
        model += W_ESS * up[x] <= train_lim * n[x]
        model += W_ESS * ret[x] + pulp.lpSum(float(W_CARGO[q[5]]) * cargo_cars[q] for q in ckeys) \
                 <= train_lim * n[x]
        model += e_up[x] >= SOC_LO * up[x]
        model += e_up[x] <= CAP * up[x]
        available_charge_hours = sum(1 for th in CHG_H if th < h)
        model += e_up[x] <= (SOC_LO + RATE * available_charge_hours) * up[x]
    for q in CPK:
        model += cargo_ton[q] <= float(Q_CARGO[q[5]]) * cargo_cars[q]

    # 날짜 간 화물 누적재고 및 처리거점 용량
    for d in D:
        for v in cargo_nodes:
            for k in TYPES:
                shipped = pulp.lpSum(cargo_ton[q] for q in CPK if q[0] == d and q[4] == v and q[5] == k)
                model += cargo_inv[d + 1][v][k] == cargo_inv[d][v][k] + get_arr(d, v, k) - shipped
        for c in CH:
            received = pulp.lpSum(cargo_ton[q] for q in CPK if q[0] == d and q[1] == c)
            model += received <= float(SS[c]["cargo_intake_tpd"])

    # 연속 7일 ESS 컨테이너·SOC 흐름
    metro_cap = int(float(sched["metro_stock_cap_ratio"]) * NCAR)
    for d in D:
        if d > 0:
            for c in CH:
                model += home_c[d][c][0] == home_c[d - 1][c][24]
                model += home_e[d][c][0] == home_e[d - 1][c][24]
            for m in DI:
                model += metro_c[d][m][0] == metro_c[d - 1][m][24]
                model += metro_e[d][m][0] == metro_e[d - 1][m][24]
        arr_home = {(c, t): [] for c in CH for t in hh}
        arr_metro = {(m, t): [] for m in DI for t in hh}
        for c, m, h in UPH:
            arr_metro[(m, h + T[(c, m)])].append((d, c, m, h))
            arr_home[(c, RH + T[(c, m)])].append((d, c, m, h))
        for h in hours:
            for c in CH:
                outgoing = pulp.lpSum(up[(d, c, m, h)] for m in DI if (d, c, m, h) in up)
                incoming_keys = arr_home[(c, h + 1)]
                incoming = pulp.lpSum(ret[x] for x in incoming_keys)
                model += home_c[d][c][h + 1] == home_c[d][c][h] - outgoing + incoming
                model += home_e[d][c][h + 1] == home_e[d][c][h] + ETA_C * chg[d][c][h] \
                         - pulp.lpSum(e_up[(d, c, m, h)] for m in DI if (d, c, m, h) in e_up) \
                         + SOC_LO * incoming
            for m in DI:
                arriving_keys = arr_metro[(m, h + 1)]
                arriving_c = pulp.lpSum(up[x] for x in arriving_keys)
                arriving_e = pulp.lpSum(e_up[x] for x in arriving_keys)
                departing = pulp.lpSum(ret[(d, c, m, hp)] for c, mm, hp in UPH if mm == m) if h == RH else 0
                model += metro_c[d][m][h + 1] == metro_c[d][m][h] + arriving_c - departing
                model += metro_e[d][m][h + 1] == metro_e[d][m][h] + arriving_e \
                         - dis[d][m][h] / ETA_D - (SOC_LO * departing if h == RH else 0)
        for h in hh:
            for c in CH:
                model += home_e[d][c][h] >= SOC_LO * home_c[d][c][h]
                model += home_e[d][c][h] <= CAP * home_c[d][c][h]
            for m in DI:
                model += metro_e[d][m][h] >= SOC_LO * metro_c[d][m][h]
                model += metro_e[d][m][h] <= CAP * metro_c[d][m][h]
            model += pulp.lpSum(metro_c[d][m][h] for m in DI) <= metro_cap
        for h in hours:
            for c in CH:
                if h in CHG_H:
                    model += chg[d][c][h] <= RATE * home_c[d][c][h]
                    model += chg[d][c][h] <= min(get_cur(d, c, h), float(SS[c]["substation_capacity_mw"]))
                else:
                    model += chg[d][c][h] == 0
            for m in DI:
                if h in DIS_H:
                    model += dis[d][m][h] <= RATE * metro_c[d][m][h]
                    model += dis[d][m][h] <= min(get_dem(d, m, h), float(SS[m]["grid_injection_capacity_mw"]))
                else:
                    model += dis[d][m][h] == 0

    # 목적함수와 비용 세부항목
    charge_total = pulp.lpSum(chg[d][c][h] for d in D for c in CH for h in hours)
    discharge_total = pulp.lpSum(dis[d][m][h] for d in D for m in DI for h in hours)
    sell_revenue = pulp.lpSum(dis[d][m][h] * get_smp(d, h) * 1000 for d in D for m in DI for h in hours)
    cargo_revenue = pulp.lpSum(cargo_ton[q] * float(econ["cargo_revenue_krw_per_ton"][q[5]]) for q in CPK)
    buy_cost = charge_total * float(econ["charge_price_krw_per_kwh"]) * 1000
    degradation_cost = charge_total * float(econ["battery_degradation_krw_per_mwh"])
    operation_cost = (charge_total + discharge_total) * float(econ["charge_discharge_cost_krw_per_mwh"])
    train_cost = pulp.lpSum(n[x] for x in P) * 2 * float(econ["train_cost_krw_per_one_way_trip"])
    haul_hours = pulp.lpSum((up[x] + ret[x]) * T[(x[1], x[2])] for x in P) + \
        pulp.lpSum(cargo_cars[q] * T[(q[1], q[2])] for q in CPK)
    haul_cost = haul_hours * float(econ["car_haul_cost_krw_per_car_hour"])
    handling_cost = pulp.lpSum(cargo_cars[q] for q in CPK) * float(econ["cargo_handling_krw_per_car"])
    holding_cost = pulp.lpSum(cargo_inv[d + 1][v][k] for d in D for v in cargo_nodes for k in TYPES) \
        * float(econ["cargo_holding_krw_per_ton_day"])
    terminal_penalty = pulp.lpSum(cargo_inv[7][v][k] for v in cargo_nodes for k in TYPES) \
        * float(econ["terminal_cargo_penalty_krw_per_ton"])
    total_cost = buy_cost + degradation_cost + operation_cost + train_cost + haul_cost + handling_cost \
        + holding_cost + terminal_penalty
    profit = sell_revenue + cargo_revenue - total_cost
    model += profit

    solver_name, solver = _solver(p["solver"])
    started = time.time()
    model.solve(solver)
    elapsed = time.time() - started
    status = pulp.LpStatus[model.status]
    solver_status = status
    mip_gap = None
    objective_bound = None
    if solver_name == "HiGHS" and getattr(model, "solverModel", None) is not None:
        try:
            info = model.solverModel.getInfo()
            solver_status = model.solverModel.modelStatusToString(model.solverModel.getModelStatus())
            mip_gap = float(info.mip_gap)
            objective_bound = float(info.mip_dual_bound)
            if "Time limit" in solver_status and status == "Optimal":
                status = "Feasible"
        except Exception:  # pragma: no cover - highspy 버전별 API 차이
            pass
    if status not in {"Optimal", "Feasible"}:
        raise RuntimeError(f"최적화 실패: {status}")

    V = lambda x: float(pulp.value(x) or 0.0)
    N = lambda x: int(round(V(x)))
    trips = []
    for x in P:
        if N(n[x]) == 0:
            continue
        d, c, m, h = x
        cargo_detail = []
        for q in [q for q in CPK if q[:4] == x and N(cargo_cars[q]) > 0]:
            cargo_detail.append({"load_node": q[4], "cargo_type": q[5],
                                 "cars": N(cargo_cars[q]), "ton": round(V(cargo_ton[q]), 3)})
        trips.append({
            "date": days[d], "from": c, "to": m, "depart_hour": h,
            "arrive_hour": h + T[(c, m)], "return_depart_hour": RH,
            "return_arrive_hour": RH + T[(c, m)], "trains": N(n[x]),
            "up_ess_containers": N(up[x]), "return_ess_containers": N(ret[x]),
            "up_energy_mwh": round(V(e_up[x]), 3), "cargo": cargo_detail,
        })

    daily = []
    hourly = []
    for d in D:
        dt = [t for t in trips if t["date"] == days[d]]
        daily.append({
            "date": days[d], "cargo_day": bool(N(z[d])),
            "trains": sum(t["trains"] for t in dt),
            "up_ess_containers": sum(t["up_ess_containers"] for t in dt),
            "cargo_cars": sum(sum(q["cars"] for q in t["cargo"]) for t in dt),
            "cargo_ton": round(sum(sum(q["ton"] for q in t["cargo"]) for t in dt), 3),
            "charged_mwh": round(sum(V(chg[d][c][h]) for c in CH for h in hours), 3),
            "discharged_mwh": round(sum(V(dis[d][m][h]) for m in DI for h in hours), 3),
            "ending_cargo_inventory_ton": round(sum(V(cargo_inv[d + 1][v][k]) for v in cargo_nodes for k in TYPES), 3),
        })
        for h in hours:
            hourly.append({
                "date": days[d], "hour": h, "smp_krw_per_kwh": get_smp(d, h),
                "chargeable_mwh": round(sum(get_cur(d, c, h) for c in CH), 3),
                "charged_mwh": round(sum(V(chg[d][c][h]) for c in CH), 3),
                "deficit_mwh": round(sum(get_dem(d, m, h) for m in DI), 3),
                "discharged_mwh": round(sum(V(dis[d][m][h]) for m in DI), 3),
            })

    costs = {
        "energy_purchase": V(buy_cost), "battery_degradation": V(degradation_cost),
        "charge_discharge_operation": V(operation_cost), "train_operation": V(train_cost),
        "car_haul": V(haul_cost), "cargo_handling": V(handling_cost),
        "cargo_holding": V(holding_cost), "terminal_cargo_penalty": V(terminal_penalty),
    }
    result = {
        "schema_version": "1.0", "model_version": p["meta"]["version"],
        "generated_at_epoch": int(time.time()), "status": status, "solver": solver_name,
        "solver_status": solver_status, "mip_gap": mip_gap, "objective_bound": objective_bound,
        "solve_seconds": round(elapsed, 3), "planning_start_date": days[0],
        "planning_end_date": days[-1], "days": days,
        "model_size": {"variables": len(model.variables()), "constraints": len(model.constraints)},
        "summary": {
            "round_trips": sum(t["trains"] for t in trips),
            "cargo_days": [x["date"] for x in daily if x["cargo_day"]],
            "charged_mwh": round(V(charge_total), 3), "discharged_mwh": round(V(discharge_total), 3),
            "cargo_cars": sum(sum(q["cars"] for q in t["cargo"]) for t in trips),
            "cargo_ton": round(sum(sum(q["ton"] for q in t["cargo"]) for t in trips), 3),
            "sell_revenue_krw": round(V(sell_revenue)), "cargo_revenue_krw": round(V(cargo_revenue)),
            "total_cost_krw": round(V(total_cost)), "net_profit_krw": round(V(profit)),
            "net_profit_per_day_krw": round(V(profit) / 7),
            "ending_cargo_inventory_ton": round(sum(V(cargo_inv[7][v][k]) for v in cargo_nodes for k in TYPES), 3),
        },
        "cost_breakdown_krw": {k: round(v) for k, v in costs.items()},
        "daily": daily, "trips": trips, "hourly": hourly,
        "ending_state": {
            "home_containers": {c: N(home_c[6][c][24]) for c in CH},
            "home_energy_mwh": {c: round(V(home_e[6][c][24]), 3) for c in CH},
            "metro_containers": {m: N(metro_c[6][m][24]) for m in DI},
            "metro_energy_mwh": {m: round(V(metro_e[6][m][24]), 3) for m in DI},
            "cargo_inventory_ton": {f"{v}|{k}": round(V(cargo_inv[7][v][k]), 3)
                                    for v in cargo_nodes for k in TYPES},
        },
    }
    return result


def run(params_path: str | Path, start_date: str, output_path: str | Path,
        initial_state_path: str | Path | None = None) -> dict:
    params_path = Path(params_path).resolve()
    base_dir = params_path.parent
    p = load_params(params_path)
    forecasts = load_forecasts(p, start_date, base_dir)
    state = None
    if initial_state_path:
        with Path(initial_state_path).open("r", encoding="utf-8") as f:
            state = json.load(f)
    result = solve(p, forecasts, state)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
