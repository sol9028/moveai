import unittest
from datetime import date, datetime
from pathlib import Path

from agent import ETrainAgent
from forecast import build_forecast_bundle
from forecast.demand import predict_demand_from_csv
from optimization import solve_for_state
from simulation import build_demo_state, simulate
from digital_twin import CargoType, RailNetwork, RailSegment

DEMAND_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "demand_shortage.csv"


class ETrainCoreTest(unittest.TestCase):
    def test_black_mass_weekly_supply_constraint_is_reflected(self):
        state = build_demo_state()
        forecast = build_forecast_bundle(state).cargo
        schedule = state.cargo_inventory.get_supply_schedule(CargoType.BLACK_MASS)

        self.assertIsNotNone(schedule)
        self.assertEqual(schedule.dispatches_per_week, 2)
        self.assertEqual(schedule.max_wagons_per_dispatch, 4)
        self.assertEqual(schedule.max_ton_per_dispatch, 80.0)
        self.assertEqual(schedule.weekly_nominal_capacity_ton, 160.0)
        self.assertEqual(forecast.current_ton["black_mass"], 80.0)
        self.assertEqual(forecast.current_dispatch_limit_wagons["black_mass"], 4)
        self.assertEqual(forecast.weekly_nominal_capacity_ton["black_mass"], 160.0)

    def test_black_mass_never_exceeds_four_wagons_in_current_dispatch(self):
        state = build_demo_state()
        forecasts = build_forecast_bundle(state)
        for policy in ("balanced", "energy_priority", "cargo_priority"):
            _, result = solve_for_state(state, forecasts, "HONAM_MAIN", "METRO_MAIN", policy=policy)
            self.assertTrue(result.success)
            self.assertLessEqual(result.cargo_wagons_by_type["black_mass"], 4)
            self.assertEqual(result.diagnostics["cargo_car_upper_bounds"]["black_mass"], 4)

    def test_policy_mix_changes_with_supply_and_objective(self):
        state = build_demo_state()
        forecasts = build_forecast_bundle(state)
        expected = {
            "balanced": (12, 8, 20),
            "energy_priority": (17, 3, 20),
            # 화물 공급이 총 8량뿐이라 20량을 억지로 채우지 않는다.
            "cargo_priority": (8, 8, 16),
        }
        for policy, expected_mix in expected.items():
            _, result = solve_for_state(state, forecasts, "HONAM_MAIN", "METRO_MAIN", policy=policy)
            self.assertTrue(result.success)
            self.assertEqual(
                (result.return_ess_count, result.total_cargo_wagons, result.total_return_wagons),
                expected_mix,
            )
            self.assertFalse(result.extra_return_train)

    def test_supply_schedule_next_dispatch(self):
        state = build_demo_state(datetime(2026, 8, 13, 9, 0))  # Thursday demo dispatch
        schedule = state.cargo_inventory.get_supply_schedule(CargoType.BLACK_MASS)
        self.assertEqual(schedule.next_dispatch_time(state.current_time), datetime(2026, 8, 17, 9, 0))
        self.assertEqual(schedule.next_dispatch_time(datetime(2026, 8, 17, 9, 1)), datetime(2026, 8, 20, 9, 0))

    def test_simulation_completes_bidirectional_cycle(self):
        state = build_demo_state()
        forecasts = build_forecast_bundle(state)
        _, opt = solve_for_state(state, forecasts, "HONAM_MAIN", "METRO_MAIN", policy="balanced")
        plan = ETrainAgent(scenario_count=1)._to_operation_plan(
            state, opt, "HONAM_MAIN", "METRO_MAIN", 20, opt.diagnostics["required_honam_ess_next"]
        )
        result = simulate(state, plan, horizon_hours=18)
        self.assertTrue(result.metrics.train_completed_cycle)
        self.assertGreater(result.metrics.delivered_energy_mwh, 0)
        self.assertGreater(result.metrics.cargo_delivered_ton, 0)
        self.assertLessEqual(opt.cargo_wagons_by_type["black_mass"], 4)
        self.assertFalse(result.metrics.honam_ess_shortage)

    def test_agent_selects_balanced_under_demo_state(self):
        result = ETrainAgent(scenario_count=3, seed=7).run_once(build_demo_state()).to_dict()
        self.assertEqual(result["selected"]["policy"], "balanced")
        self.assertEqual(result["selected"]["plan"]["return_ess_count"], 12)
        self.assertEqual(result["selected"]["plan"]["cargo_wagons_by_type"]["black_mass"], 4)
        self.assertEqual(result["selected"]["plan"]["total_return_wagons"], 20)
        rejected = {x["policy"] for x in result["briefing"]["rejected_alternatives"]}
        self.assertEqual(rejected, {"energy_priority", "cargo_priority"})

    def test_rail_route_and_hourly_slot(self):
        rail = RailNetwork()
        rail.add_segment(RailSegment("A_B", "A", "B", 100, 100, capacity_per_hour=1))
        rail.add_segment(RailSegment("B_C", "B", "C", 100, 100, capacity_per_hour=1))
        route = rail.find_route("A", "C")
        self.assertEqual(route[1], ["A_B", "B_C"])
        t = datetime(2026, 8, 13, 10, 0)
        rail.reserve_route("A", "C", t)
        self.assertFalse(rail.can_reserve_route("A", "C", t))
        next_t = rail.next_available_departure("A", "C", t)
        self.assertGreaterEqual(next_t.hour, 11)

    def test_extra_train_only_when_inventory_need_exceeds_one_train(self):
        state = build_demo_state()
        forecasts = build_forecast_bundle(state, curtailment_growth_factor=2.8)
        _, result = solve_for_state(state, forecasts, "HONAM_MAIN", "METRO_MAIN", policy="energy_priority")
        self.assertTrue(result.extra_return_train)
        self.assertGreater(result.return_ess_count, 20)
        self.assertLessEqual(result.total_return_wagons, 40)

    def test_demand_forecast_uses_real_csv_when_metadata_path_is_set(self):
        state = build_demo_state()
        state.metadata["demand_csv_path"] = str(DEMAND_CSV_PATH)
        forecasts = build_forecast_bundle(state)
        self.assertEqual(forecasts.demand.source, "CSV_DATA")
        self.assertGreater(forecasts.demand.current_mwh, 0)

    def test_demand_forecast_falls_back_to_simulated_without_csv_path(self):
        state = build_demo_state()  # metadata에 demand_csv_path 없음
        forecasts = build_forecast_bundle(state)
        self.assertEqual(forecasts.demand.source, "SIMULATED_FORECAST")

    def test_predict_demand_from_csv_picks_nearest_next_date(self):
        forecast = predict_demand_from_csv(DEMAND_CSV_PATH, target_date=date(2025, 5, 4))
        self.assertEqual(forecast.source, "CSV_DATA")
        # 2025-05-04 데이터가 있으므로 그대로 '현재'로 쓰이고, 가장 가까운 다른 날짜가 '다음 기간'
        self.assertGreater(forecast.current_mwh, 0)
        self.assertGreater(forecast.next_period_mwh, 0)


if __name__ == "__main__":
    unittest.main()
