from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from simulation import (
    ActuatedSignalController,
    ArrivalRateSegment,
    FixedTimeSignalController,
    IntersectionSimulation,
    SignalController,
)


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_TABLE = PROJECT_ROOT / "arrival_parameter_estimates_by_approach_time_period.csv"
RESULTS_DIR = PROJECT_ROOT

APPROACHES = ("E", "W", "N", "S")
DEMAND_SEGMENTS = [
    ("Morning peak (07-09)", 0.0, 3 * 3600.0),
    ("Midday (10-15)", 3 * 3600.0, 9 * 3600.0),
    ("Evening peak (16-18)", 9 * 3600.0, 12 * 3600.0),
]

TOTAL_SIMULATION_SECONDS = 43_200.0
WARM_UP_SECONDS = 1_800.0
DATA_COLLECTION_SECONDS = TOTAL_SIMULATION_SECONDS - WARM_UP_SECONDS
CROSSING_TIME_SECONDS = 1.0
REPLICATIONS = 30
BASE_SEED = 1000


@dataclass
class ScenarioConfig:
    stage: str
    name: str
    signal_control: str
    cycle_time_seconds: float | None = None
    ns_green_ratio: float | None = None
    ew_green_ratio: float | None = None
    min_green_seconds: float | None = None
    max_green_seconds: float | None = None


SCENARIOS = [
    ScenarioConfig("Stage 1", "S1_fixed_90s_50_50", "fixed_time", 90.0, 0.50, 0.50),
    ScenarioConfig("Stage 1", "S2_fixed_90s_60_40_EW", "fixed_time", 90.0, 0.40, 0.60),
    ScenarioConfig("Stage 1", "S3_fixed_90s_70_30_EW", "fixed_time", 90.0, 0.30, 0.70),
    ScenarioConfig("Stage 1", "S4_fixed_90s_80_20_EW", "fixed_time", 90.0, 0.20, 0.80),
    ScenarioConfig("Stage 2", "C1_fixed_60s_70_30_EW", "fixed_time", 60.0, 0.30, 0.70),
    ScenarioConfig("Stage 2", "C2_fixed_90s_70_30_EW", "fixed_time", 90.0, 0.30, 0.70),
    ScenarioConfig("Stage 2", "C3_fixed_120s_70_30_EW", "fixed_time", 120.0, 0.30, 0.70),
    ScenarioConfig("Stage 2", "C4_fixed_60s_80_20_EW", "fixed_time", 60.0, 0.20, 0.80),
    ScenarioConfig("Stage 2", "C5_fixed_90s_80_20_EW", "fixed_time", 90.0, 0.20, 0.80),
    ScenarioConfig("Stage 2", "C6_fixed_120s_80_20_EW", "fixed_time", 120.0, 0.20, 0.80),
    ScenarioConfig("Stage 3", "F_best_fixed_60s_80_20_EW", "fixed_time", 60.0, 0.20, 0.80),
    ScenarioConfig("Stage 3", "A1_actuated_10_45", "actuated_queue", min_green_seconds=10.0, max_green_seconds=45.0),
    ScenarioConfig("Stage 3", "A2_actuated_15_60", "actuated_queue", min_green_seconds=15.0, max_green_seconds=60.0),
    ScenarioConfig("Stage 3", "A3_actuated_10_75", "actuated_queue", min_green_seconds=10.0, max_green_seconds=75.0),
    ScenarioConfig("Stage 3", "A4_actuated_20_90", "actuated_queue", min_green_seconds=20.0, max_green_seconds=90.0),
]


def load_arrival_rate_schedule() -> list[ArrivalRateSegment]:
    parameters = pd.read_csv(INPUT_TABLE)
    schedule = []

    for label, start_time, end_time in DEMAND_SEGMENTS:
        selected = parameters[parameters["time_period"] == label]
        rates = dict(zip(selected["approach"], selected["lambda_vehicles_per_second"], strict=True))
        missing = set(APPROACHES) - set(rates)
        if missing:
            raise ValueError(f"Missing arrival rates for {label}: {sorted(missing)}")

        schedule.append(
            ArrivalRateSegment(
                label=label,
                start_time=start_time,
                end_time=end_time,
                rates_per_second=rates,
            )
        )

    return schedule


def build_signal_controller(config: ScenarioConfig) -> SignalController:
    if config.signal_control == "fixed_time":
        return FixedTimeSignalController(
            cycle_time_seconds=config.cycle_time_seconds,
            ns_green_ratio=config.ns_green_ratio,
            ew_green_ratio=config.ew_green_ratio,
            initial_phase="EW",
        )

    if config.signal_control == "actuated_queue":
        return ActuatedSignalController(
            min_green_seconds=config.min_green_seconds,
            max_green_seconds=config.max_green_seconds,
            queue_threshold=0,
            initial_phase="EW",
        )

    raise ValueError(f"Unknown signal control type: {config.signal_control}")


def run_one_replication(
    config: ScenarioConfig,
    arrival_schedule: list[ArrivalRateSegment],
    replication: int,
) -> tuple[dict[str, float | int | str | None], list[dict[str, float | int | str]]]:
    seed = BASE_SEED + replication
    simulation = IntersectionSimulation(
        signal=build_signal_controller(config),
        simulation_duration_seconds=TOTAL_SIMULATION_SECONDS,
        crossing_time_seconds=CROSSING_TIME_SECONDS,
        arrival_rate_schedule=arrival_schedule,
        random_seed=seed,
        warm_up_seconds=WARM_UP_SECONDS,
        max_events=1_200_000,
    )
    result = simulation.run()

    scenario_row = {
        "stage": config.stage,
        "scenario": config.name,
        "replication": replication,
        "seed": seed,
        **result.scenario,
    }
    approach_rows = [
        {
            "stage": config.stage,
            "scenario": config.name,
            "replication": replication,
            "seed": seed,
            **row,
        }
        for row in result.approach_rows
    ]

    return scenario_row, approach_rows


def aggregate_results(replication_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = replication_results.groupby(["stage", "scenario"], sort=False)

    for (stage, scenario), group in grouped:
        delay = group["average_delay_seconds"]
        ci_half_width = stats.t.ppf(0.975, len(delay) - 1) * delay.std(ddof=1) / np.sqrt(len(delay))
        rows.append(
            {
                "stage": stage,
                "scenario": scenario,
                "mean_delay_s": delay.mean(),
                "ci_low_s": delay.mean() - ci_half_width,
                "ci_high_s": delay.mean() + ci_half_width,
                "time_avg_queue": group["time_weighted_average_total_queue_length"].mean(),
                "max_queue": group["max_total_queue_length"].mean(),
                "throughput": group["total_throughput"].mean(),
                "remaining": group["vehicles_remaining_in_system"].mean(),
            }
        )

    return pd.DataFrame(rows)


def run_final_experiments() -> pd.DataFrame:
    RESULTS_DIR.mkdir(exist_ok=True)
    arrival_schedule = load_arrival_rate_schedule()

    scenario_rows = []
    approach_rows = []
    for config in SCENARIOS:
        for replication in range(1, REPLICATIONS + 1):
            scenario_row, scenario_approach_rows = run_one_replication(
                config=config,
                arrival_schedule=arrival_schedule,
                replication=replication,
            )
            scenario_rows.append(scenario_row)
            approach_rows.extend(scenario_approach_rows)

    replication_results = pd.DataFrame(scenario_rows)
    approach_results = pd.DataFrame(approach_rows)
    final_results = aggregate_results(replication_results)

    replication_results.to_csv(RESULTS_DIR / "replication_results_generated.csv", index=False)
    approach_results.to_csv(RESULTS_DIR / "approach_results_generated.csv", index=False)
    final_results.to_csv(RESULTS_DIR / "final_12h_results_generated.csv", index=False)

    return final_results


if __name__ == "__main__":
    results = run_final_experiments()
    print(results.to_string(index=False))
