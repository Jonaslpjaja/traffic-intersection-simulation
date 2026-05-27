from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
from typing import Deque

import numpy as np


APPROACHES = ("E", "W", "N", "S")
PHASE_APPROACHES = {
    "EW": ("E", "W"),
    "NS": ("N", "S"),
}
TIME_EPSILON = 1e-9


@dataclass
class Vehicle:
    vehicle_id: int
    approach: str
    arrival_time: float
    start_cross_time: float | None = None
    departure_time: float | None = None


@dataclass(order=True)
class Event:
    time: float
    sequence: int
    event_type: str = field(compare=False)
    approach: str | None = field(default=None, compare=False)
    vehicle: Vehicle | None = field(default=None, compare=False)


class SignalController:
    """Base signal controller.

    Green(i) from the event graph is represented by the active signal phase.
    Signal changes are handled as control logic that updates this active phase.
    """

    control_type = "base"

    def __init__(self, initial_phase: str = "EW") -> None:
        if initial_phase not in PHASE_APPROACHES:
            raise ValueError("initial_phase must be either 'EW' or 'NS'.")
        self.initial_phase = initial_phase
        self.current_phase = initial_phase
        self.phase_start_time = 0.0
        self.scheduled_signal_event_time: float | None = None

    def start(self, simulation: "IntersectionSimulation") -> None:
        self.current_phase = self.initial_phase
        self.phase_start_time = 0.0
        self.schedule_next_signal_change(simulation)

    def get_green_approaches(self) -> tuple[str, ...]:
        return PHASE_APPROACHES[self.current_phase]

    def is_green(self, approach: str) -> bool:
        return approach in self.get_green_approaches()

    def schedule_next_signal_change(self, simulation: "IntersectionSimulation") -> None:
        raise NotImplementedError

    def handle_signal_event(self, simulation: "IntersectionSimulation") -> None:
        raise NotImplementedError

    def on_queue_state_changed(
        self, simulation: "IntersectionSimulation", approach: str
    ) -> None:
        return None

    def scenario_fields(self) -> dict[str, float | int | str | None]:
        return {
            "control_type": self.control_type,
            "initial_phase": self.initial_phase,
        }

    def _opposing_phase(self) -> str:
        return "NS" if self.current_phase == "EW" else "EW"

    def _switch_phase(self, simulation: "IntersectionSimulation") -> None:
        self.current_phase = self._opposing_phase()
        self.phase_start_time = simulation.time
        self.scheduled_signal_event_time = None

    def _schedule_signal_event(
        self, simulation: "IntersectionSimulation", event_time: float
    ) -> None:
        event_time = max(event_time, simulation.time)
        if (
            self.scheduled_signal_event_time is not None
            and self.scheduled_signal_event_time <= event_time
        ):
            return
        if event_time <= simulation.simulation_duration_seconds:
            self.scheduled_signal_event_time = event_time
            simulation._schedule_event(event_time, "signal_change")

    def _is_current_signal_event(self, simulation: "IntersectionSimulation") -> bool:
        if self.scheduled_signal_event_time is None:
            return False
        return abs(simulation.time - self.scheduled_signal_event_time) <= TIME_EPSILON


@dataclass(init=False)
class FixedTimeSignalController(SignalController):
    cycle_time_seconds: float
    ns_green_ratio: float
    ew_green_ratio: float
    initial_phase: str

    control_type = "fixed_time"

    def __init__(
        self,
        cycle_time_seconds: float = 90.0,
        ns_green_ratio: float = 0.5,
        ew_green_ratio: float = 0.5,
        initial_phase: str = "EW",
    ) -> None:
        super().__init__(initial_phase=initial_phase)
        self.cycle_time_seconds = cycle_time_seconds
        self.ns_green_ratio = ns_green_ratio
        self.ew_green_ratio = ew_green_ratio

        if self.cycle_time_seconds <= 0:
            raise ValueError("cycle_time_seconds must be positive.")
        if self.ns_green_ratio <= 0 or self.ew_green_ratio <= 0:
            raise ValueError("Green ratios must be positive.")
        if not np.isclose(self.ns_green_ratio + self.ew_green_ratio, 1.0):
            raise ValueError("ns_green_ratio + ew_green_ratio must equal 1.0.")

    @property
    def ew_green_seconds(self) -> float:
        return self.cycle_time_seconds * self.ew_green_ratio

    @property
    def ns_green_seconds(self) -> float:
        return self.cycle_time_seconds * self.ns_green_ratio

    def phase_duration(self, phase: str) -> float:
        if phase == "EW":
            return self.ew_green_seconds
        if phase == "NS":
            return self.ns_green_seconds
        raise ValueError(f"Unknown phase: {phase}")

    def schedule_next_signal_change(self, simulation: "IntersectionSimulation") -> None:
        next_change_time = simulation.time + self.phase_duration(self.current_phase)
        self._schedule_signal_event(simulation, next_change_time)

    def handle_signal_event(self, simulation: "IntersectionSimulation") -> None:
        if not self._is_current_signal_event(simulation):
            return

        self._switch_phase(simulation)
        self.schedule_next_signal_change(simulation)

    def scenario_fields(self) -> dict[str, float | int | str | None]:
        return {
            **super().scenario_fields(),
            "cycle_time_seconds": self.cycle_time_seconds,
            "ns_green_ratio": self.ns_green_ratio,
            "ew_green_ratio": self.ew_green_ratio,
            "ns_green_seconds": self.ns_green_seconds,
            "ew_green_seconds": self.ew_green_seconds,
            "min_green_seconds": None,
            "max_green_seconds": None,
            "queue_threshold": None,
        }


@dataclass(init=False)
class ActuatedSignalController(SignalController):
    min_green_seconds: float
    max_green_seconds: float
    queue_threshold: int
    initial_phase: str

    control_type = "actuated_queue"

    def __init__(
        self,
        min_green_seconds: float = 15.0,
        max_green_seconds: float = 60.0,
        queue_threshold: int = 0,
        initial_phase: str = "EW",
    ) -> None:
        super().__init__(initial_phase=initial_phase)
        self.min_green_seconds = min_green_seconds
        self.max_green_seconds = max_green_seconds
        self.queue_threshold = queue_threshold

        if self.min_green_seconds <= 0:
            raise ValueError("min_green_seconds must be positive.")
        if self.max_green_seconds < self.min_green_seconds:
            raise ValueError("max_green_seconds must be at least min_green_seconds.")
        if self.queue_threshold < 0:
            raise ValueError("queue_threshold cannot be negative.")

    def schedule_next_signal_change(self, simulation: "IntersectionSimulation") -> None:
        min_check_time = self.phase_start_time + self.min_green_seconds
        max_check_time = self.phase_start_time + self.max_green_seconds

        if simulation.time + TIME_EPSILON < min_check_time:
            self._schedule_signal_event(simulation, min_check_time)
        elif simulation.time + TIME_EPSILON < max_check_time:
            self._schedule_signal_event(simulation, max_check_time)

    def handle_signal_event(self, simulation: "IntersectionSimulation") -> None:
        if not self._is_current_signal_event(simulation):
            return

        self.scheduled_signal_event_time = None
        if self._should_switch(simulation):
            self._switch_phase(simulation)
            self.schedule_next_signal_change(simulation)
            return

        elapsed = simulation.time - self.phase_start_time
        if elapsed + TIME_EPSILON < self.max_green_seconds:
            max_check_time = self.phase_start_time + self.max_green_seconds
            self._schedule_signal_event(simulation, max_check_time)

    def on_queue_state_changed(
        self, simulation: "IntersectionSimulation", approach: str
    ) -> None:
        if (
            simulation.time - self.phase_start_time + TIME_EPSILON
            < self.min_green_seconds
        ):
            return
        if self._should_switch(simulation):
            self._schedule_signal_event(simulation, simulation.time)

    def scenario_fields(self) -> dict[str, float | int | str | None]:
        return {
            **super().scenario_fields(),
            "cycle_time_seconds": None,
            "ns_green_ratio": None,
            "ew_green_ratio": None,
            "ns_green_seconds": None,
            "ew_green_seconds": None,
            "min_green_seconds": self.min_green_seconds,
            "max_green_seconds": self.max_green_seconds,
            "queue_threshold": self.queue_threshold,
        }

    def _should_switch(self, simulation: "IntersectionSimulation") -> bool:
        elapsed = simulation.time - self.phase_start_time
        if elapsed + TIME_EPSILON < self.min_green_seconds:
            return False

        current_queue = simulation.phase_queue_length(self.current_phase)
        opposing_queue = simulation.phase_queue_length(self._opposing_phase())

        if elapsed + TIME_EPSILON >= self.max_green_seconds:
            return opposing_queue > 0

        return opposing_queue > current_queue + self.queue_threshold


@dataclass
class SimulationResult:
    scenario: dict[str, float | int | str]
    approach_rows: list[dict[str, float | int | str]]
    time_series_rows: list[dict[str, float | int | str | bool | None]]


@dataclass
class ArrivalRateSegment:
    label: str
    start_time: float
    end_time: float
    rates_per_second: dict[str, float]

    def __post_init__(self) -> None:
        if self.end_time <= self.start_time:
            raise ValueError("Arrival-rate segment end_time must be after start_time.")
        missing = set(APPROACHES) - set(self.rates_per_second)
        if missing:
            raise ValueError(f"Missing arrival rates for approaches: {sorted(missing)}")
        for approach, rate in self.rates_per_second.items():
            if approach not in APPROACHES:
                raise ValueError(f"Unknown approach in arrival-rate segment: {approach}")
            if rate < 0:
                raise ValueError("Arrival rates cannot be negative.")


class IntersectionSimulation:
    """Small event-based model following Arrival -> Queue -> StartCross -> Departure."""

    def __init__(
        self,
        signal: SignalController,
        simulation_duration_seconds: float,
        crossing_time_seconds: float = 1.0,
        arrival_rates_per_second: dict[str, float] | None = None,
        arrival_rate_schedule: list[ArrivalRateSegment] | None = None,
        random_seed: int = 1,
        warm_up_seconds: float = 0.0,
        sample_interval_seconds: float | None = None,
        max_events: int | None = None,
    ) -> None:
        if simulation_duration_seconds <= 0:
            raise ValueError("simulation_duration_seconds must be positive.")
        if crossing_time_seconds <= 0:
            raise ValueError("crossing_time_seconds must be positive.")
        if warm_up_seconds < 0:
            raise ValueError("warm_up_seconds cannot be negative.")
        if warm_up_seconds >= simulation_duration_seconds:
            raise ValueError("warm_up_seconds must be shorter than the simulation duration.")
        if sample_interval_seconds is not None and sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive when provided.")
        if max_events is not None and max_events <= 0:
            raise ValueError("max_events must be positive when provided.")
        if arrival_rates_per_second is None and arrival_rate_schedule is None:
            raise ValueError("Either arrival_rates_per_second or arrival_rate_schedule is required.")
        if arrival_rates_per_second is not None and arrival_rate_schedule is not None:
            raise ValueError("Use either constant arrival rates or an arrival-rate schedule, not both.")

        self.signal = signal
        self.simulation_duration_seconds = simulation_duration_seconds
        self.crossing_time_seconds = crossing_time_seconds
        self.rng = np.random.default_rng(random_seed)
        self.random_seed = random_seed
        self.warm_up_seconds = warm_up_seconds
        self.sample_interval_seconds = sample_interval_seconds
        self.max_events = max_events
        self.events_processed = 0
        self.collection_duration_seconds = simulation_duration_seconds - warm_up_seconds
        self.arrival_rate_schedule = self._build_arrival_rate_schedule(
            arrival_rates_per_second, arrival_rate_schedule
        )

        self.time = 0.0
        self.event_sequence = 0
        self.vehicle_sequence = 0
        self.event_calendar: list[Event] = []

        self.queues: dict[str, Deque[Vehicle]] = {
            approach: deque() for approach in APPROACHES
        }
        self.in_service: dict[str, Vehicle | None] = {
            approach: None for approach in APPROACHES
        }

        self.generated_arrivals_by_approach = dict.fromkeys(APPROACHES, 0)
        self.total_departures_by_approach = dict.fromkeys(APPROACHES, 0)
        self.arrivals_by_approach = dict.fromkeys(APPROACHES, 0)
        self.departures_by_approach = dict.fromkeys(APPROACHES, 0)
        self.delays_by_approach: dict[str, list[float]] = {
            approach: [] for approach in APPROACHES
        }
        self.queue_area_by_approach = dict.fromkeys(APPROACHES, 0.0)
        self.max_queue_by_approach = dict.fromkeys(APPROACHES, 0)
        self.max_total_queue_length = 0
        self.time_series_rows: list[dict[str, float | int | str | bool | None]] = []

    def run(self) -> SimulationResult:
        if self.warm_up_seconds > 0:
            self._schedule_event(self.warm_up_seconds, "warm_up")
        self._schedule_sample_events()
        self._schedule_arrivals()
        self.signal.start(self)

        while self.event_calendar:
            event = heapq.heappop(self.event_calendar)
            if event.time > self.simulation_duration_seconds:
                break

            self.events_processed += 1
            if self.max_events is not None and self.events_processed > self.max_events:
                raise RuntimeError(
                    "Maximum event limit exceeded. "
                    f"events_processed={self.events_processed}, "
                    f"time={event.time}, event_type={event.event_type}"
                )

            self._advance_time(event.time)
            if event.event_type == "arrival":
                self._handle_arrival(event.approach)
            elif event.event_type == "signal_change":
                self._handle_signal_change()
            elif event.event_type == "departure":
                self._handle_departure(event.approach, event.vehicle)
            elif event.event_type == "warm_up":
                self._handle_warm_up()
            elif event.event_type == "sample":
                self._handle_sample()
            else:
                raise ValueError(f"Unknown event type: {event.event_type}")

        self._advance_time(self.simulation_duration_seconds)
        return self._build_result()

    def _schedule_event(
        self,
        event_time: float,
        event_type: str,
        approach: str | None = None,
        vehicle: Vehicle | None = None,
    ) -> None:
        self.event_sequence += 1
        heapq.heappush(
            self.event_calendar,
            Event(
                time=event_time,
                sequence=self.event_sequence,
                event_type=event_type,
                approach=approach,
                vehicle=vehicle,
            ),
        )

    def _schedule_sample_events(self) -> None:
        if self.sample_interval_seconds is None:
            return

        sample_time = 0.0
        while sample_time <= self.simulation_duration_seconds:
            self._schedule_event(sample_time, "sample")
            sample_time += self.sample_interval_seconds

    def _build_arrival_rate_schedule(
        self,
        arrival_rates_per_second: dict[str, float] | None,
        arrival_rate_schedule: list[ArrivalRateSegment] | None,
    ) -> list[ArrivalRateSegment]:
        if arrival_rate_schedule is not None:
            return arrival_rate_schedule

        if arrival_rates_per_second is None:
            raise ValueError("arrival_rates_per_second is required for a constant-rate model.")

        return [
            ArrivalRateSegment(
                label="Constant demand",
                start_time=0.0,
                end_time=self.simulation_duration_seconds,
                rates_per_second=arrival_rates_per_second,
            )
        ]

    def _schedule_arrivals(self) -> None:
        for segment in self.arrival_rate_schedule:
            segment_start = max(0.0, segment.start_time)
            segment_end = min(self.simulation_duration_seconds, segment.end_time)
            if segment_end <= segment_start:
                continue

            for approach in APPROACHES:
                self._schedule_arrivals_for_segment(
                    approach=approach,
                    arrival_rate=segment.rates_per_second[approach],
                    start_time=segment_start,
                    end_time=segment_end,
                )

    def _schedule_arrivals_for_segment(
        self,
        approach: str,
        arrival_rate: float,
        start_time: float,
        end_time: float,
    ) -> None:
        if arrival_rate <= 0:
            return

        next_time = start_time + self._draw_interarrival_time(arrival_rate)
        while next_time <= end_time:
            self._schedule_event(next_time, "arrival", approach=approach)
            next_time += self._draw_interarrival_time(arrival_rate)

    def _draw_interarrival_time(self, arrival_rate: float) -> float:
        return max(float(self.rng.exponential(1 / arrival_rate)), 1e-9)

    def _advance_time(self, new_time: float) -> None:
        elapsed = new_time - self.time
        if elapsed < 0:
            raise ValueError("Event calendar attempted to move backwards in time.")

        collection_elapsed = max(0.0, new_time - max(self.time, self.warm_up_seconds))
        for approach in APPROACHES:
            self.queue_area_by_approach[approach] += (
                len(self.queues[approach]) * collection_elapsed
            )
        self.time = new_time

    def _handle_warm_up(self) -> None:
        # Reset statistical counters while preserving the physical system state.
        self.arrivals_by_approach = dict.fromkeys(APPROACHES, 0)
        self.departures_by_approach = dict.fromkeys(APPROACHES, 0)
        self.delays_by_approach = {approach: [] for approach in APPROACHES}
        self.queue_area_by_approach = dict.fromkeys(APPROACHES, 0.0)
        self.max_queue_by_approach = {
            approach: len(self.queues[approach]) for approach in APPROACHES
        }
        self.max_total_queue_length = sum(len(queue) for queue in self.queues.values())

    def _handle_sample(self) -> None:
        delays = [
            delay
            for approach_delays in self.delays_by_approach.values()
            for delay in approach_delays
        ]
        row = {
            "time_seconds": self.time,
            "after_warm_up": self.time >= self.warm_up_seconds,
            "total_queue_length": sum(len(queue) for queue in self.queues.values()),
            "throughput_so_far": sum(self.departures_by_approach.values()),
            "average_delay_so_far_seconds": float(np.mean(delays)) if delays else None,
        }
        for approach in APPROACHES:
            row[f"queue_{approach}"] = len(self.queues[approach])
        self.time_series_rows.append(row)

    def _handle_arrival(self, approach: str | None) -> None:
        if approach is None:
            raise ValueError("Arrival event requires an approach.")

        self.vehicle_sequence += 1
        vehicle = Vehicle(
            vehicle_id=self.vehicle_sequence,
            approach=approach,
            arrival_time=self.time,
        )
        self.queues[approach].append(vehicle)
        self.generated_arrivals_by_approach[approach] += 1
        if self._is_collecting_statistics():
            self.arrivals_by_approach[approach] += 1
            self.max_queue_by_approach[approach] = max(
                self.max_queue_by_approach[approach], len(self.queues[approach])
            )
            self.max_total_queue_length = max(
                self.max_total_queue_length,
                sum(len(queue) for queue in self.queues.values()),
            )

        self._try_start_crossing(approach)
        self.signal.on_queue_state_changed(self, approach)

    def _handle_signal_change(self) -> None:
        self.signal.handle_signal_event(self)

        for approach in self.signal.get_green_approaches():
            self._try_start_crossing(approach)

    def _handle_departure(
        self, approach: str | None, vehicle: Vehicle | None
    ) -> None:
        if approach is None or vehicle is None:
            raise ValueError("Departure event requires an approach and a vehicle.")

        vehicle.departure_time = self.time
        self.total_departures_by_approach[approach] += 1
        if self._is_collecting_statistics():
            self.departures_by_approach[approach] += 1
        self.in_service[approach] = None

        self.signal.on_queue_state_changed(self, approach)
        if not self._signal_change_scheduled_now():
            self._try_start_crossing(approach)

    def _try_start_crossing(self, approach: str) -> None:
        if not self._is_green(approach):
            return
        if self.in_service[approach] is not None:
            return
        if not self.queues[approach]:
            return

        vehicle = self.queues[approach].popleft()
        vehicle.start_cross_time = self.time
        self.in_service[approach] = vehicle
        if (
            self._is_collecting_statistics()
            and vehicle.arrival_time >= self.warm_up_seconds
        ):
            self.delays_by_approach[approach].append(self.time - vehicle.arrival_time)

        departure_time = self.time + self.crossing_time_seconds
        self._schedule_event(
            departure_time,
            "departure",
            approach=approach,
            vehicle=vehicle,
        )

    def _is_green(self, approach: str) -> bool:
        return self.signal.is_green(approach)

    def _is_collecting_statistics(self) -> bool:
        return self.time >= self.warm_up_seconds

    def _signal_change_scheduled_now(self) -> bool:
        scheduled_time = self.signal.scheduled_signal_event_time
        return (
            scheduled_time is not None
            and abs(self.time - scheduled_time) <= TIME_EPSILON
        )

    def phase_queue_length(self, phase: str) -> int:
        return sum(len(self.queues[approach]) for approach in PHASE_APPROACHES[phase])

    def _build_result(self) -> SimulationResult:
        approach_rows = []
        all_delays = []
        total_queue_area = 0.0
        total_arrivals = 0
        total_departures = 0

        for approach in APPROACHES:
            delays = self.delays_by_approach[approach]
            all_delays.extend(delays)
            queue_area = self.queue_area_by_approach[approach]
            total_queue_area += queue_area
            generated_arrivals = self.generated_arrivals_by_approach[approach]
            arrivals = self.arrivals_by_approach[approach]
            departures = self.departures_by_approach[approach]
            total_departures_physical = self.total_departures_by_approach[approach]
            total_arrivals += arrivals
            total_departures += departures

            approach_rows.append(
                {
                    "approach": approach,
                    "generated_arrivals": generated_arrivals,
                    "arrivals": arrivals,
                    "throughput": departures,
                    "total_departures_physical": total_departures_physical,
                    "vehicles_remaining_in_queue": len(self.queues[approach]),
                    "vehicle_in_crossing_at_end": int(
                        self.in_service[approach] is not None
                    ),
                    "average_delay_seconds": float(np.mean(delays)) if delays else 0.0,
                    "max_delay_seconds": float(np.max(delays)) if delays else 0.0,
                    "time_weighted_average_queue_length": (
                        queue_area / self.collection_duration_seconds
                    ),
                    "max_queue_length": self.max_queue_by_approach[approach],
                }
            )

        scenario = {
            **self.signal.scenario_fields(),
            "simulation_duration_seconds": self.simulation_duration_seconds,
            "warm_up_seconds": self.warm_up_seconds,
            "data_collection_seconds": self.collection_duration_seconds,
            "crossing_time_seconds": self.crossing_time_seconds,
            "events_processed": self.events_processed,
            "total_generated_arrivals": sum(self.generated_arrivals_by_approach.values()),
            "total_arrivals": total_arrivals,
            "total_throughput": total_departures,
            "total_departures_physical": sum(self.total_departures_by_approach.values()),
            "vehicles_remaining_in_system": sum(len(q) for q in self.queues.values())
            + sum(vehicle is not None for vehicle in self.in_service.values()),
            "average_delay_seconds": float(np.mean(all_delays)) if all_delays else 0.0,
            "max_delay_seconds": float(np.max(all_delays)) if all_delays else 0.0,
            "time_weighted_average_total_queue_length": (
                total_queue_area / self.collection_duration_seconds
            ),
            "max_total_queue_length": self.max_total_queue_length,
        }

        return SimulationResult(
            scenario=scenario,
            approach_rows=approach_rows,
            time_series_rows=self.time_series_rows,
        )
