# Traffic Intersection Simulation

This repository contains the main Python simulation code and final result files for a discrete-event simulation of a simplified four-way signalized traffic intersection.

## Model Scope

The model follows the event-graph logic used in the report:

```text
Arrival(i) -> StartCross(i) -> Departure(i)
```

where `i` is one of the four approaches `N`, `S`, `E`, and `W`.

The FIFO stop-line queue is represented by the state variable `Q_i`. At `Arrival(i)`, the queue length is increased. At `StartCross(i)`, one vehicle is removed from the queue and starts crossing if:

```text
Q_i > 0 AND Green(i) = 1 AND M_i = 1
```

The simulation includes one FIFO queue per approach, two signal phases (`EW` and `NS`), fixed-time signal control, queue-based actuated signal control, warm-up removal, replications, common random numbers, and the performance measures used in the report.

The model does not include pedestrians, cyclists, turning lanes, lane changing, yellow/all-red phases, or detailed lane occupancy.

## Files

| File | Purpose |
|---|---|
| `simulation.py` | Core discrete-event simulation model |
| `run_experiments.py` | Final 12-hour experiment setup and scenario runner |
| `arrival_parameter_estimates_by_approach_time_period.csv` | Derived arrival-rate input used by the simulation |
| `final_12h_results_combined.csv` | Main aggregate result table used in the report |
| `best_control_approach_results.csv` | Approach-level results for the fixed-time versus actuated comparison |
| `requirements.txt` | Python package requirements |

## Running

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the final experiment script:

```bash
python3 run_experiments.py
```

The submitted results were generated using:

- original demand input
- `t_s = 1.0` second effective approach-level discharge headway
- 12-hour simulation horizon
- 1800-second warm-up
- 30 replications
- common random numbers across scenarios

## Main Result

The best overall configuration among the tested alternatives was the queue-based actuated controller with:

```text
min green = 10 seconds
max green = 75 seconds
```

This setting achieved the lowest mean delay and lowest time-average queue length in the final comparison.
