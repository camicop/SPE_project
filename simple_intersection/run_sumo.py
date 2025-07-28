import numpy as np
import subprocess
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import math
from scipy import stats


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from traffic_flow import TrafficFlow, TrafficFlowManager, VehicleType
from simulation_utils import run_simulation, analyze_tripinfo, plot_vehicle_counts_over_time, estimate_warmup_time, plot_multiple_time_series, run_adaptive_simulation

# Simulation Constants
HOUR_DURATION = 3600       # duration of one hour in seconds
NUM_HOURS = 3              # number of hours to simulate

NORTH_SOUTH_VEHICLES_PER_MINUTE = 10  # Vehicles per minute (North-South/South-North)
WEST_EAST_VEHICLES_PER_MINUTE = 6     # Vehicles per minute (West-East/East-West)

# Derived Parameters
LAMBDA_RATE_NS = NORTH_SOUTH_VEHICLES_PER_MINUTE / 60  # vehicles per second
LAMBDA_RATE_WE = WEST_EAST_VEHICLES_PER_MINUTE / 60    # vehicles per second
SIMULATION_DURATION = HOUR_DURATION * NUM_HOURS        # total simulation time in seconds

# Files
route_file = "poisson_routes.rou.xml"
config_file = "poisson_config.sumocfg"
tripinfo_file = "tripinfo.xml"

# Vertical Flows
north_south_flow = TrafficFlow(
    start_time=0,
    end_time=SIMULATION_DURATION,
    vehicles_per_minute=NORTH_SOUTH_VEHICLES_PER_MINUTE,
    route_id="north2south",
    edges="north2center center2south",
    vehicle_type=VehicleType(id="north2south_car", max_speed_kmh=50)
)
south_north_flow = TrafficFlow(
    start_time=0,
    end_time=SIMULATION_DURATION,
    vehicles_per_minute=NORTH_SOUTH_VEHICLES_PER_MINUTE,
    route_id="south2north",
    edges="south2center center2north",
    vehicle_type=VehicleType(id="south2north_car", max_speed_kmh=50)
)
# Horizontal Flows
west_east_flow = TrafficFlow(
    start_time=0,
    end_time=SIMULATION_DURATION,
    vehicles_per_minute=WEST_EAST_VEHICLES_PER_MINUTE,
    route_id="west2east",
    edges="west2center center2east",
    vehicle_type=VehicleType(id="west2east_car", max_speed_kmh=30)
)
east_west_flow = TrafficFlow(
    start_time=0,
    end_time=SIMULATION_DURATION,
    vehicles_per_minute=WEST_EAST_VEHICLES_PER_MINUTE,
    route_id="east2west",
    edges="east2center center2west",
    vehicle_type=VehicleType(id="east2west_car", max_speed_kmh=30)
)

# Route generation
def generate_routes(seed=None):
    if seed is not None:
        np.random.seed(seed)
        print(f"Using random seed: {seed}")

    with open(route_file, "w") as f:
        trafficManager = TrafficFlowManager([north_south_flow, south_north_flow, west_east_flow, east_west_flow])
        f.write(trafficManager.generate_routes_xml())

    print(f"Generated vehicles in {route_file}")



def mean_confidence_interval(data, confidence=0.95):

    a = np.array(data)
    n = len(a)
    mean = np.mean(a)
    sem = stats.sem(a)  # standard error of the mean
    margin = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - margin, mean + margin

if __name__ == "__main__":
    SIMULATION_DURATION = 10800
    N_WARMUP_RUNS = 3
    N_FINAL_RUNS = 5

    strategies = {
        "fixed": {
            "config": "poisson_config.sumocfg",
            "runner": run_simulation,
            "tripinfo": "tripinfo.xml"
        },
        "adaptive": {
            "config": "poisson_config_adaptive.sumocfg",
            "runner": run_adaptive_simulation,
            "tripinfo": "tripinfo_adaptive.xml"
        }
    }

    print("\n=== MANUAL SIMULATION WITH GUI (fixed) ===")
    generate_routes(seed=999)
    run_simulation(strategies["fixed"]["config"], SIMULATION_DURATION, gui=True)

    strategy_warmups = {}

    # Warm-up estimation loop
    for strategy, params in strategies.items():
        print(f"\n--- WARM-UP ESTIMATION: {strategy.upper()} ---")
        warmup_times = []
        all_series = []

        for i in range(N_WARMUP_RUNS):
            generate_routes(seed=1000 + i)
            params["runner"](params["config"], SIMULATION_DURATION, tripinfo_file=params["tripinfo"])

            df = pd.read_xml(params["tripinfo"])
            warmup = estimate_warmup_time(params["tripinfo"], SIMULATION_DURATION)
            warmup_times.append(warmup)

            series = [0] * (SIMULATION_DURATION + 1)
            for _, row in df.iterrows():
                depart = int(row["depart"])
                arrival = int(row["arrival"])
                for t in range(depart, min(arrival + 1, SIMULATION_DURATION + 1)):
                    series[t] += 1
            all_series.append(series)

        avg_warmup = int(sum(warmup_times) / len(warmup_times))
        strategy_warmups[strategy] = avg_warmup

        print(f"Average warm-up time for {strategy.upper()}: {avg_warmup} s")
        plot_multiple_time_series(all_series, f"Warm-up {strategy.upper()}", "Number of Vehicles")

    print("\n=== STRATEGY COMPARISON ===")
    strategy_results = {}

    # Final runs collecting statistics with mean, variance and confidence intervals
    for strategy, params in strategies.items():
        print(f"\nSimulations for: {strategy.upper()}")
        warmup = strategy_warmups[strategy]
        waitings, durations, losses, vehicles_counts = [], [], [], []

        for i in range(N_FINAL_RUNS):
            generate_routes(seed=2000 + i)
            params["runner"](params["config"], SIMULATION_DURATION, tripinfo_file=params["tripinfo"])

            df = pd.read_xml(params["tripinfo"])
            df = df[df["depart"] >= warmup]

            waitings.append(df["waitingTime"].mean())
            durations.append(df["duration"].mean())
            losses.append(df["timeLoss"].mean())
            vehicles_counts.append(len(df))

        # Calculate mean, confidence intervals, and variance
        w_mean, w_ci_low, w_ci_high = mean_confidence_interval(waitings)
        d_mean, d_ci_low, d_ci_high = mean_confidence_interval(durations)
        l_mean, l_ci_low, l_ci_high = mean_confidence_interval(losses)

        w_var = np.var(waitings, ddof=1)
        d_var = np.var(durations, ddof=1)
        l_var = np.var(losses, ddof=1)

        strategy_results[strategy] = {
            "vehicles": sum(vehicles_counts) / N_FINAL_RUNS,
            "waiting": w_mean,
            "waiting_ci": (w_ci_low, w_ci_high),
            "waiting_var": w_var,
            "duration": d_mean,
            "duration_ci": (d_ci_low, d_ci_high),
            "duration_var": d_var,
            "loss": l_mean,
            "loss_ci": (l_ci_low, l_ci_high),
            "loss_var": l_var,
            "warmup": warmup
        }

    # Print results with variance and confidence intervals
    print("\n" + "=" * 90)
    print(f"{'Strategy':<10} {'Warm-up(s)':<10} {'Vehicles':<10} {'Waiting(s) [CI] (Var)':<25} {'Duration(s) [CI] (Var)':<25} {'Loss(s) [CI] (Var)':<25}")
    print("-" * 90)

    for name, r in strategy_results.items():
        print(f"{name.upper():<10} {r['warmup']:<10} {r['vehicles']:<10.0f} "
              f"{r['waiting']:.2f} [{r['waiting_ci'][0]:.2f}, {r['waiting_ci'][1]:.2f}] ({r['waiting_var']:.2f}) "
              f"{r['duration']:.2f} [{r['duration_ci'][0]:.2f}, {r['duration_ci'][1]:.2f}] ({r['duration_var']:.2f}) "
              f"{r['loss']:.2f} [{r['loss_ci'][0]:.2f}, {r['loss_ci'][1]:.2f}] ({r['loss_var']:.2f})")

    # Print improvements between adaptive and fixed strategies
    if "fixed" in strategy_results and "adaptive" in strategy_results:
        f = strategy_results["fixed"]
        a = strategy_results["adaptive"]

        print(f"\n{'IMPROVEMENTS ADAPTIVE vs FIXED (CRN):':<50}")
        print("-" * 50)
        print(f"Waiting time reduction:   {(f['waiting'] - a['waiting']) / f['waiting'] * 100:+.1f}%")
        print(f"Duration reduction:       {(f['duration'] - a['duration']) / f['duration'] * 100:+.1f}%")
        print(f"Time loss reduction:      {(f['loss'] - a['loss']) / f['loss'] * 100:+.1f}%")

    labels = list(strategy_results.keys())
    waiting_means = [strategy_results[s]["waiting"] for s in labels]
    duration_means = [strategy_results[s]["duration"] for s in labels]
    loss_means = [strategy_results[s]["loss"] for s in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, waiting_means, width, label='Waiting (s)')
    rects2 = ax.bar(x, duration_means, width, label='Duration (s)')
    rects3 = ax.bar(x + width, loss_means, width, label='Loss (s)')

    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Performance Metrics by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in labels])
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
