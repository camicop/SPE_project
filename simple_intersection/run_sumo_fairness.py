import numpy as np
import subprocess
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import math
from scipy import stats
import seaborn as sns


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from traffic_flow import TrafficFlow, TrafficFlowManager, VehicleType
from simulation_utils import run_simulation, analyze_tripinfo, estimate_warmup_time, plot_multiple_time_series, run_adaptive_simulation

# Simulation Constants
HOUR_DURATION = 3600       # duration of one hour in seconds
NUM_HOURS = 3              # number of hours to simulate

NORTH_SOUTH_VEHICLES_PER_MINUTE = 10  # Vehicles per minute (North-South/South-North)
WEST_EAST_VEHICLES_PER_MINUTE = 6     # Vehicles per minute (West-East/East-West)

# Derived Parameters
LAMBDA_RATE_NS = NORTH_SOUTH_VEHICLES_PER_MINUTE / 60  # vehicles per second
LAMBDA_RATE_WE = WEST_EAST_VEHICLES_PER_MINUTE / 60    # vehicles per second
SIMULATION_DURATION = HOUR_DURATION * NUM_HOURS        # total simulation time in seconds
WARMUP_TIME = 4000                                     # warmup time in seconds
NUM_RUNS = 5                                           # number of runs

# Metrics
all_durations = []
all_time_losses = []
all_waiting_times = []

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

flows = [north_south_flow, south_north_flow, west_east_flow, east_west_flow]

# Route generation
def generate_routes():
    with open(route_file, "w") as f:
        trafficManager = TrafficFlowManager(flows)
        f.write(trafficManager.generate_routes_xml())

# Jain 
def jain_index(x):
    x = np.array(x)
    if np.sum(x) == 0:
        return 1.0  
    return (np.sum(x) ** 2) / (len(x) * np.sum(x ** 2))

# Lorenz
def lorenz_curve(data):
    sorted_data = np.sort(data)
    cum_data = np.cumsum(sorted_data)
    total = cum_data[-1]
    if total == 0:
        return np.linspace(0, 1, len(data)), np.linspace(0, 1, len(data))
    lorenz = cum_data / total
    x_vals = np.linspace(0, 1, len(data))
    return x_vals, lorenz



def run_multiple_simulations_and_analyze(num_runs=NUM_RUNS):
    for i in range(num_runs):
        print(f"\n--- Simulation Run {i+1}/{num_runs} ---")
        generate_routes()
        run_simulation(config_file, SIMULATION_DURATION, gui=False)

        if not os.path.exists(tripinfo_file):
            print(f"Error: {tripinfo_file} not found.")
            continue

        df = pd.read_xml(tripinfo_file)

        df_filtered = df[df["arrival"] >= WARMUP_TIME]

        all_durations.extend(df_filtered["duration"].values)
        all_time_losses.extend(df_filtered["timeLoss"].values)
        all_waiting_times.extend(df_filtered["waitingTime"].values)

    # Analysis
    metrics = {
        "duration": np.array(all_durations),
        "timeLoss": np.array(all_time_losses),
        "waitingTime": np.array(all_waiting_times)
    }

    print(f"\n--- Jain's Fairness Index ---")
    for key, values in metrics.items():
        fairness = jain_index(values)
        print(f"{key:<12}: {fairness:.4f}")

    # Lorenz Curves
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, (key, values) in enumerate(metrics.items()):
        x_vals, lorenz_vals = lorenz_curve(values)
        axs[i].plot(x_vals, lorenz_vals, label=f'Lorenz - {key}')
        axs[i].plot([0, 1], [0, 1], 'k--', label='Perfect Equality')
        axs[i].set_title(f"Lorenz Curve - {key}")
        axs[i].set_xlabel("Fraction of vehicles")
        axs[i].set_ylabel("Cumulative fraction")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

    # Distributions
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, (key, values) in enumerate(metrics.items()):
        sns.histplot(values, bins=30, kde=True, ax=axs[i])
        axs[i].set_title(f"Distribution - {key}")
        axs[i].set_xlabel("Seconds")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_multiple_simulations_and_analyze()