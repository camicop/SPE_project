import numpy as np
import subprocess
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import math
from scipy import stats
import seaborn as sns
from matplotlib import cm
import xml.etree.ElementTree as ET


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulation.traffic_flow import TrafficFlow, TrafficFlowManager, VehicleType
from simulation.simulation_utils import run_simulation, write_traffic_light_file, get_inserted_vehicle_count, get_loaded_vehicle_count

# Simulation Constants
HOUR_DURATION = 3600       # duration of one hour in seconds
NUM_HOURS = 3              # number of hours to simulate
WARMUP_TIME = 4000         # warmup time in seconds
NUM_RUNS = 15               # number of runs per setup

NORTH_SOUTH_VEHICLES_PER_MINUTE = 10  # Vehicles per minute (North-South/South-North)
WEST_EAST_VEHICLES_PER_MINUTE = 6     # Vehicles per minute (West-East/East-West)

# Derived Parameters
LAMBDA_RATE_NS = NORTH_SOUTH_VEHICLES_PER_MINUTE / 60              # vehicles per second
LAMBDA_RATE_WE = WEST_EAST_VEHICLES_PER_MINUTE / 60                # vehicles per second
SIMULATION_DURATION = HOUR_DURATION * NUM_HOURS + WARMUP_TIME      # total simulation time in seconds

# Traffic Light Setups - different green durations for North-South
TRAFFIC_LIGHT_SETUPS = [
    {"ns_green": 30, "ew_green": 30, "name": "NS30_EW30"},
    {"ns_green": 40, "ew_green": 30, "name": "NS40_EW30"}, 
    {"ns_green": 50, "ew_green": 30, "name": "NS50_EW30"},
    {"ns_green": 35, "ew_green": 20, "name": "NS35_EW20"},
    {"ns_green": 45, "ew_green": 25, "name": "NS45_EW25"}
]

# Files
route_file = "configs/poisson_routes.rou.xml"
config_file = "configs/poisson_config.sumocfg"
tripinfo_file = "configs/tripinfo.xml"
trafficlight_file = "configs/trafficlight.tll.xml"

# Traffic Flows
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

def generate_routes():
    """Generate route file"""
    with open(route_file, "w") as f:
        trafficManager = TrafficFlowManager(flows)
        f.write(trafficManager.generate_routes_xml())

def jain_index(x):
    """Calculate Jain's Fairness Index"""
    x = np.array(x)
    if np.sum(x) == 0:
        return 1.0  
    return (np.sum(x) ** 2) / (len(x) * np.sum(x ** 2))

def lorenz_curve(data):
    """Calculate Lorenz curve coordinates"""
    sorted_data = np.sort(data)
    cum_data = np.cumsum(sorted_data)
    total = cum_data[-1]
    if total == 0:
        return np.linspace(0, 1, len(data)), np.linspace(0, 1, len(data))
    lorenz = cum_data / total
    x_vals = np.linspace(0, 1, len(data))
    return x_vals, lorenz

def run_setup_simulations(setup, num_runs=NUM_RUNS):
    """Run multiple simulations for a specific traffic light setup"""
    print(f"\n=== Running simulations for setup: {setup['name']} ===")
    print(f"NS Green: {setup['ns_green']}s, EW Green: {setup['ew_green']}s")
    
    setup_durations = []
    setup_time_losses = []
    setup_waiting_times = []
    setup_vehicles_not_inserted = []
    
    # Generate traffic light configuration for this setup
    write_traffic_light_file(
        trafficlight_path=trafficlight_file,
        ns_green_duration=setup['ns_green'],
        ew_green_duration=setup['ew_green'],
        yellow_duration=5
    )
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}", end=" ")
        
        # Generate routes for this run
        generate_routes()
        
        # Run simulation 
        run_simulation(config_file, SIMULATION_DURATION, tripinfo_file=tripinfo_file, gui=False, quiet=True)
        vehicles_not_inserted = get_loaded_vehicle_count(route_file_path=route_file) - get_inserted_vehicle_count(tripinfo_path=tripinfo_file)
        
        if not os.path.exists(tripinfo_file):
            print(f"Error: {tripinfo_file} not found.")
            continue
        
        # Parse results
        df = pd.read_xml(tripinfo_file)
        df_filtered = df[df["arrival"] >= WARMUP_TIME]
        
        setup_durations.extend(df_filtered["duration"].values)
        setup_time_losses.extend(df_filtered["timeLoss"].values)
        setup_waiting_times.extend(df_filtered["waitingTime"].values)
        
        # Store vehicles not inserted for this run
        setup_vehicles_not_inserted.append(vehicles_not_inserted)
    
    return {
        "duration": np.array(setup_durations),
        "timeLoss": np.array(setup_time_losses),
        "waitingTime": np.array(setup_waiting_times),
        "vehicles_not_inserted": np.array(setup_vehicles_not_inserted)
    }

def plot_comparative_lorenz_curves(all_results):
    """Plot Lorenz curves for all setups on the same graphs"""
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    metrics = ["duration", "timeLoss", "waitingTime"]
    colors = cm.tab10(np.linspace(0, 1, len(TRAFFIC_LIGHT_SETUPS)))
    
    for i, metric in enumerate(metrics):
        # Plot perfect equality line
        axs[i].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality', linewidth=2)
        
        # Plot Lorenz curve for each setup
        for j, (setup, results) in enumerate(all_results.items()):
            x_vals, lorenz_vals = lorenz_curve(results[metric])
            axs[i].plot(x_vals, lorenz_vals, 
                       color=colors[j], 
                       label=f'{setup}', 
                       linewidth=2,
                       alpha=0.8)
        
        axs[i].set_title(f"Lorenz Curves - {metric}", fontsize=16, fontweight='bold')
        axs[i].set_xlabel("Fraction of vehicles", fontsize=18)
        axs[i].set_ylabel("Cumulative fraction", fontsize=18)
        axs[i].legend(fontsize=10)
        axs[i].grid(True, alpha=0.3)
        axs[i].tick_params(axis='both', labelsize=15)
    
    plt.tight_layout()
    plt.show()

def plot_radar_chart(all_results):
    """Plot radar chart comparing setups across metrics"""
    # Calculate metrics for each setup
    setup_names = list(all_results.keys())
    
    # Raw metrics
    raw_metrics = {}
    for setup, results in all_results.items():
        raw_metrics[setup] = {
            'Jain Duration': jain_index(results['duration']),
            'Jain TimeLoss': jain_index(results['timeLoss']),  
            'Jain WaitingTime': jain_index(results['waitingTime']),
            'Avg Duration': np.mean(results['duration']),
            'Avg TimeLoss': np.mean(results['timeLoss']),
            'Avg WaitingTime': np.mean(results['waitingTime']),
            'Vehicles Not Inserted': np.mean(results['vehicles_not_inserted'])
        }
    
    # Print metric values
    print("\n" + "="*100)
    print("METRICS before normalization for radar chart")
    print("="*100)
    print(f"{'Setup':<15} {'Jain Dur':<10} {'Jain TL':<10} {'Jain WT':<10} {'Avg Dur':<10} {'Avg TL':<10} {'Avg WT':<10} {'Not Ins':<10}")
    print("-" * 100)
    
    for setup in setup_names:
        metrics = raw_metrics[setup]
        print(f"{setup:<15} {metrics['Jain Duration']:<10.4f} {metrics['Jain TimeLoss']:<10.4f} {metrics['Jain WaitingTime']:<10.4f} "
              f"{metrics['Avg Duration']:<10.1f} {metrics['Avg TimeLoss']:<10.1f} {metrics['Avg WaitingTime']:<10.1f} {metrics['Vehicles Not Inserted']:<10.1f}")
    
    metrics_for_radar = ['Jain Duration', 'Jain TimeLoss', 'Jain WaitingTime', 
                        'Avg Duration', 'Avg TimeLoss', 'Avg WaitingTime', 'Vehicles Not Inserted']
    
    # Normalize metrics for radar chart (0-1 scale, higher is better)
    normalized_data = {}
    for setup in setup_names:
        normalized_data[setup] = []
    
    for metric in metrics_for_radar:
        values = [raw_metrics[setup][metric] for setup in setup_names]
        
        if 'Jain' in metric:
            # Jain index: already 0-1
            normalized_values = values
        else:
            # Time metrics and vehicles not inserted: need to invert
            max_val = max(values)
            min_val = min(values)
            if max_val > min_val:
                normalized_values = [(max_val - v) / (max_val - min_val) for v in values]
            else:
                normalized_values = [1.0] * len(values)
        
        for i, setup in enumerate(setup_names):
            normalized_data[setup].append(normalized_values[i])
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]  
    
    colors = cm.tab10(np.linspace(0, 1, len(setup_names)))
    
    for i, setup in enumerate(setup_names):
        values = normalized_data[setup] + [normalized_data[setup][0]]  
        
        ax.plot(angles, values, 'o-', linewidth=2, label=setup, color=colors[i], alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    # Customized labels
    label_mapping = {
        'Jain Duration': 'Fairness\nDuration',
        'Jain TimeLoss': 'Fairness\nTime Loss', 
        'Jain WaitingTime': 'Fairness\nWaiting Time',
        'Avg Duration': 'Efficiency\nDuration',
        'Avg TimeLoss': 'Efficiency\nTime Loss',
        'Avg WaitingTime': 'Efficiency\nWaiting Time',
        'Vehicles Not Inserted': 'System\nAbility to let all cars enter the system'
    }
    
    labels = [label_mapping[metric] for metric in metrics_for_radar]
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=16)
    ax.set_title("Traffic Light Setup\nMetrics are normalized: a Higher value is Better", 
                 fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    all_results = {}
    
    # Run simulations for each setup
    for setup in TRAFFIC_LIGHT_SETUPS:
        results = run_setup_simulations(setup)
        all_results[setup['name']] = results
    
    # Plot comparative Lorenz curves
    plot_comparative_lorenz_curves(all_results)
    
    # Plot radar chart
    plot_radar_chart(all_results)
