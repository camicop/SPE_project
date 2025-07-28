import numpy as np
import subprocess
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from traffic_flow import TrafficFlow, TrafficFlowManager, VehicleType
from simulation_utils import (
    run_simulation, plot_multiple_time_series
)

# Simulation Constants
HOUR_DURATION = 3600  # seconds
NUM_HOURS = 3         # number of hours of simulation

NORTH_SOUTH_VEHICLES_PER_MINUTE = 10
WEST_EAST_VEHICLES_PER_MINUTE = 6
SIMULATION_DURATION = HOUR_DURATION * NUM_HOURS

# Files
route_file = "poisson_routes.rou.xml"
config_file = "poisson_config.sumocfg"
tripinfo_file = "tripinfo.xml"

# Define flows
north_south_flow = TrafficFlow(
    start_time=0, end_time=SIMULATION_DURATION,
    vehicles_per_minute=NORTH_SOUTH_VEHICLES_PER_MINUTE,
    route_id="north2south", edges="north2center center2south",
    vehicle_type=VehicleType(id="north2south_car", max_speed_kmh=50)
)
south_north_flow = TrafficFlow(
    start_time=0, end_time=SIMULATION_DURATION,
    vehicles_per_minute=NORTH_SOUTH_VEHICLES_PER_MINUTE,
    route_id="south2north", edges="south2center center2north",
    vehicle_type=VehicleType(id="south2north_car", max_speed_kmh=50)
)
west_east_flow = TrafficFlow(
    start_time=0, end_time=SIMULATION_DURATION,
    vehicles_per_minute=WEST_EAST_VEHICLES_PER_MINUTE,
    route_id="west2east", edges="west2center center2east",
    vehicle_type=VehicleType(id="west2east_car", max_speed_kmh=30)
)
east_west_flow = TrafficFlow(
    start_time=0, end_time=SIMULATION_DURATION,
    vehicles_per_minute=WEST_EAST_VEHICLES_PER_MINUTE,
    route_id="east2west", edges="east2center center2west",
    vehicle_type=VehicleType(id="east2west_car", max_speed_kmh=30)
)

flows = [north_south_flow, south_north_flow, west_east_flow, east_west_flow]

def number_of_cars_warmup(num_runs):
    all_series = []

    # Run simulations and collect time series
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        generate_routes()
        run_simulation(config_file, SIMULATION_DURATION, gui=False)
        df = pd.read_xml(tripinfo_file)
        series = np.zeros(SIMULATION_DURATION+1)
        for _, row in df.iterrows():
            d, a = int(row['depart']), int(row['arrival'])
            series[d:a+1] += 1
        all_series.append(series)

    # Compute average series
    avg_series = np.mean(all_series, axis=0)

    # Plot all runs + average
    combined = all_series + [avg_series]
    plot_multiple_time_series(
        combined, 
        title="Number of vehicles in the system over time - All Runs", 
        y_label="Number of vehicles", 
        show_legend=False, 
        apply_smoothing=False  
        ) 

    # Plot just the average of all runs
    plot_multiple_time_series(
        [avg_series], 
        title="Average number of vehicles in system over time", 
        y_label="Average vehicle count", 
        show_legend=False, 
        apply_smoothing=False  
        )

    # Plot cumulative average of the mean series
    cum_avg = np.cumsum(avg_series) / np.arange(1, len(avg_series)+1)
    plot_multiple_time_series(
        [cum_avg],
        title="Cumulative average of vehicle counts over time",
        y_label="Cumulative average vehicles",
        show_legend=False, 
        apply_smoothing=False  
        )


def travel_duration_warmup(num_runs):
    def compute_instantaneous_travel_time_series_fast(arrivals, durations, sim_duration):
        arrivals = np.array(arrivals)
        durations = np.array(durations)
        departures = arrivals + durations
        
        avg_series = np.full(sim_duration + 1, np.nan)
        
        for t in range(sim_duration + 1):
            # Vectorized check: vehicles present at time t
            in_system = (arrivals <= t) & (t < departures)
            
            if np.any(in_system):
                # Average duration of vehicles currently in system
                avg_series[t] = np.mean(durations[in_system])
        
        return avg_series

    # Rest of the function remains the same
    all_duration_series = []
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        generate_routes()
        run_simulation(config_file, SIMULATION_DURATION, gui=False)
        
        df = pd.read_xml(tripinfo_file)
        arrivals = df['arrival'].astype(float).values
        durations = df['duration'].astype(float).values
        
        series = compute_instantaneous_travel_time_series_fast(arrivals, durations, SIMULATION_DURATION)
        all_duration_series.append(series)

    # Average across all runs
    avg_series = np.nanmean(all_duration_series, axis=0)

    # Plot all runs + average
    combined = all_duration_series + [avg_series]
    labels = [f"Run {i+1}" for i in range(num_runs)] + ["Average"]
    plot_multiple_time_series(
        combined,
        title="Instantaneous Average Travel Time - All Runs",
        y_label="Average travel time (s)",
        show_legend=False,
        apply_smoothing=False
    )

    # Plot just the average of all runs
    plot_multiple_time_series(
        [avg_series],
        title="Average Travel Time Across All Runs",
        y_label="Average travel time (s)",
        show_legend=False,
        apply_smoothing=False
    )

    # Plot cumulative average of the mean series
    valid_mask = ~np.isnan(avg_series)
    if np.any(valid_mask):
        # Calculate cumulative average only on valid values
        valid_values = avg_series[valid_mask]
        valid_indices = np.where(valid_mask)[0]
       
        cum_avg = np.cumsum(valid_values) / np.arange(1, len(valid_values) + 1)
       
        # Recreate full array with NaN where necessary
        full_cum_avg = np.full_like(avg_series, np.nan)
        full_cum_avg[valid_indices] = cum_avg
       
        plot_multiple_time_series(
            [full_cum_avg],
            title="Cumulative Average Travel Time",
            y_label="Cumulative average travel time (s)",
            show_legend=False,
            apply_smoothing=False
        )



# Route generation
def generate_routes():
    with open(route_file, "w") as f:
        manager = TrafficFlowManager(flows)
        f.write(manager.generate_routes_xml())
    print(f"Generated routes in {route_file}")

if __name__ == "__main__":
    #number_of_cars_warmup(50)
    travel_duration_warmup(50)

