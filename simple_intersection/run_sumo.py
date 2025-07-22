import numpy as np
import subprocess
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from traffic_flow import TrafficFlow, TrafficFlowManager, VehicleType
from simulation_utils import run_simulation, analyze_tripinfo, plot_vehicle_counts_over_time

# Simulation Constants 
HOUR_DURATION = 3600       # duration of an hour in seconds
NUM_HOURS = 3              # number of hours of simulation

NORTH_SOUTH_VEHICLES_PER_MINUTE = 10  # NORTH-SOUTH/SOUTH-NORTH vehicles per minute
WEST_EAST_VEHICLES_PER_MINUTE = 6    # WEST-EAST/EAST-WEST vehicles per minute

# Derived Parameters 
LAMBDA_RATE_NS = NORTH_SOUTH_VEHICLES_PER_MINUTE / 60       # vehicles per second, lamba value for poisson random variable
LAMBDA_RATE_WE = WEST_EAST_VEHICLES_PER_MINUTE / 60         # vehicles per second, lamba value for poisson random variable
SIMULATION_DURATION = HOUR_DURATION * NUM_HOURS             # total simulation duration

# Files 
route_file = "poisson_routes.rou.xml"
config_file = "poisson_config.sumocfg"
tripinfo_file = "tripinfo.xml"

# VERTICAL FLOWS
north_south_flow = TrafficFlow(
    start_time=0,
    end_time=SIMULATION_DURATION, 
    vehicles_per_minute=NORTH_SOUTH_VEHICLES_PER_MINUTE, 
    route_id="north2south", 
    edges="north2center center2south",
    vehicle_type=VehicleType(id="north2south_car", max_speed_kmh=50))
south_north_flow = TrafficFlow(
    start_time=0, 
    end_time=SIMULATION_DURATION, 
    vehicles_per_minute=NORTH_SOUTH_VEHICLES_PER_MINUTE, 
    route_id="south2north", 
    edges="south2center center2north",
    vehicle_type=VehicleType(id="south2north_car", max_speed_kmh=50))
# HORIZONTAL FLOWS
west_east_flow = TrafficFlow(
    start_time=0, 
    end_time=SIMULATION_DURATION, 
    vehicles_per_minute=WEST_EAST_VEHICLES_PER_MINUTE, 
    route_id="west2east", 
    edges="west2center center2east",
    vehicle_type=VehicleType(id="west2east_car", max_speed_kmh=30))
east_west_flow = TrafficFlow(
    start_time=0, 
    end_time=SIMULATION_DURATION, 
    vehicles_per_minute=WEST_EAST_VEHICLES_PER_MINUTE, 
    route_id="east2west", 
    edges="east2center center2west",
    vehicle_type=VehicleType(id="east2west_car", max_speed_kmh=30))

# Car flows generation
def generate_routes():
    with open(route_file, "w") as f:
        trafficManager = TrafficFlowManager([north_south_flow, south_north_flow, west_east_flow, east_west_flow])
        f.write(trafficManager.generate_routes_xml())

    print(f"Generated vehicles in {route_file}")

if __name__ == "__main__":
    gui = False
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ["gui", "--gui", "-g"]:
            gui = True

    generate_routes()
    run_simulation(config_file, SIMULATION_DURATION, gui=gui)
    analyze_tripinfo(tripinfo_file)
    plot_vehicle_counts_over_time(tripinfo_file, SIMULATION_DURATION)
