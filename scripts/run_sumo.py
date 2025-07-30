import numpy as np
import subprocess
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulation.traffic_flow import TrafficFlow, TrafficFlowManager, VehicleType
from simulation.simulation_utils import run_simulation, analyze_tripinfo

# Simulation Constants
HOUR_DURATION = 3600       # duration of one hour in seconds
NUM_HOURS = 3              # number of hours to simulate
WARMUP_TIME = 4000         # warmup time in seconds

NORTH_SOUTH_VEHICLES_PER_MINUTE = 10  # Vehicles per minute (North-South/South-North)
WEST_EAST_VEHICLES_PER_MINUTE = 6     # Vehicles per minute (West-East/East-West)

# Derived Parameters
LAMBDA_RATE_NS = NORTH_SOUTH_VEHICLES_PER_MINUTE / 60              # vehicles per second
LAMBDA_RATE_WE = WEST_EAST_VEHICLES_PER_MINUTE / 60                # vehicles per second
SIMULATION_DURATION = HOUR_DURATION * NUM_HOURS + WARMUP_TIME      # total simulation time in seconds

# Files
route_file = "configs/poisson_routes.rou.xml"
config_file = "configs/poisson_config.sumocfg"
tripinfo_file = "configs/tripinfo.xml"


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
    run_simulation(config_file, SIMULATION_DURATION, tripinfo_file, gui=gui)
    analyze_tripinfo(tripinfo_file, warmup_time=WARMUP_TIME)
