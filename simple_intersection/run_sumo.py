import numpy as np
import subprocess
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from traffic_flow import TrafficFlow, TrafficFlowManager, VehicleType
from simulation_utils import run_simulation, analyze_tripinfo, plot_vehicle_counts_over_time, estimate_warmup_time, plot_multiple_time_series

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

#if __name__ == "__main__":
    #gui = False
    #if len(sys.argv) > 1:
     #   if sys.argv[1].lower() in ["gui", "--gui", "-g"]:
      #      gui = True

    #generate_routes()
    #run_simulation(config_file, SIMULATION_DURATION, gui=gui)
    #analyze_tripinfo(tripinfo_file)
    #plot_vehicle_counts_over_time(tripinfo_file, SIMULATION_DURATION)


def first_manual_gui_run():
    print("\n=== PRIMA SIMULAZIONE CON GUI ===")
    generate_routes()
    run_simulation(config_file, SIMULATION_DURATION, gui=True)

    # Estrai la curva dei veicoli
    df = pd.read_xml(tripinfo_file)
    series = np.zeros(SIMULATION_DURATION + 1)
    for _, row in df.iterrows():
        depart = int(row["depart"])
        arrival = int(row["arrival"])
        series[depart:arrival + 1] += 1

    return series  # Restituisce la curva della simulazione manuale




def final_automatic_simulations(n_runs, warmup_time):
    results = []
    for i in range(n_runs):
        print(f"\n--- RUN {i + 1}/{n_runs} ---")
        generate_routes()
        run_simulation(config_file, SIMULATION_DURATION, gui=False)

        df = pd.read_xml(tripinfo_file)
        df = df[df["depart"] >= warmup_time]

        results.append({
            "vehicles": len(df),
            "avg_waiting": df["waitingTime"].mean(),
            "avg_duration": df["duration"].mean(),
            "avg_loss": df["timeLoss"].mean()
        })

    print("\n=== STATISTICHE AGGREGATE ===")
    print(f"Numero simulazioni: {n_runs}")
    print(f"Media veicoli simulati: {int(sum(r['vehicles'] for r in results) / n_runs)}")
    print(f"Media tempi attesa: {sum(r['avg_waiting'] for r in results)/n_runs:.2f} s")
    print(f"Media durata viaggio: {sum(r['avg_duration'] for r in results)/n_runs:.2f} s")
    print(f"Media perdita di tempo: {sum(r['avg_loss'] for r in results)/n_runs:.2f} s")

def estimate_average_warmup(n_runs=3, extra_series=None):
    warmups = []
    all_time_series = []

    if extra_series is not None:
        all_time_series.append(extra_series)

    for i in range(n_runs):
        print(f"\n--- WARM-UP SIMULATION {i+1}/{n_runs} ---")
        generate_routes()
        run_simulation(config_file, SIMULATION_DURATION, gui=False)

        warmup_time = estimate_warmup_time(tripinfo_file, SIMULATION_DURATION)
        warmups.append(warmup_time)

        df = pd.read_xml(tripinfo_file)
        series = np.zeros(SIMULATION_DURATION + 1)
        for _, row in df.iterrows():
            depart = int(row["depart"])
            arrival = int(row["arrival"])
            series[depart:arrival + 1] += 1
        all_time_series.append(series)

    # Mostra grafico combinato con label per il primo
    plot_multiple_time_series(all_time_series, highlight_index=0)

    avg_warmup = int(sum(warmups) / len(warmups))
    print(f"\n Warm-up medio stimato su {n_runs} simulazioni: {avg_warmup} secondi")
    return avg_warmup


if __name__ == "__main__":
    # 1. Manuale con GUI
    manual_series = first_manual_gui_run()  # curva manuale


    # 2. Stima del warm-up su più simulazioni
    estimated_warmup = estimate_average_warmup(n_runs=3, extra_series=manual_series)
    # 3. Simulazioni finali
    final_automatic_simulations(n_runs=10, warmup_time=estimated_warmup)
