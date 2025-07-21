import numpy as np
import subprocess
import pandas as pd
import os
import sys

# Parametri simulazione
lambda_rate = 0.2   # arrivi veicoli al secondo (es. 12 veicoli/min)
simulation_time = 600      # durata simulazione in secondi

# File generati
route_file = "poisson_routes.rou.xml"
config_file = "poisson_config.sumocfg"
tripinfo_file = "tripinfo.xml"

def generate_routes():
    current_time = 0
    car_id = 0

    with open(route_file, "w") as f:
        f.write('<routes>\n')
        f.write('  <vType id="car" accel="1.0" decel="4.5" maxSpeed="13.9" length="5"/>\n')
        f.write('  <route id="north2south" edges="north2center center2south"/>\n')
        f.write('  <route id="south2north" edges="south2center center2north"/>\n')
        f.write('  <route id="east2west" edges="east2center center2west"/>\n')
        f.write('  <route id="west2east" edges="west2center center2east"/>\n')

        while current_time < simulation_time:
            inter_arrival = np.random.exponential(1 / lambda_rate)
            current_time += inter_arrival

            if current_time >= simulation_time:
                break

            route_id = np.random.choice(["north2south", "south2north", "east2west", "west2east"])

            f.write(f'  <vehicle id="car{car_id}" type="car" route="{route_id}" depart="{int(current_time)}"/>\n')
            car_id += 1

        f.write('</routes>')

    print(f"Generated {car_id} vehicles in {route_file}")


def run_simulation(gui=False):
    print("Starting SUMO simulation...")
    command = ["sumo-gui" if gui else "sumo", "-c", config_file]
    result = subprocess.run(command, capture_output=True, text=True)

    print("SUMO simulation finished.")
    if result.returncode != 0:
        print("SUMO error:", result.stderr)
    else:
        print(result.stdout)

def analyze_tripinfo():
    if not os.path.exists(tripinfo_file):
        print(f"Error: {tripinfo_file} not found.")
        return

    df = pd.read_xml(tripinfo_file)

    mean_wait = df["waitSteps"].mean()
    mean_duration = df["duration"].mean()
    mean_loss = df["timeLoss"].mean()

    print("\n--- Simulation statistics ---")
    print(f"Number of vehicles simulated: {len(df)}")
    print(f"Average waiting time at traffic lights (seconds): {mean_wait:.2f}")
    print(f"Average trip duration (seconds): {mean_duration:.2f}")
    print(f"Average time loss (seconds): {mean_loss:.2f}")

if __name__ == "__main__":
    gui = False
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ["gui", "--gui", "-g"]:
            gui = True

    generate_routes()
    run_simulation(gui=gui)
    analyze_tripinfo()
