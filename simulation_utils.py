import os
import subprocess
import pandas as pd

def run_simulation(config_file, simulation_duration, gui=False):
    update_config_end_time(config_file, simulation_duration)
    print("Starting SUMO simulation...")
    command = ["sumo-gui" if gui else "sumo", "-c", config_file]
    result = subprocess.run(command, capture_output=True, text=True)

    print("SUMO simulation finished.")
    if result.returncode != 0:
        print("SUMO error:", result.stderr)
    else:
        print(result.stdout)

def analyze_tripinfo(tripinfo_file):
    if not os.path.exists(tripinfo_file):
        print(f"Error: {tripinfo_file} not found.")
        return

    df = pd.read_xml(tripinfo_file)

    mean_wait = df["waitingTime"].mean()
    mean_duration = df["duration"].mean()
    mean_loss = df["timeLoss"].mean()

    print("\n--- Simulation statistics ---")
    print(f"Number of vehicles simulated: {len(df)}")
    print(f"Average waiting time at traffic lights (seconds): {mean_wait:.2f}")
    print(f"Average trip duration (seconds): {mean_duration:.2f}")
    print(f"Average time loss (seconds): {mean_loss:.2f}")

def update_config_end_time(config_path, new_end_time):
    from xml.etree import ElementTree as ET

    tree = ET.parse(config_path)
    root = tree.getroot()

    for time_elem in root.findall("time"):
        for child in time_elem:
            if child.tag == "end":
                child.set("value", str(new_end_time))
                break
        else:
            # Se non esiste, aggiungilo
            end_elem = ET.SubElement(time_elem, "end")
            end_elem.set("value", str(new_end_time))

    tree.write(config_path)