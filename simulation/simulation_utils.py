import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import traci
import xml.etree.ElementTree as ET

import numpy as np

from scipy.signal import savgol_filter

def run_simulation(config_file, simulation_duration, tripinfo_file, gui=False):
    update_config_end_time(config_file, simulation_duration)
    print("Starting SUMO simulation...")

    command = ["sumo-gui" if gui else "sumo", "-c", config_file, "--no-step-log"]

    result = subprocess.run(command, capture_output=True, text=True)

    print("SUMO simulation finished.")
    if result.returncode != 0:
        print("SUMO error:", result.stderr)
    else:
        print(result.stdout)

def analyze_tripinfo(tripinfo_file, warmup_time):
    if not os.path.exists(tripinfo_file):
        print(f"Error: {tripinfo_file} not found.")
        return

    df = pd.read_xml(tripinfo_file)

    df = df[df["depart"] >= warmup_time]

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


def plot_multiple_time_series(series_list, title, y_label, labels=None, show_legend=True, apply_smoothing=True):
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(12, 6))
    for i, series in enumerate(series_list):
        data = savgol_filter(series, window_length=31, polyorder=3) if apply_smoothing else series
        label = labels[i] if labels and i < len(labels) else f"Simulation {i}"
        plt.plot(data, '-', label=label)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(y_label)
    if show_legend:
        plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def update_traffic_light_file(
    trafficlight_path: str,
    ns_green_duration: int,
    ew_green_duration: int,
    yellow_duration: int = 5,
    tl_id: str = "center",
    program_id: str = "myProgram",
    tl_type: str = "static",
    offset: str = "0"
) -> None:

    tree = ET.parse(trafficlight_path)
    root = tree.getroot()

    if root.tag != "additional":
        additional = root.find("additional")
        if additional is None:
            raise RuntimeError(f"No <additional> root found in {trafficlight_path}")
        root = additional

    for old in root.findall(f"tlLogic[@id='{tl_id}']"):
        root.remove(old)

    tl_logic = ET.SubElement(
        root, "tlLogic",
        id=tl_id,
        type=tl_type,
        programID=program_id,
        offset=offset
    )

    # Green Nord-Sud
    ET.SubElement(tl_logic, "phase", duration=str(ns_green_duration), state="GrGr")
    # Yellow Nord-Sud
    ET.SubElement(tl_logic, "phase", duration=str(yellow_duration),  state="yryr")
    # Green Est-Ovest
    ET.SubElement(tl_logic, "phase", duration=str(ew_green_duration), state="rGrG")
    # Yellow Est-Ovest
    ET.SubElement(tl_logic, "phase", duration=str(yellow_duration),  state="ryry")

    tree.write(trafficlight_path, encoding='utf-8', xml_declaration=True)