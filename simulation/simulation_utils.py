import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import traci
import xml.etree.ElementTree as ET

import numpy as np

from scipy.signal import savgol_filter

def run_simulation(config_file, simulation_duration, tripinfo_file, gui=False, quiet=False):
    update_config_end_time(config_file, simulation_duration)
    
    if not quiet:
        print("Starting SUMO simulation...")

    command = ["sumo-gui" if gui else "sumo", "-c", config_file, "--no-step-log"]

    result = subprocess.run(command, capture_output=True, text=True)

    if not quiet:
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
            end_elem = ET.SubElement(time_elem, "end")
            end_elem.set("value", str(new_end_time))

    tree.write(config_path)


def plot_multiple_time_series(series_list, title, y_label, labels=None, show_legend=True):
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(12, 6))
    for i, series in enumerate(series_list):
        label = labels[i] if labels and i < len(labels) else f"Simulation {i}"
        plt.plot(series, '-', label=label)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(y_label)
    if show_legend:
        plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def write_traffic_light_file(
    trafficlight_path: str,
    ns_green_duration: int,
    ew_green_duration: int,
    yellow_duration: int = 5,
    tl_id: str = "center",
    program_id: str = "myProgram",
    tl_type: str = "static",
    offset: str = "0"
) -> None:

    # Prepara le linee da scrivere
    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<additional>',
        f'  <tlLogic id="{tl_id}" type="{tl_type}" programID="{program_id}" offset="{offset}">',
        f'    <!-- Green Nord-Sud -->',
        f'    <phase duration="{ns_green_duration}" state="GrGr"/>',
        f'    <!-- Yellow Nord-Sud -->',
        f'    <phase duration="{yellow_duration}" state="yryr"/>',
        f'    <!-- Green Est-Ovest -->',
        f'    <phase duration="{ew_green_duration}" state="rGrG"/>',
        f'    <!-- Yellow Est-Ovest -->',
        f'    <phase duration="{yellow_duration}" state="ryry"/>',
        '  </tlLogic>',
        '</additional>',
        ''  # newline finale
    ]

    # Scrivi il file sovrascrivendo tutto
    with open(trafficlight_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_lines))


def get_inserted_vehicle_count(tripinfo_path: str) -> int:
    tree = ET.parse(tripinfo_path)
    root = tree.getroot()
    return len(root.findall("tripinfo"))  # Ogni <tripinfo> corrisponde a un veicolo inserito

def get_loaded_vehicle_count(route_file_path: str) -> int:
    tree = ET.parse(route_file_path)
    root = tree.getroot()
    return len(root.findall("vehicle")) + len(root.findall("trip"))  # Se usi <trip> al posto di <vehicle>