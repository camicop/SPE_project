import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

import traci
import xml.etree.ElementTree as ET

import numpy as np

from scipy.signal import savgol_filter

def run_simulation(config_file, simulation_duration, gui=False, tripinfo_file="tripinfo.xml"):
    update_config_end_time(config_file, simulation_duration)
    print("Starting SUMO simulation...")
    command = ["sumo-gui" if gui else "sumo", "-c", config_file]
    command += ["--tripinfo-output", tripinfo_file]
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

def plot_vehicle_counts_over_time(tripinfo_file, simulation_duration, warmup_time=0):
    if not os.path.exists(tripinfo_file):
        print(f"Error: {tripinfo_file} not found.")
        return

    df = pd.read_xml(tripinfo_file)

    # Inizializza il dizionario dei conteggi
    time_series = {}
    vehicle_types = df["vType"].unique()
    for vtype in vehicle_types:
        time_series[vtype] = np.zeros(simulation_duration + 1)

    # Per ogni veicolo, aggiungi 1 nei secondi in cui è nel sistema
    for _, row in df.iterrows():
        vtype = row["vType"]
        depart = int(row["depart"])
        arrival = int(row["arrival"])
        # Incrementa i secondi in cui l'auto è nel sistema
        time_series[vtype][depart:arrival + 1] += 1

    # Plot
    plt.figure(figsize=(12, 6))
    for vtype, counts in time_series.items():
        smoothed = savgol_filter(counts, window_length=31, polyorder=3)  # regola i valori se necessario
    start_plot = warmup_time if warmup_time else 0
    plt.plot(range(start_plot, simulation_duration + 1), smoothed[start_plot:], label=vtype)

    plt.title("Numero di veicoli nel sistema nel tempo")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Veicoli presenti")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def estimate_warmup_time(tripinfo_file, simulation_duration, threshold=0.01, window=60):
    import numpy as np
    if not os.path.exists(tripinfo_file):
        print(f"Error: {tripinfo_file} not found.")
        return 0

    df = pd.read_xml(tripinfo_file)

    # Costruisce la serie temporale come nel plot
    time_series = np.zeros(simulation_duration + 1)
    for _, row in df.iterrows():
        depart = int(row["depart"])
        arrival = int(row["arrival"])
        time_series[depart:arrival + 1] += 1

    # Calcolo della media mobile su finestra
    rolling_avg = pd.Series(time_series).rolling(window=window).mean()

    # Trova il primo punto dove la variazione percentuale si stabilizza sotto la soglia
    for t in range(window, simulation_duration - window):
        delta = abs(rolling_avg[t] - rolling_avg[t - 1]) / (rolling_avg[t - 1] + 1e-6)
        if delta < threshold:
            print(f"Estimated warm-up ends at second {t}")
            return t

    print("No stable warm-up period detected; defaulting to 0.")
    return 0

def plot_multiple_time_series(series_list, title, y_label, highlight_index=None):
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    plt.figure(figsize=(12, 6))
    for i, series in enumerate(series_list):
        smoothed = savgol_filter(series, window_length=31, polyorder=3)
        label = "Simulazione manuale (GUI)" if i == highlight_index else f"Simulazione {i}"
        style = '-' if i == highlight_index else '--'
        plt.plot(smoothed, style, label=label)

    plt.title(title)
    plt.xlabel("Tempo (s)")
    plt.ylabel(y_label)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def _update_config_for_adaptive(config_path, new_end_time, tripinfo_output):
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

    output_elem = root.find("output")
    if output_elem is None:
        output_elem = ET.SubElement(root, "output")
    tripinfo_elem = output_elem.find("tripinfo-output")
    if tripinfo_elem is None:
        tripinfo_elem = ET.SubElement(output_elem, "tripinfo-output")
    tripinfo_elem.set("value", tripinfo_output)

    tree.write(config_path)


def run_adaptive_simulation(config_file, simulation_duration, tripinfo_file="tripinfo_adaptive.xml"):
    _update_config_for_adaptive(config_file, simulation_duration, tripinfo_file)

    sumo_cmd = ["sumo", "-c", config_file, "--tripinfo-output", tripinfo_file]
    traci.start(sumo_cmd)

    tls_id = traci.trafficlight.getIDList()[0]
    current_phase = 0
    phase_duration = 10
    phase_timer = 0

    while traci.simulation.getTime() < simulation_duration:
        traci.simulationStep()
        phase_timer += 1

        if phase_timer >= phase_duration:
            ns_queue = (
                traci.lane.getLastStepVehicleNumber("north2center_0") +
                traci.lane.getLastStepVehicleNumber("south2center_0")
            )
            ew_queue = (
                traci.lane.getLastStepVehicleNumber("east2center_0") +
                traci.lane.getLastStepVehicleNumber("west2center_0")
            )

            if ns_queue > ew_queue:
                current_phase = 0  # Nord-Sud verde
            else:
                current_phase = 2  # Est-Ovest verde

            traci.trafficlight.setPhase(tls_id, current_phase)
            phase_timer = 0

    traci.close()