import os
import subprocess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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

    import matplotlib.pyplot as plt
import numpy as np

def plot_vehicle_counts_over_time(tripinfo_file, simulation_duration):
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
        plt.plot(range(simulation_duration + 1), smoothed, label=vtype)

    plt.title("Numero di veicoli nel sistema nel tempo")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Veicoli presenti")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def jain_index(x):
    x = np.array(x)
    if np.sum(x) == 0:
        return 1.0  
    return (np.sum(x) ** 2) / (len(x) * np.sum(x ** 2))

def lorenz_curve(data):
    sorted_data = np.sort(data)
    cum_data = np.cumsum(sorted_data)
    total = cum_data[-1]
    if total == 0:
        return np.linspace(0, 1, len(data)), np.linspace(0, 1, len(data))
    lorenz = cum_data / total
    x_vals = np.linspace(0, 1, len(data))
    return x_vals, lorenz

def analyze_fairness_and_distribution(tripinfo_file, vtypes):
    if not os.path.exists(tripinfo_file):
        print(f"Error: {tripinfo_file} not found.")
        return

    df = pd.read_xml(tripinfo_file)
    df_filtered = df[df["vType"].isin(vtypes)]

    if df_filtered.empty:
        print("No vehicles found for selected vTypes.")
        return

    metrics = {
        "duration": df_filtered["duration"].values,
        "timeLoss": df_filtered["timeLoss"].values,
        "waitingTime": df_filtered["waitingTime"].values
    }

    print(f"\n--- Jain's Fairness Index ---")
    for key, values in metrics.items():
        fairness = jain_index(values)
        print(f"{key:<12}: {fairness:.4f}")

    # Lorenz curves
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, (key, values) in enumerate(metrics.items()):
        x_vals, lorenz_vals = lorenz_curve(values)
        axs[i].plot(x_vals, lorenz_vals, label=f'Lorenz - {key}')
        axs[i].plot([0, 1], [0, 1], 'k--', label='Uguaglianza perfetta')
        axs[i].set_title(f"Lorenz Curve - {key}")
        axs[i].set_xlabel("Frazione veicoli")
        axs[i].set_ylabel("Frazione cumulata")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

    # Distribuzioni
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, (key, values) in enumerate(metrics.items()):
        sns.histplot(values, bins=30, kde=True, ax=axs[i])
        axs[i].set_title(f"Distribuzione - {key}")
        axs[i].set_xlabel("Secondi")
        axs[i].set_ylabel("Frequenza")
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()
