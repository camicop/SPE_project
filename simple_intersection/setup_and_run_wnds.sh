#!/bin/bash

set -e

# Percorso assoluto a Python (modifica se serve)
PYTHON="python"

# 1. Sumo network construction
netconvert -n nodes.nod.xml -e edges.edg.xml -x connections.con.xml -o simple_intersection.net.xml

# 2. Creation of venv
if [ ! -d "sim-venv" ]; then
    $PYTHON -m venv sim-venv
fi

# 3. Activation of venv on winndows
source sim-venv/Scripts/activate

# 4. Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run simulation
$PYTHON run_sumo.py gui
