#!/bin/bash

# Stop script if something goes wrong
set -e

# 1. Sumo network construction
netconvert -n nodes.nod.xml -e edges.edg.xml -x connections.con.xml -o simple_intersection.net.xml

# 2. Creation of venv
if [ ! -d "sim-venv" ]; then
    python3 -m venv sim-venv
fi

# 3. Activation of venv
source sim-venv/bin/activate

# 4. Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run simulation
python3 run_sumo_poisson.py gui
