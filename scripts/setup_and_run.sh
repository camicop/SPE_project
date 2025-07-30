#!/bin/bash

# Stop script if something goes wrong
set -e

PYTHON="python3"

# 1. Create SUMO network
netconvert \
    -n configs/nodes.nod.xml \
    -e configs/edges.edg.xml \
    -x configs/connections.con.xml \
    -o configs/simple_intersection.net.xml

# 2. Create virtual environment
if [ ! -d "sim-venv" ]; then
    $PYTHON -m venv sim-venv
fi

# 3. Activate virtual environment
if [ -f "sim-venv/bin/activate" ]; then
    source sim-venv/bin/activate
elif [ -f "sim-venv/Scripts/activate" ]; then
    source sim-venv/Scripts/activate
else
    echo "ERROR: Could not find the virtualenv activation script."
    exit 1
fi

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run simulation using the script
#$PYTHON -m scripts.run_sumo_warmup
$PYTHON -m scripts.run_sumo_fairness
#$PYTHON -m scripts.run_sumo "$@"