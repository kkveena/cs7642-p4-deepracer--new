#!/bin/bash
set -e

echo "=========================================================="
echo "   🎥 STARTING ROBUST EVALUATION"
echo "=========================================================="

# 1. CLEANUP
echo "[1/4] Cleaning up..."
pkill -u $USER -9 python
pkill -u $USER -9 deepracer
pkill -u $USER -9 apptainer
/usr/sbin/fuser -k -n tcp 9194 2>/dev/null
rm /dev/shm/*$USER* 2>/dev/null

# 2. ENVIRONMENT
echo "[2/4] Setting Environment..."
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate deepracer

# CRITICAL EXPORTS FOR ZMQ CONNECTION
export NO_PROXY=localhost,127.0.0.1,::1,.gatech.edu
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=localhost
export SDL_VIDEODRIVER=dummy
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export SINGULARITYENV_PYTHONPATH=$HOME/scratch/fake_libs
export PYTHONPATH=$PWD

# 3. LAUNCH SIMULATOR (EVALUATION MODE)
echo "[3/4] Launching Simulator (Eval Mode)..."
# Added '-E true' to tell DeepRacer this is NOT training
nohup xvfb-run -a -s "-screen 0 1024x768x24" ./scripts/start_deepracer.sh -E true > eval_sim.log 2>&1 &

# 4. WAIT & RUN PYTHON
echo "[4/4] Waiting 90s for handshake..."
sleep 90

echo "🚀 Running Python Client..."
python scripts/evaluate_wide.py
