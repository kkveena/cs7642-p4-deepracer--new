#!/bin/bash
set -e

# --- CONFIGURATION ---
STEPS="${1:-300000}"
SEED="${2:-999}"
LOGFILE="training_part1.log"

echo "=========================================================="
echo "   🚀 LAUNCHING PART 1 (Robust Mode - Fixed Conda)"
echo "   STEPS: $STEPS | SEED: $SEED"
echo "=========================================================="

# 1. CLEANUP
echo "[1/4] Cleaning up..."
pkill -u $USER -9 python > /dev/null 2>&1 || true
pkill -u $USER -9 deepracer > /dev/null 2>&1 || true
pkill -u $USER -9 apptainer > /dev/null 2>&1 || true
/usr/sbin/fuser -k -n tcp 9194 > /dev/null 2>&1 || true
rm /dev/shm/*$USER* 2>/dev/null || true

# 2. PREPARE MOCK LIBRARIES (Netifaces Fix)
mkdir -p $HOME/scratch/fake_libs
cat <<EOM > $HOME/scratch/fake_libs/netifaces.py
AF_INET = 2
def interfaces(): return ['lo']
def ifaddresses(iface): return {2: [{'addr': '127.0.0.1'}]}
EOM

# 3. ENVIRONMENT SETUP
echo "[2/4] Setting Environment..."

# --- FIXED CONDA INIT ---
# Ensure we can use the module command
source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda3

# Initialize Conda for this script shell (Standard method)
eval "$(conda shell.bash hook)"
conda activate deepracer
# ------------------------

# Critical Exports
export NO_PROXY=localhost,127.0.0.1,::1,.gatech.edu
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=localhost
export SDL_VIDEODRIVER=dummy
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export SINGULARITYENV_PYTHONPATH=$HOME/scratch/fake_libs

# 4. LAUNCH WITH THROTTLING
echo "[3/4] Launching Training..."

# We use 'nice -n 10' to lower priority so Simulator gets CPU preference
nohup nice -n 10 xvfb-run -a -s "-screen 0 1024x768x24" ./scripts/train_tt.sh \
  --env baseline \
  --agent baseline \
  --reward baseline \
  --hparams baseline \
  --steps "$STEPS" \
  --seed "$SEED" > "$LOGFILE" 2>&1 &

PID=$!
echo "✅ Training Started (PID: $PID)"
echo "📋 Monitoring: tail -f $LOGFILE"
