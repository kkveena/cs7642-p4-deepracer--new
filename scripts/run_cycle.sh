#!/bin/bash

# Default Parameters (Can be overridden by arguments)
STEPS="${1:-300000}"           # Default to 300k steps if not provided
SEED="${2:-999}"               # Default to seed 999
LOGFILE="${3:-training.log}"   # Default log filename

echo "=========================================================="
echo "   🚀 DEEPRACER LAUNCH CYCLE"
echo "   STEPS: $STEPS | SEED: $SEED | LOG: $LOGFILE"
echo "=========================================================="

# --- STEP 1: CLEANUP (The Nuclear Option) ---
echo "[1/5] Cleaning up old processes..."
# We use '|| true' to suppress errors if no processes are found
pkill -u $USER -9 python > /dev/null 2>&1 || true
pkill -u $USER -9 deepracer > /dev/null 2>&1 || true
pkill -u $USER -9 apptainer > /dev/null 2>&1 || true

# Kill the specific port (Silent kill)
/usr/sbin/fuser -k -n tcp 9194 > /dev/null 2>&1 || true

# Clean shared memory
rm /dev/shm/*$USER* 2>/dev/null || true

# --- STEP 2: PERMISSIONS ---
echo "[2/5] Ensuring script permissions..."
chmod +x scripts/*.sh

# --- STEP 3: ENVIRONMENT SETUP ---
echo "[3/5] Loading Environment..."
# Ensure Conda is available in this subshell
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepracer

# --- STEP 4: EXPORTS (The Critical Fixes) ---
export NO_PROXY=localhost,127.0.0.1,::1,.gatech.edu
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=localhost
export SDL_VIDEODRIVER=dummy
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# --- STEP 5: LAUNCH ---
echo "[4/5] Launching Training Background Job..."

nohup xvfb-run -a -s "-screen 0 1024x768x24" ./scripts/train_tt.sh \
  --env baseline \
  --agent baseline \
  --reward baseline \
  --hparams baseline \
  --steps "$STEPS" \
  --seed "$SEED" > "$LOGFILE" 2>&1 &

PID=$!
echo "✅ SUCCESS! Training started with PID: $PID"
echo "📋 Monitoring log file: $LOGFILE"
echo "=========================================================="
echo "Use 'tail -f $LOGFILE' to watch progress."
echo "Use 'kill $PID' to stop."

