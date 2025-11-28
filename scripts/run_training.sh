#!/bin/bash
set -e

# --- INPUT PARAMETERS ---
TRACK_NAME="${1:-reInvent2019_wide}"  # Default: Wide track
STEPS="${2:-300000}"                  # Default: 300k steps
SEED="${3:-999}"                      # Default: Seed 999
LOGFILE="${4:-training_${TRACK_NAME}.log}"

echo "=========================================================="
echo "   🚀 STARTING TRAINING SESSION"
echo "   TRACK: $TRACK_NAME | STEPS: $STEPS | SEED: $SEED"
echo "   LOG:   $LOGFILE"
echo "=========================================================="

# 1. DYNAMIC CONFIGURATION UPDATE
# We update the WORLD_NAME in the yaml file to match your requested track
# This uses 'sed' to find the line starting with WORLD_NAME and replace it.
CONFIG_FILE="configs/environment_params.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "[Config] Setting WORLD_NAME to $TRACK_NAME in $CONFIG_FILE..."
    sed -i "s/^WORLD_NAME:.*/WORLD_NAME: \"$TRACK_NAME\"/" "$CONFIG_FILE"
else
    echo "❌ Error: $CONFIG_FILE not found!"
    exit 1
fi

# 2. CLEANUP (The "Victory" Protocol)
echo "[Cleanup] Killing old processes..."
pkill -u $USER -9 python > /dev/null 2>&1 || true
pkill -u $USER -9 deepracer > /dev/null 2>&1 || true
pkill -u $USER -9 apptainer > /dev/null 2>&1 || true
/usr/sbin/fuser -k -n tcp 9194 > /dev/null 2>&1 || true
rm /dev/shm/*$USER* 2>/dev/null || true

# 3. ENVIRONMENT SETUP
echo "[Setup] Loading Environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepracer
chmod +x scripts/*.sh

# Critical Exports for H100/PACE
export NO_PROXY=localhost,127.0.0.1,::1,.gatech.edu
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=localhost
export SDL_VIDEODRIVER=dummy
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 4. LAUNCH TRAINING
echo "[Launch] Starting Training in background..."
nohup xvfb-run -a -s "-screen 0 1024x768x24" ./scripts/train_tt.sh \
  --env baseline \
  --agent baseline \
  --reward baseline \
  --hparams baseline \
  --steps "$STEPS" \
  --seed "$SEED" > "$LOGFILE" 2>&1 &

PID=$!
echo "✅ Training Started! (PID: $PID)"
echo "📋 Monitor with: tail -f $LOGFILE"
