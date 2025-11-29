#!/bin/bash
set -e

# --- INPUTS ---
PRETRAINED_MODEL="$1"   # Path to the .pt file
NEW_TRACK="$2"          # e.g. reInvent2019_track
STEPS="${3:-15000}"     # Default 15k steps
SEED="${4:-42}"
LOGFILE="training_${NEW_TRACK}.log"

if [ -z "$PRETRAINED_MODEL" ] || [ -z "$NEW_TRACK" ]; then
  echo "Usage: ./scripts/run_transfer.sh <PATH_TO_MODEL> <NEW_TRACK_NAME> [STEPS] [SEED]"
  exit 1
fi

echo "=========================================================="
echo "   🔄 TRANSFER LEARNING SESSION"
echo "   BASE MODEL: $PRETRAINED_MODEL"
echo "   TARGET TRACK: $NEW_TRACK"
echo "=========================================================="

# 1. UPDATE CONFIG: SET NEW TRACK
sed -i "s/^WORLD_NAME:.*/WORLD_NAME: \"$NEW_TRACK\"/" configs/environment_params.yaml

# 2. UPDATE CONFIG: INJECT PRETRAINED PATH
sed -i '/pretrained_path:/d' configs/hyper_params.yaml
echo "pretrained_path: \"$PRETRAINED_MODEL\"" >> configs/hyper_params.yaml

# 3. CLEANUP
echo "[Cleanup] Killing old processes..."
pkill -u $USER -9 python > /dev/null 2>&1 || true
pkill -u $USER -9 deepracer > /dev/null 2>&1 || true
pkill -u $USER -9 apptainer > /dev/null 2>&1 || true
/usr/sbin/fuser -k -n tcp 9194 > /dev/null 2>&1 || true
rm /dev/shm/*$USER* 2>/dev/null || true

# 4. SETUP ENVIRONMENT (Robust)
echo "[Setup] Loading Environment..."
source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate deepracer

# 5. CRITICAL EXPORTS (CPU Mode + Mock Libs)
export CUDA_VISIBLE_DEVICES=""
export NO_PROXY=localhost,127.0.0.1,::1,.gatech.edu
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=localhost
export SDL_VIDEODRIVER=dummy
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export SINGULARITYENV_PYTHONPATH=$HOME/scratch/fake_libs

# 6. LAUNCH
echo "[Launch] Fine-tuning on $NEW_TRACK..."
# Using 'nice -n 0' for standard priority on CPU
nohup xvfb-run -a -s "-screen 0 1024x768x24" ./scripts/train_tt.sh \
  --env baseline \
  --agent baseline \
  --reward baseline \
  --hparams baseline \
  --steps "$STEPS" \
  --seed "$SEED" > "$LOGFILE" 2>&1 &

PID=$!
echo "✅ Transfer Training Started (PID: $PID)"
echo "📋 Monitor: tail -f $LOGFILE"
