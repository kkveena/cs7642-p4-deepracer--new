#!/bin/bash
set -e
LOGFILE="training_vegas.log"
echo "🚀 LAUNCHING VEGAS TRACK TRAINING (15k Steps)..."

# Environment
module load anaconda3
conda activate deepracer
export CUDA_VISIBLE_DEVICES="" 
export NO_PROXY=localhost,127.0.0.1,::1,.gatech.edu
export ROS_IP=127.0.0.1
export ROS_HOSTNAME=localhost
export SDL_VIDEODRIVER=dummy
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export SINGULARITYENV_PYTHONPATH=$HOME/scratch/fake_libs

# Launch
nohup xvfb-run -a -s "-screen 0 1024x768x24" ./scripts/train_tt.sh \
  --env baseline \
  --agent baseline \
  --reward baseline \
  --hparams baseline \
  --steps 15000 \
  --seed 42 > "$LOGFILE" 2>&1 &

echo "✅ Started! Monitor: tail -f $LOGFILE"
