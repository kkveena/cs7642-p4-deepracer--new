CS 7642 Project 4: AWS DeepRacer Optimization and Adaptation

This repository contains the implementation for Project 4, focusing on training a reinforcement learning agent for autonomous racing using the AWS DeepRacer simulation environment.

Note: This project was executed on the Georgia Tech PACE ICE cluster using a CPU. The setup instructions below are  for this high-performance computing environment using Apptainer (Singularity).

1. Environment Setup (PACE ICE)

Prerequisites

Access to PACE ICE interactive session (VS Code recommended).

Compute node with NVIDIA GPU (Tested on H100/H200).

apptainer module available.

anaconda3 module available.

Installation

Clone the Repository:

git clone [https://github.com/kkveena/cs7642-p4-deepracer--new.git](https://github.com/kkveena/cs7642-p4-deepracer--new.git)
cd cs7642-p4-deepracer


Create Conda Environment:

module load anaconda3
conda create -y -n deepracer python=3.9
conda activate deepracer
pip install -r requirements.txt
# Install additional dependencies for plotting/evaluation
pip install pandas seaborn matplotlib tensorboard


Build the Apptainer Container:

If deepracer.sif is missing, build it using the provided definition file. (Note: On PACE, this might require specific flags or fakeroot workarounds depending on cluster configuration. See scripts/start_deepracer.sh logic).

2. Project Structure & Custom Scripts

This repository includes custom scripts to manage the training lifecycle on PACE, ensuring stability and preventing resource conflicts.

configs/: Configuration files for Agent, Environment, Reward Function, and Hyperparameters.

src/: Source code for PPO Agent and utilities.

scripts/:

train_tt.sh: The main entry point for the training job.

start_deepracer.sh: Launches the Apptainer container with correct network/GPU settings.

run_training.sh: Wrapper to launch a training session for a specific track.

run_transfer.sh: Wrapper to launch Transfer Learning (fine-tuning) on a new track.

evaluate_wide.py / evaluate_smile.py / evaluate_h2b.py: Track-specific evaluation scripts that generate metrics and videos.

make_plots.py: Generates performance plots from TensorBoard logs.

3. Execution Guide


Part 1: Time Trial (Wide Track)

Train:

./scripts/run_training.sh reInvent2019_wide 15000 999 training_wide.log


Monitor progress: tail -f training_wide.log

Output model: runs/<run_name>/model_15000.pt

Evaluate:
After training completes:

# 1. Stage the model
cp runs/<run_name>/model_15000.pt saved_models_part1/model_wide_track.pt

# 2. Launch Simulator & Run Eval
export PYTHONPATH=$PWD
python scripts/evaluate_wide.py


Output: final_videos/reInvent2019_wide.mp4, final_report_part1/metrics.json

Part 2a: Object Avoidance (Smile Track)

Train (Transfer Learning or Fresh):
We used a fresh training run for stability on the H100.

./scripts/run_smile_training.sh


Evaluate:

# 1. Stage model
cp runs/<run_name>/model_15000.pt saved_models_part2/model_oa_track.pt

# 2. Run Eval
python scripts/evaluate_oa.py


Part 2b: Head-to-Bot (Smile Track)

Train:

./scripts/run_vegas_training.sh # (Renamed/Adapted for H2B config)


Note: Ensure configs/environment_params.yaml is set to 3 Bots / 0 Obstacles.

Evaluate:

# 1. Stage model
cp runs/<run_name>/model_15000.pt saved_models_part2/model_h2b_track.pt

# 2. Run Eval
python scripts/evaluate_h2b.py


4. Reproducibility Notes

CPU Mode: To prevent synchronization crashes between the fast H100 GPU and the single-threaded simulator, training was forced to CPU-Only mode via CUDA_VISIBLE_DEVICES="". This ensures stability.

Mock Libraries: A fake_libs/netifaces.py shim is injected via SINGULARITYENV_PYTHONPATH to resolve a Python 2.7/3.x dependency conflict inside the container.

Video Recording: Video generation is disabled during training to save resources (render_mode=None) and enabled only during the Evaluation phase (render_mode='rgb_array').

5. Results

Plots: Training metrics and Evaluation progress charts are