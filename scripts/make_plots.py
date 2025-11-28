import os
import sys
import glob
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
RUNS_DIR = "runs"
EVAL_DIR = "final_videos"
OUTPUT_DIR = "final_report_plots"
TRACK_NAME = "reInvent2019_wide"

def get_run_dir():
    """Determines which run directory to use."""
    # 1. If user provided a path arg, use it
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        if os.path.exists(run_path):
            return run_path
        else:
            print(f"❌ Error: Provided path '{run_path}' does not exist.")
            sys.exit(1)
            
    # 2. Fallback: Find latest
    run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "*")))
    if not run_dirs:
        return None
    return run_dirs[-1]

def extract_training_data(run_dir):
    """Reads TensorBoard logs to get Reward and Length."""
    print(f"📂 Reading logs from: {run_dir}")
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        print("❌ No event file found.")
        return None

    ea = EventAccumulator(event_files[0])
    ea.Reload()

    steps = []
    rewards = []
    lengths = []

    if 'charts/episodic_return' in ea.scalars.Keys():
        for e in ea.Scalars('charts/episodic_return'):
            steps.append(e.step)
            rewards.append(e.value)
    
    if 'charts/episodic_length' in ea.scalars.Keys():
        for e in ea.Scalars('charts/episodic_length'):
            lengths.append(e.value)

    min_len = min(len(steps), len(rewards), len(lengths))
    
    if min_len == 0:
        print("⚠️ Log file exists but contains no data (Empty Run).")
        return None

    df = pd.DataFrame({
        "Step": steps[:min_len],
        "Reward": rewards[:min_len],
        "Length": lengths[:min_len]
    })
    
    df["Efficiency"] = df["Reward"] / df["Length"]
    df["Reward_Smooth"] = df["Reward"].rolling(window=10).mean()
    
    return df

def plot_training_metrics(df, output_dir):
    if df is None or df.empty:
        print("⚠️ No training data found to plot.")
        return

    print("📊 Generating Training Plots...")
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.lineplot(data=df, x="Step", y="Reward", alpha=0.3, ax=axes[0], color="blue", label="Raw")
    sns.lineplot(data=df, x="Step", y="Reward_Smooth", ax=axes[0], color="darkblue", label="Smoothed (10)")
    axes[0].set_title("Metric 1: Episodic Reward")
    
    sns.lineplot(data=df, x="Step", y="Length", ax=axes[1], color="green")
    axes[1].set_title("Metric 2: Episode Length")
    
    sns.lineplot(data=df, x="Step", y="Efficiency", ax=axes[2], color="orange")
    axes[2].set_title("Metric 3: Driving Efficiency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    print(f"✅ Saved: {output_dir}/training_metrics.png")

def plot_evaluation_metrics(eval_dir, output_dir):
    json_path = os.path.join(eval_dir, f"{TRACK_NAME}_metrics.json")
    if not os.path.exists(json_path):
        print(f"⚠️ Evaluation JSON not found: {json_path}")
        return

    print(f"📂 Reading Evaluation: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df["Lap"] = range(1, len(df) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.barplot(data=df, x="Lap", y="progress", ax=axes[0], palette="Blues_d")
    axes[0].set_title(f"Evaluation: Progress % ({TRACK_NAME})")
    axes[0].set_ylim(0, 110)
    axes[0].axhline(100, color='red', linestyle='--')
    
    completed_laps = df.dropna(subset=['lap_time'])
    if not completed_laps.empty:
        sns.barplot(data=completed_laps, x="Lap", y="lap_time", ax=axes[1], palette="Greens_d")
        axes[1].set_title(f"Evaluation: Lap Time ({TRACK_NAME})")
    else:
        axes[1].text(0.5, 0.5, "No Complete Laps", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"))
    print(f"✅ Saved: {output_dir}/evaluation_metrics.png")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    run_dir = get_run_dir()
    if run_dir:
        df_train = extract_training_data(run_dir)
        plot_training_metrics(df_train, OUTPUT_DIR)
    
    plot_evaluation_metrics(EVAL_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()
