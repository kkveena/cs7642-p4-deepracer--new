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
EVAL_DIR = "final_report_part1"  # Updated to match your output folder
OUTPUT_DIR = "final_report_plots"
TRACK_NAME = "reInvent2019_wide"

def get_run_dir():
    # Use the specific run provided arg, or find latest
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    # Fallback: Find largest run directory (most likely the valid one)
    run_dirs = glob.glob(os.path.join(RUNS_DIR, "*"))
    if not run_dirs: return None
    # Sort by size is safer than time if you have many empty starts
    # But simple time sort is usually fine
    return max(run_dirs, key=os.path.getmtime)

def extract_training_data(run_dir):
    print(f"📂 Reading Training Logs from: {run_dir}")
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        print("❌ No event file found.")
        return None

    ea = EventAccumulator(event_files[0])
    ea.Reload()

    steps, rewards, lengths = [], [], []
    
    if 'charts/episodic_return' in ea.scalars.Keys():
        for e in ea.Scalars('charts/episodic_return'):
            steps.append(e.step)
            rewards.append(e.value)
            
    if 'charts/episodic_length' in ea.scalars.Keys():
        for e in ea.Scalars('charts/episodic_length'):
            lengths.append(e.value)

    min_len = min(len(steps), len(rewards), len(lengths))
    df = pd.DataFrame({
        "Step": steps[:min_len],
        "Reward": rewards[:min_len],
        "Length": lengths[:min_len]
    })
    df["Efficiency"] = df["Reward"] / df["Length"]
    df["Reward_Smooth"] = df["Reward"].rolling(window=50).mean() # Smooth line
    return df

def plot_training(df, output_dir):
    if df is None or df.empty: return
    print("📊 Generating Training Plots...")
    sns.set_theme(style="darkgrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Reward
    sns.lineplot(data=df, x="Step", y="Reward", alpha=0.3, ax=axes[0], color="blue")
    sns.lineplot(data=df, x="Step", y="Reward_Smooth", ax=axes[0], color="darkblue", linewidth=2)
    axes[0].set_title(f"Training Reward ({TRACK_NAME})")
    
    # Length
    sns.lineplot(data=df, x="Step", y="Length", ax=axes[1], color="green")
    axes[1].set_title("Episode Length")
    
    # Efficiency
    sns.lineplot(data=df, x="Step", y="Efficiency", ax=axes[2], color="orange")
    axes[2].set_title("Reward per Step")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))

def plot_evaluation(eval_dir, output_dir):
    json_path = os.path.join(eval_dir, f"{TRACK_NAME}_metrics.json")
    print(f"📂 Reading Evaluation Data: {json_path}")
    
    if not os.path.exists(json_path):
        print("❌ Evaluation JSON not found!")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df["Lap"] = range(1, len(df) + 1)
    
    print(f"📊 Plotting Eval Metrics: Progress={df['progress'].mean():.2f}%")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Progress
    sns.barplot(data=df, x="Lap", y="progress", ax=axes[0], palette="Blues_d")
    axes[0].set_title("Evaluation Progress %")
    axes[0].set_ylim(0, 105)
    axes[0].axhline(100, color='red', linestyle='--')
    
    # Lap Time
    if df['lap_time'].isnull().all():
        axes[1].text(0.5, 0.5, "No Completed Laps", ha='center', va='center')
    else:
        sns.barplot(data=df, x="Lap", y="lap_time", ax=axes[1], palette="Greens_d")
        
    axes[1].set_title("Lap Times")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"))
    print(f"✅ Saved evaluation plots to {output_dir}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Training
    # Replace this with your ACTUAL successful run folder name
    # You can find it using: ls -lh runs/
    latest_run = get_run_dir() 
    if latest_run:
        df = extract_training_data(latest_run)
        plot_training(df, OUTPUT_DIR)
        
    # 2. Evaluation
    plot_evaluation(EVAL_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()
