import os
import sys
import glob
import json
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
RUNS_DIR = "runs"
EVAL_DIR = "final_report_smile"
OUTPUT_DIR = "final_plots_smile"
TRACK_NAME = "reInvent2019_track"
TEXT_LOG_FILE = "training_smile.log"  # <--- UPDATED FILENAME

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_run_dir():
    if len(sys.argv) > 1:
        return sys.argv[1]
    run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "*")))
    return run_dirs[-1] if run_dirs else None

def parse_text_log(logfile):
    print(f"⚠️ TensorBoard failed. Parsing text log: {logfile}...")
    if not os.path.exists(logfile):
        print(f"❌ Log file {logfile} not found.")
        return None
        
    data = {"Step": [], "Reward": [], "Length": []}
    
    with open(logfile, 'r') as f:
        for line in f:
            if "Step=" in line and "Return=" in line:
                try:
                    # Line format: ... | src.run:run:51 - Step=14607 | Ep=330 | Return=12.01 | ...
                    parts = line.split("|")
                    step_part = [p for p in parts if "Step=" in p][0]
                    step = int(step_part.split("=")[1].strip())
                    
                    ret_part = [p for p in parts if "Return=" in p][0]
                    ret = float(ret_part.split("=")[1].strip())
                    
                    # Some logs might not have Length, but usually they do
                    length = 1
                    len_parts = [p for p in parts if "episodic_length" in p or "Length" in p]
                    # If not found, check "Ep=..." logic or just default
                    
                    data["Step"].append(step)
                    data["Reward"].append(ret)
                    data["Length"].append(length) 
                except:
                    continue
                    
    if not data["Step"]:
        print("❌ Text log was empty or format didn't match.")
        return None
        
    df = pd.DataFrame(data)
    # If Length is missing/1, Efficiency is just Reward
    df["Efficiency"] = df["Reward"]
    df["Reward_Smooth"] = df["Reward"].rolling(window=10).mean()
    return df

def extract_training_data(run_dir):
    print(f"📂 Reading logs from: {run_dir}")
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    
    # If TensorBoard file is missing or corrupt, fallback immediately
    if not event_files:
        return parse_text_log(TEXT_LOG_FILE)

    try:
        ea = EventAccumulator(event_files[0])
        ea.Reload()
        
        steps, rewards = [], []
        if 'charts/episodic_return' in ea.scalars.Keys():
            for e in ea.Scalars('charts/episodic_return'):
                steps.append(e.step)
                rewards.append(e.value)

        if not steps:
            return parse_text_log(TEXT_LOG_FILE)

        df = pd.DataFrame({"Step": steps, "Reward": rewards})
        df["Length"] = 1 # Placeholder if TB missing length
        df["Efficiency"] = df["Reward"]
        df["Reward_Smooth"] = df["Reward"].rolling(window=10).mean()
        return df
        
    except Exception as e:
        return parse_text_log(TEXT_LOG_FILE)

def plot_training(df, output_dir):
    if df is None or df.empty: return
    print(f"📊 Generating Training Plots ({len(df)} episodes)...")
    sns.set_theme(style="darkgrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.lineplot(data=df, x="Step", y="Reward", alpha=0.3, ax=axes[0], color="blue")
    sns.lineplot(data=df, x="Step", y="Reward_Smooth", ax=axes[0], color="darkblue")
    axes[0].set_title("Metric 1: Episodic Reward")
    
    # Since we might lack Length data from text logs, we label carefully
    sns.lineplot(data=df, x="Step", y="Reward", ax=axes[1], color="green")
    axes[1].set_title("Metric 2: Training Stability")
    axes[1].set_ylabel("Reward")
    
    sns.lineplot(data=df, x="Step", y="Efficiency", ax=axes[2], color="orange")
    axes[2].set_title("Metric 3: Efficiency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    print(f"✅ Saved training plots to {output_dir}")

def plot_evaluation(eval_dir, output_dir):
    json_path = os.path.join(eval_dir, f"{TRACK_NAME}_metrics.json")
    if not os.path.exists(json_path): return

    print(f"📂 Reading Evaluation: {json_path}")
    with open(json_path, 'r') as f: data = json.load(f)
    df = pd.DataFrame(data)
    df["Lap"] = range(1, len(df) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=df, x="Lap", y="progress", ax=axes[0], palette="Blues_d", hue="Lap", legend=False)
    axes[0].set_title("Evaluation Progress %")
    axes[0].axhline(100, color='r', linestyle='--')
    
    # Handle missing lap times
    if df['lap_time'].isnull().all():
        axes[1].text(0.5, 0.5, "No Completed Laps", ha='center')
    else:
        sns.barplot(data=df, x="Lap", y="lap_time", ax=axes[1], palette="Greens_d", hue="Lap", legend=False)
    axes[1].set_title("Lap Times")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"))
    print(f"✅ Saved evaluation plots to {output_dir}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Training
    run_dir = get_run_dir()
    if run_dir:
        df = extract_training_data(run_dir)
        plot_training(df, OUTPUT_DIR)
        
    # 2. Evaluation
    plot_evaluation(EVAL_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()
