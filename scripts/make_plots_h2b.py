import os
import glob
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# CONFIG
RUN_DIR = "runs/deepracer-v0__h2b_track__42__1764424928"
EVAL_DIR = "final_report_h2b"
OUTPUT_DIR = "final_plots_h2b"
TRACK_NAME = "reInvent2019_track"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📊 Generating Plots for Head-to-Bot...")
    
    # 1. TRAINING DATA
    event_files = glob.glob(os.path.join(RUN_DIR, "events.out.tfevents.*"))
    if event_files:
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
        df = pd.DataFrame({"Step": steps[:min_len], "Reward": rewards[:min_len], "Length": lengths[:min_len]})
        df["Efficiency"] = df["Reward"] / df["Length"]
        df["Reward_Smooth"] = df["Reward"].rolling(window=10).mean()
        
        # Plot Training
        sns.set_theme(style="darkgrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.lineplot(data=df, x="Step", y="Reward", alpha=0.3, ax=axes[0], color="blue")
        sns.lineplot(data=df, x="Step", y="Reward_Smooth", ax=axes[0], color="darkblue")
        axes[0].set_title("Reward")
        sns.lineplot(data=df, x="Step", y="Length", ax=axes[1], color="green")
        axes[1].set_title("Episode Length")
        sns.lineplot(data=df, x="Step", y="Efficiency", ax=axes[2], color="orange")
        axes[2].set_title("Efficiency")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/training_metrics.png")
        print(f"   ✅ Training plots saved.")
    else:
        print("   ❌ Training logs not found.")

    # 2. EVALUATION DATA
    json_path = f"{EVAL_DIR}/{TRACK_NAME}_metrics.json"
    if os.path.exists(json_path):
        with open(json_path) as f: data = json.load(f)
        df = pd.DataFrame(data)
        df["Lap"] = range(1, len(df)+1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(data=df, x="Lap", y="progress", ax=axes[0], palette="Blues_d")
        axes[0].set_title("Evaluation Progress %")
        axes[0].axhline(100, color='r', linestyle='--')
        
        completed = df.dropna(subset=['lap_time'])
        if not completed.empty:
            sns.barplot(data=completed, x="Lap", y="lap_time", ax=axes[1], palette="Greens_d")
        else:
            axes[1].text(0.5, 0.5, "No Complete Laps", ha='center')
        axes[1].set_title("Lap Times")
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/evaluation_metrics.png")
        print(f"   ✅ Evaluation plots saved.")
    else:
        print("   ⚠️ Evaluation JSON not found (Run eval first).")

if __name__ == "__main__":
    main()
