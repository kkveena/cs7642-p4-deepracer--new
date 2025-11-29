import os
import sys
# Re-use the logic from your main plotting script
from make_plots import extract_training_data, plot_training, plot_evaluation_metrics

RUN_DIR_HINT = "vegas_track" # Will look for this in runs/
EVAL_DIR = "final_report_vegas"
OUTPUT_DIR = "final_plots_vegas"
TRACK_NAME = "Vegas_track"
TEXT_LOG = "training_vegas.log" # Fallback

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📊 Processing Vegas Plots...")
    
    # 1. Training Plots (Try Text Log first as it's most reliable for you)
    from make_plots import parse_text_log
    df = parse_text_log(TEXT_LOG)
    if df is not None:
        plot_training(df, OUTPUT_DIR)
    else:
        print("⚠️ Could not parse training log.")

    # 2. Eval Plots
    # Temporarily hijack the global TRACK_NAME in make_plots is hard, 
    # so we assume the JSON filename matches the track name inside EVAL_DIR
    # We manually call the plotting logic here to be safe:
    
    import json
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    json_path = os.path.join(EVAL_DIR, f"{TRACK_NAME}_metrics.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f: data = json.load(f)
        df = pd.DataFrame(data)
        df["Lap"] = range(1, len(df) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(data=df, x="Lap", y="progress", ax=axes[0], palette="Blues_d")
        axes[0].set_title(f"Evaluation: Progress % ({TRACK_NAME})")
        axes[0].set_ylim(0, 110)
        axes[0].axhline(100, color='r', linestyle='--')
        
        completed = df.dropna(subset=['lap_time'])
        if not completed.empty:
            sns.barplot(data=completed, x="Lap", y="lap_time", ax=axes[1], palette="Greens_d")
            axes[1].set_title(f"Lap Time ({TRACK_NAME})")
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_metrics.png"))
        print(f"✅ Saved evaluation plots to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
