import os
import sys
import glob
import json
import shutil
import time
from src.utils import evaluate, demo

# --- CONFIGURATION ---
# [cite_start]The 3 tracks required for Part 1 [cite: 92-94]
EVAL_TRACKS = ["reInvent2019_wide", "reInvent2019_track", "Vegas_track"]

def find_latest_model():
    """Finds the most recent model file."""
    # Look for .pt files in runs/ directory
    files = glob.glob("runs/**/*.pt", recursive=True)
    if not files:
        print("❌ Error: No model (.pt) files found in runs/. Have you finished training?")
        sys.exit(1)
    return max(files, key=os.path.getmtime)

def main():
    # 1. Identify Model
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = find_latest_model()
    
    model_name = os.path.basename(os.path.dirname(model_path)) # e.g. deepracer-v0__time_trial...
    print(f"🔍 Analyzing Model: {model_path}")

    # 2. Create Unique Output Directory (No Overwrites!)
    # Structure: analysis_results/<model_name>_<timestamp>/
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_output_dir = os.path.join("analysis_results", f"{model_name}_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"📂 Saving results to: {base_output_dir}")

    # 3. Run Evaluation on All Tracks
    results_summary = {}
    
    for track in EVAL_TRACKS:
        print(f"\n🏎️  Processing Track: {track}")
        track_dir = os.path.join(base_output_dir, track)
        os.makedirs(track_dir, exist_ok=True)

        # [cite_start]A. Evaluation (5 episodes) [cite: 84]
        print(f"   ...Running Evaluation (5 episodes)...")
        # Note: evaluate() saves raw logs internally, we capture the metrics return
        metrics = evaluate(
            agent_name="TimeTrial_Agent",
            world_name=track,
            model_path=model_path,
            episodes=5
        )
        results_summary[track] = metrics
        
        # Save track-specific metrics
        with open(os.path.join(track_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # [cite_start]B. Video Generation [cite: 128]
        print(f"   ...Generating Video...")
        video_path = os.path.join(track_dir, f"{track}_demo.mp4")
        try:
            demo(
                world_name=track,
                model_path=model_path,
                output_path=video_path
            )
            print(f"   ✅ Video saved: {video_path}")
        except Exception as e:
            print(f"   ⚠️ Video generation failed: {e}")

    # 4. Save Summary
    summary_path = os.path.join(base_output_dir, "full_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\n🎉 Analysis Complete. Data saved in: {base_output_dir}")

if __name__ == "__main__":
    main()
