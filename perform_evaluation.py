import os
import glob
import json
import shutil
from src.utils import evaluate, demo, plot_metrics

# --- CONFIGURATION ---
# The tracks required for Part 1
TRACKS = ["reInvent2019_wide", "reInvent2019_track", "Vegas_track"]
# Directory to store final report artifacts
ARTIFACT_DIR = "final_report_artifacts"

def find_latest_model():
    """Finds the most recently created model in the runs/ directory."""
    # Look for all .pt files inside runs/ subdirectories
    files = glob.glob("runs/**/*.pt", recursive=True)
    if not files:
        raise FileNotFoundError("No model (.pt) files found in runs/!")
    # Sort by modification time (newest first)
    latest_model = max(files, key=os.path.getmtime)
    print(f"✅ Found latest model: {latest_model}")
    return latest_model

def main():
    # 1. Setup Artifact Directory
    if os.path.exists(ARTIFACT_DIR):
        shutil.rmtree(ARTIFACT_DIR)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"📂 Artifacts will be saved to: {ARTIFACT_DIR}")

    # 2. Get the Model
    model_path = find_latest_model()
    
    # 3. Evaluate on ALL Tracks
    all_metrics = {}
    
    for track in TRACKS:
        print(f"\n🏎️  EVALUATING ON TRACK: {track}")
        # Evaluate 5 episodes per track as per requirements
        metrics = evaluate(
            agent_name="TimeTrial_Agent",
            world_name=track,
            model_path=model_path,
            episodes=5
        )
        all_metrics[track] = metrics
        print(f"📊 {track} Results: {metrics}")

        # 4. Generate Video (Demo)
        print(f"🎥 Generating Video for {track}...")
        video_filename = f"{track}_video.mp4"
        output_video_path = os.path.join(ARTIFACT_DIR, video_filename)
        
        # Note: demo() usually saves to 'demos/' folder. We will move it.
        demo(
            world_name=track,
            model_path=model_path,
            output_path=output_video_path
        )
        print(f"✅ Video saved: {output_video_path}")

    # 5. Save Combined Metrics
    metrics_file = os.path.join(ARTIFACT_DIR, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"✅ Metrics saved to {metrics_file}")

    print("\n🎉 Part 1 Automation Complete. Download the folder:", ARTIFACT_DIR)

if __name__ == "__main__":
    main()
