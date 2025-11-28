import os
import sys
import torch
import json

# Fix path to find src
sys.path.append(os.getcwd())

from src.ppo import PPOAgent
# IMPORT evaluate_track SPECIFICALLY
from src.utils import make_environment, demo, evaluate_track 

# Configuration
# Update this to your ACTUAL trained model path
MODEL_PATH = "saved_models_part1/model_wide_track.pt" 
TRACK = "reInvent2019_wide"
DEVICE = torch.device("cpu")
OUTPUT_DIR = "final_report_part1"

def main():
    print(f"🚀 STARTING EVALUATION FOR: {TRACK}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Agent
    # Init dummy env for dims
    env = make_environment("deepracer-v0")
    agent = PPOAgent(env).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        exit(1)
        
    print(f"   ...Loading weights from {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    agent.actor.load_state_dict(state_dict)
    agent.eval()
    env.close()
    
    # 2. Metrics (Using evaluate_track to avoid TypeError)
    print(f"   ...Collecting Metrics (5 Laps)...")
    metrics = evaluate_track(
        agent=agent,
        world_name=TRACK,
        environment_name="deepracer-v0",
        episodes=5
    )
    print(f"   📊 Results: {metrics}")
    
    # Save Metrics
    with open(f"{OUTPUT_DIR}/{TRACK}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 3. Video
    video_path = f"{OUTPUT_DIR}/{TRACK}.mp4"
    try:
        demo(agent, TRACK, video_path)
        print(f"   ✅ Video Saved: {video_path}")
    except Exception as e:
        print(f"   ⚠️ Video failed: {e}")

if __name__ == "__main__":
    main()
