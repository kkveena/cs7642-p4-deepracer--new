import os
import sys
import torch
import json

# Fix path to find src
sys.path.append(os.getcwd())

from src.ppo import PPOAgent
from src.utils import make_environment, demo, evaluate_track

# --- CONFIGURATION ---
MODEL_PATH = "saved_models_part1/model_smile_track.pt"
TRACK = "reInvent2019_track"  # Smile Track
DEVICE = torch.device("cpu")
OUTPUT_DIR = "final_report_smile"

def main():
    print(f"🚀 STARTING EVALUATION FOR: {TRACK}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Agent
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
    
    # 2. Metrics (5 Laps)
    print(f"   ...Collecting Metrics (5 Laps)...")
    metrics = evaluate_track(
        agent=agent,
        world_name=TRACK,
        environment_name="deepracer-v0",
        episodes=5,
        directory=OUTPUT_DIR
    )
    print(f"   📊 Results: {metrics}")
    
    # 3. Video
    video_path = f"{OUTPUT_DIR}/{TRACK}.mp4"
    print(f"   ...Rendering Video to {video_path}...")
    
    try:
        demo(agent, TRACK, video_path)
        print(f"   ✅ Video Saved: {video_path}")
    except Exception as e:
        print(f"   ⚠️ Video generation error: {e}")

if __name__ == "__main__":
    main()
