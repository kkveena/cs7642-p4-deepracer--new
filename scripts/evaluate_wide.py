import os
import sys
import torch
import json

# Add root directory to path so we can import src
sys.path.append(os.getcwd())

from src.ppo import PPOAgent
from src.utils import make_environment, demo, evaluate

# Configuration
MODEL_PATH = "saved_models_part1/model_wide_track.pt"
TRACK = "reInvent2019_wide" 
DEVICE = torch.device("cpu")

def main():
    print(f"🚀 STARTING EVALUATION FOR: {TRACK}")
    
    # 1. Setup
    env = make_environment("deepracer-v0")
    agent = PPOAgent(env).to(DEVICE)
    
    # 2. Load Weights
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        exit(1)
        
    print(f"   ...Loading weights from {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    agent.actor.load_state_dict(state_dict)
    agent.eval()
    
    # 3. Metrics
    print(f"   ...Running 5 evaluation laps...")
    metrics = evaluate(agent=agent, world_name=TRACK, episodes=5)
    print(f"   📊 Results: {metrics}")
    
    # 4. Video
    os.makedirs("final_videos", exist_ok=True)
    video_path = f"final_videos/{TRACK}.mp4"
    print(f"   ...Rendering Video to {video_path}...")
    
    try:
        demo(agent=agent, world_name=TRACK, output_path=video_path)
        print(f"✅ Video Saved: {video_path}")
    except Exception as e:
        print(f"   ⚠️ Video generation error: {e}")

if __name__ == "__main__":
    main()