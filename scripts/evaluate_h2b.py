import os
import sys
import torch
import json
sys.path.append(os.getcwd())
from src.ppo import PPOAgent
from src.utils import make_environment, demo, evaluate_track

# --- CONFIGURATION ---
MODEL_PATH = "saved_models_part2/model_h2b_track.pt"
TRACK = "reInvent2019_track"  # Smile Track
OUTPUT_DIR = "final_report_h2b"
DEVICE = torch.device("cpu")

def main():
    print(f"🚀 STARTING HEAD-TO-BOT EVALUATION")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Agent
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        exit(1)
        
    env = make_environment("deepracer-v0")
    agent = PPOAgent(env).to(DEVICE)
    
    print(f"   ...Loading weights: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    agent.actor.load_state_dict(state_dict)
    agent.eval()
    env.close()
    
    # 2. Metrics (5 Laps)
    print(f"   ...Collecting Metrics...")
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
    print(f"   ...Rendering Video...")
    try:
        demo(agent, TRACK, video_path)
        print(f"   ✅ Video Saved: {video_path}")
    except Exception as e:
        print(f"   ⚠️ Video Error: {e}")

if __name__ == "__main__":
    main()
