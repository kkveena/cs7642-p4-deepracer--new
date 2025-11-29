import os
import sys
import torch
import json
sys.path.append(os.getcwd())
from src.ppo import PPOAgent
from src.utils import make_environment, demo, evaluate_track

# CONFIGURATION
MODEL_PATH = "saved_models_part1/model_smile_track.pt"
TRACK = "reInvent2019_track"
OUTPUT_DIR = "final_report_oa_baseline"
DEVICE = torch.device("cpu")
"""
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
"""

def main():
    print(f"🚀 RUNNING BASELINE: Time Trial Model on Obstacle Track")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Agent (Time Trial Brain)
    env = make_environment("deepracer-v0")
    agent = PPOAgent(env).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        exit(1)
        
    print(f"   ...Loading weights: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    agent.actor.load_state_dict(state_dict)
    agent.eval()
    env.close()
    
    # 2. Metrics (Expect poor performance/crashes)
    print(f"   ...Collecting Baseline Metrics...")
    metrics = evaluate_track(
        agent=agent, 
        world_name=TRACK, 
        environment_name="deepracer-v0", 
        episodes=5,
        directory=OUTPUT_DIR
    )
    print(f"   📊 Baseline Results: {metrics}")
    
    # 3. Video (Will show the car hitting obstacles)
    video_path = f"{OUTPUT_DIR}/baseline_obstacle.mp4"
    print(f"   ...Rendering Baseline Video...")
    try:
        demo(agent, TRACK, video_path)
        print(f"   ✅ Video Saved: {video_path}")
    except Exception as e:
        print(f"   ⚠️ Video Error: {e}")

if __name__ == "__main__":
    main()
