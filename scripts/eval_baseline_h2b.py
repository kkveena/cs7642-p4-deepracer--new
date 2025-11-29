import os
import sys
import torch
import json
sys.path.append(os.getcwd())
from src.ppo import PPOAgent
from src.utils import make_environment, demo, evaluate_track

MODEL_PATH = "saved_models_part1/model_wide_track.pt"  # Time Trial Model
TRACK = "reInvent2019_track"
OUTPUT_DIR = "final_report_h2b_baseline"
DEVICE = torch.device("cpu")

def main():
    print(f"🚀 RUNNING H2B BASELINE")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = make_environment("deepracer-v0")
    agent = PPOAgent(env).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    agent.actor.load_state_dict(state_dict)
    agent.eval()
    env.close()
    
    metrics = evaluate_track(agent, TRACK, "deepracer-v0", 5, OUTPUT_DIR)
    print(f"   📊 Results: {metrics}")
    
    try:
        demo(agent, TRACK, f"{OUTPUT_DIR}/baseline_h2b.mp4")
        print(f"   ✅ Video Saved")
    except Exception as e:
        print(f"   ⚠️ Video Error: {e}")

if __name__ == "__main__":
    main()
