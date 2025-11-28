import os
import glob
import json
import torch
from src.ppo import PPOAgent
from src.utils import make_environment, demo, evaluate

# --- CONFIGURATION ---
TRACKS = ["reInvent2019_wide", "reInvent2019_track", "Vegas_track"]
DEVICE = torch.device("cpu") # Use CPU for evaluation to be safe

def load_trained_agent(model_path, track_name):
    """
    Loads the agent from a state_dict checkpoint.
    We must re-initialize the PPOAgent structure first.
    """
    # 1. Create a dummy environment to get dimensions
    env = make_environment(track_name)
    
    # 2. Initialize a fresh agent
    # Note: hyperparameters don't matter for inference, just dimensions
    agent = PPOAgent(env).to(DEVICE)
    
    # 3. Load the weights
    print(f"   ...Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=DEVICE)
    agent.actor.load_state_dict(state_dict)
    
    # 4. Set to Eval mode
    agent.eval()
    return agent

def find_latest_model():
    files = glob.glob("runs/**/*.pt", recursive=True)
    if not files:
        raise FileNotFoundError("❌ No .pt model found! Training might have failed.")
    return max(files, key=os.path.getmtime)

def main():
    # Find model
    model_path = find_latest_model()
    print(f"🚀 FOUND MODEL: {model_path}")
    
    # Output setup
    os.makedirs("final_report_output", exist_ok=True)
    results = {}

    for track in TRACKS:
        print(f"\n🎥 PROCESSING TRACK: {track}")
        
        # Load agent (Fresh for each track to ensure clean env)
        agent = load_trained_agent(model_path, track)
        
        # 1. Evaluate Metrics (5 Laps)
        print(f"   ...Running 5 evaluation laps...")
        metrics = evaluate(
            agent=agent,
            world_name=track,
            environment_name="deepracer-v0",
            directory="final_report_output"
        )
        results[track] = metrics
        print(f"   📊 Results: {metrics}")
        
        # 2. Generate Video
        video_path = f"final_report_output/{track}.mp4"
        print(f"   ...Rendering Video to {video_path}...")
        try:
            # We need to manually run the demo logic because src.utils.demo might assume old agent structure
            # But let's try the standard utility first since our agent inherits from src.Agent
            demo(agent=agent, world_name=track, directory="final_report_output")
            print("   ✅ Video Generation command sent.")
        except Exception as e:
            print(f"   ⚠️ Video generation warning: {e}")

    # Save Final Summary
    with open("final_report_output/summary.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n✅ ALL DONE. Download the 'final_report_output' folder.")

if __name__ == "__main__":
    main()