import os
import torch
import json
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, FlattenObservation, RecordEpisodeStatistics
from loguru import logger
import shutil
import glob

# Import your agent and standard tools
from src.ppo import PPOAgent
from src.utils import make_environment, evaluate, get_race_type

# --- CONFIGURATION ---
TRACK_NAME = "reInvent2019_wide"
MODEL_PATH = "saved_models_part1/model_wide_track.pt"
OUTPUT_DIR = "final_report_part1"
DEVICE = torch.device("cpu")

# --- CUSTOM VIDEO FUNCTION (Bypasses src/utils.py limitations) ---
def demo_with_video(agent, world_name, output_path):
    """
    A standalone video generator that guarantees RecordVideo is active,
    regardless of what is commented out in src/utils.py.
    """
    logger.info(f"🎥 Recording Demo for {world_name}...")
    video_folder = os.path.dirname(output_path)
    os.makedirs(video_folder, exist_ok=True)
    
    # 1. Force 'rgb_array' mode (Required for video)
    env = gym.make("deepracer-v0", render_mode='rgb_array')
    env = FlattenObservation(env)
    
    # 2. Explicitly Apply RecordVideo (Restoring the lines you commented out)
    # We use a temp prefix to avoid naming collisions
    temp_prefix = f"temp_{world_name}"
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=temp_prefix,
        episode_trigger=lambda x: True
    )
    
    # 3. Run One Episode
    obs, _ = env.reset()
    done = False
    while not done:
        # Use deterministic action for best video
        if hasattr(agent, 'get_action'):
            action, _, _ = agent.get_action(obs, train=False)
        else:
            action = agent.get_action(obs)
            
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    
    # 4. Handle File Renaming (Gym saves as prefix-episode-0.mp4)
    generated_file = os.path.join(video_folder, f"{temp_prefix}-episode-0.mp4")
    if os.path.exists(generated_file):
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.move(generated_file, output_path)
        logger.info(f"✅ Video saved to: {output_path}")
    else:
        logger.error(f"❌ Video file generation failed. Expected: {generated_file}")

def main():
    print(f"🚀 STARTING PART 1 EVALUATION: {TRACK_NAME}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Agent
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        exit(1)
        
    # Initialize env just to get dimensions for the agent
    setup_env = make_environment("deepracer-v0")
    agent = PPOAgent(setup_env).to(DEVICE)
    
    print(f"   ...Loading weights from {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    agent.actor.load_state_dict(state_dict)
    agent.eval()
    setup_env.close()
    
    # 2. Run Metrics Evaluation (Uses standard src/utils.py)
    # We temporarily set the config file to match the track for the Simulator's benefit
    # (Ideally done via shell script, but this is a safe fallback)
    print(f"   ...Collecting Metrics (5 Laps)...")
    metrics = evaluate(
        agent=agent,
        world_name=TRACK_NAME,
        environment_name="deepracer-v0",
        episodes=5
    )
    print(f"   📊 Results: {metrics}")
    
    # Save Metrics
    with open(f"{OUTPUT_DIR}/{TRACK_NAME}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 3. Run Video Generation (Uses our CUSTOM function)
    video_path = f"{OUTPUT_DIR}/{TRACK_NAME}.mp4"
    try:
        demo_with_video(agent, TRACK_NAME, video_path)
    except Exception as e:
        print(f"   ⚠️ Video failed: {e}")

    print(f"\n✅ Part 1 Analysis Complete. Check '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()
