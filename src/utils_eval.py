import os
import json
import yaml
import torch
import random
import shutil
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, FlattenObservation, RecordEpisodeStatistics
from loguru import logger
import glob

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def device():
    return torch.device("cpu")

def make_environment(environment_name="deepracer-v0", **kwargs):
    if "world_name" in kwargs:
        del kwargs["world_name"]
    env = gym.make(environment_name, **kwargs)
    env = FlattenObservation(env)
    env = RecordEpisodeStatistics(env)
    return env

def evaluate_track(agent, world_name, environment_name="deepracer-v0", episodes=5, directory='./evaluations'):
    logger.info(f"📊 Evaluating on {world_name} for {episodes} episodes...")
    
    env = make_environment(environment_name)
    progress_list = []
    lap_times = []
    
    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # Deterministic Action
            if hasattr(agent, 'get_action'):
                try:
                    # Unpack PPO tuple (action, log_prob, entropy)
                    action, _, _ = agent.get_action(obs, train=False)
                except:
                    action = agent.get_action(torch.Tensor(obs)[None, :])
            else:
                action = agent.get_action(obs)

            if not isinstance(action, np.ndarray) and torch.is_tensor(action):
                action = action.cpu().detach().numpy()

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if done:
                prog = info.get('progress', 0.0)
                time_val = info.get('episode', {}).get('t', 0.0)
                progress_list.append(prog)
                lap_times.append(time_val if prog >= 99.0 else None)
                logger.info(f"   Ep {i+1}: Progress={prog:.2f}%")

    env.close()
    
    # Save metrics
    os.makedirs(directory, exist_ok=True)
    metrics = {
        "progress": progress_list,
        "lap_time": lap_times,
        "success_rate": sum(1 for p in progress_list if p >= 99.0) / episodes
    }
    
    # Save to JSON (This line failed before)
    with open(f'{directory}/{world_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

def evaluate(agent, world_name=None, environment_name="deepracer-v0", episodes=5, directory='./evaluations'):
    agent.eval()
    if world_name:
        return evaluate_track(agent, world_name, environment_name, episodes, directory)
    
    logger.warning("Bulk evaluation requested.")
    results = {}
    for track in ['reInvent2019_wide', 'reInvent2019_track', 'Vegas_track']:
        results[track] = evaluate_track(agent, track, environment_name, episodes, directory)
    return results

def demo(agent, world_name, output_path, environment_name="deepracer-v0", directory="."):
    logger.info(f"🎥 Recording Demo for {world_name}...")
    video_folder = os.path.dirname(output_path)
    os.makedirs(video_folder, exist_ok=True)
    
    env = gym.make(environment_name, render_mode='rgb_array')
    env = FlattenObservation(env)
    
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=f"temp_{world_name}",
        episode_trigger=lambda x: True
    )
    
    obs, _ = env.reset()
    done = False
    while not done:
        if hasattr(agent, 'get_action'):
            try:
                action, _, _ = agent.get_action(obs, train=False)
            except:
                action = agent.get_action(obs)
        else:
            action = agent.get_action(obs)
            
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    
    generated_file = os.path.join(video_folder, f"temp_{world_name}-episode-0.mp4")
    if os.path.exists(generated_file):
        shutil.move(generated_file, output_path)
        logger.info(f"✅ Video saved to: {output_path}")
    else:
        files = sorted(glob.glob(f"{video_folder}/*.mp4"))
        if files:
            shutil.move(files[-1], output_path)
            logger.info(f"✅ Video saved (fallback): {output_path}")
