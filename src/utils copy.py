import os
import json
import yaml
import torch
import random
import shutil
import enlighten
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import gymnasium as gym
from loguru import logger
from gymnasium import spaces
import matplotlib.pyplot as plt
from gymnasium.wrappers import (
    RecordVideo,
    FlattenObservation,
    RecordEpisodeStatistics
)
from IPython.display import Video, display, clear_output
from src.agents import Agent

PROGRESS_MANAGER = enlighten.get_manager()
FS_TICK: int = 12
FS_LABEL: int = 18
PLOT_DPI: int=1200
PLOT_FORMAT: str='pdf'
RC_PARAMS: dict = {
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'xtick.color': 'black',
    'ytick.color': 'black',
}
ENVIRONMENT_PARAMS_PATH: str='configs/environment_params.yaml'
ENVIRONMENT_NAME: str='deepracer-v0'
MAX_DEMO_STEPS: int = 1_000
MAX_EVAL_STEPS: int = 1_000
EVAL_EPISODES: int = 5
ONLY_CPU: bool = True  # Forced to True for stability
SEED: int=42

def set_seed(seed: int=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Force CPU seeds only
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f'Random seed set as {seed}.')

def device():
    # Force CPU to match training environment and avoid sync crashes
    logger.info('Using cpu device (Forced for stability).')
    return torch.device('cpu')

def make_environment(environment_name: str=ENVIRONMENT_NAME, seed: int=SEED, **kwargs):
    # Clean kwargs to prevent TypeError in gym.make
    # DeepRacer env doesn't accept 'world_name' in constructor
    if 'world_name' in kwargs:
        del kwargs['world_name']
        
    environment = gym.make(environment_name, **kwargs)
    environment = FlattenObservation(environment)
    environment = RecordEpisodeStatistics(environment)
    
    # Seeding
    environment.action_space.seed(seed)
    environment.observation_space.seed(seed)
    return environment

def get_world_name(environment_params_path: str=ENVIRONMENT_PARAMS_PATH):
    with open(environment_params_path, 'r') as f:
        environment_params = yaml.safe_load(f)
    if 'WORLD_NAME' not in environment_params:
        raise ValueError(f'WORLD_NAME not defined in {environment_params_path}')
    return environment_params['WORLD_NAME']

def get_race_type(environment_params_path: str=ENVIRONMENT_PARAMS_PATH):
    with open(environment_params_path, 'r') as f:
        environment_params = yaml.safe_load(f)
    obstacles = int(environment_params.get('NUMBER_OF_OBSTACLES', 0))
    bots = int(environment_params.get('NUMBER_OF_BOT_CARS', 0))
    
    if obstacles == 0 and bots == 0:
        return 'time_trial'
    elif obstacles == 6 and bots == 0:
        return 'obstacle_avoidance'
    elif obstacles == 0 and bots == 3:
        return 'head_to_bot'
    else:
        # Fallback or custom
        return 'custom_race'

def demo(agent: Agent, environment_name: str=ENVIRONMENT_NAME, directory: str='./demos'):
    race_type = get_race_type(environment_params_path=ENVIRONMENT_PARAMS_PATH)
    world_name = get_world_name(environment_params_path=ENVIRONMENT_PARAMS_PATH)
    
    # Force CPU for evaluation
    demo_device = torch.device('cpu')
    # Check if agent is a PPOAgent (has .actor) or generic Agent
    if hasattr(agent, 'actor'):
        agent.actor.to(demo_device)
    
    agent.eval() # Set eval mode
    os.makedirs(directory, exist_ok=True)

    # Initialize Env with RGB Array mode for video
    # Note: We do NOT pass world_name here to avoid the TypeError
    demo_environment = gym.make(environment_name, render_mode='rgb_array')
    demo_environment = FlattenObservation(demo_environment)

    # Apply video recording wrapper
    demo_environment = RecordVideo(
        demo_environment,
        video_folder=directory,
        episode_trigger=lambda x: True,
        name_prefix=f'{world_name}-{race_type}-agent'
    )

    observation, _ = demo_environment.reset()
    
    demo_progress = PROGRESS_MANAGER.counter(
        total=MAX_DEMO_STEPS, desc=f'{world_name} {race_type} demo', unit='steps', leave=False
    )
    
    for t in range(MAX_DEMO_STEPS):
        # Get action (train=False for deterministic)
        # Handle different agent types (your custom PPO vs provided Agent)
        if hasattr(agent, 'get_action'):
             # Check signature of get_action
             try:
                 action, _, _ = agent.get_action(observation, train=False)
             except:
                 action = agent.get_action(torch.Tensor(observation)[None, :])

        if not isinstance(action, np.ndarray) and torch.is_tensor(action):
            action = action.cpu().detach().numpy()
        
        if isinstance(demo_environment.action_space, spaces.Discrete):
            action = action.item()
        
        # Execute
        observation, _, terminated, truncated, _ = demo_environment.step(action)
        demo_progress.update()
        
        if terminated or truncated:
            break
    
    demo_environment.close()
    demo_progress.close()
    
    # Rename/Move file
    # RecordVideo creates "prefix-episode-0.mp4"
    source_file = os.path.join(directory, f"{world_name}-{race_type}-agent-episode-0.mp4")
    final_file = os.path.join(directory, f"{world_name}.mp4")
    
    if os.path.exists(source_file):
        shutil.move(source_file, final_file)
        logger.info(f"✅ Video saved to: {final_file}")
    else:
        # Fallback: check for any mp4 if naming failed
        files = sorted(glob.glob(f"{directory}/*.mp4"))
        if files:
            logger.warning(f"Renaming latest video {files[-1]} to {final_file}")
            shutil.move(files[-1], final_file)

def command_exists(command: str) -> bool:
    return shutil.which(command) is not None

def run_command(command):    
    result=subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        logger.error(result.stderr)
    else:
        logger.info(result.stdout)

def evaluate(agent, world_name, environment_name=ENVIRONMENT_NAME, episodes=EVAL_EPISODES, directory='./evaluations'):
    """
    Evaluates agent performance (Metrics only, no video).
    """
    logger.info(f"Starting evaluation on {world_name}...")
    
    # Force CPU
    agent.eval()
    
    env = make_environment(environment_name)
    progress_list = []
    lap_times = []
    
    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # PPO Agent get_action signature
            try:
                action, _, _ = agent.get_action(obs, train=False)
            except:
                # Fallback for generic agent
                action = agent.get_action(torch.Tensor(obs)[None, :])
                if torch.is_tensor(action): action = action.cpu().detach().numpy()

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if done:
                prog = info.get('progress', 0.0)
                time_val = info.get('episode', {}).get('t', 0.0)
                progress_list.append(prog)
                lap_times.append(time_val if prog >= 99.0 else None)
                logger.info(f"   Ep {i+1}: Progress={prog:.2f}%")

    env.close()
    
    success_rate = sum(1 for p in progress_list if p >= 99.0) / episodes
    return {
        "progress": progress_list,
        "lap_time": lap_times,
        "success_rate": success_rate
    }

def lap_time(info):
    if info['reward_params']['progress'] >= 100:
        if isinstance(info['episode']['t'], np.ndarray):
            return info['episode']['t'].mean()
        else:
            return info['episode']['t']
    else:
        return np.nan

def plot_metrics(data, title, directory='./plots'):
    # (Keep your existing plot code here, it is fine)
    pass