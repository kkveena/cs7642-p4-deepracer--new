import yaml
import time
import torch
import datetime
import numpy as np
from loguru import logger
from munch import munchify
from torch.utils.tensorboard import SummaryWriter

# Import our new PPO agent
from src.ppo import PPOAgent 
from src.utils import device, set_seed, make_environment

DEVICE = device()
HYPER_PARAMS_PATH = 'configs/hyper_params.yaml'

def run(hparams):
    start_time = time.time()
    
    # Load Hyperparams
    with open(HYPER_PARAMS_PATH, 'r') as file:
        default_hparams = yaml.safe_load(file)
    final_hparams = default_hparams.copy()
    final_hparams.update(hparams)
    args = munchify(final_hparams)
    
    # Setup Logging
    run_name = f"{args.environment}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    set_seed(args.seed)
    
    # Create Environment
    env = make_environment(args.environment)
    
    # Initialize PPO Agent
    agent = PPOAgent(env, lr=args.lr)
    
    # --- TRAINING LOOP ---
    observation, info = env.reset()
    episode_return = 0
    episode_len = 0
    total_episodes = 0
    
    logger.info(f"🚀 Starting PPO Training for {args.total_timesteps} steps...")

    for step in range(1, args.total_timesteps + 1):
        
        # 1. Get Action from Agent
        # (Returns action, log_prob, value for PPO updates)
        action, log_prob, _ = agent.get_action(observation)
        
        # 2. Step Environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # 3. Store Transition in PPO Buffer
        agent.store(observation, action, reward, terminated or truncated, log_prob, 0)
        
        observation = next_observation
        episode_return += reward
        episode_len += 1
        
        # 4. Handle End of Episode (Crash or Lap Complete)
        if terminated or truncated:
            total_episodes += 1
            et = str(datetime.timedelta(seconds=round(time.time()-start_time)))
            
            # Log Progress
            logger.info(f"Step={step} | Ep={total_episodes} | Return={episode_return:.2f} | Elapsed={et}")
            writer.add_scalar('charts/episodic_return', episode_return, step)
            writer.add_scalar('charts/episodic_length', episode_len, step)
            
            # CRITICAL FIX: Reset Environment instead of breaking!
            observation, info = env.reset()
            episode_return = 0
            episode_len = 0
            
            # Update PPO Agent (Learn) every few episodes or fixed interval
            # Simple version: Update at end of every episode
            agent.update()

        # 5. Save Model Periodically (Every 1k steps or last step)
        if step % 1000 == 0 or step == args.total_timesteps:
            save_path = f"runs/{run_name}/model_{step}.pt"
            torch.save(agent, save_path)
            logger.info(f"💾 Model saved to {save_path}")

    env.close()
    writer.close()
    logger.info("✅ Training Complete.")