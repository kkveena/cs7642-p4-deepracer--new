import yaml
import time
import torch
import datetime
import numpy as np
from loguru import logger
from munch import munchify
from torch.utils.tensorboard import SummaryWriter

from src.ppo import PPOAgent
from src.utils import device, set_seed, make_environment

DEVICE = device()
HYPER_PARAMS_PATH = 'configs/hyper_params.yaml'

def run(hparams):
    start_time = time.time()
    
    with open(HYPER_PARAMS_PATH, 'r') as file:
        default_hparams = yaml.safe_load(file)
    final_hparams = default_hparams.copy()
    final_hparams.update(hparams)
    args = munchify(final_hparams)
    
    run_name = f"{args.environment}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    set_seed(args.seed)
    env = make_environment(args.environment)
    agent = PPOAgent(env, lr=args.lr)
    
    observation, info = env.reset()
    episode_return = 0
    episode_len = 0
    total_episodes = 0
    
    logger.info(f"🚀 Starting PPO Training for {args.total_timesteps} steps...")

    for step in range(1, args.total_timesteps + 1):
        action, log_prob, _ = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        agent.store(observation, action, reward, terminated or truncated, log_prob, 0)
        
        observation = next_observation
        episode_return += reward
        episode_len += 1
        
        if terminated or truncated:
            total_episodes += 1
            et = str(datetime.timedelta(seconds=round(time.time()-start_time)))
            logger.info(f"Step={step} | Ep={total_episodes} | Return={episode_return:.2f} | Elapsed={et}")
            writer.add_scalar('charts/episodic_return', episode_return, step)
            
            observation, info = env.reset()
            episode_return = 0
            episode_len = 0
            agent.update()

        # --- FIX: SAVE STATE_DICT ONLY ---
        if step % 5000 == 0 or step == args.total_timesteps:
            save_path = f"runs/{run_name}/model_{step}.pt"
            # We save the actor network state, which is all we need for evaluation
            torch.save(agent.actor.state_dict(), save_path)
            logger.info(f"💾 Model weights saved to {save_path}")

    env.close()
    writer.close()
    logger.info("✅ Training Complete.")