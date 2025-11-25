# src/cli.py
import argparse
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

from src.run import run
from src.utils import evaluate, demo

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs"

@contextmanager
def _swap_file(active_path: Path, desired_path: Path):
    """
    Temporarily swap a config file so the simulator/tools see the desired variant
    without permanently overwriting your baseline.
    """
    active_path = active_path.resolve()
    desired_path = desired_path.resolve()
    backup_path = active_path.with_suffix(active_path.suffix + ".bak")

    if desired_path == active_path:
        yield
        return

    if not desired_path.exists():
        raise FileNotFoundError(f"Config not found: {desired_path}")

    if active_path.exists():
        shutil.copy2(active_path, backup_path)
    try:
        shutil.copy2(desired_path, active_path)
        yield
    finally:
        if backup_path.exists():
            shutil.move(backup_path, active_path)

def _resolve_env_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / p).resolve()

def _resolve_agent_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / p).resolve()

def _resolve_reward_variant(reward: str) -> Path:
    short = reward.lower()
    if short in {"baseline", "base", "default"}:
        return CFG_DIR / "reward_function.py"
    if short in {"stage2", "s2"}:
        return CFG_DIR / "reward_function.stage2.py"
    if short in {"stage3", "s3"}:
        return CFG_DIR / "reward_function.stage3.py"
    q = Path(reward)
    return q if q.is_absolute() else (ROOT / reward).resolve()

def _resolve_hparams_variant(hp: str) -> Path:
    short = hp.lower()
    if short in {"baseline", "base", "default"}:
        return CFG_DIR / "hyper_params.yaml"
    if short in {"stage2", "s2"}:
        return CFG_DIR / "hyper_params.stage2.yaml"
    if short in {"stage3", "s3"}:
        return CFG_DIR / "hyper_params.stage3.yaml"
    q = Path(hp)
    return q if q.is_absolute() else (ROOT / hp).resolve()

def main():
    ap = argparse.ArgumentParser("DeepRacer Part I CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- train ---
    tr = sub.add_parser("train", help="Train a Time-Trial agent")
    tr.add_argument("--steps", type=int, default=300000)
    tr.add_argument("--lr", type=float, default=None, help="Override learning rate (optional)")
    tr.add_argument("--seed", type=int, default=None, help="Override seed (optional)")
    tr.add_argument("--experiment-name", default=None, help="Override experiment name (optional)")

    tr.add_argument("--env-config", default="configs/environment_params.yaml")
    tr.add_argument("--agent-config", default="configs/agent_params.json")
    tr.add_argument("--reward", default="baseline",
                    help="baseline|stage2|stage3 or path to reward_function*.py")
    tr.add_argument("--hparams-config", default="baseline",
                    help="baseline|stage2|stage3 or path to YAML (e.g., configs/hyper_params.stage2.yaml)")
    tr.add_argument("--logdir", default=None)

    # --- eval ---
    ev = sub.add_parser("eval", help="Evaluate a trained model across tracks")
    ev.add_argument("--model", default="models/ppo_time_trial.pt")
    ev.add_argument("--worlds", nargs="+",
                    default=["reInvent2019_wide", "reInvent2019_track", "Vegas_track"])
    ev.add_argument("--episodes", type=int, default=5)

    # --- demo ---
    dm = sub.add_parser("demo", help="Render a demo video for a single track")
    dm.add_argument("--model", default="models/ppo_time_trial.pt")
    dm.add_argument("--world", default="reInvent2019_wide")
    dm.add_argument("--out", default="videos/time_trial.mp4")

    args = ap.parse_args()

    if args.cmd == "train":
        env_cfg_path    = _resolve_env_path(args.env_config)
        agent_cfg_path  = _resolve_agent_path(args.agent_config)
        reward_src      = _resolve_reward_variant(args.reward)
        hparams_path    = _resolve_hparams_variant(args.hparams_config)

        # Active files consumed by env/sim and run()
        active_env     = CFG_DIR / "environment_params.yaml"
        active_agent   = CFG_DIR / "agent_params.json"
        active_reward  = CFG_DIR / "reward_function.py"
        active_hparams = CFG_DIR / "hyper_params.yaml"

        with _swap_file(active_env, env_cfg_path), \
             _swap_file(active_agent, agent_cfg_path), \
             _swap_file(active_reward, reward_src), \
             _swap_file(active_hparams, hparams_path):

            # Build overrides (CLI flags can override YAML)
            overrides = {}
            if args.lr is not None:   overrides["lr"] = float(args.lr)
            if args.seed is not None: overrides["seed"] = int(args.seed)
            if args.experiment_name:  overrides["experiment_name"] = str(args.experiment_name)
            if args.steps is not None:overrides["total_timesteps"] = int(args.steps)
            if args.logdir:           overrides["logdir"] = args.logdir

            run(overrides)

    elif args.cmd == "eval":
        for w in args.worlds:
            metrics = evaluate(world_name=w, model_path=args.model, episodes=args.episodes)
            print(f"[EVAL] {w}: {metrics}")

    elif args.cmd == "demo":
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        demo(world_name=args.world, model_path=args.model, output_path=args.out)

if __name__ == "__main__":
    sys.exit(main())
