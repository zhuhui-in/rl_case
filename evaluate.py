"""Greedy rollout of the trained DQN; export trajectory.json for the web viewer."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from dqn_agent import DQNAgent, DQNConfig
from magic_tower_env import CONFIG, MagicTowerEnv

CKPT_PATH = Path("checkpoints/best.pt")
TRAJ_PATH = Path("viz/trajectory.json")

ACTION_NAMES = {0: "FIGHT", 1: "BUY_DEF", 2: "BUY_HEAL"}


def main() -> None:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}. Run `python train.py` first.")

    env = MagicTowerEnv()
    cfg = DQNConfig()
    agent = DQNAgent(cfg)
    agent.load_state_dict(torch.load(CKPT_PATH, map_location=cfg.device))
    agent.online.eval()

    obs, _ = env.reset()
    steps: list[dict] = []
    initial_state = dict(env.raw_state)

    done = False
    total_reward = 0.0
    win = False

    print("Step | Action     | HP  d  $  killed | reward")
    print("-" * 60)
    print(f"  0  | {'INIT':<10} | {initial_state['hp']:3d} {initial_state['d']:2d} {initial_state['money']:2d} {initial_state['killed']:6d} | -")

    step_idx = 0
    while not done:
        action = agent.act(obs, eps=0.0)
        before_state = dict(env.raw_state)
        next_obs, reward, terminated, truncated, info = env.step(action)
        after_state = {k: int(v) for k, v in env.raw_state.items()}
        step_idx += 1
        steps.append(
            {
                "step": step_idx,
                "action": int(action),
                "action_name": ACTION_NAMES[int(action)],
                "reward": float(reward),
                "before": {k: int(v) for k, v in before_state.items()},
                "after": after_state,
            }
        )
        obs = next_obs
        total_reward += reward
        done = terminated or truncated
        if info.get("win"):
            win = True
        print(
            f"{step_idx:3d}  | {ACTION_NAMES[int(action)]:<10} | "
            f"{after_state['hp']:3d} {after_state['d']:2d} {after_state['money']:2d} {after_state['killed']:6d} | "
            f"{reward:+.2f}"
        )

    print("-" * 60)
    print(f"Done. win={win}  total_reward={total_reward:.2f}  steps={step_idx}")
    if not win:
        raise RuntimeError("Greedy policy did not win; cannot export trajectory.")

    TRAJ_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": CONFIG,
        "initial": {k: int(v) for k, v in initial_state.items()},
        "steps": steps,
        "total_reward": total_reward,
        "win": win,
    }
    with open(TRAJ_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Trajectory written to {TRAJ_PATH}.")


if __name__ == "__main__":
    main()
