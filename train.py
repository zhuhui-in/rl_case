"""Train a DQN agent on MagicTowerEnv."""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch

from dqn_agent import DQNAgent, DQNConfig
from magic_tower_env import MagicTowerEnv

CKPT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
CKPT_PATH = CKPT_DIR / "best.pt"
METRICS_PATH = LOG_DIR / "metrics.csv"

EVAL_EVERY = 50
EVAL_EPISODES = 5
EARLY_STOP_PATIENCE = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(agent: DQNAgent, env: MagicTowerEnv, n_episodes: int) -> tuple[float, float]:
    total_reward = 0.0
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        win = False
        while not done:
            a = agent.act(obs, eps=0.0)
            obs, r, terminated, truncated, info = env.step(a)
            ep_r += r
            done = terminated or truncated
            if info.get("win"):
                win = True
        total_reward += ep_r
        wins += int(win)
    return total_reward / n_episodes, wins / n_episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = MagicTowerEnv()
    eval_env = MagicTowerEnv()
    cfg = DQNConfig()
    agent = DQNAgent(cfg)

    f = open(METRICS_PATH, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["episode", "train_reward", "avg_loss", "eval_avg_reward", "eval_win_rate", "epsilon"])

    global_step = 0
    consecutive_full_wins = 0
    best_saved = False

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        losses: list[float] = []
        while not done:
            eps = agent.epsilon(global_step)
            a = agent.act(obs, eps)
            next_obs, r, terminated, truncated, _ = env.step(a)
            agent.push(obs, a, r, next_obs, float(terminated))
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            ep_reward += r
            done = terminated or truncated
            global_step += 1

        avg_loss = float(np.mean(losses)) if losses else float("nan")

        if ep % EVAL_EVERY == 0:
            avg_r, win_rate = evaluate(agent, eval_env, EVAL_EPISODES)
            writer.writerow([ep, f"{ep_reward:.3f}", f"{avg_loss:.5f}", f"{avg_r:.3f}", f"{win_rate:.3f}", f"{agent.epsilon(global_step):.3f}"])
            f.flush()
            print(
                f"ep={ep:4d}  step={global_step:6d}  eps={agent.epsilon(global_step):.3f}  "
                f"train_r={ep_reward:6.2f}  loss={avg_loss:7.4f}  eval_r={avg_r:6.2f}  win_rate={win_rate:.2f}"
            )

            if win_rate >= 1.0:
                consecutive_full_wins += 1
                if not best_saved or consecutive_full_wins == 1:
                    torch.save(agent.state_dict(), CKPT_PATH)
                    best_saved = True
            else:
                consecutive_full_wins = 0

            if consecutive_full_wins >= EARLY_STOP_PATIENCE:
                print(f"Early stop at episode {ep}: {EARLY_STOP_PATIENCE} consecutive full-win evals.")
                break

    if not best_saved:
        torch.save(agent.state_dict(), CKPT_PATH)
        print("Training finished without reaching 100% win-rate; saved final weights anyway.")
    else:
        print(f"Best checkpoint saved to {CKPT_PATH}.")
    f.close()


if __name__ == "__main__":
    main()
