"""Plot training curves from logs/metrics.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

METRICS_PATH = Path("logs/metrics.csv")
PLOT_PATH = Path("logs/training_curves.png")


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"{METRICS_PATH} not found. Run `python train.py` first.")

    episodes: list[int] = []
    train_rewards: list[float] = []
    losses: list[float] = []
    eval_rewards: list[float] = []
    win_rates: list[float] = []
    with open(METRICS_PATH, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            episodes.append(int(row["episode"]))
            train_rewards.append(float(row["train_reward"]))
            losses.append(float(row["avg_loss"]))
            eval_rewards.append(float(row["eval_avg_reward"]))
            win_rates.append(float(row["eval_win_rate"]))

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(episodes, eval_rewards, label="eval_avg_reward", color="tab:blue")
    axes[0].plot(episodes, train_rewards, label="train_reward", color="tab:orange", alpha=0.4)
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(episodes, losses, label="avg_td_loss", color="tab:red")
    axes[1].set_ylabel("TD Loss (Huber)")
    axes[1].set_yscale("symlog")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(episodes, win_rates, label="eval_win_rate", color="tab:green")
    axes[2].set_ylabel("Win Rate")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("Episode")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle("Magic Tower DQN Training")
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"Saved {PLOT_PATH}")


if __name__ == "__main__":
    main()
