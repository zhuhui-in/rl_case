"""Deterministic Magic-Tower environment for the fixed CONFIG in instructions.md."""
from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


CONFIG: dict[str, int] = {
    "N": 15,
    "A": 5,
    "T": 2,
    "H0": 60,
    "d0": 1,
    "G": 3,
    "Cd": 10,
    "Ch": 8,
    "R": 12,
}

ACTION_FIGHT = 0
ACTION_BUY_DEF = 1
ACTION_BUY_HEAL = 2

D_NORM = 10.0
M_NORM = 50.0

REWARD_KILL = 1.0
REWARD_WIN = 50.0
REWARD_DEAD = -50.0
REWARD_INVALID = -0.1
REWARD_NOOP = 0.0

MAX_STEPS = 200


class MagicTowerEnv(gym.Env):
    """Deterministic MDP for the magic-tower task."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=np.float32(np.inf), shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._hp: int = 0
        self._d: int = 0
        self._money: int = 0
        self._killed: int = 0
        self._steps: int = 0

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                self._hp / CONFIG["H0"],
                self._d / D_NORM,
                self._money / M_NORM,
                self._killed / CONFIG["N"],
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._hp = CONFIG["H0"]
        self._d = CONFIG["d0"]
        self._money = 0
        self._killed = 0
        self._steps = 0
        return self._obs(), {"win": False}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = int(action)
        self._steps += 1
        reward = REWARD_NOOP
        terminated = False
        truncated = False
        win = False

        if action == ACTION_FIGHT:
            loss = max(0, CONFIG["A"] - self._d) * CONFIG["T"]
            self._hp -= loss
            if self._hp <= 0:
                reward = REWARD_DEAD
                terminated = True
            else:
                self._killed += 1
                self._money += CONFIG["G"]
                reward = REWARD_KILL
                if self._killed >= CONFIG["N"]:
                    reward += REWARD_WIN
                    terminated = True
                    win = True
        elif action == ACTION_BUY_DEF:
            if self._money >= CONFIG["Cd"]:
                self._money -= CONFIG["Cd"]
                self._d += 1
            else:
                reward = REWARD_INVALID
        elif action == ACTION_BUY_HEAL:
            if self._money >= CONFIG["Ch"]:
                self._money -= CONFIG["Ch"]
                self._hp += CONFIG["R"]
            else:
                reward = REWARD_INVALID
        else:
            reward = REWARD_INVALID

        if not terminated and self._steps >= MAX_STEPS:
            truncated = True

        info = {
            "win": win,
            "hp": self._hp,
            "d": self._d,
            "money": self._money,
            "killed": self._killed,
        }
        return self._obs(), float(reward), terminated, truncated, info

    @property
    def raw_state(self) -> dict[str, int]:
        return {
            "hp": self._hp,
            "d": self._d,
            "money": self._money,
            "killed": self._killed,
        }
