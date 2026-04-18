"""DQN agent: Q-network, replay buffer, target net, epsilon-greedy."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DQNConfig:
    obs_dim: int = 4
    n_actions: int = 3
    hidden: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer: int = 1_000
    target_sync_steps: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 5_000
    grad_clip: float = 10.0
    device: str = "cpu"


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            np.stack(obs).astype(np.float32),
            np.array(act, dtype=np.int64),
            np.array(rew, dtype=np.float32),
            np.stack(next_obs).astype(np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent:
    def __init__(self, cfg: DQNConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.online = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.target = QNet(cfg.obs_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.train_steps = 0

    def epsilon(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.cfg.n_actions)
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        q = self.online(x)
        return int(torch.argmax(q, dim=1).item())

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float | None:
        if len(self.buffer) < max(self.cfg.batch_size, self.cfg.min_buffer):
            return None
        obs, act, rew, next_obs, done = self.buffer.sample(self.cfg.batch_size)
        obs_t = torch.as_tensor(obs).to(self.device)
        act_t = torch.as_tensor(act).to(self.device)
        rew_t = torch.as_tensor(rew).to(self.device)
        next_obs_t = torch.as_tensor(next_obs).to(self.device)
        done_t = torch.as_tensor(done).to(self.device)

        q = self.online(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(next_obs_t).max(dim=1).values
            target = rew_t + self.cfg.gamma * q_next * (1.0 - done_t)
        loss = F.smooth_l1_loss(q, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "train_steps": self.train_steps,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.online.load_state_dict(sd["online"])
        self.target.load_state_dict(sd["target"])
        self.train_steps = sd.get("train_steps", 0)
