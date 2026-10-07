"""A new, self-contained cooperative navigation task with coupled dynamics.

This is a compact validation environment, not the report's unpublished bounded
cooperative-navigation implementation. Rewards depend on each agent's own state
and action. Neighbor states affect movement, making independent progress coupled.
"""

from __future__ import annotations

import numpy as np


class LinkedNavigation:
    """Carriers on parallel lanes must move to a shared endpoint and hold there.

    A chain tether prevents a carrier moving farther from a neighbor when their
    separation is already ``coupling_distance - 1``. Simultaneous independent
    moves therefore cannot stretch any link beyond ``coupling_distance``.
    Rewards are computed BEFORE moving: -normalized distance + 0.5 on the goal,
    minus 0.01 for a non-stay action. Team reward is exactly their sum. Episodes
    have a finite horizon (included in every observation) and never stop early.
    """

    n_actions = 3  # left, stay, right

    def __init__(self, n_agents: int = 3, length: int = 7,
                 horizon: int = 16, coupling_distance: int = 2):
        if n_agents < 2 or length < 5 or horizon < 2:
            raise ValueError("Need n_agents >= 2, length >= 5 and horizon >= 2")
        if not 2 <= coupling_distance < length:
            raise ValueError("coupling_distance must be in [2, length)")
        self.n_agents = n_agents
        self.length = length
        self.horizon = horizon
        self.coupling_distance = coupling_distance
        # Own position/goal/time, then neighbor relative positions and masks.
        self.obs_dim = 7
        self.state_dim = 2 * n_agents + 1
        self.positions = np.zeros(n_agents, dtype=np.int64)
        self.goals = np.zeros(n_agents, dtype=np.int64)
        self.t = 0

    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        center = int(rng.integers(1, self.length - 1))
        # Two adjacent start cells always satisfy the chain constraint.
        self.positions = np.clip(center + rng.integers(0, 2, self.n_agents),
                                 0, self.length - 1)
        self.goals[:] = int(rng.choice([0, self.length - 1]))
        self.t = 0
        return self.observe(), self.state()

    def observe(self) -> np.ndarray:
        scale = self.length - 1
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i in range(self.n_agents):
            obs[i, :3] = (self.positions[i] / scale, self.goals[i] / scale,
                          (self.horizon - self.t) / self.horizon)
            for column, neighbor in ((3, i - 1), (5, i + 1)):
                if 0 <= neighbor < self.n_agents:
                    obs[i, column] = (self.positions[neighbor] - self.positions[i]) / scale
                    obs[i, column + 1] = 1
        return obs

    def state(self) -> np.ndarray:
        return np.concatenate((self.positions / (self.length - 1),
                               self.goals / (self.length - 1),
                               [(self.horizon - self.t) / self.horizon])).astype(np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        actions = np.asarray(actions)
        if self.t >= self.horizon:
            raise RuntimeError("Episode ended; call reset before stepping")
        if actions.shape != (self.n_agents,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("Expected one integer action per agent")
        if np.any((actions < 0) | (actions >= self.n_actions)):
            raise ValueError("Actions must be 0 (left), 1 (stay), or 2 (right)")
        rewards = (-np.abs(self.positions - self.goals) / (self.length - 1)
                   + 0.5 * (self.positions == self.goals)
                   - 0.01 * (actions != 1)).astype(np.float32)
        moves = actions - 1
        blocked = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            for neighbor in (i - 1, i + 1):
                if 0 <= neighbor < self.n_agents:
                    separation = self.positions[i] - self.positions[neighbor]
                    if abs(separation) >= self.coupling_distance - 1 and separation * moves[i] > 0:
                        blocked[i] = True
        self.positions = np.clip(self.positions + np.where(blocked, 0, moves),
                                 0, self.length - 1)
        self.t += 1
        info = {"team_reward": float(rewards.sum()),
                "success": bool(np.all(self.positions == self.goals)),
                "blocked_moves": int(blocked.sum())}
        return self.observe(), self.state(), rewards, self.t == self.horizon, info


def partition_rewards(local_rewards: np.ndarray, partitions: list[list[int]]) -> np.ndarray:
    """Sum additive agent rewards; works on a transition or a replay batch."""
    return np.stack([local_rewards[..., part].sum(axis=-1) for part in partitions], axis=-1)
