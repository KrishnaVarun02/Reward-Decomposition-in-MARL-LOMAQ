"""Centralized neural TD learning, local-only execution, and full checkpoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from marl.environment import LinkedNavigation
from marl.networks import ValueSystem
from marl.neural_config import validate_config


def make_environment(config: dict) -> LinkedNavigation:
    return LinkedNavigation(**{k: v for k, v in config["environment"].items() if k != "name"})


class ReplayBuffer:
    """CPU tensor ring buffer; checkpoints preserve its exact sampling population."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: dict[str, torch.Tensor] = {}
        self.size = 0
        self.position = 0

    def add(self, **transition) -> None:
        values = {key: torch.as_tensor(value) for key, value in transition.items()}
        if not self.data:
            self.data = {key: torch.empty((self.capacity, *value.shape), dtype=value.dtype)
                         for key, value in values.items()}
        for key, value in values.items():
            self.data[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError("Not enough replay transitions")
        indices = rng.integers(0, self.size, batch_size)
        return {key: value[indices] for key, value in self.data.items()}

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "size": self.size, "position": self.position,
                "data": {k: v[:self.size].clone() for k, v in self.data.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.capacity, self.size, self.position = (state[k] for k in ("capacity", "size", "position"))
        self.data = {}
        for key, value in state["data"].items():
            self.data[key] = torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
            self.data[key][:self.size] = value


def td_targets(rewards: torch.Tensor, next_values: torch.Tensor,
               terminated: torch.Tensor, gamma: float) -> torch.Tensor:
    """Finite-horizon terminals never bootstrap (horizon is part of the state)."""
    return rewards + gamma * (1 - terminated.float().unsqueeze(-1)) * next_values


class NeuralLearner:
    def __init__(self, config: dict):
        validate_config(config)
        self.config = copy.deepcopy(config)
        seed = config["training"]["seed"]
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        env = make_environment(config)
        self.online = ValueSystem(config, env.obs_dim, env.state_dim, env.n_actions)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["algorithm"]["learning_rate"])
        self.replay = ReplayBuffer(config["training"]["replay_capacity"])
        self.env_steps = 0
        self.updates = 0
        self.episodes = 0
        self.history: list[dict] = []

    def epsilon(self) -> float:
        train = self.config["training"]
        fraction = min(1.0, self.env_steps / train["epsilon_decay_steps"])
        return train["epsilon_start"] + fraction * (train["epsilon_end"] - train["epsilon_start"])

    @torch.no_grad()
    def act(self, observations: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """Only per-agent observations enter this path; mixing is training-only."""
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        values = self.online.utilities(torch.as_tensor(observations, dtype=torch.float32).unsqueeze(0))[0]
        actions = values.argmax(dim=-1).numpy()
        # Pure greedy evaluation consumes no training randomness.
        if epsilon:
            mask = self.rng.random(len(actions)) < epsilon
            actions = np.where(mask, self.rng.integers(0, values.shape[-1], len(actions)), actions)
        return actions

    def observe(self, **transition) -> float | None:
        self.replay.add(**transition)
        self.env_steps += 1
        train = self.config["training"]
        if (self.env_steps < train["learning_starts"] or self.replay.size < train["batch_size"]
                or self.env_steps % train["train_every"]):
            return None
        return self.update()

    def reward_targets(self, local: torch.Tensor) -> torch.Tensor:
        alg = self.config["algorithm"]
        if alg["name"] == "lomaq":
            return torch.stack([local[:, part].sum(-1) for part in alg["partitions"]], dim=-1)
        if alg["name"] == "iql_local":
            return local
        team = local.sum(-1, keepdim=True)
        return team.expand_as(local) if alg["name"] == "iql" else team

    def update(self) -> float:
        train = self.config["training"]
        batch = self.replay.sample(train["batch_size"], self.rng)
        chosen = self.online.selected_values(batch["observations"], batch["state"], batch["actions"])
        with torch.no_grad():
            # Double DQN: online local utilities choose, frozen target evaluates.
            next_actions = self.online.utilities(batch["next_observations"]).argmax(dim=-1)
            next_values = self.target.selected_values(batch["next_observations"], batch["next_state"], next_actions)
            targets = td_targets(self.reward_targets(batch["local_rewards"]), next_values,
                                 batch["terminated"], self.config["algorithm"]["gamma"])
        loss = F.mse_loss(chosen, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite TD loss")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), train["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % train["target_update_steps"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1, "config": self.config,
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
            "rng_json": json.dumps(self.rng.bit_generator.state), "torch_rng": torch.get_rng_state(),
            "env_steps": self.env_steps, "updates": self.updates, "episodes": self.episodes,
            "history": self.history,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> NeuralLearner:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        learner = cls(checkpoint["config"])
        learner.online.load_state_dict(checkpoint["online"])
        learner.target.load_state_dict(checkpoint["target"])
        learner.optimizer.load_state_dict(checkpoint["optimizer"])
        learner.replay.load_state_dict(checkpoint["replay"])
        learner.rng.bit_generator.state = json.loads(checkpoint["rng_json"])
        torch.set_rng_state(checkpoint["torch_rng"])
        for name in ("env_steps", "updates", "episodes", "history"):
            setattr(learner, name, checkpoint[name])
        return learner
