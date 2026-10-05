"""Tabular utilities with a monotone linear partition mixer.

This is a minimal instance of the report's LOMAQ update. It accepts local rewards
per partition, trains with joint transitions, and selects actions from local
utilities alone. It does not reproduce the report's neural experiments.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Hashable, Sequence


class LOMAQ:
    def __init__(self, n_agents: int, n_actions: int,
                 partitions: Sequence[Sequence[int]], *, gamma: float = 0.99,
                 learning_rate: float = 0.1, epsilon: float = 0.1,
                 seed: int = 7):
        if n_agents < 1 or n_actions < 2:
            raise ValueError("Need at least one agent and two actions")
        if sorted(i for part in partitions for i in part) != list(range(n_agents)) or any(not part for part in partitions):
            raise ValueError("Partitions must cover each agent exactly once")
        if not 0 <= gamma <= 1 or not 0 < learning_rate <= 1 or not 0 <= epsilon <= 1:
            raise ValueError("Invalid gamma, learning rate or epsilon")
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.partitions = tuple(tuple(part) for part in partitions)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.random = random.Random(seed)
        self.utilities: dict[tuple[int, Hashable], list[float]] = defaultdict(
            lambda: [0.0] * n_actions
        )
        self.weights = [[1.0] * len(part) for part in self.partitions]
        self.bias = [0.0] * len(self.partitions)

    def act(self, observations: Sequence[Hashable], *, explore: bool = True) -> tuple[int, ...]:
        if len(observations) != self.n_agents:
            raise ValueError("One local observation is required per agent")
        actions = []
        for agent, observation in enumerate(observations):
            if explore and self.random.random() < self.epsilon:
                actions.append(self.random.randrange(self.n_actions))
            else:
                utilities = self.utilities[agent, observation]
                actions.append(max(range(self.n_actions), key=utilities.__getitem__))
        return tuple(actions)

    def partition_value(self, partition: int, observations: Sequence[Hashable],
                        actions: Sequence[int]) -> float:
        return self.bias[partition] + sum(
            weight * self.utilities[agent, observations[agent]][actions[agent]]
            for agent, weight in zip(self.partitions[partition], self.weights[partition])
        )

    def update(self, observations: Sequence[Hashable], actions: Sequence[int],
               local_rewards: Sequence[float], next_observations: Sequence[Hashable],
               *, done: bool = False) -> tuple[float, ...]:
        if len(observations) != self.n_agents or len(next_observations) != self.n_agents or len(actions) != self.n_agents:
            raise ValueError("One state and action is required per agent")
        if len(local_rewards) != len(self.partitions):
            raise ValueError("One local reward is required per partition")
        if any(not 0 <= action < self.n_actions for action in actions):
            raise ValueError("Action outside discrete action space")
        greedy_next = self.act(next_observations, explore=False)
        errors = []
        for j, part in enumerate(self.partitions):
            current = self.partition_value(j, observations, actions)
            next_value = 0.0 if done else self.partition_value(j, next_observations, greedy_next)
            error = float(local_rewards[j]) + self.gamma * next_value - current
            # Read the old parameters before applying all gradients.
            old_weights = self.weights[j][:]
            old_utilities = [self.utilities[agent, observations[agent]][actions[agent]]
                             for agent in part]
            for pos, agent in enumerate(part):
                self.utilities[agent, observations[agent]][actions[agent]] += (
                    self.learning_rate * error * old_weights[pos]
                )
                self.weights[j][pos] = max(
                    0.0, old_weights[pos] + self.learning_rate * error * old_utilities[pos]
                )
            self.bias[j] += self.learning_rate * error
            errors.append(error)
        return tuple(errors)
