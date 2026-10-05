"""Evaluate a neural checkpoint and seeded random baseline without exploration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from marl.learner import NeuralLearner, make_environment


def evaluate(learner: NeuralLearner, episodes: int, seed: int, *, random_policy: bool = False) -> dict:
    if episodes < 1 or seed < 0:
        raise ValueError("episodes must be positive and seed nonnegative")
    env = make_environment(learner.config)
    # Evaluation has its own environment and RNG; it never modifies replay/training RNG.
    rng = np.random.default_rng(seed)
    returns, successes, arrival_steps = [], [], []
    for episode in range(episodes):
        observations, _ = env.reset(seed + episode)
        total = 0.0
        first_arrival = env.horizon + 1
        for step in range(env.horizon):
            actions = (rng.integers(0, env.n_actions, env.n_agents) if random_policy
                       else learner.act(observations))
            observations, _, _, done, info = env.step(actions)
            total += info["team_reward"]
            if info["success"] and first_arrival == env.horizon + 1:
                first_arrival = step + 1
            if done:
                break
        returns.append(total)
        successes.append(float(info["success"]))
        arrival_steps.append(first_arrival)
    return {"episodes": episodes, "seed": seed, "policy": "random" if random_policy else "greedy",
            "return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
            "success_rate": float(np.mean(successes)),
            "first_arrival_step_mean": float(np.mean(arrival_steps)),
            "return_per_episode": returns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    learner = NeuralLearner.load(args.checkpoint)
    result = {"algorithm": learner.config["algorithm"]["name"],
              "trained_episodes": learner.episodes, "env_steps": learner.env_steps,
              "evaluation": evaluate(learner, args.episodes, args.seed),
              "random_baseline": evaluate(learner, args.episodes, args.seed, random_policy=True)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
