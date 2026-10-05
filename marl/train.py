"""Train neural LOMAQ and local/global reward comparisons on linked navigation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from marl.evaluate import evaluate
from marl.learner import NeuralLearner, make_environment
from marl.neural_config import read_config, validate_config


def train(config: dict, output: Path, *, resume: Path | None = None) -> tuple[NeuralLearner, dict]:
    validate_config(config)
    if resume is not None:
        learner = NeuralLearner.load(resume)
        previous, requested = copy.deepcopy(learner.config), copy.deepcopy(config)
        # Only the requested total episode count can change on an exact resume.
        previous["training"].pop("episodes")
        requested["training"].pop("episodes")
        if previous != requested:
            raise ValueError("Resume only permits changing training.episodes")
        if config["training"]["episodes"] <= learner.episodes:
            raise ValueError("Resume episodes must exceed completed episodes")
        learner.config = copy.deepcopy(config)
    else:
        if (output / "checkpoint.pt").exists():
            raise FileExistsError("Output already has a checkpoint; use --resume or another output")
        learner = NeuralLearner(config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    settings = config["training"]
    before = evaluate(learner, settings["eval_episodes"], settings["eval_seed"])
    random_baseline = evaluate(learner, settings["eval_episodes"], settings["eval_seed"], random_policy=True)
    env = make_environment(config)
    starting_episode = learner.episodes
    for episode in range(starting_episode, settings["episodes"]):
        observations, state = env.reset(settings["seed"] + episode)
        episode_return, losses = 0.0, []
        for _ in range(env.horizon):
            actions = learner.act(observations, learner.epsilon())
            next_observations, next_state, rewards, done, info = env.step(actions)
            loss = learner.observe(observations=observations, state=state, actions=actions,
                                   local_rewards=rewards, next_observations=next_observations,
                                   next_state=next_state, terminated=done)
            if loss is not None:
                losses.append(loss)
            episode_return += info["team_reward"]
            observations, state = next_observations, next_state
            if done:
                break
        learner.episodes = episode + 1
        learner.history.append({"episode": learner.episodes, "env_steps": learner.env_steps,
                                "return": episode_return, "epsilon": learner.epsilon(),
                                "loss_mean": float(np.mean(losses)) if losses else None,
                                "success": info["success"]})
        if learner.episodes % settings["checkpoint_every"] == 0:
            learner.save(output / "checkpoint.pt")
    learner.save(output / "checkpoint.pt")
    after = evaluate(learner, settings["eval_episodes"], settings["eval_seed"])
    result = {"algorithm": config["algorithm"]["name"], "starting_episode": starting_episode,
              "episodes": learner.episodes, "env_steps": learner.env_steps, "updates": learner.updates,
              "before": before, "after": after, "random_baseline": random_baseline,
              "versions": {"numpy": np.__version__, "torch": str(torch.__version__)}}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in learner.history))
    return learner, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-config", type=Path)
    parser.add_argument("--algorithm-config", type=Path)
    parser.add_argument("--test-config", type=Path)
    parser.add_argument("--episodes", type=int, help="Total episodes, including episodes before resume")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        if args.environment_config or args.algorithm_config or args.test_config:
            parser.error("Resume uses the checkpoint configuration; only --episodes may change")
        config = NeuralLearner.load(args.resume).config
    else:
        config = read_config(args.environment_config, args.algorithm_config, args.test_config)
    if args.episodes is not None:
        config["training"]["episodes"] = args.episodes
    _, result = train(config, args.output, resume=args.resume)
    # Full per-episode returns are saved in metrics.json; keep CLI output compact.
    for name in ("before", "after", "random_baseline"):
        result[name].pop("return_per_episode")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
