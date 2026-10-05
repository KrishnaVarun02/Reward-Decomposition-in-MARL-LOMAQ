"""Validated four-layer configuration for neural experiments."""

from __future__ import annotations

import math
from pathlib import Path

from marl.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "neural"


def read_config(environment: Path | None = None, algorithm: Path | None = None,
                test: Path | None = None) -> dict:
    config = load_config(CONFIG_ROOT / "default.json",
                         environment or CONFIG_ROOT / "environment" / "linked_navigation.json",
                         algorithm or CONFIG_ROOT / "algorithm" / "lomaq.json",
                         test or CONFIG_ROOT / "test" / "default.json")
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    schema = {
        "environment": {"name", "n_agents", "length", "horizon", "coupling_distance"},
        "algorithm": {"name", "partitions", "hidden_dim", "mixer_dim", "gamma", "learning_rate"},
        "training": {"seed", "episodes", "batch_size", "replay_capacity", "learning_starts",
                     "train_every", "target_update_steps", "epsilon_start", "epsilon_end",
                     "epsilon_decay_steps", "gradient_clip", "eval_episodes", "eval_seed",
                     "checkpoint_every"},
    }
    if set(config) != set(schema):
        raise ValueError(f"Expected config sections {sorted(schema)}")
    for section, keys in schema.items():
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f"Invalid/missing {section} keys; expected {sorted(keys)}")
    env, alg, train = (config[k] for k in ("environment", "algorithm", "training"))
    if env["name"] != "linked_navigation":
        raise ValueError("Only linked_navigation is available")
    if alg["name"] not in {"lomaq", "qmix", "iql", "iql_local", "vdn"}:
        raise ValueError("Unknown algorithm")
    integers = [(env, k) for k in ("n_agents", "length", "horizon", "coupling_distance")]
    integers += [(alg, k) for k in ("hidden_dim", "mixer_dim")]
    integers += [(train, k) for k in ("episodes", "batch_size", "replay_capacity", "learning_starts",
                                    "train_every", "target_update_steps", "epsilon_decay_steps",
                                    "eval_episodes", "checkpoint_every")]
    for group, key in integers:
        if type(group[key]) is not int or group[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "eval_seed"):
        if type(train[key]) is not int or train[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if env["n_agents"] < 2 or env["length"] < 5 or env["horizon"] < 2:
        raise ValueError("Need n_agents >= 2, length >= 5, horizon >= 2")
    if not 2 <= env["coupling_distance"] < env["length"]:
        raise ValueError("Invalid coupling_distance")
    partitions = alg["partitions"]
    if (not isinstance(partitions, list) or not partitions
            or any(not isinstance(part, list) or not part for part in partitions)
            or any(type(i) is not int for part in partitions for i in part)
            or sorted(i for part in partitions for i in part) != list(range(env["n_agents"]))):
        raise ValueError("Partitions must cover each agent exactly once")
    numbers = [(alg, k) for k in ("gamma", "learning_rate")]
    numbers += [(train, k) for k in ("epsilon_start", "epsilon_end", "gradient_clip")]
    for group, key in numbers:
        if not isinstance(group[key], (int, float)) or not math.isfinite(group[key]):
            raise ValueError(f"{key} must be a finite number")
    if not 0 <= alg["gamma"] <= 1 or alg["learning_rate"] <= 0 or train["gradient_clip"] <= 0:
        raise ValueError("Invalid gamma, learning_rate or gradient_clip")
    if not 0 <= train["epsilon_end"] <= train["epsilon_start"] <= 1:
        raise ValueError("Epsilon must satisfy 0 <= end <= start <= 1")
    if train["replay_capacity"] < train["batch_size"]:
        raise ValueError("Replay capacity must be at least batch_size")
    # Disjoint reset-seed ranges make before/after evaluation truly held out.
    if max(train["seed"], train["eval_seed"]) < min(
            train["seed"] + train["episodes"], train["eval_seed"] + train["eval_episodes"]):
        raise ValueError("Training and evaluation reset seed ranges must be disjoint")
