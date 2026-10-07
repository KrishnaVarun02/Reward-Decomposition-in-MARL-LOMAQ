"""Run an illustrative discrete payoff matrix, not the archived experiments."""

import argparse
from pathlib import Path

from marl.config import load_config
from marl.lomaq import LOMAQ


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-config", type=Path,
                        default=Path("configs/test/payoff_matrix.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    test_config = args.test_config
    if not test_config.is_absolute():
        test_config = root / test_config
    config = load_config(root / "configs/default.json",
                         root / "configs/environment/payoff_matrix.json",
                         root / "configs/algorithm/lomaq.json",
                         test_config)
    learner = LOMAQ(**config["algorithm"], n_agents=config["environment"]["n_agents"],
                    n_actions=config["environment"]["n_actions"],
                    partitions=config["environment"]["partitions"])
    observations = ("only_state",) * learner.n_agents
    for _ in range(config["episodes"]):
        actions = learner.act(observations)
        # An illustrative cooperative payoff, chosen solely for this demo.
        reward = 1.0 if all(action == 1 for action in actions) else 0.0
        learner.update(observations, actions, [reward], observations, done=True)
    print({"greedy_actions": learner.act(observations, explore=False),
           "episodes": config["episodes"]})


if __name__ == "__main__":
    main()
