"""Behavioral tests for neural learning, locality, monotonicity, and persistence."""

import copy
import itertools
import json

import numpy as np
import pytest
import torch

from marl.environment import LinkedNavigation, partition_rewards
from marl.evaluate import evaluate
from marl.learner import NeuralLearner, ReplayBuffer, make_environment, td_targets
from marl.networks import PartitionMixer, QMixMixer
from marl.neural_config import CONFIG_ROOT, read_config, validate_config
from marl.train import train


def tiny_config(episodes=4):
    config = read_config()
    config["training"].update(episodes=episodes, batch_size=8, replay_capacity=64,
                              learning_starts=8, target_update_steps=5,
                              eval_episodes=4, checkpoint_every=2)
    return config


def test_local_rewards_coupled_dynamics_and_finite_horizon():
    env = LinkedNavigation(n_agents=2, horizon=4)
    env.reset(7)
    env.positions[:] = [2, 1]
    env.goals[:] = 6
    _, _, rewards, done, info = env.step(np.array([2, 2]))
    # Agent 0 cannot stretch its tether; agent 1 can catch up.
    np.testing.assert_array_equal(env.positions, [2, 2])
    np.testing.assert_allclose(rewards, [-4 / 6 - 0.01, -5 / 6 - 0.01])
    assert info["team_reward"] == float(rewards.sum())
    assert not done
    np.testing.assert_allclose(partition_rewards(rewards, [[0], [1]]).sum(), info["team_reward"])
    np.testing.assert_allclose(partition_rewards(rewards, [[0, 1]]), [info["team_reward"]])
    # Change neighbor state; the same own state/action has the same own reward.
    env.positions[:] = [2, 2]
    _, _, changed, _, _ = env.step(np.array([2, 2]))
    assert changed[0] == rewards[0]
    assert env.positions[0] == 3
    env.step(np.ones(2, dtype=np.int64))
    obs, _, _, done, _ = env.step(np.ones(2, dtype=np.int64))
    assert done and np.all(obs[:, 2] == 0)
    with pytest.raises(RuntimeError):
        env.step(np.array([1, 1]))


def test_tethers_hold_for_random_rollouts_and_action_validation():
    env = LinkedNavigation(n_agents=5, horizon=40)
    rng = np.random.default_rng(11)
    for seed in range(20):
        env.reset(seed)
        for _ in range(env.horizon):
            env.step(rng.integers(0, 3, env.n_agents))
            assert np.max(np.abs(np.diff(env.positions))) <= env.coupling_distance
    env.reset(1)
    for invalid in ([0, 1], [0.0] * 5, [3] * 5):
        with pytest.raises(ValueError):
            env.step(np.asarray(invalid))


@pytest.mark.parametrize("kind", ["lomaq", "qmix"])
def test_monotonicity_and_joint_greedy_consistency(kind):
    torch.manual_seed(2)
    mixer = PartitionMixer(3, 2, 8) if kind == "lomaq" else QMixMixer(3, 7, 8)
    utilities = torch.randn(5, 3, requires_grad=True)
    state = torch.randn(5, 7)
    values = mixer(utilities, state)
    for partition in range(values.shape[1]):
        gradient = torch.autograd.grad(values[:, partition].sum(), utilities, retain_graph=True)[0]
        # Every partition depends monotonically on EVERY utility, outside J too.
        assert torch.all(gradient > 0)
    action_values = torch.randn(3, 3)
    joint_actions = list(itertools.product(range(3), repeat=3))
    selected = torch.stack([action_values[range(3), action] for action in joint_actions])
    scores = mixer(selected, state[:1].expand(len(joint_actions), -1))
    greedy = tuple(action_values.argmax(-1).tolist())
    torch.testing.assert_close(scores[joint_actions.index(greedy)], scores.max(dim=0).values)


def test_decentralized_utility_is_independent_of_other_observations():
    learner = NeuralLearner(tiny_config())
    observations, _ = make_environment(learner.config).reset(11)
    original = learner.online.utilities(torch.tensor(observations).unsqueeze(0))
    changed = observations.copy()
    changed[1:] += 10
    revised = learner.online.utilities(torch.tensor(changed).unsqueeze(0))
    torch.testing.assert_close(original[:, 0], revised[:, 0])
    # No state or mixer call occurs at execution time.
    def forbidden(*args, **kwargs):
        raise AssertionError("A mixer was used to select decentralized actions")
    learner.online.mixer.forward = forbidden
    assert learner.act(observations).shape == (3,)


def test_terminal_mask_and_reward_targets():
    rewards = torch.tensor([[1., 2.], [3., 4.]])
    target = td_targets(rewards, torch.full((2, 2), 10.), torch.tensor([True, False]), .9)
    torch.testing.assert_close(target, torch.tensor([[1., 2.], [12., 13.]]))
    config = tiny_config()
    local = torch.tensor([[1., 2., 4.]])
    for name, expected in [("lomaq", [[3., 4.]]), ("qmix", [[7.]]), ("vdn", [[7.]]),
                           ("iql", [[7., 7., 7.]]), ("iql_local", [[1., 2., 4.]])]:
        config["algorithm"].update(name=name, partitions=[[0, 1], [2]])
        learner = NeuralLearner(config)
        torch.testing.assert_close(learner.reward_targets(local), torch.tensor(expected))
        if name == "vdn":
            obs, state = make_environment(config).reset(1)
            obs, state = torch.tensor(obs)[None], torch.tensor(state)[None]
            actions = torch.ones(1, 3, dtype=torch.long)
            values = learner.online.selected_values(obs, state, actions)
            utilities = learner.online.utilities(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            assert values.shape == (1, 1)
            torch.testing.assert_close(values, utilities.sum(-1, keepdim=True))


def test_replay_ring_roundtrip():
    replay = ReplayBuffer(3)
    for value in range(5):
        replay.add(value=np.array([value], dtype=np.float32))
    restored = ReplayBuffer(1)
    restored.load_state_dict(replay.state_dict())
    assert restored.size == 3 and restored.position == 2
    torch.testing.assert_close(replay.sample(3, np.random.default_rng(4))["value"],
                               restored.sample(3, np.random.default_rng(4))["value"])


def test_double_dqn_uses_online_action_and_frozen_target_value():
    config = tiny_config()
    config["algorithm"]["name"] = "iql_local"
    learner = NeuralLearner(config)
    with torch.no_grad():
        for system in (learner.online, learner.target):
            for parameter in system.parameters():
                parameter.zero_()
        for agent in learner.online.utilities.agents:
            agent[-1].bias.copy_(torch.tensor([0., 1., 0.]))
        for agent in learner.target.utilities.agents:
            agent[-1].bias.copy_(torch.tensor([100., 2., 0.]))
    obs, state = make_environment(config).reset(1)
    for _ in range(8):
        learner.replay.add(observations=obs, state=state, actions=np.full(3, 2, dtype=np.int64),
                           local_rewards=np.zeros(3, dtype=np.float32), next_observations=obs,
                           next_state=state, terminated=False)
    # Online chooses action 1; target evaluates it as 2, rather than its own max 100.
    assert learner.update() == pytest.approx((config["algorithm"]["gamma"] * 2) ** 2)
    assert all(parameter.grad is None for parameter in learner.target.parameters())


@pytest.mark.parametrize("change", [
    {"algorithm": {"partitions": [[0], [0, 2]]}},
    {"training": {"epsilon_end": 2}},
    {"training": {"eval_seed": 8}},
    {"training": {"batch_size": 0}},
    {"algorithm": {"learning_rate": float("nan")}},
    {"environment": {"unknown": 1}},
])
def test_invalid_neural_configuration(change):
    config = tiny_config()
    for section, overrides in change.items():
        config[section].update(overrides)
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.integration
def test_checkpoint_resume_is_identical_to_uninterrupted_training(tmp_path):
    config = tiny_config(8)
    continuous, _ = train(config, tmp_path / "continuous")
    partial_config = copy.deepcopy(config)
    partial_config["training"]["episodes"] = 3
    partial, _ = train(partial_config, tmp_path / "resumed")
    loaded = NeuralLearner.load(tmp_path / "resumed" / "checkpoint.pt")
    obs, _ = make_environment(config).reset(111)
    np.testing.assert_array_equal(partial.act(obs), loaded.act(obs))
    resumed, _ = train(config, tmp_path / "resumed", resume=tmp_path / "resumed" / "checkpoint.pt")
    assert continuous.history == resumed.history
    assert continuous.env_steps == resumed.env_steps == 128
    assert continuous.updates == resumed.updates > 0
    for key, tensor in continuous.online.state_dict().items():
        torch.testing.assert_close(tensor, resumed.online.state_dict()[key], rtol=0, atol=0)
    for key, tensor in continuous.target.state_dict().items():
        torch.testing.assert_close(tensor, resumed.target.state_dict()[key], rtol=0, atol=0)
    rng_before = json.dumps(resumed.rng.bit_generator.state)
    evaluate(resumed, 4, 30000)
    assert rng_before == json.dumps(resumed.rng.bit_generator.state)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["qmix", "iql", "iql_local", "vdn"])
def test_comparisons_train_save_reload_and_evaluate(tmp_path, algorithm):
    config = tiny_config()
    config["algorithm"]["name"] = algorithm
    initial = NeuralLearner(config)
    learner, result = train(config, tmp_path / algorithm)
    assert learner.updates > 0 and np.isfinite(result["after"]["return_mean"])
    assert any(not torch.equal(value, learner.online.state_dict()[key])
               for key, value in initial.online.state_dict().items())
    loaded = NeuralLearner.load(tmp_path / algorithm / "checkpoint.pt")
    assert evaluate(loaded, 4, 10000) == result["after"]


@pytest.mark.integration
def test_neural_lomaq_learns_on_held_out_reset_seeds(tmp_path):
    config = read_config(test=CONFIG_ROOT / "test" / "smoke.json")
    learner, result = train(config, tmp_path / "learned")
    assert learner.updates >= 3000
    assert result["after"]["return_mean"] > result["random_baseline"]["return_mean"] + 15
    assert result["after"]["return_mean"] > result["before"]["return_mean"] + 15
    assert result["after"]["success_rate"] >= .75
    reloaded = NeuralLearner.load(tmp_path / "learned" / "checkpoint.pt")
    assert evaluate(reloaded, 32, 10000) == result["after"]
