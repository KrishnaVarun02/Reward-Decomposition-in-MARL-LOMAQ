# Reward Decomposition in MARL — neural LOMAQ

This repository contains the 2023 project report, the preserved standard-library tabular example, and a runnable PyTorch implementation of neural LOMAQ with training, evaluation, and exact checkpoint resume. Agent utility networks learn from replayed joint transitions; each partition has a monotonic neural mixer supervised by that partition's additive local reward. At execution, each agent selects actions using only its own observation and utility network.

The included **Linked Navigation** environment is a new, compact cooperative task for validating the implementation. It is not the report's unpublished bounded-cooperative-navigation environment. None of the numerical results in the report are claimed to have been reproduced.

## Install and run

Use Python 3.10+ from the repository root. All neural commands run on CPU with one Torch thread and deterministic operations.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

# Train 250 episodes (4,000 sequential environment steps).
python -m marl.train --test-config configs/neural/test/smoke.json --output runs/lomaq

# Load the saved model and evaluate greedily on new reset seeds.
python -m marl.evaluate --checkpoint runs/lomaq/checkpoint.pt \
  --episodes 100 --seed 20000 --output runs/lomaq/evaluation.json

# Continue to 400 TOTAL episodes with the original exploration schedule.
python -m marl.train --resume runs/lomaq/checkpoint.pt \
  --episodes 400 --output runs/lomaq

python -m pytest -q
python -m ruff check .
```

Without `--test-config`, training uses 400 episodes and the default exploration schedule. `--episodes` overrides the total training budget. A new run refuses to overwrite an existing `checkpoint.pt`. Resume uses its checkpoint's saved configuration and permits only changing the total episode count. Evaluation is greedy (`epsilon=0`), uses a separate environment/RNG, and does not advance training RNGs or counters.

Each output folder contains:

- `config.json`: fully resolved configuration;
- `checkpoint.pt`: online utilities/mixers, target networks, Adam state, replay buffer, NumPy/Torch RNGs, counters and history;
- `history.jsonl`: per-episode returns, epsilon, TD loss and final success;
- `metrics.json`: before/after greedy evaluation, a seeded random-policy comparison, raw episode returns, versions and update counts.

Checkpoints are written atomically at episode boundaries. Resume reconstructs episode reset seeds from the saved count. The tests compare uninterrupted training with interrupted/resumed training **exactly**, including online/target parameters and history. Bitwise identity is only expected on the same supported CPU software stack. `torch.load(..., weights_only=True)` is used. Models and replay artifacts are not committed.

### Comparisons under the same budget

```bash
for algorithm in qmix iql iql_local vdn; do
  python -m marl.train \
    --algorithm-config configs/neural/algorithm/$algorithm.json \
    --test-config configs/neural/test/smoke.json \
    --output runs/$algorithm
  python -m marl.evaluate --checkpoint runs/$algorithm/checkpoint.pt \
    --episodes 100 --seed 20000 --output runs/$algorithm/evaluation.json
done
```

All algorithms use identical environment settings, network width, replay budget, optimizer, exploration schedule and training/evaluation reset seeds. LOMAQ trains one value per partition; QMIX uses a state-conditioned monotonic mixer and a team-reward target; VDN sums utility values and uses a team-reward target; IQL gives each independent learner the team reward; IQL-local gives each learner its own reward. These are fresh implementations for this environment, not restored historical experiment code. A single training seed is smoke evidence, not an algorithm ranking; vary `training.seed` in test configuration and use multiple independent runs for scientific comparisons.

## Linked Navigation task and method

Three carriers move left/stay/right along parallel lanes of length seven toward a shared, randomly selected endpoint. Neighboring carriers are linked in a chain. A carrier cannot move farther away from a neighbor if their current separation is at least `coupling_distance - 1`. This maintains the maximum tether length even with simultaneous movement, and an agent that falls behind can delay a teammate. Initial positions vary across resets; each episode is 16 transitions.

Agent `i` receives its own normalized position, goal and remaining time, plus relative positions and presence masks for its immediate neighbors. These are **neighborhood-local observations**, not a centralized global-state input. Utilities are independent two-hidden-layer MLPs; the report formulates Markov local-state utilities and does not prescribe a recurrent architecture. This implementation does not claim recurrent policies or solve arbitrary partial observability.

The reward is computed from the **pre-transition** own state and action:

```text
r_i = -abs(position_i - goal_i) / (length - 1)
      + 0.5 * [position_i == goal_i] - 0.01 * [action_i != stay]
r_team = sum_i r_i
r_J = sum_{i in J} r_i
```

The reward is local even though dynamics depend on neighbor states. Agents maximize a common sum of rewards. Episodes continue after reaching the goal so agents learn to hold their positions. The fixed horizon is a true terminal of the finite-horizon task, and the remaining time is observed; terminal TD targets therefore do not bootstrap. Evaluation reports undiscounted team return mean/population standard deviation and the fraction with **all agents on goal at the final step**. `first_arrival_step_mean` uses `horizon + 1` for episodes that never jointly arrive. A high return can coexist with lower final success if a policy moves away on its last transition.

For every partition `J`, `F_J` consumes the **full vector of all agent utilities**, as the report's `F_J: R^n -> R` requires. A partition chooses the local reward target; it does not restrict the mixer to that partition's utility inputs. Softplus-constrained positive weights and monotonic ELU activations ensure `dF_J/dU_i >= 0` structurally throughout training. Thus independently greedy utilities maximize every mixed partition value, enabling centralized training with decentralized execution.

The learner uses epsilon-greedy exploration with a linear step schedule, replay minibatches, Double-DQN action selection, frozen periodically updated target networks, squared TD loss, Adam, and gradient clipping. Replay, target networks and Double DQN are explicit practical neural-training choices: the report does not provide these exact architecture/hyperparameter specifications. Its projected monotonic update is implemented through nonnegative parameterization instead of a numerical projection.

## Configuration and code map

Neural JSON layers in `configs/neural/` follow the report's order: **default < environment < algorithm < test**; nested dictionaries are merged and validated. `--episodes` is a final CLI override. Unknown keys, malformed partitions, invalid numeric values and overlapping training/evaluation reset-seed ranges are rejected. Changing agent count also requires partitions covering those agents exactly once.

| Report / workflow requirement | Implementation |
| --- | --- |
| Chapters 3–4: local utilities and monotonic partition functions | `marl/networks.py`: independent MLPs and full-vector `PartitionMixer` |
| Chapters 3–4: additive local and partition rewards | `marl/environment.py`, `NeuralLearner.reward_targets` |
| Chapter 4.1: decentralized greedy actions and exploration | `NeuralLearner.act`, epsilon schedule in `marl/learner.py` |
| Neural centralized TD training | `marl/learner.py`: replay, Double DQN, targets, terminal masking, optimizer |
| Chapter 5.1: four configuration layers | `marl/config.py`, `marl/neural_config.py`, `configs/neural/` |
| Run, resume, evaluate and persist | `marl/train.py`, `marl/evaluate.py` |
| Chapter 5 / Figure 5.2: available comparisons | Neural QMIX, IQL, IQL-local and VDN on the new task |
| Earlier tabular example | `marl/lomaq.py`, `run_demo.py`, original `configs/` preserved |

## Validation evidence

`python -m pytest -q` passes 24 tests, including actual neural learning on held-out reset seeds, every baseline's training/save/load/evaluation path, exact resume, Double-DQN target selection, terminal masking, additive reward locality, decentralized action independence, monotonic gradients for every utility, joint greedy consistency, replay persistence, and configuration validation. `python -m ruff check .` passes. GitHub Actions runs both commands and the original demo.

Recorded local CPU runs used Python 3.12.14, NumPy 2.5.3, Torch 2.14.1, Pytest 8.4.2 and Ruff 0.16.10; exact versions are in `requirements-tested.txt`. With `configs/neural/test/smoke.json`, each algorithm ran **250 episodes, 4,000 environment steps and 3,873 gradient updates**, training seed 7 and 32 evaluation reset seeds starting at 10000. The full fresh-run metrics and resolved configurations are in [`validation/smoke-results.json`](validation/smoke-results.json).

| Policy | Mean return | Episode return std | Final success |
| --- | ---: | ---: | ---: |
| Seeded random policy | -21.980 | 14.093 | 3.125% |
| Untrained local MLPs | -25.038 | 13.532 | 0% |
| LOMAQ | 14.333 | 4.852 | 100% |
| QMIX | 14.367 | 4.856 | 43.750% |
| IQL (team reward) | 14.244 | 4.839 | 56.250% |
| IQL-local | 14.246 | 4.841 | 100% |
| VDN | 14.222 | 4.842 | 100% |

These measurements validate executable learning and persistence on a small task. Reset seeds are disjoint from training; the finite task can still revisit the same states. They are not results from the report and do not establish comparative scientific performance. The learning smoke test requires LOMAQ to exceed both the untrained and random-policy return by at least 15, achieve at least 75% final success, perform at least 3,000 gradient updates, and preserve evaluation after loading.

## Authoritative report and remaining gaps

[`ExploReport.pdf`](ExploReport.pdf) is the only report in the current tree and Git history. It is the May 2023 report, originally added in commit `53e1c7e`; SHA-256 is `7838cb5fb0ed5a50d3fa5997a28cb11dfb7541a81cbec8787faf3afe706c774e`. The report is unchanged. The pre-existing `imp1` implementation added in `30a8554` contained only tabular utilities, linear partition mixers, a one-state payoff example and four tests. No original neural source, notebooks, experiment archives, checkpoints, raw logs or historical environment configurations exist in this repository's history.

The following reported results remain **unverified**:

- **Table 5.1 (multi-particle returns):** the exact scenario/environment implementation and version, agent count, local reward definition, graph and partitions, episode/training budgets, seeds, checkpoint and evaluation protocol are missing. This includes the table's QTRAN, IQL, IQL-local, QMIX and LOMAQ results.
- **Figure 5.1 (test 2 curve):** original run configuration, checkpoints and raw learning/evaluation logs are missing.
- **Figure 5.2 (coupled multi-cart-pole and bounded cooperative navigation):** the original environment code, cart coupling/dynamics/bounds, navigation rules, partitions, run configurations, seeds and raw returns are missing. The plot's VDN and LOMAQ+RD curves cannot be reconstructed from it.
- **Traffic control and payoff-matrix experiments:** the original traffic simulator/version, road network and control/reward settings, and original payoff matrices are absent. The existing `run_demo.py` matrix is explicitly illustrative.
- **QTRAN, COMA, GraphMIX and learned reward decomposition (LOMAQ+RD):** the report mentions comparisons but does not supply their implementation or enough experiment/architecture/training specifications to restore them. They are not implemented here. Dynamic graph discovery and automatic partition/decomposition learning remain future work in Chapter 6.

Restoring the historical artifacts is necessary before testing those results. The self-contained neural workflow and comparisons above run without them.

## Preserved tabular demonstration

```bash
python run_demo.py
python -m unittest discover -s tests -p test_lomaq.py -v
```

The demo and its original four `unittest` cases still require only the Python standard library. The full neural suite uses `pytest` and the dependencies above.

## Authors and reference

G. Unureddy, K. V. K. Varun, and V. N. V. S. S. Jayadithya — Department of Computer Science and Engineering, Indian Institute of Technology (Banaras Hindu University), Varanasi.

The project report is based on *Locality Matters: A Scalable Value Decomposition Approach for Cooperative Multi-Agent Reinforcement Learning* by R. Zohar, S. Mannor and G. Tennenholtz. This repository's new implementation follows the report's stated mathematical decomposition; its new task and engineering choices are documented above.
