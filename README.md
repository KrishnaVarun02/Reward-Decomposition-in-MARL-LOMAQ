# Reward Decomposition in MARL — LOMAQ

An undergraduate research-oriented study of **cooperative multi-agent reinforcement learning**, structural credit assignment, monotonic value decomposition, and decentralized execution.

The repository contains the project report together with a runnable PyTorch implementation, controlled comparisons, automated validation, and reproducibility artifacts. The executable experiments are explicitly separated from numerical results in the report that cannot be reconstructed from the preserved historical artifacts.

## Research question

How can a cooperative multi-agent value-decomposition system use decomposable rewards while preserving the monotonic structure needed for decentralized greedy action selection?

## Methodology

- Independent per-agent MLP utility networks map local observations to action utilities.
- A monotonic mixer is defined for each reward partition. Every mixer receives the full vector of agent utilities while the partition determines the additive reward target.
- Positive parameterization and increasing ELU activations enforce nonnegative sensitivity to utility inputs.
- Centralized training uses replay, Double-DQN target selection, frozen target networks, Adam optimization, and gradient clipping.
- Decentralized execution uses only each agent's own observation and utility network; the centralized mixer is not used to select actions.
- The **Linked Navigation** environment contains three coupled agents moving toward a shared target with neighborhood-local observations and a finite horizon.
- QMIX, VDN, IQL, and IQL-local provide comparison methods under matched environment, network, optimizer, exploration, and evaluation settings.

## Experiments / Evaluation

The repository supports configurable experiments, deterministic CPU execution, disjoint training/evaluation reset seeds, greedy evaluation after checkpoint reload, exact checkpoint persistence, interrupted/resumed training checks, and automated behavioral tests.

The validation suite tests reward locality, decentralized action independence, monotonic gradients, terminal masking, Double-DQN target selection, persistence, and configuration validation.

A controlled smoke evaluation uses **250 training episodes, 4,000 environment steps, 3,873 gradient updates, and 32 evaluation reset seeds**.

## Results / Findings

| Policy | Mean return | Episode return std | Final joint-goal success |
| --- | ---: | ---: | ---: |
| Seeded random policy | -21.980 | 14.093 | 3.125% |
| Untrained local MLPs | -25.038 | 13.532 | 0% |
| LOMAQ | **14.333** | 4.852 | **100%** |
| QMIX | 14.367 | 4.856 | 43.750% |
| IQL | 14.244 | 4.839 | 56.250% |
| IQL-local | 14.246 | 4.841 | 100% |
| VDN | 14.222 | 4.842 | 100% |

These measurements demonstrate executable learning and validation on the repository's compact task. They do **not** establish general comparative superiority across MARL benchmarks.

## Scope and limitations

The current neural experiments use a small custom Linked Navigation environment rather than a standard large-scale benchmark. Broader performance claims would require additional benchmark environments and independent runs.

The preserved project report contains additional experiments whose environments, checkpoints, configurations, and raw logs are not available in the current repository. Those numerical results are not presented here as reproduced results.

## Technical details

- marl/networks.py — local utility networks and monotonic mixers
- marl/environment.py — Linked Navigation environment
- marl/learner.py — centralized TD learning and checkpointing
- marl/train.py and marl/evaluate.py — training and evaluation workflows
- configs/neural/ — layered experiment configurations
- tests/ — behavioral and reproducibility tests
- validation/ — recorded evaluation evidence
- ExploReport.pdf — preserved project report

## Reproduce

Use Python 3.10+:

~~~
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

python -m marl.train --test-config configs/neural/test/smoke.json --output runs/lomaq
python -m marl.evaluate --checkpoint runs/lomaq/checkpoint.pt --episodes 100 --seed 20000 --output runs/lomaq/evaluation.json

python -m pytest -q
python -m ruff check .
~~~

## Academic context

The project is part of undergraduate academic work in Computer Science and Engineering at **IIT (BHU) Varanasi**, under the guidance of **Dr. Lakshmanan Kailasam**.
