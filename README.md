# Reward Decomposition in MARL - LOMAQ

This repository contains the project report and a small, runnable implementation of **LOMAQ** (Local Multi-Agent Q-learning), a value-based method for cooperative Multi-Agent Reinforcement Learning (MARL).

LOMAQ addresses multi-agent credit assignment by decomposing rewards and learning local, partition-based value functions. It follows the Centralized Training, Decentralized Execution (CTDE) paradigm: training can use joint information, while each agent acts from its local observations at execution time.

## Repository contents

- [`ExploReport.pdf`](ExploReport.pdf) - complete project report, including the method, experimental setup, results, and references.
- [`marl/lomaq.py`](marl/lomaq.py) - tabular local utilities and monotone linear partition mixers for discrete actions.
- [`marl/config.py`](marl/config.py) and [`configs/`](configs/) - default, environment, algorithm, and test configuration layers.
- [`run_demo.py`](run_demo.py) - an illustrative two-agent payoff-matrix example.

## Project scope

The report covers:

- reward decomposition for cooperative MARL;
- the LOMAQ algorithm and its monotonic value-decomposition assumption;
- comparisons with IQL, QMIX, QTRAN, COMA, and GraphMIX;
- experiments in multi-particle, coupled multi-cart-pole, bounded cooperative navigation, payoff-matrix, and traffic-signal-control environments.

The reported experiments found LOMAQ to perform particularly well in coupled multi-cart-pole and bounded cooperative-navigation settings, with lower return variability than selected baselines.

## Read the report

Open `ExploReport.pdf` in any PDF reader, or from a terminal on macOS:

```bash
open ExploReport.pdf
```

## Run the implementation

Python 3.10+ is sufficient; the demo uses only the standard library.

```bash
python run_demo.py
python -m unittest discover -s tests -v
```

`LOMAQ.update` takes one local observation and action per agent, a reward per partition, and the next local observations. Each partition value is a nonnegative weighted sum of agent utilities plus a bias, so its gradient with respect to each utility is nonnegative. The weights are projected back to nonnegative values after each temporal-difference update. `act` chooses each agent's action from its own utility table, with optional epsilon exploration; the joint transition and partition rewards are used during training.

The configuration loader applies default, environment, algorithm, then test JSON, with later values overriding earlier ones as the report specifies. The payoff matrix in `run_demo.py` is illustrative and was chosen for a smoke test; it is not the report's unpublished matrix.

## Authors

- G. Unureddy
- K. V. K. Varun
- V. N. V. S. S. Jayadithya

Department of Computer Science and Engineering  
Indian Institute of Technology (Banaras Hindu University), Varanasi

## Reference

The work is based on *Locality Matters: A Scalable Value Decomposition Approach for Cooperative Multi-Agent Reinforcement Learning* by R. Zohar, S. Mannor, and G. Tennenholtz.

## Status

The report's partition-based local rewards, monotone decomposition, decentralized greedy actions, and four configuration layers now have a small executable example. The report's original neural implementation, experiment configurations, environment code, trained models, and raw results were not committed. This implementation does not reproduce the reported multi-particle, coupled cart-pole, bounded navigation, or traffic-signal experiments, nor the IQL/QMIX/QTRAN/COMA/GraphMIX comparisons. The report's future work on learning dynamic dependency graphs is not specified as an implemented feature.
