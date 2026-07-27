# Reward Decomposition in MARL - LOMAQ

This repository contains the project report for an exploration of **LOMAQ** (Local Multi-Agent Q-learning), a value-based method for cooperative Multi-Agent Reinforcement Learning (MARL).

LOMAQ addresses multi-agent credit assignment by decomposing rewards and learning local, partition-based value functions. It follows the Centralized Training, Decentralized Execution (CTDE) paradigm: training can use joint information, while each agent acts from its local observations at execution time.

## Repository contents

- [`ExploReport.pdf`](ExploReport.pdf) - complete 2023 project report, including the method, experimental setup, results, and references.

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

## Authors

- G. Unureddy
- K. V. K. Varun
- V. N. V. S. S. Jayadithya

Department of Computer Science and Engineering  
Indian Institute of Technology (Banaras Hindu University), Varanasi

## Reference

The work is based on *Locality Matters: A Scalable Value Decomposition Approach for Cooperative Multi-Agent Reinforcement Learning* by R. Zohar, S. Mannor, and G. Tennenholtz.

## Status

At present, this repository archives the report only; it does not include the experiment source code, environment configurations, or trained models described in the document.
