"""Neural utilities and structurally monotonic partition/QMIX mixers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AgentUtilities(nn.Module):
    """Separate local MLPs: no joint state or other utility enters an action head."""

    def __init__(self, n_agents: int, obs_dim: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.agents = nn.ModuleList([
            nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, n_actions))
            for _ in range(n_agents)
        ])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.stack([agent(observations[:, i])
                            for i, agent in enumerate(self.agents)], dim=1)


class PositiveLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Approximately unit row sums, avoiding huge initial mixed Q values.
        initial = torch.full((output_dim, input_dim), 1 / input_dim)
        self.raw_weight = nn.Parameter(torch.log(torch.expm1(initial)))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, F.softplus(self.raw_weight), self.bias)


class PartitionMixer(nn.Module):
    """One F_J: R^n -> R per partition; ALL n utilities enter every F_J.

    Nonnegative weights and increasing ELU give dF_J/dU_i >= 0 at every update.
    A partition selects the REWARD TARGET, not the set of utility inputs.
    """

    def __init__(self, n_agents: int, n_partitions: int, hidden_dim: int):
        super().__init__()
        self.mixers = nn.ModuleList([
            nn.Sequential(PositiveLinear(n_agents, hidden_dim), nn.ELU(),
                          PositiveLinear(hidden_dim, 1))
            for _ in range(n_partitions)
        ])

    def forward(self, utilities: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        return torch.cat([mixer(utilities) for mixer in self.mixers], dim=-1)


class QMixMixer(nn.Module):
    """QMIX-style two-layer mixer with state-conditioned positive weights."""

    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.w2 = nn.Linear(state_dim, hidden_dim)
        self.b1 = nn.Linear(state_dim, hidden_dim)
        self.b2 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, utilities: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        weight1 = F.softplus(self.w1(state)).view(-1, self.n_agents, self.hidden_dim)
        weight2 = F.softplus(self.w2(state)).view(-1, self.hidden_dim, 1)
        hidden = F.elu(torch.bmm(utilities.unsqueeze(1), weight1)
                       + self.b1(state).unsqueeze(1))
        return (torch.bmm(hidden, weight2).squeeze(1) + self.b2(state))


class ValueSystem(nn.Module):
    def __init__(self, config: dict, obs_dim: int, state_dim: int, n_actions: int):
        super().__init__()
        env, alg = config["environment"], config["algorithm"]
        self.utilities = AgentUtilities(env["n_agents"], obs_dim, n_actions, alg["hidden_dim"])
        self.name = alg["name"]
        if self.name == "lomaq":
            self.mixer = PartitionMixer(env["n_agents"], len(alg["partitions"]), alg["mixer_dim"])
        elif self.name == "qmix":
            self.mixer = QMixMixer(env["n_agents"], state_dim, alg["mixer_dim"])
        else:
            self.mixer = None

    def selected_values(self, observations: torch.Tensor, state: torch.Tensor,
                        actions: torch.Tensor) -> torch.Tensor:
        chosen = self.utilities(observations).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        if self.name == "vdn":
            return chosen.sum(dim=-1, keepdim=True)
        return self.mixer(chosen, state) if self.mixer is not None else chosen
