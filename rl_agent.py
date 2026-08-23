"""
rl_agent.py
===========
Distributional RL agents for the S-SM crush spread strategy.

Classes
-------
ReplayBuffer   – uniform experience replay
IQNAgent       – Implicit Quantile Network (main algorithm)
QRDQNAgent     – Quantile Regression DQN  (comparison)
DQNAgent       – Standard DQN             (comparison)

IQN reference: Dabney et al. (2018) "Implicit Quantile Networks for
Distributional Reinforcement Learning"
CVaR action selection: sample τ from U[0, α] instead of U[0,1],
giving a risk-averse policy without modifying the training target.
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========================================================================
# REPLAY BUFFER
# =========================================================================

class ReplayBuffer:
    """Fixed-capacity uniform experience replay buffer."""

    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self.pos      = 0
        self.size     = 0

        self.states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        i = self.pos % self.capacity
        self.states[i]      = state
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = float(done)
        self.pos  += 1
        self.size  = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.tensor(x, device=self.device)
        return (
            to_t(self.states[idx]),
            to_t(self.actions[idx]),
            to_t(self.rewards[idx]),
            to_t(self.next_states[idx]),
            to_t(self.dones[idx]),
        )

    def __len__(self):
        return self.size


# =========================================================================
# IQN NETWORK
# =========================================================================

class IQNNet(nn.Module):
    """
    Implicit Quantile Network.

    forward(state, tau) → Q of shape (B, N, n_actions)
      state : (B, state_dim)
      tau   : (B, N)  quantile levels in (0, 1)
    """

    def __init__(self, state_dim: int, n_actions: int,
                 hidden_dim: int = 128, n_cos: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_cos      = n_cos
        self.n_actions  = n_actions

        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

        # Quantile embedding: cos features → hidden_dim
        self.tau_embed = nn.Linear(n_cos, hidden_dim)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        B, N = tau.shape

        # Encode state: (B, hidden_dim)
        feat = self.encoder(state)

        # Cosine quantile embedding: (B, N, n_cos) → (B, N, hidden_dim)
        i_pi = (torch.arange(1, self.n_cos + 1,
                              device=tau.device, dtype=torch.float32) * np.pi)
        cos_feat = torch.cos(tau.unsqueeze(-1) * i_pi)          # (B, N, n_cos)
        tau_emb  = torch.relu(self.tau_embed(cos_feat))          # (B, N, hidden_dim)

        # Element-wise product then decode
        feat_exp = feat.unsqueeze(1).expand(B, N, self.hidden_dim)
        Q = self.head(feat_exp * tau_emb)                        # (B, N, n_actions)
        return Q

    @torch.no_grad()
    def q_mean(self, state: torch.Tensor,
               N: int = 32, cvar_alpha: float = 1.0,
               deterministic: bool = False) -> torch.Tensor:
        """Mean Q-values over N quantile samples; CVaR when cvar_alpha < 1."""
        B   = state.shape[0]
        if deterministic:
            # Midpoints of N equal-probability bins give reproducible
            # validation/test decisions while still approximating the integral.
            grid = (torch.arange(N, device=state.device, dtype=torch.float32)
                    + 0.5) / N
            tau = grid.unsqueeze(0).expand(B, N) * cvar_alpha
        else:
            tau = torch.rand(B, N, device=state.device) * cvar_alpha
        return self(state, tau).mean(dim=1)                      # (B, n_actions)


def _quantile_huber_loss(predicted: torch.Tensor,
                         tau:       torch.Tensor,
                         targets:   torch.Tensor,
                         kappa:     float = 1.0) -> torch.Tensor:
    """
    Huber quantile regression loss.

    predicted : (B, N, 1)  Z(s, a, τ_i) for chosen action
    tau       : (B, N)     quantile levels
    targets   : (B, Np)    Bellman targets
    """
    B, N, _ = predicted.shape
    Np = targets.shape[1]

    u   = targets.unsqueeze(1) - predicted          # (B, N, Np) via broadcast
    tau = tau.unsqueeze(2)                          # (B, N, 1)

    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa,
                        0.5 * u.pow(2),
                        kappa * (abs_u - 0.5 * kappa))
    rho = (tau - (u.detach() < 0).float()).abs() * huber / kappa
    return rho.mean(dim=2).sum(dim=1).mean()


# =========================================================================
# IQN AGENT
# =========================================================================

class IQNAgent:

    def __init__(self, state_dim: int, n_actions: int = 3,
                 hidden_dim: int = 128, n_cos: int = 64,
                 lr: float = 3e-4, gamma: float = 0.99,
                 n_tau_train: int = 64, n_tau_target: int = 32,
                 cvar_alpha: float = 1.0,
                 device: str = "cpu"):
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.n_tau_train  = n_tau_train
        self.n_tau_target = n_tau_target
        self.cvar_alpha   = cvar_alpha
        self.device       = torch.device(device)

        self.online = IQNNet(state_dim, n_actions, hidden_dim, n_cos).to(self.device)
        self.target = IQNNet(state_dim, n_actions, hidden_dim, n_cos).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float = 0.0,
            deterministic: bool = False) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        q = self.online.q_mean(s, N=self.n_tau_target,
                               cvar_alpha=self.cvar_alpha,
                               deterministic=deterministic)
        return int(q.argmax(dim=1).item())

    def update(self, batch, kappa: float = 1.0) -> float:
        states, actions, rewards, next_states, dones = batch
        B = states.shape[0]

        # ---- Double DQN targets (no grad) ----
        with torch.no_grad():
            # action selection: online net. Must use the SAME cvar_alpha as
            # act() -- otherwise a risk-averse agent (cvar_alpha<1) would
            # bootstrap its target off a risk-NEUTRAL action choice,
            # inconsistent with the policy it actually executes.
            a_prime = self.online.q_mean(next_states, N=self.n_tau_target,
                                          cvar_alpha=self.cvar_alpha).argmax(1)  # (B,)

            # action evaluation: target net
            tau_p   = torch.rand(B, self.n_tau_target, device=self.device)
            Q_t     = self.target(next_states, tau_p)                     # (B, Np, na)
            idx_p   = a_prime.view(B, 1, 1).expand(B, self.n_tau_target, 1)
            Q_sa_p  = Q_t.gather(2, idx_p).squeeze(2)                    # (B, Np)

            targets = (rewards.unsqueeze(1) +
                       self.gamma * (1 - dones.unsqueeze(1)) * Q_sa_p)   # (B, Np)

        # ---- Online net ----
        tau   = torch.rand(B, self.n_tau_train, device=self.device)
        Q_on  = self.online(states, tau)                                  # (B, N, na)
        a_idx = actions.view(B, 1, 1).expand(B, self.n_tau_train, 1)
        Q_sa  = Q_on.gather(2, a_idx)                                     # (B, N, 1)

        loss = _quantile_huber_loss(Q_sa, tau, targets, kappa)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"online": self.online.state_dict(),
                    "optim":  self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.sync_target()
        self.optimizer.load_state_dict(ckpt["optim"])


# =========================================================================
# QR-DQN AGENT
# =========================================================================

class QRDQNNet(nn.Module):
    """Quantile Regression DQN with N fixed quantiles."""

    def __init__(self, state_dim: int, n_actions: int,
                 hidden_dim: int = 128, n_quantiles: int = 32):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_actions   = n_actions
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions * n_quantiles),
        )
        # Fixed quantile levels: (2i-1)/(2N) for i=1..N
        tau = (2 * torch.arange(1, n_quantiles + 1) - 1) / (2 * n_quantiles)
        self.register_buffer("tau", tau.float())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        out = self.net(state).view(B, self.n_quantiles, self.n_actions)
        return out   # (B, N, n_actions)

    @torch.no_grad()
    def q_mean(self, state: torch.Tensor) -> torch.Tensor:
        return self(state).mean(dim=1)   # (B, n_actions)


class QRDQNAgent:

    def __init__(self, state_dim: int, n_actions: int = 3,
                 hidden_dim: int = 128, n_quantiles: int = 32,
                 lr: float = 3e-4, gamma: float = 0.99,
                 device: str = "cpu"):
        self.gamma     = gamma
        self.n_actions = n_actions
        self.device    = torch.device(device)

        self.online = QRDQNNet(state_dim, n_actions, hidden_dim, n_quantiles).to(self.device)
        self.target = QRDQNNet(state_dim, n_actions, hidden_dim, n_quantiles).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float = 0.0, **_) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return int(self.online.q_mean(s).argmax(1).item())

    def update(self, batch, kappa: float = 1.0) -> float:
        states, actions, rewards, next_states, dones = batch
        B = states.shape[0]
        N = self.online.n_quantiles

        with torch.no_grad():
            # Double DQN: online selects action, target evaluates value
            a_prime  = self.online.q_mean(next_states).argmax(1)         # (B,)
            Q_t      = self.target(next_states)                           # (B, N, na)
            idx_p    = a_prime.view(B, 1, 1).expand(B, N, 1)
            Q_sa_p   = Q_t.gather(2, idx_p).squeeze(2)                   # (B, N)
            targets  = (rewards.unsqueeze(1) +
                        self.gamma * (1 - dones.unsqueeze(1)) * Q_sa_p)  # (B, N)

        Q_on  = self.online(states)                                       # (B, N, na)
        a_idx = actions.view(B, 1, 1).expand(B, N, 1)
        Q_sa  = Q_on.gather(2, a_idx)                                     # (B, N, 1)

        tau   = self.online.tau.view(1, N, 1).expand(B, N, 1)
        loss  = _quantile_huber_loss(Q_sa, tau.squeeze(2), targets, kappa)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"online": self.online.state_dict(),
                    "optim":  self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.sync_target()
        self.optimizer.load_state_dict(ckpt["optim"])


# =========================================================================
# STANDARD DQN AGENT
# =========================================================================

class DQNNet(nn.Module):

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:

    def __init__(self, state_dim: int, n_actions: int = 3,
                 hidden_dim: int = 128,
                 lr: float = 3e-4, gamma: float = 0.99,
                 device: str = "cpu"):
        self.gamma     = gamma
        self.n_actions = n_actions
        self.device    = torch.device(device)

        self.online = DQNNet(state_dim, n_actions, hidden_dim).to(self.device)
        self.target = DQNNet(state_dim, n_actions, hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float = 0.0, **_) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        return int(self.online(s).argmax(1).item())

    def update(self, batch, **_) -> float:
        states, actions, rewards, next_states, dones = batch
        with torch.no_grad():
            # Double DQN: online selects action, target evaluates value
            a_prime  = self.online(next_states).argmax(1)
            q_next   = self.target(next_states).gather(
                           1, a_prime.unsqueeze(1)).squeeze(1)
            targets  = rewards + self.gamma * (1 - dones) * q_next

        q_cur = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss  = F.smooth_l1_loss(q_cur, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"online": self.online.state_dict(),
                    "optim":  self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.sync_target()
        self.optimizer.load_state_dict(ckpt["optim"])


# =========================================================================
# FACTORY
# =========================================================================

def make_agent(agent_type: str, state_dim: int, cfg: dict):
    """Create an agent from a config dict."""
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    common = dict(state_dim=state_dim, n_actions=3,
                  hidden_dim=cfg["hidden_dim"], lr=cfg["lr"],
                  gamma=cfg["gamma"], device=device)
    if agent_type == "IQN":
        return IQNAgent(**common,
                        n_cos=cfg.get("n_cos", 64),
                        n_tau_train=cfg["n_quantiles"],
                        n_tau_target=cfg["n_quantiles"] // 2,
                        cvar_alpha=cfg.get("cvar_alpha", 1.0))
    elif agent_type == "QRDQN":
        return QRDQNAgent(**common,
                          n_quantiles=cfg.get("qrdqn_n_quantiles", 32))
    elif agent_type == "DQN":
        return DQNAgent(**{**common,
                           "hidden_dim": cfg.get("dqn_hidden_dim",
                                                 cfg["hidden_dim"])})
    else:
        raise ValueError(f"Unknown agent_type '{agent_type}'")
