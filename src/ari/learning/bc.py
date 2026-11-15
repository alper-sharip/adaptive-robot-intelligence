"""M14: behaviour cloning in plain PyTorch.

Behaviour cloning is supervised learning: given demonstrations (observation, expert action),
fit a network so that policy(observation) ~= expert action. Nothing about the task's goal
enters the loss; the network only imitates.

    motion outputs (dx, dy, dz, dyaw) : regression, mean-squared error on NORMALIZED actions
    gripper output                    : classification (open/closed), binary cross-entropy

Normalization matters: positions are ~0.5 (metres), deltas ~0.005. Without rescaling every
input and output to zero mean and unit variance, the tiny deltas would be ignored.

Train/validation split is by EPISODE, not by sample: neighbouring samples of one episode are
nearly identical, so a per-sample split would leak the answers into validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ari.learning.spaces import ACT_DIM, OBS_DIM


class MLP(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 256, layers: int = 3):
        super().__init__()
        dims = [obs_dim] + [hidden] * layers
        body = []
        for a, b in zip(dims[:-1], dims[1:]):
            body += [nn.Linear(a, b), nn.ReLU()]
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(hidden, act_dim)   # 4 motion values + 1 gripper logit

    def forward(self, obs):
        return self.head(self.body(obs))


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden: int = 256
    layers: int = 3
    val_fraction: float = 0.1
    gripper_weight: float = 1.0
    seed: int = 0
    history: list = field(default_factory=list)


def split_by_episode(episode_index: np.ndarray, val_fraction: float, seed: int):
    eps = np.unique(episode_index)
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_val = max(1, int(round(len(eps) * val_fraction)))
    val = np.isin(episode_index, eps[:n_val])
    return ~val, val


def train(data: dict, cfg: TrainConfig | None = None, verbose: bool = True):
    """Returns (policy, cfg). cfg.history holds per-epoch train/validation losses."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    obs = torch.as_tensor(data["observation.state"], dtype=torch.float32)
    act = torch.as_tensor(data["action"], dtype=torch.float32)
    tr, va = split_by_episode(data["episode_index"], cfg.val_fraction, cfg.seed)
    tr, va = torch.as_tensor(tr), torch.as_tensor(va)

    # Normalization statistics from the TRAINING split only.
    obs_mean, obs_std = obs[tr].mean(0), obs[tr].std(0).clamp_min(1e-6)
    act_mean, act_std = act[tr, :4].mean(0), act[tr, :4].std(0).clamp_min(1e-6)

    net = MLP(obs.shape[1], act.shape[1], cfg.hidden, cfg.layers)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    def losses(mask_or_idx):
        o = (obs[mask_or_idx] - obs_mean) / obs_std
        a = act[mask_or_idx]
        out = net(o)
        return mse(out[:, :4], (a[:, :4] - act_mean) / act_std), bce(out[:, 4], a[:, 4])

    idx_tr = torch.nonzero(tr).squeeze(1)
    g = torch.Generator().manual_seed(cfg.seed)
    cfg.history = []
    for epoch in range(cfg.epochs):
        net.train()
        perm = idx_tr[torch.randperm(len(idx_tr), generator=g)]
        for i in range(0, len(perm), cfg.batch_size):
            lm, lg = losses(perm[i:i + cfg.batch_size])
            opt.zero_grad()
            (lm + cfg.gripper_weight * lg).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            (tm, tg), (vm, vg) = losses(tr), losses(va)
        cfg.history.append(dict(epoch=epoch + 1, train_motion=tm.item(), train_gripper=tg.item(),
                                val_motion=vm.item(), val_gripper=vg.item()))
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:3d}  motion MSE train {tm:.4f} val {vm:.4f} | "
                  f"gripper BCE train {tg:.4f} val {vg:.4f}")
    return BCPolicy(net, obs_mean, obs_std, act_mean, act_std), cfg


class BCPolicy:
    """Wraps the network with its normalization: obs (numpy) -> action (numpy)."""

    def __init__(self, net, obs_mean, obs_std, act_mean, act_std):
        self.net = net.eval()
        self.obs_mean, self.obs_std, self.act_mean, self.act_std = obs_mean, obs_std, act_mean, act_std

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        o = (torch.as_tensor(obs, dtype=torch.float32) - self.obs_mean) / self.obs_std
        out = self.net(o.unsqueeze(0))[0]
        motion = out[:4] * self.act_std + self.act_mean
        return np.concatenate([motion.numpy(), [float(torch.sigmoid(out[4]) > 0.5)]])

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        lin = [m for m in self.net.body if isinstance(m, nn.Linear)]
        torch.save({"state_dict": self.net.state_dict(), "obs_mean": self.obs_mean, "obs_std": self.obs_std,
                    "act_mean": self.act_mean, "act_std": self.act_std,
                    "obs_dim": lin[0].in_features, "hidden": lin[0].out_features, "layers": len(lin)}, path)

    @classmethod
    def load(cls, path: Path) -> "BCPolicy":
        ck = torch.load(path, weights_only=True)
        net = MLP(ck["obs_dim"], ACT_DIM, ck["hidden"], ck["layers"])
        net.load_state_dict(ck["state_dict"])
        return cls(net, ck["obs_mean"], ck["obs_std"], ck["act_mean"], ck["act_std"])
