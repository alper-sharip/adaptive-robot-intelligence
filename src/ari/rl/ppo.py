"""M17: Proximal Policy Optimization (Schulman et al., 2017), written out in full.

The loop:
  1. COLLECT   run the current policy for a few thousand steps; the policy is stochastic
               (a Gaussian around its preferred action), which is how it explores.
  2. CREDIT    for every step, estimate the ADVANTAGE: was the action better or worse than
               what the policy normally achieves from that state? Uses a learned value
               function V(s) and Generalized Advantage Estimation (GAE).
  3. IMPROVE   make better-than-usual actions more likely and worse ones less likely, with
               the probability ratio clipped to [1 - eps, 1 + eps] so one batch of noisy data
               cannot move the policy far. Then throw the data away and repeat.

Pieces a learned policy must provide (see GaussianPolicy and rl/residual.py):
    act(obs)                 -> env_action, raw_action, log_prob, value
    evaluate(obs, raw)       -> log_prob, entropy, value
Collection runs in persistent worker processes, one environment each.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, act=nn.Tanh, zero_last: bool = False) -> nn.Sequential:
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        lin = nn.Linear(a, b)
        last = i == len(sizes) - 2
        nn.init.orthogonal_(lin.weight, gain=(0.01 if last else np.sqrt(2)))
        nn.init.zeros_(lin.bias)
        if last and zero_last:
            nn.init.zeros_(lin.weight)
        layers += [lin] if last else [lin, act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Actor (mean of a Gaussian, state-independent std) and critic, with input normalization."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, log_std: float = -0.5,
                 fixed_norm: tuple | None = None):
        super().__init__()
        self.actor = mlp([obs_dim, hidden, hidden, act_dim])
        self.critic = mlp([obs_dim, hidden, hidden, 1])
        self.critic[-1].weight.data.mul_(100.0)          # undo the tiny init for the value head
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std)))
        mean, std = fixed_norm if fixed_norm is not None else (torch.zeros(obs_dim), torch.ones(obs_dim))
        self.register_buffer("obs_mean", torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer("obs_std", torch.as_tensor(std, dtype=torch.float32).clone())
        self.register_buffer("norm_count", torch.zeros(()))
        self.fixed_norm = fixed_norm is not None

    # -- normalization: statistics of every observation seen so far ------------------------
    def update_norm(self, obs: torch.Tensor) -> None:
        if self.fixed_norm:
            return
        n, m = self.norm_count.item(), len(obs)
        mean = (self.obs_mean * n + obs.sum(0)) / (n + m)
        var = (self.obs_std**2 * n + ((obs - mean) ** 2).sum(0)) / (n + m) if n > 0 else obs.var(0)
        self.obs_mean.copy_(mean)
        self.obs_std.copy_(var.sqrt().clamp_min(1e-3))
        self.norm_count.fill_(n + m)

    def _n(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    # -- the two hooks subclasses override --------------------------------------------------
    def mean(self, obs_n: torch.Tensor) -> torch.Tensor:
        return self.actor(obs_n)

    def to_env(self, obs: torch.Tensor, raw: torch.Tensor) -> np.ndarray:
        return raw.clamp(-1, 1).numpy()

    # -- PPO interface ------------------------------------------------------------------------
    def dist(self, obs):
        return torch.distributions.Normal(self.mean(self._n(obs)), self.log_std.exp())

    def value(self, obs):
        return self.critic(self._n(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        d = self.dist(o)
        raw = d.mean if deterministic else d.sample()
        return self.to_env(o, raw)[0], raw[0].numpy(), d.log_prob(raw).sum(-1).item(), self.value(o).item()

    def evaluate(self, obs, raw):
        d = self.dist(obs)
        return d.log_prob(raw).sum(-1), d.entropy().sum(-1), self.value(obs)


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Generalized Advantage Estimation (Schulman et al., 2016).

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)     one-step surprise
    A_t     = sum_k (gamma * lam)^k * delta_{t+k}    discounted sum of surprises
    lam = 0: trust the value function fully (low variance, biased);
    lam = 1: trust only observed rewards (unbiased, high variance).
    `dones[t]` is True when the episode ended at step t; nothing after it counts.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    nxt, running = last_value, 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * nxt * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
        nxt = values[t]
    return adv, adv + np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------- collection workers
def _worker(conn, build, build_args, worker_id):
    torch.set_num_threads(1)
    env, policy = build(worker_id=worker_id, **build_args)
    obs, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    while True:
        msg = conn.recv()
        if msg is None:
            break
        state, n_steps, gamma = msg
        policy.load_state_dict(state)
        buf = {k: [] for k in ("obs", "raw", "logp", "val", "rew", "done")}
        episodes = []
        for _ in range(n_steps):
            a_env, raw, logp, val = policy.act(obs)
            nobs, r, term, trunc, info = env.step(a_env)
            ep_ret, ep_len = ep_ret + r, ep_len + 1
            if trunc and not term:
                # Time limit: the episode was cut, not finished. Add the value of what would
                # have followed, so "running out of time" is not mistaken for "nothing more to gain".
                r += gamma * policy.value(torch.as_tensor(nobs, dtype=torch.float32).unsqueeze(0)).item()
            for k, v in zip(buf, (obs, raw, logp, val, r, term or trunc)):
                buf[k].append(v)
            obs = nobs
            if term or trunc:
                episodes.append({"return": ep_ret, "length": ep_len, **{k: float(v) for k, v in info.items()
                                                                        if isinstance(v, (bool, float, int))}})
                obs, _ = env.reset()
                ep_ret, ep_len = 0.0, 0
        last_value = policy.value(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
        conn.send(({k: np.asarray(v, dtype=np.float32) for k, v in buf.items()}, last_value, episodes))


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    steps_per_worker: int = 2048
    workers: int = 2
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 10
    minibatch: int = 256
    lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03      # stop an update early if the policy has moved this far
    critic_warmup_iters: int = 0        # iterations that train ONLY the value function
    seed: int = 0
    log: list = field(default_factory=list)


def train(build, build_args: dict, cfg: PPOConfig, on_iteration=None, verbose: bool = True):
    """build(worker_id=..., **build_args) -> (env, policy); must be importable (used in subprocesses)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    _, policy = build(worker_id=-1, **build_args)
    actor_params = [p for n, p in policy.named_parameters() if not n.startswith("critic")]
    opt_actor = torch.optim.Adam(actor_params, lr=cfg.lr, eps=1e-5)
    opt_critic = torch.optim.Adam(policy.critic.parameters(), lr=cfg.critic_lr, eps=1e-5)

    ctx = get_context("spawn")
    pipes, procs = [], []
    for w in range(cfg.workers):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(child, build, build_args, w), daemon=True)
        p.start()
        pipes.append(parent), procs.append(p)

    steps, it, t0 = 0, 0, time.time()
    recent = []
    try:
        while steps < cfg.total_steps:
            it += 1
            state = {k: v.clone() for k, v in policy.state_dict().items()}
            for c in pipes:
                c.send((state, cfg.steps_per_worker, cfg.gamma))
            O, R, LP, ADV, RET, eps = [], [], [], [], [], []
            for c in pipes:
                buf, last_value, episodes = c.recv()
                adv, ret = gae(buf["rew"], buf["val"], buf["done"], last_value, cfg.gamma, cfg.lam)
                O.append(buf["obs"]), R.append(buf["raw"]), LP.append(buf["logp"]), ADV.append(adv), RET.append(ret)
                eps += episodes
            obs, raw = torch.as_tensor(np.concatenate(O)), torch.as_tensor(np.concatenate(R))
            logp_old, adv, ret = (torch.as_tensor(np.concatenate(x)) for x in (LP, ADV, RET))
            steps += len(obs)
            policy.update_norm(obs)

            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
            train_actor = it > cfg.critic_warmup_iters
            kl = 0.0
            for epoch in range(cfg.epochs):
                perm = torch.randperm(len(obs))
                for i in range(0, len(obs), cfg.minibatch):
                    b = perm[i:i + cfg.minibatch]
                    logp, ent, val = policy.evaluate(obs[b], raw[b])
                    v_loss = ((val - ret[b]) ** 2).mean()
                    opt_critic.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(policy.critic.parameters(), cfg.max_grad_norm)
                    opt_critic.step()
                    if train_actor:
                        ratio = (logp - logp_old[b]).exp()
                        surr = torch.min(ratio * adv_n[b], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv_n[b])
                        loss = -surr.mean() - cfg.entropy_coef * ent.mean()
                        opt_actor.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
                        opt_actor.step()
                with torch.no_grad():
                    kl = (logp_old - policy.evaluate(obs, raw)[0]).mean().item()
                if train_actor and cfg.target_kl is not None and kl > cfg.target_kl:
                    break

            recent = (recent + eps)[-100:]
            row = {"iteration": it, "steps": steps, "episodes": len(eps), "kl": kl,
                   "value_loss": v_loss.item(), "minutes": (time.time() - t0) / 60,
                   "return": float(np.mean([e["return"] for e in recent])) if recent else float("nan"),
                   "length": float(np.mean([e["length"] for e in recent])) if recent else float("nan"),
                   "success": float(np.mean([e.get("is_success", 0.0) for e in recent])) if recent else float("nan"),
                   "std": policy.log_std.exp().mean().item()}
            for key in ("lifted", "closest_to_cube"):          # task-specific extras, when the env reports them
                if recent and key in recent[0]:
                    row[key] = float(np.mean([e[key] for e in recent]))
            cfg.log.append(row)
            if verbose:
                print(f"it {it:3d}  steps {steps:7d}  return {row['return']:7.2f}  success {100 * row['success']:5.1f}%  "
                      f"len {row['length']:5.1f}  kl {kl:.4f}  std {row['std']:.3f}  {row['minutes']:.1f} min", flush=True)
            if on_iteration is not None:
                on_iteration(policy, row)
    finally:
        for c in pipes:
            c.send(None)
        for p in procs:
            p.join(timeout=5)
    return policy
