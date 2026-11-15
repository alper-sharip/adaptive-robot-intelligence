"""M19: one network from (instruction text, scene) to gripper actions.

    words --GRU--> sentence vector --> two queries: "which cube?"  "which target?"
    each object carries a colour tag;  attention = softmax(query . tag embedding)
    attended cube pose + attended target position + gripper state --> MLP --> action

Training signal: behaviour cloning ONLY (the reactive expert's actions). The network is
never told which object an instruction refers to. If the attention ends up on the right
cube, it learned what "the blue cube" means from watching what the expert did.

Object slots are shuffled in every training batch, so "slot 2" means nothing; the only way
to find the blue cube is to match the word against the colour tags.

At run time the attention is made hard (the single best-matching object is used): a soft
average that puts 1% of its weight on a cube 15 cm away would shift the grasp by 1.5 mm,
and Stage 4 showed millimetres matter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ari.envs.multi_object import CUBE_COLORS, TARGET_COLORS
from ari.language import grammar
from ari.language.parser import GRUParser
from ari.learning.spaces import tcp_yaw

COLOR_IDS = {c: i for i, c in enumerate(sorted(set(CUBE_COLORS) | set(TARGET_COLORS)))}
CUBE_TAGS = np.array([COLOR_IDS[c] for c in CUBE_COLORS])
TARGET_TAGS = np.array([COLOR_IDS[c] for c in TARGET_COLORS])


def scene_obs(env) -> dict:
    """What the language policy observes: gripper state and EVERY cube and target."""
    T = env.tcp_pose()
    yaw = tcp_yaw(T)
    cubes = []
    for i in range(len(CUBE_COLORS)):
        p, a = env.cube_position(i), env.cube_yaw_of(i)
        cubes.append([*p, np.sin(4 * a), np.cos(4 * a)])
    return {"proprio": np.array([*T[:3, 3], np.sin(yaw), np.cos(yaw), env.finger_width()], dtype=np.float32),
            "cubes": np.array(cubes, dtype=np.float32),
            "targets": np.array([env.target_position(j)[:2] for j in range(len(TARGET_COLORS))], dtype=np.float32)}


class LanguagePolicy(nn.Module):
    def __init__(self, hidden: int = 256, tag_dim: int = 16):
        super().__init__()
        self.text = GRUParser()                       # reuse the encoder architecture; its heads are unused
        self.tags = nn.Embedding(len(COLOR_IDS), tag_dim)
        # Attention must START near uniform. With large or unnormalized sentence vectors the
        # softmax begins saturated on arbitrary objects and passes almost no gradient to fix
        # itself: a pretrained text encoder (vector norm 9) converged to a confidently WRONG,
        # consistent target assignment, 0% correct (devlog.md). Hence: normalize the sentence
        # vector, small initial query weights, scale starting at 1.
        self.norm = nn.LayerNorm(128)
        self.q_cube, self.q_target = nn.Linear(128, tag_dim), nn.Linear(128, tag_dim)
        for q in (self.q_cube, self.q_target):
            q.weight.data.mul_(0.1)
            q.bias.data.zero_()
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.mlp = nn.Sequential(nn.Linear(20, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 5))
        for name, n in (("obs_mean", 20), ("obs_std", 20), ("act_mean", 4), ("act_std", 4)):
            self.register_buffer(name, torch.ones(n) if name.endswith("std") else torch.zeros(n))

    def attention(self, tokens, cube_tags, target_tags):
        h = self.norm(self.text.encode(tokens))
        s = self.log_scale.exp()
        ac = torch.softmax(s * torch.einsum("bd,bnd->bn", self.q_cube(h), self.tags(cube_tags)), dim=1)
        at = torch.softmax(s * torch.einsum("bd,bnd->bn", self.q_target(h), self.tags(target_tags)), dim=1)
        return ac, at

    def forward(self, tokens, proprio, cubes, targets, cube_tags, target_tags, hard: bool = False):
        ac, at = self.attention(tokens, cube_tags, target_tags)
        if hard:
            ac = torch.nn.functional.one_hot(ac.argmax(1), ac.shape[1]).float()
            at = torch.nn.functional.one_hot(at.argmax(1), at.shape[1]).float()
        cube = (ac.unsqueeze(-1) * cubes).sum(1)             # (B, 5): position, sin 4a, cos 4a
        tgt = (at.unsqueeze(-1) * targets).sum(1)            # (B, 2)
        p = proprio[:, :3]
        # The same 20 numbers as Stage 4's observation, built from the ATTENDED objects.
        obs = torch.cat([proprio, cube, tgt, cube[:, :3] - p, tgt - p[:, :2], cube[:, :2] - tgt], dim=1)
        return self.mlp((obs - self.obs_mean) / self.obs_std), ac, at


# ---------------------------------------------------------------- data
def collect_chunk(args):
    """Reactive expert in the multi-object scene, one instructed (cube, target) pair per episode."""
    seeds, = args
    from ari.envs.multi_object import MultiObjectEnv
    from ari.learning import demos
    from ari.skills.reactive_pick_place import ReactiveExpert

    env, expert = MultiObjectEnv(), ReactiveExpert()
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 900_000)
        cube, target = grammar.TRAIN_PAIRS[rng.integers(len(grammar.TRAIN_PAIRS))]
        env.reset(seed)
        env.focus(cube, target)
        ex = demos.ActionExecutor(env, joint_interp=True)
        rows = []

        def step(_):
            a = expert(env)
            rows.append((scene_obs(env), a))
            return a

        ok, _ = demos.run_policy(env, ex, step, max_time=15.0)
        if ok:
            out.append({"proprio": np.stack([r[0]["proprio"] for r in rows]),
                        "cubes": np.stack([r[0]["cubes"] for r in rows]),
                        "targets": np.stack([r[0]["targets"] for r in rows]),
                        "action": np.stack([r[1] for r in rows]),
                        "cube": CUBE_COLORS.index(cube), "target": TARGET_COLORS.index(target), "seed": seed})
    return out


def collect(n: int, start_seed: int = 40_000, workers: int = 2) -> dict:
    from ari.learning.evaluate import _pmap

    seeds = list(range(start_seed, start_seed + n))
    eps = [e for c in _pmap(collect_chunk, [(seeds[i::workers * 2],) for i in range(workers * 2)], workers) for e in c]
    eps.sort(key=lambda e: e["seed"])
    lens = [len(e["action"]) for e in eps]
    return {"proprio": np.concatenate([e["proprio"] for e in eps]), "cubes": np.concatenate([e["cubes"] for e in eps]),
            "targets": np.concatenate([e["targets"] for e in eps]), "action": np.concatenate([e["action"] for e in eps]),
            "cube": np.repeat([e["cube"] for e in eps], lens), "target": np.repeat([e["target"] for e in eps], lens),
            "episode_index": np.repeat(np.arange(len(eps)), lens)}


# ---------------------------------------------------------------- training
def train(data: dict, epochs: int = 60, batch: int = 256, lr: float = 1e-3, seed: int = 0, verbose: bool = True,
          text_init: Path | None = None, attention_lr_scale: float = 1.0):
    """text_init: start the text encoder from a GRU parser trained on text (language/parser.py).
    Everything is still trained end to end on actions afterwards. Without it, the from-scratch
    policy never learned target-first instructions with a blue target (devlog.md)."""
    from ari.learning.bc import split_by_episode

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    T = {k: torch.as_tensor(v) for k, v in data.items()}
    tr, va = split_by_episode(data["episode_index"], 0.1, seed)
    idx_tr, idx_va = np.nonzero(tr)[0], np.nonzero(va)[0]
    net = LanguagePolicy()
    if text_init is not None:
        from ari.language import parser as _parser

        net.text.load_state_dict(_parser.load(text_init).state_dict())

    # Normalization statistics of the 20-D observation, computed with the true objects selected.
    def true_obs(idx):
        i = torch.as_tensor(idx)
        cube = T["cubes"][i, T["cube"][i]]
        tgt = T["targets"][i, T["target"][i]]
        p = T["proprio"][i]
        return torch.cat([p, cube, tgt, cube[:, :3] - p[:, :3], tgt - p[:, :2], cube[:, :2] - tgt], dim=1)

    o = true_obs(idx_tr)
    net.obs_mean.copy_(o.mean(0)), net.obs_std.copy_(o.std(0).clamp_min(1e-6))
    a = T["action"][torch.as_tensor(idx_tr), :4]
    net.act_mean.copy_(a.mean(0)), net.act_std.copy_(a.std(0).clamp_min(1e-6))

    def batch_of(idx, shuffle_slots: bool):
        i = torch.as_tensor(idx)
        n = len(i)
        # A fresh paraphrase for every sample: free language augmentation.
        texts = [grammar.render(grammar.TRAIN_TEMPLATES[rng.integers(len(grammar.TRAIN_TEMPLATES))],
                                CUBE_COLORS[c], TARGET_COLORS[t], rng)
                 for c, t in zip(T["cube"][i].tolist(), T["target"][i].tolist())]
        tokens = torch.as_tensor(np.stack([grammar.encode(x) for x in texts]))
        cubes, targets = T["cubes"][i], T["targets"][i]
        ctags, ttags = torch.as_tensor(CUBE_TAGS).repeat(n, 1), torch.as_tensor(TARGET_TAGS).repeat(n, 1)
        if shuffle_slots:
            pc = torch.argsort(torch.rand(n, 3), dim=1)
            pt = torch.argsort(torch.rand(n, 3), dim=1)
            cubes = torch.gather(cubes, 1, pc.unsqueeze(-1).expand(-1, -1, 5))
            targets = torch.gather(targets, 1, pt.unsqueeze(-1).expand(-1, -1, 2))
            ctags, ttags = torch.gather(ctags, 1, pc), torch.gather(ttags, 1, pt)
        return tokens, T["proprio"][i], cubes, targets, ctags, ttags, T["action"][i]

    # The attention commits during the first epoch and then barely moves (a saturated softmax
    # passes almost no gradient). attention_lr_scale < 1 slows the language side down relative
    # to the motion network, to test whether committing too early is what goes wrong (devlog.md).
    slow = [q for n, q in net.named_parameters() if not n.startswith("mlp.")]
    opt = torch.optim.AdamW([{"params": net.mlp.parameters()},
                             {"params": slow, "lr": lr * attention_lr_scale}], lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(epochs):
        net.train()
        perm = rng.permutation(idx_tr)
        for k in range(0, len(perm), batch):
            *x, act = batch_of(perm[k:k + batch], True)
            out, _, _ = net(*x)
            loss = mse(out[:, :4], (act[:, :4] - net.act_mean) / net.act_std) + bce(out[:, 4], act[:, 4])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            sub = idx_va[:: max(1, len(idx_va) // 4000)]
            *x, act = batch_of(sub, False)
            out, ac, at = net(*x, hard=True)
            vm = mse(out[:, :4], (act[:, :4] - net.act_mean) / net.act_std).item()
            g = ((ac.argmax(1) == T["cube"][torch.as_tensor(sub)]) &
                 (at.argmax(1) == T["target"][torch.as_tensor(sub)])).float().mean().item()
        history.append({"epoch": epoch + 1, "val_motion": vm, "grounding": g})
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:3d}  val motion MSE {vm:.4f} | attention on the instructed objects {100 * g:.1f}%",
                  flush=True)
    return net, history


# ---------------------------------------------------------------- run time
class InstructedPolicy:
    """The trained network bound to one instruction, as an `obs -> action` rollout policy."""

    def __init__(self, env, net: LanguagePolicy, instruction: str):
        self.env, self.net = env, net.eval()
        self.tokens = torch.as_tensor(grammar.encode(instruction)).unsqueeze(0)
        self.ctags, self.ttags = torch.as_tensor(CUBE_TAGS).unsqueeze(0), torch.as_tensor(TARGET_TAGS).unsqueeze(0)
        with torch.no_grad():
            ac, at = net.attention(self.tokens, self.ctags, self.ttags)
        self.attended = (int(ac.argmax()), int(at.argmax()))      # which cube / target it will act on

    @torch.no_grad()
    def __call__(self, _obs=None) -> np.ndarray:
        s = scene_obs(self.env)
        out, _, _ = self.net(self.tokens, *(torch.as_tensor(s[k]).unsqueeze(0) for k in ("proprio", "cubes", "targets")),
                             self.ctags, self.ttags, hard=True)
        motion = out[0, :4] * self.net.act_std + self.net.act_mean
        return np.concatenate([motion.numpy(), [float(out[0, 4] > 0)]]).astype(np.float32)


def save(net: LanguagePolicy, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "language_policy", "state_dict": net.state_dict(), "vocab": grammar.VOCAB}, path)


def load(path: Path) -> LanguagePolicy:
    ck = torch.load(path, weights_only=True)
    assert ck["vocab"] == grammar.VOCAB, "policy was trained with a different vocabulary"
    net = LanguagePolicy()
    net.load_state_dict(ck["state_dict"])
    return net.eval()
