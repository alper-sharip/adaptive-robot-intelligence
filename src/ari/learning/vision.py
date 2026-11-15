"""M15: a policy that sees camera images instead of object positions.

Teacher-student setup ("privileged" imitation):
    teacher  ReactiveExpert, reads exact object poses from the simulator
    student  VisuomotorNet, gets two small camera images + the gripper's own pose
The student never receives the cube or target position; it has to find them in the pixels.
The teacher can label any state the student reaches, so DAgger works unchanged.

Student observation:
    front camera  64x64 RGB   whole table: roughly where the cube and target are
    wrist camera  64x64 RGB   rides on the hand: precise cube position when it matters
    proprioception (6)        gripper position, yaw (sin, cos), finger opening. A real robot
                              measures these with its joint encoders, so this is not cheating.

Network: one small CNN per camera, each ending in a SPATIAL SOFTMAX: for each of 32 feature
channels it outputs the (x, y) image location where that channel is most active. The image
is thereby reduced to 32 "keypoints", which suits manipulation: what matters is WHERE things
are (Levine, Finn et al., 2016). Keypoints + proprioception go through an MLP to the action.

No image augmentation by random shifts: the cameras are fixed and calibrated, and the policy
must turn pixel positions into millimetres. Shifting images would teach it the wrong geometry.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn

from ari.learning.spaces import ACT_DIM, tcp_yaw

CAMERAS = ("front_cam", "wrist_cam")
IMG = 64
PROPRIO_DIM = 6


class ImageObserver:
    """Renders the student's observation from the simulator."""

    def __init__(self, env, size: int = IMG):
        self.env = env
        self.renderer = mujoco.Renderer(env.model, height=size, width=size)
        # Shadows and floor reflections cost ~10x in software rendering (200 ms -> 16 ms per
        # image on CPU). They are switched off for speed; turning them on is a robustness
        # experiment for a GPU machine.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

    def images(self) -> np.ndarray:
        """uint8 array (2, 3, H, W): front and wrist camera, channels first."""
        out = []
        for cam in CAMERAS:
            self.renderer.update_scene(self.env.data, camera=cam)
            out.append(self.renderer.render().transpose(2, 0, 1))
        return np.stack(out)

    def proprio(self) -> np.ndarray:
        T = self.env.tcp_pose()
        yaw = tcp_yaw(T)
        return np.array([*T[:3, 3], np.sin(yaw), np.cos(yaw), self.env.finger_width()], dtype=np.float32)


class SpatialSoftmax(nn.Module):
    """(B, C, H, W) feature maps -> (B, 2C) expected (x, y) coordinates in [-1, 1]."""

    def forward(self, f):
        B, C, H, W = f.shape
        p = torch.softmax(f.reshape(B, C, H * W), dim=-1).reshape(B, C, H, W)
        ys = torch.linspace(-1, 1, H, device=f.device)
        xs = torch.linspace(-1, 1, W, device=f.device)
        return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], dim=1)


def _encoder(keypoints: int = 32) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(),      # 64 -> 30
        nn.Conv2d(32, 32, 3), nn.ReLU(),               # 28
        nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),     # 13
        nn.Conv2d(64, keypoints, 3),                   # 11
        SpatialSoftmax(),
    )


class VisuomotorNet(nn.Module):
    def __init__(self, keypoints: int = 32, hidden: int = 256):
        super().__init__()
        self.encoders = nn.ModuleList([_encoder(keypoints) for _ in CAMERAS])
        self.head = nn.Sequential(
            nn.Linear(2 * keypoints * len(CAMERAS) + PROPRIO_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ACT_DIM),
        )

    def forward(self, images, proprio):
        """images: float (B, 2, 3, H, W) in [0, 1]; proprio: normalized (B, 6)."""
        feats = [enc(images[:, i]) for i, enc in enumerate(self.encoders)]
        return self.head(torch.cat(feats + [proprio], dim=1))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():       # Apple Silicon (M1/M2/...)
        return torch.device("mps")
    return torch.device("cpu")


def fit(data: dict, net: VisuomotorNet | None = None, stats: dict | None = None, *, epochs: int = 10,
        batch_size: int = 64, lr: float = 1e-3, val_fraction: float = 0.1, seed: int = 0,
        device: torch.device | None = None, verbose: bool = True):
    """Train (or continue training) on data = {images, proprio, action, episode_index}.

    Pass an existing `net` and its `stats` to fine-tune (used by DAgger); normalization
    statistics are then kept fixed so the network's inputs keep their meaning.
    Returns (net, stats, history).
    """
    from ari.learning.bc import split_by_episode

    device = device or pick_device()
    torch.manual_seed(seed)
    imgs = torch.as_tensor(data["images"])                       # uint8, stays on CPU
    prop = torch.as_tensor(data["proprio"], dtype=torch.float32)
    act = torch.as_tensor(data["action"], dtype=torch.float32)
    tr, va = split_by_episode(data["episode_index"], val_fraction, seed)
    idx_tr, idx_va = np.nonzero(tr)[0], np.nonzero(va)[0]
    if stats is None:
        stats = {"prop_mean": prop[idx_tr].mean(0), "prop_std": prop[idx_tr].std(0).clamp_min(1e-6),
                 "act_mean": act[idx_tr, :4].mean(0), "act_std": act[idx_tr, :4].std(0).clamp_min(1e-6)}
    net = (net or VisuomotorNet()).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * int(np.ceil(len(idx_tr) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
    sd = {k: v.to(device) for k, v in stats.items()}

    def losses(idx):
        x = imgs[idx].to(device).float() / 255.0
        p = (prop[idx].to(device) - sd["prop_mean"]) / sd["prop_std"]
        a = act[idx].to(device)
        out = net(x, p)
        return mse(out[:, :4], (a[:, :4] - sd["act_mean"]) / sd["act_std"]), bce(out[:, 4], a[:, 4])

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        net.train()
        perm = rng.permutation(idx_tr)
        run = 0.0
        for i in range(0, len(perm), batch_size):
            lm, lg = losses(perm[i:i + batch_size])
            opt.zero_grad()
            (lm + lg).backward()
            opt.step()
            sched.step()
            run += lm.item()
        net.eval()
        with torch.no_grad():
            vm = np.mean([[l.item() for l in losses(idx_va[i:i + 512])] for i in range(0, len(idx_va), 512)], axis=0)
        history.append(dict(epoch=epoch + 1, train_motion=run / max(1, len(perm) // batch_size),
                            val_motion=float(vm[0]), val_gripper=float(vm[1])))
        if verbose:
            h = history[-1]
            print(f"  epoch {h['epoch']:2d}  motion MSE train {h['train_motion']:.4f} val {h['val_motion']:.4f} | "
                  f"gripper BCE val {h['val_gripper']:.4f}", flush=True)
    return net.cpu(), stats, history


class ImagePolicy:
    """The student as a rollout policy. It ignores the state observation it is handed and
    looks through its own cameras instead."""

    kind = "vision"

    def __init__(self, env, net: VisuomotorNet, stats: dict):
        self.net, self.stats = net.eval(), stats
        self.obs = ImageObserver(env)
        self.last = None          # (images, proprio) of the latest decision, for DAgger recording

    @torch.no_grad()
    def __call__(self, _state_obs=None) -> np.ndarray:
        im, pr = self.obs.images(), self.obs.proprio()
        self.last = (im, pr)
        x = torch.as_tensor(im).unsqueeze(0).float() / 255.0
        p = ((torch.as_tensor(pr) - self.stats["prop_mean"]) / self.stats["prop_std"]).unsqueeze(0)
        out = self.net(x, p)[0]
        motion = out[:4] * self.stats["act_std"] + self.stats["act_mean"]
        return np.concatenate([motion.numpy(), [float(out[4] > 0)]])


def save(path: Path, net: VisuomotorNet, stats: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "vision", "state_dict": net.state_dict(), **stats}, path)


def load(path: Path):
    ck = torch.load(path, weights_only=True)
    net = VisuomotorNet()
    net.load_state_dict(ck["state_dict"])
    return net, {k: ck[k] for k in ("prop_mean", "prop_std", "act_mean", "act_std")}
