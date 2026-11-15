"""M15 (optional, needs a GPU for real training): ACT from the LeRobot library on our data.

ACT (Action Chunking with Transformers, Zhao et al. 2023) predicts the next `chunk_size`
actions at once instead of one. Chunks give the policy short-term commitment, which is the
"memory" route around the problem Stage 4 solved by redesigning the expert.

Steps:
    python scripts/lerobot_act.py convert                 # data/image_demos.npz -> LeRobotDataset
    python scripts/lerobot_act.py train --steps 20000     # GPU: ~1-2 h on a Colab T4 (not measured here)
    python scripts/lerobot_act.py eval --n 100            # roll out in our simulator, seeds 0-99

Requires `pip install lerobot` (written against lerobot 0.7.0; its API changes between releases).
Status: convert / a few training steps / one rollout are smoke-tested on CPU. A full training
run has NOT been done in this repository; no success rate is claimed for ACT.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import ari  # first: selects the headless renderer before mujoco is imported

ARI_VERSION = ari.__version__

ROOT = Path("data/lerobot_pickplace")
REPO_ID = "local/ari_pickplace"
OUT = Path("outputs/models/act")
TASK = "pick up the red cube and place it on the green target"
FPS = 20
FEATURES = {
    "observation.images.front": {"dtype": "image", "shape": (64, 64, 3), "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "image", "shape": (64, 64, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (6,),
                          "names": ["tcp_x", "tcp_y", "tcp_z", "sin_yaw", "cos_yaw", "finger_width"]},
    "action": {"dtype": "float32", "shape": (5,), "names": ["dx", "dy", "dz", "dyaw", "gripper"]},
}


def act_config(device: str, chunk: int):
    from lerobot.policies.act.configuration_act import ACTConfig

    # No ImageNet weights: 64x64 simulator images are far from ImageNet, and it avoids a download.
    return ACTConfig(chunk_size=chunk, n_action_steps=chunk // 2, pretrained_backbone_weights=None,
                     dim_model=256, dim_feedforward=1024, device=device)


def convert(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    d = np.load(args.src)
    n_eps = int(d["episode_index"].max()) + 1 if args.episodes is None else args.episodes
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ds = LeRobotDataset.create(REPO_ID, fps=FPS, features=FEATURES, root=ROOT, use_videos=False)
    for ep in range(n_eps):
        for i in np.nonzero(d["episode_index"] == ep)[0]:
            ds.add_frame({
                "observation.images.front": d["images"][i, 0].transpose(1, 2, 0),
                "observation.images.wrist": d["images"][i, 1].transpose(1, 2, 0),
                "observation.state": d["proprio"][i].astype(np.float32),
                "action": d["action"][i].astype(np.float32),
                "task": TASK,
            })
        ds.save_episode()
    ds.finalize()
    print(f"wrote {n_eps} episodes to {ROOT}")


def train(args):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    from ari.learning.vision import pick_device

    device = str(pick_device())
    cfg = act_config(device, args.chunk)
    meta = LeRobotDatasetMetadata(REPO_ID, root=ROOT)
    ds = LeRobotDataset(REPO_ID, root=ROOT, delta_timestamps=resolve_delta_timestamps(cfg, meta))
    policy = make_policy(cfg, ds_meta=meta)
    pre, post = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=args.workers)
    print(f"device {device}; {len(ds)} frames; {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M parameters")
    policy.train()
    step, t0, log = 0, time.time(), []
    while step < args.steps:
        for batch in loader:
            loss, info = policy.forward(pre(batch))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            step += 1
            if step % args.log_every == 0 or step == args.steps:
                log.append({"step": step, "loss": loss.item(), **{k: float(v) for k, v in (info or {}).items()}})
                print(f"step {step:6d}  loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)
            if step >= args.steps:
                break
    OUT.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(OUT)
    pre.save_pretrained(OUT)
    post.save_pretrained(OUT)
    (OUT / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {OUT}")


class ACTAsPolicy:
    """Adapter: a LeRobot policy as an `obs -> action` callable for ari.learning.demos.rollout."""

    def __init__(self, env, path=OUT):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        from ari.learning.vision import ImageObserver

        self.policy = ACTPolicy.from_pretrained(path).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(path))
        self.obs = ImageObserver(env)

    def reset(self):
        self.policy.reset()       # clears the queue of not-yet-executed chunk actions

    @torch.no_grad()
    def __call__(self, _state_obs=None):
        im = torch.as_tensor(self.obs.images()).float() / 255.0          # (2, 3, H, W) in [0, 1]
        batch = {"observation.images.front": im[0:1], "observation.images.wrist": im[1:2],
                 "observation.state": torch.as_tensor(self.obs.proprio()).unsqueeze(0), "task": [TASK]}
        a = self.post(self.policy.select_action(self.pre(batch)))
        a = a.squeeze(0).cpu().numpy().astype(np.float32)
        a[4] = float(a[4] > 0.5)
        return a


def evaluate(args):
    from ari.envs.pick_place import PickPlaceEnv
    from ari.learning import demos

    env = PickPlaceEnv()
    policy = ACTAsPolicy(env)
    results = [demos.rollout(env, policy, s, max_time=args.max_time) for s in range(args.n)]
    k = sum(r.success for r in results)
    print(f"ACT: {k}/{args.n} successes; failures: "
          f"{ {f: sum(r.failure == f for r in results) for f in set(r.failure for r in results) if f} }")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--src", type=Path, default=Path("data/image_demos.npz"))
    c.add_argument("--episodes", type=int, default=None)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--chunk", type=int, default=20, help="actions predicted per inference (20 = 1 s)")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--log-every", type=int, default=200)
    e = sub.add_parser("eval")
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--max-time", type=float, default=25.0)
    args = p.parse_args()
    {"convert": convert, "train": train, "eval": evaluate}[args.cmd](args)
