"""M15: train the image-based policy (teacher-student imitation, then DAgger).

Run (CPU-sized, what the repo's results were produced with):
    python scripts/train_vision.py --demos 200 --epochs 12 --dagger-iters 2 --name vision_cpu
Run (GPU or Apple-Silicon sized):
    python scripts/train_vision.py --demos 500 --epochs 30 --dagger-iters 3 --name vision_full

Demonstration seeds 10000+, DAgger seeds 20000+, evaluation seeds 0-99 (scripts/evaluate_bc.py).
The image dataset is cached in data/ and reused when it already holds enough demonstrations.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ari.learning import vision
from ari.learning.evaluate import append_dataset, collect_images

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=int, default=200)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--dagger-iters", type=int, default=0)
    p.add_argument("--dagger-episodes", type=int, default=60)
    p.add_argument("--dagger-epochs", type=int, default=8,
                   help="fine-tuning epochs per DAgger iteration. Too few and the policy gets WORSE: "
                        "3 epochs took the student from 23/60 to 1/60 (docs/devlog.md)")
    p.add_argument("--init", type=Path, default=None, help="continue from an existing checkpoint")
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(args.workers)
    device = vision.pick_device()
    print(f"device: {device}")

    cache = Path("data/image_demos.npz")
    data = dict(np.load(cache)) if cache.exists() else None
    if data is None or len(data["episode_seed"]) < args.demos:
        t0 = time.time()
        data, _ = collect_images("reactive", range(10_000, 10_000 + args.demos), workers=args.workers)
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, **data)
        print(f"collected {len(data['episode_seed'])} image demonstrations, {len(data['action'])} frames, "
              f"{time.time() - t0:.0f}s")
    else:
        keep = data["episode_index"] < args.demos
        data = {k: (v[keep] if k != "episode_seed" else v[:args.demos]) for k, v in data.items()}
        print(f"using {args.demos} cached demonstrations, {len(data['action'])} frames")

    out = Path("outputs/models") / f"{args.name}.pt"
    log = {"args": vars(args), "device": str(device), "stages": []}
    t0 = time.time()
    net, stats = vision.load(args.init) if args.init else (None, None)
    net, stats, hist = vision.fit(data, net, stats, epochs=args.epochs, lr=5e-4 if args.init else 1e-3,
                                  seed=args.seed, device=device)
    vision.save(out, net, stats)
    log["stages"].append({"tag": "bc", "frames": int(len(data["action"])), "history": hist,
                          "minutes": (time.time() - t0) / 60})
    for it in range(1, args.dagger_iters + 1):
        seeds = range(20_000 + (it - 1) * args.dagger_episodes, 20_000 + it * args.dagger_episodes)
        new, ok = collect_images(str(out), seeds, workers=args.workers)
        print(f"[dagger {it}] student succeeded on {ok}/{args.dagger_episodes} rollouts; "
              f"adding {len(new['action'])} teacher-labelled frames", flush=True)
        data = append_dataset(data, new)
        t0 = time.time()
        net, stats, hist = vision.fit(data, net, stats, epochs=args.dagger_epochs, lr=5e-4, seed=args.seed + it,
                                      device=device)
        vision.save(out, net, stats)
        log["stages"].append({"tag": f"dagger{it}", "student_success": ok / args.dagger_episodes,
                              "frames": int(len(data["action"])), "history": hist,
                              "minutes": (time.time() - t0) / 60})
    out.with_suffix(".json").write_text(json.dumps(log, indent=1, default=str))
    print(f"saved {out}")
