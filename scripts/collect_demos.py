"""M13: collect demonstrations from an expert.

Run:  python scripts/collect_demos.py --expert reactive --n 500     # state-feedback expert
      python scripts/collect_demos.py --expert scripted --n 500     # clock-driven Stage 2 expert
Output: data/demos_<expert>.npz  (not committed to git; regenerate with these commands)
Training seeds start at 10000 and never overlap the evaluation seeds 0-99.
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ari.learning import demos


def _chunk(args):
    start, stop, kind = args
    from ari.envs.pick_place import PickPlaceEnv
    from ari.skills.reactive_pick_place import ReactiveExpert

    env = PickPlaceEnv()
    out = []
    for seed in range(start, stop):
        if kind == "reactive":
            o, a, ok = demos.record_reactive(env, seed, ReactiveExpert())
        else:
            o, a, ok = demos.record_episode(env, seed)
        out.append((seed, o, a, ok))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--expert", choices=["reactive", "scripted"], default="reactive")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    args.out = args.out or Path(f"data/demos_{args.expert}.npz")

    t0 = time.time()
    s0 = demos.TRAIN_SEED_START
    edges = np.linspace(s0, s0 + args.n, args.workers * 4 + 1).astype(int)
    with Pool(args.workers) as pool:
        results = [r for chunk in pool.map(_chunk, [(a, b, args.expert) for a, b in zip(edges[:-1], edges[1:])]) for r in chunk]
    good = [r for r in results if r[3]]
    data = {
        "observation.state": np.concatenate([r[1] for r in good]),
        "action": np.concatenate([r[2] for r in good]),
        "episode_index": np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(good)]),
        "frame_index": np.concatenate([np.arange(len(r[1])) for r in good]),
        "episode_seed": np.array([r[0] for r in good]),
        "skipped_failed_expert_episodes": np.array(len(results) - len(good)),
    }
    data["timestamp"] = data["frame_index"] * demos.POLICY_DT
    demos.save(data, args.out)
    print(f"{len(good)} successful demonstrations ({len(results) - len(good)} expert failures skipped), "
          f"{len(data['action'])} samples, {time.time() - t0:.0f}s -> {args.out}")
