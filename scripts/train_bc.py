"""M14: train a behaviour-cloning policy, optionally followed by DAgger iterations.

Run:  python scripts/train_bc.py --demos data/demos_reactive.npz --name bc_reactive_500
      python scripts/train_bc.py --demos data/demos_reactive.npz --n-demos 20 --name bc_reactive_20
      python scripts/train_bc.py --demos data/demos_reactive.npz --dagger-iters 3 --name bc_dagger
      python scripts/train_bc.py --demos data/demos_scripted.npz --name bc_scripted_500

DAgger (Ross et al., 2011): the learner's own mistakes take it to states the expert never
visited, where it has no training data. Each iteration runs the current learner, asks the
expert what it would do in every state the learner reached, adds those pairs to the dataset
and retrains. DAgger seeds start at 20000 (disjoint from demos 10000+ and evaluation 0-99).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ari.learning import bc, demos
from ari.learning.evaluate import dagger_collect

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demos", type=Path, required=True)
    p.add_argument("--n-demos", type=int, default=None, help="use only the first N demonstrations")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--dagger-iters", type=int, default=0)
    p.add_argument("--dagger-episodes", type=int, default=100, help="learner rollouts per DAgger iteration")
    p.add_argument("--keep-repeats", action="store_true",
                   help="keep repeated stuck states in DAgger data (the first, harmful version)")
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.workers)

    data = demos.load(args.demos, args.n_demos)
    n_eps = len(np.unique(data["episode_index"]))
    out = Path("outputs/models") / f"{args.name}.pt"
    log = {"args": {k: str(v) for k, v in vars(args).items()}, "stages": []}

    def fit(tag):
        print(f"[{tag}] training on {n_eps} episodes, {len(data['action'])} samples")
        policy, cfg = bc.train(data, bc.TrainConfig(epochs=args.epochs, seed=args.seed))
        policy.save(out)
        log["stages"].append({"tag": tag, "episodes": int(n_eps), "samples": int(len(data["action"])),
                              "final": cfg.history[-1], "history": cfg.history})

    fit("bc")
    for it in range(1, args.dagger_iters + 1):
        seeds = range(20_000 + (it - 1) * args.dagger_episodes, 20_000 + it * args.dagger_episodes)
        O, A, ok = dagger_collect(str(out), seeds, workers=args.workers, dedupe=not args.keep_repeats)
        print(f"[dagger {it}] learner succeeded on {ok}/{len(O)} rollouts; "
              f"adding {sum(len(o) for o in O)} expert-labelled states")
        log["stages"].append({"tag": f"dagger{it}_rollouts", "learner_success": ok / len(O)})
        for o, a in zip(O, A):
            data["observation.state"] = np.concatenate([data["observation.state"], o])
            data["action"] = np.concatenate([data["action"], a])
            data["episode_index"] = np.concatenate([data["episode_index"], np.full(len(o), n_eps)])
            n_eps += 1
        fit(f"dagger{it}")
    out.with_suffix(".json").write_text(json.dumps(log, indent=1))
    print(f"saved {out}")
