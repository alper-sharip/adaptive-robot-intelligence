"""Stage 6: train the language models.

    python scripts/train_language.py parser            # text -> (cube, target): bag of words vs GRU
    python scripts/train_language.py policy --demos 500 --epochs 60   # instruction + scene -> actions

Demonstration seeds 40000+ (multi-object scene); evaluation uses seeds 0+ (scripts/evaluate_language.py).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ari.language import grammar, parser, policy

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["parser", "policy"])
    p.add_argument("--demos", type=int, default=500)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pretrained-text", action="store_true",
                   help="policy: start the text encoder from outputs/models/language_parser.pt")
    p.add_argument("--attention-lr-scale", type=float, default=1.0,
                   help="policy: learning-rate multiplier for the text encoder and attention (experiment, see devlog)")
    p.add_argument("--name", type=str, default="language_policy")
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path("outputs/models")

    if args.what == "parser":
        table = {}
        for name, cls in (("bag_of_words", parser.BagOfWords), ("gru", parser.GRUParser)):
            for seed in (0, 1, 2):
                m = parser.train(cls(), seed=seed)
                table[f"{name}_seed{seed}"] = {s: parser.accuracy(m, s) for s in grammar.SPLITS}
                print(f"{name:13s} seed {seed}  " + "  ".join(f"{s}: {100 * a:5.1f}%" for s, a in table[f"{name}_seed{seed}"].items()))
                if name == "gru" and seed == args.seed:
                    parser.save(m, out / "language_parser.pt")
        (out / "language_parser.json").write_text(json.dumps(table, indent=1))
    else:
        cache = Path("data/language_demos.npz")
        if cache.exists() and len(np.unique(np.load(cache)["episode_index"])) >= args.demos * 0.95:
            data = dict(np.load(cache))
        else:
            t0 = time.time()
            data = policy.collect(args.demos)
            cache.parent.mkdir(exist_ok=True)
            np.savez_compressed(cache, **data)
            print(f"collected {len(np.unique(data['episode_index']))} demonstrations, {len(data['action'])} samples, "
                  f"{time.time() - t0:.0f}s")
        net, history = policy.train(data, epochs=args.epochs, seed=args.seed,
                                    text_init=out / "language_parser.pt" if args.pretrained_text else None,
                                    attention_lr_scale=args.attention_lr_scale)
        policy.save(net, out / f"{args.name}.pt")
        (out / f"{args.name}.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
        print("saved", out / f"{args.name}.pt")
