"""Stage 5: train with PPO.

    python scripts/train_rl.py pendulum --steps 300000           # checks the PPO code on a standard benchmark
    python scripts/train_rl.py reach --steps 150000              # M16/M17: reach a point, from scratch
    python scripts/train_rl.py pick --steps 300000               # M17: pick-and-place from scratch
    python scripts/train_rl.py residual --base outputs/models/bc_reactive_100.pt --steps 300000   # M18

Training episodes use seeds 30000+; evaluation (scripts/evaluate_bc.py --joint-interp) uses 0-99.
Learning curves are written next to the model as <name>.json.
"""

import argparse
import json
from pathlib import Path

from ari.rl import ppo, tasks

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["pendulum", "reach", "pick", "residual"])
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", type=str, default="outputs/models/bc_reactive_100.pt")
    p.add_argument("--disturb-prob", type=float, default=0.0, help="fraction of training episodes with the cube moved")
    p.add_argument("--phi-offset", type=float, default=0.0, help="pick only: shift the shaping potential (see gym_envs.py)")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    name = args.name or f"{args.task}_s{args.seed}"
    out = Path("outputs/models") / f"rl_{name}.pt"
    common = dict(total_steps=args.steps, workers=args.workers, seed=args.seed)
    if args.task == "pendulum":
        build, bargs, kind = tasks.build_gym, {"env_id": "Pendulum-v1", "seed": args.seed}, None
        cfg = ppo.PPOConfig(gamma=0.9, lr=1e-3, steps_per_worker=1024, minibatch=64, **common)
    elif args.task == "reach":
        build, bargs, kind = tasks.build_reach, {"seed": args.seed}, "ppo_reach"
        cfg = ppo.PPOConfig(**common)
    elif args.task == "pick":
        build, kind = tasks.build_pick, "ppo_pick"
        bargs = {"seed": args.seed, "disturb_prob": args.disturb_prob, "phi_offset": args.phi_offset}
        cfg = ppo.PPOConfig(entropy_coef=0.001, **common)
    else:
        build = tasks.build_residual
        bargs, kind = {"base_path": args.base, "seed": args.seed, "disturb_prob": args.disturb_prob}, "residual"
        # The value function starts from nothing while the policy starts good: let the critic
        # catch up for a few iterations before the policy is allowed to change.
        cfg = ppo.PPOConfig(lr=1e-4, target_kl=0.02, critic_warmup_iters=3, **common)

    def checkpoint(policy, row):
        if kind is not None and row["iteration"] % 5 == 0:
            tasks.save(out, policy, kind)

    if args.steps == 0:      # save the untrained policy (for the residual: the clone + exploration noise)
        policy = build(worker_id=-1, **bargs)[1]
    else:
        policy = ppo.train(build, bargs, cfg, on_iteration=checkpoint)
    if kind is not None:
        tasks.save(out, policy, kind)
    result = {"args": vars(args), "log": cfg.log}
    if args.task == "reach":
        result["eval_deterministic"] = tasks.evaluate_reach(policy)
        result["eval_p_controller"] = tasks.evaluate_reach(None)
        print("learned policy :", result["eval_deterministic"])
        print("P-controller   :", result["eval_p_controller"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=str))
    print(f"saved {out.with_suffix('.json')}")
