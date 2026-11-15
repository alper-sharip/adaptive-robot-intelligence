"""Stage 6 evaluation.

    python scripts/evaluate_language.py m19            # instruction following: 150 episodes per system
    python scripts/evaluate_language.py m20            # two-step tasks, with and without injected faults

Results are written to outputs/eval/language_results.json.
"""

import argparse
import json
from pathlib import Path

from ari.language import evaluate as lev

M = "outputs/models"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["m19", "m20"])
    p.add_argument("--policy", default=f"{M}/language_policy.pt")
    p.add_argument("--skill", default=f"{M}/bc_dagger.pt")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--systems", nargs="*", default=None)
    args = p.parse_args()
    out = Path("outputs/eval/language_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    store = json.loads(out.read_text()) if out.exists() else {}
    models = {"policy": args.policy, "parser": f"{M}/language_parser.pt", "skill": args.skill}

    if args.what == "m19":
        for system in args.systems or ["language_policy", "parser+skill", "oracle+expert"]:
            r = lev.run_m19(system, models, n=args.n or 150)
            key = f"m19|{system}" + (f"|{Path(args.policy).stem}" if system == "language_policy" else "")
            store[key] = r
            print(f"{key:55s} " + " | ".join(
                f"{s}: {100 * v['success']:.0f}% (understood {100 * v['understood']:.0f}%)" for s, v in r["by_split"].items()),
                flush=True)
            out.write_text(json.dumps(store, indent=1, default=float))
    else:
        configs = [(skill, mon, faults, fb) for faults in (False, True)
                   for skill, mon, fb in (("scripted", False, None), ("scripted", True, None), (args.skill, False, None),
                                          (args.skill, True, None), (args.skill, True, "reactive"),
                                          ("reactive", True, None))]
        if args.systems == ["fallback"]:
            configs = [c for c in configs if c[3]]
        for skill, mon, faults, fb in configs:
            r = lev.run_m20(skill, mon, faults, models["parser"], n=args.n or 100, fallback_spec=fb)
            key = (f"m20|{Path(skill).stem}{'+fallback' if fb else ''}|{'monitor' if mon else 'no-monitor'}|"
                   f"{'faults' if faults else 'clean'}")
            store[key] = r
            print(f"{key:48s} success {100 * r['success']:5.1f}% (CI {100 * r['ci95'][0]:.0f}-{100 * r['ci95'][1]:.0f})  "
                  f"plans right {100 * r['plan_ok']:.0f}%  attempts {r['mean_attempts']:.2f}  "
                  f"time {r['mean_sim_time']:.0f}s  faulted episodes {r['episodes_with_faults']}"
                  + (f" -> {100 * r['success_when_faulted']:.0f}% of those succeed" if r["episodes_with_faults"] else ""),
                  flush=True)
            out.write_text(json.dumps(store, indent=1, default=float))
