"""Deterministic evaluation: the number you actually report.

Ten fixed seeds, deterministic actions, and the same success criterion the
resolved-rate controller had to meet in Section 2 (drawer travel >= 0.24 m).

    python section3/evaluate_policy.py --model section3/runs/ppo_seed0/best_model.zip
    python section3/evaluate_policy.py --model ... --no-randomize

Writes section3/output/eval_results.json and section3/output/eval_plot.png.
You complete TODO 3.7.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from drawer_env import SUCCESS_DISPLACEMENT, make_drawer_env  # noqa: E402

OUTPUT = HERE / "output"


def run_episode(env, model, seed: int) -> dict:
    """TODO 3.7 - one deterministic episode, fully instrumented.

        1. obs, info = env.reset(seed=seed)
        2. loop:
             action, _ = model.predict(obs, deterministic=True)
             step the environment
             accumulate the return
             append info["drawer_opening"] and step * dt to the traces
             track max |action| and the time at which is_success first turns True
             stop on terminated or truncated
        3. fill and return the record below.

    deterministic=True is not optional: the training policy is stochastic, and
    a sampled action changes the number you are about to put in your report.
    """
    dt = env.unwrapped.frame_skip * env.unwrapped.model.opt.timestep
    record = {
        "seed": seed,
        "return": 0.0,
        "steps": 0,
        "success": False,
        "time_to_open_s": None,
        "max_opening_m": 0.0,
        "max_abs_action": 0.0,
        "opening_trace": [],
        "time_trace": [],
        "dt": dt,
    }
    raise NotImplementedError("TODO 3.7")


def summarize(records: list[dict]) -> dict:
    successes = [r for r in records if r["success"]]
    returns = np.array([r["return"] for r in records], dtype=float)
    return {
        "episodes": len(records),
        "success_rate": len(successes) / max(len(records), 1),
        "mean_return": float(returns.mean()) if returns.size else 0.0,
        "std_return": float(returns.std()) if returns.size else 0.0,
        "mean_time_to_open_s": (
            float(np.mean([r["time_to_open_s"] for r in successes])) if successes else None
        ),
        "mean_max_opening_m": float(np.mean([r["max_opening_m"] for r in records])),
        "max_abs_action": float(np.max([r["max_abs_action"] for r in records])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--no-randomize", action="store_true")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = make_drawer_env(randomize=not args.no_randomize)
    model = PPO.load(args.model)

    records = [run_episode(env, model, args.start_seed + i) for i in range(args.episodes)]
    env.close()

    summary = summarize(records)
    summary["model"] = str(args.model)
    summary["randomize"] = not args.no_randomize
    summary["success_displacement_m"] = SUCCESS_DISPLACEMENT

    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "eval_results.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(7, 6))
    for record in records:
        axes[0].plot(record["time_trace"], record["opening_trace"], alpha=0.6)
    axes[0].axhline(SUCCESS_DISPLACEMENT, color="tab:red", linestyle="--", label="target")
    axes[0].set_ylabel("drawer opening (m)")
    axes[0].set_xlabel("time (s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].bar(range(len(records)), [r["return"] for r in records])
    axes[1].set_ylabel("episode return")
    axes[1].set_xlabel("evaluation episode")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "eval_plot.png", dpi=160)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT / 'eval_results.json'} and {OUTPUT / 'eval_plot.png'}")


if __name__ == "__main__":
    main()
