"""Validate the environment before spending a training budget on it.

Supplied - run it, do not edit it. It answers four questions in order:

  1. Does the model compile and does every named object exist?
  2. Do the declared spaces match the arrays the environment actually returns?
  3. Does SB3's check_env accept the API (dtypes, reset signature, five-tuple)?
  4. What does a random policy score? That is the number learning must beat.

    python section3/check_drawer_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drawer_env import (  # noqa: E402
    MAX_EPISODE_STEPS,
    SUCCESS_DISPLACEMENT,
    DrawerEnv,
    make_drawer_env,
)

N_EPISODES = 5
SEED = 0


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    section("1. Model")
    env = DrawerEnv()
    model = env.model
    print(f"  nq / nv / nu     : {model.nq} / {model.nv} / {model.nu}")
    print(f"  timestep         : {model.opt.timestep} s")
    print(f"  frame_skip       : {env.frame_skip}")
    print(f"  control rate     : {1.0 / (env.frame_skip * model.opt.timestep):.1f} Hz")
    print(f"  episode budget   : {MAX_EPISODE_STEPS} steps "
          f"= {MAX_EPISODE_STEPS * env.frame_skip * model.opt.timestep:.1f} s")
    print(f"  success at       : drawer_slide >= {SUCCESS_DISPLACEMENT} m")

    section("2. Spaces and one transition")
    print(f"  observation_space: {env.observation_space}")
    print(f"  action_space     : {env.action_space}")
    obs, info = env.reset(seed=SEED)
    print(f"  reset obs shape  : {obs.shape} dtype={obs.dtype}")
    if obs.shape != env.observation_space.shape:
        raise SystemExit("TODO 3.1 incomplete: observation shape does not match the space")
    if not np.all(np.isfinite(obs)):
        raise SystemExit("observation contains non-finite values")
    if np.allclose(obs, 0.0):
        print("  ! every observation entry is zero - TODO 3.1 is probably still empty")

    print(f"  ee position      : {np.round(env.ee_position, 3)}")
    print(f"  handle position  : {np.round(env.handle_position, 3)}")
    print(f"  tool -> handle   : {np.linalg.norm(env.handle_position - env.ee_position):.3f} m")

    action = env.action_space.sample()
    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except NotImplementedError as exc:
        raise SystemExit(
            f"  incomplete environment: {exc}\n"
            "  Finish TODO 3.2 (and 3.3, 3.4, 3.5) in drawer_env.py, then re-run."
        ) from exc
    print(f"  step reward      : {reward:.4f}")
    print(f"  reward terms     : {info['reward_terms']}")
    print(f"  terminated/trunc : {terminated} / {truncated}")
    env.close()

    section("3. SB3 API check")
    try:
        from stable_baselines3.common.env_checker import check_env

        check_env(make_drawer_env(), warn=True)
        print("  check_env passed")
    except ImportError:
        print("  stable-baselines3 not installed - skipped")

    section("4. Random-policy baseline")
    env = make_drawer_env()
    returns, lengths, openings, successes = [], [], [], 0
    for episode in range(N_EPISODES):
        obs, info = env.reset(seed=SEED + episode)
        env.action_space.seed(SEED + episode)
        total, steps, best = 0.0, 0, 0.0
        while True:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            steps += 1
            best = max(best, info["drawer_opening"])
            if terminated or truncated:
                break
        successes += int(info["is_success"])
        returns.append(total)
        lengths.append(steps)
        openings.append(best)
        print(
            f"  episode {episode}: return={total:8.2f} length={steps:4d} "
            f"max_opening={best:.3f} m success={info['is_success']}"
        )
    env.close()

    print()
    print(f"  mean return      : {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"  mean length      : {np.mean(lengths):.1f}")
    print(f"  mean max opening : {np.mean(openings):.3f} m")
    print(f"  success rate     : {successes}/{N_EPISODES}")
    print()
    print("Record these numbers. A trained policy that does not clearly beat them")
    print("has not learned anything, no matter how good the reward curve looks.")


if __name__ == "__main__":
    main()
