"""Train PPO on the drawer task. Headless, seeded, and recorded.

Sized for one week: 150k steps, 4 parallel environments, roughly 12-20 minutes
on a laptop CPU. Everything around the algorithm is supplied - vectorised envs,
Monitor logging, a separate deterministic evaluation env, EvalCallback,
TensorBoard, and a run directory holding the exact configuration.

    python section3/train_ppo.py --seed 0
    python section3/train_ppo.py --seed 1
    python section3/train_ppo.py --seed 0 --sparse       # the required ablation

    tensorboard --logdir section3/runs

You complete TODO 3.6.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from drawer_env import MAX_EPISODE_STEPS, RewardConfig, make_drawer_env  # noqa: E402

RUNS = HERE / "runs"

# Defaults that are known to be reasonable for this task. Change them if you
# can say why - and report what you changed.
PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=512,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    device="cpu",
)


def env_factory(seed: int, rank: int, **env_kwargs):
    def _init():
        env = make_drawer_env(max_episode_steps=MAX_EPISODE_STEPS, **env_kwargs)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def build_callback(eval_env, run_dir: Path, eval_freq: int):
    """TODO 3.6 - configure the evaluation callback.

    Return a stable_baselines3.common.callbacks.EvalCallback with:

        deterministic=True          training samples actions, evaluation must not
        n_eval_episodes=10          one episode is not a measurement
        best_model_save_path=run_dir
        log_path=run_dir
        eval_freq=eval_freq         already divided by the number of envs

    Two questions for your report: why must evaluation be deterministic when
    training is stochastic, and why is eval_freq counted per environment?
    """
    raise NotImplementedError("TODO 3.6")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--sparse", action="store_true", help="reward ablation")
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    env_kwargs = dict(
        randomize=not args.no_randomize,
        reward_config=RewardConfig(sparse=args.sparse),
    )
    run_name = args.run_name or f"ppo_seed{args.seed}" + ("_sparse" if args.sparse else "")
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_env = VecMonitor(
        vec_cls([env_factory(args.seed, r, **env_kwargs) for r in range(args.n_envs)]),
        filename=str(run_dir / "monitor"),
    )
    eval_env = VecMonitor(
        DummyVecEnv([env_factory(args.seed + 10_000, 0, **env_kwargs)]),
        filename=str(run_dir / "eval_monitor"),
    )

    model = PPO(
        env=train_env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        **PPO_KWARGS,
    )
    callback = build_callback(eval_env, run_dir, max(args.eval_freq // args.n_envs, 1))

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "algo": "PPO",
                "ppo_kwargs": {k: str(v) for k, v in PPO_KWARGS.items()},
                "timesteps": args.timesteps,
                "seed": args.seed,
                "n_envs": args.n_envs,
                "sparse": args.sparse,
                "randomize": not args.no_randomize,
                "env": make_drawer_env(**env_kwargs).unwrapped.config(),
            },
            indent=2,
        )
        + "\n"
    )

    started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    wall = time.perf_counter() - started

    model.save(run_dir / "final_model")
    (run_dir / "training_time.json").write_text(
        json.dumps({"wall_clock_s": wall, "timesteps": args.timesteps}, indent=2) + "\n"
    )
    train_env.close()
    eval_env.close()

    print(f"\nrun directory : {run_dir}")
    print(f"wall clock    : {wall / 60:.1f} min for {args.timesteps} steps")
    print("best model    : best_model.zip (evaluate this one, not final_model)")


if __name__ == "__main__":
    main()
