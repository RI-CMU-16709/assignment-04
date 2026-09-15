"""Record the trained policy - offscreen for the video, viewer for debugging.

Supplied. Two rendering paths, one policy:

    python section3/record_policy.py --model section3/runs/ppo_seed0/best_model.zip
    python section3/record_policy.py --model ... --viewer        # interactive
    python section3/record_policy.py --model ... --episodes 3 --video drawer.mp4

Offscreen rendering needs a GL backend but no display:

    export MUJOCO_GL=egl      # headless machine with an NVIDIA GPU
    export MUJOCO_GL=osmesa   # headless machine, CPU only
    export MUJOCO_GL=glfw     # laptop with a display (default on macOS)

macOS: run the --viewer mode through `mjpython record_policy.py ...`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from drawer_env import make_drawer_env  # noqa: E402

OUTPUT = HERE / "output"


def load_model(path: Path, algo: str):
    import stable_baselines3 as sb3

    cls = {"ppo": sb3.PPO, "sac": sb3.SAC, "td3": sb3.TD3, "a2c": sb3.A2C}[algo.lower()]
    return cls.load(str(path))


def record_video(model, episodes: int, seed: int, video: Path, fps: int) -> None:
    import imageio.v2 as imageio

    env = make_drawer_env(render_mode="rgb_array")
    frames = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                print(
                    f"episode {episode}: success={info['is_success']} "
                    f"opening={info['drawer_opening']:.3f} m"
                )
                break
    env.close()

    OUTPUT.mkdir(exist_ok=True)
    imageio.mimsave(video, frames, fps=fps)
    print(f"wrote {video} ({len(frames)} frames)")


def run_viewer(model, episodes: int, seed: int) -> None:
    env = make_drawer_env(render_mode="human")
    dt = env.unwrapped.frame_skip * env.unwrapped.model.opt.timestep
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        while True:
            started = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            # Pace to wall clock so the motion is watchable. Training has no
            # such sleep, which is most of the throughput difference.
            remaining = dt - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
            if terminated or truncated:
                print(
                    f"episode {episode}: success={info['is_success']} "
                    f"opening={info['drawer_opening']:.3f} m "
                    f"max|a|={np.max(np.abs(action)):.2f}"
                )
                break
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--algo", default="ppo")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--video", default=str(OUTPUT / "policy.mp4"))
    parser.add_argument("--viewer", action="store_true", help="interactive instead of video")
    args = parser.parse_args()

    model = load_model(Path(args.model), args.algo)
    if args.viewer:
        run_viewer(model, args.episodes, args.seed)
    else:
        record_video(model, args.episodes, args.seed, Path(args.video), args.fps)


if __name__ == "__main__":
    main()
