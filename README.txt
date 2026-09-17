Assignment 4 (combined, one week): One Drawer, Two Controllers
=============================================================

Covers Lecture 6 (MuJoCo modelling, composition, sensing, numerical behaviour)
and Lecture 7 (Gymnasium, reward design, SB3, evaluation) on a single task: a
Menagerie Franka Panda opening a drawer.

A one-week assignment. It keeps the
evaluation logic - diagnose before you trust a model, validate before you spend
compute, report a protocol rather than a screenshot - at roughly a third of the
workload. Read Assignment-04.pdf before editing anything.

Directory layout
----------------

setup_model.py
  Copies mujoco_menagerie/franka_emika_panda into scene/models/ and writes
  panda_task.xml (keyframe removed, end_effector site added, meshdir fixed).
  Run this first; it explains every edit it makes.

scene/
  environment.xml           Menagerie-style floor, skybox, haze, lighting.
  drawer_scene_bad.xml      Loads and runs, and is wrong in five ways.
  DEFECTS.txt               What you must diagnose, fix, and report.
  drawer_scene.xml          YOU create this: your repaired scene. Everything
                            downstream (Sections 2 and 3) loads this file.

section1/
  inspect_scene.py          Supplied validator: model summary, five defect
                            checks, a 10 s stability run, and a timestep
                            convergence test. Writes output/scene_report.txt.
  capture_wrist.py          Wrist RGB + depth capture. TODO 1.1 and TODO 1.2.

section2/
  drawer_controller.py      Resolved-rate Cartesian controller.
                            TODO 2.1 - TODO 2.5.

section3/
  drawer_env.py             Gymnasium wrapper. TODO 3.1 - TODO 3.5.
  check_drawer_env.py       Supplied validator and random-policy baseline.
  train_ppo.py              SB3 PPO training. TODO 3.6.
  evaluate_policy.py        Deterministic evaluation protocol. TODO 3.7.
  record_policy.py          Supplied: offscreen video and interactive viewer.

section4/
  COMPARISON.txt            The head-to-head you must report.

Environment
-----------

  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt

Menagerie models: the course MuJoCo tree already has them
(Lecture_6/mujoco/mujoco-3.10.0/mujoco_menagerie). Otherwise:

  git clone https://github.com/google-deepmind/mujoco_menagerie

Suggested one-week order
------------------------

Day 1  python setup_model.py --menagerie /path/to/mujoco_menagerie
       python -m mujoco.viewer --mjcf=scene/drawer_scene_bad.xml
       Diagnose, repair, save scene/drawer_scene.xml
       python section1/inspect_scene.py                    (must print 5/5 PASS)

Day 2  Add the wrist camera to scene/models/franka_emika_panda/panda_task.xml
       python section1/capture_wrist.py

Day 3  Complete section2/drawer_controller.py
       python section2/drawer_controller.py

Day 4  Complete section3/drawer_env.py
       python section3/check_drawer_env.py                 (must pass)

Day 5  python section3/train_ppo.py --seed 0 --timesteps 150000
       python section3/train_ppo.py --seed 1 --timesteps 150000
       python section3/train_ppo.py --seed 0 --sparse --timesteps 150000

Day 6  python section3/evaluate_policy.py --model section3/runs/ppo_seed0/best_model.zip
       python section3/record_policy.py  --model section3/runs/ppo_seed0/best_model.zip
       Fill section4/COMPARISON.txt

Day 7  Record the video, check the submission list, submit.

Compute budget: one training run of 150k steps is roughly 12-20 min on a
laptop CPU with 4 parallel environments. Three runs total. Training is headless;
never render during training.

Submission
----------

Follow the exact submission list in Assignment-04.pdf. Do not submit .venv/,
scene/models/ (the copied Menagerie meshes), or intermediate checkpoints.
