"""Resolved-rate Cartesian controller: open the drawer without learning anything.

Simulation, task sequencing, metrics, plotting, and JSON output are supplied.
You complete the five decisions that make it a controller:

    TODO 2.1  site Jacobian
    TODO 2.2  Cartesian error
    TODO 2.3  damped least-squares resolved-rate update
    TODO 2.4  approach -> grasp transition
    TODO 2.5  success test

Background, in brief - the handout has the references if you want more.

  A feedback controller measures where the system is, compares that with where
  it should be, and turns the difference into a command. Here the measurement
  is the tool position x (m), the target is x_des (m), and the command is a
  joint position sent to the actuators. Nothing about the arm's dynamics is
  modelled: the loop simply keeps shrinking the error it can see.

  The Jacobian J is the matrix that converts joint rates into tool velocity,
  v = J qdot, where v is 3 x 1 (m/s), qdot is 7 x 1 (rad/s) and J is 3 x 7
  (m/rad). It is the local linearisation of the forward kinematics: at this
  pose, moving each joint at 1 rad/s moves the tool by the corresponding
  column of J. It changes with every configuration, so it is recomputed every
  step.

  Going the other way - what joint rates give the tool velocity I want - means
  inverting a matrix that is not square (7 joints, 3 constraints) and that
  loses rank in some poses. Damped least squares handles both at once:

      qdot = J^T (J J^T + lambda^2 I)^-1 v

  The lambda^2 I term (lambda in m/rad) keeps the matrix invertible near a
  singularity, trading a little tracking accuracy for a bounded qdot instead
  of an unbounded one.

    python section2/drawer_controller.py                 # with the viewer
    python section2/drawer_controller.py --headless      # faster, no window

Writes section2/output/results.json and section2/output/controller_plot.png.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_PATH = ROOT / "scene" / "drawer_scene.xml"
OUTPUT = HERE / "output"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
EE_SITE = "end_effector"
HANDLE_SITE = "handle_site"
SUCCESS_DISPLACEMENT = 0.24     # m of drawer travel
TIME_LIMIT = 12.0               # s


@dataclass
class TaskState:
    phase: str = "approach"
    phase_start_time: float = 0.0
    pull_start: np.ndarray | None = None
    time_log: list[float] = field(default_factory=list)
    error_log: list[float] = field(default_factory=list)
    drawer_log: list[float] = field(default_factory=list)
    max_control: float = 0.0
    max_joint_velocity: float = 0.0
    q_ref: np.ndarray | None = None      # persistent joint reference, rad


def object_id(model, obj_type, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return idx


def arm_addresses(model) -> tuple[np.ndarray, np.ndarray]:
    qpos, dof = [], []
    for name in ARM_JOINTS:
        jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos.append(model.jnt_qposadr[jid])
        dof.append(model.jnt_dofadr[jid])
    return np.asarray(qpos), np.asarray(dof)


def site_position(model, data, name: str) -> np.ndarray:
    return data.site_xpos[object_id(model, mujoco.mjtObj.mjOBJ_SITE, name)].copy()


def drawer_opening(model, data) -> float:
    jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    return float(data.qpos[model.jnt_qposadr[jid]])


def activate_weld(model, data) -> None:
    """Supplied. Freeze the current hand-drawer relative pose, then weld.

    A weld drives body2 to body1's pose transformed by eq_data[3:10]. Writing
    the *current* relative pose there means the constraint holds the drawer
    where the hand caught it instead of snapping the arm to the model default.
    """
    eq = object_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
    b1, b2 = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])

    neg_q1 = np.zeros(4)
    mujoco.mju_negQuat(neg_q1, data.xquat[b1])
    rel_pos = np.zeros(3)
    mujoco.mju_rotVecQuat(rel_pos, data.xpos[b2] - data.xpos[b1], neg_q1)
    rel_quat = np.zeros(4)
    mujoco.mju_mulQuat(rel_quat, neg_q1, data.xquat[b2])

    model.eq_data[eq, 0:3] = 0.0
    model.eq_data[eq, 3:6] = rel_pos
    model.eq_data[eq, 6:10] = rel_quat
    data.eq_active[eq] = 1
    mujoco.mj_forward(model, data)


# --------------------------------------------------------------------- TODOs
def translational_jacobian(model, data, site_name: str) -> np.ndarray:
    """TODO 2.1 - return the 3 x nv translational Jacobian of the named site.

    The function you want is mujoco.mj_jacSite. Look it up in the MuJoCo
    documentation and work out for yourself what it expects you to pass, what
    it writes into rather than returns, and which part of what it produces you
    actually need here:

        https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
        https://mujoco.readthedocs.io/en/stable/python.html

    Reading an API signature and getting the shapes and dtypes right from the
    documentation is the point of this TODO, so no skeleton is given.

    Returns: ndarray of shape (3, model.nv).
    """
    site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    raise NotImplementedError("TODO 2.1")


def resolved_rate_target(model, data, state, target, gain=3.0, damping=0.08):
    """Map a Cartesian target into seven joint position commands."""
    qpos_adr, dof_adr = arm_addresses(model)
    current = site_position(model, data, EE_SITE)

    # The joint reference is integrated in its own state, seeded once from the
    # keyframe pose. Do not rebuild it from data.qpos on every call: these
    # actuators are position servos, so a command of "wherever you are, plus a
    # little" leaves the servo with almost no error, no holding torque against
    # gravity, and a tracking rate reduced by kp*dt/kv. state.q_ref keeps the
    # reference ahead of the measurement, which is what makes the arm move.
    if state.q_ref is None:
        state.q_ref = data.qpos[qpos_adr].copy()

    # TODO 2.2: the Cartesian position error that the controller drives to zero.
    error = np.zeros(3)

    jacobian = translational_jacobian(model, data, EE_SITE)[:, dof_adr]

    # TODO 2.3: damped least-squares resolved rate.
    #
    #     dq = J^T (J J^T + lambda^2 I)^-1 (gain * error)
    #
    # Use np.linalg.solve. Never form an explicit inverse, and be ready to
    # explain what lambda does near a singularity.
    dq = np.zeros(7)

    state.q_ref = state.q_ref + model.opt.timestep * dq
    q_target = state.q_ref.copy()
    for i, name in enumerate(ARM_JOINTS):
        jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q_target[i] = np.clip(q_target[i], model.jnt_range[jid, 0], model.jnt_range[jid, 1])
    return q_target, float(np.linalg.norm(error))


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 3.0 * value**2 - 2.0 * value**3


def desired_position(model, data, state: TaskState):
    handle = site_position(model, data, HANDLE_SITE)
    tool = site_position(model, data, EE_SITE)

    if state.phase == "approach":
        # A standoff keeps the tool off the handle while it approaches. Whether
        # you keep one, and how big it is, interacts with the threshold you pick
        # in TODO 2.4: a standoff larger than that threshold can never trigger
        # the transition, because the controller converges to the standoff and
        # stops there. Change this line if your threshold needs you to.
        target = handle + np.array([-0.06, 0.0, 0.0])

        # TODO 2.4: switch to "grasp" once the tool is close enough to the
        # handle. Choose the threshold and record data.time in
        # state.phase_start_time. Say in your report what a threshold that is
        # too tight or too loose does to the task.

        return target, False

    if state.phase == "grasp":
        activate_weld(model, data)
        state.pull_start = tool.copy()
        state.phase_start_time = data.time
        state.phase = "pull"
        return tool, False

    if state.phase == "pull":
        assert state.pull_start is not None
        progress = smoothstep((data.time - state.phase_start_time) / 2.5)
        target = state.pull_start + np.array([-0.27 * progress, 0.0, 0.0])

        # TODO 2.5: success when the drawer has travelled far enough.
        # drawer_opening(model, data) gives the current displacement.
        success = False
        return target, success

    raise ValueError(f"Unknown phase: {state.phase}")


# ------------------------------------------------------------------ supplied
def record(model, data, state, error_norm, dof_adr, act_ids):
    state.time_log.append(float(data.time))
    state.error_log.append(error_norm)
    state.drawer_log.append(drawer_opening(model, data))
    state.max_control = max(state.max_control, float(np.max(np.abs(data.ctrl[act_ids]))))
    state.max_joint_velocity = max(
        state.max_joint_velocity, float(np.max(np.abs(data.qvel[dof_adr])))
    )


def save_results(state: TaskState, success: bool) -> None:
    OUTPUT.mkdir(exist_ok=True)
    results = {
        "controller": "resolved-rate with damped least squares",
        "success": bool(success),
        "completion_time_s": state.time_log[-1] if state.time_log else None,
        "drawer_displacement_m": max(state.drawer_log, default=0.0),
        "maximum_position_error_m": max(state.error_log, default=0.0),
        "maximum_joint_velocity_rad_s": state.max_joint_velocity,
        "maximum_absolute_control": state.max_control,
    }
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    axes[0].plot(state.time_log, state.drawer_log)
    axes[0].axhline(SUCCESS_DISPLACEMENT, color="tab:red", linestyle="--")
    axes[0].set_ylabel("drawer (m)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(state.time_log, state.error_log)
    axes[1].set_ylabel("position error (m)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "controller_plot.png", dpi=160)
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gain", type=float, default=3.0)
    parser.add_argument("--damping", type=float, default=0.08)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found - finish Section 1 first.")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(model, data)

    _, dof_adr = arm_addresses(model)
    act_ids = np.asarray(
        [object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS]
    )
    state, success = TaskState(), False

    def loop_body():
        nonlocal success
        target, success = desired_position(model, data, state)
        q_target, error_norm = resolved_rate_target(
            model, data, state, target, args.gain, args.damping
        )
        data.ctrl[act_ids] = q_target
        mujoco.mj_step(model, data)
        record(model, data, state, error_norm, dof_adr, act_ids)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            raise RuntimeError("Simulation became non-finite - go back to Section 1.")

    if args.headless:
        while data.time < TIME_LIMIT and not success:
            loop_body()
    else:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < TIME_LIMIT and not success:
                loop_body()
                viewer.sync()

    print(f"Drawer opened at t={data.time:.2f} s" if success
          else "Task did not reach the required drawer displacement.")
    save_results(state, success)


if __name__ == "__main__":
    main()
