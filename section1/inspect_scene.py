"""Diagnose the scene, then prove the repair. Supplied - run it, do not edit it.

    python section1/inspect_scene.py                       # checks scene/drawer_scene.xml
    python section1/inspect_scene.py --scene scene/drawer_scene_bad.xml

Five static checks (one per defect), then two dynamic ones that no amount of
attribute tweaking can fake:

  stability   10 s of simulation with the arm holding its keyframe pose:
              every state stays finite and bounded.
  convergence the same 4 s motion at dt and at dt/2 ends in the same place.
              If halving the timestep changes the answer, the answer was the
              integrator's, not the model's.

Writes section1/output/scene_report.txt. Submit that file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / "output"

MAX_TIMESTEP = 0.002
MAX_PENETRATION = 0.002       # m
MASS_RANGE = (0.2, 20.0)      # kg
INERTIA_RANGE = (1e-4, 1.0)   # kg m^2, per principal axis
MIN_HANDLE_RADIUS = 0.010     # m
APPROACH_SPEED = 0.35         # m/s, nominal closing speed during the approach
TOOL_HALF_EXTENT = 0.0085     # m, Panda fingertip pad half-size facing the handle
MIN_CONTACT_STEPS = 20        # consecutive steps the pair must be able to overlap
MAX_SPEED = 50.0              # rad/s
CONVERGENCE_TOL = 0.005       # m

INTEGRATOR = {0: "EULER", 1: "RK4", 2: "IMPLICIT", 3: "IMPLICITFAST"}
lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def gid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def contact_geoms(contact) -> tuple[int, int]:
    try:
        return int(contact.geom1), int(contact.geom2)
    except AttributeError:                     # newer bindings expose .geom
        return int(contact.geom[0]), int(contact.geom[1])


def check(number: int, title: str, ok: bool, detail: str) -> bool:
    say(f"  [{'PASS' if ok else 'FAIL'}] defect {number}: {title}")
    say(f"         {detail}")
    return ok


def hold_ctrl(model, data, key_id):
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return data.ctrl.copy()


def run(model, data, key_id, seconds, wiggle=False):
    """Deterministic rollout. With wiggle=True the arm sweeps two joints."""
    base = hold_ctrl(model, data, key_id)
    n_steps = int(seconds / model.opt.timestep)
    max_speed = 0.0
    for _ in range(n_steps):
        if wiggle:
            target = base.copy()
            target[0] = base[0] + 0.5 * np.sin(2.0 * np.pi * 0.5 * data.time)
            target[3] = base[3] + 0.25 * np.sin(2.0 * np.pi * 0.5 * data.time)
            data.ctrl[:] = np.clip(
                target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
        mujoco.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            return None, np.inf
        max_speed = max(max_speed, float(np.max(np.abs(data.qvel))))
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return data.xpos[hand].copy(), max_speed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=str(ROOT / "scene" / "drawer_scene.xml"))
    args = parser.parse_args()

    path = Path(args.scene)
    if not path.exists():
        raise SystemExit(
            f"{path} not found.\n"
            "Repair scene/drawer_scene_bad.xml and save it as scene/drawer_scene.xml."
        )

    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    say(f"Scene: {path}")
    say()
    say("Model")
    say("-----")
    say(f"  nq / nv / nu   : {model.nq} / {model.nv} / {model.nu}")
    say(f"  bodies / geoms : {model.nbody} / {model.ngeom}")
    say(f"  timestep       : {model.opt.timestep} s")
    say(f"  integrator     : {INTEGRATOR.get(int(model.opt.integrator), '?')}")
    say(f"  solver iters   : {model.opt.iterations} (ls {model.opt.ls_iterations})")
    say()

    say("Static checks")
    say("-------------")
    results = []

    integrator = INTEGRATOR.get(int(model.opt.integrator), "?")
    ok = model.opt.timestep <= MAX_TIMESTEP and integrator != "RK4"
    results.append(check(
        1, "integration settings", ok,
        f"timestep={model.opt.timestep} (limit {MAX_TIMESTEP}), integrator={integrator}"
        " - implicitfast is the Menagerie default for this arm",
    ))

    hold_ctrl(model, data, key_id)
    shell, front = gid(model, "cabinet_shell"), gid(model, "drawer_front")
    # Both are axis-aligned boxes in this scene, so an interval test is exact.
    delta = np.abs(data.geom_xpos[shell] - data.geom_xpos[front])
    reach = model.geom_size[shell][:3] + model.geom_size[front][:3]
    overlap = float(np.min(reach - delta))          # > 0 means the boxes intersect
    collidable = bool(
        (model.geom_contype[shell] & model.geom_conaffinity[front])
        or (model.geom_contype[front] & model.geom_conaffinity[shell])
    )
    worst = 0.0
    for i in range(data.ncon):
        worst = max(worst, -float(data.contact[i].dist))
    ok = (overlap <= 0.0) or (not collidable)
    results.append(check(
        2, "the cabinet shell does not fight the drawer", ok,
        f"shell/drawer_front boxes {'intersect by' if overlap > 0 else 'clear by'} "
        f"{abs(overlap) * 1000:.0f} mm, shell collidable={collidable}; "
        f"{data.ncon} contacts at t=0, deepest penetration {worst * 1000:.1f} mm",
    ))

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    dof = int(model.jnt_dofadr[jid])
    span = float(model.jnt_range[jid, 1] - model.jnt_range[jid, 0])
    damping = float(model.dof_damping[dof])
    ok = bool(model.jnt_limited[jid]) and 0.1 <= span <= 1.0 and damping > 0.0
    results.append(check(
        3, "drawer joint is limited and damped", ok,
        f"limited={bool(model.jnt_limited[jid])}, range span={span:.3f} m, damping={damping}",
    ))

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
    mass = float(model.body_mass[bid])
    inertia = np.asarray(model.body_inertia[bid], dtype=float)
    worst_inertia = float(np.max(inertia))
    arm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in [f"link{i}" for i in range(1, 8)] + ["hand"]
    ]
    arm_ids = [i for i in arm_ids if i >= 0]
    arm_mass = float(np.max(model.body_mass[arm_ids])) if arm_ids else float("nan")
    arm_inertia = float(np.max(model.body_inertia[arm_ids])) if arm_ids else float("nan")
    mass_ok = MASS_RANGE[0] <= mass <= MASS_RANGE[1]
    inertia_ok = INERTIA_RANGE[0] <= worst_inertia <= INERTIA_RANGE[1]
    ok = mass_ok and inertia_ok
    results.append(check(
        4, "drawer inertial properties are plausible", ok,
        f"mass={mass:.4f} kg ({'ok' if mass_ok else 'OUT OF RANGE'}, allowed "
        f"{MASS_RANGE[0]}-{MASS_RANGE[1]} kg); "
        f"largest principal inertia={worst_inertia:.4g} kg m^2 "
        f"({'ok' if inertia_ok else 'OUT OF RANGE'}, allowed "
        f"{INERTIA_RANGE[0]}-{INERTIA_RANGE[1]} kg m^2); "
        f"for scale, the heaviest Panda link is {arm_mass:.2f} kg with a largest "
        f"principal inertia of {arm_inertia:.4g} kg m^2. A body carries mass AND "
        "a rotational inertia tensor; both have to be physical.",
    ))

    handles = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith("drawer_handle"):
            collidable = bool(model.geom_contype[g]) or bool(model.geom_conaffinity[g])
            handles.append((name, float(model.geom_size[g, 0]), collidable))

    collidable_handles = [h for h in handles if h[2]]
    if collidable_handles:
        radius = min(h[1] for h in collidable_handles)
        window = 2.0 * (radius + TOOL_HALF_EXTENT)          # m of overlap on one pass
        steps = window / (APPROACH_SPEED * model.opt.timestep)
        ok = radius >= MIN_HANDLE_RADIUS and steps >= MIN_CONTACT_STEPS
    else:
        radius, steps, ok = 0.0, 0.0, False

    inventory = ", ".join(
        f"{n} r={r * 1000:.1f} mm {'COLLIDABLE' if c else 'visual only'}"
        for n, r, c in handles
    ) or "no geom named drawer_handle*"
    results.append(check(
        5, "the handle the gripper actually meets can be grasped", ok,
        f"{inventory}. Thinnest collidable radius={radius * 1000:.1f} mm "
        f"(minimum {MIN_HANDLE_RADIUS * 1000:.0f} mm); at {APPROACH_SPEED} m/s and "
        f"dt={model.opt.timestep} s the tool and that geom can overlap for about "
        f"{steps:.0f} consecutive steps (minimum {MIN_CONTACT_STEPS}). "
        "Contact is re-detected every step, so a pair that overlaps for only a "
        "few steps never builds up force.",
    ))

    say()
    say("Dynamic checks")
    say("--------------")
    _, speed = run(model, data, key_id, 10.0)
    stable = np.isfinite(speed) and speed < MAX_SPEED
    say(f"  [{'PASS' if stable else 'FAIL'}] 10 s stability: max |qvel| = "
        f"{speed:.2f} rad/s (limit {MAX_SPEED})")

    hand_a, _ = run(model, data, key_id, 4.0, wiggle=True)
    coarse = model.opt.timestep
    model.opt.timestep = coarse / 2.0
    hand_b, _ = run(model, data, key_id, 4.0, wiggle=True)
    model.opt.timestep = coarse
    if hand_a is None or hand_b is None:
        converged, delta = False, float("inf")
    else:
        delta = float(np.linalg.norm(hand_a - hand_b))
        converged = delta <= CONVERGENCE_TOL
    say(f"  [{'PASS' if converged else 'FAIL'}] timestep convergence: hand position "
        f"differs by {delta * 1000:.2f} mm between dt={coarse} and dt={coarse / 2}"
        f" (tolerance {CONVERGENCE_TOL * 1000:.0f} mm)")

    say()
    passed = sum(results)
    say(f"Result: {passed}/5 static checks, stability {'ok' if stable else 'FAILED'}, "
        f"convergence {'ok' if converged else 'FAILED'}")
    if passed == 5 and stable and converged:
        say("The model is sound. Explaining *why* each defect mattered is still part")
        say("of the grade - a passing check with no explanation earns no marks.")
    else:
        say("Do not start Section 2 yet: a controller tuned against a broken model")
        say("teaches you nothing about control.")

    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "scene_report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUTPUT / 'scene_report.txt'}")


if __name__ == "__main__":
    main()
