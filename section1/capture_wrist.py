"""Wrist camera: one RGB frame, one depth frame, and the frame reasoning.

Before running this you must mount the camera on the hand. Open

    scene/models/franka_emika_panda/panda_task.xml

and edit it in three places. Use the values given below verbatim: everyone in
the class renders from the same viewpoint, so the images are comparable and the
grading is about your reasoning, not about how patiently you nudged numbers.

1. In <asset>, register the supplied mesh and a material for it. setup_model.py
   has already copied the STL into assets/, so meshdir resolves the file name:

       <mesh name="zed_cam" file="zed_cam_monocular.stl" scale="0.001 0.001 0.001"/>
       <material class="panda" name="zed_gray" rgba="0.20 0.20 0.24 1"/>

   The scale is not decoration - work out what unit the STL is in and why the
   factor is exactly 0.001.

2. Inside <body name="hand">, add a body named cam_body carrying BOTH the
   camera and its physical shell, so the two can never drift apart. You write
   this block yourself; it needs four things:

     a) the body, mounted at pos="0.048 0 0.01" in the hand frame - use this
        value so every submission renders the same view;
     b) an <inertial> giving the housing mass="0.16" (a real ZED X One) and a
        small diagonal inertia, around 1e-4 per axis. Say what MuJoCo would do
        about the shell's mass if you left this out, and why that matters on an
        arm whose actuator gains were tuned for the shipped inertias;
     c) a <geom type="mesh"> using the zed_cam mesh and the zed_gray material,
        with euler="1.5708 0 0" to stand the housing upright, in
        class="visual" - say why a camera housing should not collide here;
     d) a <camera name="wrist_cam"> offset 0.03 along the body's own +Z, so the
        optical centre sits at the front face of the housing rather than inside
        it. Its orientation is yours to derive, below.

ALIGNING THE CAMERA AXES
------------------------
The one attribute not given to you is the camera's orientation, because it is
the part with the reasoning in it. Do not rotate the camera until the picture
looks right - derive it from two conventions:

    - A MuJoCo camera looks along its own -Z axis, with +Y up in the image.
    - The Panda gripper approaches along the hand frame's +Z.

So a wrist camera that sees what the gripper is about to grasp must have its
own -Z pointing along the hand's +Z. Fix the roll as well, so that everyone's
images match: the image's "up" direction must be the hand's +X.

The <camera> element takes no angle. It takes

    xyaxes="x1 x2 x3  y1 y2 y3"

which are the camera's own X and Y axes written in the parent frame, and MuJoCo
completes the frame with z = x cross y. That is the tag you use to align a
camera exactly, and it is why no trial and error is needed.

In your video:

    a) write x and y as vectors in the hand frame,
    b) compute z = x cross y,
    c) show that -z is the hand's +Z, i.e. the approach direction.

Two constraints fix the frame completely, so there is exactly one right answer.

Then demonstrate it, do not assert it. Open the scene in the viewer, switch the
view to the wrist camera (the camera selector is in the viewer's left panel;
cycling cameras with the "[" and "]" keys also works), and capture a screenshot
showing the arm and the view down the camera. Submit it as wrist_view.png. A
camera pointing backwards produces a perfectly valid render of the wrong thing,
and the screenshot is what proves you did not ship one.

MATCHING THE REAL CAMERA
------------------------
A MuJoCo <camera> defaults to fovy="45", a vertical field of view in degrees.
Real cameras are specified by a focal length and a sensor size instead, and
MuJoCo can take those directly with the focal, sensorsize and resolution
attributes. Answer these in your video:

    1. The mesh you mounted is a ZED X One 4K with a 3 mm lens. Look up its
       sensor size and give the fovy that would match that real camera. Is the
       45 degree default wider or narrower than the real thing?
    2. Re-specify the camera with focal, sensorsize and resolution instead of
       fovy, reload, and report the fovy MuJoCo derives. In one sentence: why
       is this the form to use when simulated images are meant to train
       something that will later run on the real camera?

Then:

    python section1/capture_wrist.py
    export MUJOCO_GL=egl      # headless GPU;  osmesa = CPU only;  glfw = laptop

Writes section1/output/wrist_rgb.png, wrist_depth.npy, wrist_depth.png, and
wrist_pose.txt. Complete TODO 1.1 and TODO 1.2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / "output"

CAMERA = "wrist_cam"
WIDTH, HEIGHT = 640, 480


def render_rgb(model, data, camera: str) -> np.ndarray:
    """TODO 1.1 - render one RGB frame from the named wrist camera.

    Create a mujoco.Renderer(model, height=HEIGHT, width=WIDTH), point it at the
    named camera with renderer.update_scene(data, camera=camera), and return
    renderer.render(). Close the renderer when you are done: it owns an
    offscreen framebuffer.

    A camera you can see in the viewer is not the same thing as a camera you can
    render from. If this returns the third-person view, you passed the wrong
    camera.
    """
    raise NotImplementedError("TODO 1.1")


def render_depth(model, data, camera: str) -> np.ndarray:
    """TODO 1.2 - render the depth buffer, in metres.

    Same idea, plus renderer.enable_depth_rendering() before update_scene.
    MuJoCo returns depth already in metres along the camera's viewing axis, so
    no conversion is needed - but you must say in your report what the values at
    the sky and at the floor mean, and why depth has no colour channels.
    """
    raise NotImplementedError("TODO 1.2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=str(ROOT / "scene" / "drawer_scene.xml"))
    parser.add_argument("--camera", default=CAMERA)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if cam_id < 0:
        raise SystemExit(
            f'No camera named "{args.camera}" in {args.scene}.\n'
            "Add it to scene/models/franka_emika_panda/panda_task.xml - see the "
            "docstring at the top of this file."
        )

    OUTPUT.mkdir(exist_ok=True)
    rgb = render_rgb(model, data, args.camera)
    depth = render_depth(model, data, args.camera)

    imageio.imwrite(OUTPUT / "wrist_rgb.png", rgb)
    np.save(OUTPUT / "wrist_depth.npy", depth)
    finite = np.isfinite(depth) & (depth < 5.0)
    shown = np.zeros_like(depth)
    if finite.any():
        lo, hi = depth[finite].min(), depth[finite].max()
        shown[finite] = (depth[finite] - lo) / max(hi - lo, 1e-6)
    imageio.imwrite(OUTPUT / "wrist_depth.png", (shown * 255).astype(np.uint8))

    # The camera pose MuJoCo actually used, not the one you intended.
    pos = data.cam_xpos[cam_id]
    mat = data.cam_xmat[cam_id].reshape(3, 3)
    report = [
        f"camera            : {args.camera}",
        f"world position    : {np.round(pos, 4).tolist()}",
        f"x axis (image right): {np.round(mat[:, 0], 4).tolist()}",
        f"y axis (image up)   : {np.round(mat[:, 1], 4).tolist()}",
        f"viewing direction   : {np.round(-mat[:, 2], 4).tolist()}   (= -Z)",
        f"image size        : {rgb.shape}",
        f"depth range (m)   : {float(depth[finite].min()):.3f} - "
        f"{float(depth[finite].max()):.3f}" if finite.any() else "depth range: empty",
        f"pixels closer than 2 m: {float(np.mean(depth < 2.0)) * 100:.1f} %",
    ]
    (OUTPUT / "wrist_pose.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nwrote four files to {OUTPUT}")
    print("Check that the drawer front and handle are visible and roughly centred.")
    print("Then move the arm (change the keyframe qpos), re-run, and confirm the")
    print("camera moved with the hand: that is what 'rigidly attached' means.")


if __name__ == "__main__":
    main()
