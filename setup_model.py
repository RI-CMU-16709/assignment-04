"""Prepare the Menagerie Panda for this assignment.

Three edits are needed before the Menagerie Panda can be composed into a scene
that adds a moving drawer. Each is printed as it happens - you are expected to
be able to explain all three.

  1. Remove the <keyframe> block.
     panda.xml ships a "home" keyframe with nq = 9 values. Adding the drawer's
     slide joint makes nq = 10, and MuJoCo rejects a keyframe whose qpos length
     does not match nq. The scene file defines its own 10-value keyframe.

  2. Add an end_effector site inside <body name="hand">.
     Section 2 measures the tool-to-handle distance and takes a site Jacobian;
     both need a named frame on the tool.

  3. Fix meshdir.
     Includes and asset directories resolve relative to the *main* MJCF file,
     which will be scene/drawer_scene.xml. So meshdir becomes
     models/franka_emika_panda/assets.

It also copies the supplied ZED camera mesh, scene/zed_cam_monocular.STL, into
the model's assets/ directory so that meshdir resolves it. The mesh is only
placed on disk - referencing it from the MJCF is your job.

What this script does NOT do: the wrist camera. Adding <camera name="wrist_cam">
to the hand body, mounting the ZED mesh at the same pose, and getting the
optical convention right is Task 1.3. See section1/capture_wrist.py.

Usage
-----
    python setup_model.py --menagerie /path/to/mujoco_menagerie
    python setup_model.py            # searches a few likely locations
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / "scene" / "models" / "franka_emika_panda"

CANDIDATES = (
    Path.home() / "mujoco_menagerie",
    HERE.parents[2] / "Lecture_6" / "mujoco" / "mujoco-3.10.0" / "mujoco_menagerie",
    Path.cwd() / "mujoco_menagerie",
)

EE_SITE = '<site name="end_effector" pos="0 0 0.1034" size="0.01" rgba="1 0.2 0.1 1"/>'


def find_menagerie(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not (path / "franka_emika_panda" / "panda.xml").exists():
            raise SystemExit(f"No franka_emika_panda/panda.xml under {path}")
        return path
    for path in CANDIDATES:
        if (path / "franka_emika_panda" / "panda.xml").exists():
            return path.resolve()
    raise SystemExit(
        "Could not find mujoco_menagerie. Pass --menagerie /path/to/mujoco_menagerie\n"
        "or clone it: git clone https://github.com/google-deepmind/mujoco_menagerie"
    )


def strip_keyframe(xml: str) -> str:
    new, n = re.subn(r"[ \t]*<keyframe>.*?</keyframe>[ \t]*\n", "", xml, flags=re.S)
    print("  - removed <keyframe>: the drawer joint changes nq from 9 to 10"
          if n else "  ! no <keyframe> block found (already removed?)")
    return new


def add_end_effector_site(xml: str) -> str:
    if 'name="end_effector"' in xml:
        print("  ! end_effector site already present")
        return xml
    match = re.search(r'^([ \t]*)<body name="hand".*?>[ \t]*\n', xml, flags=re.M)
    if not match:
        raise SystemExit('Could not find <body name="hand"> in panda.xml')
    indent = match.group(1) + "  "
    print('  - added <site name="end_effector"> inside <body name="hand">')
    return xml[: match.end()] + indent + EE_SITE + "\n" + xml[match.end() :]


def fix_meshdir(xml: str) -> str:
    new, n = re.subn(
        r'meshdir="assets"', 'meshdir="models/franka_emika_panda/assets"', xml
    )
    if n != 1:
        raise SystemExit('Expected exactly one meshdir="assets" in panda.xml')
    print("  - meshdir now resolves from the main file (scene/)")
    return new


def install_camera_mesh() -> None:
    """Put the supplied ZED mesh where meshdir can find it."""
    src = HERE / "scene" / "zed_cam_monocular.STL"
    if not src.exists():
        print(f"  ! {src.name} not found next to the scene - skipping the camera mesh")
        return
    dest = DEST / "assets" / "zed_cam_monocular.stl"
    shutil.copyfile(src, dest)
    print(f"  - copied {src.name} to assets/{dest.name} (Task 1.3 references it)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menagerie", default=None)
    parser.add_argument("--force", action="store_true", help="overwrite scene/models")
    args = parser.parse_args()

    menagerie = find_menagerie(args.menagerie)
    print(f"Menagerie: {menagerie}")

    if DEST.exists():
        if not args.force:
            print(f"{DEST} already exists. Re-run with --force to overwrite.")
            sys.exit(0)
        shutil.rmtree(DEST)

    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(menagerie / "franka_emika_panda", DEST)
    print(f"Copied the complete Menagerie model directory to {DEST}")
    print("  (meshes, LICENSE, and README are preserved - do not delete them)")

    xml = (DEST / "panda.xml").read_text()
    print("Writing panda_task.xml:")
    xml = strip_keyframe(xml)
    xml = add_end_effector_site(xml)
    xml = fix_meshdir(xml)
    xml = xml.replace('<mujoco model="panda">', '<mujoco model="panda_task">', 1)
    (DEST / "panda_task.xml").write_text(xml)
    install_camera_mesh()

    print("\nDone. Next:")
    print("  python -m mujoco.viewer --mjcf=scene/drawer_scene_bad.xml")
    print("  read scene/DEFECTS.txt, repair the scene, save scene/drawer_scene.xml")
    print("  python section1/inspect_scene.py")


if __name__ == "__main__":
    main()
