"""Gymnasium environment: a Menagerie Franka Panda opening a drawer.

Same scene as Section 2, same success criterion. What changes is who writes
the controller.

Everything mechanical is supplied: model loading, name lookups, the weld helper,
reset with domain randomization, rendering, and the step loop. You complete the
five decisions that define the learning problem:

    TODO 3.1  what the policy observes            -> _get_obs
    TODO 3.2  what an action means                -> _apply_action
    TODO 3.3  when the grasp happens              -> _maybe_grasp
    TODO 3.4  when an episode terminates          -> _terminated
    TODO 3.5  what behaviour the reward pays for  -> compute_reward

Observation layout (23 values, float32) - keep this order, check_drawer_env.py
and the observation_space below assume it:

     0: 6    arm joint positions        qpos[joint1..joint7]      (7)
     7: 13   arm joint velocities       qvel[joint1..joint7]      (7)
    14: 16   end-effector position      site "end_effector"       (3)
    17: 19   handle minus end effector  vector the policy chases  (3)
    20      drawer opening              qpos[drawer_slide]        (1)
    21      drawer opening rate         qvel[drawer_slide]        (1)
    22      grasped flag                0.0 or 1.0                (1)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_PATH = ROOT / "scene" / "drawer_scene.xml"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
EE_SITE = "end_effector"
HANDLE_SITE = "handle_site"
DRAWER_JOINT = "drawer_slide"
WELD = "grasp_weld"

OBS_DIM = 23
SUCCESS_DISPLACEMENT = 0.24   # metres of drawer travel
MAX_EPISODE_STEPS = 300       # 300 * 0.02 s = 6 s of task time at frame_skip=10


@dataclass
class RewardConfig:
    """Starting weights. You may change them - and you must justify what you change."""

    reach: float = 1.0            # shaping toward the handle
    open: float = 10.0            # drawer displacement
    success: float = 50.0         # one-off bonus
    grasp: float = 5.0            # one-off bonus when the weld engages
    action_cost: float = 0.01     # ||a||^2
    action_rate_cost: float = 0.01  # ||a - a_prev||^2
    sparse: bool = False          # ablation switch: reward only success


class DrawerEnv(gym.Env):
    """Franka Panda + sliding drawer, joint-space delta control."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,
        max_joint_delta: float = 0.05,
        grasp_threshold: float = 0.03,
        randomize: bool = True,
        reward_config: RewardConfig | None = None,
    ) -> None:
        super().__init__()

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} missing - repair the scene in Section 1 first"
            )
        if not (ROOT / "scene" / "models" / "franka_emika_panda" / "panda_task.xml").exists():
            raise FileNotFoundError(
                "scene/models/franka_emika_panda/panda_task.xml missing - run setup_model.py"
            )

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = int(frame_skip)
        self.max_joint_delta = float(max_joint_delta)
        self.grasp_threshold = float(grasp_threshold)
        self.randomize = bool(randomize)
        self.reward_config = reward_config or RewardConfig()

        # ---- name -> index lookups, resolved once ---------------------------
        self._arm_qpos = np.array([self._jnt(name)[0] for name in ARM_JOINTS])
        self._arm_dof = np.array([self._jnt(name)[1] for name in ARM_JOINTS])
        self._arm_act = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS]
        )
        self._ee_site = self._id(mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self._handle_site = self._id(mujoco.mjtObj.mjOBJ_SITE, HANDLE_SITE)
        self._drawer_qpos, self._drawer_dof = self._jnt(DRAWER_JOINT)
        self._weld = self._id(mujoco.mjtObj.mjOBJ_EQUALITY, WELD)
        self._home_key = self._id(mujoco.mjtObj.mjOBJ_KEY, "home")
        self._cabinet = self._id(mujoco.mjtObj.mjOBJ_BODY, "cabinet")
        self._cabinet_home = self.model.body_pos[self._cabinet].copy()

        self._ctrl_low = self.model.actuator_ctrlrange[self._arm_act, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[self._arm_act, 1].copy()

        # ---- spaces ---------------------------------------------------------
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # ---- episode state ---------------------------------------------------
        self._grasped = False
        self._prev_action = np.zeros(7, dtype=np.float32)
        self._elapsed = 0
        self._success = False

        # ---- rendering -------------------------------------------------------
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None

    # ------------------------------------------------------------------ utils
    def _id(self, obj_type, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return idx

    def _jnt(self, name: str) -> tuple[int, int]:
        jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])

    @property
    def ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site].copy()

    @property
    def handle_position(self) -> np.ndarray:
        return self.data.site_xpos[self._handle_site].copy()

    @property
    def drawer_opening(self) -> float:
        return float(self.data.qpos[self._drawer_qpos])

    def _activate_weld(self) -> None:
        """Freeze the current hand-drawer relative pose and switch the weld on.

        Supplied. A weld drives body2 to the pose of body1 transformed by
        eq_data[3:10] (position + quaternion). Writing the *current* relative
        pose there means the constraint holds the drawer where the hand caught
        it, instead of snapping the arm to the model default.
        """
        model, data = self.model, self.data
        b1 = int(model.eq_obj1id[self._weld])   # hand
        b2 = int(model.eq_obj2id[self._weld])   # drawer

        neg_q1 = np.zeros(4)
        mujoco.mju_negQuat(neg_q1, data.xquat[b1])

        delta = data.xpos[b2] - data.xpos[b1]
        rel_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(rel_pos, delta, neg_q1)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, neg_q1, data.xquat[b2])

        model.eq_data[self._weld, 0:3] = 0.0            # anchor at body2 origin
        model.eq_data[self._weld, 3:6] = rel_pos
        model.eq_data[self._weld, 6:10] = rel_quat
        data.eq_active[self._weld] = 1
        mujoco.mj_forward(model, data)
        self._grasped = True

    # ------------------------------------------------------------------ TODOs
    def _get_obs(self) -> np.ndarray:
        """TODO 3.1 - build the observation described in the module docstring.

        Use self.data.qpos[self._arm_qpos], self.data.qvel[self._arm_dof],
        self.ee_position, self.handle_position, self.drawer_opening,
        self.data.qvel[self._drawer_dof], and self._grasped.

        Return float32 of shape (OBS_DIM,). Nothing here may be a Python list:
        SB3 will copy this array millions of times.
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        return obs

    def _apply_action(self, action: np.ndarray) -> None:
        """TODO 3.2 - turn a normalised action into actuator commands.

        The seven arm actuators are POSITION actuators: data.ctrl is a target
        joint angle, not a torque. Interpret the action as a bounded increment
        of the current target:

            ctrl <- clip(ctrl + action * self.max_joint_delta,
                         self._ctrl_low, self._ctrl_high)

        Write only the seven arm actuators (self._arm_act); leave the gripper
        actuator alone. Explain in your video why an unbounded, absolute action
        would make this task much harder to learn.
        """
        raise NotImplementedError("TODO 3.2")

    def _maybe_grasp(self) -> None:
        """TODO 3.3 - engage the weld the first time the tool reaches the handle.

        Condition: not already grasped and ||ee - handle|| < self.grasp_threshold.
        Call self._activate_weld() - it does the rest. Think about what the
        threshold trades off: too large and the drawer teleports to the hand,
        too small and the policy may never trigger it.
        """
        return None

    def _terminated(self) -> bool:
        """TODO 3.4 - decide when the episode ends because of the task itself.

        Terminate on success: drawer opening >= SUCCESS_DISPLACEMENT. Set
        self._success so the info dict and the evaluation script can see it.

        Also terminate if the simulation state stops being finite. "State" here
        means MuJoCo's own state vector - self.data.qpos and self.data.qvel -
        and it means all of it, not just the seven arm joints: the drawer slide
        and the two finger joints are integrated in the same step, so a
        divergence in any of them has already corrupted the transition you are
        about to return. np.isfinite over both arrays is the whole test.

        A NaN or an inf means the solver diverged and the transition is
        meaningless, so the episode must end rather than be learned from. Note
        that this is a test of numerical validity, not of bad behaviour: an arm
        moving fast, or a drawer slammed to its limit, is a poor episode but a
        perfectly finite one, and it must NOT terminate here. Terminating on
        "large" instead of "not finite" silently teaches the policy that moving
        quickly ends the episode.

        The 6-second time budget is NOT handled here. It belongs to a TimeLimit
        wrapper, and it produces truncated=True, not terminated=True.
        """
        return False

    def compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        """TODO 3.5 - the reward function is the specification of the behaviour.

        Fill in the terms below using self.reward_config weights. Suggested
        starting point (you are free to do better, and to say why):

            reach          -cfg.reach * ||ee - handle||        (only before grasp)
            open           +cfg.open  * drawer_opening
            grasp          +cfg.grasp once, on the step the weld engages
            success        +cfg.success once, when the drawer passes the target
            action_cost    -cfg.action_cost * ||a||^2
            action_rate    -cfg.action_rate_cost * ||a - a_prev||^2

        If cfg.sparse is True, return only the success term: that ablation is
        required in Task 3.3 and it should visibly fail to learn.

        Return (reward, terms) where terms maps each name to its scalar value.
        The dict is logged, so keep the keys stable across runs.
        """
        cfg = self.reward_config
        terms = {
            "reach": 0.0,
            "open": 0.0,
            "grasp": 0.0,
            "success": 0.0,
            "action_cost": 0.0,
            "action_rate": 0.0,
        }
        reward = float(sum(terms.values()))
        return reward, terms

    # ------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        self.data.eq_active[self._weld] = 0
        self._grasped = False
        self._success = False
        self._elapsed = 0
        self._prev_action[:] = 0.0

        if self.randomize:
            # Small start-state noise and a cabinet that is not always in the
            # same place: a policy that only works from one pose has memorised
            # a trajectory, not a controller.
            self.data.qpos[self._arm_qpos] += self.np_random.uniform(
                -0.03, 0.03, size=7
            )
            jitter = self.np_random.uniform([-0.02, -0.04, 0.0], [0.02, 0.04, 0.0])
            self.model.body_pos[self._cabinet] = self._cabinet_home + jitter
        else:
            self.model.body_pos[self._cabinet] = self._cabinet_home

        self.data.ctrl[self._arm_act] = self.data.qpos[self._arm_qpos]
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = {
            "is_success": False,
            "drawer_opening": self.drawer_opening,
            "grasped": False,
        }
        return observation, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        self._apply_action(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._maybe_grasp()
        mujoco.mj_forward(self.model, self.data)

        reward, terms = self.compute_reward(action)
        terminated = bool(self._terminated())

        self._prev_action = action.copy()
        self._elapsed += 1

        observation = self._get_obs()
        info = {
            "is_success": bool(self._success),
            "drawer_opening": self.drawer_opening,
            "grasped": bool(self._grasped),
            "tool_to_handle": float(
                np.linalg.norm(self.handle_position - self.ee_position)
            ),
            "reward_terms": terms,
        }

        if self.render_mode == "human":
            self.render()

        # truncated stays False: the TimeLimit wrapper owns the time budget.
        return observation, float(reward), terminated, False, info

    # --------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            camera = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "task_view"
            )
            self._renderer.update_scene(self.data, camera=camera if camera >= 0 else -1)
            return self._renderer.render()

        if self.render_mode == "human":
            from mujoco import viewer as mj_viewer

            if self._viewer is None:
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------- helpers
    def config(self) -> dict:
        """Everything that defines this environment, for the run record."""
        return {
            "frame_skip": self.frame_skip,
            "control_hz": 1.0 / (self.frame_skip * self.model.opt.timestep),
            "max_joint_delta": self.max_joint_delta,
            "grasp_threshold": self.grasp_threshold,
            "randomize": self.randomize,
            "success_displacement": SUCCESS_DISPLACEMENT,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "reward": asdict(self.reward_config),
        }


def make_drawer_env(
    render_mode: str | None = None,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    **kwargs,
) -> gym.Env:
    """Factory used by every script: the env plus its time limit.

    TimeLimit is what turns the 6-second budget into truncated=True. Keeping it
    outside the env is the reason your evaluation can change the budget without
    touching the task definition.
    """
    env = DrawerEnv(render_mode=render_mode, **kwargs)
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
