"""Curriculum-aware Panda Pick-and-Place environment.

Mirrors panda-gym's PandaPickAndPlaceEnv but uses CurriculumPickAndPlace task,
exposing set_difficulty() for use with CurriculumCallback during training.
"""

from typing import Optional

import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

from envs.curriculum_task import CurriculumPickAndPlace


class CurriculumPandaPickAndPlaceEnv(RobotTaskEnv):
    """Panda pick-and-place with curriculum learning support.

    Args:
        render_mode: "human" for GUI, "rgb_array" for offscreen rendering.
        reward_type: "sparse" (-1/0) or "dense" (negative distance).
        control_type: "ee" (end-effector) or "joints".
        renderer: "Tiny" (headless) or "OpenGL" (for rgb_array recording).
        initial_difficulty: Starting difficulty in [0, 1].
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        initial_difficulty: float = 0.0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(
            sim,
            block_gripper=False,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type=control_type,
        )
        task = CurriculumPickAndPlace(
            sim,
            reward_type=reward_type,
            initial_difficulty=initial_difficulty,
        )
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

    def set_difficulty(self, level: float) -> None:
        """Set curriculum difficulty level in [0, 1]."""
        self.task.set_difficulty(level)

    def get_difficulty(self) -> float:
        """Get current curriculum difficulty level."""
        return self.task.difficulty
