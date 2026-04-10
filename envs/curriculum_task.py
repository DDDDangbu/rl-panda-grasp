"""Curriculum-aware Pick-and-Place task with adaptive difficulty scaling.

Extends panda-gym's PickAndPlace with a continuous difficulty parameter [0, 1]
that controls goal/object sampling ranges, enabling curriculum learning.
"""

import numpy as np
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.pybullet import PyBullet


class CurriculumPickAndPlace(PickAndPlace):
    """Pick-and-place task with difficulty-scaled sampling ranges.

    Difficulty levels map to four phases:
        0.0-0.2  Reaching:  goal near object on table, minimal randomization
        0.2-0.5  Pushing:   goal on table surface, wider spawn range
        0.5-0.8  Lifting:   goal above table, requires grasping
        0.8-1.0  Full P&P:  full 3D goal space with max randomization
    """

    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        initial_difficulty: float = 0.0,
    ) -> None:
        # Set difficulty BEFORE super().__init__() because it calls _create_scene
        # which indirectly triggers reset -> _sample_goal / _sample_object
        self._difficulty = np.clip(initial_difficulty, 0.0, 1.0)
        self._table_goal_prob = 0.5  # will be updated by _apply_difficulty

        # Initialize parent with max ranges (we override sampling methods)
        super().__init__(
            sim,
            reward_type=reward_type,
            distance_threshold=distance_threshold,
            goal_xy_range=0.3,
            goal_z_range=0.2,
            obj_xy_range=0.3,
        )
        # Apply difficulty scaling to sampling ranges
        self._apply_difficulty()

    @property
    def difficulty(self) -> float:
        return self._difficulty

    def set_difficulty(self, level: float) -> None:
        """Set task difficulty in [0, 1] and update sampling ranges."""
        self._difficulty = np.clip(level, 0.0, 1.0)
        self._apply_difficulty()

    def _apply_difficulty(self) -> None:
        """Map difficulty level to concrete sampling ranges."""
        d = self._difficulty

        # Linearly interpolate ranges
        max_goal_xy = 0.05 + d * 0.25        # 0.05 -> 0.30
        max_goal_z = d * 0.2                  # 0.00 -> 0.20
        max_obj_xy = 0.05 + d * 0.25         # 0.05 -> 0.30

        self.goal_range_low = np.array([-max_goal_xy, -max_goal_xy, 0.0])
        self.goal_range_high = np.array([max_goal_xy, max_goal_xy, max_goal_z])
        self.obj_range_low = np.array([-max_obj_xy, -max_obj_xy, 0.0])
        self.obj_range_high = np.array([max_obj_xy, max_obj_xy, 0.0])

        # Probability of placing goal on table (easier): decreases with difficulty
        self._table_goal_prob = max(0.0, 0.5 - d * 0.5)  # 0.5 -> 0.0

    def _sample_goal(self) -> np.ndarray:
        """Sample goal position with difficulty-scaled ranges."""
        goal = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < self._table_goal_prob:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Sample object position with difficulty-scaled ranges."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position
