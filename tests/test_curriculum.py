"""Tests for curriculum learning task and difficulty scaling."""

import numpy as np
import pytest
from panda_gym.pybullet import PyBullet

from envs.curriculum_task import CurriculumPickAndPlace
from envs.curriculum_env import CurriculumPandaPickAndPlaceEnv


@pytest.fixture
def env():
    """Create a test environment."""
    e = CurriculumPandaPickAndPlaceEnv(
        render_mode="rgb_array",
        renderer="Tiny",
        initial_difficulty=0.0,
    )
    yield e
    e.close()


class TestDifficultyScaling:
    """Test that difficulty scaling correctly adjusts sampling ranges."""

    def test_initial_difficulty(self, env):
        assert env.get_difficulty() == 0.0

    def test_set_difficulty(self, env):
        env.set_difficulty(0.5)
        assert env.get_difficulty() == 0.5

    def test_difficulty_clipping_high(self, env):
        env.set_difficulty(1.5)
        assert env.get_difficulty() == 1.0

    def test_difficulty_clipping_low(self, env):
        env.set_difficulty(-0.5)
        assert env.get_difficulty() == 0.0

    def test_low_difficulty_ranges(self, env):
        """At low difficulty, goal range should be small."""
        env.set_difficulty(0.0)
        task = env.task
        # At difficulty 0: max_goal_xy = 0.05
        assert task.goal_range_high[0] <= 0.06
        assert task.goal_range_high[2] <= 0.01  # Almost no lifting

    def test_high_difficulty_ranges(self, env):
        """At high difficulty, goal range should be large."""
        env.set_difficulty(1.0)
        task = env.task
        # At difficulty 1.0: max_goal_xy = 0.30
        assert task.goal_range_high[0] >= 0.25
        assert task.goal_range_high[2] >= 0.15  # Significant lifting


class TestGoalSampling:
    """Test that goal/object sampling respects difficulty."""

    def test_easy_goals_near_table(self, env):
        """At low difficulty, goals should cluster near the table surface."""
        env.set_difficulty(0.0)
        goal_heights = []
        for _ in range(100):
            obs, _ = env.reset()
            goal_heights.append(obs["desired_goal"][2])

        mean_height = np.mean(goal_heights)
        # At difficulty 0, most goals should be near table (z ~= object_size/2)
        assert mean_height < 0.1, f"Mean goal height {mean_height} too high for easy difficulty"

    def test_hard_goals_include_lifting(self, env):
        """At high difficulty, some goals should be above the table."""
        env.set_difficulty(1.0)
        goal_heights = []
        for _ in range(100):
            obs, _ = env.reset()
            goal_heights.append(obs["desired_goal"][2])

        max_height = np.max(goal_heights)
        # At difficulty 1.0, at least some goals should require lifting
        assert max_height > 0.05, f"Max goal height {max_height} too low for hard difficulty"

    def test_difficulty_progression(self, env):
        """Goal range should monotonically increase with difficulty."""
        prev_range = 0.0
        for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
            env.set_difficulty(d)
            current_range = env.task.goal_range_high[0]
            assert current_range >= prev_range, (
                f"Goal range decreased at difficulty {d}: {current_range} < {prev_range}"
            )
            prev_range = current_range


class TestCurriculumIntegration:
    """Test curriculum works with full env step cycle."""

    def test_env_works_after_difficulty_change(self, env):
        """Environment should still function after changing difficulty."""
        env.set_difficulty(0.0)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["observation"].shape == (19,)

        # Change difficulty mid-episode
        env.set_difficulty(1.0)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["observation"].shape == (19,)

    def test_incremental_difficulty(self, env):
        """Simulate the CurriculumCallback's incremental adjustments."""
        for level in np.arange(0.0, 1.1, 0.1):
            env.set_difficulty(level)
            obs, _ = env.reset()
            assert obs is not None
            assert env.get_difficulty() == pytest.approx(min(level, 1.0), abs=1e-6)
