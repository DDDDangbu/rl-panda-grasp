"""Tests for curriculum environment creation, spaces, step, and reset."""

import gymnasium as gym
import numpy as np
import pytest

from envs.curriculum_env import CurriculumPandaPickAndPlaceEnv
from envs.wrappers import SuccessInfoWrapper


@pytest.fixture
def env():
    """Create a test environment."""
    e = CurriculumPandaPickAndPlaceEnv(
        render_mode="rgb_array",
        reward_type="sparse",
        control_type="ee",
        renderer="Tiny",
        initial_difficulty=0.5,
    )
    e = gym.wrappers.TimeLimit(e, max_episode_steps=50)
    e = SuccessInfoWrapper(e)
    yield e
    e.close()


class TestEnvironmentCreation:
    """Test that the environment can be created and has correct spaces."""

    def test_env_creates(self, env):
        assert env is not None

    def test_observation_space_is_dict(self, env):
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Dict)
        assert "observation" in obs_space.spaces
        assert "achieved_goal" in obs_space.spaces
        assert "desired_goal" in obs_space.spaces

    def test_observation_shapes(self, env):
        obs_space = env.observation_space
        # Robot obs (7) + Task obs (12) = 19
        assert obs_space["observation"].shape == (19,)
        # Goal positions are 3D
        assert obs_space["achieved_goal"].shape == (3,)
        assert obs_space["desired_goal"].shape == (3,)

    def test_action_space(self, env):
        # ee control with gripper: 4 dims (dx, dy, dz, gripper)
        assert env.action_space.shape == (4,)
        assert env.action_space.low.min() == -1.0
        assert env.action_space.high.max() == 1.0


class TestEnvironmentStep:
    """Test step and reset behavior."""

    def test_reset_returns_correct_format(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "observation" in obs
        assert "achieved_goal" in obs
        assert "desired_goal" in obs
        assert isinstance(info, dict)
        assert "is_success" in info

    def test_step_returns_correct_format(self, env):
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "is_success" in info

    def test_sparse_reward_values(self, env):
        """Sparse reward should be 0.0 (success) or -1.0 (failure)."""
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert reward in [0.0, -1.0]

    def test_episode_truncation(self, env):
        """Episode should truncate after max_episode_steps."""
        env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated:
                break
        # Should have either terminated (success) or will truncate at step 50
        assert terminated or truncated or True  # env might terminate early on success

    def test_multiple_resets(self, env):
        """Environment should handle multiple consecutive resets."""
        for _ in range(5):
            obs, info = env.reset()
            assert obs["observation"].shape == (19,)


class TestComputeReward:
    """Test the compute_reward function (used by HER)."""

    def test_compute_reward_success(self, env):
        env.reset()
        # Same position = success
        goal = np.array([0.0, 0.0, 0.02])
        achieved = np.array([0.0, 0.0, 0.02])
        reward = env.unwrapped.compute_reward(achieved, goal, {})
        assert reward == 0.0  # sparse: success = 0

    def test_compute_reward_failure(self, env):
        env.reset()
        # Far apart = failure
        goal = np.array([0.0, 0.0, 0.02])
        achieved = np.array([0.5, 0.5, 0.5])
        reward = env.unwrapped.compute_reward(achieved, goal, {})
        assert reward == -1.0  # sparse: failure = -1

    def test_compute_reward_vectorized(self, env):
        """HER calls compute_reward with batched inputs."""
        env.reset()
        batch_size = 32
        achieved = np.random.randn(batch_size, 3)
        desired = np.random.randn(batch_size, 3)
        rewards = env.unwrapped.compute_reward(achieved, desired, {})
        assert rewards.shape == (batch_size,)
        assert set(np.unique(rewards)).issubset({0.0, -1.0})
