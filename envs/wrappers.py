"""Gymnasium wrappers for training stability and compatibility."""

import gymnasium as gym
import numpy as np


class SuccessInfoWrapper(gym.Wrapper):
    """Ensure info dict always contains 'is_success' as a float.

    Some SB3 callbacks (e.g., EvalCallback) expect info["is_success"]
    to be present and numeric. This wrapper guarantees that.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["is_success"] = float(info.get("is_success", terminated))
        return obs, reward, terminated, truncated, info


class TimeFeatureWrapper(gym.ObservationWrapper):
    """Append remaining episode time as a feature to observations.

    Useful for sparse-reward tasks where the agent benefits from knowing
    how much time remains in the episode (see SB3 docs).

    Only modifies the 'observation' key of dict observations.
    """

    def __init__(self, env: gym.Env, max_steps: int = 50):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

        # Extend observation space
        obs_space = self.observation_space
        obs_low = obs_space["observation"].low
        obs_high = obs_space["observation"].high
        new_low = np.append(obs_low, 0.0)
        new_high = np.append(obs_high, 1.0)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(new_low, new_high, dtype=np.float32),
            "achieved_goal": obs_space["achieved_goal"],
            "desired_goal": obs_space["desired_goal"],
        })

    def reset(self, **kwargs):
        self.current_step = 0
        return super().reset(**kwargs)

    def observation(self, obs):
        time_feature = np.array(
            [1.0 - self.current_step / self.max_steps], dtype=np.float32
        )
        obs["observation"] = np.append(obs["observation"], time_feature)
        self.current_step += 1
        return obs
