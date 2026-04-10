"""Environment factory for creating vectorized training/evaluation envs."""

from typing import Callable, Dict, Optional

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.curriculum_env import CurriculumPandaPickAndPlaceEnv
from envs.wrappers import SuccessInfoWrapper


def make_env(
    rank: int = 0,
    seed: int = 0,
    reward_type: str = "sparse",
    control_type: str = "ee",
    initial_difficulty: float = 0.0,
    max_episode_steps: int = 50,
    render_mode: str = "rgb_array",
    renderer: str = "Tiny",
) -> Callable:
    """Return a factory function that creates a single environment instance.

    Args:
        rank: Index for this env in the vectorized env.
        seed: Base random seed.
        reward_type: "sparse" or "dense".
        control_type: "ee" or "joints".
        initial_difficulty: Starting curriculum difficulty [0, 1].
        max_episode_steps: Episode truncation length.
        render_mode: Gymnasium render mode.
        renderer: PyBullet renderer ("Tiny" for headless).

    Returns:
        A callable that creates and returns a wrapped environment.
    """

    def _init() -> gym.Env:
        env = CurriculumPandaPickAndPlaceEnv(
            render_mode=render_mode,
            reward_type=reward_type,
            control_type=control_type,
            renderer=renderer,
            initial_difficulty=initial_difficulty,
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = SuccessInfoWrapper(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vec_env(
    n_envs: int = 4,
    seed: int = 42,
    reward_type: str = "sparse",
    control_type: str = "ee",
    initial_difficulty: float = 0.0,
    max_episode_steps: int = 50,
) -> SubprocVecEnv:
    """Create a vectorized environment for training.

    Uses SubprocVecEnv for n_envs > 1 (parallel), DummyVecEnv for n_envs == 1.
    """
    env_fns = [
        make_env(
            rank=i,
            seed=seed,
            reward_type=reward_type,
            control_type=control_type,
            initial_difficulty=initial_difficulty,
            max_episode_steps=max_episode_steps,
        )
        for i in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def make_eval_env(
    seed: int = 0,
    reward_type: str = "sparse",
    control_type: str = "ee",
    difficulty: float = 1.0,
    max_episode_steps: int = 50,
) -> DummyVecEnv:
    """Create a single evaluation environment at fixed difficulty."""
    return DummyVecEnv([
        make_env(
            rank=0,
            seed=seed + 1000,
            reward_type=reward_type,
            control_type=control_type,
            initial_difficulty=difficulty,
            max_episode_steps=max_episode_steps,
        )
    ])
