"""Agent builder: constructs SAC/TD3 with HER from configuration."""

from typing import Any, Dict

import torch as th
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

import numpy as np


def build_agent(config: Dict[str, Any], env: VecEnv):
    """Build an RL agent (SAC or TD3) with HER from config.

    Args:
        config: Training configuration dictionary.
        env: Vectorized training environment.

    Returns:
        A Stable-Baselines3 model instance (SAC or TD3).
    """
    algorithm = config["algorithm"]
    her_config = config.get("her", {})
    policy_config = config.get("policy", {})

    # HER replay buffer kwargs
    replay_buffer_kwargs = dict(
        n_sampled_goal=her_config.get("n_sampled_goal", 4),
        goal_selection_strategy=her_config.get("strategy", "future"),
    )

    # Policy network architecture
    policy_kwargs = dict(
        net_arch=policy_config.get("net_arch", [256, 256, 256]),
        activation_fn=th.nn.ReLU,
    )

    # Common kwargs shared by SAC and TD3
    common_kwargs = dict(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        policy_kwargs=policy_kwargs,
        learning_rate=config.get("learning_rate", 0.001),
        buffer_size=config.get("buffer_size", 1_000_000),
        batch_size=config.get("batch_size", 256),
        tau=config.get("tau", 0.005),
        gamma=config.get("gamma", 0.95),
        learning_starts=config.get("learning_starts", 1000),
        train_freq=config.get("train_freq", 1),
        gradient_steps=config.get("gradient_steps", 1),
        tensorboard_log=config.get("tensorboard_log", "results/logs"),
        seed=config.get("seed", 42),
        device="auto",
        verbose=1,
    )

    if algorithm == "sac":
        model = SAC(
            **common_kwargs,
            ent_coef=config.get("ent_coef", "auto"),
        )
    elif algorithm == "td3":
        # Add exploration noise for TD3
        noise_config = config.get("action_noise", {})
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=noise_config.get("sigma", 0.1) * np.ones(n_actions),
        )
        model = TD3(
            **common_kwargs,
            action_noise=action_noise,
            policy_delay=config.get("policy_delay", 2),
            target_policy_noise=config.get("target_policy_noise", 0.2),
            target_noise_clip=config.get("target_noise_clip", 0.5),
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'sac' or 'td3'.")

    return model
