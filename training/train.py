"""Main training script for Panda pick-and-place with curriculum learning.

Usage:
    python -m training.train --config configs/sac_her.yaml
    python -m training.train --config configs/td3_her.yaml --seed 123
    python -m training.train --config configs/sac_her.yaml --total_timesteps 100000
"""

import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from agents.builder import build_agent
from agents.callbacks import CurriculumCallback, SuccessRateCallback
from envs.env_factory import make_eval_env, make_vec_env
from utils.config import load_config
from utils.logger import setup_logger
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent for Panda pick-and-place")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--n_envs", type=int, default=None, help="Override number of parallel envs")
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for logging")
    parser.add_argument("--no_curriculum", action="store_true", help="Disable curriculum learning")
    return parser.parse_args()


def train(config, run_name: str = "default"):
    """Run the training pipeline.

    Args:
        config: Training configuration dictionary.
        run_name: Name for this training run (used in log dirs).
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    env_config = config.get("env", {})
    curriculum_config = config.get("curriculum", {})
    curriculum_enabled = curriculum_config.get("enabled", True)
    initial_difficulty = curriculum_config.get("initial_difficulty", 0.0) if curriculum_enabled else 1.0

    logger = setup_logger(log_dir=f"results/logs/{run_name}")
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Algorithm: {config['algorithm']}, Seed: {seed}")
    logger.info(f"Curriculum: {'enabled' if curriculum_enabled else 'disabled'}")

    # Create environments
    logger.info("Creating training environments...")
    train_env = make_vec_env(
        n_envs=env_config.get("n_envs", 4),
        seed=seed,
        reward_type=env_config.get("reward_type", "sparse"),
        control_type=env_config.get("control_type", "ee"),
        initial_difficulty=initial_difficulty,
        max_episode_steps=env_config.get("max_episode_steps", 50),
    )

    logger.info("Creating evaluation environment...")
    eval_env = make_eval_env(
        seed=seed,
        reward_type=env_config.get("reward_type", "sparse"),
        control_type=env_config.get("control_type", "ee"),
        difficulty=initial_difficulty if curriculum_enabled else 1.0,
        max_episode_steps=env_config.get("max_episode_steps", 50),
    )

    # Build agent
    logger.info("Building agent...")
    config["tensorboard_log"] = f"results/logs/{run_name}"
    model = build_agent(config, train_env)

    # Setup callbacks
    save_path = f"results/models/{run_name}"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    callbacks = []

    # Evaluation callback (save best model)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=f"results/logs/{run_name}",
        eval_freq=config.get("eval_freq", 5000),
        n_eval_episodes=config.get("n_eval_episodes", 20),
        deterministic=True,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get("checkpoint_freq", 25000),
        save_path=save_path,
        name_prefix="checkpoint",
    )
    callbacks.append(checkpoint_callback)

    # Curriculum callback
    if curriculum_enabled:
        curriculum_eval_env = make_eval_env(
            seed=seed + 2000,
            reward_type=env_config.get("reward_type", "sparse"),
            control_type=env_config.get("control_type", "ee"),
            difficulty=initial_difficulty,
            max_episode_steps=env_config.get("max_episode_steps", 50),
        )
        curriculum_callback = CurriculumCallback(
            eval_env=curriculum_eval_env,
            eval_freq=curriculum_config.get("eval_freq", 10000),
            n_eval_episodes=curriculum_config.get("n_eval_episodes", 20),
            promotion_threshold=curriculum_config.get("promotion_threshold", 0.6),
            demotion_threshold=curriculum_config.get("demotion_threshold", 0.1),
            difficulty_step=curriculum_config.get("difficulty_step", 0.1),
            patience=curriculum_config.get("patience", 3),
            initial_difficulty=initial_difficulty,
        )
        callbacks.append(curriculum_callback)

    # Success rate tracking
    callbacks.append(SuccessRateCallback())

    # Train
    total_timesteps = config.get("total_timesteps", 500000)
    logger.info(f"Starting training for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        tb_log_name=run_name,
    )

    # Save final model
    final_path = f"{save_path}/final_model"
    model.save(final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")

    train_env.close()
    eval_env.close()

    return model


def main():
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.seed is not None:
        config["seed"] = args.seed
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.n_envs is not None:
        config.setdefault("env", {})["n_envs"] = args.n_envs
    if args.no_curriculum:
        config.setdefault("curriculum", {})["enabled"] = False

    # Determine run name
    run_name = args.run_name
    if run_name is None:
        algo = config["algorithm"]
        curriculum = "curriculum" if config.get("curriculum", {}).get("enabled", True) else "no_curriculum"
        reward = config.get("env", {}).get("reward_type", "sparse")
        seed = config.get("seed", 42)
        run_name = f"{algo}_her_{curriculum}_{reward}_s{seed}"

    train(config, run_name)


if __name__ == "__main__":
    main()
