"""Evaluate a trained model and compute performance metrics.

Usage:
    python -m evaluation.evaluate --model results/models/sac_her_curriculum/best_model.zip
    python -m evaluation.evaluate --model results/models/sac_her_curriculum/best_model.zip --episodes 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC, TD3

from envs.curriculum_env import CurriculumPandaPickAndPlaceEnv
from envs.wrappers import SuccessInfoWrapper
import gymnasium as gym


def evaluate_model(
    model_path: str,
    n_episodes: int = 50,
    difficulty: float = 1.0,
    reward_type: str = "sparse",
    deterministic: bool = True,
    verbose: bool = True,
) -> dict:
    """Evaluate a trained model at a specific difficulty level.

    Args:
        model_path: Path to saved model (.zip).
        n_episodes: Number of evaluation episodes.
        difficulty: Environment difficulty [0, 1].
        reward_type: "sparse" or "dense".
        deterministic: Use deterministic policy.
        verbose: Print progress.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Create environment
    env = CurriculumPandaPickAndPlaceEnv(
        render_mode="rgb_array",
        reward_type=reward_type,
        control_type="ee",
        renderer="Tiny",
        initial_difficulty=difficulty,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    env = SuccessInfoWrapper(env)

    # Load model (auto-detect algorithm from file)
    try:
        model = SAC.load(model_path, env=env)
    except Exception:
        model = TD3.load(model_path, env=env)

    # Run evaluation
    successes = []
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        successes.append(float(info.get("is_success", False)))
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if verbose and (ep + 1) % 10 == 0:
            print(
                f"Episode {ep+1}/{n_episodes} | "
                f"Success rate: {np.mean(successes):.2%} | "
                f"Avg reward: {np.mean(episode_rewards):.2f}"
            )

    env.close()

    metrics = {
        "model_path": model_path,
        "difficulty": difficulty,
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(successes)),
        "success_std": float(np.std(successes)),
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
    }

    if verbose:
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"  Difficulty:       {difficulty:.2f}")
        print(f"  Success rate:     {metrics['success_rate']:.2%} +/- {metrics['success_std']:.2%}")
        print(f"  Mean reward:      {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
        print(f"  Mean ep length:   {metrics['mean_episode_length']:.1f} +/- {metrics['std_episode_length']:.1f}")
        print("=" * 50)

    return metrics


def evaluate_across_difficulties(
    model_path: str,
    difficulties: list = None,
    n_episodes: int = 30,
) -> list:
    """Evaluate model across multiple difficulty levels.

    Returns:
        List of metric dictionaries, one per difficulty level.
    """
    if difficulties is None:
        difficulties = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    all_metrics = []
    for d in difficulties:
        print(f"\n--- Evaluating at difficulty {d:.1f} ---")
        metrics = evaluate_model(
            model_path, n_episodes=n_episodes, difficulty=d, verbose=True
        )
        all_metrics.append(metrics)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--episodes", type=int, default=50, help="Number of eval episodes")
    parser.add_argument("--difficulty", type=float, default=1.0, help="Difficulty level")
    parser.add_argument("--sweep", action="store_true", help="Evaluate across all difficulty levels")
    parser.add_argument("--output", type=str, default=None, help="Save metrics to JSON file")
    args = parser.parse_args()

    if args.sweep:
        metrics = evaluate_across_difficulties(args.model, n_episodes=args.episodes)
    else:
        metrics = evaluate_model(
            args.model, n_episodes=args.episodes, difficulty=args.difficulty
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
