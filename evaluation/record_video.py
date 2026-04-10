"""Record demo videos of a trained agent performing pick-and-place.

Usage:
    python -m evaluation.record_video --model results/models/sac_her_curriculum/best_model.zip
    python -m evaluation.record_video --model results/models/sac_her_curriculum/best_model.zip --episodes 10
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import SAC, TD3

from envs.curriculum_env import CurriculumPandaPickAndPlaceEnv
from envs.wrappers import SuccessInfoWrapper
import gymnasium as gym


def record_episodes(
    model_path: str,
    output_dir: str = "results/videos",
    n_episodes: int = 5,
    difficulty: float = 1.0,
    fps: int = 25,
    create_gif: bool = True,
) -> None:
    """Record demo videos of the trained agent.

    Args:
        model_path: Path to saved model (.zip).
        output_dir: Directory to save videos.
        n_episodes: Number of episodes to record.
        difficulty: Environment difficulty level.
        fps: Frames per second for video.
        create_gif: Also create a GIF version (for README).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create environment with rendering enabled
    env = CurriculumPandaPickAndPlaceEnv(
        render_mode="rgb_array",
        reward_type="sparse",
        control_type="ee",
        renderer="OpenGL",
        initial_difficulty=difficulty,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=50)
    env = SuccessInfoWrapper(env)

    # Load model
    try:
        model = SAC.load(model_path, env=env)
    except Exception:
        model = TD3.load(model_path, env=env)

    all_frames = []
    successes = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_frames = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)
            done = terminated or truncated

        success = info.get("is_success", False)
        successes += int(success)
        status = "SUCCESS" if success else "FAIL"
        print(f"Episode {ep+1}/{n_episodes}: {status} ({len(episode_frames)} frames)")

        all_frames.extend(episode_frames)

        # Add a brief pause between episodes (10 blank frames)
        if ep < n_episodes - 1 and episode_frames:
            all_frames.extend([episode_frames[-1]] * 10)

    env.close()

    print(f"\nTotal: {successes}/{n_episodes} successes ({successes/n_episodes:.0%})")

    if not all_frames:
        print("No frames captured. Check renderer settings.")
        return

    # Save MP4
    mp4_path = f"{output_dir}/demo_d{difficulty:.1f}.mp4"
    imageio.mimsave(mp4_path, all_frames, fps=fps)
    print(f"Saved video: {mp4_path}")

    # Save GIF (subsampled for smaller file size)
    if create_gif:
        gif_frames = all_frames[::3]  # Every 3rd frame
        # Resize for GIF if frames are large
        gif_path = f"{output_dir}/demo_d{difficulty:.1f}.gif"
        imageio.mimsave(gif_path, gif_frames, fps=fps // 3, loop=0)
        print(f"Saved GIF: {gif_path}")


def record_difficulty_sweep(
    model_path: str,
    output_dir: str = "results/videos",
    difficulties: list = None,
    n_episodes: int = 3,
) -> None:
    """Record demo videos across multiple difficulty levels."""
    if difficulties is None:
        difficulties = [0.3, 0.6, 1.0]

    for d in difficulties:
        print(f"\n{'='*40}")
        print(f"Recording at difficulty {d:.1f}")
        print(f"{'='*40}")
        record_episodes(
            model_path=model_path,
            output_dir=output_dir,
            n_episodes=n_episodes,
            difficulty=d,
        )


def main():
    parser = argparse.ArgumentParser(description="Record demo videos of trained agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--output", type=str, default="results/videos", help="Output directory")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to record")
    parser.add_argument("--difficulty", type=float, default=1.0, help="Difficulty level")
    parser.add_argument("--sweep", action="store_true", help="Record across difficulty levels")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS")
    parser.add_argument("--no_gif", action="store_true", help="Skip GIF generation")
    args = parser.parse_args()

    if args.sweep:
        record_difficulty_sweep(args.model, args.output)
    else:
        record_episodes(
            model_path=args.model,
            output_dir=args.output,
            n_episodes=args.episodes,
            difficulty=args.difficulty,
            fps=args.fps,
            create_gif=not args.no_gif,
        )


if __name__ == "__main__":
    main()
