"""Generate publication-quality training curves and comparison plots.

Usage:
    python -m evaluation.plot_results --log_dir results/logs
    python -m evaluation.plot_results --log_dir results/logs --output results/plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use a clean style
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def load_evaluations(log_dir: str, run_name: str) -> dict:
    """Load evaluation data from SB3 EvalCallback output.

    Args:
        log_dir: Base log directory.
        run_name: Name of the training run.

    Returns:
        Dictionary with 'timesteps', 'results', 'ep_lengths' arrays.
    """
    eval_path = Path(log_dir) / run_name / "evaluations.npz"
    if not eval_path.exists():
        return None
    data = np.load(eval_path)
    return {
        "timesteps": data["timesteps"],
        "results": data["results"],
        "ep_lengths": data["ep_lengths"],
    }


def compute_success_rate(results: np.ndarray, threshold: float = -50.0) -> np.ndarray:
    """Compute success rate from evaluation rewards.

    An episode is successful if total reward > threshold.
    For sparse reward with max 50 steps: success=0, failure=-50.
    """
    return np.mean(results > threshold, axis=1)


def smooth(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_success_rate_comparison(log_dir: str, output_dir: str) -> None:
    """Plot success rate curves for different experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    experiments = {
        "sac_her_curriculum_sparse_s42": ("SAC+HER+Curriculum", "#2196F3"),
        "td3_her_curriculum_sparse_s42": ("TD3+HER+Curriculum", "#FF9800"),
        "sac_her_no_curriculum_sparse_s42": ("SAC+HER (no curriculum)", "#4CAF50"),
    }

    for run_name, (label, color) in experiments.items():
        data = load_evaluations(log_dir, run_name)
        if data is None:
            continue

        timesteps = data["timesteps"]
        success_rates = compute_success_rate(data["results"])

        # Plot with smoothing
        if len(success_rates) > 5:
            smoothed = smooth(success_rates, window=5)
            t_smooth = timesteps[2:-2] if len(timesteps) > 4 else timesteps
            ax.plot(t_smooth[:len(smoothed)], smoothed, label=label, color=color, linewidth=2)
            # Shaded region for raw variance
            ax.fill_between(
                timesteps, success_rates - 0.05, success_rates + 0.05,
                alpha=0.15, color=color,
            )
        else:
            ax.plot(timesteps, success_rates, label=label, color=color, linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Success Rate")
    ax.set_title("Pick-and-Place Success Rate Comparison")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_rate_comparison.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/success_rate_comparison.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/success_rate_comparison.png")


def plot_curriculum_progression(log_dir: str, output_dir: str) -> None:
    """Plot curriculum difficulty progression over training."""
    # This reads from TensorBoard logs
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("TensorBoard not installed, skipping curriculum progression plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    run_dir = Path(log_dir) / "sac_her_curriculum_sparse_s42"
    tb_dirs = list(run_dir.glob("sac_her_*"))
    if not tb_dirs:
        print(f"No TensorBoard logs found in {run_dir}")
        return

    ea = EventAccumulator(str(tb_dirs[0]))
    ea.Reload()

    # Plot difficulty
    if "curriculum/difficulty" in ea.Tags().get("scalars", []):
        events = ea.Scalars("curriculum/difficulty")
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax1.plot(steps, values, color="#2196F3", linewidth=2)
        ax1.set_ylabel("Difficulty Level")
        ax1.set_title("Curriculum Learning Progression")
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)

    # Plot success rate at current difficulty
    if "curriculum/success_rate" in ea.Tags().get("scalars", []):
        events = ea.Scalars("curriculum/success_rate")
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax2.plot(steps, values, color="#4CAF50", linewidth=2)
        ax2.set_ylabel("Success Rate")
        ax2.set_xlabel("Training Steps")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/curriculum_progression.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/curriculum_progression.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/curriculum_progression.png")


def plot_ablation_bar_chart(log_dir: str, output_dir: str) -> None:
    """Generate bar chart comparing final success rates across experiments."""
    experiments = [
        ("sac_her_curriculum_sparse_s42", "SAC+HER\n+Curriculum"),
        ("td3_her_curriculum_sparse_s42", "TD3+HER\n+Curriculum"),
        ("sac_her_no_curriculum_sparse_s42", "SAC+HER\n(no curriculum)"),
    ]

    names = []
    success_rates = []
    stds = []

    for run_name, label in experiments:
        data = load_evaluations(log_dir, run_name)
        if data is None:
            continue
        sr = compute_success_rate(data["results"])
        # Use last 5 evaluations for final performance
        final_sr = sr[-5:] if len(sr) >= 5 else sr
        names.append(label)
        success_rates.append(np.mean(final_sr))
        stds.append(np.std(final_sr))

    if not names:
        print("No evaluation data found for ablation bar chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]
    bars = ax.bar(range(len(names)), success_rates, yerr=stds,
                  color=colors[:len(names)], capsize=5, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, sr in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{sr:.1%}", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Final Success Rate")
    ax.set_title("Ablation Study: Pick-and-Place Performance")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_results.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}/ablation_results.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/ablation_results.png")


def plot_training_reward(log_dir: str, output_dir: str) -> None:
    """Plot mean evaluation reward curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    experiments = {
        "sac_her_curriculum_sparse_s42": ("SAC+HER+Curriculum", "#2196F3"),
        "td3_her_curriculum_sparse_s42": ("TD3+HER+Curriculum", "#FF9800"),
    }

    for run_name, (label, color) in experiments.items():
        data = load_evaluations(log_dir, run_name)
        if data is None:
            continue

        timesteps = data["timesteps"]
        mean_rewards = np.mean(data["results"], axis=1)
        std_rewards = np.std(data["results"], axis=1)

        ax.plot(timesteps, mean_rewards, label=label, color=color, linewidth=2)
        ax.fill_between(
            timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards,
            alpha=0.2, color=color,
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title("Training Reward Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_reward.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/training_reward.png")


def main():
    parser = argparse.ArgumentParser(description="Generate training plots")
    parser.add_argument("--log_dir", type=str, default="results/logs", help="Log directory")
    parser.add_argument("--output", type=str, default="results/plots", help="Output directory")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_success_rate_comparison(args.log_dir, args.output)
    plot_training_reward(args.log_dir, args.output)
    plot_curriculum_progression(args.log_dir, args.output)
    plot_ablation_bar_chart(args.log_dir, args.output)
    print("\nAll plots generated.")


if __name__ == "__main__":
    main()
