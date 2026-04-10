"""Ablation study runner: batch execution of multiple experiment configurations.

Usage:
    python -m training.ablation
    python -m training.ablation --timesteps 200000  # shorter runs for testing
"""

import argparse
import copy

from utils.config import load_config
from training.train import train


# Ablation experiment definitions
EXPERIMENTS = [
    {
        "name": "E1_sac_her_curriculum",
        "description": "SAC + HER + Curriculum (main baseline)",
        "config_path": "configs/sac_her.yaml",
        "overrides": {},
    },
    {
        "name": "E2_td3_her_curriculum",
        "description": "TD3 + HER + Curriculum (algorithm comparison)",
        "config_path": "configs/td3_her.yaml",
        "overrides": {},
    },
    {
        "name": "E3_sac_her_no_curriculum",
        "description": "SAC + HER without Curriculum",
        "config_path": "configs/sac_her.yaml",
        "overrides": {"curriculum": {"enabled": False}},
    },
    {
        "name": "E4_sac_dense_no_her",
        "description": "SAC + Dense reward, no HER",
        "config_path": "configs/sac_her.yaml",
        "overrides": {
            "env": {"reward_type": "dense"},
            "curriculum": {"enabled": False},
            # Remove HER by not using HerReplayBuffer
        },
    },
    {
        "name": "E5_sac_her_final_strategy",
        "description": "SAC + HER (final strategy) + Curriculum",
        "config_path": "configs/sac_her.yaml",
        "overrides": {"her": {"strategy": "final"}},
    },
]


def run_ablation(timesteps_override=None):
    """Run all ablation experiments sequentially."""
    print("=" * 60)
    print("ABLATION STUDY: RL Panda Pick-and-Place")
    print("=" * 60)

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(EXPERIMENTS)}: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"{'='*60}\n")

        config = load_config(exp["config_path"])

        # Apply experiment-specific overrides (deep merge)
        for key, value in exp["overrides"].items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

        if timesteps_override is not None:
            config["total_timesteps"] = timesteps_override

        # Special handling: E4 removes HER
        is_no_her = exp["name"] == "E4_sac_dense_no_her"

        if is_no_her:
            # For no-HER experiment, we need to modify builder behavior
            config["_no_her"] = True

        try:
            train(config, run_name=exp["name"])
            print(f"\n[OK] {exp['name']} completed successfully.")
        except Exception as e:
            print(f"\n[FAIL] {exp['name']} failed: {e}")
            continue

    print(f"\n{'='*60}")
    print("All ablation experiments completed.")
    print(f"Results saved to: results/logs/ and results/models/")
    print(f"View with: tensorboard --logdir results/logs")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study experiments")
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total timesteps for all experiments (useful for testing)",
    )
    args = parser.parse_args()
    run_ablation(timesteps_override=args.timesteps)


if __name__ == "__main__":
    main()
