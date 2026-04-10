"""Custom SB3 callbacks for curriculum learning and metrics tracking."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


class CurriculumCallback(BaseCallback):
    """Adaptive curriculum learning callback.

    Monitors evaluation success rate and adjusts environment difficulty:
    - Promote (increase difficulty) when success rate exceeds threshold
      for consecutive evaluations.
    - Demote (decrease difficulty) when success rate drops too low.

    Args:
        eval_env: Evaluation environment (should be at current difficulty).
        eval_freq: Steps between evaluations.
        n_eval_episodes: Episodes per evaluation.
        promotion_threshold: Success rate to trigger difficulty increase.
        demotion_threshold: Success rate to trigger difficulty decrease.
        difficulty_step: Amount to change difficulty on promotion.
        patience: Consecutive successes before promotion.
        initial_difficulty: Starting difficulty level.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        promotion_threshold: float = 0.6,
        demotion_threshold: float = 0.1,
        difficulty_step: float = 0.1,
        patience: int = 3,
        initial_difficulty: float = 0.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.difficulty_step = difficulty_step
        self.patience = patience
        self.current_difficulty = initial_difficulty
        self.consecutive_successes = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Evaluate at current difficulty
        success_rate = self._evaluate()

        # Log metrics
        self.logger.record("curriculum/difficulty", self.current_difficulty)
        self.logger.record("curriculum/success_rate", success_rate)
        self.logger.record("curriculum/consecutive_passes", self.consecutive_successes)

        # Adaptive difficulty adjustment
        if success_rate >= self.promotion_threshold:
            self.consecutive_successes += 1
            if self.consecutive_successes >= self.patience:
                self._increase_difficulty()
                self.consecutive_successes = 0
        elif success_rate < self.demotion_threshold:
            self._decrease_difficulty()
            self.consecutive_successes = 0
        else:
            self.consecutive_successes = 0

        return True

    def _evaluate(self) -> float:
        """Run evaluation episodes and return success rate."""
        successes = 0
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = self.eval_env.step(action)
                done = dones[0]
                if done and infos[0].get("is_success", False):
                    successes += 1
        return successes / self.n_eval_episodes

    def _increase_difficulty(self) -> None:
        """Increase difficulty by one step."""
        old = self.current_difficulty
        self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_step)
        self._apply_difficulty()
        if self.verbose:
            print(
                f"[Curriculum] Difficulty increased: {old:.2f} -> "
                f"{self.current_difficulty:.2f}"
            )

    def _decrease_difficulty(self) -> None:
        """Decrease difficulty by half a step (safety fallback)."""
        old = self.current_difficulty
        self.current_difficulty = max(
            0.0, self.current_difficulty - self.difficulty_step * 0.5
        )
        self._apply_difficulty()
        if self.verbose:
            print(
                f"[Curriculum] Difficulty decreased: {old:.2f} -> "
                f"{self.current_difficulty:.2f}"
            )

    def _apply_difficulty(self) -> None:
        """Propagate difficulty to training and eval environments."""
        # Training env (VecEnv)
        self.training_env.env_method("set_difficulty", self.current_difficulty)
        # Eval env
        self.eval_env.env_method("set_difficulty", self.current_difficulty)


class SuccessRateCallback(BaseCallback):
    """Log per-episode success rate to TensorBoard.

    Tracks a rolling window of episode outcomes and logs the success rate.

    Args:
        window_size: Number of recent episodes to average over.
        verbose: Verbosity level.
    """

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.successes = []

    def _on_step(self) -> bool:
        # Check for episode completion in info dicts
        infos = self.locals.get("infos", [])
        for info in infos:
            if "is_success" in info:
                # Episode just ended (either terminated or truncated)
                maybe_ep_info = info.get("episode")
                if maybe_ep_info is not None or info.get("TimeLimit.truncated", False):
                    self.successes.append(float(info["is_success"]))

        # Log rolling success rate
        if len(self.successes) > 0:
            recent = self.successes[-self.window_size:]
            self.logger.record("rollout/success_rate", np.mean(recent))

        return True
