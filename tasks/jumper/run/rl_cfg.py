"""Hyper-parameters for jumper.run.

Flat's numbers, stated in full -- **only this file changes this task's training
hyper-parameters**, and nothing this task does not state is inherited silently
(`tests/test_ppo_cfg.py` checks the set).

One departure: `entropy_coef` is doubled to 0.02. This task's optimum sits at
the edge of what the robot can physically do, and getting there asks the policy
to keep exploring gaits past the walking local optimum flat converges to;
microduck-playground's running task made the same move (0.01 -> 0.02) for the
same reason. Every other value matches flat so that a difference in outcome
reads as the reward and the ceiling, not the optimiser.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.run/<date-time>`, with the outermost segment decided by
    `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.run",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # ── PPO ──
        entropy_coef=0.02,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        num_steps_per_env=24,
        max_iterations=10_000,
    )


def runner_cls() -> type:
    """mjlab's velocity runner, which logs extra velocity tracking metrics, with
    the curriculum levels added to its checkpoints (`common/runner.py`) -- what
    makes `--resume` into a raised ceiling carry on at the right rung."""
    from ..common.runner import CurriculumRunner

    return CurriculumRunner

def play_status():
    """The line `scripts/play.py` prints while replaying: achieved speed against
    commanded. Shared by the velocity tasks; see `common/play.py`."""
    from ..common.play import velocity_status

    return velocity_status
