"""`jumper.run`'s own reward: pay for forward speed, up to the command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def forward_progress(
    env: "ManagerBasedRlEnv",
    cap: float,
    command_name: str,
    command_threshold: float = 0.05,
    gate_floor: float = 0.03,
    gate_ratio: float = 0.3,
) -> torch.Tensor:
    """`clamp(vx, 0, min(cmd_x, cap)) / cap`, gated on a forward command and on
    actually tracking it.

    **Linear in `vx` up to the command, flat past it.** The velocity-tracking
    terms pay `exp(-err^2 / std^2)`, whose optimum is to sit exactly on the
    command; a task asking "how fast can this robot go" adds a term that keeps
    paying for speed, so the optimum at the top of the command range is the
    ceiling rather than a comfortable walk.

    **The payment is capped at the command, and that cap is measured, not
    aesthetic.** The first version of this term paid `clamp(vx, 0, cap) / cap`
    with no regard to the command (microduck-playground's shape), on the
    reasoning that overshoot should keep paying. Run on this task (stage A,
    2026-10-01, AutoDL 4090D), the policy took the exploit: the exp-kernel
    tracking loss is bounded by the tracking weight, so once tracking is fully
    sacrificed, *more* speed is pure profit. By iteration ~1800 the policy had
    collapsed to "always run forward at ~0.4 m/s whatever the command" --
    `Curriculum/command/lin_err` sat at 0.44-0.46 against `lin_err_bar` 0.21,
    **worse than a motionless robot scores (0.71 * range ~= 0.18 at the level-0
    range 0.25, per the calibration in `common/mdp/curriculum.py`)**, and the
    command curriculum could never promote. Capping the payment at `cmd_x`
    removes the profit from overshoot while keeping the linear pull toward the
    command; past the command, tracking owns the gradient again. microduck
    could afford the uncapped shape because its commands were always
    forward-at-the-target; this task's command mix (20% standing, lateral and
    backward envs, and a promotion gate on tracking error) cannot.

    The outer cap prevents a physics glitch from collecting a jackpot: an
    unrewarded-and-unbounded term pays arbitrarily for a numerical excursion
    that has nothing to do with running. Set it to the task's command ceiling,
    which is by construction the fastest this policy is ever asked to go.

    **Gated on tracking, because a sprint-crash cycle must not be paid.**
    Stage B (2026-10-01, ceiling raised 1.2 -> 2.0 on resume) showed what the
    ungated term buys: falls per env-minute in the training distribution
    climbed monotonically from stage A's 17.4 through 27.0 (50 iterations in),
    37.0 and 37.8, while the curriculum's `lin_err` kept reading 0.18 -- the
    policy was collecting this term through the stumble phase of a sprint and
    crash, and every episode too short to reach `min_steps` was invisible to
    the gate that was supposed to notice. So payment now also requires the
    planar velocity to actually be on the command *this step*:

        tracking = ||[vx - cmd_x, vy - cmd_y]|| < gate_floor + gate_ratio * |cmd_xy|

    The tolerance is the curriculum's own bar shape, with its measured
    constants (`CommandRangeCurriculum`'s `gate_floor` / `gate_ratio`): 0.03
    of exploration noise plus 0.3 of the commanded planar speed -- 0.63 at a
    2.0 m/s command, the same bar the curriculum promotes against. A robot
    holding its command sits far inside it (the healthy policy's per-step
    ripple is a few cm/s); a stumbling one is far outside, so the stumble
    phase of a sprint-crash cycle pays exactly nothing. The acceleration
    transient after a resample pays nothing either, which is correct: this
    term is for *holding* speed, and the tracking terms own getting there.
    A hard gate rather than a smooth factor because PPO never differentiates
    the reward -- the only thing a smoother shape would add is a second
    constant to calibrate.

    Stage C (same day) then measured this gate's limit: with it in force and
    the curriculum re-climbed properly from the old range, ceiling 2.0 still
    collapsed the policy (28.75 -> 37.72 falls/env/min, killed at the red
    line). The gate removes a perverse incentive; it cannot make an
    unattainable command attainable. The full record is at `RUN_TOP_LIN`.

    **Gated on the command, because 20% of environments are told to stand.**
    `rel_standing_envs` is 0.2 on this skeleton, and an ungated forward reward
    would pay those environments for walking away from a stand command --
    directly against the tracking terms, which pay them for holding still.
    Environments commanded sideways or backwards also pay nothing: the
    question this task answers is about forward speed, and a pure lateral
    command says nothing about it.

    Reads the body-frame velocity, as `track_yaw_velocity` does for yaw. This is
    training-side information -- nothing here is observed by the actor, so the
    sim2real contract is untouched.
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"command {command_name!r} not found"
    vel_xy = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    cmd_xy = command[:, :2]
    vx, cmd_x = vel_xy[:, 0], cmd_xy[:, 0]
    gate = (cmd_x > command_threshold).float()
    err = torch.linalg.norm(vel_xy - cmd_xy, dim=1)
    tol = gate_floor + gate_ratio * torch.linalg.norm(cmd_xy, dim=1)
    tracking = (err < tol).float()
    paid = torch.minimum(vx.clamp(min=0.0), cmd_x.clamp(min=0.0, max=cap))
    return gate * tracking * paid / cap
