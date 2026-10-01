"""`forward_progress` must not pay a sprint-crash cycle.

Stage B (2026-10-01, ceiling raised 1.2 -> 2.0 on resume) paid for speed
through the stumble: falls per env-minute climbed 17.4 -> 37.8 while the
curriculum's `lin_err` kept reading 0.18. The fix is the tracking gate --
payment requires the planar velocity to be on the command *this step*. These
cases pin the gate's shape: a robot holding its command is paid exactly as
before, a stumbling or overshooting one collects nothing, and the standing
and lateral command gates from stage A are untouched.

Needs no GPU and no env: the term reads two tensors off the env, so a
namespace stands in for it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="the reward is implemented in torch")

from tasks.jumper.run.mdp.rewards import forward_progress  # noqa: E402

CAP = 2.0


def score(cmd: list[float], vel: list[float], **params) -> float:
    """One environment's payment for one step."""
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([cmd], dtype=torch.float32)
        ),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_lin_vel_b=torch.tensor([vel], dtype=torch.float32)
                )
            )
        },
    )
    return float(forward_progress(env, cap=CAP, command_name="twist", **params))


# ── Paid exactly as before the gate ─────────────────────────────────────


def test_holding_the_command_pays_unchanged() -> None:
    # err = sqrt(0.1² + 0.05²) ≈ 0.112 < tol = 0.03 + 0.3·1.0 = 0.33
    assert score([1.0, 0.0], [0.9, 0.05]) == pytest.approx(0.9 / CAP)


def test_holding_the_ceiling_command_pays_full() -> None:
    assert score([2.0, 0.0], [2.0, 0.1]) == pytest.approx(1.0)


# ── The stage-B leak, closed ─────────────────────────────────────────────


def test_stumble_pays_nothing() -> None:
    # The sprint-crash shape: speed collapsing and the body going sideways,
    # err far past the 0.63 tolerance at a 2.0 command.
    assert score([2.0, 0.0], [0.2, 1.5]) == 0.0


def test_overshoot_past_command_pays_nothing() -> None:
    # Asked for 0.5, barrelling at 1.5: previously paid 0.5/CAP through any
    # phase of the motion; now err = 1.0 > tol = 0.18.
    assert score([0.5, 0.0], [1.5, 0.0]) == 0.0


def test_error_exactly_at_tolerance_pays_nothing() -> None:
    # The gate is strict: err == tol is not tracking.
    tol = 0.03 + 0.3 * 2.0
    assert score([2.0, 0.0], [2.0 - tol, 0.0]) == 0.0


# ── The older gates, untouched ───────────────────────────────────────────


def test_standing_command_pays_nothing() -> None:
    assert score([0.0, 0.0], [0.5, 0.0]) == 0.0


def test_lateral_command_pays_nothing() -> None:
    assert score([0.0, 0.8], [0.0, 0.8]) == 0.0
