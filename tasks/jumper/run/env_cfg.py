"""Environment config for jumper.run.

`jumper.flat`'s skeleton and every one of its parameters, aimed past its
measured ceiling. The two deliberate departures are named where they are made:
`RUN_TOP_LIN` below, and the `forward_progress` term at the end of `env_cfg`.
Everything else matches flat **on purpose** -- the question this task answers is
"how fast can the robot go", and any other difference would confound the
comparison against flat's 0.80 m/s.

## The staged ceiling

`jumper.flat`'s 0.80 m/s is "the fastest a trained policy has been measured
holding", with nothing tested above it (see the note at `FLAT_TOP_LIN` in
`tasks/jumper/flat/env_cfg.py`). This task probes upward in stages: train at one
ceiling until the command curriculum tops out or stalls, then raise the constant
and `--resume`. The ladder keeps its shared shape (`curriculum.ladder`), so each
stage re-scales the same rungs and the checkpoint's level still means something
-- the top rung of the old ladder is not the top of the new one, and the policy
earns the widened range from where it was.

A stage that stalls is the measurement, not a failure: `Curriculum/command/
lin_err` stuck above `lin_err_bar` at a rung below the top is a speed the policy
cannot track, and the previous stage's ceiling is the reachable candidate --
exactly the diagnostic `jumper.flat`'s comment prescribes for its own 0.80.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..common.mdp.curriculum import (
    STD_ANG_RATIO,
    STD_LIN_RATIO,
)
from ..common.tof import tof_sensor
from ..common.velocity_env import velocity_env_cfg
from .mdp.rewards import forward_progress

#: The linear ceiling of the current stage, in m/s.
#:
#: **Stage B: 2.0, resumed from stage A's final checkpoint -- a measured
#: negative result.** Stage A (1.2) ran 2026-10-01 02:00--06:07 on an AutoDL
#: RTX 4090 D (10000 iterations, 4096 envs, 4h02m wall): the curriculum reached
#: the top rung in ~38 minutes and `lin_err` settled at 0.18 against the 0.21
#: bar. The final checkpoint's command sweep (eval_top_speed.py, 512 envs,
#: warp/cuda, eval/run_stageA_final.json) tracks commands up to 1.6 and
#: plateaus at **1.91 m/s** (vx 1.906 and 1.905 at commands 2.2 and 2.4, 100%
#: survival) -- 2.4x `jumper.flat`'s measured 0.80. The planned 1.6 stage was
#: skipped on that measurement: the policy already tracks 1.6 without ever
#: being trained past 1.2, so a 1.6 stage would re-measure a known result. 2.0
#: was the first ceiling above the measured plateau -- the first stage whose
#: answer was not already in.
#:
#: Stage B ran 2026-10-01 06:17--10:14 on the same machine (10000 resumed
#: iterations, 3h57m wall, run directory `2026-10-01_06-17-50`). The widening
#: to 2.0 re-scaled the restored top rung on the first resumed iteration, and
#: the policy broke **immediately and irreversibly**: the same command sweep
#: shows every stage-B checkpoint from `model_10050.pt` (50 iterations in)
#: onward terminating en masse from a standing start at *every* pinned command
#: 0.4--2.4, in clean play conditions where stage A's `model_9999.pt` walks
#: indefinitely (eval/run_stageB_final.json; A re-verified in the same session
#: at cmd 1.2, vx 1.199, 100% survival). In the training distribution itself
#: (train mode, sampler untouched, 512 envs, 60 s) falls per env-minute climb
#: monotonically through the stage: A-final 17.4, B-10050 27.0, B-15000 37.0,
#: B-final 37.8. The weights are clean (no NaN/Inf in actor, critic or
#: normalizer); the behaviour is the failure, not the checkpoint format. The
#: curriculum's `lin_err` read 0.18 against the 0.63 bar to the end and never
#: saw the collapse: it is sampled only at resets from episodes that lived at
#: least `min_steps`, normalised per command-resample window, so a population
#: that sprints and crashes still reads as tracking. Two compounding causes
#: are recorded for the next attempt: the top rung re-scaled by the new
#: ceiling hit the policy with commands past its measured plateau on iteration
#: one (a re-climb from the old top rung's *range*, not the old level number,
#: would have ramped instead), and `forward_progress` keeps paying for speed
#: through the stumble phase of a sprint-crash cycle. The verifiable top speed
#: therefore stays **stage A's 1.91 m/s**, and the 2.4 stage was not started:
#: it would have inherited a broken policy and repeated the measurement.
#:
#: One trade was recorded in the stage-A sweep and deliberately accepted: at
#: commands <= 0.6 the polished policy terminates en masse (survival 0--0.4%,
#: where iteration 2300 still walked 0.6 at 100%). High-speed specialisation
#: ate low-speed stability between iterations 2300 and 9999. This task's
#: question is top speed; a task that needs both would add a low-speed
#: retention rung rather than accept this. Stage B reads as the same erosion
#: crossing the whole command range once the ceiling passed the plateau.
#:
#: Stage C (2026-10-01 12:18--12:47, same machine, same ceiling 2.0, resumed
#: from stage A's model_9999) re-ran stage B with both of its recorded causes
#: removed, and **collapsed the same way**. The re-climb used
#: `MJRL_COMMAND_LEVEL=1`: at this ceiling the ladder's ranges are (1.0, 1.4,
#: 2.0) plus the precision rung, so level 1 put the policy on a 1.4 range it
#: measurably holds (vx 1.40 at cmd 1.4, 100% survival in the stage-A sweep)
#: and the performance gate plus dwell ramped it upward -- which worked
#: exactly as designed (level 1 -> 3 in 26 minutes, stage A took 38). And
#: `forward_progress` was gated to pay only while the planar velocity is
#: actually on the command (see the term's docstring), so the sprint-crash
#: cycle paid nothing (`Episode_Reward/forward_progress` read ~0.0002). The
#: collapse came anyway: falls per env-minute in the training distribution
#: (same probe as stage B, eval/debug_train_dist_stageC.py) went 28.75 at
#: model_10050 to 37.72 at model_11000 -- stage B's 27.0 -> 37.8 almost to
#: the digit -- and the run was killed at that red line, ~1050 iterations
#: in. Two independent attempts at ceiling 2.0 now agree, so the driver is
#: the ceiling itself: commands past the 1.91 plateau in the *training*
#: distribution (stochastic policy, disturbances, 20% of commands at the top
#: of the range) destabilise the policy, and the curriculum's error sampling
#: cannot see it because episodes too short to reach `min_steps` never
#: count. The gate stays -- a perverse incentive removed is removed -- but
#: it was not the binding constraint. The verifiable top speed remains
#: **stage A's 1.91 m/s**, and raising this constant again wants a mechanism
#: that keeps unattainable commands out of the training distribution, not
#: another reward patch.
#:
#: Raise only after the curriculum has topped out, and record the outcome of
#: each stage here with the machine it ran on, per the repository's rule that
#: measurements belong next to the decision.
RUN_TOP_LIN = 2.0

#: The angular ceiling. Not what this task probes, so it stays at flat's value:
#: yaw tracking was measured unsaturated up to 1.00 rad/s there.
RUN_TOP_ANG = 1.0

#: Weight of the `forward_progress` term, microduck's running weight. Against
#: the tracking terms' combined 4.0 it is the same order, so the term can
#: bargain with "sit exactly on the command" without drowning it -- the tracking
#: terms are still the curriculum's guide and the source of direction control.
FORWARD_PROGRESS_WEIGHT = 5.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        play: replay mode -- observation noise and external disturbances off,
            longer episodes.
    """
    cfg = velocity_env_cfg(
        controls=Path(__file__).parent / "controls.yaml",
        asset=asset,
        play=play,
        # ── Identical to `jumper.flat`, deliberately ──
        # Each value's provenance is documented at flat's call site; they are
        # restated here because a parameter this task does not write down is a
        # parameter it silently inherits, which is the failure
        # `velocity_env_cfg` raises rather than defaulting for.
        foot_target_height=0.025,
        action_rate_weight=-0.1,
        pose_weight=1.0,
        command_lin_ceiling=RUN_TOP_LIN,
        command_ang_ceiling=RUN_TOP_ANG,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
    )

    # ── Shared with `jumper.flat`, for the same reason flat sets them ──
    # The measurements behind all three are quoted at flat's `env_cfg`; this
    # task is flat aimed at a higher ceiling, so it keeps flat's marking.
    cfg.rewards["foot_clearance"].weight = -4.0
    cfg.rewards["foot_swing_height"].weight = -3.0
    cfg.commands["twist"].rel_standing_envs = 0.2

    # ── The departure: pay for forward speed itself ──
    # Tracking pays for being *on* the command; this pays for being *fast*,
    # linearly up to the command and flat past it, gated so environments told
    # to stand are not paid for moving. The cap at the command is measured,
    # not aesthetic: the uncapped microduck shape let the policy profit from
    # ignoring commands entirely (see the function's docstring for the numbers
    # from the 2026-10-01 stage-A attempt).
    cfg.rewards["forward_progress"] = RewardTermCfg(
        func=forward_progress,
        weight=FORWARD_PROGRESS_WEIGHT,
        params={"command_name": "twist", "cap": RUN_TOP_LIN},
    )

    # The dToF, replay only, as in the other locomotion tasks.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg
