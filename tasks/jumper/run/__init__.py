"""jumper.run -- how fast can this robot go; flat's skeleton aimed at the ceiling.

Built on `jumper.flat`'s configuration -- no gait prior, so no phase clock caps
the cadence -- with two deliberate departures, both in `env_cfg.py`:

- the linear command ceiling is raised past flat's measured 0.80 m/s, in stages,
  to find where tracking can no longer follow;
- a `forward_progress` reward (`mdp/rewards.py`) pays linearly for body-forward
  speed without saturating at the command, so the optimum is "as fast as
  possible" rather than "on command". The velocity-tracking terms stay as the
  curriculum's guide.

It is **not** a fifth control for the four gait tasks -- the extra reward term
and the staged ceiling are two differences against each of them, so it is listed
in `NOT_LOCOMOTION` in `tests/test_task_parity.py`.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the
register call below), its environment config (`env_cfg.py`), its hyper-parameters
(`rl_cfg.py`) and its one reward function (`mdp/`).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.run",
    assets=JUMPER_ASSETS,
    description="jumper hexapod top-speed probe; flat's skeleton plus an unsaturated forward-progress reward",
    tags=("locomotion", "jumper", "run", "speed"),
)
