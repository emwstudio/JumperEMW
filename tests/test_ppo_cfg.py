"""The jumper tasks' PPO configs are independent, and each states its own numbers.

Two silent failures are pinned here, both of the shape "training runs, converges
to something, and the number you tuned was not the one in effect".

1. **Two tasks aliasing one mutable config.** `jumper_ppo_baseline` builds
   `symmetry_cfg` and `distribution_cfg` as dict literals. Hoisting either to a
   module constant -- an obvious tidy-up, and the file used to do exactly that
   with `HIDDEN_DIMS` -- would have every task share one object, so tuning one
   task's mirror loss retunes the other three. Nothing raises; the run is just
   answering a question nobody asked.

2. **A task silently inheriting the numbers it is supposed to own.** A fifth jumper
   task copy-pasted from a fourth, with the hyper-parameter block dropped,
   compiles and trains. It then moves whenever someone edits a default in
   `common/ppo.py` for an unrelated task.

Both tests carry a control group, because both assertions can pass vacuously --
the first if the configs are compared by value rather than by identity, the
second if the required-keyword set is empty.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import tasks
from tasks.jumper.common.ppo import jumper_ppo_baseline

#: The jumper tasks built on `jumper_ppo_baseline`.
JUMPER_TASKS = ("jumper.flat", "jumper.tripod", "jumper.tetrapod", "jumper.ripple",
                "jumper.run")

#: The keywords every jumper task must pass for itself.
#:
#: Not "every argument the constructor takes" -- that would make adding an
#: optional knob a four-file change for no benefit. This is the set where a task
#: silently getting someone else's value is a real experiment-invalidating
#: outcome: the network shape, and the PPO settings that decide what is learned
#: rather than how it is plumbed.
REQUIRED_KWARGS = frozenset({
    "experiment_name",
    "actor_hidden_dims",
    "critic_hidden_dims",
    "init_std",
    "entropy_coef",
    "learning_rate",
    "desired_kl",
    "gamma",
    "lam",
    "num_learning_epochs",
    "num_mini_batches",
    "num_steps_per_env",
    "max_iterations",
})


def _rl_cfg_path(task_id: str) -> Path:
    """`jumper.tripod` -> `tasks/jumper/tripod/rl_cfg.py`, the same mapping the
    registry uses: a task id is its module path."""
    return Path("tasks", *task_id.split("."), "rl_cfg.py")


def _baseline_call_keywords(source: str) -> set[str]:
    """The keywords passed to `jumper_ppo_baseline(...)` anywhere in `source`.

    Parsed rather than imported so that what is checked is what the file *says*.
    A test that imported and compared values would pass just as happily on a file
    that passed nothing and inherited every default, which is the case this is
    here to catch.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name == "jumper_ppo_baseline":
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    return set()


@pytest.mark.parametrize("task_id", JUMPER_TASKS)
def test_task_states_its_own_hyperparameters(task_id: str) -> None:
    """Every jumper task passes the numbers it owns, rather than inheriting them."""
    path = _rl_cfg_path(task_id)
    assert path.is_file(), f"{path} does not exist"
    passed = _baseline_call_keywords(path.read_text())
    assert passed, (
        f"{path} never calls jumper_ppo_baseline with keywords -- either it does "
        f"not build its config here any more, or the parse above needs updating"
    )
    missing = REQUIRED_KWARGS - passed
    assert not missing, (
        f"{path} does not state {sorted(missing)}. Each jumper task owns its own "
        f"hyper-parameters; inheriting these silently ties this task to an edit "
        f"made for another one. State the value, even if it equals the default."
    )


def test_required_kwargs_is_not_empty() -> None:
    """Control for the test above, which passes trivially on an empty set.

    Also pins the set against `jumper_ppo_baseline`'s actual signature: a keyword
    required here but renamed there would make every task fail with a confusing
    message, and one *removed* there would leave a requirement nothing can meet.
    """
    import inspect

    assert REQUIRED_KWARGS, "an empty requirement makes the test above vacuous"
    params = set(inspect.signature(jumper_ppo_baseline).parameters)
    unknown = REQUIRED_KWARGS - params
    assert not unknown, (
        f"REQUIRED_KWARGS names {sorted(unknown)}, which jumper_ppo_baseline does "
        f"not accept -- the constructor was changed without this list"
    )


def test_agent_cfgs_do_not_alias_mutable_state() -> None:
    """Mutating one task's config must not reach another's.

    The dicts are the exposure: `symmetry_cfg` and `distribution_cfg` are the two
    places a shared object would hide, because a dataclass field holding a dict
    is copied by reference.
    """
    cfgs = {t: tasks.load_agent_cfg(t) for t in JUMPER_TASKS}
    victim, *others = JUMPER_TASKS

    before = {
        t: (
            cfgs[t].algorithm.symmetry_cfg["mirror_loss_coeff"],
            cfgs[t].actor.distribution_cfg["init_std"],
            cfgs[t].algorithm.entropy_coef,
        )
        for t in others
    }

    cfgs[victim].algorithm.symmetry_cfg["mirror_loss_coeff"] = 99.0
    cfgs[victim].actor.distribution_cfg["init_std"] = 99.0
    cfgs[victim].algorithm.entropy_coef = 99.0

    for t in others:
        assert (
            cfgs[t].algorithm.symmetry_cfg["mirror_loss_coeff"],
            cfgs[t].actor.distribution_cfg["init_std"],
            cfgs[t].algorithm.entropy_coef,
        ) == before[t], (
            f"changing {victim} changed {t}: the two share a mutable config "
            f"object. Build it inside jumper_ppo_baseline, not at module level."
        )


def test_aliasing_check_can_fail() -> None:
    """Control for the test above.

    That test would pass on any four configs that merely *happen* to be equal, so
    it has to be shown capable of failing. Two references to one config are what
    the bug looks like; the same assertion must catch it.
    """
    shared = jumper_ppo_baseline(experiment_name="a")
    alias = shared  # what a module-level dict would produce, in miniature

    alias.algorithm.symmetry_cfg["mirror_loss_coeff"] = 99.0
    assert shared.algorithm.symmetry_cfg["mirror_loss_coeff"] == 99.0, (
        "the aliasing assertion cannot detect aliasing, so the test above proves "
        "nothing"
    )


def test_separate_calls_do_not_alias() -> None:
    """The positive half of the control: two calls give two independent dicts."""
    a = jumper_ppo_baseline(experiment_name="a")
    b = jumper_ppo_baseline(experiment_name="b")

    assert a.algorithm.symmetry_cfg is not b.algorithm.symmetry_cfg
    assert a.actor.distribution_cfg is not b.actor.distribution_cfg
    assert a.actor.distribution_cfg is not a.critic.__dict__.get("distribution_cfg")


@pytest.mark.parametrize("task_id", JUMPER_TASKS)
def test_experiment_name_is_the_task_id(task_id: str) -> None:
    """The log path's middle segment is the task id.

    `tests/test_log_layout.py` covers the path `train.py` composes; this covers
    the one input it takes from the task. A task that copy-pasted another's
    `experiment_name` would write its checkpoints into the other's directory,
    where `--resume` would then find them.
    """
    assert tasks.load_agent_cfg(task_id).experiment_name == task_id
