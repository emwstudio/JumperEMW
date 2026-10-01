"""The live window is drawn on a thread of its own, from a copy the simulation takes.

Two properties, and both fail without a sound:

1. **A slow frame does not slow the simulation.** `handle.sync()` copies the whole
   model into the viewer -- 5.9 ms on jumper -- and while it ran on the simulation
   thread a replay of jumper.posture on one warp environment went from 92 control
   steps a second headless to 55 with the window at 60 Hz. Nothing reported that;
   the world simply ran slower than real time, and asking for a smoother picture
   made it slower still.
2. **What is drawn is a copy, never live state.** The frame thread works while the
   simulation writes the next step. Drawing live state from there would tear, and a
   torn picture cannot be told apart from a policy doing something strange.

Neither needs a window: the handoff is driven here with a stand-in for MuJoCo's
viewer handle.
"""

from __future__ import annotations

import threading
import time

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="b" pos="0 0 1"><freejoint/>
      <geom name="s1" type="sphere" size="0.2"/>
      <geom name="s2" type="sphere" size="0.1" pos="0.3 0 0"/></body>
  </worldbody>
</mujoco>
"""

#: How long the stand-in's `sync()` takes. Ten times what jumper's real one does,
#: so a simulation thread that waits for it cannot hide inside timing noise.
SLOW_SYNC_S = 0.05


class _SlowHandle:
    """What `LiveViewer` uses of `mujoco.viewer.Handle`, with a sync that is slow on
    purpose and remembers which thread called it."""

    def __init__(self, model) -> None:
        self.user_scn = mujoco.MjvScene(model, 1000)
        self.sync_threads: list[str] = []
        self.viewport = mujoco.MjrRect(0, 0, 640, 480)

    def sync(self, state_only: bool = False) -> None:
        del state_only
        self.sync_threads.append(threading.current_thread().name)
        time.sleep(SLOW_SYNC_S)

    def set_images(self, pairs) -> None:
        pass

    def clear_images(self) -> None:
        pass

    def is_running(self) -> bool:
        return True

    def close(self) -> None:
        pass


def _open_without_a_window(viewer, handle) -> None:
    """Everything `LiveViewer.start()` does except `launch_passive`."""
    model, data = viewer._pick_render_target()
    viewer._model, viewer._data = model, data
    viewer._scratch = mujoco.MjData(model)
    viewer._vopt = mujoco.MjvOption()
    viewer._pert = mujoco.MjvPerturb()
    viewer._catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
    viewer._resolve_draw_limit(model)
    viewer._apply(viewer._take_frame())
    viewer._handle = handle
    viewer._start_frame_thread()


def _native_sim(num_envs: int):
    from mjrl.backend.native_sim import NativeSimulation

    return NativeSimulation(num_envs, None, mujoco.MjModel.from_xml_string(XML), "cpu")


def test_a_slow_frame_does_not_hold_up_the_simulation_thread() -> None:
    """`maybe_sync` -- what the simulation calls after every step -- has to return
    in the time it takes to copy the state, however long drawing takes.

    Run against the synchronous design -- `handle.sync()` called inline from
    `maybe_sync` -- the worst call measured 50.1 ms and the first assertion failed.
    """
    from mjrl.viewer.live import LiveViewer

    sim = _native_sim(3)
    viewer = LiveViewer(sim, env_index=0, fps=1000.0, show_all_envs=True)
    handle = _SlowHandle(viewer._sim.env_mjdata(0)[0])
    try:
        _open_without_a_window(viewer, handle)
        worst = 0.0
        for _ in range(5):
            viewer._last_sync = 0.0  # due now, whatever the clock says
            t = time.perf_counter()
            viewer.maybe_sync()
            worst = max(worst, time.perf_counter() - t)
            time.sleep(0.002)
        assert viewer.taken == 5
        assert worst < SLOW_SYNC_S / 5, (
            f"maybe_sync took {worst * 1e3:.1f} ms against a {SLOW_SYNC_S * 1e3:.0f} ms "
            f"sync: the simulation thread is waiting for the window"
        )

        deadline = time.monotonic() + 2.0
        while viewer.synced == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert viewer.synced >= 1, "no frame was ever drawn"
        # Frames that arrive while one is being drawn replace each other rather
        # than queue, so fewer are drawn than taken -- and none is drawn twice.
        assert viewer.synced <= viewer.taken
        assert handle.sync_threads, "the window was never synced"
        assert set(handle.sync_threads) == {"mjrl-live-viewer-frames"}, (
            f"handle.sync() ran on {sorted(set(handle.sync_threads))}"
        )
    finally:
        viewer.stop()
        sim.close()
    assert viewer._frame_thread is None, "stop() left the frame thread behind"


def test_a_frame_is_a_copy_of_the_state_not_a_view_of_it() -> None:
    """The frame thread reads a frame while the simulation writes the next step,
    so a frame holding a view of live state would change under it.

    The control is the first assertion: `env_mjdata` really does hand out a
    reference, so writing to it is visible -- which is exactly what a frame built
    from views would inherit.
    """
    from mjrl.viewer.live import LiveViewer

    sim = _native_sim(2)
    try:
        viewer = LiveViewer(sim, env_index=0, show_all_envs=True)
        viewer._model, viewer._data = viewer._pick_render_target()
        viewer._draw_order = [1]
        frame = viewer._take_frame()
        before = frame.qpos.copy()

        for i in (0, 1):
            sim.env_mjdata(i)[1].qpos[2] += 1.0
        assert sim.env_mjdata(0)[1].qpos[2] == before[0, 2] + 1.0, (
            "control: env_mjdata is expected to return live state by reference"
        )
        assert np.array_equal(frame.qpos, before), (
            "the frame changed when live state did: it holds a view, not a copy"
        )
    finally:
        sim.close()


def test_the_window_is_filled_from_the_frame_rows_in_draw_order() -> None:
    """Row 0 of a frame is the followed environment and the rest follow
    `_draw_order`; the window and the extra environments have to be filled from
    the right rows, or the camera follows one robot while drawing another's pose.
    """
    from mjrl.viewer.live import LiveViewer

    sim = _native_sim(4)
    try:
        for i in range(4):
            sim.env_mjdata(i)[1].qpos[0] = 10.0 * i  # tell them apart by x
        viewer = LiveViewer(sim, env_index=2, show_all_envs=True)
        viewer._model, viewer._data = viewer._pick_render_target()
        viewer._draw_order = [3, 0]
        frame = viewer._take_frame()
        assert list(frame.qpos[:, 0]) == [20.0, 30.0, 0.0]
        viewer._apply(frame)
        assert viewer._data.qpos[0] == 20.0, "the window shows the wrong environment"
        assert viewer._data is not sim.env_mjdata(2)[1]
    finally:
        sim.close()
