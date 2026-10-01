"""Sensor previews in the live window's corner: the right picture, in the right place.

A sensor with a `preview(env_index)` method is drawn in the top-right corner of the
live window (`mjrl/viewer/live.py`, "Sensor insets"). Every way this goes wrong is
quiet:

1. **The picture of another environment.** The window follows one environment and
   the inset has to show that one's sensor; the wrong index gives a perfectly good
   image of a robot that is not on screen.
2. **A rectangle that does not match its image.** MuJoCo's `set_images` raises when
   the two differ, on the frame thread, which then stops drawing -- a window that
   freezes with one line of output buried in the run's log.
3. **A broken preview.** It must not reach the simulation, and it must not fail again
   on every frame after the first.

The robot's own camera goes in the top-left corner, and its mistakes are as quiet: a
name that misses the scene's prefix shows nothing anywhere, an inset kept while the
window looks through that very camera repeats the window, and a picture upside down,
mirrored or of another environment looks like any view from a robot.

None needs a window: the viewer is opened as far as taking a frame, and the handle is
a stand-in that checks what the real one checks.

The speed chip at the top centre (`mjrl/viewer/live.py`, "The speed readout") has
quiet mistakes of its own: a world-frame speed reads 0 for a falling robot the
terminal says is moving, and a chip placed without looking at the corner insets
draws over one of them on a narrow window.

Checked against the viewer broken four ways, one at a time -- the preview always of
environment 0, each rectangle a pixel narrower than its image, a broken preview kept
on, the corner cleared on every empty frame -- and each fails a test here.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="b" pos="0 0 1"><freejoint/><geom type="sphere" size="0.2"/></body>
  </worldbody>
</mujoco>
"""

#: The shape of the jumper's dToF zone image, (rows, columns).
ZONES = (42, 54)


class _Previewing:
    """A sensor that shows which environment it was asked for: every pixel is the index."""

    class cfg:  # noqa: N801 -- mirrors a sensor's `cfg.name`
        name = "fake_tof"

    def __init__(self) -> None:
        self.asked: list[int] = []

    def preview(self, env_index: int):
        self.asked.append(env_index)
        return np.full((*ZONES, 3), env_index, dtype=np.uint8)


class _Handle:
    """What the insets use of `mujoco.viewer.Handle`, checking what `set_images` checks."""

    def __init__(self, width: int, height: int) -> None:
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        self.images: list = []
        self.cleared = 0

    def set_images(self, pairs) -> None:
        for rect, image in pairs:
            if image.shape[:2] != (rect.height, rect.width):
                raise ValueError(f"Image shape {image.shape[:2]} does not match target "
                                 f"shape {(rect.height, rect.width)}")
        self.images = list(pairs)

    def clear_images(self) -> None:
        self.cleared += 1
        self.images = []


def _viewer(sources, env_index: int = 0, num_envs: int = 3, xml: str = XML):
    """A viewer on a native simulation whose sensor context holds `sources`, opened as
    far as taking a frame."""
    from mjrl.backend.native_sim import NativeSimulation
    from mjrl.viewer.live import LiveViewer

    sim = NativeSimulation(num_envs, None, mujoco.MjModel.from_xml_string(xml), "cpu")
    sim._sensor_context = type("Ctx", (), {"raycast_sensors": sources, "camera_sensors": []})()
    viewer = LiveViewer(sim, env_index=env_index, show_all_envs=False)
    viewer._model, viewer._data = viewer._pick_render_target()
    return viewer


def test_the_inset_shows_the_followed_environments_sensor() -> None:
    plain = type("Plain", (), {})()          # a sensor with nothing to show
    shown = _Previewing()
    viewer = _viewer([plain, shown], env_index=2)
    assert viewer._previews == [shown], "only sensors with a preview are drawn"

    frame = viewer._take_frame()
    assert shown.asked == [2]
    assert len(frame.images) == 1 and int(frame.images[0].max()) == 2
    # The control: follow another environment and the picture changes with it.
    other = _viewer([_Previewing()], env_index=1)
    assert int(other._take_frame().images[0].max()) == 1


def test_insets_sit_in_the_top_right_corner_as_whole_blocks() -> None:
    from mjrl.viewer.live import _INSET_FRACTION, _INSET_MARGIN

    viewer = _viewer([])
    handle = _Handle(1280, 720)
    rng = np.random.default_rng(0)
    first = rng.integers(0, 255, (*ZONES, 3), dtype=np.uint8)
    second = rng.integers(0, 255, (*ZONES, 3), dtype=np.uint8)
    viewer._show_insets(handle, (first, second))

    assert len(handle.images) == 3
    (rect_a, block_a), (rect_b, block_b), _ = handle.images
    scale = round(1280 * _INSET_FRACTION / ZONES[1])
    assert (rect_a.width, rect_a.height) == (ZONES[1] * scale, ZONES[0] * scale)
    # Flush with the top-right corner, less the margin, and the second below it.
    assert rect_a.left + rect_a.width == 1280 - _INSET_MARGIN
    assert rect_a.bottom + rect_a.height == 720 - _INSET_MARGIN
    assert rect_b.bottom + rect_b.height == rect_a.bottom - _INSET_MARGIN
    # Every zone a solid block of its own colour, not an interpolated smear.
    np.testing.assert_array_equal(block_a[::scale, ::scale], first)
    np.testing.assert_array_equal(block_a, np.repeat(np.repeat(first, scale, 0), scale, 1))
    np.testing.assert_array_equal(block_b[::scale, ::scale], second)

    # The control for the stand-in: a rectangle that disagrees with its image raises,
    # as MuJoCo's own does, so the assertions above could have failed.
    with pytest.raises(ValueError):
        handle.set_images([(mujoco.MjrRect(0, 0, 10, 10), first)])


def test_insets_that_no_longer_fit_are_cleared_once() -> None:
    viewer = _viewer([])
    handle = _Handle(1280, 720)
    image = np.zeros((*ZONES, 3), dtype=np.uint8)
    viewer._show_insets(handle, (image,))
    assert len(handle.images) == 2  # the dToF and the speed chip

    handle.viewport = mujoco.MjrRect(0, 0, 40, 30)     # smaller than one zone image
    viewer._show_insets(handle, (image,))
    assert handle.images == [] and handle.cleared == 1, "nothing is drawn over the edge"
    viewer._show_insets(handle, ())
    assert handle.cleared == 1, "and it is cleared once, not on every frame"


def test_a_broken_preview_is_dropped_once_and_never_raised(capsys) -> None:
    class Raising(_Previewing):
        def preview(self, env_index):
            raise RuntimeError("no zones")

    class Floats(_Previewing):
        def preview(self, env_index):
            return np.zeros((*ZONES, 3), dtype=np.float32)

    good = _Previewing()
    viewer = _viewer([Raising(), Floats(), good])
    frame = viewer._take_frame()                        # must not raise
    assert len(frame.images) == 1 and viewer._previews == [good]
    out = capsys.readouterr().out
    assert out.count("no longer shown") == 2, out
    viewer._take_frame()
    assert "no longer shown" not in capsys.readouterr().out, "one message, not one per frame"


# ── The onboard camera, top left ─────────────────────────────────────────────

#: A robot that carries the dToF's camera and its own, under the prefix a scene
#: attaches it with, looking along +x at a red box and, turned a quarter to the left,
#: at a green one. Both are off the camera's axis, so a mirrored or upside-down
#: picture differs from the right one. `offwidth` is small so the buffer's limit is
#: reachable.
CAMERAS_XML = """
<mujoco>
  <visual><global offwidth="320" offheight="240"/></visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.4 0.4 0.4 1"/>
    <geom name="red" type="box" pos="2 0.4 1.3" size="0.3 0.3 0.3" rgba="1 0 0 1"/>
    <geom name="green" type="box" pos="-0.6 2 0.7" size="0.3 0.3 0.3" rgba="0 1 0 1"/>
    <body name="b" pos="0 0 1"><freejoint/><geom type="sphere" size="0.2"/>
      <camera name="robot/tof" pos="0.25 0 0.05" xyaxes="0 -1 0 0 0 1"
              resolution="54 42" fovy="42"/>
      <camera name="robot/onboard" pos="0.25 0 0" xyaxes="0 -1 0 0 0 1"
              resolution="40 30" fovy="60"/>
    </body>
  </worldbody>
</mujoco>
"""


class _Stub:
    """Stands in for `_OnboardInset.render`: remembers what it was asked to draw."""

    def __init__(self) -> None:
        self.asked: list = []

    def __call__(self, data, width: int, height: int):
        self.asked.append((data, width, height))
        return np.full((height, width, 3), 7, dtype=np.uint8)


def _onboard_viewer(monkeypatch, *, stub: bool = True, env_index: int = 0, sources=()):
    """A viewer on `CAMERAS_XML` with its onboard inset found, opened as far as
    drawing a frame; with `stub`, rendering is `_Stub` and no GL context is made."""
    from mjrl.backend.native_sim import NativeSimulation
    from mjrl.viewer import live

    if stub:
        monkeypatch.setattr(live, "_offscreen_unavailable", lambda: None)
    sim = NativeSimulation(3, None, mujoco.MjModel.from_xml_string(CAMERAS_XML), "cpu")
    sim._sensor_context = type("Ctx", (), {"raycast_sensors": list(sources),
                                           "camera_sensors": []})()
    viewer = live.LiveViewer(sim, env_index=env_index, show_all_envs=False)
    viewer._model, viewer._data = viewer._pick_render_target()
    viewer._onboard = viewer._find_onboard(viewer._model, stripped=False)
    if stub and viewer._onboard is not None:
        viewer._onboard.render = _Stub()
    return viewer


def _camera(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)


class _WindowHandle(_Handle):
    """`_Handle` with the window's camera, which starts out free as MuJoCo's does."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self.cam = mujoco.MjvCamera()


def test_the_onboard_camera_is_found_under_the_scenes_prefix(monkeypatch) -> None:
    """A scene names it `robot/onboard`; matching the bare name finds nothing, and
    the inset is then missing from every real scene without a word."""
    viewer = _onboard_viewer(monkeypatch)
    assert viewer._onboard is not None
    assert viewer._onboard.camera == _camera(viewer._model, "robot/onboard")
    assert viewer._onboard.aspect == 40 / 30, "the camera's own shape, not the window's"
    # The control: a model with no such camera gets no inset.
    bare = _viewer([])
    assert bare._find_onboard(bare._model, stripped=False) is None


def test_the_onboard_inset_is_top_left_at_the_right_hand_insets_height(monkeypatch) -> None:
    from mjrl.viewer.live import _INSET_FRACTION, _INSET_MARGIN

    viewer = _onboard_viewer(monkeypatch)
    handle = _WindowHandle(1280, 720)
    zones = np.zeros((*ZONES, 3), dtype=np.uint8)
    viewer._show_insets(handle, (zones,))

    assert len(handle.images) == 3
    (tof, _), (onboard, image), _ = handle.images
    assert onboard.height == tof.height, "the two corners line up"
    assert onboard.width == round(tof.height * 40 / 30)
    assert (onboard.left, onboard.bottom + onboard.height) == (_INSET_MARGIN, 720 - _INSET_MARGIN)
    assert int(image.max()) == 7
    data, width, height = viewer._onboard.render.asked[-1]
    assert data is viewer._data, "the window's own state -- the followed environment's"
    assert (width, height) == (onboard.width, onboard.height)

    # With nothing on the right it takes the same fraction of the width.
    viewer._show_insets(handle, ())
    (alone, _), _chip = handle.images
    assert alone.width == round(1280 * _INSET_FRACTION)
    assert abs(alone.height - alone.width * 30 / 40) <= 0.5


def test_the_onboard_inset_goes_while_the_window_looks_through_that_camera(monkeypatch) -> None:
    viewer = _onboard_viewer(monkeypatch)
    handle = _WindowHandle(1280, 720)
    viewer._show_insets(handle, ())
    assert len(handle.images) == 2

    handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    handle.cam.fixedcamid = _camera(viewer._model, "robot/onboard")
    rendered = len(viewer._onboard.render.asked)
    viewer._show_insets(handle, ())
    assert len(handle.images) == 1, "the inset would repeat the window; the chip stays"
    assert len(viewer._onboard.render.asked) == rendered, "and it is not rendered either"

    # The control: another fixed camera -- the dToF's -- is not the onboard one.
    handle.cam.fixedcamid = _camera(viewer._model, "robot/tof")
    viewer._show_insets(handle, ())
    assert len(handle.images) == 2


def test_the_onboard_inset_stays_clear_of_the_right_hand_ones(monkeypatch) -> None:
    from mjrl.viewer.live import _INSET_MARGIN

    viewer = _onboard_viewer(monkeypatch)
    zones = np.zeros((*ZONES, 3), dtype=np.uint8)
    # The dToF at scale 1 leaves 46 px between the two margins; the inset is 56 wide.
    handle = _WindowHandle(130, 200)
    viewer._show_insets(handle, (zones,))
    assert len(handle.images) == 1, "the dToF is kept and the onboard inset left out"
    # The control: wide enough, and it is there, short of the dToF by the margin.
    handle = _WindowHandle(400, 300)
    viewer._show_insets(handle, (zones,))
    (tof, _), (onboard, _) = handle.images
    assert onboard.left + onboard.width <= tof.left - _INSET_MARGIN


def test_the_onboard_inset_never_exceeds_the_offscreen_buffer(monkeypatch) -> None:
    viewer = _onboard_viewer(monkeypatch)
    handle = _WindowHandle(3840, 2160)          # a fifth of it is 845 px, the buffer 320
    viewer._show_insets(handle, ())
    (rect, _), _chip = handle.images
    assert (rect.width, rect.height) == (320, 240)


def test_a_failing_onboard_render_turns_the_inset_off_once(monkeypatch, capsys) -> None:
    viewer = _onboard_viewer(monkeypatch)

    def broken(data, width, height):
        raise RuntimeError("no GL here")

    inset = viewer._onboard
    inset.render = broken
    handle = _WindowHandle(1280, 720)
    zones = np.zeros((*ZONES, 3), dtype=np.uint8)
    viewer._show_insets(handle, (zones,))              # must not raise
    assert len(handle.images) == 2 and viewer._onboard is None
    assert capsys.readouterr().out.count("no longer shown") == 1
    viewer._show_insets(handle, (zones,))
    assert "no longer shown" not in capsys.readouterr().out, "one message, not one per frame"


def test_the_onboard_inset_is_the_followed_environment_through_that_camera(monkeypatch) -> None:
    """Rendered for real and compared with `mujoco.Renderer` through the same camera.

    What this pins is quiet: an image upside down or mirrored, of another camera or
    another environment, all look like a perfectly good view from a robot. The
    followed environment is turned a quarter to the left, towards the green box, so
    it sees something the others do not.

    Measured on Mesa llvmpipe (2026-09-29): identical to the bit; upside down it is
    52/255 away on average and mirrored 15, so the tolerance of 1 is far from both.
    """
    viewer = _onboard_viewer(monkeypatch, stub=False, env_index=2)
    if viewer._onboard is None:
        pytest.skip("no off-screen GL context can be made off the main thread here")
    model = viewer._model
    try:
        mujoco.Renderer(model, height=8, width=8).close()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot create an off-screen context here: {type(e).__name__}: {e}")
    _, turned = viewer._sim.env_mjdata(2)
    turned.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    viewer._apply(viewer._take_frame())

    handle = _WindowHandle(320, 240)
    inset = viewer._onboard
    try:
        viewer._show_insets(handle, ())
    finally:
        inset.close()
    assert viewer._onboard is inset, "rendering failed where a Renderer works"
    rect, image = handle.images[0]

    def through_onboard(qpos):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        r = mujoco.Renderer(model, height=rect.height, width=rect.width)
        try:
            r.update_scene(data, camera="robot/onboard")
            return r.render().astype(int)
        finally:
            r.close()

    expected = through_onboard(turned.qpos)
    got = image.astype(int)
    assert np.abs(got - expected).mean() < 1.0
    green = (got[..., 1] > 150) & (got[..., 0] < 80)
    assert green.sum() > 20, "the followed environment is looking at the green box"

    # The controls: each of the quiet mistakes is far from it.
    _, straight = viewer._sim.env_mjdata(0)
    assert np.abs(got - through_onboard(straight.qpos)).mean() > 10, "another environment"
    assert np.abs(got[::-1] - expected).mean() > 10, "upside down"
    assert np.abs(got[:, ::-1] - expected).mean() > 10, "mirrored"


def test_glvnd_patching_is_turned_off_before_the_window_opens(monkeypatch) -> None:
    """Without it the inset never appears beside a real window on NVIDIA, and no test
    without a window can tell.

    libglvnd patches the GL entry points for the first vendor made current, and the
    window's GLX context then keeps the inset's EGL one from ever being made current
    -- measured on an RTX 5090 D, driver 580, X11: refused beside GLX, current with
    `__GLVND_DISALLOW_PATCHING=1`, which libglvnd reads when a context is first made
    current. So it has to be set by `_find_onboard`, which `start()` calls before
    `launch_passive`; this pins that it is, and that an explicit value is kept.
    """
    monkeypatch.delenv("__GLVND_DISALLOW_PATCHING", raising=False)
    import os

    _onboard_viewer(monkeypatch)
    assert os.environ.get("__GLVND_DISALLOW_PATCHING") == "1"

    monkeypatch.setenv("__GLVND_DISALLOW_PATCHING", "0")
    _onboard_viewer(monkeypatch)
    assert os.environ["__GLVND_DISALLOW_PATCHING"] == "0", "an explicit choice is kept"

    # The control: with no onboard camera there is no second context, and the
    # process's environment is left alone.
    monkeypatch.delenv("__GLVND_DISALLOW_PATCHING")
    bare = _viewer([])
    assert bare._find_onboard(bare._model, stripped=False) is None
    assert "__GLVND_DISALLOW_PATCHING" not in os.environ


# ── The speed readout, top centre ─────────────────────────────────────────


def test_the_speed_chip_shows_the_body_frame_horizontal_speed() -> None:
    """The chip's number is the terminal's: body frame, so a fall counts.

    A world-frame speed is the quiet mistake here -- it reads 0.00 for a robot
    dropping straight down, next to a terminal that says it is moving. The
    window's own MjData is driven directly: no window, no frame thread.
    """
    from mjrl.viewer.live import _INSET_MARGIN, _speed_chip

    viewer = _viewer([])
    handle = _Handle(1280, 720)
    # Upright, moving at (1.2, 0.5) horizontally and 3.0 straight down: the
    # vertical part must not count -- hypot(1.2, 0.5) = 1.3 exactly.
    viewer._data.qvel[:3] = [1.2, 0.5, 3.0]
    viewer._show_insets(handle, ())
    (rect, image), = handle.images
    np.testing.assert_array_equal(image, _speed_chip("1.30 m/s"))
    assert rect.bottom + rect.height == 720 - _INSET_MARGIN, "flush with the top"
    assert rect.left == (1280 - rect.width) // 2, "centred between the margins"

    # The control: pitched 90 degrees about y, a world-vertical velocity is
    # body-forward -- the body frame counts it, the world frame would say 0.
    viewer._data.qpos[3:7] = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]
    viewer._data.qvel[:3] = [0.0, 0.0, 0.77]
    viewer._speed_cache = None
    viewer._speed_smooth = None
    viewer._show_insets(handle, ())
    _, pitched = handle.images[0]
    np.testing.assert_array_equal(pitched, _speed_chip("0.77 m/s"))
    assert not np.array_equal(pitched, _speed_chip("0.00 m/s")), "world frame would read 0"


def test_the_speed_chip_centres_in_the_space_the_insets_leave(monkeypatch) -> None:
    """Between the two corners, not over them: the chip is centred in the gap
    between the onboard inset and the dToF's, and dropped where it cannot fit."""
    from mjrl.viewer.live import _INSET_MARGIN

    viewer = _onboard_viewer(monkeypatch)
    zones = np.zeros((*ZONES, 3), dtype=np.uint8)
    handle = _WindowHandle(1280, 720)
    viewer._show_insets(handle, (zones,))
    assert len(handle.images) == 3
    (tof, _), (onboard, _), (chip, _) = handle.images
    lo = onboard.left + onboard.width + _INSET_MARGIN
    hi = tof.left - _INSET_MARGIN
    assert chip.left == lo + (hi - lo - chip.width) // 2
    top_edge = 720 - _INSET_MARGIN
    span = min(tof.bottom, onboard.bottom)
    assert chip.bottom == span + (top_edge - span - chip.height) // 2, (
        "vertically centred on the insets' column"
    )

    # The control: at 400 px the gap is 140 px and the chip 259 -- it is left
    # out rather than drawn over the dToF, which stays.
    handle = _WindowHandle(400, 300)
    viewer._show_insets(handle, (zones,))
    assert len(handle.images) == 2


def test_no_speed_chip_without_a_floating_base() -> None:
    """A model with no free joint has no robot speed to name, and no chip."""
    welded = XML.replace("<freejoint/>", "")
    viewer = _viewer([], xml=welded)
    assert viewer._speed_readout() is None

    handle = _Handle(1280, 720)
    image = np.zeros((*ZONES, 3), dtype=np.uint8)
    viewer._show_insets(handle, (image,))            # must not raise
    assert len(handle.images) == 1, "the dToF alone; nothing at the top centre"

    # The control: the same scene with its free joint gets one.
    free = _viewer([])
    assert free._speed_readout() is not None


def test_the_speed_chip_is_cached_by_text() -> None:
    """Re-rendered when the number changes, reused when it does not -- the chip
    is redrawn every frame while driving."""
    viewer = _viewer([])
    first = viewer._speed_readout()
    assert viewer._speed_readout() is first
    viewer._data.qvel[0] = 0.5
    viewer._speed_smooth = None               # smoothing aside; it is pinned below
    assert viewer._speed_readout() is not first
    assert viewer._speed_cache[0] == "0.50 m/s"


def test_the_displayed_speed_is_smoothed(monkeypatch) -> None:
    """The raw speed bobs with every stride, which flickered the second decimal
    every frame; the chip shows an exponential moving average instead. The
    terminal's number is not touched."""
    import time

    from mjrl.viewer.live import _SPEED_SMOOTH_S, _speed_chip

    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])

    viewer = _viewer([])
    viewer._data.qvel[0] = 1.3
    assert viewer._speed_smooth is None
    np.testing.assert_array_equal(viewer._speed_readout(), _speed_chip("1.30 m/s"))

    # A jump in the raw value does not jump the display: at the same instant
    # the average has moved nowhere.
    viewer._data.qvel[0] = 0.0
    np.testing.assert_array_equal(viewer._speed_readout(), _speed_chip("1.30 m/s"))
    # The control: with the smoothing state cleared the same state reads raw.
    viewer._speed_smooth = None
    np.testing.assert_array_equal(viewer._speed_readout(), _speed_chip("0.00 m/s"))

    # Half a time constant later it has moved halfway towards the new value.
    viewer._speed_smooth = (now[0], 1.3)
    now[0] += _SPEED_SMOOTH_S * 0.5
    np.testing.assert_array_equal(viewer._speed_readout(), _speed_chip("0.79 m/s"))
