"""Watch while training: a window showing one environment live during training.
**Both backends are supported.**

Rendering does not conflict with warp's CUDA graphs. A captured graph is
invalidated when the arrays it holds are **replaced** (reallocated -- which is why
`expand_model_fields` is followed by `create_graph()`), not when they are **read**.
Rendering only reads. mjlab's own `play` renders live on warp for the same reason
(`mjlab/viewer/native/viewer.py`).

The real difference is not capability but **what taking a frame costs the
simulation**:

| | native:cpu | warp:cuda |
|---|---|---|
| Where the state is | Host memory, already an `MjData` | Device memory |
| Taking a frame | Copy qpos/qvel/ctrl/mocap into host arrays | **One** batched device-to-host transfer |
| After DR expanded visual fields | as above | Also fetch those `sim.model` fields |

Everything after that -- `mj_forward`, the other environments, `handle.sync()` --
happens on the viewer's own thread. Frames are taken at most `fps` times a second
(30 by default when training, 60 when replaying), not every physics step.

## A frame is taken on the simulation thread and drawn on another

**Taking** a frame hangs off `step()`, on the simulation's thread: the state is
consistent the moment `step()` returns, so a copy taken then cannot tear. A thread
reading live state on a timer would catch the simulation halfway through writing
it, giving a jittery or interpenetrating picture -- and **no way to tell a
rendering problem from a policy problem**, which would make this tool actively
misleading. So nothing but the simulation thread ever reads live state: the
window draws an `MjData` of its own, filled from the copy.

**Drawing** does not hang off `step()`, because it is not cheap and does not have
to be. `handle.sync()` copies the whole model into the viewer on every frame --
measured 5.9 ms on jumper, against 0.06 ms for the state alone -- and while it ran
on the simulation thread it was the largest cost of watching. jumper.posture on
one warp environment (RTX 5090 D, 2026-09-26) ran 92 control steps a second
headless, 74 with the window at 30 Hz and **55 at 60**: asking for a smoother
picture made the world run slower.
`handle.sync()`, `mj_forward` and `mjv_addGeoms` all release the GIL (measured: a
busy Python loop keeps 98-100% of its rate while another thread calls any of them
back to back), so on a thread of their own they cost the simulation nothing but
the copy.

If drawing falls behind, frames are **dropped, not queued**: the viewer draws the
newest copy there is, and the simulation never waits for it.

**What this does not touch is MuJoCo's own render loop**, and when training on warp
that is most of what watching costs. It redraws the scene continuously on the GPU
the simulation runs on, however rarely a frame arrives, and with 128 environments
in the scene the two contend for the card. jumper.swing at 4096 environments (RTX
5090 D, 2026-09-26), seconds an iteration:

    headless                                   1.69
    window, 30 Hz                              3.92
    window, 1 Hz                               4.09    the frame rate is not the cost
    window, 30 Hz, one environment drawn       1.98    `--viewer-env-num 1`
    window, 30 Hz, drawn on the sim thread     4.09    the design before this one

So moving the drawing off the simulation thread is worth little to a training run,
and the rest of the training cost is the picture itself. Where it is worth a lot
is replay: there the window's share of a step went from 40% to 8% (jumper.posture,
one environment, 60 Hz; 92 headless against 55 with the window before, 127
against 117 after).

How taking a frame hooks in differs, because what can be modified differs:

- **native**: `NativeSimulation.attach_viewer()` is our own code and registers
  explicitly.
- **warp**: `Simulation` is vendored and will not be modified for one viewer.
  Instead `sim.step` is wrapped at **instance** level and restored by `stop()`.
  Only that one instance is affected, never the class.

## Sensor insets

A sensor that can show itself is drawn in the window's top-right corner, for the
followed environment: the jumper's dToF zone image, in every task and scene that
declares it, under `play` and `play --app` alike. The protocol is one method,

    sensor.preview(env_index) -> uint8 array [H, W, 3], row 0 at the top, or None

and the viewer finds such sensors itself, in the simulation's sensor context --
nothing registers, and `scripts/` passes nothing, so the window knows no sensor by
name. What the picture means (colours, scale) is the sensor's business.

The picture is taken with the frame, **on the simulation thread**, so it shows the
same step as the robot beside it, and placed on the frame thread with
`handle.set_images`. Each is scaled by a whole number, so a zone stays a square
block, to about a fifth of the window's width, and further ones stack below. A
preview that raises is dropped with one message: it never reaches the simulation,
and it does not fail again on every frame.

## The onboard camera inset

The robot's own camera -- the one a model names `onboard` -- is drawn in the
**top-left** corner whenever the window is looking through anything else, so the
tracking view and what the robot sees are on screen together. Switch the window to
that camera (`]` / `[` in MuJoCo's viewer) and the inset goes away, since it
would only repeat the window.

It is the same height as the first inset on the right -- the dToF's -- so the two
corners line up, and as wide as the camera's own aspect ratio makes it: the
camera's field of view, not a crop of it to the dToF's shape. With no inset on the
right it takes the same fifth of the window's width.

Unlike a sensor's preview this one is not taken on the simulation thread. It is
rendered **on the frame thread**, from the window's own `MjData`, which that thread
has just filled from the frame -- so it shows the same step as the robot beside it
and costs the simulation nothing. It needs a GL context of its own for that, made
on the first frame that shows the inset and freed when the thread ends. EGL
(mjlab's default on Linux), OSMesa and CGL (macOS) can be made on any thread; a
GLFW one is a window, which GLFW allows only on the main thread and which would be
created while MuJoCo's render thread is calling GLFW -- so under GLFW (Windows, or
`MUJOCO_GL=glfw`) the inset is left out, with one message saying so. It is a view
for the person watching, not an observation: no camera sensor is involved, and
nothing a policy is given changes with it.

### GLX and EGL in one process

On Linux the window's context is GLX (GLFW on X11) and the inset's is EGL, and
libglvnd counts the two as different vendors even when both are NVIDIA's. By default
it **patches** the GL entry points for the first vendor whose context is made
current, and from then on a context of any other vendor cannot be made current on
any thread while one of the first vendor's still is -- `eglMakeCurrent` returns
false and `eglGetError` says `EGL_SUCCESS`, so nothing says why. Measured with a
GLX context current on one thread and an EGL one made current on another (RTX 5090
D, driver 580.105.08, X11, 2026-09-29): EGL alone, current; beside GLX, refused;
beside GLX with `__GLVND_DISALLOW_PATCHING=1`, current, and the inset renders.

libglvnd reads that variable when a context is first made current, not when it
is loaded, so the viewer sets it (unless it is already set) before the window
opens, whenever it has an onboard inset to draw. Unpatched, every GL call goes
through one more indirection, which a window drawing one robot does not notice.
It is too late if something made a context current before the viewer was built --
a camera sensor on the native backend does -- and then the inset fails its first
render and is dropped with one message.

## The speed readout

The followed robot's speed -- the number the play loop prints on the terminal,
the base's horizontal speed in the **body frame** (`scripts/play.py`'s
`_speed_text`; body frame so a fall counts, not just travel) -- is drawn at the
top centre of the window, between the two corners' insets. It is an image
placed by the same `handle.set_images` as the insets: `handle.set_texts` knows
only the window's four corners, and both top corners are taken.

The value is drawn as seven-segment digits and the unit ("m/s") smaller in a
5x7 bitmap -- a digital dash's layout -- both from this module's own tables,
not PIL: the framework does not depend on it (only `tools/readme_media.py`
does, a dev tool), and fixed geometry at fixed scales renders the same on every
machine, so a test can compare the chip by the bit. Edges are anti-aliased by
supersampling. The chip is centred in the space the corner insets leave --
horizontally between them, vertically on their column -- and left out when
there is no room rather than drawn over one of them. A model without a floating
base has no speed worth naming -- a speed then belongs to a joint, not to the
robot -- and gets no chip.

## Usage

```python
from mjrl.viewer.live import LiveViewer

with LiveViewer(env.sim, env_index=0, fps=30):
    runner.learn(...)          # training runs as usual, env 0 shown live
```

Closing the window stops taking frames; training continues unaffected.

> **On macOS, launch through `mjpython`** -- a MuJoCo constraint: the passive
> viewer needs its GLFW context created on the main thread. Plain `python` is fine
> on Linux.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
import time
from typing import Any

from mjrl.viewer import keys

__all__ = ["LiveViewer"]

#: How many environments the live viewer draws at most (including the followed
#: one). Fewer environments than this and all of them are drawn.
#:
#: A fixed number rather than one adapted to frame time: an adaptive number drifts
#: with machine, backend and contact density, so the same command draws a different
#: number of environments on different runs and it becomes hard to say what you are
#: looking at. 128 is enough to see -- more is just a denser field of robots, at a
#: linearly growing cost.
_MAX_DRAW_ENVS: int = 128

#: A corner inset's width as a fraction of the window's, before rounding the scale
#: to a whole number: big enough to read a 54 x 42 zone image, small enough to leave
#: the robot in view.
_INSET_FRACTION = 0.22
#: The largest whole-number scale an inset gets, however wide the window.
_INSET_MAX_SCALE = 12
#: Pixels between an inset and the window's edge, and between two insets.
_INSET_MARGIN = 10

#: The camera drawn in the top-left corner: the name `build_jumper.py` gives the
#: robot's own. A scene attaches the robot under a prefix (`robot/onboard`), so
#: only the name's last segment is compared. A model without one gets no inset.
_ONBOARD_CAMERA = "onboard"

#: The speed chip's unit glyphs, 5x7: the characters of "m/s". The value is
#: seven-segment (`_SEG_DIGITS`); only the unit uses this table.
_SPEED_GLYPHS = {
    " ": (".....", ".....", ".....", ".....", ".....", ".....", "....."),
    "-": (".....", ".....", ".....", "#####", ".....", ".....", "....."),
    ".": (".....", ".....", ".....", ".....", ".....", ".##..", ".##.."),
    "/": ("....#", "...#.", "...#.", "..#..", ".#...", ".#...", "#...."),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", "..#..", "..#..", "..#.."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    "m": (".....", ".....", "##.#.", "#.#.#", "#.#.#", "#...#", "#...#"),
    "s": (".....", ".....", ".####", "#....", ".###.", "....#", "####."),
}

#: The scale the value's seven-segment digits are drawn at: one grid unit in
#: pixels. Five gives 50 x 75 px digits on a 91 px chip, readable in a video
#: frame rather than only at the desk.
_VALUE_SCALE = 5
#: The scale the unit's 5x7 glyphs are drawn at -- smaller than the value, the
#: way a real dash prints its unit.
_UNIT_SCALE = 3
#: The chip's colours: a dark plate, a bright value and a dimmer unit.
_SPEED_BG = (20, 23, 28)
_SPEED_FG = (232, 236, 240)
_SPEED_DIM = (110, 118, 128)
#: Padding inside the chip, px (horizontal, vertical).
_SPEED_PAD = (12, 8)
#: The chip is drawn this many times too large and block-averaged back down, so
#: the segments' and pixels' square edges get anti-aliased instead of staircases.
_SPEED_SUPERSAMPLE = 4
#: The displayed speed's smoothing time constant, seconds. The raw speed bobs
#: with every stride, which flickered the display every frame; 0.3 s stills
#: that while a launch still reads as a launch.
_SPEED_SMOOTH_S = 0.3

#: Seven-segment geometry: stroke thickness and segment length in grid units.
_SEG_T = 2
_SEG_L = 6
#: A digit cell's grid size.
_SEG_W = _SEG_L + 2 * _SEG_T
_SEG_H = 2 * _SEG_L + 3 * _SEG_T
#: Each segment's rectangle on the grid, (x, y, w, h).
_SEG_RECT = {
    "a": (_SEG_T, 0, _SEG_L, _SEG_T),
    "b": (_SEG_T + _SEG_L, _SEG_T, _SEG_T, _SEG_L),
    "c": (_SEG_T + _SEG_L, 2 * _SEG_T + _SEG_L, _SEG_T, _SEG_L),
    "d": (_SEG_T, 2 * _SEG_T + 2 * _SEG_L, _SEG_L, _SEG_T),
    "e": (0, 2 * _SEG_T + _SEG_L, _SEG_T, _SEG_L),
    "f": (0, _SEG_T, _SEG_T, _SEG_L),
    "g": (_SEG_T, _SEG_T + _SEG_L, _SEG_L, _SEG_T),
}
#: Which segments each character lights.
_SEG_DIGITS = {
    "-": "g",
    "0": "abcdef",
    "1": "bc",
    "2": "abdeg",
    "3": "abcdg",
    "4": "bcfg",
    "5": "acdfg",
    "6": "acdefg",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
}


def _speed_chip(text: str):
    """`text` ("2.06 m/s") as a uint8 [H, W, 3] image: the value in
    seven-segment digits, the unit smaller and dimmer in the module's 5x7
    bitmap, bottom-aligned -- a digital dash's layout. Edges anti-aliased by
    supersampling (`_SPEED_SUPERSAMPLE`).

    Deterministic -- fixed geometry at fixed scales with no rasteriser
    involved, so tests compare it by the bit and no two machines render it
    differently.
    """
    import numpy as np

    value, _, unit = text.partition(" ")
    ss = _SPEED_SUPERSAMPLE
    sv, su = _VALUE_SCALE * ss, _UNIT_SCALE * ss
    pad_x, pad_y = _SPEED_PAD[0] * ss, _SPEED_PAD[1] * ss
    digit_h = _SEG_H * sv
    gap = _SEG_T * sv

    def advance(char: str) -> int:
        return (_SEG_T if char == "." else _SEG_W) * sv

    value_w = sum(advance(c) for c in value) + gap * (len(value) - 1)
    unit_cell, unit_gap = 5 * su, su
    unit_w = len(unit) * unit_cell + (len(unit) - 1) * unit_gap
    unit_gap_x = 3 * su
    width = value_w + unit_gap_x + unit_w
    image = np.full((digit_h + 2 * pad_y, width + 2 * pad_x, 3), _SPEED_BG, np.uint8)

    x = pad_x
    for char in value:
        if char == ".":
            image[pad_y + (_SEG_H - _SEG_T) * sv: pad_y + _SEG_H * sv,
                  x: x + _SEG_T * sv] = _SPEED_FG
        else:
            for seg in _SEG_DIGITS.get(char, ""):
                rx, ry, rw, rh = _SEG_RECT[seg]
                image[pad_y + ry * sv: pad_y + (ry + rh) * sv,
                      x + rx * sv: x + (rx + rw) * sv] = _SPEED_FG
        x += advance(char) + gap

    # The unit rides smaller and dimmer, bottom-aligned with the digits.
    x = pad_x + value_w + unit_gap_x
    y0 = pad_y + digit_h - 7 * su
    for char in unit:
        glyph = _SPEED_GLYPHS.get(char, _SPEED_GLYPHS[" "])
        for row, line in enumerate(glyph):
            for col, on in enumerate(line):
                if on == "#":
                    image[y0 + row * su: y0 + (row + 1) * su,
                          x + col * su: x + (col + 1) * su] = _SPEED_DIM
        x += unit_cell + unit_gap

    rows, cols = image.shape[:2]
    return (image.reshape(rows // ss, ss, cols // ss, ss, 3)
            .mean(axis=(1, 3)).round().astype(np.uint8))


def _followed_speed(model, data) -> float | None:
    """The followed robot's horizontal speed in the body frame, m/s -- the
    number `play` prints on the terminal (`scripts/play.py`'s `_speed_text`,
    which reads `root_link_lin_vel_b`).

    None when the model has no floating base: a speed then belongs to a joint,
    not to the robot, and there is no chip. Reads the window's own `MjData` --
    the followed environment's state as of the frame being drawn -- so it runs
    on the frame thread and shows the same step as the robot beside it.
    """
    import mujoco
    import numpy as np

    if model.nq < 7 or model.nv < 6:
        return None
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(data.qpos[3:7], dtype=np.float64))
    vel = rot.reshape(3, 3).T @ np.asarray(data.qvel[:3], dtype=np.float64)
    return float(np.hypot(vel[0], vel[1]))


def _max_geom() -> int:
    """The geom capacity `launch_passive` gives `user_scn`.

    Once full, `mjv_addGeoms` **silently drops** further geoms (printing one
    warning that scrolls away in the training log), so how many environments to
    draw has to be computed here rather than left to it.

    Read from upstream rather than hardcoded to 100000: that value is
    `mujoco.viewer._Simulate.MAX_GEOM`, and if upstream changes it a hardcoded
    number either draws needlessly few (too small) or goes back to silent
    truncation (too large). The known value is only a fallback.
    """
    try:
        import mujoco.viewer

        return int(mujoco.viewer._Simulate.MAX_GEOM)
    except Exception:  # noqa: BLE001
        return 100_000

#: Model fields that affect CPU-side rendering. When domain randomisation has
#: expanded them, they have to be copied from sim.model back into MjModel every
#: frame, or the window shows nominal geometry rather than what this environment
#: actually uses. The list comes from mjlab's `viewer/model_sync.py` so the two
#: cannot drift apart.
def _viewer_model_fields() -> frozenset[str]:
    from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS

    return VIEWER_MODEL_FIELDS


class _Frame:
    """One copy of the state, taken on the simulation thread as a step ends.

    Row 0 of `qpos` / `qvel` is the followed environment and the rest follow
    `LiveViewer._draw_order`; the other fields are the followed environment's
    alone. `fields` holds the per-environment model fields domain randomisation
    expanded, when there are any.

    Every array here is a host array that nothing else holds a reference to. That
    is the whole point of the class: the frame thread can take as long as it likes
    over one while the simulation writes the next step.
    """

    __slots__ = ("ctrl", "fields", "images", "mocap_pos", "mocap_quat", "qpos", "qvel",
                 "xfrc_applied")

    def __init__(self, qpos, qvel, ctrl, mocap_pos, mocap_quat, xfrc_applied, fields,
                 images=()):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl
        self.mocap_pos = mocap_pos
        self.mocap_quat = mocap_quat
        self.xfrc_applied = xfrc_applied
        self.fields = fields
        #: The sensor previews, taken with the state: see "Sensor insets".
        self.images = images


def no_display_reason() -> str | None:
    """Return a human-readable reason when no display is usable, else None.

    Only **cheap** checks: whether a GL context can really be created is not known
    until `launch_passive`, and the caller's try/except covers that (see
    `_maybe_viewer` in `scripts/train.py`).

    It lives here rather than in the caller because it is the viewer's own
    precondition, and because `scripts/` is not a package and tests cannot import
    from it directly.
    """
    import os
    import sys

    if sys.platform == "darwin":
        # On macOS the passive viewer must create its GLFW context on the main
        # thread, which means mjpython.
        #
        # **Do not look at `sys.executable`**: mjpython is a shell that execs an
        # ordinary interpreter, so `sys.executable` is always `.../bin/python`.
        # Judging by basename means the window can **never** open on macOS -- and
        # the message would say "launch through mjpython", which does not help
        # because that is already what happened.
        #
        # The real signal is `mujoco.viewer._MJPYTHON`: mjpython sets it to the UI
        # thread proxy, and it is None under an ordinary interpreter.
        # `launch_passive` tests exactly this (viewer.py:494, "Initialize GLFW if
        # not using mjpython").
        import mujoco.viewer

        if getattr(mujoco.viewer, "_MJPYTHON", None) is None:
            return "the live viewer needs mjpython on macOS (.venv/bin/mjpython)"
        return None
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return "no DISPLAY / WAYLAND_DISPLAY"
    return None


def _sensor_name(sensor) -> str:
    """What a sensor is called in messages: its config's name, else its class."""
    return getattr(getattr(sensor, "cfg", None), "name", None) or type(sensor).__name__


def _offscreen_unavailable() -> str | None:
    """Why the onboard camera cannot be rendered beside the window, else None.

    See "The onboard camera inset": the context is made on the frame thread, and
    only a GLFW one cannot be.
    """
    from mujoco import gl_context

    context = getattr(gl_context, "GLContext", None)
    if context is None:
        return "MUJOCO_GL turns off-screen rendering off"
    if context.__module__ == "mujoco.glfw":
        return ("its off-screen context would be a GLFW window, which may only be made on "
                "the main thread (MUJOCO_GL=egl or osmesa makes one that is not)")
    return None


class _OnboardInset:
    """The robot's own camera, rendered off-screen for the window's top-left corner.

    **Frame thread only**, every method: a GL context is current on the thread that
    made it, so it is made on the first frame that shows the inset and freed by
    `close()` as the frame thread ends. See "The onboard camera inset".
    """

    def __init__(self, model, camera: int, geom_groups) -> None:
        import mujoco

        from mjrl.sensor.camera import scene_option

        self._model = model
        #: The camera's id in the model the window draws.
        self.camera = camera
        width, height = (int(x) for x in model.cam_resolution[camera])
        #: Width over height, from the camera's own resolution, so the inset has the
        #: camera's field of view whatever shape the window is.
        self.aspect = width / height
        #: The off-screen buffer `MjrContext` allocates; the inset never exceeds it.
        self.buffer = (int(model.vis.global_.offwidth), int(model.vis.global_.offheight))
        #: What the camera would see: the geoms the window shows, and none of the
        #: sites, joints or tendons the window may be drawing on top of them.
        self._opt = scene_option(geom_groups)
        self._catmask = int(mujoco.mjtCatBit.mjCAT_STATIC) | int(mujoco.mjtCatBit.mjCAT_DYNAMIC)
        self._gl = self._con = self._scn = None

    def render(self, data, width: int, height: int):
        """`data` through the camera, as uint8 [height, width, 3] with row 0 at the top."""
        import mujoco
        import numpy as np

        if self._con is None:
            self._open()
        # Every time, as `mujoco.Renderer` does: another context made current on
        # this thread in between would otherwise take the picture.
        self._gl.make_current()
        rect = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self._model, data, self._opt, None, self._cam, self._catmask, self._scn
        )
        mujoco.mjr_render(rect, self._scn, self._con)
        image = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(image, None, rect, self._con)
        return image[::-1]  # OpenGL's row 0 is the bottom one

    def _open(self) -> None:
        import mujoco
        from mujoco import gl_context

        self._gl = gl_context.GLContext(*self.buffer)
        self._gl.make_current()  # `MjrContext` is made in the current context
        self._con = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._con)
        self._scn = mujoco.MjvScene(self._model, maxgeom=10_000)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._cam.fixedcamid = self.camera

    def close(self) -> None:
        """Free the GL resources; the next `render` makes them again. Never raises:
        EGL's teardown can fail at exit with the picture long since drawn (see
        `NativeCameraRenderer.close`)."""
        con, gl = self._con, self._gl
        self._con = self._gl = self._scn = None
        if con is not None:
            with contextlib.suppress(Exception):
                con.free()
        if gl is not None:
            with contextlib.suppress(Exception):
                gl.free()


class LiveViewer:
    """The live training window, shared by the native and warp backends.

    Args:
        sim: an `mjlab.sim.Simulation` or `mjrl.backend.native_sim.NativeSimulation`.
        env_index: which environment the camera follows. **`None` (the default)
            means the one at the centre of the scene** -- environment 0 sits at a
            corner of the grid and following it leaves two sides empty. See
            `_resolve_env_index`.
        fps: how many frames a second to take. `step()` calls between frames
            cost one time comparison; a frame taken costs the simulation a copy
            of the state and nothing else -- the drawing happens on the viewer's
            own thread.
        show_all_envs: draw the other environments too. **True** by default, but
            **how many is capped** (`min(num_envs, 128)`; see
            `_resolve_draw_limit`). Environments are offset on a grid by
            `scene.env_spacing` (2 m by default), so what you see is a field of
            robots rather than one pile.
        max_draw_envs: how many environments to draw at most (including the
            followed one). `None` (the default) means `_MAX_DRAW_ENVS` (128). The
            geom capacity still caps it a second time.
        env_origins: each environment's grid origin, `[N, 3]`. Used to pick the
            ones **nearest the followed environment**. Without it the current base
            positions are used, and failing that the first few by index -- which
            for 4096 environments (a 64x64 grid) would draw only the two outermost
            rows.
    """

    def __init__(
        self,
        sim: Any,
        env_index: int | None = None,
        fps: float = 30.0,
        show_all_envs: bool = True,
        env_origins: Any = None,
        max_draw_envs: int | None = None,
        show_ui: bool = False,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        num_envs = getattr(sim, "num_envs", 0)

        self._sim = sim
        self._show_ui = show_ui
        self._show_all = show_all_envs
        #: Each environment's grid origin [N,3], used to pick the nearest ones.
        #: See _nearest_envs.
        self._env_origins = (
            None if env_origins is None
            else __import__("numpy").asarray(
                env_origins.detach().cpu().numpy()
                if hasattr(env_origins, "detach") else env_origins
            )
        )
        self._env_index = self._resolve_env_index(env_index, num_envs)
        if not 0 <= self._env_index < num_envs:
            raise IndexError(
                f"environment index {self._env_index} out of range (of {num_envs})"
            )
        self._scratch = None  # scratch MjData for the other environments, lazy
        self._min_interval = 1.0 / fps
        self._last_sync = 0.0
        self._handle = None
        #: MuJoCo's own render thread, started by `launch_passive`.
        self._viewer_thread = None
        #: Ours: takes the newest copy of the state and pushes it into the window.
        #: See the module docstring for why it is not the simulation's thread.
        self._frame_thread: threading.Thread | None = None
        self._frame_ready = threading.Event()
        self._frame_stop = threading.Event()
        self._frame_lock = threading.Lock()
        #: The newest copy of the state, waiting to be drawn. A single slot: a
        #: copy the frame thread has not reached yet is replaced, not queued.
        self._latest: _Frame | None = None
        self._orig_step = None
        #: Copies of the state taken on the simulation thread. `maybe_sync` is
        #: called far more often; the difference is what the frame-rate limiter
        #: skipped.
        self.taken = 0
        #: Frames actually pushed to the window, on the frame thread. Fewer than
        #: `taken` when drawing falls behind, which costs the simulation nothing.
        self.synced = 0

        #: native holds real MjData in host memory, so a frame is a host copy;
        #: warp holds it on the device, so a frame is a transfer.
        self._host_state = hasattr(sim, "env_mjdata")

        #: Cap on environments drawn into the window, excluding the followed one.
        #: See `_resolve_draw_limit`.
        self._draw_limit = 0
        self._draw_order: list[int] = []
        #: Device-side index of the followed environment followed by
        #: `_draw_order`, used by warp to take a frame in one transfer; built once
        self._frame_idx = None
        self._geoms_per_env = 0
        #: geom groups to turn off (the collision-only ones); computed in start()
        self._hide_groups: set = set()
        #: How many environments to draw at most, including the followed one.
        #: None means `_MAX_DRAW_ENVS`.
        self._max_draw = int(max_draw_envs) if max_draw_envs else _MAX_DRAW_ENVS
        #: Sensors drawn as corner insets; see "Sensor insets" in the module docstring.
        self._previews = self._preview_sources()
        #: The robot's own camera for the top-left corner, found in `start()` once the
        #: model the window draws is known; see "The onboard camera inset".
        self._onboard: _OnboardInset | None = None
        #: Whether the window holds insets, so they are cleared once, not every frame.
        self._insets_shown = False
        #: The top-centre speed readout's cache, (text, image); re-rendered only
        #: when the number changes. See "The speed readout" in the module docstring.
        self._speed_cache: tuple | None = None
        #: The readout's smoothing state, (monotonic time, value); see
        #: `_speed_readout`. None until the first sample.
        self._speed_smooth: tuple | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @staticmethod
    def _collision_only_groups(model) -> set:
        """Work out which geom groups to turn off to draw appearance only.

        The default `MjvOption.geomgroup` is `[1,1,1,0,0,0]`, and a model carrying
        collision hulls in one visible group and appearance meshes in another draws
        **both**, with the hulls wrapped around the outside of the meshes (measured
        on the hexapod: 31 -> 62 geoms per environment). It is most obvious on
        warp, which does not strip visual meshes, so both sets are present as
        authored.

        The test is applied per group, and a group is turned off only when **both**
        hold:

          1. every geom in it is a collision geom (contype or conaffinity
             non-zero); and
          2. **every body** owning those geoms also has a non-colliding geom -- an
             appearance stand-in.

        The second condition is not optional. **Without it the floor disappears**:
        mjlab's terrain is the `terrain` body (bodyid=1, not worldbody), its plane
        is a collision geom, it has group 0 to itself and it has no appearance
        stand-in -- turn that off and nothing is left, with the robots hanging in
        the void. The same protects a link that has collision geometry but no
        appearance mesh.

        **Groups are only turned off, never on**: the caller clears `geomgroup` for
        this set and leaves the rest as they were. **The model is never modified**
        either -- on warp `mj_model` *is* the physics model and camera sensors read
        it too, so changing `geom_group` there would silently change observations.

        An empty set is returned when the two cannot be separated: if appearance
        and collision share a group (group numbers are the asset author's choice
        and may well be written that way), grouping cannot distinguish them -- and
        such an asset could not draw appearance only in any case.
        """
        colliding = [
            bool(model.geom_contype[i] or model.geom_conaffinity[i])
            for i in range(model.ngeom)
        ]
        # body -> does it have a non-colliding geom (an appearance stand-in)
        has_visual = set()
        for i in range(model.ngeom):
            if not colliding[i]:
                has_visual.add(int(model.geom_bodyid[i]))

        hide = set()
        for g in {int(x) for x in model.geom_group}:
            ids = [i for i in range(model.ngeom) if int(model.geom_group[i]) == g]
            if not all(colliding[i] for i in ids):
                continue  # the group holds appearance geoms; must stay on
            if all(int(model.geom_bodyid[i]) in has_visual for i in ids):
                hide.add(g)
        return hide

    def _pick_render_target(self):
        """Pick the `(model, data)` the window draws, and remember where frames
        are copied from.

        **The window never draws live state**, on either backend: it gets an
        `MjData` of its own, filled from a copy the simulation thread takes as a
        step ends. The frame thread works on that `MjData` while the simulation is
        writing the next step, which it could not do to the live one without
        tearing. native used to hand the window its live `MjData` -- free, when
        drawing ran on the simulation thread, and a torn picture once it does not.

        The model used for drawing need **not** be the physics one either. native
        strips visual meshes to save memory (4096 copies at 41.8 MiB do not fit),
        but what stripping saves is *copying N of them* -- drawing needs only one,
        so the sim keeps a `render_model` with the appearance meshes intact. When
        that is available it is used, so **physics runs on decimated hulls while
        the window draws the real appearance**. qpos and qvel are interchangeable
        between the two, so the same copy fills either.

        warp has no `render_model` attribute and draws the host model `sim.mj_model`.
        This is a separate method so the selection logic can be tested **without
        opening a window**.
        """
        import mujoco

        #: native: the followed environment's **live** MjData, which frames are
        #: copied from on the simulation thread. None on warp.
        self._live_main = None
        #: Whether the window draws native's `render_model` rather than the model
        #: physics runs on.
        self._render_proxy = False
        if not self._host_state:
            model = self._sim.mj_model
            return model, mujoco.MjData(model)
        model, self._live_main = self._sim.env_mjdata(self._env_index)
        render_m = getattr(self._sim, "render_model", None)
        if render_m is not None and render_m is not model:
            self._render_proxy, model = True, render_m
        return model, mujoco.MjData(model)

    def start(self) -> "LiveViewer":
        """Open the window, attach the hook that takes frames and start the
        thread that draws them. Idempotent."""
        if self._handle is not None:
            return self

        import mujoco
        import mujoco.viewer

        model, data = self._pick_render_target()

        # **Assign before taking a frame**: `_take_frame` and `_apply` use
        # `self._model`. Written the other way round, the first frame raises
        # AttributeError the moment the window opens -- and when the warp path
        # alone took one, nothing caught it until a window was really opened on a
        # GPU machine.
        self._model, self._data = model, data
        # Turn off the geom groups that hold collision geometry only. **Both
        # backends take this path**: warp's model carries appearance and collision
        # as authored, and so does native's render_model when there is one. The one
        # case where nothing may be turned off is the degraded one -- visual meshes
        # stripped and no render_model -- where collision geometry is all there is
        # to draw.
        stripped_no_render = (
            getattr(self._sim, "visuals_stripped", False) and not self._render_proxy
        )
        self._hide_groups = (
            set() if stripped_no_render else self._collision_only_groups(model)
        )
        if self._show_all:
            self._scratch = mujoco.MjData(model)
            self._vopt = mujoco.MjvOption()
            self._pert = mujoco.MjvPerturb()
            self._catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
            if stripped_no_render:
                # After stripping visual meshes the remaining collision geometry
                # sits in **geom group 4**, and the default `MjvOption.geomgroup`
                # is [1,1,1,0,0,0] -- group 4 is off. Without turning it on, all
                # that gets drawn is the six foot `*_meshcol` meshes: **nothing but
                # toe tips and finger tips**, not one link capsule.
                for g in range(len(self._vopt.geomgroup)):
                    self._vopt.geomgroup[g] = 1
                print(
                    "[mjrl] live viewer: visual meshes were stripped, showing "
                    "collision geometry instead (capsules / spheres / boxes plus "
                    "the foot meshes)"
                )
            else:
                for g in self._hide_groups:
                    self._vopt.geomgroup[g] = 0
            self._resolve_draw_limit(model)
            self._widen_shadows(model)
        self._onboard = self._find_onboard(model, stripped_no_render)
        # Fill one frame first, so the window does not open on the model's default
        # pose. Here, on the simulation thread, before anything else exists to
        # draw it.
        self._apply(self._take_frame())
        # `launch_passive` puts the render loop in a **daemon thread** and does
        # not hand it back. At exit, atexit runs LIFO: mujoco's `exit_simulate`
        # first (which only signals, it does not wait), then glfw's registered
        # `terminate()` -- so GLFW is torn down while that thread is still inside
        # `_launch_internal`, and the process **segfaults**.
        #
        # Since the thread has no recognisable name, the only way to get hold of it
        # is to diff `threading.enumerate()` around the call.
        before = set(threading.enumerate())
        # `key_callback` is wired unconditionally to the registry in
        # `mjrl.viewer.keys`, which is empty unless something registered a
        # handler, and which also sees each key come up -- MuJoCo reports the
        # press alone (`keys.from_viewer`). That keeps this file free of any knowledge of what keys mean:
        # the viewer is built in `scripts/_cli.py`, which has no task context, so
        # a constructor argument would have to be plumbed through code that should
        # not have to care. See the module docstring in `keys.py`.
        # MuJoCo's own panels, off by default. They are the `simulate` UI --
        # rendering flags, visualisation options, the model tree -- and they are
        # genuinely useful for looking at a policy, which is why `show_ui` exists.
        #
        # Off by default because this viewer is on during **training** too, where
        # the panels cost frame time and screen for controls nobody is going to
        # touch across a two-hour run.
        #
        # One asymmetry to know about when they are on. The panel edits
        # `handle.opt`, which governs the environment the camera follows; the
        # other environments are appended to the scene under `self._vopt`, which
        # the panel cannot reach. So toggling a geom group in the UI changes the
        # followed robot and leaves the rest of the field as it was. That is a
        # consequence of drawing many environments into one scene, not something
        # this can fix from here.
        self._handle = mujoco.viewer.launch_passive(
            model, data,
            show_left_ui=self._show_ui, show_right_ui=self._show_ui,
            key_callback=keys.from_viewer,
        )
        new_threads = set(threading.enumerate()) - before
        self._viewer_thread = next(iter(new_threads), None)
        # Backstop for a caller that forgets stop(): the thread still has to be
        # joined before exit. This atexit is registered **after** glfw's (glfw
        # registers inside launch_passive), so under LIFO it runs **first**, just
        # ahead of glfw.terminate().
        # `self._vopt` governs only the environments **appended** to the scene;
        # the followed one is rendered by the window itself through `handle.opt`.
        # Both have to be set, or the robot the camera follows keeps its hulls.
        for g in self._hide_groups:
            self._handle.opt.geomgroup[g] = 0
        if self._hide_groups:
            what = "full appearance meshes" if self._render_proxy else "appearance meshes"
            print(
                f"[mjrl] live viewer: drawing {what} only; turned off the "
                f"collision-only geom groups {sorted(self._hide_groups)}"
            )
        atexit.register(self.stop)
        self._aim_camera(model)
        # Started **after** the `threading.enumerate()` diff above, or it would be
        # taken for MuJoCo's render thread.
        self._start_frame_thread()
        self._hook()
        self._last_sync = time.perf_counter()
        return self

    def _start_frame_thread(self) -> None:
        """Start the thread that draws frames. A method of its own so the handoff
        can be tested without opening a window."""
        self._frame_stop.clear()
        self._frame_thread = threading.Thread(
            target=self._frame_loop, name="mjrl-live-viewer-frames", daemon=True
        )
        self._frame_thread.start()

    def stop(self) -> None:
        """Close the window, detach the hook and restore the wrapped `step`.
        Training continues unaffected.

        **Our frame thread is stopped first**, before the window closes: it may be
        inside `handle.sync()`, and closing the window under it is the same race
        as the one below, one thread earlier.

        **MuJoCo's render thread has to be waited for**, not just `close()`d: that
        only signals, and the thread still has to join its own side_thread and call
        `destroy()`. Without waiting it collides with glfw's `atexit terminate()`
        and the process exits with a segfault. See `start()`.
        """
        self._unhook()
        self._frame_stop.set()
        self._frame_ready.set()  # wake it if it is waiting for a frame
        f = self._frame_thread
        self._frame_thread = None
        if f is not None and f is not threading.current_thread():
            f.join(timeout=5.0)
            if f.is_alive():
                print("[mjrl] the live viewer's frame thread did not finish within "
                      "5 s; closing the window anyway")
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        t = self._viewer_thread
        if t is not None:
            self._viewer_thread = None
            try:
                t.join(timeout=5.0)
            except (AssertionError, RuntimeError):
                # Under mjpython (macOS) it cannot be joined: there
                # `launch_passive` keeps the UI on the **main thread**, the diff
                # catches a dummy thread created on the C side, and `Thread.join`
                # goes straight to `assert False, "cannot join a dummy thread"`.
                #
                # Not waiting is fine there: the precondition for the segfault does
                # not hold on that path -- GLFW is managed by mjpython itself, and
                # `viewer.py:494` only calls `glfw.init()` +
                # `atexit.register(glfw.terminate)` when **not** running under
                # mjpython, so there is no "glfw torn down while the render thread
                # runs" to begin with.
                #
                # Swallowing it matters: otherwise every training run on macOS ends
                # in an AssertionError.
                pass
            else:
                if t.is_alive():
                    # Say so if it does not finish: if a segfault follows, at
                    # least the cause is known
                    print("[mjrl] the live viewer's render thread did not finish "
                          "within 5 s; exiting anyway")
        with contextlib.suppress(Exception):
            atexit.unregister(self.stop)

    def __enter__(self) -> "LiveViewer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ── Hooking in ────────────────────────────────────────────────────────

    def _hook(self) -> None:
        if self._host_state:
            self._sim.attach_viewer(self)
            return
        # warp: Simulation is vendored, so wrap the instance, never the class
        orig = self._sim.step
        self._orig_step = orig

        def step_then_sync(*args: Any, **kwargs: Any) -> Any:
            out = orig(*args, **kwargs)
            self.maybe_sync()
            return out

        self._sim.step = step_then_sync

    def _unhook(self) -> None:
        if self._host_state:
            self._sim.attach_viewer(None)
        elif self._orig_step is not None:
            self._sim.step = self._orig_step
            self._orig_step = None

    # ── Taking a frame: the simulation thread ─────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._handle is not None and self._handle.is_running()

    def maybe_sync(self) -> None:
        """Called after every physics step, **on the simulation thread**: takes a
        frame, subject to the fps cap, and hands it to the frame thread.

        This is the only place live state is read, and the copy is all it costs
        the simulation; the drawing is `_frame_loop`'s. The handoff never waits --
        a frame the frame thread has not reached yet is replaced by this one.

        Detaches itself once the user closes the window -- training should neither
        keep paying for frames nobody is watching, nor crash because the window
        went away.
        """
        if self._handle is None:
            return
        if not self._handle.is_running():
            self.stop()
            return

        now = time.perf_counter()
        if now - self._last_sync < self._min_interval:
            return
        self._last_sync = now
        if self._frame_thread is None or not self._frame_thread.is_alive():
            return  # nothing is drawing any more; `_frame_loop` printed why
        frame = self._take_frame()
        with self._frame_lock:
            self._latest = frame
        self.taken += 1
        self._frame_ready.set()

    def _take_frame(self) -> _Frame:
        """Copy what drawing needs out of live state. **Simulation thread only.**"""
        frame = self._copy_host_state() if self._host_state else self._pull_state()
        frame.images = self._take_previews()
        return frame

    def _pull_state(self) -> _Frame:
        """Take a frame from the GPU: the followed environment and the ones drawn
        beside it, in **one** transfer.

        The field selection matches mjlab's
        `NativeViewer._sync_env_state_to_mjdata`: rendering needs qpos/qvel/ctrl/
        mocap, and the remaining derived quantities are recomputed by
        `mj_forward`.

        **One transfer, not one per field and environment**, because every
        `.cpu()` is a synchronisation as well as a copy, and they are paid here on
        the simulation thread. 128 environments at two transfers each was 254
        small transfers dominated by launch overhead; batching them per field took
        a frame from 7.70 to 5.49 ms (-29%) with bit-identical geom positions, and
        one `torch.cat` takes the rest of the fields along in the same transfer.
        """
        import numpy as np
        import torch

        sim, k, m = self._sim, self._env_index, self._model
        d = sim.data
        if self._frame_idx is None:
            self._frame_idx = torch.as_tensor(
                [k, *self._draw_order], device=d.qpos.device
            )
        idx = self._frame_idx
        n = len(self._draw_order) + 1
        parts = [d.qpos[idx].reshape(-1), d.qvel[idx].reshape(-1)]
        sizes = [n * m.nq, n * m.nv]
        if m.nu > 0:
            parts.append(d.ctrl[k].reshape(-1))
            sizes.append(m.nu)
        if m.nmocap > 0:
            parts += [d.mocap_pos[k].reshape(-1), d.mocap_quat[k].reshape(-1)]
            sizes += [3 * m.nmocap, 4 * m.nmocap]
        parts.append(d.xfrc_applied[k].reshape(-1))
        sizes.append(6 * m.nbody)
        flat = torch.cat([p.to(parts[0].dtype) for p in parts]).cpu().numpy()
        chunks = iter(np.split(flat.astype(np.float64), np.cumsum(sizes)[:-1]))

        qpos = next(chunks).reshape(n, m.nq)
        qvel = next(chunks).reshape(n, m.nv)
        ctrl = next(chunks) if m.nu > 0 else None
        mocap_pos = next(chunks) if m.nmocap > 0 else None
        mocap_quat = next(chunks) if m.nmocap > 0 else None
        xfrc = next(chunks)

        # When domain randomisation has expanded visual fields, the model has to
        # follow this environment too. Rare, and each is a transfer of its own:
        # they are integer and float arrays of different shapes.
        fields = getattr(sim, "expanded_fields", frozenset()) & _viewer_model_fields()
        values = {f: getattr(sim.model, f)[k].cpu().numpy() for f in fields}
        return _Frame(qpos, qvel, ctrl, mocap_pos, mocap_quat, xfrc, values)

    def _copy_host_state(self) -> _Frame:
        """Take a frame from native's live `MjData`s: host copies, microseconds.

        Only qpos and qvel for the environments beside the followed one, as when
        they were drawn straight from live state; the followed one carries
        everything `_pull_state` fetches on warp.
        """
        import numpy as np

        live = self._live_main
        n = len(self._draw_order) + 1
        qpos = np.empty((n, live.qpos.size))
        qvel = np.empty((n, live.qvel.size))
        qpos[0], qvel[0] = live.qpos, live.qvel
        for row, i in enumerate(self._draw_order, start=1):
            _, d = self._sim.env_mjdata(i)
            qpos[row], qvel[row] = d.qpos, d.qvel
        return _Frame(
            qpos, qvel, live.ctrl.copy(), live.mocap_pos.copy(),
            live.mocap_quat.copy(), live.xfrc_applied.copy(), {},
        )

    # ── Drawing a frame: the frame thread ─────────────────────────────────

    def _frame_loop(self) -> None:
        """Draw the newest frame whenever there is one. The frame thread's body.

        Everything here is off the simulation thread: filling the window's own
        `MjData`, `mj_forward`, the other environments and `handle.sync()`, which
        copies the whole model into the viewer and is the expensive part. All of
        them release the GIL.

        A frame that fails to draw stops the drawing and says why. It must never
        reach the simulation, which does not wait for this thread -- and an
        exception raised here would end the thread without anyone hearing of it.

        However the loop ends, the onboard camera's GL context is freed here: it is
        current on this thread and on no other.
        """
        try:
            while not self._frame_stop.is_set():
                if not self._frame_ready.wait(timeout=0.1):
                    continue
                self._frame_ready.clear()
                with self._frame_lock:
                    frame, self._latest = self._latest, None
                handle = self._handle
                if frame is None or handle is None:
                    continue
                if not handle.is_running():
                    return  # the window was closed; `maybe_sync` notices and stops
                try:
                    self._apply(frame)
                    if self._show_all:
                        self._draw_other_envs(frame)
                    self._show_insets(handle, frame.images)
                    handle.sync()
                except Exception as e:  # noqa: BLE001
                    print(f"[mjrl] live viewer: drawing a frame failed, the window stops "
                          f"updating: {type(e).__name__}: {e}")
                    return
                self.synced += 1
        finally:
            if self._onboard is not None:
                self._onboard.close()

    def _apply(self, frame: _Frame) -> None:
        """Fill the window's own `MjData` from a frame and bring its derived
        quantities up to date.

        On the frame thread, except for the very first frame, which `start()`
        fills before the frame thread exists.
        """
        import mujoco

        m, data = self._model, self._data
        data.qpos[:] = frame.qpos[0]
        data.qvel[:] = frame.qvel[0]
        if m.nu > 0:
            data.ctrl[:] = frame.ctrl
        if m.nmocap > 0:
            data.mocap_pos[:] = frame.mocap_pos.reshape(data.mocap_pos.shape)
            data.mocap_quat[:] = frame.mocap_quat.reshape(data.mocap_quat.shape)
        data.xfrc_applied[:] = frame.xfrc_applied.reshape(data.xfrc_applied.shape)
        for name, value in frame.fields.items():
            dst = getattr(m, name)
            dst[:] = value.reshape(dst.shape)
        mujoco.mj_forward(m, data)

    # ── Sensor insets ─────────────────────────────────────────────────────

    def _preview_sources(self) -> list:
        """The sensors that can show themselves, from the simulation's sensor context.

        Both backends keep the context the environment registered with
        `set_sensor_context` as `_sensor_context`. No context, no insets.
        """
        ctx = getattr(self._sim, "_sensor_context", None)
        sensors = [*getattr(ctx, "raycast_sensors", ()), *getattr(ctx, "camera_sensors", ())]
        found = [s for s in sensors if callable(getattr(s, "preview", None))]
        if found:
            names = ", ".join(_sensor_name(s) for s in found)
            print(f"[mjrl] live viewer: showing {names} in the top-right corner")
        return found

    def _take_previews(self) -> tuple:
        """Every source's picture of the followed environment. **Simulation thread.**"""
        import numpy as np

        images = []
        for source in list(self._previews):
            try:
                image = source.preview(self._env_index)
                if image is None:
                    continue
                image = np.asarray(image)
                if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                    raise ValueError(
                        f"expected a uint8 [H, W, 3] image, got {image.dtype} {image.shape}"
                    )
            except Exception as e:  # noqa: BLE001
                self._previews.remove(source)
                print(f"[mjrl] live viewer: {_sensor_name(source)}'s preview failed and "
                      f"is no longer shown: {type(e).__name__}: {e}")
                continue
            images.append(np.array(image))  # a copy: the frame thread outlives the step
        return tuple(images)

    def _find_onboard(self, model, stripped: bool) -> _OnboardInset | None:
        """The robot's own camera in the model the window draws, or None.

        `stripped`: visual meshes were stripped and there is no render model, so the
        collision geometry in every group is all there is to see -- as for the
        environments drawn beside the followed one.

        Called before the window opens, because that is the last moment libglvnd's
        entry-point patching can still be turned off -- see "GLX and EGL in one
        process" in the module docstring.
        """
        import os

        import mujoco

        cameras = [
            i for i in range(model.ncam)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or "")
            .rsplit("/", 1)[-1] == _ONBOARD_CAMERA
        ]
        if not cameras:
            return None
        reason = _offscreen_unavailable()
        if reason:
            print(f"[mjrl] live viewer: not showing the onboard camera: {reason}")
            return None
        default = mujoco.MjvOption().geomgroup
        groups = (
            range(len(default)) if stripped
            else [g for g, on in enumerate(default) if on and g not in self._hide_groups]
        )
        # Read by libglvnd when a context is first made current, so it has to be in
        # the environment before the window's is; `setdefault`, so an explicit 0 wins.
        os.environ.setdefault("__GLVND_DISALLOW_PATCHING", "1")
        print("[mjrl] live viewer: showing the onboard camera in the top-left corner "
              "while the window looks through another")
        return _OnboardInset(model, cameras[0], groups)

    def _show_insets(self, handle, images) -> None:
        """Place the previews in the top-right corner of the window, and the onboard
        camera in the top-left. **Frame thread.**

        Read against the viewport every frame, so resizing the window or opening the
        UI panels moves them along. One that no longer fits is left out rather than
        drawn over the edge, and when none is left the window's are cleared.
        """
        import mujoco
        import numpy as np

        placed = []
        chip = self._speed_readout()
        vp = handle.viewport if images or self._onboard is not None or chip is not None else None
        if vp is not None and vp.width > 0 and vp.height > 0:
            right = vp.left + vp.width - _INSET_MARGIN
            top = vp.bottom + vp.height - _INSET_MARGIN
            for image in images:
                h, w = image.shape[:2]
                scale = max(1, min(_INSET_MAX_SCALE, round(vp.width * _INSET_FRACTION / w)))
                left, bottom = right - w * scale, top - h * scale
                if left < vp.left or bottom < vp.bottom:
                    break
                block = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
                placed.append((mujoco.MjrRect(left, bottom, w * scale, h * scale), block))
                top = bottom - _INSET_MARGIN
            rightmost = min((r.left for r, _ in placed), default=vp.left + vp.width)
            onboard = self._place_onboard(handle, vp, placed)
            if onboard is not None:
                placed.append(onboard)
            if chip is not None:
                # Centred in the space the corner insets leave, horizontally
                # and vertically on their column; left out when there is no
                # room rather than drawn over one of them.
                ch, cw = chip.shape[:2]
                lo = vp.left + _INSET_MARGIN
                if onboard is not None:
                    rect, _ = onboard
                    lo = rect.left + rect.width + _INSET_MARGIN
                hi = rightmost - _INSET_MARGIN
                left = lo + (hi - lo - cw) // 2
                top_edge = vp.bottom + vp.height - _INSET_MARGIN
                if placed:
                    # Vertically centred on the insets' column, from its top
                    # edge to the lowest inset's bottom.
                    span = min(r.bottom for r, _ in placed)
                    bottom = span + max(0, (top_edge - span - ch) // 2)
                else:
                    bottom = top_edge - ch
                if cw <= hi - lo and bottom >= vp.bottom:
                    placed.append((mujoco.MjrRect(left, bottom, cw, ch), chip))
        if placed:
            handle.set_images(placed)
            self._insets_shown = True
        elif self._insets_shown:
            handle.clear_images()
            self._insets_shown = False

    def _speed_readout(self):
        """The followed robot's speed as a chip image for the window's top
        centre, or None when the model has no floating base. **Frame thread.**

        The displayed value is an exponential moving average over
        `_SPEED_SMOOTH_S`: the raw speed bobs with every stride, which flickers
        the second decimal every frame, and a number that cannot be read says
        less than one a quarter-second behind. The terminal's number is not
        smoothed.

        Cached by text: the number changes most frames while driving, the
        pixels only when it does.
        """
        import math
        import time

        raw = _followed_speed(self._model, self._data)
        if raw is None:
            return None
        now = time.monotonic()
        if self._speed_smooth is None:
            self._speed_smooth = (now, raw)
        else:
            t, value = self._speed_smooth
            value += (raw - value) * (1.0 - math.exp(-max(0.0, now - t) / _SPEED_SMOOTH_S))
            self._speed_smooth = (now, value)
        text = f"{self._speed_smooth[1]:.2f} m/s"
        if self._speed_cache is None or self._speed_cache[0] != text:
            self._speed_cache = (text, _speed_chip(text))
        return self._speed_cache[1]

    def _place_onboard(self, handle, vp, right_insets):
        """The onboard camera's `(rect, image)` for the top-left corner, or None.
        **Frame thread.**

        None while the window itself looks through that camera, and when the inset
        would run into `right_insets` or off the bottom. Sized by "The onboard camera
        inset": the height of the top inset on the right, or a fifth of the window's
        width when there is none, and never larger than the off-screen buffer.

        A render that fails turns the inset off with one message. Letting it raise
        would stop the whole window updating, over a corner of it.
        """
        import mujoco

        inset = self._onboard
        if inset is None:
            return None
        cam = handle.cam
        if cam.type == mujoco.mjtCamera.mjCAMERA_FIXED and cam.fixedcamid == inset.camera:
            return None
        if right_insets:
            height = right_insets[0][0].height
            width = round(height * inset.aspect)
        else:
            width = round(vp.width * _INSET_FRACTION)
            height = round(width / inset.aspect)
        buffer_w, buffer_h = inset.buffer
        if width > buffer_w or height > buffer_h:
            height = min(buffer_h, int(buffer_w / inset.aspect))
            width = min(buffer_w, round(height * inset.aspect))
        left = vp.left + _INSET_MARGIN
        bottom = vp.bottom + vp.height - _INSET_MARGIN - height
        right = min((r.left for r, _ in right_insets), default=vp.left + vp.width)
        if width < 1 or height < 1 or left + width > right - _INSET_MARGIN or bottom < vp.bottom:
            return None
        try:
            image = inset.render(self._data, width, height)
        except Exception as e:  # noqa: BLE001
            self._onboard = None
            inset.close()
            print(f"[mjrl] live viewer: rendering the onboard camera failed and it is no "
                  f"longer shown: {type(e).__name__}: {e}")
            return None
        return mujoco.MjrRect(left, bottom, width, height), image

    def _resolve_env_index(self, given: "int | None", num_envs: int) -> int:
        """Follow the environment at the **centre of the scene** when none is given.

        Environments are laid out on a grid by `env_spacing` and environment 0 sits
        at a corner: 4096 environments span 126x126 m with 0 at (63, -63).
        Following it, the neighbours that get drawn are only on two sides and the
        other two are empty. Following the central one puts robots on all four.

        The one nearest the centroid of all origins is chosen rather than computing
        grid row and column -- generated terrain need not produce a regular grid,
        and the centroid criterion holds for any layout.
        """
        if given is not None:
            return int(given)
        pos = self._env_origins
        if pos is None or len(pos) != num_envs or num_envs == 0:
            return 0
        import numpy as np

        p = np.asarray(pos, dtype=float)
        return int(((p - p.mean(axis=0)) ** 2).sum(axis=1).argmin())

    def _aim_camera(self, model) -> None:
        """Make the camera actually follow environment `env_index`.

        `launch_passive` gives a default free camera that looks at the model's
        `stat.center` (near the origin), while environments are spread out on a
        grid by `env_spacing`: 4096 environments at 2 m spacing form a 64x64 grid
        spanning 128 m. So "follow environment 0" would in practice mean "look at
        the origin, with the robots a hundred metres away".

        MuJoCo's own `mjCAMERA_TRACKING` is used instead: the camera centre is
        locked to a body while the user can still rotate and zoom freely. Better
        than rewriting `lookat` every frame, which fights the user's mouse.

        With no body to follow it falls back to the free camera with `lookat` set
        to that environment's origin.
        """
        import mujoco

        cam = self._handle.cam
        body = self._followed_body(model)
        if body is not None:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = body
        elif self._env_origins is not None:
            cam.lookat[:] = self._env_origins[self._env_index]
        else:
            return
        # Stand back a little and look down: one environment's robot is about a
        # metre across and a field of neighbours is drawn around it, so a close
        # camera would show only the middle one.
        cam.distance = max(float(model.stat.extent) * 3.0, 6.0)
        cam.elevation = -25.0

    def _followed_body(self, model) -> "int | None":
        """Which body the camera follows: the followed environment's **base**, the
        one with the free joint.

        Found by looking for the free joint rather than by a hardcoded index: body
        0 is the world, body 1 is usually the terrain, and where the robot base
        lands depends on how the scene was assembled.
        """
        import mujoco

        free = int(mujoco.mjtJoint.mjJNT_FREE)
        for i in range(model.nbody):
            if model.body_jntnum[i] > 0 and model.jnt_type[model.body_jntadr[i]] == free:
                return i
        return None

    def _nearest_envs(self) -> list[int]:
        """Return the other environments' indices, sorted by **distance from the
        followed one**.

        Taking the first `limit` indices instead would be wrong: environments form
        a grid by `env_spacing`, 4096 of them a 64x64 one, so the first 128 are
        exactly the **two outermost rows** while the camera follows environment 0
        at a corner. What that looks like in practice is a field of robots stuck to
        the edge of the view.

        Positions come from the scene's `env_origins` (the fixed grid origins) when
        available, and fall back to current base positions. The latter move as the
        robots walk, but the choice is made once when the window opens, which is
        enough.
        """
        n, k = self._sim.num_envs, self._env_index
        pos = self._env_origins
        if pos is None:
            try:
                pos = self._sim.data.qpos[:, :3].detach().cpu().numpy()
            except Exception:  # noqa: BLE001
                pos = None
        others = [i for i in range(n) if i != k]
        if pos is None or len(pos) != n:
            return others
        import numpy as np

        p = np.asarray(pos, dtype=float)
        d2 = ((p - p[k]) ** 2).sum(axis=1)
        return sorted(others, key=lambda i: d2[i])

    def _widen_shadows(self, model) -> None:
        """Make the shadow map cover the environments actually being drawn.

        MuJoCo fits a directional light's shadow map to the model, scaled by
        `vis.map.shadowclip`. The model here is **one environment** -- extent
        around a metre for a hexapod -- while the environments being drawn are
        spread over the whole grid: 128 of them at 2 m spacing is more than 20 m
        across, and 4096 spans 126 m. Everything outside that little box renders
        with no shadow at all, which is why most of the robots look pasted onto
        the ground while the followed one does not.

        Nothing reports it. The picture is complete, correctly lit and correctly
        shaded; the shadows are simply missing from all but a handful, and the eye
        reads that as the robots floating rather than as a rendering setting.

        Widening costs resolution, and there is a lot to spend: mjlab's template
        asks for an 8192 shadow map, so even a 126 m box is 15 mm per texel.

        Only `vis`, and only on the model the window draws. Physics does not read
        it, and neither do the camera sensors -- they use `vis.map.znear/zfar` and
        `stat.extent`, which are left alone.
        """
        origins = self._env_origins
        if origins is None or not self._draw_order:
            return
        import numpy as np

        pts = np.asarray(origins, dtype=float)[
            list(self._draw_order) + [self._env_index]
        ][:, :2]
        radius = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
        extent = float(model.stat.extent) or 1.0
        # Two of the radius, plus the model itself: the box is centred on the
        # model rather than on what is being drawn, and a shadow is cast some way
        # from the thing casting it when the sun is this low.
        needed = 2.0 * (radius + extent) / extent
        if needed > float(model.vis.map.shadowclip):
            model.vis.map.shadowclip = needed
            print(f"[mjrl] live viewer: widened the shadow map to cover "
                  f"{radius:.0f} m of environments (shadowclip {needed:.0f})")

    def _resolve_draw_limit(self, model) -> None:
        """Decide how many environments to draw -- `min(num_envs, 128)` -- and
        print the actual number.

        ## Why there has to be a cap

        Without one there are two traps, **neither of which reports an error**:

        1. **Silent geom truncation.** `launch_passive` gives `user_scn` a capacity
           of `MAX_GEOM = 100000`. Once full, `mjv_addGeoms` prints **one** warning
           (MuJoCo prints each message once, and it scrolls away in the training
           log) and then **silently drops** the rest. Measured with warp at 4096
           environments: 253890 geoms expected, 100000 arrived -- 1613 environments
           actually drawn (39%), while the picture still shows a field of robots
           and nothing looks missing.
        2. **Time budget.** Measured on native with 4096 environments, drawing all
           of them costs **405 ms per frame**. That was when drawing hung off
           `sim.step()` and came straight out of training -- 0.4 s of a 1.05 s
           control step. It is on the frame thread now, but the copy each frame
           starts from is still taken on the simulation thread and grows with the
           number drawn, and a frame thread that needs 0.4 s a frame shows a
           slideshow.

        `_MAX_DRAW_ENVS` covers both: 128 environments are far below the geom
        capacity and the frame budget on either backend. `geom_cap` is still
        computed as a backstop -- geoms per environment is a property of the model,
        and a sufficiently complex robot could fill 100000 with 128 environments.
        Hitting that has to be said out loud rather than going back to silent
        truncation.
        """
        import mujoco

        others = self._nearest_envs()
        cap = _max_geom()
        # How many geoms one environment contributes: measure once with a scratch
        # scene rather than guessing
        probe = mujoco.MjvScene(model=model, maxgeom=cap)
        mujoco.mj_forward(model, self._scratch)
        mujoco.mjv_addGeoms(
            model, self._scratch, self._vopt, self._pert, self._catmask, probe
        )
        self._geoms_per_env = max(1, probe.ngeom)
        geom_cap = cap // self._geoms_per_env

        self._draw_limit = min(len(others), self._max_draw - 1, geom_cap)
        self._draw_order = others[: self._draw_limit]
        self._frame_idx = None
        shown = self._draw_limit + 1  # plus the followed one
        if shown < self._sim.num_envs:
            why = (
                f"cap {self._max_draw}"
                if self._draw_limit == self._max_draw - 1
                else f"geom capacity {cap} / {self._geoms_per_env} per env = {geom_cap}"
            )
            print(
                f"[mjrl] live viewer: {self._sim.num_envs} environments, "
                f"drawing {shown} of them ({why})"
            )
        else:
            print(f"[mjrl] live viewer: drawing all {self._sim.num_envs} environments")

    def _draw_other_envs(self, frame: _Frame) -> None:
        """Append the environments other than the followed one as extra geoms.

        The approach matches mjlab's own `NativeViewer._render_other_env_geoms`:
        push each environment's state into a scratch `MjData`, compute the geometry
        with `mj_forward`, and append with `mjv_addGeoms`. **`user_scn.ngeom` must
        be zeroed every frame**, or geoms accumulate frame by frame until the scene
        capacity is exceeded, getting slower all the way.

        On the frame thread, from `frame`'s rows 1.. -- never from live state. This
        loop is most of a frame's cost when training (127 forwards), and it used to
        run on the simulation thread.
        """
        import mujoco

        assert self._handle is not None and self._scratch is not None
        scn = self._handle.user_scn
        scn.ngeom = 0
        for row in range(1, len(frame.qpos)):
            self._scratch.qpos[:] = frame.qpos[row]
            self._scratch.qvel[:] = frame.qvel[row]
            mujoco.mj_forward(self._model, self._scratch)
            mujoco.mjv_addGeoms(
                self._model, self._scratch, self._vopt, self._pert, self._catmask, scn
            )
