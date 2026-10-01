# JumperEMW

**English** | [简体中文](README.zh.md)

<img src="docs/media/run-top-speed.gif" align="right" width="300" alt="Jumper sprinting at 2 m/s with a live speed readout">

Personal experiments on **Jumper**, a 22-DoF 3D-printed crab robot:
reinforcement-learning motion training in MuJoCo, with each experiment
documented, reproducible, and open.

This is an independent playground by [EMW Studio](https://github.com/emwstudio),
not an official KingKong Robotics release. It builds on the open-source
[Jumper training repository](https://github.com/KingKongRobotics/jumper);
upstream history and attribution are preserved in this repository's git
history.

<br clear="right">

## Experiments

Animated previews play directly in the table. Click one to open the complete
video. Results are simulation rollouts; hardware validation is still ahead.

<table>
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Preview</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Top-speed running</strong></td>
      <td>
        <a href="docs/media/run-top-speed.mp4">
          <img src="docs/media/run-top-speed.gif" width="280" alt="Jumper sprinting at a pinned 2.0 m/s command with a live speed readout on top">
        </a>
      </td>
      <td>
        Velocity policy at a pinned 2.0 m/s command: <strong>2.15 m/s</strong>
        peak, 2.03 m/s on the smoothed readout, and no fall in any of five
        10-second runs. The overlay is the same live speed readout the
        <code>play</code> window draws.<br>
        <a href="docs/media/run-top-speed.mp4">Full video (4K)</a> ·
        <a href="docs/TUTORIAL.md">Training tutorial</a>
      </td>
    </tr>
  </tbody>
</table>

## Quick start

Python 3.10–3.13. On an NVIDIA machine install the GPU build of PyTorch;
anywhere else, install the CPU build and select the native backend.

```bash
git clone https://github.com/emwstudio/JumperEMW.git
cd JumperEMW
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .

python scripts/train.py --list
python scripts/train.py --task jumper.run
python scripts/play.py --task jumper.run --checkpoint <path>/model_9999.pt
```

The play window is drivable: **W** runs, **Backspace** resets the episode,
and the live speed readout sits on top. See the
[training tutorial](docs/TUTORIAL.md) for the full path from training to a
deployable bundle, and [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) for
setup details on Linux, macOS and Windows.

## Repository layout

```text
tasks/jumper/run/      the sprint task: env config, rewards, controls
tasks/jumper/          the upstream gaits, dances, gestures and more
rl/mjrl/               training glue and the live play viewer
scripts/               train, play, export, deploy
deploy/                from a checkpoint to the robot: bundle, RKNN, hosts
docs/                  manuals and guides, bilingual
tests/                 the suite that guards all of the above
```

## Scope

These are simulation experiments, not hardware certifications. The 2.15 m/s
figure is a simulation result on flat ground; the real robot has not run it
yet. Hardware results will be reported here when they exist.

## Upstream and license

The Jumper product lives at
[KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper);
appearance and scene creation use the separate
[jumper-design](https://github.com/KingKongRobotics/jumper-design) repository.

Copyright 2026 KingKong Robotics. Maintainer-owned project materials are
licensed under Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Third-party materials remain under their respective terms. Experiments and
documentation added in this repository carry the same license.
