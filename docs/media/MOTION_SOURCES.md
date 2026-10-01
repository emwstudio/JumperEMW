# Motion showcase sources

The maintainer requested these three clips from the official [Jumper page](https://kingkong.tech/jumper)
for the README on 2026-09-30. The category mapping follows the maintainer's selection.

| README asset | Source |
|---|---|
| `official-dance.gif` | https://kingkong.tech/assets/products/robot/0929/action-move.mp4 |
| `official-jump.gif` | https://kingkong.tech/assets/products/robot/0929/action-jump.mp4 |
| `official-grasp.gif` | https://kingkong.tech/assets/products/robot/0929/action-operate.mp4 |

Converted with FFmpeg to looping GIFs at 360 pixels wide, 12.5 fps, with a 128-color
palette and Bayer dithering. Full duration and aspect ratio are preserved, without audio.
The grasp website clip is horizontally mirrored at the maintainer's request.
These website showcases are not evidence that the paired simulation policies drive the
depicted robot, nor a hardware validation result. Original simulation GIFs remain unchanged.

## Top-speed running media

`run-top-speed.gif` and `run-top-speed.mp4` are off-screen MuJoCo renders of the stage-A
`jumper.run` checkpoint `logs/jumper/jumper.run/2026-10-01_02-00-41/model_9999.pt` at a
pinned 2.0 m/s command, recorded 2026-10-01 with `eval/render_top_speed_video.py`
(outside this repository). The MP4 is the best of five 10-second runs
(2.15 m/s peak, 2.03 m/s smoothed, no falls); the GIF is a 280-pixel, 12 fps FFmpeg
conversion of the same take, and `run-hero.png`, the README's header still, is
frame 60 of it at 900 pixels wide. The overlaid speed readout is `mjrl.viewer.live`'s chip
logic, re-rendered for the video with DIN Alternate Bold. Unlike the clips drawn by
`tools/readme_media.py`, this take comes from a `logs/` checkpoint, not a committed
export -- exporting the policy to `tasks/jumper.run/out/example/` and re-rendering is
the follow-up that makes it reproducible from the repository alone.
