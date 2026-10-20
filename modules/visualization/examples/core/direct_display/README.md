# `holoviz_direct_display`: exclusive (compositor-bypass) display output

Drives a display directly through HoloViz: no window, no compositor, no window system in
the frame loop. It enumerates the displays attached to each adapter with their ownership
classification, acquires one that is available, sets a mode, and renders a slow color sweep
through the regular swapchain machinery straight to the panel. It is the reference for
`Display`, `get_display()`, and `Display::create_surface()`, and the quickest way to
learn the display names `ExclusiveDisplaySinkDesc` expects.

## Quick start

```bash
# What is connected, who owns it, and which modes it has
holoviz_direct_display --list
holoviz_direct_display --list --all-modes

# Take over a panel (name substring), largest/fastest mode, 600 frames
holoviz_direct_display --display DP-3

# Pick a mode and run longer
holoviz_direct_display --display DP-3 --width 1920 --height 1080 --hz 144 --frames 2000
```

Every `--display` run ends by releasing the display. The run is bounded by `--frames`; there is
no key to press.

## Prerequisites (Linux, DRM / `VK_EXT_acquire_drm_display`)

Exclusive display is Linux-only in this release. On other platforms `--list` still enumerates,
`CapabilityReport::exclusive_display` is false, and `--display` fails with `kNotImplemented`.

- **No display server may own the GPU.** `--list` reports `compositor-owned` for panels a running
  session holds, and acquiring one fails (`... is owned by the compositor ...` or
  `Failed to become DRM master ...`). Stop the display manager and run from a TTY or over SSH:
  `sudo systemctl stop gdm` (`sddm`/`lightdm` as appropriate), `sudo systemctl start gdm` when
  done. Turning a panel off in the desktop settings is not enough: the X server or compositor
  stays DRM master for the whole GPU.
- **DRM device access.** Membership in the `video` and `render` groups, or explicit access to
  `/dev/dri/card*` and `/dev/dri/renderD*`.
- **Names are DRM connector names** (`DP-2`, `HDMI-A-1`, ...), not the X11/RandR names of the
  same panels: a panel called `DP-0` under X may be `DP-2` here. `--display` matches a
  case-sensitive substring.
- **In a container**, pass the DRM nodes and the `video`/`render` group ids, use the NVIDIA
  runtime, and set `NVIDIA_DRIVER_CAPABILITIES=all` so the Vulkan ICD and the display libraries
  are mounted:

  ```bash
  docker run --rm --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all --device /dev/dri \
      --group-add "$(getent group video | cut -d: -f3)" \
      --group-add "$(getent group render | cut -d: -f3)" <image> holoviz_direct_display --list
  ```

  The SDK's `./run launch` already adds both group ids and the NVIDIA runtime, and the build
  image sets `NVIDIA_DRIVER_CAPABILITIES=all`. If `ls /dev/dri` is empty inside the container,
  append `--device /dev/dri` to the `launch` command (extra arguments go to `docker run`). From a
  TTY there is no `DISPLAY`, so `launch` skips X forwarding; that is expected.
- **A sleeping panel (DPMS off)** stalls presents by about a second each; wake it before
  measuring.

## Command line

| Option | Meaning | Default |
|---|---|---|
| `--list`, `-l` | Enumerate adapters and their displays: ownership, current mode, VRR capability from the EDID, best mode. Implied when `--display` is absent. | |
| `--all-modes` | With `--list`, print every mode instead of the best one. | off |
| `--display <substring>`, `-d` | Acquire the first display whose name contains the substring and render to it. | |
| `--width W`, `--height H`, `--hz R` | Mode request. Unset dimensions mean "largest", unset refresh "fastest at that size"; the closest supported mode wins and an inexact match logs the alternatives. | largest / fastest |
| `--frames N` | Frames to render before releasing the display. | 600 |
| `--fifo` | `PresentMode::kFIFO`: vblank-paced queue. | |
| `--fifo-relaxed` | `PresentMode::kFIFORelaxed`: vblank-paced, but a frame that misses its vblank presents immediately (may tear) instead of waiting for the next one. | |
| `--immediate` | `PresentMode::kImmediate`: present as soon as the frame is ready; tears at fixed refresh. | |
| (neither) | `PresentMode::kFIFOLatestReady`: vblank-paced, stale queued frames replaced by the newest. | default |
| `--images N` | Swapchain image count request (clamped to the surface's range). Two is the low-latency setting. | 2 |

## What a run does

1. `get_display(filter)` finds the panel; `display.create_surface({.mode})` acquires it and
   sets the mode. The surface is fixed-size: there is no per-frame surface tracking, and any
   `kOutOfDate` would be a genuine display loss.
2. The device is created for that surface and its capabilities printed: the present modes the
   surface supports (as numeric `PresentMode` values) and the allowed image-count range.
3. `--frames` frames are rendered: acquire, record a full-image clear whose color advances along
   a sine of the frame index, submit against the acquire and present gates, present. Once per
   second the loop prints its present rate; when the device reports `present_completion`, the
   first frames also print `Display-done spacing (ms)`, the true scanout period while the queue
   is still full.
4. The swapchain is destroyed before the surface it was created on, which releases the display.

## Reading the output

```text
Latching onto display "DP-3" (available for direct display, VRR/G-Sync advertised by EDID)
Display surface: 1920x1080
Using device: <your GPU>
exclusive_display capability: yes
Present modes: 1; 2; 4; (image count [2, 8])
Display-done spacing (ms): 6.94 6.95 6.94 6.95 6.94
  143.9 fps
  ...
Rendered 600 frames in 4.17 s (143.9 fps average); releasing the display.
```

Display names, modes, the image-count range, and the rates depend on your GPU, driver, and panel;
the transcript above is a 1920x1080 144 Hz panel.

- **VRR/G-Sync advertised by EDID** (`Display::vrr_capable`, with the range floor from
  `Display::vrr_min_refresh_mhz` in `--list`) says the panel and connector advertise variable
  refresh. It is a capability, not a measurement: whether the driver engages VRR on this path is
  not reported.
- **Present modes** are the numeric `PresentMode` values the surface supports (`1` FIFO,
  `2` FIFO-latest-ready, `4` immediate).
- **fps** lines count presents in one-second windows, not refreshes. Direct-display presents on
  current NVIDIA drivers do not block on vblank, so the number can exceed the refresh rate.

## Exit status

Zero after a completed run or listing. Non-zero, with the error on stderr, when no display
matches, the display cannot be acquired, or a swapchain operation fails.
