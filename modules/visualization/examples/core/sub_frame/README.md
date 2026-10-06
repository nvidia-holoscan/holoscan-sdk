# `holoviz_sub_frame`: sub-frame (band-by-band) visualizer with latency measurement

This example shows a video frame that arrives in pieces. A camera sensor with a
rolling shutter, or a capture card streaming slices, hands over a frame as N
horizontal bands over time; the visualizer draws each band into the presentable
image the moment its content lands, and the frame goes to the display once the
last band is in. A CUDA kernel stands in for the sensor here, signalling one
band at a time.

Two things make the example more than a demo:

- It is the 5.x translation of the holoviz 4.x sub-frame pattern
  (`DONT_CLEAR_COLOR | DONT_CLEAR_DEPTH | DONT_SWAP_BUFFERS` accumulation and
  the exclusive-display "FPO" path). The mapping table at the top of `Main.cpp`
  spells out which 4.x flag became which 5.x call.
- It measures, per frame and to a few microseconds, the time between the last
  band landing on the GPU and the first pixel of that frame leaving the display
  engine. That is the number a "present just in time for the flip" pipeline
  has to design around, and the example reports it in every output mode.

## Quick start

```bash
# Decorated window, FIFO, runs until the window is closed (ESC closes only fullscreen windows)
holoviz_sub_frame

# Borderless fullscreen on one monitor (X11/GLFW names), 2000 frames
holoviz_sub_frame --fullscreen --monitor DP-2 --frames 2000

# Fullscreen with G-Sync/VRR engaged on NVIDIA X11: the G-Sync panel must be the only
# active output, and the present rate must sit below its maximum refresh (see below)
xrandr --output DP-0 --off
holoviz_sub_frame --fullscreen --monitor DP-2 --fifo --fps 80
xrandr --output DP-0 --auto

# Exclusive display (DRM/Vulkan names; see holoviz_direct_display --list)
holoviz_sub_frame --display DP-3

# Exclusive display, bands streamed straight to the panel with no per-frame present
holoviz_sub_frame --display DP-3 --shared-continuous

# Turn the measurement off, or cross-check it against the host clock
holoviz_sub_frame --display DP-3 --no-latency
holoviz_sub_frame --display DP-3 --verify-timing
```

The Slang shader is compiled to SPIR-V at build time and embedded in the
executable; the example ships no data files.

## Command line

| Option | Meaning | Default |
|---|---|---|
| `--bands N` | Bands per frame (1 to the frame height of 600). More bands means more, smaller submits per frame. A band that owns no output rows (the window has fewer rows than bands, e.g. after a resize) is still uploaded and submitted but records no render pass. | 8 |
| `--frames N` | Stop after N frames. | windowed: until the window closes; display modes: 600 |
| `--fullscreen` | Borderless fullscreen GLFW window instead of a decorated one. Does not take exclusive ownership of the panel. ESC or closing the window exits. | off |
| `--monitor <name>` | GLFW monitor for `--fullscreen` (X11/RandR names such as `DP-0`). On Linux this also sets `__GL_SYNC_DISPLAY_DEVICE` to that name, if not already set, so the NVIDIA X11 driver paces presents on that monitor's vblank rather than the primary's. | primary |
| `--display <substring>` | Exclusive scanout to the display whose DRM/Vulkan name contains the substring. Mutually exclusive with `--fullscreen`. | |
| `--width W`, `--height H` | Video mode (with `--fullscreen`) or display mode request (with `--display`). | monitor's current / largest |
| `--hz R` | Refresh rate request (with `--fullscreen` or `--display`). | fullscreen: monitor's current; display: fastest at that size |
| `--fps N` | Cap the present rate at N per second in a window (host sleep between frames). Holding presents below the panel's maximum refresh is what lets G-Sync/VRR engage in a fullscreen window; see [G-Sync in a fullscreen window](#g-sync-in-a-fullscreen-window). Not available with `--display`, which paces on display events. | unpaced |
| `--immediate`, `--fifo` | Present mode. | FIFO in a window; FIFO-latest-ready on a display |
| `--shared-demand`, `--shared-continuous` | Shared presentable image (with `--display`): bands stream to the panel as their submits execute. | |
| `--fps-cap R` | Sleep after each present to hold R frames per second. With `--fifo` on a `--fullscreen` window this is the VRR test: the `FPO period` line reads 1/R (with a tight min/max) when the panel follows the presents, and snaps to integer multiples of the refresh period when it does not. | off |
| `--no-latency` | Skip the latency measurement even when the driver supports present timing. | measurement on |
| `--verify-timing` | Per frame, also block on `wait_for_present()`, stamp the host clock and compare it with the sample mapped through `Device::calibrated_timestamps()`. Serializes the loop. | off |

Display names differ between the two paths: `--monitor` takes the X11 name of a
panel, `--display` its DRM connector name, and the same physical panel can be
`DP-0` in one and `DP-2` in the other. `holoviz_direct_display --list`
prints the DRM names. The `--display` path has the same prerequisites as that
example (no compositor owning the panel, DRM device access); its
[README](../direct_display/README.md) lists them.

## How a frame is rendered

1. **Source image.** An 800x600 `ShareMode::kExported` image is the simulated
   sensor frame. Each band, a CUDA kernel writes a moving test pattern into its
   slice of that image on a CUDA stream and signals an `ExternalSemaphore`.
2. **Band pass.** For each band the loop records a short command list that
   samples the source image onto a full-screen quad, with the viewport covering
   the whole presentable image and the scissor limited to the band's rectangle.
   The render pass loads the existing contents (`LoadOp::kLoad`), so bands
   already drawn stay intact. The submit waits the band's CUDA semaphore with
   destination stage `ALL_COMMANDS`, so nothing in the batch runs before the
   band's data exists.
3. **Present once per frame.** All bands render into the same acquired image;
   after the last band the frame is presented. Because every frame overwrites
   every band, no stale content from `image_count` presents ago can reach the
   screen, which is what makes present-per-frame sub-frame rendering correct
   without per-image convergence tracking.
4. **Shared continuous** (`--shared-continuous`) is the 4.x FPO path: the
   presentation engine scans out one shared image continuously, so after one
   initial present each band appears on the panel as its submit executes and
   there are no further present calls. The loop is then paced by first-pixel-out
   display events. On NVIDIA drivers the present-timing feedback for shared images
   reads a fixed 60 Hz regardless of the display mode; whether scanout itself is
   capped is still being verified.

### Pacing on a display

Direct-display presents on current NVIDIA drivers do not block, so on the
`--display` path the loop paces every mode on
`Device::wait_for_display_event(kFirstPixelOut)`. In a window, FIFO paces
itself through acquire, and `--fps N` adds a host sleep so the loop presents at
most N times per second (a late frame resets the deadline rather than catching
up in a burst).

### G-Sync in a fullscreen window

Three things have to hold for the NVIDIA X11 driver to run a window at variable
refresh, and the driver's G-Sync visual indicator (`nvidia-settings`
*ShowGSYNCVisualIndicator*) reads `NORMAL` until all do:

1. **The window is flipped, not composited.** `--fullscreen` creates an
   undecorated, monitor-sized GLFW window; on X11 GLFW also sets
   `_NET_WM_BYPASS_COMPOSITOR=1`, which compositors such as Mutter honour by
   unredirecting the window. `xprop` on the window should show
   `_NET_WM_STATE_FULLSCREEN` and `_NET_WM_BYPASS_COMPOSITOR = 1`. The
   `--display` path needs none of this.
2. **The G-Sync panel is the only display on the X screen.** The driver flips a
   window only when it is "unoccluded, unredirected, and covering the entire
   desktop" (NVIDIA, [developer forums](https://forums.developer.nvidia.com/t/any-plans-to-allow-g-sync-on-an-x-window-system-that-spans-multiple-monitors-variable-non-variable-refresh-rate-monitors/51209)).
   With a second monitor on the same X screen a fullscreen window on the G-Sync
   panel covers only part of the desktop, so the driver blits instead and the
   panel stays at fixed refresh however the presents are paced. Measured here:
   with two monitors, `--fullscreen --monitor DP-2 --fifo --fps 80` reports
   `FPO period` values alternating between 6.9 and 13.9 ms (the 144 Hz
   quantisation), and without `__GL_SYNC_DISPLAY_DEVICE` the presents pace on
   the *primary's* 180 Hz vblank instead, which is the blit path's signature.
   Turn the other outputs off for the run (`xrandr --output DP-0 --off`, and
   `--auto` afterwards), put the G-Sync panel on its own X screen, or use
   `--display`, which owns the panel outright and needs none of this. Verified
   on the same rig: with DP-0 off, `--fullscreen --monitor DP-2 --fifo --fps 80`
   switches the driver's indicator from `NORMAL` to `G-SYNC`.
3. **The present rate is below the panel's maximum refresh.** An unpaced FIFO
   window presents once per refresh, so the refresh has nothing to follow and
   the panel stays at the mode's maximum. `--fps N` holds the present rate in
   the panel's VRR range (for example `--fps 80` on a 48–144 Hz panel).
   Prefer `--fifo --fps 80`: it is tear-free, and its `FPO` samples are real
   scanout times, whereas in immediate mode on a window the sample is the flip
   and follows the presents whether or not the panel does.

The panel itself must be in variable-refresh mode on the X screen
(`nvidia-settings -q [dpy:<n>]/DisplayVRREnabled` reads 1; unvalidated
Adaptive-Sync panels need the MetaMode attribute `AllowGSYNCCompatible=On`).
Checklist when the indicator stays on `NORMAL`:

| Symptom | Cause | Fix |
|---|---|---|
| `FPO period` steps in multiples of the *primary's* refresh although the window is on another panel | window is blitted, not flipped: another output is active on the X screen | `xrandr --output <other> --off` for the run, or `--display` |
| `FPO period` alternates one and two refresh periods at the panel's rate | flipping, but still fixed refresh: another output active, or presents at the maximum rate | as above, plus `--fps` below the maximum |
| frame rate equals the refresh rate | unpaced FIFO | `--fps N` inside the VRR range |
| `DisplayVRREnabled` reads 0 for the panel | panel not in VRR mode (unvalidated, or G-Sync disallowed) | MetaMode `AllowGSYNCCompatible=On`, `nvidia-settings` *Allow G-SYNC* |
| `xprop` lacks `_NET_WM_BYPASS_COMPOSITOR = 1` / fullscreen state | not a GLFW monitor window | use `--fullscreen --monitor <name>`, not a manually sized window |

Whether VRR is really engaged shows in the `FPO period` line: under VRR it
tracks the present period (about 12.5 ms at 80 fps), at fixed refresh it
quantises to multiples of the refresh period (6.9 / 13.9 ms at 144 Hz).

## The latency measurement

Enabled automatically when the driver reports `CapabilityReport::present_timing`
for the surface; the header line says so, and `Latency stats: <reason>` explains
why not otherwise. Three instants are stamped per frame, all on the GPU's clock:

| Mark | Where | What it means |
|---|---|---|
| T0' band arrival | `TimestampLocation::kStart` (top of pipe) at the head of the last band's command list | An upper bound on when the last band's CUDA signal landed: the submit cannot start before it, so this is arrival plus queue wake-up (microseconds). |
| T0 frame complete | `TimestampLocation::kEnd` (bottom of pipe) after the last band's render | Every pixel of the frame is in the swapchain image and the present gate is armed. |
| T1 first pixel out | `PresentTimingSample::first_pixel_out_ns` for the frame's present id | The display engine started scanning out this frame (`VK_EXT_present_timing`, device time domain). |

Each present carries an id; the GPU timestamps arrive one swapchain lap later
(read after the transient command list's `reset()`, which already waited for
that submit), the present-timing sample one to three refreshes later, and the
two halves meet in a map keyed by present id. Dropped presents (FIFO-latest-ready
replacing a frame), recreations, and samples that never arrive are counted, not
waited for.

Reported once per second and summarised on exit:

- `render-done->FPO`: T1 minus T0, average, min and max over the window. The
  headline number.
- `band-arrival->FPO`: T1 minus T0', normally a few microseconds more.
- `last-pixel-out`: T1 minus T0 plus one refresh period; an estimate, and only
  a lower bound under VRR.
- `FPO period`: spacing between consecutive first-pixel-out times, i.e. the
  refresh the frames actually got.
- `lost: ...`: counters for dropped, incomplete, pruned, recreated and
  anomalous records.

With `--verify-timing` a fourth line compares the host time at which
`wait_for_present()` returned against T1 mapped onto the host clock with
`Device::calibrated_timestamps()`; agreement within tens of microseconds
confirms the device-clock measurement and the calibration deviation is printed.

The header also names the driver's refresh-mode claim (fixed or variable,
`Swapchain::refresh_mode()`), and the loop prints a line when it changes. On
NVIDIA this claim reads "variable" for any Adaptive-Sync-capable panel once
frames are on screen, whether or not VRR is really engaged; use
`holoviz_direct_display --display <name>` for the measured verdict on a
display, or the `FPO period` line under `--fps` in a fullscreen window.

## Reading the output

```
Present timing: device clock, stages queue-ops-end first-pixel-out; refresh 6.945 ms (144.0 Hz); fixed refresh
    refresh mode (driver) now: driver reports variable refresh (...)
  143.9 fps | frame time avg 6.96 ms, min 6.88 ms, max 7.03 ms (144 frames)
    render-done->FPO avg 0.02 ms, min 0.01 ms, max 0.03 ms (136 samples)
    band-arrival->FPO avg 0.03 ms, min 0.02 ms, max 0.04 ms | last-pixel-out +6.95 ms
    FPO period avg 6.95 ms, min 6.90 ms, max 7.01 ms
    verify: wait_for_present - FPO avg 0.03 ms, min 0.02 ms, max 0.06 ms | calibration deviation <= 1.8 us
    lost: 0 dropped, 0 incomplete, 2 no-sample, 0 recreated, 0 anomalies, 0 gpu-readback
Overall latency (after warm-up):
    render-done->FPO ...
```

The `verify:` line appears only with `--verify-timing`, the `lost:` line only
when a counter is non-zero, and `FPO` reads `flip` for immediate mode in a
window (see below). Values here are illustrative of the format, not a
measurement.

What the numbers typically look like, and why (figures from one test rig with
a fixed-refresh 1440p@180 panel and a VRR 1080p@144 panel; the method carries
over to other hardware, the values do not):

- **Decorated window, FIFO**: several milliseconds. The frame renders in well
  under a millisecond and then waits for the next vblank, so the value is most
  of one refresh period, quantised to the refresh.
- **Fullscreen window**: about one refresh more than the decorated window; the
  borderless path goes through the flip queue rather than the compositor.
- **Immediate in a window**: tens of microseconds, but labelled `flip`: the
  driver reports the flip into the compositor, not a vblank-aligned scanout.
- **Exclusive display, any mode**: about 20 microseconds. The driver starts
  scanning out the presented image right at the present. On a fixed-refresh
  panel that is a mid-scan flip (the frame completes on screen a refresh later,
  with a tear at the boundary); under VRR the refresh itself follows the present
  and there is no tear.

The measurement lives on the device clock because present-timing samples are
reported in that domain; `--verify-timing` cross-checks it against the host clock
through `Device::calibrated_timestamps()`.

## Adapting it to a real sensor

Replace the CUDA pattern kernel with the capture path that writes bands into an
exported image and signals a semaphore per band; keep T0' as the "data arrived
on the GPU" mark. Nothing in the measurement chain changes. If the sensor drives
the timing, the numbers here tell you how long before the intended first pixel
out the last band has to be complete.

## VRR through a fullscreen window (X11)

`--fullscreen --monitor <name> --fifo --fps-cap 70` is the windowed counterpart of
`holoviz_direct_display --hz-sweep`: the first-pixel-out period either follows the
70 fps presents or snaps to 2 and 3 refreshes (144 Hz) / 3 and 4 refreshes (240 Hz).
On the X11 path the NVIDIA driver engages G-Sync only for a window it *flips*, and it
flips only an unredirected fullscreen window; a composited (redirected) window is
blitted and stays at fixed refresh whatever `nvidia-settings` says. The driver's own
overlay tells which case applies: `nvidia-settings -a ShowGraphicsVisualIndicator=1
-a ShowVRRVisualIndicator=1` prints `FLIP`/`BLIT` and `G-SYNC`/`NORMAL` in the window.
Measured 2026-09-04 (driver 595.84, GNOME Shell 46 on Xorg, Ubuntu 24.04, ASUS PG279QM at
240 Hz, `--fps-cap 70`): the window is fullscreen, monitor-sized and carries
`_NET_WM_BYPASS_COMPOSITOR=1`, yet the overlay read `BLIT - VSYNC ON` / `NORMAL` until the
legacy XEmbed tray of the Ubuntu AppIndicator extension was turned off; with it off (or the
extension disabled) the same run read `FLIP - VSYNC ON` / `G-SYNC`. A legacy tray icon (here
the RealVNC Server icon) keeps Mutter compositing the fullscreen window, and a composited
window is never flipped. The one-line fix on such a desktop is
`gsettings set org.gnome.shell.extensions.appindicator legacy-tray-enabled false`
(or `gnome-extensions disable ubuntu-appindicators@ubuntu.com`); the other Ubuntu
extensions (dock, desktop icons, tiling assistant) do not interfere. The per-display driver
state is readable with
`nvidia-settings -q [dpy:N]/DisplayVRRMode -q [dpy:N]/DisplayVRREnabled` (mode 2 =
validated G-SYNC Compatible, 3 = unvalidated, which needs `AllowGSYNCCompatible=On` in
the MetaMode before `DisplayVRREnabled` turns 1). Note that `Present timing:` still
prints `fixed refresh` at start-up because the driver reports the mode before any frame
is on screen; the overlay and the panel's own refresh read-out are the authority.
