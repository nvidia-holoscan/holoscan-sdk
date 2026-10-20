# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gray_scale_example (Python) - a 16-bit grayscale image displayed natively by HolovizOp.

    GrayScaleSource --(schema.Image, MONO16, uint16 {H, W})--> HolovizOp (image layer r16_unorm)

This is the Python port of ``../cpp``. The same smooth 16-bit ramp with a sweeping bar is rendered
here with numpy on the host instead of a CUDA kernel; HolovizOp uploads the two-byte samples as an
R16_UNORM texture and replicates the single channel to gray. The source, the transport and the
upload stay 16-bit; only the window's BGRA8 swapchain quantizes the final displayed color to 8 bits
per channel. The operator is authored through the typed declarations of
``holoscan_visualization.operators`` (``ImageLayer``, ``WindowSink``, ``ExclusiveDisplaySink``,
``LayerTrigger``), which lower to the same tagged dictionaries the facade also accepts.
"""

import argparse
import os
import signal
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
from holoscan_visualization.operators import (
    ExclusiveDisplaySink,
    HolovizOp,
    ImageFormat,
    ImageLayer,
    LayerTrigger,
    PresentMode,
    WindowSink,
)

import holoscan as hs
from holoscan.schema import Image
from holoscan.schema.image import ColorSpace, ImageEncoding

WIDTH = 640
HEIGHT = 480
USAGE = (
    "usage: gray_scale_example.py [--seconds N] [--fps N]"
    " [--fullscreen [MONITOR] | --display [NAME]] [--hz HZ]"
)
# Mirrors the C++ kMaxFps and kMaxSeconds: `seconds * fps` must fit an unsigned 64-bit count.
MAX_FPS = 1000
MAX_SECONDS = (2**64 - 1) // MAX_FPS
UINT32_MAX = 2**32 - 1


class GrayScaleSource(hs.Operator):
    """Clock-driven source of animated 16-bit grayscale frames rendered with numpy.

    Emits ``holoscan.schema.Image`` frames (MONO16-encoded, ``{height, width}`` uint16 host tensor
    data) at a fixed period, matching HolovizOp's image-kind ports. The pattern is a smooth
    diagonal ramp over the full 0..65535 range with a sweeping bright bar, so that a consumer
    misreading the samples as 8-bit would show visible banding or garbage.
    """

    frame = hs.Output("frame", Image)
    fps = hs.Param(60, range=(1, MAX_FPS))
    # Frames = seconds * fps, after which the source completes naturally (the run then drains to
    # COMPLETED); zero leaves the source unbounded.
    seconds = hs.Param(0, range=(0, MAX_SECONDS))

    def setup(self, spec: hs.OperatorSpec) -> None:
        """Declare the pooled frame output."""
        spec.output(self.frame).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLUINT, 16, 1),
                rank=2,
                transfer=hs.tensor.TensorTransferPolicy(
                    foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
                ),
            ),
            hs.tensor.byte_span(WIDTH * HEIGHT * 2),
            field="data",
        )

    def contract(self) -> hs.Contract:
        """Tick at the configured frame rate and, with --seconds, complete after its frames."""
        result = hs.Contract().trigger(hs.OnClock(period=hs.time.ns(1_000_000_000 // self.fps)))
        if self.seconds:
            result = result.complete_after(self.seconds * self.fps)
        return result

    def start(self) -> None:
        """Precompute the phase-free ramp once so each frame is two vector operations."""
        self.frame_index = 0
        y, x = np.indices((HEIGHT, WIDTH), dtype=np.float32)
        # The base is a diagonal ramp that spans the whole 16-bit range across the frame diagonal;
        # on a 640-wide frame neighbouring pixels differ by ~60 codes, far below one 8-bit step.
        self.ramp = (x + y) / np.float32(WIDTH + HEIGHT - 2)
        self.columns = np.arange(WIDTH)
        self.bar_half_width = WIDTH // 40 + 1

    def compute(self, context: hs.ExecutionContext) -> None:
        """Render one frame and publish it."""
        # Same arithmetic as the CUDA kernel in ../cpp/gray_scale_source.cu; float32 rounding may
        # differ from the device by one code on a few pixels, which is invisible on screen.
        phase = np.float32((self.frame_index % 1024) / 1024.0)
        t = self.ramp + phase
        t -= np.floor(t)  # wrap into [0, 1)
        pixels = (t * np.float32(65535.0)).astype(np.uint16)
        # A bright vertical bar sweeps left to right on top of the ramp.
        bar_x = (self.frame_index * 2) % WIDTH
        pixels[:, np.abs(self.columns - bar_x) <= self.bar_half_width] = 65535
        self.frame_index += 1

        self.frame.emit(
            Image(
                height=HEIGHT,
                width=WIDTH,
                encoding=ImageEncoding.MONO16,
                color_space=ColorSpace.UNSPECIFIED,
                data=hs.Tensor.as_tensor(pixels),
            ),
            options=hs.EmitOptions(capture_time=context.activation_time),
        )


@dataclass(frozen=True, kw_only=True)
class Arguments:
    """Parsed command line, mirroring the C++ ``Arguments`` struct."""

    seconds: int  # --seconds: frames = seconds * fps (0: run until Ctrl-C)
    fps: int  # source rate (display follows it); keep inside the VRR range
    output: Literal["window", "fullscreen", "display"]
    # --fullscreen: window-system monitor name (empty = primary). --display: case-sensitive
    # substring of the DRM connector name, e.g. "DP-2" (empty = the first display no compositor
    # owns). The two namespaces differ: `holoviz_direct_display --list` prints the DRM names.
    display_name: str
    # --fullscreen / --display. 0 means: fullscreen keeps the monitor's current rate; exclusive
    # display picks the fastest refresh at the selected (largest) resolution.
    refresh_hz: int


class _Parser(argparse.ArgumentParser):
    """Report malformed arguments the way the C++ example does and exit with status 1."""

    def error(self, message: str) -> None:
        # argparse prefixes validator messages with "argument --x: "; drop it so the diagnostic
        # starts with the same text the C++ binary prints.
        prefix, separator, detail = message.partition(": ")
        if prefix.startswith("argument ") and separator and detail.startswith("invalid "):
            message = detail
        print(message, file=sys.stderr)
        print(USAGE, file=sys.stderr)
        sys.exit(1)


def _nonnegative(option: str, text: str, *, low: int, high: int, expected: str) -> int:
    # int() would accept "+5", " 5", "5_0" and "-0"; std::from_chars accepts none of them.
    if not text.isdigit() or not low <= int(text) <= high:
        raise argparse.ArgumentTypeError(f"invalid {option} value: '{text}' ({expected})")
    return int(text)


def _seconds(text: str) -> int:
    return _nonnegative(
        "--seconds", text, low=0, high=MAX_SECONDS, expected="expected a nonnegative integer"
    )


def _fps(text: str) -> int:
    return _nonnegative("--fps", text, low=1, high=MAX_FPS, expected="expected 1..1000")


def _hz(text: str) -> int:
    return _nonnegative(
        "--hz", text, low=0, high=UINT32_MAX, expected="expected a nonnegative refresh rate"
    )


def parse_arguments(argv: list[str]) -> Arguments:
    """Parse the command line with the same options and diagnostics as the C++ example."""
    parser = _Parser(
        prog="gray_scale_example.py",
        usage=USAGE,
        description=__doc__.split("\n\n")[0],
        allow_abbrev=False,
    )
    parser.add_argument("--seconds", type=_seconds, default=0, metavar="N")
    parser.add_argument("--fps", type=_fps, default=60, metavar="N")
    output = parser.add_mutually_exclusive_group()
    # An optional monitor/display name follows unless the next token is another option.
    output.add_argument("--fullscreen", nargs="?", const="", default=None, metavar="MONITOR")
    output.add_argument("--display", nargs="?", const="", default=None, metavar="NAME")
    parser.add_argument("--hz", type=_hz, default=0, metavar="HZ")
    args = parser.parse_args(argv)

    if args.display is not None:
        mode, name = "display", args.display
    elif args.fullscreen is not None:
        mode, name = "fullscreen", args.fullscreen
    else:
        mode, name = "window", ""
    if args.hz and mode == "window":
        parser.error("--hz requires --fullscreen or --display")
    return Arguments(
        seconds=args.seconds, fps=args.fps, output=mode, display_name=name, refresh_hz=args.hz
    )


def make_sink(arguments: Arguments) -> WindowSink | ExclusiveDisplaySink:
    """Describe where HolovizOp presents.

    Exclusive display: width/height 0 select the display's largest resolution and refresh_hz 0 the
    fastest refresh at that resolution, so a modeset may occur even with the defaults; there is no
    window, so the run ends with Ctrl-C or --seconds. A fullscreen window takes the monitor's
    current video mode (width/height 0) and closes on ESC, since an undecorated window has no close
    button. It also pins plain FIFO presentation: the default AUTO prefers FIFO_LATEST_READY when
    the driver offers it, which the NVIDIA X11 driver flips with G-Sync as well but reports as
    "VSYNC OFF" in its overlay; FIFO is the unambiguous vsync path for a G-Sync demonstration.
    """
    if arguments.output == "display":
        return ExclusiveDisplaySink(
            display_name=arguments.display_name, refresh_hz=arguments.refresh_hz
        )
    fullscreen = arguments.output == "fullscreen"
    return WindowSink(
        title="gray_scale_example (16-bit, Python)",
        width=0 if fullscreen else WIDTH,
        height=0 if fullscreen else HEIGHT,
        present_mode=PresentMode.FIFO if fullscreen else PresentMode.AUTO,
        fullscreen=fullscreen,
        display_name=arguments.display_name,
        refresh_hz=arguments.refresh_hz,
        close_on_escape=fullscreen,
    )


def build_graph(arguments: Arguments) -> hs.Graph:
    """Compose source -> HolovizOp."""
    graph = hs.Graph("gray-scale-example", default_clock=hs.RealtimeClock)

    # The source sets the frame rate (--fps); HolovizOp presents each frame as it arrives (the
    # LayerTrigger on layer 0), so the display follows the source with no clock in between. For
    # G-Sync/VRR the rate has to sit inside the panel's variable-refresh range and below its
    # maximum: a 60 Hz default on a 60..144 Hz panel is at the floor, where jitter makes the driver
    # frame-double instead of following the presents.
    source = graph.op(GrayScaleSource, "source", fps=arguments.fps, seconds=arguments.seconds)

    # The layer format tells HolovizOp the tensor holds one uint16 per pixel; the operator checks
    # the incoming dtype and channel count against it and drops mismatching samples.
    visualizer = HolovizOp(
        graph,
        "holoviz",
        layers=[
            ImageLayer(
                port_name="gray",
                format=ImageFormat.R16_UNORM,
                max_width=WIDTH,
                max_height=HEIGHT,
            )
        ],
        sink=make_sink(arguments),
        trigger=LayerTrigger(layer=0),
    )
    # Depth-1 "latest frame" mailbox: never backpressure the source while the window starts up.
    graph.add_flow(
        source.frame, visualizer.gray, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1
    )
    graph.partition("display").add(visualizer)
    graph.auto_partition_remaining()
    return graph


def _raise_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: list[str]) -> int:
    """Run the example until Ctrl-C, SIGTERM, or the --seconds frame count completes."""
    arguments = parse_arguments(argv)
    # Python installs no SDK signal handler: SIGINT already arrives as KeyboardInterrupt, and
    # SIGTERM is routed to the same cooperative stop below.
    signal.signal(signal.SIGTERM, _raise_interrupt)

    # The NVIDIA ICD reads __GL_SYNC_DISPLAY_DEVICE once, when Vulkan first loads in the process.
    # HolovizOp creates its Vulkan instance in start(), so exporting it here, before the graph
    # runs, is early enough; HolovizOp itself only logs a hint. An explicit setting wins.
    if os.name != "nt" and arguments.output == "fullscreen" and arguments.display_name:
        os.environ.setdefault("__GL_SYNC_DISPLAY_DEVICE", arguments.display_name)

    plan = hs.compile(build_graph(arguments))
    if not plan:
        print(plan.json(), file=sys.stderr)
        return 1

    # Like the C++ binary: with --seconds the source completes after seconds * fps frames,
    # HolovizOp drains, and the run ends COMPLETED (exit 0). Ctrl-C or SIGTERM requests a
    # cooperative stop, which is reported as a nonzero exit.
    with hs.run_async(plan) as session:
        try:
            while not session.wait(timeout=0.05):
                pass
        except KeyboardInterrupt:
            session.request_stop()
        session.wait()

    for diagnostic in session.diagnostics:
        print(f"diagnostic: {diagnostic.token} vertex={diagnostic.source.vertex}", file=sys.stderr)
    if session.termination is not hs.RunTermination.COMPLETED or session.diagnostics:
        print(
            f"gray_scale_example: run ended with termination={session.termination.name}"
            f" diagnostics={len(session.diagnostics)}",
            file=sys.stderr,
        )
        return 2
    print("gray_scale_example complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
