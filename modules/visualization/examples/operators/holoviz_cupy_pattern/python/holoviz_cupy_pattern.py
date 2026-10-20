# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Animate a CuPy RGBA pattern through a native Python Image source and Holoviz.

Run with one visible CUDA device. The duration covers the session wait;
separate qualification tests check emitted and rendered image values.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys

from holoscan_visualization.operators import HolovizOp

import holoscan as hs
from holoscan.schema import Image
from holoscan.schema.image import ColorSpace, ImageEncoding

SKIP_EXIT_CODE = 77


class UnavailableError(RuntimeError):
    """An optional prerequisite is absent, so a smoke run cannot execute."""


def load_cupy():
    """Import the optional array library without hiding broken installations."""
    try:
        return importlib.import_module("cupy")
    except ModuleNotFoundError as error:
        if error.name == "cupy":
            raise UnavailableError("CuPy dependency is not installed") from error
        raise


def require_single_device(cupy) -> int:
    """Keep allocation and renderer device selection in one qualified configuration."""
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        if error.status == 100:  # cudaErrorNoDevice
            raise UnavailableError("no CUDA device available") from error
        raise
    if count == 0:
        raise UnavailableError("no CUDA device available")
    if count != 1:
        raise ValueError("this example requires exactly one visible CUDA device")
    return 0


def parse_args(argv=None):
    """Validate options before importing CuPy or constructing a graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0, help="session duration in seconds")
    parser.add_argument(
        "--fps", type=float, default=30.0, help="pattern updates per second (up to 1000)"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args(argv)
    for option in ("seconds", "fps"):
        value = getattr(args, option)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"invalid --{option.replace('_', '-')} value: {value}")
    if args.fps > 1000:
        parser.error("--fps must not exceed 1000")
    for option in ("width", "height"):
        if not 1 <= getattr(args, option) <= 8192:
            parser.error(f"--{option} must be between 1 and 8192")
    return args


def fill_pattern(cupy, tensor, shape, coordinates, frame):
    """Fill a borrowed Image loan on the current runtime stream, without retaining its view."""
    dtype = tensor.dtype
    if (
        tensor.shape != shape
        or tensor.nbytes != math.prod(shape)
        or (dtype.code, dtype.bits, dtype.lanes) != (1, 8, 1)
        or tensor.__dlpack_device__() != (2, 0)
        or tensor.is_read_only
        or not tensor.is_contiguous()
        or not tensor.data
    ):
        raise ValueError("expected a writable contiguous RGBA8 Image loan on CUDA device 0")
    # UnownedMemory retains this Tensor but neither frees nor recycles the provider buffer.
    # The caller holds the write guard until this scoped view and its writes are submitted.
    memory = cupy.cuda.UnownedMemory(tensor.data, tensor.nbytes, owner=tensor, device_id=0)
    rgba = cupy.ndarray(shape, dtype=cupy.uint8, memptr=cupy.cuda.MemoryPointer(memory, 0))
    x, y = coordinates
    phase = frame * 0.2
    rgba[:, :, 0] = (127.5 * (1 + cupy.sin(x + phase))).astype(cupy.uint8)
    rgba[:, :, 1] = (127.5 * (1 + cupy.sin(y - phase))).astype(cupy.uint8)
    rgba[:, :, 2] = (127.5 * (1 + cupy.sin(x + y + phase))).astype(cupy.uint8)
    rgba[:, :, 3] = 255


def pattern_source(config, cupy):
    """Create a zero-argument hosted operator with invocation-local configuration."""

    class PatternSource(hs.Operator):
        """Fill and publish provider-backed RGBA8 Images without overwriting held frames."""

        out = hs.Output("out", Image)

        def setup(self, spec):
            """Declare runtime-pool CUDA storage for the Image's RGBA8 tensor field."""
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="runtime_pool",
                    memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                    dtype=hs.tensor.DLDataType(1, 8, 1),
                    rank=3,
                ),
                hs.tensor.byte_span(config.height * config.width * 4),
                field="data",
            )

        def contract(self):
            """Pace invocations with a clock trigger at the configured frame rate."""
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ns(int(1e9 / config.fps))))

        def start(self):
            """Initialize the next frame index and reusable coordinate vectors."""
            self.frame = 0
            self.coordinates = None

        def compute(self, context):
            """Generate pixels in an Image pool loan and publish on guard exit."""
            stream = context.stream()
            with cupy.cuda.Device(0), stream:
                if self.coordinates is None:
                    self.coordinates = (
                        cupy.arange(config.width, dtype=cupy.float32)[None, :] * 0.05,
                        cupy.arange(config.height, dtype=cupy.float32)[:, None] * 0.05,
                    )
                shape = (config.height, config.width, 4)
                image = Image(
                    height=config.height,
                    width=config.width,
                    encoding=ImageEncoding.RGBA8,
                    color_space=ColorSpace.UNSPECIFIED,
                )
                request = hs.tensor.LoanRequest(shape=shape, dtype=hs.tensor.DLDataType(1, 8, 1))
                try:
                    loan = self.out.allocate_tensors(image, {"data": request})
                except hs.errors.ResourceExhaustedError:
                    return  # A consumer still holds the bounded pool; try the next clock tick.
                with loan:
                    fill_pattern(cupy, loan.tensors["data"], shape, self.coordinates, self.frame)
            self.frame += 1

    return PatternSource


def add_visualizer(graph, config):
    """Add a window displaying the RGBA image layer."""
    return HolovizOp(
        graph,
        "visualizer",
        layers=[
            dict(
                kind="image",
                port_name="video",
                format="rgba8_unorm",
                max_width=config.width,
                max_height=config.height,
            )
        ],
        sink=dict(
            kind="window", width=config.width, height=config.height, title="CuPy Image pattern"
        ),
        trigger=dict(kind="layer", layer=0),
    )


def compile_options():
    """Bind the source Tensor output to the sole visible CUDA device."""
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_tensor_output_device(
        operator_path="source", output_port="out", device=hs.execution.DeviceId(0)
    )
    return hs.CompileOptions(deployment=deployment)


def build_graph(config, cupy):
    """Connect the paced producer to a bounded, drop-oldest display edge."""
    graph = hs.Graph("holoviz-cupy-pattern", default_clock=hs.RealtimeClock)
    source = graph.op(pattern_source(config, cupy), "source")
    visualizer = add_visualizer(graph, config)
    graph.add_flow(
        source.out, visualizer.video, queue_policy=hs.QueuePolicy.DROP_OLDEST, queue_depth=1
    )
    graph.partition("display").add(visualizer)
    graph.auto_partition_remaining()
    return graph


NO_WINDOW_ROUTE = "[holoviz] No window system route"


def display_unavailable(session) -> bool:
    """Recognize only the native, positively identified missing-window-route failure."""

    if not session.diagnostics:
        return False
    for diagnostic in session.diagnostics:
        if diagnostic.token != "RUNTIME_LIFECYCLE_CALLBACK_FAILED":
            return False
        detail = diagnostic.detail
        if (
            detail is None
            or detail.stage is not hs.lifecycle.LifecycleStage.START
            or detail.operator_path != "visualizer"
        ):
            return False
        error = detail.error
        if (
            error is None
            or error.code is not hs.errors.ErrorCode.NOT_FOUND
            or NO_WINDOW_ROUTE not in error.message
        ):
            return False
    return True


def final_result(session):
    """Inspect the authoritative session outcome, allowing only proven prerequisite skips."""
    if session.diagnostics:
        if display_unavailable(session):
            print("no display available; skipping")
            return SKIP_EXIT_CODE
        for diagnostic in session.diagnostics:
            detail = diagnostic.detail
            message = detail.error.message if detail is not None and detail.error else ""
            print(f"run failed: {diagnostic.token}: {message}", file=sys.stderr)
        return 1
    if session.termination is not hs.RunTermination.STOPPED:
        print(f"unexpected run termination: {session.termination}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    """Run for the requested session duration and report the final lifecycle result."""
    config = parse_args(argv)
    try:
        cupy = load_cupy()
        require_single_device(cupy)
    except UnavailableError as error:
        print(f"skipping: {error}")
        return SKIP_EXIT_CODE
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    plan = hs.compile(build_graph(config, cupy), compile_options())
    if not plan:
        print(plan.summary(), file=sys.stderr)
        return 1
    with hs.run_async(plan) as session:
        ended_early = session.wait(timeout=config.seconds)
        session.request_stop()
        if not session.wait(timeout=10.0):
            print(
                "shutdown timeout after 10s; cleanup may still block", file=sys.stderr, flush=True
            )
            return 1
    result = final_result(session)
    if result == 0 and ended_early:
        print("session ended before the requested duration", file=sys.stderr)
        return 1
    if result == 0:
        print(f"session ran for {config.seconds:g}s and stopped cleanly")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
