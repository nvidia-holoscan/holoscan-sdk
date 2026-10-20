# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a HolovizOp graph fed by an external image topic, authored entirely in Python.

The graph is the smallest real one the Visualization bindings support: an imported
``holoscan.schema.Image`` topic drives a single image layer in a window.

A passing run says the graph compiled, the session wait lasted for the requested
interval, and the session stopped without diagnostics. Session waits observe
termination; they do not confirm that all START callbacks completed before the
interval. Strict startup qualification belongs to the non-installed test driver.

Nothing publishes to ``holomq://camera`` here, so no frame reaches the layer.
This example does not validate rendering or window-event handling.

The trigger is left at its default, which is what a display application wants: activation follows
frame arrival on layer 0. Nothing here depends on that choice -- with no publisher the operator
starts and stops cleanly under either trigger.

Point a peer process at ``holomq://camera`` to actually see frames.

Exits 77 when retained diagnostics prove there is no display route, so CTest can
report that failure as skipped.
"""

from __future__ import annotations

import argparse
import math
import sys

from holoscan_visualization.operators import HolovizOp

import holoscan as hs

SKIP_EXIT_CODE = 77
FRAME_HEIGHT = 480
FRAME_WIDTH = 640


def image_payload():
    """Resolve the Image payload identity.

    Deliberately not resolved at import: the registry row is installed by
    ``libholoscan_sensor_io``, which reaches this process through the Visualization operator
    library. Resolving here keeps the example independent of module import order.
    """

    return hs.payload.resolve("holoscan.schema.Image")


def frame_contract() -> hs.tensor.TensorBoundaryContract:
    """Bound one host RGBA frame; HolovizOp accepts host- or device-resident image data."""

    return hs.tensor.TensorBoundaryContract(
        representation=hs.tensor.TensorRepresentation(
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype=hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLUINT, 8, 1),
            rank=3,
            strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
        ),
        bounds=hs.tensor.TensorBounds(max_shape=(FRAME_HEIGHT, FRAME_WIDTH, 4)),
    )


def build_graph() -> hs.Graph:
    """Wire the imported image topic into a single-image-layer window."""

    graph = hs.Graph("holoviz-image-topic")
    clock = graph.add_clock(hs.RealtimeClock, "clock")
    graph.set_default_clock(clock)

    camera = graph.import_topic(
        "holomq://camera",
        image_payload(),
        endpoint_name="camera",
        capture_time=clock,
        tensor_fields={"data": frame_contract()},
    )

    visualizer = HolovizOp(
        graph,
        "visualizer",
        layers=[
            {
                "kind": "image",
                "port_name": "video",
                "format": "rgba8_unorm",
                "max_width": FRAME_WIDTH,
                "max_height": FRAME_HEIGHT,
            }
        ],
        sink={
            "kind": "window",
            "title": "Holoviz image topic",
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
        },
    )

    # Depth-1 drop-oldest, so a stalled display drops frames instead of backpressuring a
    # publisher. Nothing publishes here, so this is the shape a real graph wants, not
    # behaviour the example exercises.
    graph.add_flow(
        camera.output,
        visualizer.video,
        queue_policy=hs.QueuePolicy.DROP_OLDEST,
        queue_depth=1,
    )
    graph.partition("display").add(visualizer)
    graph.auto_partition_remaining()
    return graph


def display_unavailable(session) -> bool:
    """Recognize only visualizer START failures with an explicit absent display route."""

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
            or "[holoviz] No window system route" not in error.message
        ):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=2.0, help="session wait interval")
    arguments = parser.parse_args()
    if not math.isfinite(arguments.seconds) or arguments.seconds <= 0.0:
        parser.error(f"invalid --seconds value: '{arguments.seconds:g}'")

    plan = hs.compile(build_graph())
    if not plan:
        for diagnostic in plan.diagnostics:
            print(f"compile rejected: {diagnostic.token}", file=sys.stderr, flush=True)
        return 1

    with hs.run_async(plan) as session:
        # A timed session wait observes termination, not completion of START.
        ended_early = session.wait(timeout=arguments.seconds)
        if ended_early:
            print("session ended before the requested interval", file=sys.stderr, flush=True)
        session.request_stop()
        if not session.wait(timeout=10.0):
            print(
                "shutdown timeout after 10s; cleanup may still block", file=sys.stderr, flush=True
            )
            return 1
    if session.diagnostics:
        if display_unavailable(session):
            print("no display available; skipping")
            return SKIP_EXIT_CODE
        for diagnostic in session.diagnostics:
            print(f"run failed: {diagnostic.token}", file=sys.stderr, flush=True)
        return 1
    if session.termination is not hs.RunTermination.STOPPED:
        print(f"unexpected run termination: {session.termination}", file=sys.stderr, flush=True)
        return 1
    if ended_early:
        return 1
    print(
        f"session wait completed for {arguments.seconds:g}s and stopped cleanly; "
        "no frames published"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
