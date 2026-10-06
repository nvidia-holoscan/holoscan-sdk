# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a HolovizOp graph fed by an external image topic, authored entirely in Python.

The graph is the smallest real one the Visualization bindings support: an imported
``holoscan.schema.Image`` topic drives a single image layer in a window.

A passing run tells you the operator lifecycle held: the graph compiled, the session ran for the
requested duration without terminating, and it stopped cleanly. A failed start would have ended
the session and recorded diagnostics instead. **It tells you nothing about rendering.**
Nothing publishes to ``holomq://camera`` here, so no frame ever reaches the layer, and the example
does not inspect presented pixels or window-event handling. A run that starts and stops cleanly
says the binding path is sound, not that anything was drawn correctly.

The trigger is left at its default, which is what a display application wants: activation follows
frame arrival on layer 0. Nothing here depends on that choice -- with no publisher the operator
starts and stops cleanly under either trigger.

Point a peer process at ``holomq://camera`` to actually see frames.

Exits 77 when no display is available so CTest can report the run as skipped.
"""

from __future__ import annotations

import argparse
import math

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


# A missing display surfaces as a window-creation failure during start. Matching the message keeps
# the skip narrow: a CUDA or renderer startup failure carries the same diagnostic token, and
# reporting one of those as "no display" would hide a real break behind a skipped test.
DISPLAY_FAILURE_MARKERS = ("create_window failed", "Failed to initialize GLFW")


def display_unavailable(session) -> bool:
    """True only when every failure is a start-time window-creation failure."""

    if not session.diagnostics:
        return False
    for diagnostic in session.diagnostics:
        if diagnostic.token != "RUNTIME_LIFECYCLE_CALLBACK_FAILED":
            return False
        detail = diagnostic.detail
        if detail is None or detail.stage is not hs.lifecycle.LifecycleStage.START:
            return False
        error = detail.error
        message = "" if error is None else error.message
        if not any(marker in message for marker in DISPLAY_FAILURE_MARKERS):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds", type=float, default=2.0, help="how long to run before stopping"
    )
    arguments = parser.parse_args()
    if not math.isfinite(arguments.seconds) or arguments.seconds <= 0.0:
        parser.error(f"invalid --seconds value: '{arguments.seconds:g}'")

    plan = hs.compile(build_graph())
    if not plan:
        for diagnostic in plan.diagnostics:
            print(f"compile rejected: {diagnostic.token}")
        return 1

    # Wait on the session rather than sleeping beside it. A HolovizOp whose start() fails
    # terminates the session and records diagnostics, so surviving the full duration is the
    # strongest startup evidence this API offers: Phase 1 exposes no per-operator lifecycle state
    # to Python, and `is_running` is already true while start() is still in progress.
    with hs.run_async(plan) as session:
        ended_early = session.wait(timeout=arguments.seconds)
        session.request_stop()
        session.wait()

    if session.diagnostics:
        if display_unavailable(session):
            print("no display available; skipping")
            return SKIP_EXIT_CODE
        for diagnostic in session.diagnostics:
            print(f"run failed: {diagnostic.token}")
        return 1

    # Without this the smoke is vacuous: a session that ended on its own records no diagnostics
    # either, so absence of failure alone would read as success.
    if ended_early:
        print(f"session ended before the requested {arguments.seconds:g}s")
        return 1

    print(f"started, ran for {arguments.seconds:g}s and stopped cleanly; no frames published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
