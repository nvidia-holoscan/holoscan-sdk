# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author and run the camera/IMU/control graph from Python."""

import ctypes
from argparse import ArgumentParser
from contextlib import suppress
from pathlib import Path

import holoscan as hs

IMU_ENDPOINT = "baseline.camera_imu.imu"
CONTROL_ENDPOINT = "baseline.camera_imu.control"


class CameraSource(hs.Operator):
    """Publish a fixed host byte frame with activation time as capture metadata."""

    frame = hs.Output("frame", hs.payload.Tensor)

    def setup(self, spec):
        spec.output(self.frame).produces_tensor(
            hs.tensor.OutputSpec(
                storage=hs.tensor.OutputStorage.RUNTIME_POOL,
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=1,
                bounds=hs.tensor.TensorBounds(max_byte_span=64),
            )
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(20)))

    def start(self):
        self.sequence = 0

    def compute(self, ctx):
        with self.frame.allocate_tensor(
            shape=(64,),
            dtype=hs.tensor.DLDataType(1, 8, 1),
            options=hs.EmitOptions(capture_time=ctx.activation_time),
        ) as loan:
            storage = (ctypes.c_uint8 * loan.tensor.nbytes).from_address(loan.tensor.data)
            storage[:] = [self.sequence] * loan.tensor.nbytes
        self.sequence = (self.sequence + 1) % 256


class WindowFusion(hs.Operator):
    """Fuse each camera frame with its bounded IMU event-time selection."""

    camera = hs.Input("camera", hs.payload.Tensor)
    imu = hs.Input("imu", hs.payload.uint32)
    control = hs.Output("control", hs.payload.uint32)

    def setup(self, spec):
        spec.input(self.camera, queue_depth=4).expects_tensor(
            hs.tensor.InputSpec(
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=1,
            )
        )
        spec.input(self.imu, queue_depth=32)
        spec.output(self.control)

    def contract(self):
        return (
            hs.Contract()
            .trigger(
                hs.OnWindow(anchor=self.camera, max_wait=hs.time.ms(20), max_pending_windows=2)
            )
            .input(self.camera, hs.Exact())
            .input(
                self.imu,
                hs.BracketedWindow(lookbehind=1, lookahead=1, max_samples=16),
            )
            .on_missing(self.imu, hs.SafeDefault())
        )

    def compute(self, context):
        frame = self.camera.receive_data()
        integrated = 0
        if self.imu.selection().state is hs.InputState.SELECTED:
            integrated = sum(sample.data for sample in self.imu.selected_batch())
        with suppress(hs.errors.NotReadyError):
            self.control.emit(frame.nbytes + integrated)


class FusionGraph(hs.Graph):
    """Embed camera production, IMU import, and windowed fusion."""

    control = hs.Output("control", hs.payload.uint32)

    def compose(self) -> None:
        clock = self.add_clock(hs.RealtimeClock, "capture-clock")
        self.set_default_clock(clock)
        imu = self.import_topic(
            f"holomq://{IMU_ENDPOINT}",
            hs.payload.uint32,
            endpoint_name=IMU_ENDPOINT,
            capture_time=clock,
        )
        camera = self.op(CameraSource, "camera")
        fusion = self.op(WindowFusion, "window-fusion")
        self.add_flow(camera.frame, fusion.camera, queue_depth=4)
        self.add_flow(imu.output, fusion.imu, queue_depth=32)
        self.export_output(self.control, fusion.control)


def build_graph() -> hs.Graph:
    """Build the application root and its stable external control topic."""
    graph = hs.Graph("holoscan-camera-imu-control")
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "runtime-clock"))
    control = graph.export_topic(
        f"holomq://{CONTROL_ENDPOINT}",
        hs.payload.uint32,
        endpoint_name=CONTROL_ENDPOINT,
    )
    fusion = graph.embed(FusionGraph, "fusion")
    graph.add_flow(fusion.control, control.input, queue_depth=8)
    return graph


def build_plan() -> hs.Plan:
    """Compile the example graph and return its endpoint-bearing plan."""
    return hs.compile(build_graph())


def parse_args() -> Path:
    """Return the requested endpoint-catalog output path."""
    parser = ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("endpoint-catalog.json"),
        help="path for the canonical endpoint catalog",
    )
    return parser.parse_args().catalog


def main() -> None:
    """Save the catalog and exercise the camera-only safe-default path."""
    plan = build_plan()
    if not plan:
        raise SystemExit(plan.summary())

    catalog_path = parse_args()
    plan.endpoint_catalog.save_json(catalog_path)
    with hs.run_async(plan) as session:
        if session.wait(timeout=0.25):
            raise RuntimeError("camera/IMU graph ended before the caller requested stop")
        session.request_stop()
        if not session.wait(timeout=5.0):
            raise TimeoutError("camera/IMU graph did not stop within five seconds")
    if session.termination is not hs.RunTermination.STOPPED or session.diagnostics:
        raise RuntimeError(f"unexpected run result: {session!r}")

    print(f"catalog={catalog_path}")
    print("camera/IMU Python graph complete (camera-only safe-default run)")


if __name__ == "__main__":
    main()
