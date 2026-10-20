#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assert SDK callbacks add no imported-Tensor snapshot events on SDK streams."""

import argparse
import ctypes
import os
import time

try:
    import cupy as cp
except ImportError:
    raise SystemExit(77) from None

import holoscan as hs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("shared", "per_partition"), required=True)
    parser.add_argument("--cudart", required=True)
    args = parser.parse_args()
    try:
        if not cp.cuda.runtime.getDeviceCount():
            return 77
    except RuntimeError:
        return 77
    cp.cuda.runtime.free(0)
    runtime = ctypes.CDLL(args.cudart, mode=ctypes.RTLD_GLOBAL)
    shim = ctypes.CDLL(None)
    records = shim.hsdk_test_thread_event_records
    records.restype = ctypes.c_uint64
    create = runtime.cudaEventCreateWithFlags
    create.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint)
    record = shim.cudaEventRecord
    record.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    event = ctypes.c_void_p()
    before = records()
    assert create(ctypes.byref(event), 2) == 0
    assert record(event, None) == 0
    assert records() == before + 1, "CUDA call counter did not observe its positive control"
    assert runtime.cudaEventSynchronize(event) == 0
    assert runtime.cudaEventDestroy(event) == 0
    previous = {}
    received = []

    def checkpoint(name, expected_new=0):
        count = records()
        if name in previous:
            assert count - previous[name] == expected_new, (
                f"{name}: unexpected CUDA event records: "
                f"{count - previous[name]} (expected {expected_new})"
            )
        previous[name] = count

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="pass_through",
                    memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                    bounds=hs.tensor.TensorBounds(max_rank=1, max_byte_span=64),
                )
            )
            self.sent = 0

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            checkpoint("source")
            if self.sent >= 12:
                return
            with ctx.stream():
                array = cp.empty(64, dtype=cp.uint8)
                self.out.emit(array)
            self.sent += 1
            # Imports record one producer-readiness event; callback exit must add none.
            checkpoint("source", expected_new=1)

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor)

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            checkpoint("sink")
            with ctx.stream():
                tensor = self.inp.receive_data()
                view = cp.from_dlpack(tensor)
                assert view.nbytes == 64
                received.append(len(received))
            checkpoint("sink")

    graph = hs.Graph(f"event-count-{os.getpid()}", default_clock=hs.RealtimeClock)
    source, sink = graph.op(Source, "source"), graph.op(Sink, "sink")
    graph.add_flow(source.out, sink["in"], queue_depth=2)
    graph.partition("source").add(source)
    graph.partition("sink").add(sink)
    deployment = hs.execution.DeploymentBinding()
    deployment.bind_tensor_output_device(
        operator_path="source", output_port="out", device=hs.execution.DeviceId(0)
    )
    policy = (
        hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
        if args.policy == "shared"
        else hs.execution.CudaComputeStreamPolicy.PER_EXECUTION_PARTITION
    )
    plan = hs.compile(
        graph,
        hs.CompileOptions(
            deployment=deployment,
            cuda_execution=hs.execution.CudaExecutionCompileOptions(compute_stream_policy=policy),
        ),
    )
    assert plan.ok, plan.summary()
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 5
        while len(received) < 12 and time.monotonic() < deadline:
            time.sleep(0.001)
        session.request_stop()
        assert session.wait(timeout=5)
    assert not session.diagnostics, session.diagnostics
    assert len(received) == 12
    print(f"{args.policy}: 12 frames; zero SDK callback snapshot events; positive control passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
