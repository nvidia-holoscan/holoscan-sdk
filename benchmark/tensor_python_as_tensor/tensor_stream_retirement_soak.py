#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe queued-work retirement without retaining per-frame benchmark metadata."""

import argparse
import gc
import json
import os
import threading
import time
import weakref
from pathlib import Path

import numpy as np

try:
    import cupy as cp
except ImportError as exc:
    print(exc)
    raise SystemExit(77) from exc

import holoscan as hs
from holoscan._core import _test_drain_tensor_releases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--policy", choices=("shared", "per_partition"), default="shared")
    parser.add_argument("--tensor-bytes", type=int, default=4096)
    parser.add_argument("--gpu-work-us", type=int, default=3000)
    parser.add_argument("--sample-seconds", type=float, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.gpu_work_us <= ((1 << 64) - 1) // 1000:
        parser.error("GPU work must be a nonnegative representable nanosecond duration")
    if args.seconds <= 0 or args.sample_seconds <= 0 or args.tensor_bytes <= 0:
        parser.error("duration, sampling interval and Tensor size must be positive")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            return 77
    except RuntimeError:
        return 77
    spin = cp.RawKernel(
        r"""
    extern "C" __global__ void spin(unsigned long long duration) {
      unsigned long long start, now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
      do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); }
      while (now - start < duration);
    }
    """,
        "spin",
    )
    spin((1,), (1,), (np.uint64(1),))
    cp.cuda.runtime.deviceSynchronize()
    owners = {}
    counts = {"emitted": 0, "received": 0, "released": 0, "retries": 0, "forward_dropped": 0}
    stop_source = threading.Event()
    samples = []
    source_instances = []

    def sample(label):
        pool = cp.get_default_memory_pool()
        rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        row = dict(
            label=label,
            elapsed_seconds=time.monotonic() - started,
            rss_bytes=rss,
            live_owners=len(owners),
            pool_total_bytes=pool.total_bytes(),
            pool_used_bytes=pool.used_bytes(),
            **counts,
        )
        samples.append(row)
        args.output.write_text(
            json.dumps(
                {"configuration": vars(args) | {"output": str(args.output)}, "samples": samples},
                indent=2,
            )
        )
        print(json.dumps(row), flush=True)

    def output(spec, port):
        spec.output(port).produces_tensor(
            hs.tensor.OutputSpec(
                storage="pass_through",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                bounds=hs.tensor.TensorBounds(max_rank=1, max_byte_span=args.tensor_bytes),
            )
        )

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            output(spec, self.out)
            self.pending = None
            self.next_frame = 0
            source_instances.append(self)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.us(100)))

        def compute(self, ctx):
            if stop_source.is_set():
                return
            with ctx.stream():
                if self.pending is None:
                    self.pending = cp.empty(args.tensor_bytes, dtype=cp.uint8)
                    frame = self.next_frame
                    self.next_frame += 1

                    def released(_, frame=frame):
                        owners.pop(frame, None)
                        counts["released"] += 1

                    owners[frame] = weakref.ref(self.pending, released)
                    spin((1,), (1,), (np.uint64(args.gpu_work_us * 1000),))
                try:
                    self.out.emit(self.pending)
                except hs.errors.BackpressuredError:
                    counts["retries"] += 1
                else:
                    self.pending = None
                    counts["emitted"] += 1

    class Forward(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor, as_=cp)
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.input(self.inp)
            output(spec, self.out)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with ctx.stream():
                array = self.inp.receive_data()
                try:
                    self.out.emit(array)
                except hs.errors.BackpressuredError:
                    # Discard this pre-submit frame rather than adding a Python queue.
                    # Its producer GPU work is already queued and still needs retirement.
                    counts["forward_dropped"] += 1

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor, as_=cp)

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with ctx.stream():
                assert self.inp.receive_data().nbytes == args.tensor_bytes
                counts["received"] += 1

    graph = hs.Graph(f"retirement-soak-{os.getpid()}", default_clock=hs.RealtimeClock)
    source, forward, sink = [
        graph.op(kind, name)
        for kind, name in ((Source, "source"), (Forward, "forward"), (Sink, "sink"))
    ]
    graph.add_flow(source.out, forward["in"], queue_depth=2)
    graph.add_flow(forward.out, sink["in"], queue_depth=2)
    for node, name in ((source, "source"), (forward, "forward"), (sink, "sink")):
        graph.partition(name).add(node)
    deployment = hs.execution.DeploymentBinding()
    for name in ("source", "forward"):
        deployment.bind_tensor_output_device(
            operator_path=name, output_port="out", device=hs.execution.DeviceId(0)
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
    if not plan.ok:
        raise RuntimeError(plan.summary())
    started = time.monotonic()
    with hs.run_async(plan) as session:
        sample("start")
        deadline = started + args.seconds
        while time.monotonic() < deadline:
            time.sleep(min(args.sample_seconds, max(0, deadline - time.monotonic())))
            sample("running")
            if session.wait(timeout=0):
                raise RuntimeError(f"Soak graph terminated early: {session.diagnostics}")
        stop_source.set()
        sample("stop_requested")
        session.request_stop()
        if not session.wait(timeout=max(60, args.seconds)):
            raise TimeoutError("GPU quiescence exceeded the soak duration")
        sample("quiesced")
    if session.diagnostics:
        raise RuntimeError(session.diagnostics)
    del session
    # A source retry can hold an array which was never emitted when shutdown starts.
    for instance in source_instances:
        instance.pending = None
    source_instances.clear()
    gc.collect()
    _test_drain_tensor_releases(timeout=30)
    sample("drained")
    if owners:
        raise RuntimeError("CuPy source owners remain after final drain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
