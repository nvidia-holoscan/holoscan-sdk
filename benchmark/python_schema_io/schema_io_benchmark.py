# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare owning and selected Python schema reads and synchronous forwarding."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import holoscan as hs
from holoscan.schema import Header, Imu, Time
from holoscan.schema.geometry import Matrix3f

MODES = (
    "receive_metadata",
    "selected_metadata",
    "receive_data",
    "selected_data",
    "receive_forward",
    "selected_forward",
)


def operation(mode, inp, out):
    """Keep owning and borrowed operation boundaries identical across measurement passes."""
    if mode == "receive_metadata":
        return inp.receive().metadata
    if mode == "selected_metadata":
        return inp.selected_sample().metadata
    if mode == "receive_data":
        return inp.receive_data()
    if mode == "selected_data":
        return inp.selected_sample().data
    if mode == "receive_forward":
        out.forward(inp.receive())
    else:
        out.forward(inp.selected_sample())
    return None


def run_case(mode, frame_bytes, iterations, warmup, *, memory=False):
    """Time only the consumer/relay API operation, excluding graph setup and validation."""
    payload = Imu(
        header=Header(frame_id="f" * frame_bytes, stamp=Time(time_ns=123), device_sequence=7),
        orientation_covariance=Matrix3f(),
        angular_velocity_covariance=Matrix3f(),
        linear_acceleration_covariance=Matrix3f(),
    )
    forwarding = mode.endswith("forward")
    total = iterations + warmup
    samples = []
    peaks = []
    consumed = 0
    delivered = 0
    produced = 0

    class Source(hs.Operator):
        """Publish the same payload outside the measured callback operation."""

        out = hs.Output("out", Imu)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.us(10)))

        def compute(self, context):
            nonlocal produced
            retired = delivered if forwarding else consumed
            if produced == total or produced != retired:
                return
            try:
                self.out.emit(payload, options=hs.EmitOptions(frame_id=91))
            except hs.errors.BackpressuredError:
                # Retry only unaccepted source publications; relay failure remains fatal.
                return
            produced += 1

    class Consumer(hs.Operator):
        """Read or relay every admitted sample with one selected API mode."""

        inp = hs.Input("in", Imu)
        out = hs.Output("out", Imu)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            nonlocal consumed
            measured = consumed >= warmup
            if memory and measured:
                before, _ = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
            begin = time.perf_counter_ns()
            value = operation(mode, self.inp, self.out)
            elapsed = time.perf_counter_ns() - begin
            if memory and measured:
                _, peak = tracemalloc.get_traced_memory()
                peaks.append(max(0, peak - before))
            elif measured:
                samples.append(elapsed)
            # All projections and Runtime metadata are checked outside the measured operation.
            if mode.endswith("metadata"):
                assert value.frame_id == 91
                assert self.inp.selected_sample().data == payload
            elif not forwarding:
                assert value == payload
            consumed += 1
            if consumed == total and not forwarding:
                context.request_completion()

    class Sink(hs.Operator):
        """Validate every forwarded payload and inherited metadata outside relay timing."""

        inp = hs.Input("in", Imu)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            nonlocal delivered
            sample = self.inp.receive()
            assert sample.data == payload
            assert sample.metadata.frame_id == 91
            delivered += 1
            if delivered == total:
                context.request_completion()

    graph = hs.Graph("python-schema-io", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    consumer = graph.op(Consumer, "consumer")
    graph.add_flow(source.out, consumer["in"], queue_depth=1)
    partition = graph.partition("serial").add(source).add(consumer)
    if forwarding:
        sink = graph.op(Sink, "sink")
        graph.add_flow(consumer.out, sink["in"], queue_depth=1)
        partition.add(sink)
    plan = hs.compile(graph)
    gc.collect()
    if memory:
        tracemalloc.start()
    try:
        session = hs.run_async(plan)
        if not session.wait(timeout=30.0):
            session.request_stop()
            session.wait(timeout=5.0)
            raise RuntimeError(f"{mode}: benchmark timed out")
        if session.diagnostics:
            raise RuntimeError(f"{mode}: {session.diagnostics}")
        assert session.termination is hs.RunTermination.REQUESTED_COMPLETED
        assert produced == consumed == total
        assert not forwarding or delivered == total
        assert len(peaks if memory else samples) == iterations
    finally:
        if memory:
            tracemalloc.stop()
    return peaks if memory else samples


def benchmark(args):
    """Use separate untraced timing and traced-memory passes for each matched pair."""
    results = []
    for frame_bytes in args.frame_bytes:
        # Rotate mode order across repeats to reduce a fixed-order cache/thermal preference.
        timings = {mode: [] for mode in MODES}
        for repeat in range(args.repeats):
            order = MODES[repeat % len(MODES) :] + MODES[: repeat % len(MODES)]
            for mode in order:
                samples = run_case(mode, frame_bytes, args.iterations, args.warmup)
                timings[mode].append(statistics.mean(samples))
        for mode in MODES:
            peaks = run_case(mode, frame_bytes, args.memory_iterations, args.warmup, memory=True)
            results.append(
                {
                    "name": mode,
                    "frame_bytes": frame_bytes,
                    "iterations": args.iterations,
                    "repeats": args.repeats,
                    "median_ns_per_op": statistics.median(timings[mode]),
                    "min_ns_per_op": min(timings[mode]),
                    "max_ns_per_op": max(timings[mode]),
                    "repeat_means_ns": timings[mode],
                    "memory_iterations": args.memory_iterations,
                    "median_python_peak_bytes": statistics.median(peaks),
                    "max_python_peak_bytes": max(peaks),
                }
            )
    return {
        "context": {
            "benchmark": "python_schema_io",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "configuration": args.configuration,
            "schema": "holoscan.schema.Imu",
            "partition_count": 1,
            "queue_depth": 1,
            "warmup": args.warmup,
            "timing_scope": "callback API operation; excludes setup, source, sink and validation",
            "memory_scope": "tracemalloc peak above pre-operation baseline; excludes C++ new",
        },
        "benchmarks": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--memory-iterations", type=int, default=100)
    parser.add_argument("--frame-bytes", type=int, nargs="+", default=[8, 128])
    parser.add_argument("--configuration", default="unspecified")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--benchmark_out")
    parser.add_argument("--benchmark_out_format", choices=["json"])
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.memory_iterations) <= 0 or args.warmup < 0:
        parser.error("iterations and repeats must be positive; warmup must be non-negative")
    if any(size < 0 or size > 128 for size in args.frame_bytes):
        parser.error("frame bytes must be in [0, 128] to fit the populated Imu wire budget")
    result = benchmark(args)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{'operation':22} {'frame B':>8} {'median ns/op':>14} {'Python peak B':>14}")
        for row in result["benchmarks"]:
            print(
                f"{row['name']:22} {row['frame_bytes']:8} "
                f"{row['median_ns_per_op']:14.1f} {row['median_python_peak_bytes']:14.0f}"
            )
    if args.benchmark_out:
        Path(args.benchmark_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
