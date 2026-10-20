# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure Python payload conversion for byte vectors, numeric arrays and nested schemas."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import platform
import statistics
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

from vector_record import VectorRecord

import holoscan as hs
from holoscan.schema import Header, Imu, Time
from holoscan.schema.geometry import Matrix3f

# Load the native benchmark layouts before resolving their public payload tokens.
importlib.import_module("_pod_fixtures")

MODES = ("emit", "receive_data", "receive", "selected_data")
CASES = (
    "schema_bytes_small",
    "schema_bytes_camera",
    "schema_numeric_small",
    "schema_numeric_large",
    "schema_nested",
    "pod_bytes_small",
    "pod_bytes_large",
    "pod_uint8_small",
    "pod_uint8_large",
    "pod_numeric_small",
    "pod_numeric_large",
)


def payload_case(name, schema_byte_input="list"):
    """Return a public port type, its value, and the populated vector's storage size."""
    if name == "schema_nested":
        return (
            Imu,
            Imu(
                header=Header(stamp=Time(time_ns=123), frame_id="camera", device_sequence=7),
                orientation_covariance=Matrix3f(),
                angular_velocity_covariance=Matrix3f(),
                linear_acceleration_covariance=Matrix3f(),
            ),
            0,
        )
    byte_case = "bytes" in name or "uint8" in name
    count = (
        (
            (407 * 1024 if name.startswith("schema") else 16 * 1024)
            if name.endswith(("camera", "large"))
            else 8
        )
        if byte_case
        else (4096 if name.endswith("large") else 3)
    )
    data = (bytes(range(256)) * ((count + 255) // 256))[:count] if byte_case else [0.5] * count
    size = count if byte_case else count * 8
    if name.startswith("schema"):
        if byte_case and schema_byte_input == "list":
            data = list(data)
        return (
            VectorRecord,
            VectorRecord(sequence=7, **{"data" if byte_case else "readings": data}),
            size,
        )
    suffix = (
        ("LargeBytes" if name.endswith("large") else "SmallBytes")
        if byte_case
        else ("LargeNumeric" if name.endswith("large") else "SmallNumeric")
    )
    if "uint8" in name:
        suffix = "LargeUInt8" if name.endswith("large") else "SmallUInt8"
        data = tuple(data)
    return (
        hs.payload.resolve(f"holoscan.benchmark.{suffix}"),
        {
            "sequence": 7,
            "data": data if byte_case else tuple(data),
        },
        size,
    )


def run_case(
    name,
    mode,
    iterations,
    warmup,
    *,
    memory=False,
    schema_byte_input="list",
    require_byte_vectors=False,
):
    """Measure one API operation; source pacing, validation and dispatch are outside timing."""
    payload_type, payload, size = payload_case(name, schema_byte_input)
    total = iterations + warmup
    produced = consumed = 0
    measurements = []
    representation = None
    expected_bytes = bytes(payload.data) if name.startswith("schema_bytes") else None

    def measure(operation, index):
        if memory and index >= warmup:
            before, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
        begin = time.perf_counter_ns()
        value = operation()
        elapsed = time.perf_counter_ns() - begin
        if index >= warmup:
            if memory:
                _, peak = tracemalloc.get_traced_memory()
                measurements.append(max(0, peak - before))
            else:
                measurements.append(elapsed)
        return value

    class Source(hs.Operator):
        """Admit one record only after the previous record was consumed."""

        out = hs.Output("out", payload_type)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.us(10)))

        def compute(self, context):
            nonlocal produced
            if produced == total or produced != consumed:
                return
            if mode == "emit":
                measure(lambda: self.out.emit(payload), produced)
            else:
                self.out.emit(payload)
            produced += 1

    class Consumer(hs.Operator):
        """Validate every received value and its Python representation outside measurement."""

        inp = hs.Input("in", payload_type)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            nonlocal consumed, representation
            if mode == "receive":
                value = measure(lambda: self.inp.receive().data, consumed)
            elif mode == "selected_data":
                sample = self.inp.selected_sample()
                value = measure(lambda: sample.data, consumed)
            elif mode == "receive_data":
                value = measure(self.inp.receive_data, consumed)
            else:
                value = self.inp.receive_data()
            if name.startswith("schema_bytes"):
                representation = type(value.data).__name__
                assert value.sequence == payload.sequence
                assert value.readings == payload.readings
                assert bytes(value.data) == expected_bytes
                if require_byte_vectors:
                    assert isinstance(value.data, bytes)
            else:
                assert value == payload
                field = (
                    value["data"]
                    if name.startswith("pod")
                    else value.readings
                    if name.startswith("schema_numeric")
                    else value
                )
                representation = type(field).__name__
                if name.startswith("pod_bytes"):
                    assert isinstance(field, bytes)
                elif name.startswith("pod"):
                    assert isinstance(field, tuple)
            consumed += 1
            if consumed == total:
                context.request_completion()

    graph = hs.Graph("python-payload-conversion", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    consumer = graph.op(Consumer, "consumer")
    graph.add_flow(source.out, consumer["in"], queue_depth=1)
    graph.partition("serial").add(source).add(consumer)
    plan = hs.compile(graph)
    gc.collect()
    if memory:
        tracemalloc.start()
    try:
        session = hs.run_async(plan)
        if not session.wait(timeout=30):
            session.request_stop()
            session.wait(timeout=5)
            raise RuntimeError(f"{name}/{mode}: timed out")
        assert not session.diagnostics, session.diagnostics
        assert session.termination is hs.RunTermination.REQUESTED_COMPLETED
        assert produced == consumed == total
        assert len(measurements) == iterations
        # Detect accidental per-byte Python expansion without a machine-specific latency threshold.
        if memory and "bytes" in name and size > 8 and mode != "emit" and representation == "bytes":
            assert max(measurements) < size * 4
    finally:
        if memory:
            tracemalloc.stop()
    return measurements, representation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-iterations", type=int, default=16)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--schema-byte-input", choices=["list", "bytes"], default="list")
    parser.add_argument("--require-byte-vectors", action="store_true")
    parser.add_argument("--configuration", default="unspecified")
    parser.add_argument("--revision", default="unspecified")
    parser.add_argument("--benchmark_out")
    parser.add_argument("--benchmark_out_format", choices=["json"])
    args = parser.parse_args()
    if min(args.iterations, args.repeats, args.memory_iterations) <= 0 or args.warmup < 0:
        parser.error("iterations and repeats must be positive; warmup must be non-negative")
    rows = []
    for case in args.cases:
        timings = {mode: [] for mode in args.modes}
        for repeat in range(args.repeats):
            order = list(args.modes[repeat % len(args.modes) :]) + list(
                args.modes[: repeat % len(args.modes)]
            )
            for mode in order:
                timings[mode].append(
                    statistics.mean(
                        run_case(
                            case,
                            mode,
                            args.iterations,
                            args.warmup,
                            schema_byte_input=args.schema_byte_input,
                            require_byte_vectors=args.require_byte_vectors,
                        )[0]
                    )
                )
        for mode in args.modes:
            peaks, representation = run_case(
                case,
                mode,
                args.memory_iterations,
                args.warmup,
                memory=True,
                schema_byte_input=args.schema_byte_input,
                require_byte_vectors=args.require_byte_vectors,
            )
            rows.append(
                {
                    "name": f"{case}/{mode}",
                    "payload_bytes": payload_case(case)[2],
                    "median_ns_per_op": statistics.median(timings[mode]),
                    "repeat_means_ns": timings[mode],
                    "median_python_peak_bytes": statistics.median(peaks),
                    "received_representation": representation,
                }
            )
    result = {
        "context": {
            "benchmark": "python_payload_conversion",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "configuration": args.configuration,
            "revision": args.revision,
            "schema_byte_input": args.schema_byte_input,
            "require_byte_vectors": args.require_byte_vectors,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "memory_iterations": args.memory_iterations,
        },
        "benchmarks": rows,
    }
    print(f"{'case/operation':44} {'median us/op':>13} {'Python peak B':>14}")
    for row in rows:
        print(
            f"{row['name']:44} {row['median_ns_per_op'] / 1000:13.2f} "
            f"{row['median_python_peak_bytes']:14.0f}"
        )
    if args.benchmark_out:
        Path(args.benchmark_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
