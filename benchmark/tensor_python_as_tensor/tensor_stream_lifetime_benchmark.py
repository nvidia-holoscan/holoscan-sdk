#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure CuPy handoff and complete retirement, including worker CPU consumption."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import threading
import time
import weakref
from collections import deque
from pathlib import Path

import numpy as np

try:
    import cupy as cp
except ImportError as exc:
    print(f"CuPy benchmark unavailable: {exc}")
    raise SystemExit(77) from exc

import holoscan as hs

# In-tree benchmark seam: wait for actual owner release, not just queued retirement.
from holoscan._core import _test_drain_tensor_releases


def thread_cpu():
    """Read cumulative Linux task CPU time without sampling/profiling the measured path."""
    result = {}
    for task in Path("/proc/self/task").iterdir():
        try:
            fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
            name = (task / "comm").read_text().strip()
            result[task.name] = (
                name,
                (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
            )
        except FileNotFoundError:
            continue
    return result


def cpu_delta(before, after):
    return {
        f"{name}:{tid}": seconds - before.get(tid, (name, 0))[1]
        for tid, (name, seconds) in after.items()
        if "retire" in name or seconds - before.get(tid, (name, 0))[1] > 0
    }


def distribution(values):
    return {f"p{p}_ns": float(np.percentile(values, p)) for p in (50, 95, 99)}


def snapshot():
    """Capture phase CPU before reading task counters; the scan itself has measurable overhead."""
    return {
        "wall_ns": time.perf_counter_ns(),
        "process_cpu_ns": time.process_time_ns(),
        "threads": thread_cpu(),
    }


def phase_deltas(phases):
    ordered = sorted(phases.items(), key=lambda item: item[1]["wall_ns"])
    return [
        {
            "from": before_name,
            "to": after_name,
            "wall_ns": after["wall_ns"] - before["wall_ns"],
            "process_cpu_ns": after["process_cpu_ns"] - before["process_cpu_ns"],
            "thread_cpu_seconds": cpu_delta(before["threads"], after["threads"]),
        }
        for (before_name, before), (after_name, after) in zip(ordered, ordered[1:], strict=False)
    ]


def wait_for_delivery(done, session, timeout_seconds):
    """Stop waiting when the graph terminates instead of hiding its failure behind a timeout."""
    deadline = time.monotonic() + timeout_seconds
    while not done.wait(min(0.1, max(0.0, deadline - time.monotonic()))):
        if not session.is_running:
            raise RuntimeError(f"Graph stopped before delivery completed: {session.diagnostics!r}")
        if time.monotonic() >= deadline:
            return False
    return True


def drain(owners):
    cp.cuda.runtime.deviceSynchronize()
    gc.collect()
    _test_drain_tensor_releases()
    if any(owner() is not None for owner in owners):
        raise RuntimeError("Retirement did not release every measured CuPy owner")


SPIN = cp.RawKernel(
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


def pipeline(args, policy):
    total = args.frames + args.warmup
    done = threading.Event()
    owners = []
    entered = [[], [], []]
    durations = [[], [], []]
    cpu = [[], [], []]
    retries = 0
    first_emit_ns = [0] * total
    emitted_ns = [0] * total
    released_ns = [0] * total
    producer_observed_ns = [0] * total

    def owner_released(frame):
        # Capture the frame index and timestamp storage, never the owner or its operator.
        def record(_):
            released_ns[frame] = time.perf_counter_ns()

        return record

    measurement_start = 0
    phases = {}
    role_threads = [None, None, None]
    pool_samples = []
    stages = {
        name: []
        for name in ("allocate", "emit_source", "receive_forward", "emit_forward", "receive_sink")
    }

    def sample_pool(label):
        pool = cp.get_default_memory_pool()
        pool_samples.append(
            {
                "label": label,
                "wall_ns": time.perf_counter_ns(),
                "source_sent": len(entered[0]),
                "sink_received": len(entered[2]),
                "live_source_owners": sum(owner() is not None for owner in owners),
                "pool_total_bytes": pool.total_bytes(),
                "pool_used_bytes": pool.used_bytes(),
                "pool_free_blocks": pool.n_free_blocks(),
            }
        )

    def output(spec, port):
        spec.output(port).produces_tensor(
            hs.tensor.OutputSpec(
                storage="pass_through",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype="float32",
                rank=1,
                bounds=hs.tensor.TensorBounds(max_rank=1, max_byte_span=args.tensor_bytes),
            )
        )

    def finish(role, wall, thread):
        durations[role].append(time.perf_counter_ns() - wall)
        cpu[role].append(time.thread_time_ns() - thread)
        entered[role].append(wall)
        if args.diagnostics and role_threads[role] is None:
            role_threads[role] = threading.get_native_id()

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            output(spec, self.out)
            self.pending = None
            self.sent = 0
            self.inflight = deque()

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.us(args.period_us)))

        def compute(self, ctx):
            nonlocal retries, measurement_start
            if self.sent == total:
                return
            if self.sent == args.warmup and not measurement_start and args.gpu_work_us:
                if args.diagnostics:
                    phases["warmup_fence_start"] = snapshot()
                cp.cuda.runtime.deviceSynchronize()
            if self.sent == args.warmup and not measurement_start and args.diagnostics:
                phases["measurement_start"] = snapshot()
                sample_pool("measurement_start")
            wall, thread = time.perf_counter_ns(), time.thread_time_ns()
            if self.sent == args.warmup and not measurement_start:
                measurement_start = wall
            with ctx.stream():
                if self.pending is None:
                    if args.pace_window and len(self.inflight) >= args.pace_window:
                        frame, completed = self.inflight.popleft()
                        completed.synchronize()
                        if args.release_timings:
                            producer_observed_ns[frame] = time.perf_counter_ns()
                    stage = time.perf_counter_ns() if args.stage_timings else 0
                    self.pending = cp.full(args.tensor_bytes // 4, self.sent, dtype=cp.float32)
                    if args.stage_timings and self.sent >= args.warmup:
                        stages["allocate"].append(time.perf_counter_ns() - stage)
                    owners.append(
                        weakref.ref(
                            self.pending,
                            owner_released(self.sent) if args.release_timings else None,
                        )
                    )
                    if args.gpu_work_us:
                        SPIN((1,), (1,), (np.uint64(args.gpu_work_us * 1000),))
                    if args.pace_window:
                        event = cp.cuda.Event(block=True, disable_timing=True)
                        event.record()
                        self.inflight.append((self.sent, event))
                try:
                    stage = time.perf_counter_ns() if args.stage_timings else 0
                    if args.release_timings and not first_emit_ns[self.sent]:
                        first_emit_ns[self.sent] = time.perf_counter_ns()
                    self.out.emit(self.pending)
                    if args.release_timings:
                        emitted_ns[self.sent] = time.perf_counter_ns()
                    if args.stage_timings and self.sent >= args.warmup:
                        stages["emit_source"].append(time.perf_counter_ns() - stage)
                except hs.errors.BackpressuredError:
                    retries += 1
                    return
                self.pending = None
            self.sent += 1
            finish(0, wall, thread)
            if args.diagnostics and self.sent % args.sample_every == 0:
                sample_pool("progress")

    class Forward(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor, as_=cp)
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.input(self.inp)
            output(spec, self.out)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            wall, thread = time.perf_counter_ns(), time.thread_time_ns()
            with ctx.stream():
                if args.stage_timings:
                    stage = time.perf_counter_ns()
                    array = self.inp.receive_data()
                    received = time.perf_counter_ns()
                    self.out.emit(array)
                    del array  # preserve the normal expression's immediate view release
                    emitted = time.perf_counter_ns()
                    if len(entered[1]) >= args.warmup:
                        stages["receive_forward"].append(received - stage)
                        stages["emit_forward"].append(emitted - received)
                else:
                    self.out.emit(self.inp.receive_data())
            finish(1, wall, thread)

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor, as_=cp)

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            wall, thread = time.perf_counter_ns(), time.thread_time_ns()
            with ctx.stream():
                stage = time.perf_counter_ns() if args.stage_timings else 0
                array = self.inp.receive_data()
                if args.stage_timings and len(entered[2]) >= args.warmup:
                    stages["receive_sink"].append(time.perf_counter_ns() - stage)
                assert array.nbytes == args.tensor_bytes
            finish(2, wall, thread)
            if len(entered[2]) == total:
                done.set()

    graph = hs.Graph("cupy-retirement-benchmark", default_clock=hs.RealtimeClock)
    nodes = [
        graph.op(kind, name)
        for kind, name in ((Source, "source"), (Forward, "forward"), (Sink, "sink"))
    ]
    graph.add_flow(nodes[0].out, nodes[1]["in"], queue_depth=args.queue_depth)
    graph.add_flow(nodes[1].out, nodes[2]["in"], queue_depth=args.queue_depth)
    for node, name in zip(nodes, ("source", "forward", "sink"), strict=True):
        graph.partition(name).add(node)
    deployment = hs.execution.DeploymentBinding()
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) >= 5:
        for name, core in zip(("source", "forward", "sink"), cpus[1:4], strict=True):
            deployment.bind_thread_placement(
                name, hs.execution.ThreadPlacement(affinity=hs.execution.CPUAffinity([core]))
            )
        deployment.bind_support_thread_affinity(
            role=hs.execution.SupportThreadRole.CONTROLLER,
            affinity=hs.execution.CPUAffinity([cpus[0]]),
        )
        deployment.bind_support_thread_affinity(
            role=hs.execution.SupportThreadRole.CHANNEL_PUMP,
            affinity=hs.execution.CPUAffinity([cpus[4]]),
        )
    for name in ("source", "forward"):
        deployment.bind_tensor_output_device(
            operator_path=name, output_port="out", device=hs.execution.DeviceId(0)
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
    before_threads, before_cpu = thread_cpu(), time.process_time_ns()
    if args.diagnostics:
        phases["before_session"] = snapshot()
    with hs.run_async(plan) as session:
        if args.diagnostics:
            phases["session_entered"] = snapshot()
        if not wait_for_delivery(done, session, args.delivery_timeout_seconds):
            session.request_stop()
            raise TimeoutError(f"Received {len(entered[2])}/{total} frames")
        delivery_end = time.perf_counter_ns()
        if args.diagnostics:
            # The source's clock contract keeps lanes alive until request_stop().
            phases["delivery_end"] = snapshot()
            sample_pool("delivery_end")
        session.request_stop()
        if not session.wait(timeout=args.retirement_timeout_seconds):
            raise TimeoutError("Session retirement did not finish")
        if args.diagnostics:
            phases["session_wait_returned"] = snapshot()
    if session.diagnostics or any(len(times) != total for times in entered):
        raise RuntimeError(f"Incomplete graph: {session.diagnostics!r}")
    # Drop the session's last sample claims before checking owner release. Include retirement
    # and worker drain in the lifetime metric, even if their work outlives delivery callbacks.
    del session
    if args.diagnostics:
        phases["session_destroyed"] = snapshot()
    drain(owners)
    end, end_cpu, end_threads = time.perf_counter_ns(), time.process_time_ns(), thread_cpu()
    if args.diagnostics:
        phases["drain_finished"] = snapshot()
        sample_pool("drain_finished")
    start = args.warmup
    release_metrics = {}
    if args.release_timings:
        records = [
            {
                "frame": frame,
                "first_emit_ns": first_emit_ns[frame],
                "emit_return_ns": emitted_ns[frame],
                "release_ns": released_ns[frame],
                "producer_observed_ns": producer_observed_ns[frame],
            }
            for frame in range(start, total)
        ]
        if any(not r["first_emit_ns"] or r["release_ns"] < r["emit_return_ns"] for r in records):
            raise RuntimeError("Missing or out-of-order source owner release timestamps")
        steady = [r for r in records if r["release_ns"] <= delivery_end]
        release_metrics = {
            "first_emit_to_release": distribution(
                [r["release_ns"] - r["first_emit_ns"] for r in records]
            ),
            "emit_return_to_release": distribution(
                [r["release_ns"] - r["emit_return_ns"] for r in records]
            ),
            "steady_first_emit_to_release": distribution(
                [r["release_ns"] - r["first_emit_ns"] for r in steady]
            )
            if steady
            else {},
            "steady_releases": len(steady),
            "releases_after_delivery": len(records) - len(steady),
            "records": records,
        }
        if args.pace_window:
            observed = [r for r in records if r["producer_observed_ns"]]
            if any(r["producer_observed_ns"] < r["emit_return_ns"] for r in observed):
                raise RuntimeError("Out-of-order producer completion observation")
            # A wait return is a host observation, not a device timestamp. Release can
            # precede this observation; retain signed lags and leave final-window frames
            # unobserved instead of introducing extra waits to finish the measurement.
            lags = [r["release_ns"] - r["producer_observed_ns"] for r in observed]
            release_metrics["producer_completion_observation"] = {
                "observed_frames": len(observed),
                "unobserved_frames": len(records) - len(observed),
                "released_before_observation": sum(lag < 0 for lag in lags),
                "emit_return_to_observation": distribution(
                    [r["producer_observed_ns"] - r["emit_return_ns"] for r in observed]
                )
                if observed
                else {},
                "observation_to_release_signed": distribution(lags) if lags else {},
            }
    return {
        "source_owner_release": release_metrics,
        "name": "pipeline",
        "callback_thread_ids": dict(zip(("source", "forward", "sink"), role_threads, strict=True)),
        "callback_cpu_total_ns": [sum(times) for times in cpu],
        "callback_cpu_measured_ns": [sum(times[start:]) for times in cpu],
        "phases": phases,
        "phase_deltas": phase_deltas(phases),
        "pool_samples": pool_samples,
        "stage_wall": {name: distribution(times) for name, times in stages.items() if times},
        "policy": policy.name,
        "frames": args.frames,
        "retries": retries,
        "callback_wall": [distribution(times[start:]) for times in durations],
        "callback_cpu": [distribution(times[start:]) for times in cpu],
        "source_to_forward": distribution(np.subtract(entered[1][start:], entered[0][start:])),
        "forward_to_sink": distribution(np.subtract(entered[2][start:], entered[1][start:])),
        "source_to_sink": distribution(np.subtract(entered[2][start:], entered[0][start:])),
        "delivery_ns_per_frame": (delivery_end - measurement_start) / args.frames,
        "full_lifetime_ns_per_frame": (end - measurement_start) / args.frames,
        "process_cpu_ns_per_frame": (end_cpu - before_cpu) / total,
        "thread_cpu_seconds": cpu_delta(before_threads, end_threads),
        "all_owners_released": True,
    }


def import_retire(args):
    stream = cp.cuda.Stream(non_blocking=True)
    owners = []
    stages = {name: [] for name in ("allocate", "import", "export", "drop")}
    before_threads = thread_cpu()
    phases = {"submission_start": snapshot()} if args.diagnostics else {}
    start, cpu = time.perf_counter_ns(), time.process_time_ns()
    with stream:
        for _ in range(args.frames):
            stage = time.perf_counter_ns() if args.stage_timings else 0
            array = cp.empty(args.tensor_bytes // 4, dtype=cp.float32)
            owners.append(weakref.ref(array))
            if args.stage_timings:
                allocated = time.perf_counter_ns()
            tensor = hs.Tensor.as_tensor(array, stream=stream)
            if args.stage_timings:
                imported = time.perf_counter_ns()
            view = cp.from_dlpack(tensor)
            if args.stage_timings:
                exported = time.perf_counter_ns()
            del view, tensor, array
            if args.stage_timings:
                stages["allocate"].append(allocated - stage)
                stages["import"].append(imported - allocated)
                stages["export"].append(exported - imported)
                stages["drop"].append(time.perf_counter_ns() - exported)
    submission_end = time.perf_counter_ns()
    pool_at_submission = {}
    if args.diagnostics:
        phases["submission_end"] = snapshot()
        pool = cp.get_default_memory_pool()
        pool_at_submission = {
            "total_bytes": pool.total_bytes(),
            "used_bytes": pool.used_bytes(),
            "free_blocks": pool.n_free_blocks(),
        }
    drain(owners)
    end = time.perf_counter_ns()
    end_cpu = time.process_time_ns()
    if args.diagnostics:
        phases["drain_finished"] = snapshot()
    return {
        "name": "external_import_retire",
        "phases": phases,
        "phase_deltas": phase_deltas(phases),
        "pool_at_submission": pool_at_submission,
        "stage_wall": {name: distribution(times) for name, times in stages.items() if times},
        "frames": args.frames,
        "submission_ns_per_frame": (submission_end - start) / args.frames,
        "full_lifetime_ns_per_frame": (end - start) / args.frames,
        "process_cpu_ns_per_frame": (end_cpu - cpu) / args.frames,
        "thread_cpu_seconds": cpu_delta(before_threads, thread_cpu()),
        "all_owners_released": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--delivery-timeout-seconds",
        type=float,
        default=60,
        help="Positive finite pipeline delivery timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--retirement-timeout-seconds",
        type=float,
        default=10,
        help="Positive finite post-stop session retirement timeout in seconds (default: 10)",
    )
    parser.add_argument("--tensor-bytes", type=int, default=1 << 20)
    parser.add_argument("--period-us", type=int, default=1000)
    parser.add_argument(
        "--queue-depth",
        type=int,
        default=2,
        help="Lossless edge capacity; finite smoke runs reserve their whole burst (default: 2)",
    )
    parser.add_argument("--gpu-work-us", type=int, default=0)
    parser.add_argument("--mode", choices=("pipeline", "import_retire"), default="pipeline")
    parser.add_argument("--policy", choices=("shared", "per_partition"), default="shared")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Capture phase CPU, live-thread snapshots and sampled pool/owner counts",
    )
    parser.add_argument(
        "--stage-timings",
        action="store_true",
        help="Time pipeline or import/export stages separately; perturbs timing",
    )
    parser.add_argument(
        "--release-timings",
        action="store_true",
        help="Timestamp each source owner release; measures reuse, not exact GPU completion",
    )
    parser.add_argument(
        "--pace-window",
        type=int,
        default=0,
        help="Bound queued producer work using blocking events; 0 disables pacing",
    )
    parser.add_argument("--sample-every", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for timeout in (args.delivery_timeout_seconds, args.retirement_timeout_seconds):
        if not 0 < timeout <= threading.TIMEOUT_MAX:
            parser.error("completion timeouts must be positive, finite and representable")
    if not 0 <= args.gpu_work_us <= ((1 << 64) - 1) // 1000:
        parser.error("GPU work must be a nonnegative representable nanosecond duration")
    if (
        min(args.frames, args.period_us, args.tensor_bytes, args.sample_every, args.queue_depth)
        <= 0
        or args.tensor_bytes % 4
        or min(args.warmup, args.pace_window) < 0
    ):
        parser.error("positive frames/period/queue depth and a float32-aligned size are required")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            print("CuPy benchmark unavailable: no visible CUDA device")
            return 77
    except RuntimeError as exc:
        print(f"CuPy benchmark unavailable: {exc}")
        return 77
    # Compile and prime CUDA/CuPy before CPU and latency windows.
    cp.empty(args.tensor_bytes // 4, dtype=cp.float32).fill(0)
    SPIN((1,), (1,), (np.uint64(1),))
    cp.cuda.runtime.deviceSynchronize()
    policy = (
        hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
        if args.policy == "shared"
        else hs.execution.CudaComputeStreamPolicy.PER_EXECUTION_PARTITION
    )
    result = pipeline(args, policy) if args.mode == "pipeline" else import_retire(args)
    result["metadata"] = {
        "args": vars(args) | {"output": str(args.output)},
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "sdk_module": hs.__file__,
        "sdk_libraries": sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if ".so" in line and ("holoscan" in line or "holomq" in line)
            }
        ),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cpu_tick_seconds": 1 / os.sysconf("SC_CLK_TCK"),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
