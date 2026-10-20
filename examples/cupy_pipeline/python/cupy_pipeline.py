# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass strided CuPy arrays through a GPU pipeline using only public Python APIs."""

import argparse
import importlib
import threading
import time

import holoscan as hs


def run_pipeline(cp, *, device=0, frames=10, per_partition_streams=False):
    """Run a finite CuPy source, transform, and validating sink."""
    completed = threading.Event()
    received = []
    # The sink validates every frame in order, so the pipeline must be lossless. A graph-bound emit
    # that finds a full blocking queue waits only a bounded time and then fails the run, and the
    # transform cannot retry later. The source admits a frame only while fewer than max_in_flight
    # frames await validation, and the transform's edge has room for all of them plus one still
    # active in the sink. The source's own edge stays small: when the transform falls behind, for
    # example during CuPy's first kernel compilation, the source retries the same array later.
    max_in_flight = 3

    def declare_output(spec, port):
        spec.output(port).produces_tensor(
            hs.tensor.OutputSpec(
                storage="pass_through",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype="complex64",
                rank=2,
                bounds=hs.tensor.TensorBounds(max_rank=2, max_byte_span=32 * 64 * 8),
            )
        )

    class Source(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            declare_output(spec, self.out)
            self.next_frame = 1
            self.pending = None

        def contract(self):
            # Retries do not advance a frame, so complete_after() cannot count successful emits.
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, ctx):
            emitted = self.next_frame - 1
            if self.next_frame <= frames and emitted - len(received) < max_in_flight:
                with ctx.stream():
                    # Slicing keeps the complex array strided. emit() retains its CuPy owner.
                    if self.pending is None:
                        self.pending = cp.full((32, 64), self.next_frame, dtype=cp.complex64)[
                            :, ::2
                        ]
                    try:
                        self.out.emit(self.pending)
                    except hs.errors.BackpressuredError:
                        # Retry the same array on a later clock tick when the queue has room.
                        return
                    self.pending = None
                self.next_frame += 1

    class DoubleIt(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor, as_=cp)
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.input(self.inp)
            declare_output(spec, self.out)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with ctx.stream():
                array = self.inp.receive_data()
                assert array.shape == (32, 32)
                assert array.strides == (64 * 8, 2 * 8)
                self.out.emit(array * 2)

    class Sink(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor, as_=cp)

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with ctx.stream():
                # This final copy synchronizes for CPU validation; the graph's edges share storage.
                array = cp.asnumpy(self.inp.receive_data())
                expected = 2 * (len(received) + 1)
                assert array.shape == (32, 32)
                assert (array == expected).all()
                received.append(expected)
                if len(received) == frames:
                    completed.set()

    graph = hs.Graph("cupy-pipeline", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    transform = graph.op(DoubleIt, "transform")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, transform["in"], queue_depth=2)
    graph.add_flow(transform.out, sink["in"], queue_depth=max_in_flight + 1)
    for name, node in (("source", source), ("transform", transform), ("sink", sink)):
        graph.partition(name).add(node)

    deployment = hs.execution.DeploymentBinding()
    for name in ("source", "transform"):
        deployment.bind_tensor_output_device(
            operator_path=name, output_port="out", device=hs.execution.DeviceId(device)
        )
    policy = (
        hs.execution.CudaComputeStreamPolicy.PER_EXECUTION_PARTITION
        if per_partition_streams
        else hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
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
    with hs.run_async(plan) as session:
        deadline = time.monotonic() + 10
        while not completed.wait(0.01) and time.monotonic() < deadline:
            if session.wait(timeout=0):
                break
        session.request_stop()
        if not session.wait(timeout=5):
            raise TimeoutError("CuPy pipeline did not stop")
    if not completed.is_set() or session.diagnostics:
        raise RuntimeError(f"CuPy pipeline did not complete: {session.diagnostics!r}")
    if session.termination is not hs.RunTermination.STOPPED:
        raise RuntimeError(f"Unexpected termination: {session.termination}")
    print(f"cupy_pipeline: received {frames} doubled arrays (last={received[-1]})")


def main():
    """Select a CUDA device and run the graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--per-partition-streams", action="store_true")
    parser.add_argument("--skip-if-unavailable", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0 or args.frames <= 0:
        parser.error("--device must be nonnegative and --frames must be positive")

    try:
        cp = importlib.import_module("cupy")
        count = cp.cuda.runtime.getDeviceCount()
    except (ImportError, RuntimeError) as exc:
        if args.skip_if_unavailable:
            print(f"CuPy/CUDA unavailable: {exc}")
            return 77
        raise SystemExit(f"This example requires CuPy and a visible CUDA GPU: {exc}") from exc
    if args.device >= count:
        if args.skip_if_unavailable and count == 0:
            print("No visible CUDA GPU")
            return 77
        parser.error(f"device {args.device} is outside the {count} visible CUDA devices")
    run_pipeline(
        cp, device=args.device, frames=args.frames, per_partition_streams=args.per_partition_streams
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
