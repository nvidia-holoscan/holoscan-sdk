# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose and run the multi-rate slow/fast fusion pipeline."""

from datetime import timedelta

import holoscan as hs


class PeriodicSource(hs.Operator):
    """Publish sequential values with activation time as their capture coordinate."""

    output = hs.Output("output", hs.payload.uint32)
    period = hs.Param(timedelta(milliseconds=20))
    scale = hs.Param(1, range=(1, 1_000_000))

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=self.period))

    def start(self):
        self.next_value = 0

    def compute(self, ctx):
        self.next_value += 1
        self.output.emit(
            self.next_value * self.scale,
            options=hs.EmitOptions(capture_time=ctx.activation_time),
        )


class FuseLatest(hs.Operator):
    """Add each fast sample to the latest eligible slow sample."""

    slow = hs.Input("slow", hs.payload.uint32)
    fast = hs.Input("fast", hs.payload.uint32)
    fused = hs.Output("fused", hs.payload.uint32)

    def setup(self, spec):
        spec.input(self.slow, queue_depth=8)
        spec.input(self.fast, queue_depth=16)
        spec.output(self.fused)

    def contract(self):
        return (
            hs.Contract()
            .trigger(hs.OnEach(self.fast))
            .input(self.slow, hs.LatestWithin(max_age=timedelta(milliseconds=100)))
            .on_missing(self.slow, hs.Skip())
        )

    def compute(self, context):
        slow_sample = self.slow.receive()
        fast_sample = self.fast.receive()
        self.fused.forward(fast_sample, fast_sample.data + slow_sample.data)


class BoundedSink(hs.Operator):
    """Request successful completion after six fused samples."""

    input = hs.Input("input", hs.payload.uint32)

    def setup(self, spec):
        spec.input(self.input, queue_depth=8)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, ctx):
        self.input.receive()
        self.count = getattr(self, "count", 0) + 1
        if self.count == 6:
            ctx.request_completion()


def build_graph() -> hs.Graph:
    """Build the two-rate fusion graph with an explicit partition cover."""
    graph = hs.Graph("holoscan-slow-fast-fusion")
    slow = graph.op(PeriodicSource, "slow", period=timedelta(milliseconds=20), scale=100)
    fast = graph.op(PeriodicSource, "fast", period=timedelta(milliseconds=2), scale=1)
    fuse = graph.op(FuseLatest, "fuse")
    sink = graph.op(BoundedSink, "sink")
    graph.add_flow(slow.output, fuse.slow, queue_depth=8)
    graph.add_flow(fast.output, fuse.fast, queue_depth=16)
    graph.add_flow(fuse.fused, sink.input, queue_depth=8)
    graph.partition("slow-source").add(slow)
    graph.partition("fast-fusion").add(fast).add(fuse).add(sink)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "runtime-clock"))
    return graph


def build_plan() -> hs.Plan:
    """Compile the example graph and return its inspectable plan."""
    return hs.compile(build_graph())


def main() -> None:
    """Run until the bounded sink requests successful completion."""
    plan = build_plan()
    if not plan:
        raise SystemExit(plan.summary())

    session = hs.run(plan)
    if session.termination is not hs.RunTermination.REQUESTED_COMPLETED or session.diagnostics:
        raise RuntimeError(f"unexpected run result: {session!r}")
    print("slow/fast fusion complete")


if __name__ == "__main__":
    main()
