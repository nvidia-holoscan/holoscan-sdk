# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author graph-scope queue, partition, region, and clock policy."""

import threading
import time

import holoscan as hs


class CompositionObservation:
    """Track both native-Python graph paths for the final example assertions."""

    condition = threading.Condition()
    stats = {
        "blocking": {"produced": 0, "received": 0, "backpressured": 0},
        "freshest": {"produced": 0, "received": 0, "backpressured": 0},
    }

    @classmethod
    def reset(cls):
        with cls.condition:
            for path in cls.stats.values():
                path.update(produced=0, received=0, backpressured=0)

    @classmethod
    def record(cls, path, field, amount=1):
        with cls.condition:
            cls.stats[path][field] += amount
            cls.condition.notify_all()

    @classmethod
    def wait_for_burst(cls, timeout=10.0):
        deadline = time.monotonic() + timeout
        with cls.condition:
            while not (
                cls.stats["blocking"]["produced"] >= 30
                and cls.stats["freshest"]["produced"] >= 30
                and cls.stats["blocking"]["received"] >= 30
                and cls.stats["freshest"]["received"] >= 1
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                cls.condition.wait(remaining)
        return True


class BurstSource(hs.Operator):
    """Publish a bounded burst; used by both queue-policy paths in ``build_graph``."""

    output = hs.Output("output", hs.payload.uint32)
    path = hs.Param("blocking", allowed=("blocking", "freshest"))

    def setup(self, spec):
        spec.output(self.output)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(2)))

    def start(self):
        self.next_value = 1

    def compute(self, ctx):
        if self.next_value > 30:
            return
        try:
            self.output.emit(self.next_value)
        except hs.errors.BackpressuredError:
            CompositionObservation.record(self.path, "backpressured")
            return
        except hs.HoloscanError:
            if ctx.stop_requested:
                return
            raise
        self.next_value += 1
        CompositionObservation.record(self.path, "produced")


class SlowSink(hs.Operator):
    """Consume one DrainAll batch per tick; used by both paths in ``build_graph``."""

    input = hs.Input("input", hs.payload.uint32)
    path = hs.Param("blocking", allowed=("blocking", "freshest"))

    def setup(self, spec):
        spec.input(self.input, queue_depth=4)

    def contract(self):
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=hs.time.ms(20)))
            .input(self.input, hs.DrainAll())
        )

    def compute(self, context):
        batch = self.input.selected_batch()
        CompositionObservation.record(self.path, "received", len(batch))


def build_graph() -> hs.Graph:
    """Build lossless and freshest-wins paths from identical operator types."""
    graph = hs.Graph("holoscan-graph-composition")
    burst_blocking = graph.op(BurstSource, "burst_blocking", path="blocking")
    burst_freshest = graph.op(BurstSource, "burst_freshest", path="freshest")
    sink_blocking = graph.op(SlowSink, "sink_blocking", path="blocking")
    sink_freshest = graph.op(SlowSink, "sink_freshest", path="freshest")

    graph.add_flow(
        burst_blocking.output,
        sink_blocking.input,
        queue_policy=hs.QueuePolicy.BLOCKING,
        queue_depth=4,
    )
    freshest_flow = graph.add_flow(burst_freshest.output, sink_freshest.input)
    graph.configure(sink_freshest.input).queue_depth(1).diagnostic_alias("sink")
    graph.configure(freshest_flow).queue_policy(hs.QueuePolicy.DROP_OLDEST).max_inflight_samples(64)

    graph.partition("blocking-source").add(burst_blocking)
    graph.partition("blocking-sink").add(sink_blocking)
    graph.partition("freshest-source").add(burst_freshest)
    graph.partition("freshest-sink").add(sink_freshest)
    graph.region("critical").add(sink_blocking).constrain(
        hs.execution.RegionConstraint.bounded_memory()
    )

    runtime_clock = graph.add_clock(hs.RealtimeClock, "runtime")
    capture_clock = graph.add_clock(hs.ManualClock, "capture")
    graph.set_default_clock(runtime_clock)
    graph.add_clock_conversion(
        capture_clock,
        runtime_clock,
        hs.clocks.AffineClockConversion(offset_ns=500_000),
    )
    return graph


def build_plan() -> hs.Plan:
    """Compile the policy-rich graph and return its inspectable plan."""
    return hs.compile(build_graph())


def main() -> None:
    """Run the bounded bursts, then stop the recurring clocked sinks."""
    CompositionObservation.reset()
    plan = build_plan()
    if not plan:
        raise SystemExit(plan.summary())

    with hs.run_async(plan) as session:
        if not CompositionObservation.wait_for_burst():
            raise TimeoutError("bounded graph-composition burst did not complete")
        session.request_stop()
        if not session.wait(timeout=5.0):
            raise TimeoutError("graph did not stop within five seconds")
    if session.termination is not hs.RunTermination.STOPPED or session.diagnostics:
        raise RuntimeError(f"unexpected run result: {session!r}")
    if CompositionObservation.stats["blocking"]["backpressured"] == 0:
        raise RuntimeError("blocking path did not observe bounded backpressure")
    freshest = CompositionObservation.stats["freshest"]
    if freshest["received"] >= freshest["produced"]:
        raise RuntimeError("freshest path did not drop superseded samples")
    print("graph_composition: blocking and freshest-wins policies executed")
    print("graph composition complete")


if __name__ == "__main__":
    main()
