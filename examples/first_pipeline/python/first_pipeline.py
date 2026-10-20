# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose, compile, and run a minimal graph of native Python operators."""

import threading
import time

import holoscan as hs


class PipelineObservation:
    """Record sink progress so the application can request cooperative stop."""

    condition = threading.Condition()
    count = 0
    last_value = 0

    @classmethod
    def record(cls, value: int) -> None:
        """Record one received value and wake the application thread."""
        with cls.condition:
            cls.count += 1
            cls.last_value = value
            cls.condition.notify_all()

    @classmethod
    def wait_for(cls, expected: int, timeout: float) -> bool:
        """Wait until the sink has observed the requested number of values."""
        deadline = time.monotonic() + timeout
        with cls.condition:
            while cls.count < expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                cls.condition.wait(remaining)
        return True


class CounterSource(hs.Operator):
    """Emit a finite sequence while the application retains stop authority."""

    out = hs.Output("out", hs.payload.uint32)
    limit = hs.Param(10, range=(1, 1_000_000))

    def setup(self, spec: hs.OperatorSpec) -> None:
        """Declare the source output."""
        spec.output(self.out)

    def contract(self) -> hs.Contract:
        """Drive the source from the graph's default clock."""
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(10)))

    def compute(self, context: hs.ExecutionContext) -> None:
        """Emit the next value until the configured limit is reached."""
        value = getattr(self, "next_value", 1)
        if value <= self.limit:
            self.out.emit(value)
            self.next_value = value + 1


class DoubleIt(hs.Operator):
    """Double each scalar sample."""

    input = hs.Input("in", hs.payload.uint32)
    output = hs.Output("out", hs.payload.uint32)

    def setup(self, spec: hs.OperatorSpec) -> None:
        """Declare the transform ports and bounded input queue."""
        spec.input(self.input, queue_depth=4)
        spec.output(self.output)

    def contract(self) -> hs.Contract:
        """Run once for each input sample."""
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context: hs.ExecutionContext) -> None:
        """Receive, double, and publish one value."""
        self.output.emit(self.input.receive_data() * 2)


class PrintSink(hs.Operator):
    """Print and record each doubled sample."""

    input = hs.Input("in", hs.payload.uint32)

    def setup(self, spec: hs.OperatorSpec) -> None:
        """Declare the bounded sink input."""
        spec.input(self.input, queue_depth=4)

    def contract(self) -> hs.Contract:
        """Run once for each transformed sample."""
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context: hs.ExecutionContext) -> None:
        """Consume one value and make progress visible to the application."""
        value = self.input.receive_data()
        print(f"tick {value}")
        PipelineObservation.record(value)


def build_graph() -> hs.Graph:
    """Build the counter-to-transform-to-sink graph."""
    graph = hs.Graph("first-pipeline", default_clock=hs.RealtimeClock)
    source = graph.op(CounterSource, "source", limit=10)
    doubler = graph.op(DoubleIt, "doubler")
    sink = graph.op(PrintSink, "sink")
    graph.add_flow(source.out, doubler["in"], queue_depth=4)
    graph.add_flow(doubler.out, sink["in"], queue_depth=4)
    return graph


def build_plan() -> hs.Plan:
    """Compile the example graph and return its inspectable plan."""
    return hs.compile(build_graph())


def main() -> None:
    """Run the finite source and stop after all values reach the sink."""
    plan = build_plan()
    if not plan:
        raise SystemExit(plan.summary())

    with hs.run_async(plan) as session:
        observed = PipelineObservation.wait_for(10, 5.0)
        session.request_stop()
        stopped = session.wait(timeout=5.0)

    if not observed:
        raise TimeoutError(f"first_pipeline did not receive ten samples: {session.diagnostics!r}")
    if not stopped:
        raise TimeoutError("first_pipeline did not stop within five seconds")
    if session.termination is not hs.RunTermination.STOPPED or session.diagnostics:
        raise RuntimeError(f"unexpected run result: {session!r}")
    if PipelineObservation.last_value != 20:
        raise RuntimeError("first_pipeline did not preserve the expected scalar values")
    print("first_pipeline: received 10 doubled ticks (last=20) termination=stopped")


if __name__ == "__main__":
    main()
