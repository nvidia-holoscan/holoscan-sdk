# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run native Python operators through the caller-managed session surface."""

import holoscan as hs


class TickSource(hs.Operator):
    """Emit an unbounded sequence for caller-controlled runs."""

    output = hs.Output("output", hs.payload.uint32)

    def setup(self, spec: hs.OperatorSpec) -> None:
        """Declare the scalar output."""
        spec.output(self.output)

    def contract(self) -> hs.Contract:
        """Drive the source from the graph's default clock."""
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

    def compute(self, context: hs.ExecutionContext) -> None:
        """Emit the next sequence value."""
        value = getattr(self, "next_value", 1)
        self.output.emit(value)
        self.next_value = value + 1


class CompletingSource(TickSource):
    """Request successful completion through a bounded activation contract."""

    def contract(self) -> hs.Contract:
        """Complete the run after five clock activations."""
        return super().contract().complete_after(5)


class CountingSink(hs.Operator):
    """Consume every source value without choosing run termination."""

    input = hs.Input("input", hs.payload.uint32)

    def setup(self, spec: hs.OperatorSpec) -> None:
        """Declare the bounded scalar input."""
        spec.input(self.input, queue_depth=4)

    def contract(self) -> hs.Contract:
        """Run once for each available input sample."""
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context: hs.ExecutionContext) -> None:
        """Consume one value."""
        self.input.receive_data()


class FailingSink(CountingSink):
    """Raise from compute to demonstrate fail-closed Python callbacks."""

    def compute(self, context: hs.ExecutionContext) -> None:
        """Consume one value and deliberately fail the run."""
        super().compute(context)
        raise RuntimeError("intentional Python operator failure")


def compile_pipeline(
    name: str, source_type: type[TickSource], sink_type: type[CountingSink]
) -> hs.Plan:
    """Compile one source-to-sink pipeline with the selected terminal behavior."""
    graph = hs.Graph(name, default_clock=hs.RealtimeClock)
    source = graph.op(source_type, "source")
    sink = graph.op(sink_type, "sink")
    graph.add_flow(source.output, sink.input, queue_depth=4)
    return hs.compile(graph)


def require_plan(plan: hs.Plan) -> None:
    """Stop immediately when a graph is not admitted."""
    if not plan:
        raise SystemExit(plan.summary())


def main() -> None:
    """Demonstrate cooperative stop, failure, completion, and scoped cleanup."""
    cooperative = compile_pipeline("cooperative-stop", TickSource, CountingSink)
    require_plan(cooperative)
    with hs.run_async(cooperative) as session:
        if session.wait(timeout=0.05):
            raise RuntimeError("cooperative graph ended before stop was requested")
        session.request_stop()
        session.request_stop()
        if not session.wait(timeout=5.0):
            raise TimeoutError("cooperative graph did not stop within five seconds")
    if session.termination is not hs.RunTermination.STOPPED or session.diagnostics:
        raise RuntimeError(f"unexpected cooperative result: {session!r}")
    print("A: cooperative stop -> termination=STOPPED, diagnostics=0")

    failing = compile_pipeline("fail-closed", TickSource, FailingSink)
    require_plan(failing)
    failed_session = hs.run(failing)
    if failed_session.termination is not hs.RunTermination.FAILED or not failed_session.diagnostics:
        raise RuntimeError(f"unexpected failure result: {failed_session!r}")
    print(f"B: Python compute error -> diagnostic={failed_session.diagnostics[0].token}")

    completing = compile_pipeline("contract-completion", CompletingSource, CountingSink)
    require_plan(completing)
    completed_session = hs.run(completing)
    if (
        completed_session.termination is not hs.RunTermination.COMPLETED
        or completed_session.diagnostics
    ):
        raise RuntimeError(f"unexpected completion result: {completed_session!r}")
    print("C: complete_after contract -> termination=COMPLETED, diagnostics=0")

    with hs.run_async(cooperative) as scoped_session:
        if scoped_session.wait(timeout=0.05):
            raise RuntimeError("scoped graph ended before context cleanup")
    if scoped_session.termination is not hs.RunTermination.STOPPED:
        raise RuntimeError(f"unexpected scoped result: {scoped_session!r}")
    print("D: context manager retained stop/join authority")

    blocking_session = hs.run(completing)
    if blocking_session.termination is not hs.RunTermination.COMPLETED:
        raise RuntimeError(f"unexpected blocking result: {blocking_session!r}")
    print("E: caller-managed blocking run -> terminal session returned")
    print("running_and_stopping: all native Python sub-runs behaved as documented")


if __name__ == "__main__":
    main()
