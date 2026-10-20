# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect admitted, rejected, and changed native-Python execution plans."""

import threading
import time
from hashlib import sha256
from pathlib import Path

import holoscan as hs


class PipelineObservation:
    """Record sink progress for the example's short runtime check."""

    condition = threading.Condition()
    count = 0

    @classmethod
    def reset(cls) -> None:
        """Clear observations before running an admitted plan."""
        with cls.condition:
            cls.count = 0

    @classmethod
    def record(cls) -> None:
        """Record one received sample."""
        with cls.condition:
            cls.count += 1
            cls.condition.notify_all()

    @classmethod
    def wait_for(cls, expected: int, timeout: float) -> bool:
        """Wait for bounded sink progress."""
        deadline = time.monotonic() + timeout
        with cls.condition:
            while cls.count < expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                cls.condition.wait(remaining)
        return True


class CounterSource(hs.Operator):
    """Publish a finite scalar sequence for plan and runtime demonstrations."""

    output = hs.Output("output", hs.payload.uint32)
    limit = hs.Param(3, range=(1, 1_000_000))

    def setup(self, spec: hs.OperatorSpec) -> None:
        """Declare the source output."""
        spec.output(self.output)

    def contract(self) -> hs.Contract:
        """Drive the source from the graph's default clock."""
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(10)))

    def compute(self, context: hs.ExecutionContext) -> None:
        """Emit one value until the configured limit is reached."""
        value = getattr(self, "next_value", 1)
        if value <= self.limit:
            self.output.emit(value)
            self.next_value = value + 1


class CountingSink(hs.Operator):
    """Consume scalar samples and expose progress to the application thread."""

    input = hs.Input("input", hs.payload.uint32)

    def setup(self, spec: hs.OperatorSpec) -> None:
        """Declare the bounded sink input."""
        spec.input(self.input, queue_depth=8)

    def contract(self) -> hs.Contract:
        """Run once for each source sample."""
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context: hs.ExecutionContext) -> None:
        """Consume one sample and record progress."""
        self.input.receive_data()
        PipelineObservation.record()


def build_pipeline(
    name: str,
    *,
    limit: int = 3,
    queue_depth: int = 4,
    complete_partition_cover: bool = True,
) -> hs.Graph:
    """Build one source-to-sink variant used by the admission checks."""
    graph = hs.Graph(name, default_clock=hs.RealtimeClock)
    source = graph.op(CounterSource, "source", limit=limit)
    sink = graph.op(CountingSink, "sink")
    graph.add_flow(source.output, sink.input, queue_depth=queue_depth)
    if not complete_partition_cover:
        graph.partition("incomplete-cover").add(source)
    return graph


def build_plan() -> hs.Plan:
    """Compile the canonical admitted graph."""
    return hs.compile(build_pipeline("compile-and-check"))


def plan_hash(plan: hs.Plan) -> str:
    """Return a short digest of the canonical JSON exposed by Python."""
    return sha256(plan.json().encode()).hexdigest()[:16]


def main() -> None:
    """Exercise cold admission, deterministic identity, and one short run."""
    plan = build_plan()
    if not plan:
        raise SystemExit(plan.summary())
    print(f"[admitted] plan ok: yes (diagnostics: {len(plan.diagnostics)})")

    json_path = Path("compile_and_check_plan.json")
    json_path.write_text(plan.json(), encoding="utf-8")
    print(f"[admitted] plan.json written: {json_path} ({len(plan.json())} bytes)")

    rejected = hs.compile(
        build_pipeline("compile-and-check-broken", complete_partition_cover=False)
    )
    print(f"[broken] plan ok: {'yes' if rejected else 'no'}")
    for diagnostic in rejected.diagnostics:
        print(f"[broken]   {diagnostic.token} | {diagnostic.severity.name.lower()}")
    if rejected or not rejected.errors:
        raise RuntimeError("broken graph did not retain an admission diagnostic")
    try:
        rejected.require_no_errors()
    except hs.errors.CompileError:
        print("[broken] require_no_errors raised CompileError: yes")
    else:
        raise RuntimeError("require_no_errors did not raise CompileError")

    second_plan = build_plan()
    stable = plan.json() == second_plan.json()
    print(f"[digest] first  compile: {plan_hash(plan)}")
    print(f"[digest] second compile: {plan_hash(second_plan)}")
    print(f"[digest] plan.json byte-equal: {'yes' if stable else 'no'}")
    if not stable:
        raise RuntimeError("identical graphs produced different plan JSON")

    edited = hs.compile(build_pipeline("compile-and-check", limit=5, queue_depth=8))
    changed = plan.json() != edited.json()
    print(f"[digest] edited compile: {plan_hash(edited)}")
    print(f"digest changed after edit: {'yes' if changed else 'no'}")
    if not changed:
        raise RuntimeError("semantic edit did not change plan JSON")

    PipelineObservation.reset()
    with hs.run_async(plan) as session:
        observed = PipelineObservation.wait_for(3, 5.0)
        session.request_stop()
        stopped = session.wait(timeout=5.0)
    if not observed:
        raise TimeoutError(
            f"compile_and_check did not receive three samples: {session.diagnostics!r}"
        )
    if not stopped:
        raise TimeoutError("compile_and_check did not stop within five seconds")
    if session.termination is not hs.RunTermination.STOPPED or session.diagnostics:
        raise RuntimeError(f"unexpected run result: {session!r}")
    print("[run] termination=stopped diagnostics=0")
    print("compile_and_check example complete: admission, diagnostics, and identity verified")


if __name__ == "__main__":
    main()
