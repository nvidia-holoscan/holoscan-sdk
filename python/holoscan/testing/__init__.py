# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite operator tests using the SDK's optional native testing component."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from holoscan import ConnectionOptions, QueuePolicy
from holoscan.errors import HoloscanError

from ._testing import (
    CaptureFailure,
    ClockTestOutcome,
    ControlledClockRun,
    FrameRelease,
    Retain,
    SchemaSnapshot,
    TensorSnapshot,
    TestResult,
    TestRunOptions,
    _CapturePolicy,
    _Harness,
    _RunningFixture,
)

if TYPE_CHECKING:
    from types import TracebackType

    from holoscan import CompileOptions, Operator, RunTermination
    from holoscan.execution import DeviceId
    from holoscan.lifecycle import (
        LifecycleAccessPolicy,
        LifecycleControlHandle,
        LifecycleOperatorRef,
    )
    from holoscan.registry import OperatorType
    from holoscan.tensor import InputSpec, MemoryConversionPolicy, OutputSpec


def _timestamp(value: int | None) -> int | None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise TypeError("capture_timestamp_ns must be an integer or None")
    if value is not None and not -(2**63) <= value < 2**63:
        raise OverflowError("capture_timestamp_ns must fit in signed 64-bit nanoseconds")
    return value


@dataclass(frozen=True)
class TestSample:
    """One staged payload and its optional capture coordinate in nanoseconds."""

    __test__ = False

    data: object
    capture_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        _timestamp(self.capture_timestamp_ns)


@dataclass(frozen=True, kw_only=True)
class TestLifecycleOptions:
    """Immutable permissions for the tested operator; both default to denied.

    Set these between attempts. Self and external restart are independent;
    neither flag issues a request or bypasses native transaction/access policy.
    """

    __test__ = False

    allow_self_restart: bool = False
    allow_external_restart: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.allow_self_restart) is not bool
            or type(self.allow_external_restart) is not bool
        ):
            raise TypeError("lifecycle restart permissions must be bool")


def sample_at(data: object, *, capture_timestamp_ns: int) -> TestSample:
    """Create one timestamped staging envelope.

    Coordinates are signed 64-bit integer nanoseconds in the fixture clock domain; they do not
    advance time or schedule compute. Reject booleans, non-integers and out-of-range values.
    """
    return TestSample(data, _timestamp(capture_timestamp_ns))


def _checked(result: TestResult, check: bool) -> TestResult:
    if check:
        try:
            result._raise_for_status()
        except HoloscanError as error:
            error.result = result
            raise
    return result


def result_of(error: BaseException) -> TestResult | None:
    """Return the joined test result directly attached to an exception, if any.

    Preserve the original result's identity and partial outputs. Unrelated
    exceptions and attachments return None; causes are not searched. Inspection
    never executes or joins an operator.
    """
    result = vars(error).get("result")
    return result if isinstance(result, TestResult) else None


class OperatorTestHarness:
    """Stage finite inputs and run fresh instances of one registered or Python operator.

    ``push`` stages one payload, including a container or ``None`` on an admitted
    object port. Use ``push_many`` to explicitly stage multiple payloads.
    Captured results remain available after the harness is closed.
    """

    def __init__(
        self,
        operator: type[Operator] | OperatorType | str,
        *,
        params: Mapping[str, object] | None = None,
        outputs: str | Sequence[str] | None = None,
    ) -> None:
        selected = [outputs] if isinstance(outputs, str) else outputs
        self._native = _Harness(operator, {} if params is None else dict(params), selected)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"operator test cleanup also failed: {cleanup_error}")

    def close(self) -> None:
        """Release the builder resources.

        Existing results and independently owned running attempts remain valid.
        """
        self._native.close()

    def push(self, port: str, value: object) -> None:
        """Append one payload to an input.

        Containers remain one payload; None is data on an admitted object port. Reject mixing
        grouped and independent producers for an input.
        """
        self._native.stage(port, [value], [None])

    def push_many(self, port: str, values: Iterable[object]) -> None:
        """Atomically append an explicit batch in caller order.

        Reject strings, bytes, bytearrays and mappings as ambiguous batches. Conversion failure
        leaves pending input unchanged.
        """
        if isinstance(values, (str, bytes, bytearray, Mapping)):
            raise TypeError("push_many requires an explicit batch; use push for one payload")
        values = list(values)
        self._native.stage(port, values, [None] * len(values))

    def push_samples(self, port: str, samples: Iterable[TestSample]) -> None:
        """Atomically append TestSample envelopes in caller order.

        Capture coordinates do not sort inputs or advance time. Reject non-sample entries and
        invalid timestamps before staging changes.
        """
        samples = list(samples)
        if any(not isinstance(sample, TestSample) for sample in samples):
            raise TypeError("push_samples requires TestSample entries")
        self._native.stage(
            port,
            [sample.data for sample in samples],
            [_timestamp(sample.capture_timestamp_ns) for sample in samples],
        )

    def push_frame(
        self,
        entries: Mapping[str, object] | Iterable[tuple[str, object]],
        *,
        capture_timestamp_ns: int | None = None,
    ) -> None:
        """Atomically append one frame from a mapping or ordered unique (port, payload) pairs.

        Values share the optional nanosecond capture coordinate; omitted ports publish nothing.
        Reject duplicate ports and mixing grouped and independent producers.
        """
        if isinstance(entries, Mapping):
            entries = entries.items()
        self._native.push_frame(list(entries), _timestamp(capture_timestamp_ns))

    def capture(self, port: str, *, capacity: int = 1024) -> None:
        """Replace the selected output policy with value capture and a positive finite record
        limit.

        Tensor-bearing outputs require capture_snapshot or explicit capture_native. Overflow
        fails the run without silent truncation; Python objects retain identity rather than an
        implicit deep copy.
        """
        self._native.capture(port, _CapturePolicy.VALUE, capacity, 0)

    def capture_snapshot(
        self, port: str, *, capacity: int = 1024, maximum_bytes: int = 67108864
    ) -> None:
        """Replace the output policy with independent host snapshots.

        capacity bounds records; maximum_bytes bounds each complete record, including all
        admitted Tensor fields of a schema. Both limits are positive. Overflow fails capture
        without publishing a partial record.
        """
        self._native.capture(port, _CapturePolicy.SNAPSHOT, capacity, maximum_bytes)

    def capture_native(self, port: str, policy: Retain | None = None) -> None:
        """Opt into native Tensor retention, retaining source backing instead of copying to host.

        The finite source, single output, capacity and completion must satisfy native admission.
        None selects the default Retain record limit.
        """
        retain = Retain() if policy is None else policy
        self._native.capture(port, _CapturePolicy.NATIVE, retain.max_records, 0)

    def run(self, options: TestRunOptions | None = None, *, check: bool = True) -> TestResult:
        """Execute and join a fresh attempt.

        By default, failure raises the native HoloscanError subclass with .result attached;
        check=False returns that result. Invalid configuration can still raise before a result
        exists. Joining releases the GIL and rejects Runtime callback contexts.
        """
        return _checked(self._native.run(TestRunOptions() if options is None else options), check)

    def run_async(self, options: TestRunOptions | None = None) -> RunningFixture:
        """Start a fresh attempt owning resources independently of the builder.

        Inputs must already be staged; payloads become available only in joined finish results.
        One active attempt per harness is permitted.
        """
        return RunningFixture(
            self._native.run_async(TestRunOptions() if options is None else options)
        )

    def run_controlled_clock(
        self, options: ControlledClockRun, *, check: bool = True
    ) -> TestResult:
        """Run and join the restricted native clock-observation profile.

        Requires one sample per scalar input, ordinary OnClock/LatestWithin/Skip semantics and
        capture of every output. Unsupported configurations reject before staged input is
        consumed. Checked failures expose .result.
        """
        return _checked(self._native.run_controlled_clock(options), check)

    def set_input_queue_depth(self, port: str, depth: int | None) -> None:
        """Replace tested-operator input history depth, distinct from edge queue capacity.

        None clears the override. Native range/direction checks and active-attempt rejection
        preserve previous configuration on failure.
        """
        self._native.input_queue_depth(port, depth)

    def clear_input_queue_depth(self, port: str) -> None:
        """Remove this port's input queue depth override."""
        self.set_input_queue_depth(port, None)

    def set_lifecycle_options(self, options: TestLifecycleOptions | None) -> None:
        """Replace tested-operator restart permissions for subsequent fresh attempts.

        None clears both permissions. Values are copied; invalid or active-attempt
        changes preserve prior state. Permissions do not issue a restart or bypass
        native transaction capacities and runtime target/stage access policy.
        """
        if options is not None and not isinstance(options, TestLifecycleOptions):
            raise TypeError("options must be TestLifecycleOptions or None")
        self._native.set_lifecycle_options(
            False if options is None else options.allow_self_restart,
            False if options is None else options.allow_external_restart,
        )

    def clear_lifecycle_options(self) -> None:
        """Clear both restart permissions without changing staged data or compile options."""
        self.set_lifecycle_options(None)

    def set_compile_options(self, options: CompileOptions) -> None:
        """Copy compile options for subsequent fresh attempts.

        Later caller mutation does not change this copy. Native application order and active-
        attempt mutation rejection remain authoritative.
        """
        self._native.set_compile_options(options)

    def set_params(self, params: Mapping[str, object]) -> None:
        """Replace parameters atomically after validating pending inputs, capture selectors and
        overrides.

        Failed construction or validation preserves prior state. Automatic outputs follow new
        declarations; explicit selections must remain valid.
        """
        self._native.set_params(dict(params))

    def set_input_flow(
        self,
        port: str,
        *,
        queue_policy: QueuePolicy = QueuePolicy.BLOCKING,
        queue_depth: int | None = None,
        topic_name: str | None = None,
        memory_conversion: MemoryConversionPolicy | None = None,
        max_inflight_samples: int | None = None,
    ) -> None:
        """Replace the input edge options using the canonical Graph keyword rules.

        Omitted/None fields take Graph defaults. With no keywords, set default
        options; use clear_input_flow to restore native fixture derivation.
        Values are copied. Invalid options and active attempts leave prior state intact.
        """
        options = ConnectionOptions(
            queue_policy=queue_policy,
            queue_depth=queue_depth,
            topic_name=topic_name,
            memory_conversion=memory_conversion,
            max_inflight_samples=max_inflight_samples,
        )
        self._native.set_override("input_flow", port, options)

    def clear_input_flow(self, port: str) -> None:
        """Remove the input edge override and restore native fixture derivation."""
        self._native.set_override("input_flow", port, None)

    def set_output_flow(
        self,
        port: str,
        *,
        queue_policy: QueuePolicy = QueuePolicy.BLOCKING,
        queue_depth: int | None = None,
        topic_name: str | None = None,
        memory_conversion: MemoryConversionPolicy | None = None,
        max_inflight_samples: int | None = None,
    ) -> None:
        """Replace the output edge options using the canonical Graph keyword rules.

        Omitted/None fields take Graph defaults. With no keywords, set default
        options; use clear_output_flow to restore native fixture derivation.
        Values are copied. Invalid options and active attempts leave prior state intact.
        """
        options = ConnectionOptions(
            queue_policy=queue_policy,
            queue_depth=queue_depth,
            topic_name=topic_name,
            memory_conversion=memory_conversion,
            max_inflight_samples=max_inflight_samples,
        )
        self._native.set_override("output_flow", port, options)

    def clear_output_flow(self, port: str) -> None:
        """Remove the output edge override and restore native fixture derivation."""
        self._native.set_override("output_flow", port, None)

    def set_input_tensor_spec(self, port: str, spec: InputSpec | None) -> None:
        """Copy or clear an operator Tensor input contract."""
        self._native.set_override("input_tensor_spec", port, spec)

    def clear_input_tensor_spec(self, port: str) -> None:
        """Remove this port's input tensor spec override."""
        self.set_input_tensor_spec(port, None)

    def set_output_tensor_spec(self, port: str, spec: OutputSpec | None) -> None:
        """Copy or clear an operator Tensor output contract."""
        self._native.set_override("output_tensor_spec", port, spec)

    def clear_output_tensor_spec(self, port: str) -> None:
        """Remove this port's output tensor spec override."""
        self.set_output_tensor_spec(port, None)

    def set_input_source_tensor_spec(self, port: str, spec: OutputSpec | None) -> None:
        """Copy or clear a staged producer Tensor promise."""
        self._native.set_override("input_source_tensor_spec", port, spec)

    def clear_input_source_tensor_spec(self, port: str) -> None:
        """Remove this port's input source tensor spec override."""
        self.set_input_source_tensor_spec(port, None)

    def set_input_source_device(self, port: str, device: DeviceId | None) -> None:
        """Copy or clear staged-source CUDA placement."""
        self._native.set_override("input_source_device", port, device)

    def clear_input_source_device(self, port: str) -> None:
        """Remove this port's input source device override."""
        self.set_input_source_device(port, None)

    def set_output_device(self, port: str, device: DeviceId | None) -> None:
        """Copy or clear tested-operator CUDA output placement."""
        self._native.set_override("output_device", port, device)

    def clear_output_device(self, port: str) -> None:
        """Remove this port's output device override."""
        self.set_output_device(port, None)


class RunningFixture:
    """Own one attempt until joined; live observation exposes counts, never payload storage."""

    def __init__(self, native: _RunningFixture) -> None:
        if not isinstance(native, _RunningFixture):
            raise TypeError("running owners must be created by OperatorTestHarness.run_async()")
        self._native = native
        self._result = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"operator test cleanup also failed: {cleanup_error}")

    def finish(self, *, check: bool = True) -> TestResult:
        """Join normally with the GIL released and return the same result on repeated calls.

        Default checking raises with .result attached; check=False returns a failed result for
        inspection. Reject joins from Runtime callbacks.
        """
        if self._result is None:
            self._result = self._native.finish()
        return _checked(self._result, check)

    def close(self) -> None:
        """Request cooperative stop and join with the GIL released.

        This has no hard timeout and rejects Runtime callback joins. Use finish to inspect the
        resulting outcome.
        """
        self._native.close()

    def request_stop(self) -> None:
        """Request cooperative stop without joining; permitted in Runtime callbacks."""
        self._native.request_stop()

    @property
    def is_running(self) -> bool:
        """Report whether the native attempt is still active; do not join."""
        return self._native.is_running

    @property
    def termination(self) -> RunTermination:
        """Return the native termination observation without joining the attempt."""
        return self._native.termination

    def release_frame(self, *, timeout_ms: int | None = None) -> None:
        """Admit the next pre-staged manual frame.

        None or zero attempts release once without waiting; positive integer milliseconds allow
        bounded admission waiting. Success does not mean the frame has been computed or
        captured.
        """
        self._native.release_frame(0 if timeout_ms is None else timeout_ms)

    def operator_ref(self, plan_ref: LifecycleOperatorRef | None = None) -> LifecycleOperatorRef:
        """Return the tested operator reference bound to this run, or bind the supplied plan
        reference through Core.

        Foreign and stale identities reject.
        """
        return self._native.operator_ref(plan_ref)

    def lifecycle_control(self, policy: LifecycleAccessPolicy) -> LifecycleControlHandle:
        """Create Core lifecycle control restricted by run-bound targets and source stages in the
        policy.

        Graph permissions and transaction capacities remain required; creating control does not
        request restart.
        """
        return self._native.lifecycle_control(policy)

    def output_count(self, port: str) -> int:
        """Observe the captured record count without exposing unjoined payloads."""
        return self._native.output_count(port)

    def wait_for_output(self, port: str, minimum: int, *, timeout_ms: int) -> bool:
        """Wait up to timeout_ms milliseconds for a count; zero only polls.

        Return False when the count is not reached before timeout or completion.
        This releases the GIL while waiting and never joins or reruns the operator.
        """
        return self._native.wait_for_output(port, minimum, timeout_ms)


__all__ = [
    "CaptureFailure",
    "ClockTestOutcome",
    "ControlledClockRun",
    "FrameRelease",
    "OperatorTestHarness",
    "Retain",
    "RunningFixture",
    "SchemaSnapshot",
    "TensorSnapshot",
    "TestResult",
    "TestLifecycleOptions",
    "TestRunOptions",
    "TestSample",
    "sample_at",
    "result_of",
]
