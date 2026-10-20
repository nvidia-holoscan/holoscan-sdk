# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from dataclasses import dataclass
from types import TracebackType
from typing import ClassVar, Self, type_check_only

from holoscan import (
    CompileOptions,
    QueuePolicy,
    InputSelection,
    Operator,
    Plan,
    RunTermination,
    Sample,
    SampleMetadata,
)
from holoscan.diagnostics import Diagnostic, DiagnosticError
from holoscan.execution import DeviceId
from holoscan.lifecycle import LifecycleAccessPolicy, LifecycleControlHandle, LifecycleOperatorRef
from holoscan.registry import OperatorType
from holoscan.tensor import DLDataType, DLDevice, InputSpec, MemoryConversionPolicy, OutputSpec

class FrameRelease(Enum):
    AUTOMATIC = 0
    MANUAL = 1

class ClockTestOutcome(Enum):
    NOT_REQUESTED = 0
    RETIRED = 1
    SUPPRESSED = 2
    FAILED = 3

class TestRunOptions:
    """Configure one finite attempt.

    Values are copied for the run; invalid options reject when the attempt starts.
    """

    __test__: ClassVar[bool]
    def __init__(
        self,
        *,
        timeout_ms: int = 30000,
        expect_stop: bool = False,
        frames: FrameRelease = FrameRelease.AUTOMATIC,
    ) -> None: ...
    timeout_ms: int
    """Positive steady-clock watchdog in integer milliseconds; defaults to 30000.

    Expiry requests cooperative stop and fails the test, but cannot interrupt a blocked
    callback.
    """
    expect_stop: bool
    """Accept clean cooperative stop as expected.

    Errors, watchdog expiry and missing controlled-clock/retention completion proof still fail.
    """
    frames: FrameRelease
    """Select automatic or manual finite frame release.

    Manual release requires run_async and grouped input staging.
    """

class ControlledClockRun:
    """Observe one positive period-boundary tick with a shared clock initially frozen at zero.

    The startup tick is real. This restricted profile is not an operator stepping API.
    """
    def __init__(
        self, *, tick_ns: int, observation_capacity: int = 1024, timeout_ms: int = 30000
    ) -> None: ...
    tick_ns: int
    """Positive tick coordinate in integer nanoseconds, exactly on a boundary of the operator clock
    period.
    """
    observation_capacity: int
    """Positive bound on native observation records and total publication attempts; defaults to
    1024.
    """
    timeout_ms: int
    """Positive steady-clock cooperative watchdog in integer milliseconds, independent of synthetic
    time; defaults to 30000.
    """

class Retain:
    """Request native Tensor backing without a host copy.

    Admission requires a finite source with one output and proven capacity and completion;
    unsupported geometries reject.
    """
    def __init__(self, *, max_records: int = 1024) -> None: ...
    max_records: int
    """Positive finite retained-record limit; defaults to 1024.

    Overflow fails capture without truncating output.
    """

@dataclass(frozen=True)
class TestSample:
    """One payload with an optional integer nanosecond capture coordinate."""

    __test__: ClassVar[bool]
    data: object
    capture_timestamp_ns: int | None = None
    def __post_init__(self) -> None: ...

def result_of(error: BaseException) -> TestResult | None:
    """Return the joined test result directly attached to an exception, if any.

    Preserve the original result's identity and partial outputs. Unrelated
    exceptions and attachments return None; causes are not searched. Inspection
    never executes or joins an operator."""

def sample_at(data: object, *, capture_timestamp_ns: int) -> TestSample:
    """Create one timestamped staging envelope.

    Coordinates are signed 64-bit integer nanoseconds in the fixture clock domain; they do not
    advance time or schedule compute. Reject booleans, non-integers and out-of-range values.
    """

class TensorSnapshot:
    """Own contiguous host content and original Tensor metadata independently of runtime pools.

    This inspection record is not a Tensor for publication.
    """
    @property
    def bytes(self) -> bytes:
        """Return a Python bytes copy of contiguous logical Tensor content.

        Source strides and offset describe the original allocation, not this host buffer.
        """
    @property
    def shape(self) -> list[int]:
        """Return the original logical dimension extents."""
    @property
    def dtype(self) -> DLDataType:
        """Return the original DLPack element representation."""
    @property
    def source_device(self) -> DLDevice:
        """Return the original DLPack device, which can differ from the snapshot host storage."""
    @property
    def source_byte_strides(self) -> list[int]:
        """Return the original layout strides in bytes, including explicit contiguous strides."""
    @property
    def source_byte_offset(self) -> int:
        """Return the logical Tensor byte offset within its original source allocation."""
    @property
    def sample_metadata(self) -> SampleMetadata:
        """Return semantic sample metadata copied when the output was collected."""

class SchemaSnapshot:
    """Own copied schema metadata and independent host snapshots of every present admitted Tensor
    field.

    Absent optional fields retain their absence.
    """
    @property
    def metadata(self) -> object:
        """Return copied schema metadata with Tensor members unset.

        Use tensor() for field content; this metadata is not a complete payload to republish.
        """
    @property
    def sample_metadata(self) -> SampleMetadata:
        """Return semantic sample metadata copied when the schema output was collected."""
    @property
    def tensor_fields(self) -> list[str]:
        """Return all admitted Tensor field names, including absent optional fields."""
    def has_tensor(self, name: str) -> bool:
        """Return whether a known Tensor field was present.

        Raise ValueError for an unknown field.
        """
    def tensor(self, name: str) -> TensorSnapshot:
        """Return the independent host snapshot of a present Tensor field.

        Raise ValueError for an absent or unknown field.
        """

class CaptureFailure:
    """Own a failed capture's sample metadata and native error without retaining the payload or a
    finite read claim.
    """
    @property
    def metadata(self) -> SampleMetadata:
        """Return native sample metadata associated with the failed capture."""
    @property
    def error(self) -> DiagnosticError:
        """Return the native error code and detail explaining why capture failed."""

class TestResult:
    """Own joined diagnostics and captures independently of the harness.

    Inspection never executes work. Python-object identity is retained; snapshots own host
    content. Retaining the plan can retain staged Tensor leases.
    """

    __test__: ClassVar[bool]
    @property
    def ok(self) -> bool:
        """Report whether admission, capture and expected completion succeeded.

        Watchdog expiry and capture failures fail the test; a clean stopped run needs
        expect_stop or the native controlled-clock/retention completion proof.
        """
    @property
    def termination(self) -> RunTermination:
        """Return actual native run termination, independently of whether the test passed."""
    @property
    def plan(self) -> Plan:
        """Return the compiled plan, including rejected-admission diagnostics.

        Inspection does not execute it.
        """
    @property
    def diagnostics(self) -> list[Diagnostic]:
        """Return a copied list of compile and runtime diagnostics retained after join."""
    @property
    def clock_outcome(self) -> ClockTestOutcome:
        """Return the controlled tick retirement or suppression outcome; ordinary runs return
        NOT_REQUESTED.
        """
    def describe_failure(self) -> str:
        """Describe rejected admission, termination, capture errors and watchdog expiry; return an
        empty string on success.
        """
    def clock_selection(self, name: str) -> InputSelection:
        """Return the named input selection at the completed controlled tick.

        Reject ordinary runs, incomplete observations and unknown inputs.
        """
    def capture_failure(self, name: str) -> CaptureFailure | None:
        """Return the failed capture error and sample metadata, or None if collection did not fail.

        Reject unknown or uncaptured outputs. The failed payload is not retained.
        """
    def values(self, name: str) -> list[object]:
        """Return a new list of value or native-retention captures in collection order, possibly
        empty.

        Python objects are not deep-copied. Raise ValueError for unknown, unselected or snapshot
        outputs.
        """
    def value(self, name: str) -> object:
        """Return the sole value or native-retention payload.

        Raise ValueError unless exactly one record exists, or for unknown, unselected or
        snapshot outputs.
        """
    def samples(self, name: str) -> list[Sample]:
        """Return value or native-retention captures with native sample metadata, preserving
        Python-object identity.

        Raise ValueError for unknown, unselected or snapshot outputs.
        """
    def snapshots(self, name: str) -> list[TensorSnapshot | SchemaSnapshot]:
        """Return independent host Tensor or schema snapshots in collection order, possibly empty.

        Raise ValueError for unknown, unselected or value-capture outputs.
        """
    def snapshot(self, name: str) -> TensorSnapshot | SchemaSnapshot:
        """Return the sole Tensor or schema snapshot.

        Raise ValueError unless exactly one record exists, or for unknown, unselected or value-
        capture outputs.
        """

@dataclass(frozen=True, kw_only=True)
class TestLifecycleOptions:
    """Immutable tested-operator restart permissions, independent and denied by default."""

    __test__: ClassVar[bool]
    allow_self_restart: bool = False
    """Permit the tested operator to request its own restart."""
    allow_external_restart: bool = False
    """Permit external control subject to the native runtime access policy."""
    def __post_init__(self) -> None: ...

@type_check_only
class _NativeRunningOwner:
    """Opaque native owner created only by OperatorTestHarness.run_async."""

class RunningFixture:
    """Own an attempt created by OperatorTestHarness.run_async; finish joins it."""
    def __init__(self, native: _NativeRunningOwner) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def finish(self, *, check: bool = True) -> TestResult:
        """Join normally with the GIL released and return the same result on repeated calls.

        Default checking raises with .result attached; check=False returns a failed result for
        inspection. Reject joins from Runtime callbacks.
        """
    def close(self) -> None:
        """Request cooperative stop and join with the GIL released.

        This has no hard timeout and rejects Runtime callback joins. Use finish to inspect the
        resulting outcome.
        """
    def request_stop(self) -> None:
        """Request cooperative stop without joining; permitted in Runtime callbacks."""
    @property
    def is_running(self) -> bool:
        """Report whether the native attempt is still active; do not join."""
    @property
    def termination(self) -> RunTermination:
        """Return the native termination observation without joining the attempt."""
    def operator_ref(self, plan_ref: LifecycleOperatorRef | None = None) -> LifecycleOperatorRef:
        """Return the tested operator reference bound to this run, or bind the supplied plan
        reference through Core.

        Foreign and stale identities reject.
        """
    def lifecycle_control(self, policy: LifecycleAccessPolicy) -> LifecycleControlHandle:
        """Create Core lifecycle control restricted by run-bound targets and source stages in the
        policy.

        Graph permissions and transaction capacities remain required; creating control does not
        request restart.
        """
    def release_frame(self, *, timeout_ms: int | None = None) -> None:
        """Admit the next pre-staged manual frame.

        None or zero attempts release once without waiting; positive integer milliseconds allow
        bounded admission waiting. Success does not mean the frame has been computed or
        captured.
        """
    def output_count(self, port: str) -> int:
        """Observe the captured record count without exposing unjoined payloads."""
    def wait_for_output(self, port: str, minimum: int, *, timeout_ms: int) -> bool:
        """Wait up to timeout_ms milliseconds for a count; zero only polls.

        Return False when the count is not reached before timeout or completion.
        This releases the GIL while waiting and never joins or reruns the operator."""

class OperatorTestHarness:
    """Construct fresh registered C++ or Python operators through ordinary Graph services."""
    def __init__(
        self,
        operator: type[Operator] | OperatorType | str,
        *,
        params: Mapping[str, object] | None = None,
        outputs: str | Sequence[str] | None = None,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def close(self) -> None:
        """Release the builder resources.

        Existing results and independently owned running attempts remain valid.
        """
    def push(self, port: str, value: object) -> None:
        """Append one payload to an input.

        Containers remain one payload; None is data on an admitted object port. Reject mixing
        grouped and independent producers for an input.
        """
    def push_many(self, port: str, values: Iterable[object]) -> None:
        """Atomically append an explicit batch in caller order.

        Reject strings, bytes, bytearrays and mappings as ambiguous batches. Conversion failure
        leaves pending input unchanged.
        """
    def push_samples(self, port: str, samples: Iterable[TestSample]) -> None:
        """Atomically append TestSample envelopes in caller order.

        Capture coordinates do not sort inputs or advance time. Reject non-sample entries and
        invalid timestamps before staging changes.
        """
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
    def capture(self, port: str, *, capacity: int = 1024) -> None:
        """Replace the selected output policy with value capture and a positive finite record
        limit.

        Tensor-bearing outputs require capture_snapshot or explicit capture_native. Overflow
        fails the run without silent truncation; Python objects retain identity rather than an
        implicit deep copy.
        """
    def capture_snapshot(
        self, port: str, *, capacity: int = 1024, maximum_bytes: int = 67108864
    ) -> None:
        """Replace the output policy with independent host snapshots.

        capacity bounds records; maximum_bytes bounds each complete record, including all
        admitted Tensor fields of a schema. Both limits are positive. Overflow fails capture
        without publishing a partial record.
        """
    def capture_native(self, port: str, policy: Retain | None = None) -> None:
        """Opt into native Tensor retention, retaining source backing instead of copying to host.

        The finite source, single output, capacity and completion must satisfy native admission.
        None selects the default Retain record limit.
        """
    def run(self, options: TestRunOptions | None = None, *, check: bool = True) -> TestResult:
        """Execute and join a fresh attempt.

        By default, failure raises the native HoloscanError subclass with .result attached;
        check=False returns that result. Invalid configuration can still raise before a result
        exists. Joining releases the GIL and rejects Runtime callback contexts.
        """
    def run_controlled_clock(
        self, options: ControlledClockRun, *, check: bool = True
    ) -> TestResult:
        """Run and join the restricted native clock-observation profile.

        Requires one sample per scalar input, ordinary OnClock/LatestWithin/Skip semantics and
        capture of every output. Unsupported configurations reject before staged input is
        consumed. Checked failures expose .result.
        """
    def set_params(self, params: Mapping[str, object]) -> None:
        """Replace parameters atomically after validating pending inputs, capture selectors and
        overrides.

        Failed construction or validation preserves prior state. Automatic outputs follow new
        declarations; explicit selections must remain valid.
        """
    def set_compile_options(self, options: CompileOptions) -> None:
        """Copy compile options for subsequent fresh attempts.

        Later caller mutation does not change this copy. Native application order and active-
        attempt mutation rejection remain authoritative.
        """
    def set_lifecycle_options(self, options: TestLifecycleOptions | None) -> None:
        """Copy replacement permissions; None denies both.

        Reject active-attempt changes atomically.
        """
    def clear_lifecycle_options(self) -> None:
        """Clear both permissions while preserving staging and compile options."""
    def run_async(self, options: TestRunOptions | None = None) -> RunningFixture:
        """Start a fresh attempt owning resources independently of the builder.

        Inputs must already be staged; payloads become available only in joined finish results.
        One active attempt per harness is permitted.
        """
    def set_input_queue_depth(self, port: str, depth: int | None) -> None:
        """Replace tested-operator input history depth, distinct from edge queue capacity.

        None clears the override. Native range/direction checks and active-attempt rejection
        preserve previous configuration on failure.
        """
    def clear_input_queue_depth(self, port: str) -> None:
        """Remove this port's input queue depth override."""
    def set_input_flow(
        self,
        port: str,
        *,
        queue_policy: QueuePolicy = ...,
        queue_depth: int | None = None,
        topic_name: str | None = None,
        memory_conversion: MemoryConversionPolicy | None = None,
        max_inflight_samples: int | None = None,
    ) -> None:
        """Replace the input edge options using the canonical Graph keyword rules.

        Omitted/None fields take Graph defaults. With no keywords, set default
        options; use clear_input_flow to restore native fixture derivation.
        Values are copied. Invalid options and active attempts leave prior state intact."""
    def clear_input_flow(self, port: str) -> None:
        """Remove the input edge override and restore native fixture derivation."""
    def set_output_flow(
        self,
        port: str,
        *,
        queue_policy: QueuePolicy = ...,
        queue_depth: int | None = None,
        topic_name: str | None = None,
        memory_conversion: MemoryConversionPolicy | None = None,
        max_inflight_samples: int | None = None,
    ) -> None:
        """Replace the output edge options using the canonical Graph keyword rules.

        Omitted/None fields take Graph defaults. With no keywords, set default
        options; use clear_output_flow to restore native fixture derivation.
        Values are copied. Invalid options and active attempts leave prior state intact."""
    def clear_output_flow(self, port: str) -> None:
        """Remove the output edge override and restore native fixture derivation."""
    def set_input_tensor_spec(self, port: str, spec: InputSpec | None) -> None:
        """Copy or clear an operator Tensor input contract."""
    def clear_input_tensor_spec(self, port: str) -> None:
        """Remove this port's input tensor spec override."""
    def set_output_tensor_spec(self, port: str, spec: OutputSpec | None) -> None:
        """Copy or clear an operator Tensor output contract."""
    def clear_output_tensor_spec(self, port: str) -> None:
        """Remove this port's output tensor spec override."""
    def set_input_source_tensor_spec(self, port: str, spec: OutputSpec | None) -> None:
        """Copy or clear a staged producer Tensor promise."""
    def clear_input_source_tensor_spec(self, port: str) -> None:
        """Remove this port's input source tensor spec override."""
    def set_input_source_device(self, port: str, device: DeviceId | None) -> None:
        """Copy or clear staged-source CUDA placement."""
    def clear_input_source_device(self, port: str) -> None:
        """Remove this port's input source device override."""
    def set_output_device(self, port: str, device: DeviceId | None) -> None:
        """Copy or clear tested-operator CUDA output placement."""
    def clear_output_device(self, port: str) -> None:
        """Remove this port's output device override."""

__all__: list[str] = [
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
