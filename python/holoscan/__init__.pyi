# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Holoscan SDK Python authoring API."""

from holoscan.clocks import AffineClockConversion as _clocks_AffineClockConversion
from holoscan.clocks import ClockKey as _clocks_ClockKey
from holoscan.clocks import ClockRef as _clocks_ClockRef
from holoscan.clocks import ClockType as _clocks_ClockType
from holoscan.diagnostics import Diagnostic as _diagnostics_Diagnostic
from holoscan.execution import AutomaticQueueSizingOptions as _execution_AutomaticQueueSizingOptions
from holoscan.execution import CudaComputeStreamPolicy as _execution_CudaComputeStreamPolicy
from holoscan.execution import CudaExecutionCompileOptions as _execution_CudaExecutionCompileOptions
from holoscan.execution import DeploymentBinding as _execution_DeploymentBinding
from holoscan.execution import ExecutionPartitions as _execution_ExecutionPartitions
from holoscan.execution import PartitionHandle as _execution_PartitionHandle
from holoscan.execution import Profile as _execution_Profile
from holoscan.execution import RateRatio as _execution_RateRatio
from holoscan.execution import RegionHandle as _execution_RegionHandle
from holoscan.execution import SerializationDomainSplitPolicy as _execution_SerializationDomainSplitPolicy
from holoscan.execution import TensorByteSpanDerivation as _execution_TensorByteSpanDerivation
from holoscan.execution import TensorMemoryKindDerivation as _execution_TensorMemoryKindDerivation
from holoscan.lifecycle import GraphLifecycleBuilder as _lifecycle_GraphLifecycleBuilder
from holoscan.lifecycle import LifecycleAccessPolicy as _lifecycle_LifecycleAccessPolicy
from holoscan.lifecycle import LifecycleControlHandle as _lifecycle_LifecycleControlHandle
from holoscan.lifecycle import LifecycleTransactionOptions as _lifecycle_LifecycleTransactionOptions
from holoscan.lifecycle import OperatorLifecycleBuilder as _lifecycle_OperatorLifecycleBuilder
from holoscan.operators import OperatorHandle as _operators_OperatorHandle
from holoscan.params import ParamAccessPolicy as _params_ParamAccessPolicy
from holoscan.params import ParamControlHandle as _params_ParamControlHandle
from holoscan.params import ParameterCompileOptions as _params_ParameterCompileOptions
from holoscan.payload import PayloadType as _payload_PayloadType
from holoscan.registry import GraphType as _registry_GraphType
from holoscan.registry import OperatorType as _registry_OperatorType
from holoscan.tensor import DLDataType as _tensor_DLDataType
from holoscan.tensor import DLDevice as _tensor_DLDevice
from holoscan.tensor import MemoryConversionPolicy as _tensor_MemoryConversionPolicy
from holoscan.tensor import MemoryKindExpression as _tensor_MemoryKindExpression
from holoscan.tensor import ByteSpanExpression as _tensor_ByteSpanExpression
from holoscan.tensor import InputSpec as _tensor_InputSpec
from holoscan.tensor import OutputSpec as _tensor_OutputSpec
from holoscan.tensor import TensorBoundaryContract as _tensor_TensorBoundaryContract
from holoscan.tensor import LoanRequest as _tensor_LoanRequest
from holoscan.tensor import TensorLoan as _tensor_TensorLoan
from holoscan.tensor import TensorSchemaLoan as _tensor_TensorSchemaLoan
from holoscan.tensor import TensorProjection as _tensor_TensorProjection
from holoscan.time import Duration as _time_Duration
from holoscan.time import TagTime as _time_TagTime
from holoscan.time import TimePoint as _time_TimePoint
from holoscan.topics import ClockHealthOptions as _topics_ClockHealthOptions
from holoscan.topics import EndpointBounds as _topics_EndpointBounds
from holoscan.topics import EndpointCatalog as _topics_EndpointCatalog
from holoscan.topics import ExportedTopic as _topics_ExportedTopic
from holoscan.topics import ImportedTopic as _topics_ImportedTopic
from holoscan.topics import TopicAttachPolicy as _topics_TopicAttachPolicy
from holoscan.topics import TopicOptions as _topics_TopicOptions

import enum
import types
from collections.abc import Iterator, Sequence
from datetime import timedelta
from typing import Any, Self, overload

from . import clocks as clocks
from . import channel as channel
from . import config as config
from . import diagnostics as diagnostics
from . import errors as errors
from . import execution as execution
from . import graphs as graphs
from . import lifecycle as lifecycle
from . import operators as operators
from . import params as params
from . import payload as payload
from . import registry as registry
from . import schema as schema
from . import sensor_io as sensor_io
from . import tensor as tensor
from . import time as time
from . import topics as topics
from holoscan.errors import HoloscanError as HoloscanError


class Operator:
    """Base class whose subclasses define hosted-operator lifecycle callbacks."""

    def __init__(self) -> None: ...
class Param[T]:
    """Declare and validate one class-level hosted-operator parameter."""

    def __init__(self, default: T, *, description: str = '', range: tuple[T, T] | None = None, allowed: list[T] | tuple[T, ...] | None = None, recompile_required: bool = False, max_elements: int | None = None) -> None: ...

    def __set_name__(self, owner: type[object], name: str) -> None: ...

    @property
    def default(self) -> T: ...

    @property
    def description(self) -> str: ...

    @property
    def range(self) -> tuple[T, T] | None: ...

    @property
    def allowed(self) -> list[T] | tuple[T, ...] | None: ...

    @property
    def recompile_required(self) -> bool: ...

    @property
    def max_elements(self) -> int | None: ...

    @overload
    def __get__(self, instance: None, owner: type[object], /) -> Self: ...

    @overload
    def __get__(self, instance: Operator, owner: type[Operator], /) -> T: ...

    def __set__(self, arg0: object, arg1: object, /) -> None: ...

    @classmethod
    def __class_getitem__(cls, item: object, /) -> types.GenericAlias: ...

class OperatorPort:
    """Callback-lifetime typed port bound to one hosted operator instance."""

    @property
    def name(self) -> str: ...

    @property
    def direction(self) -> PortDirection: ...

    @property
    def payload_type(self) -> _payload_PayloadType: ...

    def receive(self) -> Sample[object]: ...

    def selection(self) -> InputSelection: ...

    def selected_sample(self) -> SelectedSample[object]: ...

    def selected_batch(self) -> SelectedBatch[object]: ...

    def receive_data(self) -> object: ...

    def emit(self, data: object, *, options: EmitOptions | None = None) -> None: ...

    @overload
    def forward(self, sample: Sample[object], *, options: EmitOptions | None = None) -> None:
        """
        Forward one received sample without replacing its data.

        Every field absent from ``options`` inherits from the source sample metadata. Use
        ``emit()`` for a fused result whose absent provenance fields must remain absent.
        """

    @overload
    def forward(self, sample: Sample[object], data: object, *, options: EmitOptions | None = None) -> None:
        """
        Forward one received sample while replacing its data.

        Every field absent from ``options`` inherits from the source sample metadata. Use
        ``emit()`` for a fused result whose absent provenance fields must remain absent.
        """

    @overload
    def forward(self, sample: SelectedSample[object], *, options: EmitOptions | None = None) -> None:
        """
        Forward an ordinary generated schema from this Operator's scalar selection.

        Inherit absent options from source metadata. Encode synchronously without Python data
        materialization or a native value copy. Require the creating callback and thread.
        Tensor descriptors and other payload kinds are unsupported.
        """

    def allocate_tensor(self, *, shape: Sequence[int], dtype: object, strides: Sequence[int] | None = None, options: EmitOptions | None = None) -> _tensor_TensorLoan: ...

    def allocate_tensors(self, descriptor: object, requests: dict[str, _tensor_LoanRequest], *, options: EmitOptions | None = None) -> _tensor_TensorSchemaLoan:
        """
        Allocate and publish every Tensor field of a descriptor-backed schema.

        The requests mapping must contain every canonical field name. The returned
        context manager exposes provider-owned Tensors through its read-only
        ``tensors`` mapping and publishes them atomically with descriptor metadata
        on successful exit. This API supports schemas with one or multiple Tensor
        fields.
        """

    def expects_tensor(self, spec: _tensor_InputSpec) -> OperatorPort: ...

    def accepts_projected_tensor(self) -> OperatorPort: ...

    def expected_rate(self, numerator_hz: int, denominator_hz: int = 1) -> OperatorPort: ...

    def nominal_rate(self, numerator_hz: int, denominator_hz: int = 1) -> OperatorPort: ...

    @overload
    def follows(self, input: OperatorPort, *, numerator: int = 1, denominator: int = 1) -> OperatorPort: ...

    @overload
    def follows(self, input: OperatorPort, ratio: _execution_RateRatio) -> OperatorPort: ...

    def produces_tensor(self, spec: _tensor_OutputSpec, byte_span: _tensor_ByteSpanExpression | None = None, *, memory_kind: _tensor_MemoryKindExpression | None = None, field: str | None = None) -> OperatorPort: ...

class OperatorSpec:
    """Cold-path port declaration builder passed to ``Operator.setup``."""

    def input(self, port: OperatorPort, *, queue_depth: int | None = None, optional: bool = False) -> OperatorPort: ...

    def output(self, port: OperatorPort, *, max_emits_per_compute: int | None = None) -> OperatorPort: ...

    def lifecycle(self) -> _lifecycle_OperatorLifecycleBuilder: ...

class ExecutionContext:
    """Callback-lifetime execution context passed to ``Operator.compute``."""

    @property
    def operator_path(self) -> str: ...

    def stream(self) -> object: ...

    def clock(self) -> _clocks_ClockRef: ...

    @property
    def tag_time(self) -> _time_TagTime: ...

    @property
    def activation_time(self) -> _time_TimePoint: ...

    @property
    def trace_id(self) -> int: ...

    @property
    def activation(self) -> ActivationStatus: ...

    @property
    def stop_requested(self) -> bool: ...

    def request_completion(self) -> None: ...
    def request_stop(self) -> None: ...

class ActivationReason(enum.Enum):
    NOMINAL = 0
    MAX_START_DELAY_EXCEEDED = 1
    INVARIANT_VIOLATED = 2

class ActivationStatus:
    def __init__(self, *, reason: ActivationReason = ..., policy: OnViolation = ...) -> None: ...

    @property
    def reason(self) -> ActivationReason: ...

    @property
    def policy(self) -> OnViolation: ...

    def __eq__(self, arg0: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class SampleMetadataVersion:
    @property
    def major(self) -> int: ...

    @property
    def minor(self) -> int: ...

class SampleFlags(enum.IntFlag):
    NONE = 0
    INVALID = 1
    DEGRADED = 2

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class EmitOptions:
    """
    Publication metadata overrides.

    Fields omitted from user-created options inherit from the source when passed to
    ``Output.forward()``. Options returned by ``ProvenanceAccumulator`` are authoritative and can
    only be passed to ``Output.emit()``.
    """

    def __init__(self, *, capture_time: _time_TimePoint | None = None, frame_id: int | None = None, trace_id: int | None = None, flags: SampleFlags | None = None) -> None: ...

    @property
    def capture_time(self) -> _time_TimePoint | None: ...

    @property
    def frame_id(self) -> int | None: ...

    @property
    def trace_id(self) -> int | None: ...

    @property
    def flags(self) -> SampleFlags | None: ...

    @property
    def authoritative(self) -> bool:
        """Whether absent fields are authoritative rather than forwarding inheritance requests."""

    def __eq__(self, arg0: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class PayloadKind(enum.Enum):
    INLINE_TYPED = 0
    SERIALIZED_BYTES = 1
    DESCRIPTOR_BACKED = 2
    COMPATIBILITY_PAYLOAD = 3
    CONTROL_EMPTY = 4

class SampleMetadata:
    @property
    def version(self) -> SampleMetadataVersion: ...

    @property
    def sequence(self) -> int: ...

    @property
    def capture_timestamp_ns(self) -> int: ...

    @property
    def capture_time(self) -> _time_TimePoint: ...

    @property
    def source_clock_id(self) -> int: ...

    @property
    def publish_timestamp_ns(self) -> int: ...

    @property
    def publish_time(self) -> _time_TimePoint: ...

    @property
    def publish_clock_id(self) -> int: ...

    @property
    def trace_id(self) -> int: ...

    @property
    def frame_id(self) -> int: ...

    @property
    def byte_size(self) -> int: ...

    @property
    def payload_kind(self) -> PayloadKind: ...

    @property
    def flags(self) -> SampleFlags: ...

    @property
    def valid(self) -> bool: ...

    @property
    def degraded(self) -> bool: ...

class Sample[T]:
    @property
    def data(self) -> T: ...

    @property
    def metadata(self) -> SampleMetadata: ...

    @property
    def is_stale(self) -> bool: ...

    @property
    def is_missing(self) -> bool: ...

    @classmethod
    def __class_getitem__(cls, item: object, /) -> types.GenericAlias: ...

class SelectedSample[T]:
    """Read-only callback-scoped scalar selection with lazy owning data projection."""

    @property
    def data(self) -> T: ...

    @property
    def metadata(self) -> SampleMetadata: ...

    @property
    def is_stale(self) -> bool: ...

class SelectedBatch[T]:
    """Read-only callback-scoped view; indexed samples and sliced lists own their values."""

    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, index: int) -> Sample[T]: ...

    @overload
    def __getitem__(self, index: slice) -> list[Sample[T]]: ...

    def __iter__(self) -> Iterator[Sample[T]]: ...

class InputState(enum.Enum):
    SELECTED = 0
    REUSED = 1
    SAFE_DEFAULT = 2
    ABSENT = 3
    NOT_SELECTED = 4

class SelectionReason(enum.Enum):
    NONE = 0
    NEVER_SEEN = 1
    TOO_STALE = 2
    INSUFFICIENT_HISTORY = 3
    WINDOW_TIMEOUT = 4
    WINDOW_SAMPLE_LIMIT = 5
    WINDOW_PENDING_LIMIT = 6
    END_OF_STREAM = 7
    NOT_CONNECTED = 8
    ALIGNMENT_WINDOW_MISS = 9
    NOT_SELECTED_FOR_ACTIVATION = 10
    CAPTURE_TIME_ABSENT = 11
    CLOCK_INCOMPARABLE = 12
    CONVERSION_UNAVAILABLE = 13

class InputSelection:
    def __init__(self, *, state: InputState, reason: SelectionReason) -> None: ...

    @property
    def state(self) -> InputState: ...

    @property
    def reason(self) -> SelectionReason: ...

    def __eq__(self, arg0: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class Exact:
    def __init__(self) -> None: ...

class LatestWithin:
    def __init__(self, *, max_age: timedelta | _time_Duration) -> None: ...

    @property
    def max_age(self) -> _time_Duration: ...

class DrainAll:
    def __init__(self) -> None: ...

class BracketedWindow:
    def __init__(self, *, lookbehind: int = 1, lookahead: int = 1, max_samples: int = 0) -> None: ...

    @property
    def lookbehind(self) -> int: ...

    @property
    def lookahead(self) -> int: ...

    @property
    def max_samples(self) -> int: ...

class Skip:
    def __init__(self) -> None: ...

class SafeDefault:
    def __init__(self) -> None: ...

class ReuseLast:
    def __init__(self, *, max_age: timedelta | _time_Duration) -> None: ...

    @property
    def max_age(self) -> _time_Duration: ...

class OnAnyInput:
    def __init__(self) -> None: ...

class OnEach:
    def __init__(self, port: OperatorPort) -> None: ...

    @property
    def port(self) -> OperatorPort: ...

class OnClock:
    def __init__(self, *, period: timedelta | _time_Duration, clock: _clocks_ClockKey | None = None, start_at: _time_TimePoint | None = None) -> None: ...

    @property
    def period(self) -> _time_Duration: ...

    @property
    def clock(self) -> _clocks_ClockKey: ...

    @property
    def start_at(self) -> _time_TimePoint | None: ...

class OnAligned:
    def __init__(self, *, ports: Sequence[OperatorPort], window: timedelta | _time_Duration) -> None: ...

    @property
    def ports(self) -> Sequence[OperatorPort]: ...

    @property
    def window(self) -> _time_Duration: ...

class OnWindow:
    def __init__(self, *, anchor: OperatorPort, max_wait: timedelta | _time_Duration, wait_clock: _clocks_ClockKey | None = None, max_pending_windows: int = 1) -> None: ...

    @property
    def anchor(self) -> OperatorPort: ...

    @property
    def max_wait(self) -> _time_Duration: ...

    @property
    def wait_clock(self) -> _clocks_ClockKey: ...

    @property
    def max_pending_windows(self) -> int: ...

class OnViolation(enum.Enum):
    SAFE_DEFAULT = 0
    EMIT_WITNESS = 1
    FAIL_RUN = 2

class Contract:
    def __init__(self) -> None: ...

    @overload
    def trigger(self, trigger: OnAnyInput) -> Self: ...

    @overload
    def trigger(self, trigger: OnEach) -> Self: ...

    @overload
    def trigger(self, trigger: OnClock) -> Self: ...

    @overload
    def trigger(self, trigger: OnAligned) -> Self: ...

    @overload
    def trigger(self, trigger: OnWindow) -> Self: ...

    @overload
    def input(self, port: OperatorPort, selector: Exact) -> Self: ...

    @overload
    def input(self, port: OperatorPort, selector: LatestWithin) -> Self: ...

    @overload
    def input(self, port: OperatorPort, selector: DrainAll) -> Self: ...

    @overload
    def input(self, port: OperatorPort, selector: BracketedWindow) -> Self: ...

    @overload
    def on_missing(self, port: OperatorPort, strategy: Skip) -> Self: ...

    @overload
    def on_missing(self, port: OperatorPort, strategy: SafeDefault) -> Self: ...

    @overload
    def on_missing(self, port: OperatorPort, strategy: ReuseLast) -> Self: ...

    def max_start_delay(self, budget: timedelta | _time_Duration) -> Self: ...

    def complete_after(self, limit: int) -> Self: ...

    def on_violation(self, policy: OnViolation) -> Self: ...

class Graph:
    """Own a mutable Holoscan graph authoring root."""

    def __init__(self, name: str = '', *, default_clock: _clocks_ClockType | str | None = None) -> None: ...

    @overload
    def op[HandleT](self, type_or_name: _registry_OperatorType[HandleT], instance_name: str, **params: object) -> HandleT: ...

    @overload
    def op(self, type_or_name: type[Operator], instance_name: str, **params: object) -> _operators_OperatorHandle: ...

    @overload
    def op(self, type_or_name: str, instance_name: str, **params: Any) -> _operators_OperatorHandle:  # noqa: ANN401 - dynamic registry name
        """
        Add a registered C++ operator by registry name.

        Keyword arguments are converted and validated from the operator's C++ parameter
        schema. Duration fields accept ``datetime.timedelta`` or
        ``holoscan.time.ns``/``us``/``ms`` values; bare numeric durations are rejected.
        """

    @overload
    def add_flow(self, source: Port, target: Port, *, queue_policy: QueuePolicy = ..., queue_depth: int | None = None, topic_name: str | None = None, memory_conversion: _tensor_MemoryConversionPolicy | None = None, max_inflight_samples: int | None = None) -> Flow: ...

    @overload
    def add_flow(self, source: _tensor_TensorProjection, target: Port, *, queue_policy: QueuePolicy = ..., queue_depth: int | None = None, topic_name: str | None = None, memory_conversion: _tensor_MemoryConversionPolicy | None = None, max_inflight_samples: int | None = None) -> Flow: ...

    @overload
    def import_topic(self, uri: str, payload_type: _payload_PayloadType, options: _topics_TopicOptions) -> _topics_ImportedTopic: ...

    @overload
    def import_topic(self, uri: str, payload_type: _payload_PayloadType, *, endpoint_name: str | None = None, memory_conversion: _tensor_MemoryConversionPolicy | None = None, schema: _payload_PayloadType | None = None, capture_time: _clocks_ClockKey | None = None, clock_health: _topics_ClockHealthOptions | None = None, attach: _topics_TopicAttachPolicy = ..., bounds: _topics_EndpointBounds = ..., tensor: _tensor_TensorBoundaryContract | None = None, tensor_fields: dict[str, _tensor_TensorBoundaryContract] = ...) -> _topics_ImportedTopic: ...

    @overload
    def export_topic(self, uri: str, payload_type: _payload_PayloadType, options: _topics_TopicOptions) -> _topics_ExportedTopic: ...

    @overload
    def export_topic(self, uri: str, payload_type: _payload_PayloadType, *, endpoint_name: str | None = None, memory_conversion: _tensor_MemoryConversionPolicy | None = None, schema: _payload_PayloadType | None = None, capture_time: _clocks_ClockKey | None = None, clock_health: _topics_ClockHealthOptions | None = None, attach: _topics_TopicAttachPolicy = ..., bounds: _topics_EndpointBounds = ..., tensor: _tensor_TensorBoundaryContract | None = None, tensor_fields: dict[str, _tensor_TensorBoundaryContract] = ...) -> _topics_ExportedTopic: ...

    @overload
    def configure(self, port: Port) -> PortConfigurator: ...

    @overload
    def configure(self, flow: Flow) -> FlowConfigurator: ...

    def add_clock(self, type_or_name: _clocks_ClockType | str, name: str, *, initial_time_offset: float = ..., initial_time_scale: float = ..., use_time_since_epoch: bool = ...) -> _clocks_ClockKey: ...

    def set_default_clock(self, key: _clocks_ClockKey) -> Graph: ...

    def default_clock(self) -> _clocks_ClockKey: ...

    def import_clock(self, name: str) -> _clocks_ClockKey: ...

    def add_clock_conversion(self, from_clock: _clocks_ClockKey, to_clock: _clocks_ClockKey, conversion: _clocks_AffineClockConversion) -> Graph: ...

    @overload
    def embed[HandleT](self, graph_type: _registry_GraphType[HandleT], instance_name: str, **params: object) -> HandleT: ...

    @overload
    def embed(self, graph_type: type[Graph], instance_name: str, **params: object) -> GraphHandle: ...

    @overload
    def embed(self, graph_type: str, instance_name: str, **params: object) -> GraphHandle:
        """
        Embed a reusable Graph from a Python ``Graph`` subclass or a registered C++ Graph type.

        Registered types accept a fully qualified registry name or a proxy from
        ``holoscan.graphs``/``holoscan.registry.resolve``. Keyword arguments are converted and validated
        through the C++ constructor-parameter schema.
        """

    def export_input(self, public_port: Port, internal_port: Port) -> None: ...

    def export_output(self, public_port: Port, internal_port: Port) -> None: ...

    def bind_clock(self, child: GraphHandle, child_import: _clocks_ClockKey, parent_clock: _clocks_ClockKey) -> Graph: ...

    def region(self, name: str) -> _execution_RegionHandle: ...

    def partition(self, name: str) -> _execution_PartitionHandle: ...

    def auto_partition_remaining(self) -> None: ...

    def lifecycle(self) -> _lifecycle_GraphLifecycleBuilder: ...

    def compose(self) -> None:
        """
        Author the reusable Graph body.

        Override this method in a Graph subclass. Core invokes it once for an embedded
        Graph, and ``holoscan.compile`` invokes it once for a top-level Python Graph.
        Authoring from the subclass constructor is unsupported; store constructor
        arguments there and create Graph objects in this method.
        """

    @property
    def name(self) -> str: ...

    def __repr__(self) -> str: ...

class GraphHandle:
    """
    A weak, read-only handle to a reusable Graph instance embedded in a parent Graph.
    """

    @property
    def name(self) -> str: ...

    @property
    def type_name(self) -> str: ...

    @property
    def path(self, /) -> tuple[str, ...] | None: ...

    @property
    def graph_valid(self) -> bool: ...

    @property
    def ports(self, /) -> dict[str, Port]: ...

    def imported_clock(self, name: str) -> _clocks_ClockKey: ...

    def __getitem__(self, name: str) -> Port: ...

    def __getattr__(self, name: str) -> object: ...

    def __repr__(self) -> str: ...

class Input:
    """Declare a typed input on a reusable Graph subclass."""

    def __init__(self, name: str, payload_type: _payload_PayloadType | type[object], *, as_: object | None = None) -> None: ...

    @property
    def name(self) -> str: ...

    @property
    def payload_type(self) -> _payload_PayloadType: ...

    @property
    def as_(self) -> object | None: ...

    @overload
    def __get__(self, instance: None, owner: type[object], /) -> Self: ...

    @overload
    def __get__(self, instance: Graph, owner: type[Graph], /) -> Port: ...

    @overload
    def __get__(self, instance: Operator, owner: type[Operator], /) -> OperatorPort: ...

    def __set__(self, arg0: object, arg1: object, /) -> None: ...

class Output:
    """Declare a typed output on a reusable Graph subclass."""

    def __init__(self, name: str, payload_type: _payload_PayloadType | type[object]) -> None: ...

    @property
    def name(self) -> str: ...

    @property
    def payload_type(self) -> _payload_PayloadType: ...

    @overload
    def __get__(self, instance: None, owner: type[object], /) -> Self: ...

    @overload
    def __get__(self, instance: Graph, owner: type[Graph], /) -> Port: ...

    @overload
    def __get__(self, instance: Operator, owner: type[Operator], /) -> OperatorPort: ...

    def __set__(self, arg0: object, arg1: object, /) -> None: ...

class Port:
    """
    A weak, lazily resolved view of one Graph-owned operator port.

    Index an OperatorHandle to access arbitrary or Python-keyword port names. Attribute
    access is available as sugar for ordinary identifiers.
    """

    @property
    def name(self) -> str: ...

    @property
    def direction(self) -> PortDirection: ...

    @property
    def payload_type(self) -> _payload_PayloadType: ...

    @property
    def graph_valid(self) -> bool: ...

    def project(self, field_name: str) -> _tensor_TensorProjection:
        """
        Select one named Tensor field from a descriptor-backed schema output.

        The target must be a Tensor input that opts in with
        ``accepts_projected_tensor()``. Projection currently supports one direct,
        same-process subscriber and does not support projected fanout or simultaneous
        delivery of the complete schema and projected Tensor.
        """

    def __repr__(self) -> str: ...

class Flow:
    """
    A weak handle to one authored Graph flow.

    ``tensor_projection`` retains the selected field view for a schema-to-Tensor
    flow and is ``None`` for an ordinary typed flow. ``source`` remains the original
    authored output port in either case.
    """

    @property
    def source(self) -> Port: ...

    @property
    def target(self) -> Port: ...

    @property
    def tensor_projection(self) -> _tensor_TensorProjection | None: ...

    @property
    def graph_valid(self) -> bool: ...

    def __repr__(self) -> str: ...

class PortConfigurator:
    """Configure one live Graph port through the Core authoring API."""

    def queue_depth(self, depth: int) -> PortConfigurator: ...

    def max_emits_per_compute(self, count: int = 1) -> PortConfigurator: ...

    def late_joiner_history_depth(self, depth: int) -> PortConfigurator: ...

    def diagnostic_alias(self, alias: str) -> PortConfigurator: ...

class FlowConfigurator:
    """Configure one live Graph flow through the Core authoring API."""

    def queue_policy(self, policy: QueuePolicy) -> FlowConfigurator: ...

    def queue_depth(self, depth: int) -> FlowConfigurator: ...

    def topic_name(self, name: str) -> FlowConfigurator: ...

    def memory_conversion(self, policy: _tensor_MemoryConversionPolicy) -> FlowConfigurator: ...

    def max_inflight_samples(self, count: int) -> FlowConfigurator: ...

class Tensor:
    """
    An n-dimensional array, interoperable with NumPy, CuPy, and PyTorch via DLPack and the array
    interface protocols. Wraps a DLPack managed tensor; instances are cheap handles that share the
    underlying memory and its access authority. Zero-copy imports are read-only; writable consumers
    must use an explicit allocation-and-copy path.
    """

    @property
    def ndim(self) -> int:
        """Number of dimensions (int)."""

    @property
    def shape(self, /) -> tuple[int, ...]:
        """Tuple of the tensor's dimensions."""

    @property
    def strides(self, /) -> tuple[int, ...]:
        """Tuple of the tensor's strides, in bytes."""

    @property
    def size(self) -> int:
        """Total number of elements (int)."""

    @property
    def itemsize(self) -> int:
        """Size of a single element, in bytes (int)."""

    @property
    def nbytes(self) -> int:
        """Total size of the elements, in bytes (int)."""

    @property
    def byte_offset(self) -> int:
        """
        Offset, in bytes, from the start of the underlying allocation to the tensor data (int).
        """

    @property
    def device(self) -> _tensor_DLDevice:
        """The ``DLDevice`` the tensor data resides on."""

    @property
    def dtype(self) -> _tensor_DLDataType:
        """The ``DLDataType`` of the tensor elements."""

    @property
    def data(self) -> int:
        """Integer address of the tensor's data pointer."""

    def is_contiguous(self) -> bool:
        """
        Whether the tensor has a contiguous, row-major (C-contiguous) memory layout.

        Returns
        -------
        bool
        """

    @property
    def is_read_only(self) -> bool:
        """
        Whether the tensor's effective backing authority is permanently read-only.
        """

    @property
    def __array_interface__(self, /) -> dict[str, object]:
        """
        The NumPy ``__array_interface__`` dict. Present only for host (CPU / pinned-host) tensors;
        accessing it on a device tensor raises ``AttributeError``. The dictionary carries no stream, so a
        readiness-aware tensor first waits on the host for its producer's work, such as a pinned-host
        tensor written on a CUDA stream. A tensor written on the host returns at once.
        """

    @property
    def __cuda_array_interface__(self, /) -> dict[str, object]:
        """
        The ``__cuda_array_interface__`` dict. Present only for device (CUDA / managed) tensors; accessing
        it on a host tensor raises ``AttributeError``. Readiness-aware tensors are synchronized to the
        legacy default stream before export because the returned dictionary reports ``stream=1``.
        """

    def __dlpack__(self, *, stream: object | None = None, max_version: tuple[int, int] | None = None, dl_device: tuple[int, int] | None = None, copy: bool | None = None) -> object:
        """
        Export the tensor as a DLPack capsule (the array-API ``__dlpack__`` protocol).

        Parameters
        ----------
        stream : int or None, optional
            Consumer stream for synchronization. ``None``/``1`` = legacy default stream, ``2`` = per-thread
            default, ``-1`` = no synchronization, a stream pointer as a positive integer, or an object with
            callable ``__cuda_stream__()`` returning ``(0, stream_pointer)``. Ignored for host tensors.
        max_version : tuple[int, int] or None, optional
            Maximum supported DLPack ``(major, minor)`` version; a major >= 1 selects the versioned capsule.
        dl_device : tuple[int, int] or None, optional
            Requested ``(device_type, device_id)``. A different device requests an explicit copy.
        copy : bool or None, optional
            Request an owning contiguous copy before export. ``copy=False`` rejects a different
            ``dl_device`` request.

        Returns
        -------
        PyCapsule

        Raises
        ------
        BufferError
            If the tensor is read-only and ``max_version`` does not select a versioned capsule. Legacy
            DLPack cannot carry read-only authority, so Holoscan rejects rather than widening access.

        Notes
        -----
        For imported CUDA storage, ``stream=2`` and ``stream=-1`` use conservative device retirement.
        ``stream=-1`` still skips producer-readiness ordering; the caller must establish it separately.
        Prefer explicit streams for precise retirement.
        """

    def __dlpack_device__(self) -> tuple[int, int]:
        """
        Return the tensor's ``(device_type, device_id)`` tuple (the array-API ``__dlpack_device__``
        protocol). Both values are ints. CUDA device tensors preserve the owning CUDA ordinal; CPU,
        pinned-host, and managed tensors report ``device_id == 0``.

        Returns
        -------
        tuple[int, int]
        """

    @staticmethod
    def as_tensor(obj: object, *, stream: object | None = None) -> Tensor:
        """
        Create a Tensor from a tensor-like object (zero-copy).

        Parameters
        ----------
        obj : array-like
            A NumPy array, CuPy array, PyTorch tensor, or any object supporting the DLPack protocol or an
            array interface.
        stream : int or object, optional
            CUDA import stream, as an integer or an object exposing ``__cuda_stream__()``.
            Defaults to the activation stream inside ``compute()`` and the legacy default stream outside
            a callback. DLPack producers negotiate directly on this stream. Array-interface and C exchange
            producers retain their advertised readiness and order an explicitly supplied stream after it.
            Integer ``0`` is invalid; use ``1`` or ``cupy.cuda.Stream.null`` for the default stream.

        Returns
        -------
        holoscan.Tensor

        Notes
        -----
        Use ``with ctx.stream():`` for CuPy work in ``compute()``. Prefer ``as_=cupy`` on inputs or
        ``cupy.from_dlpack(tensor)`` for conversion. Outside callbacks, keep external consumer streams
        alive until their exported views are released.

        Zero-copy imported storage is permanently read-only. The returned Tensor keeps its producer
        alive through queued consumer work, including release in a later callback and exception cleanup.
        Retained views may outlive the session. For work on the SDK callback stream, release a retained
        view in the operator that last used it, or synchronize that operator's stream before release
        elsewhere, including another partition or cyclic garbage collection. Inside callbacks, keep a
        stream you create alive until the callback returns, and synchronize it before releasing a view
        you used on it. Keep that stream on the operator so it survives callback-exit completion recording.
        CUDA array-interface, no-sync C exchange, and no-sync or thread-relative DLPack exports require
        conservative device completion at final release.

        Readiness is captured at import. Finish producing the array before importing it; after subsequent
        writes on other streams, import again or emit the array directly. Importing a read-only Tensor
        returns the same object; importing a writable Tensor creates a new read-only wrapper.

        The interchange protocol is selected by precedence: the DLPack C exchange API [1]_ first, then the
        Python DLPack protocol [1]_, [2]_, then ``__cuda_array_interface__`` [3]_, then
        ``__array_interface__`` [4]_. DLPack is preferred because it transfers ownership (so the imported
        Tensor keeps its data alive independently of the source), has explicit stream-synchronization
        semantics, and is the array-API-endorsed interchange path. The C exchange API avoids the per-call
        Python capsule path when a producer type exposes ``__dlpack_c_exchange_api__``.

        Examples
        --------
        With Tensor input/output ports declared, prefer the SDK stream in ``compute(ctx)``::

            import cupy as cp

            with ctx.stream():
                view = cp.from_dlpack(self.inp.receive_data())
                self.out.emit(view * 2)

        If you need a separate stream, keep it on the operator across callbacks and synchronize before
        dropping a view, including a view retained from an earlier callback::

            with ctx.stream():
                if not hasattr(self, "_cupy_stream"):
                    self._cupy_stream = cp.cuda.Stream(non_blocking=True)
                with self._cupy_stream:
                    view = cp.from_dlpack(self.inp.receive_data())
                    try:
                        result = view * 2
                    finally:
                        self._cupy_stream.synchronize()
                del view
                self.out.emit(result)

        References
        ----------
        .. [1] https://dmlc.github.io/dlpack/latest/python_spec.html
        .. [2] https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
        .. [3] https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
        .. [4] https://numpy.org/doc/stable/reference/arrays.interface.html
        """

    @staticmethod
    def from_dlpack(obj: object, *, stream: object | None = None) -> Tensor:
        """
        Create a Tensor from an object supporting the DLPack protocol [1]_, [2]_ (zero-copy).

        Parameters
        ----------
        obj : array-like
            An object exposing ``__dlpack__`` and ``__dlpack_device__``.
        stream : int or object, optional
            CUDA import stream, as an integer or an object exposing ``__cuda_stream__()``.
            Defaults to the activation stream inside ``compute()`` and the legacy default stream outside
            a callback. ``stream=-1`` is rejected because it supplies no readiness guarantee.
            Integer ``0`` is invalid; use ``1`` or ``cupy.cuda.Stream.null`` for the default stream.

        Returns
        -------
        holoscan.Tensor

        Notes
        -----
        Zero-copy imported storage is permanently read-only.

        CUDA imports negotiate ``__dlpack__(stream=...)`` on the selected stream and record readiness
        with a CUDA event. The Tensor retains the producer's storage until its views and queued consumer
        work have finished. SDK callbacks snapshot consumer completion before their streams are destroyed.
        Outside callbacks, keep external consumer streams alive until their exported views are released.
        For work on the SDK callback stream, release retained views in the operator that last used them,
        or synchronize that operator's stream before release elsewhere. Inside callbacks, keep a stream
        you create alive until the callback returns, and synchronize it before releasing a view you used
        on it. Keep that stream on the operator so it survives callback-exit completion recording.

        Examples
        --------
        With Tensor input/output ports declared, prefer the SDK stream in ``compute(ctx)``::

            import cupy as cp

            with ctx.stream():
                view = cp.from_dlpack(self.inp.receive_data())
                self.out.emit(view * 2)

        If you need a separate stream, keep it on the operator across callbacks and synchronize before
        dropping a view, including a view retained from an earlier callback::

            with ctx.stream():
                if not hasattr(self, "_cupy_stream"):
                    self._cupy_stream = cp.cuda.Stream(non_blocking=True)
                with self._cupy_stream:
                    view = cp.from_dlpack(self.inp.receive_data())
                    try:
                        result = view * 2
                    finally:
                        self._cupy_stream.synchronize()
                del view
                self.out.emit(result)

        References
        ----------
        .. [1] https://dmlc.github.io/dlpack/latest/python_spec.html
        .. [2] https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
        """

    __dlpack_c_exchange_api__: object = ...

class PortDirection(enum.Enum):
    INPUT = 0

    OUTPUT = 1

class QueuePolicy(enum.Enum):
    BLOCKING = 0

    DROP_OLDEST = 1

class OpaqueObjectPayloadPolicy(enum.Enum):
    ALLOW = 0

    WARN = 1

    REJECT = 2

RealtimeClock: _clocks_ClockType = ...

ManualClock: _clocks_ClockType = ...

SyntheticClock: _clocks_ClockType = ...

pyobject: _payload_PayloadType = ...

class ConnectionOptions:
    def __init__(self, *, queue_policy: QueuePolicy | None = None,
                 queue_depth: int | None = None, topic_name: str | None = None,
                 memory_conversion: _tensor_MemoryConversionPolicy | None = None,
                 max_inflight_samples: int | None = None) -> None: ...
    @property
    def queue_policy(self) -> QueuePolicy | None: ...
    @property
    def queue_depth(self) -> int | None: ...
    @property
    def topic_name(self) -> str | None: ...
    @property
    def memory_conversion(self) -> _tensor_MemoryConversionPolicy | None: ...
    @property
    def max_inflight_samples(self) -> int | None: ...

class CompileOptions:
    """Caller-owned inputs that control one Graph compilation."""

    def __init__(self, *, deployment: _execution_DeploymentBinding = _execution_DeploymentBinding(), profile: _execution_Profile = _execution_Profile.STANDARD, auto_insert_converters: bool = True, parameters: _params_ParameterCompileOptions = _params_ParameterCompileOptions(), lifecycle_transactions: _lifecycle_LifecycleTransactionOptions | None = None, automatic_queue_sizing: _execution_AutomaticQueueSizingOptions = _execution_AutomaticQueueSizingOptions(), cuda_execution: _execution_CudaExecutionCompileOptions = _execution_CudaExecutionCompileOptions(), opaque_object_payloads: OpaqueObjectPayloadPolicy = OpaqueObjectPayloadPolicy.ALLOW, serialization_domain_splits: _execution_SerializationDomainSplitPolicy = _execution_SerializationDomainSplitPolicy.WARN) -> None: ...  # noqa: B008 - default constructed by the native binding

    @property
    def deployment(self) -> _execution_DeploymentBinding: ...

    @deployment.setter
    def deployment(self, arg: _execution_DeploymentBinding, /) -> None: ...

    @property
    def profile(self) -> _execution_Profile: ...

    @profile.setter
    def profile(self, arg: _execution_Profile, /) -> None: ...

    @property
    def auto_insert_converters(self) -> bool: ...

    @auto_insert_converters.setter
    def auto_insert_converters(self, arg: bool, /) -> None: ...

    @property
    def parameters(self) -> _params_ParameterCompileOptions: ...

    @parameters.setter
    def parameters(self, arg: _params_ParameterCompileOptions, /) -> None: ...

    @property
    def lifecycle_transactions(self) -> _lifecycle_LifecycleTransactionOptions | None: ...

    @lifecycle_transactions.setter
    def lifecycle_transactions(self, arg: _lifecycle_LifecycleTransactionOptions | None, /) -> None: ...

    @property
    def automatic_queue_sizing(self) -> _execution_AutomaticQueueSizingOptions: ...

    @automatic_queue_sizing.setter
    def automatic_queue_sizing(self, arg: _execution_AutomaticQueueSizingOptions, /) -> None: ...

    @property
    def cuda_execution(self) -> _execution_CudaExecutionCompileOptions: ...

    @cuda_execution.setter
    def cuda_execution(self, arg: _execution_CudaExecutionCompileOptions, /) -> None: ...

    @property
    def opaque_object_payloads(self) -> OpaqueObjectPayloadPolicy: ...

    @opaque_object_payloads.setter
    def opaque_object_payloads(self, arg: OpaqueObjectPayloadPolicy, /) -> None: ...

    @property
    def serialization_domain_splits(self) -> _execution_SerializationDomainSplitPolicy: ...

    @serialization_domain_splits.setter
    def serialization_domain_splits(self, arg: _execution_SerializationDomainSplitPolicy, /) -> None: ...

class Plan:
    """An immutable, inspectable result of compiling one Graph generation."""

    @property
    def ok(self) -> bool: ...

    @property
    def diagnostics(self) -> list[_diagnostics_Diagnostic]: ...

    @property
    def errors(self) -> list[_diagnostics_Diagnostic]: ...

    @property
    def warnings(self) -> list[_diagnostics_Diagnostic]: ...

    @property
    def cuda_compute_stream_policy(self) -> _execution_CudaComputeStreamPolicy: ...

    @property
    def partitions(self) -> _execution_ExecutionPartitions: ...

    @property
    def endpoint_catalog(self) -> _topics_EndpointCatalog: ...

    def find_tensor_byte_span_derivation(self, operator_path: str, output_name: str, *, field_id: int | None = None, tensor_index: int | None = None) -> _execution_TensorByteSpanDerivation | None:
        """Find derived Tensor capacity evidence by authored operator and output names."""

    @property
    def tensor_memory_kind_derivations(self) -> list[_execution_TensorMemoryKindDerivation]: ...

    def find_tensor_memory_kind_derivation(self, operator_path: str, output_name: str, *, field_id: int | None = None, tensor_index: int | None = None) -> _execution_TensorMemoryKindDerivation | None:
        """Find derived Tensor memory-kind evidence by authored operator and output names."""

    def json(self) -> str: ...

    def summary(self) -> str: ...

    def require_no_errors(self) -> None: ...

    def __bool__(self) -> bool: ...

    def __repr__(self) -> str: ...

def compile(graph: object, options: object | None = None) -> Plan:
    """
    Compile a Graph into a Plan. Admission failures are retained as diagnostics.
    """

class RunSession:
    """
    The sole owner of one asynchronous run attempt.

    Use this object as a context manager. Exiting requests cooperative stop and waits for cleanup. Python
    runs leave process signals entirely caller-managed and never install an SDK signal handler.
    """

    @property
    def is_running(self) -> bool: ...

    @property
    def termination(self) -> RunTermination: ...

    @property
    def diagnostics(self) -> list[_diagnostics_Diagnostic]: ...

    def summary(self) -> str:
        """
        Return the bounded human-readable report for this session's runtime diagnostics.

        Errors render before warnings while preserving publication order within each severity. Use the
        structured ``diagnostics`` values, rather than parsing this text, for program logic.
        """

    def request_stop(self) -> None: ...

    def parameter_control(self, policy: _params_ParamAccessPolicy) -> _params_ParamControlHandle: ...

    def lifecycle_control(self, policy: _lifecycle_LifecycleAccessPolicy) -> _lifecycle_LifecycleControlHandle: ...

    def wait(self, timeout: float | None = None) -> bool:
        """
        Wait for terminal state and return True, or return False when the timeout expires.

        ``timeout`` is None for an unbounded wait or a nonnegative number of seconds. An unbounded successful
        wait also joins the session worker. A bounded successful wait reports frozen terminal state while
        the session retains strict ownership until a later unbounded wait or cleanup. This operation does
        not request stop.
        """

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> bool: ...

    def __repr__(self) -> str: ...

class RunTermination(enum.Enum):
    RUNNING = 0

    COMPLETED = 1

    REQUESTED_COMPLETED = 2

    STOPPED = 3

    FAILED = 4

def run(plan: Plan) -> RunSession:
    """
    Run a Plan to terminal completion and return its inspectable RunSession.

    The call releases the GIL while waiting. Process signals remain caller-managed; no SDK signal handler
    is installed.
    """

def run_async(plan: Plan) -> RunSession:
    """
    Start a Plan and return its RunSession without waiting.

    Process signals remain caller-managed; no SDK signal handler is installed. Prefer ``with
    run_async(plan) as session`` so scope exit always requests stop and waits.
    """

__version__: str

# ``compile`` remains directly available as ``holoscan.compile`` but is omitted here because the
# runtime wildcard export list intentionally excludes names that collide with Python builtins.
__all__: list[str] = ['__version__', 'Graph', 'GraphHandle', 'Operator', 'OperatorSpec', 'OperatorPort', 'ExecutionContext', 'ActivationReason', 'ActivationStatus', 'Param', 'Input', 'Output', 'Port', 'Flow', 'Sample', 'SelectedSample', 'SelectedBatch', 'InputSelection', 'InputState', 'SelectionReason', 'SampleMetadata', 'SampleMetadataVersion', 'SampleFlags', 'EmitOptions', 'PayloadKind', 'Contract', 'OnAnyInput', 'OnEach', 'OnClock', 'OnAligned', 'OnWindow', 'Exact', 'LatestWithin', 'DrainAll', 'BracketedWindow', 'Skip', 'SafeDefault', 'ReuseLast', 'OnViolation', 'PortConfigurator', 'FlowConfigurator', 'Tensor', 'PortDirection', 'QueuePolicy', 'OpaqueObjectPayloadPolicy', 'RealtimeClock', 'ManualClock', 'SyntheticClock', 'HoloscanError', 'ConnectionOptions', 'CompileOptions', 'Plan', 'RunSession', 'RunTermination', 'run', 'run_async', 'pyobject']

def __getattr__(name: str) -> object:
    """Load an independently packaged Holoscan module on first access."""

def __dir__() -> list[str]:
    """Include discoverable providers without importing their packages."""
