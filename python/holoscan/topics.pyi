# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""External topic and endpoint catalog values."""

from holoscan.clocks import ClockKey as _clocks_ClockKey
from holoscan.payload import PayloadType as _payload_PayloadType
from holoscan.tensor import DLDataType as _tensor_DLDataType
from holoscan.tensor import InterfaceTensorEmitPolicy as _tensor_InterfaceTensorEmitPolicy
from holoscan.tensor import MemoryKind as _tensor_MemoryKind
from holoscan.tensor import MemoryConversionPolicy as _tensor_MemoryConversionPolicy
from holoscan.tensor import TensorBoundaryContract as _tensor_TensorBoundaryContract
from holoscan.tensor import TensorStridePolicy as _tensor_TensorStridePolicy

import enum

import holoscan


class ClockHealthCapability(enum.Enum):
    UNAVAILABLE = 0

    STATUS_REPORTS = 1

class ClockHealthOptions:
    def __init__(self, *, capability: ClockHealthCapability = ClockHealthCapability.UNAVAILABLE, requirement: ClockHealthRequirement = ClockHealthRequirement.NOT_REQUIRED, max_report_age: object | None = None) -> None: ...

    @property
    def capability(self) -> ClockHealthCapability: ...

    @capability.setter
    def capability(self, arg: ClockHealthCapability, /) -> None: ...

    @property
    def requirement(self) -> ClockHealthRequirement: ...

    @requirement.setter
    def requirement(self, arg: ClockHealthRequirement, /) -> None: ...

    @property
    def max_report_age(self) -> object: ...

    @max_report_age.setter
    def max_report_age(self, arg: object | None, /) -> None: ...

class ClockHealthRequirement(enum.Enum):
    NOT_REQUIRED = 0

    FRESH_HEALTHY_REPORT = 1

class EndpointBounds:
    def __init__(self, *, publisher: object | None = None, subscriber: object | None = None) -> None: ...

    @property
    def publisher(self) -> PublisherEndpointBounds: ...

    @publisher.setter
    def publisher(self, arg: PublisherEndpointBounds | None, /) -> None: ...

    @property
    def subscriber(self) -> SubscriberEndpointBounds: ...

    @subscriber.setter
    def subscriber(self, arg: SubscriberEndpointBounds | None, /) -> None: ...

class EndpointCatalog:
    """An immutable catalog of external Graph endpoints."""

    @staticmethod
    def load(path: object) -> EndpointCatalog: ...

    @staticmethod
    def parse(text: str) -> EndpointCatalog: ...

    def json(self) -> str: ...

    def save_json(self, path: object) -> None:
        """Save canonical catalog JSON to a filesystem path if its bytes changed."""

    @property
    def digest(self) -> str: ...

    @property
    def scope(self) -> str: ...

    def endpoint(self, name: str) -> EndpointView: ...

    def compatibility_with(self, candidate: EndpointCatalog) -> EndpointCompatibilityReport: ...

class EndpointCompatibilityReport:
    @property
    def compatible(self) -> bool: ...

    @property
    def diagnostics(self) -> list[holoscan.diagnostics.Diagnostic]: ...

    def json(self) -> str: ...

class EndpointNameOrigin(enum.Enum):
    EXPLICIT = 0
    GENERATED = 1

class EndpointSchemaIdentity:
    @property
    def canonical_name(self) -> str: ...

    @property
    def compatibility_epoch(self) -> int: ...

    @property
    def family_id(self) -> bytes: ...

    @property
    def file_identifier(self) -> bytes: ...

class TensorBoundaryContractView:
    @property
    def memory_kind(self) -> _tensor_MemoryKind | None: ...

    @property
    def dtype(self) -> _tensor_DLDataType | None: ...

    @property
    def rank(self) -> int | None: ...

    @property
    def max_rank(self) -> int | None: ...

    @property
    def max_byte_span(self) -> int | None: ...

    @property
    def axes(self) -> tuple[tuple[int, int], ...]: ...

    @property
    def stride_policy(self) -> _tensor_TensorStridePolicy: ...

    @property
    def foreign_emit_policy(self) -> _tensor_InterfaceTensorEmitPolicy: ...

    @property
    def max_inline_bytes(self) -> int | None: ...

    @property
    def max_staging_byte_span(self) -> int | None: ...

class EndpointView:
    @property
    def name(self) -> str: ...

    @property
    def uri(self) -> str: ...

    @property
    def service(self) -> str: ...

    @property
    def direction(self) -> TopicDirection: ...

    @property
    def name_origin(self) -> EndpointNameOrigin: ...

    @property
    def scalar_identity(self) -> str | None: ...

    @property
    def schema_identity(self) -> EndpointSchemaIdentity | None: ...

    @property
    def schema_writer_max_serialized_bytes(self) -> int | None: ...

    @property
    def schema_required_frame_bytes(self) -> int | None: ...

    @property
    def schema_selected_inline_bucket_bytes(self) -> int | None: ...

    @property
    def schema_bucket_usable_serialized_bytes(self) -> int | None: ...

    @property
    def schema_transport_profile_tag(self) -> int | None: ...

    @property
    def attach_policy(self) -> TopicAttachPolicy: ...

    @property
    def bounds(self) -> EndpointBounds: ...

    @property
    def memory_conversion(self) -> _tensor_MemoryConversionPolicy: ...

    @property
    def clock_health(self) -> ClockHealthOptions: ...

    @property
    def capture_clock(self) -> str | None: ...

    @property
    def has_tensor_requirements(self) -> bool: ...

    @property
    def tensor_requirements(self) -> TensorBoundaryContractView | None: ...

    @property
    def has_pod_identity(self) -> bool: ...

class TopicDirection(enum.Enum):
    IMPORT = 0
    EXPORT = 1

class ExportedTopic:
    """A weak exported-topic handle exposing its Graph-side input port."""

    @property
    def input(self) -> holoscan.Port: ...

    @property
    def graph_valid(self) -> bool: ...

    def __repr__(self) -> str: ...

class ImportedTopic:
    """A weak imported-topic handle exposing its Graph-side output port."""

    @property
    def output(self) -> holoscan.Port: ...

    @property
    def graph_valid(self) -> bool: ...

    def __repr__(self) -> str: ...

class PeerRequirement(enum.Enum):
    OPTIONAL = 0

    REQUIRED = 1

class ProbeFailurePolicy(enum.Enum):
    WARN_AND_RECORD_FALLBACK = 0

    FAIL_ADMISSION = 1

class PublisherEndpointBounds:
    def __init__(self, *, max_loaned_samples: object | None = None) -> None: ...

    @property
    def max_loaned_samples(self) -> int | None: ...

    @max_loaned_samples.setter
    def max_loaned_samples(self, arg: int | None, /) -> None: ...

class SubscriberEndpointBounds:
    def __init__(self, *, queue_depth: object | None = None, max_borrowed_samples: object | None = None) -> None: ...

    @property
    def queue_depth(self) -> int | None: ...

    @queue_depth.setter
    def queue_depth(self, arg: int | None, /) -> None: ...

    @property
    def max_borrowed_samples(self) -> int | None: ...

    @max_borrowed_samples.setter
    def max_borrowed_samples(self, arg: int | None, /) -> None: ...

class TopicAttachPolicy:
    def __init__(self, *, peer: PeerRequirement = PeerRequirement.OPTIONAL, max_wait: object | None = None) -> None: ...

    @property
    def peer(self) -> PeerRequirement: ...

    @peer.setter
    def peer(self, arg: PeerRequirement, /) -> None: ...

    @property
    def max_wait(self) -> object: ...

    @max_wait.setter
    def max_wait(self, arg: object | None, /) -> None: ...

class TopicOptions:
    """
    Owned, mutable backend-neutral options for one external topic declaration.

    Use ``tensor`` for a plain Tensor payload and ``tensor_fields`` for a
    descriptor-backed schema payload. Descriptor field keys are generated schema
    names and unknown names report the complete known-field list.
    """

    def __init__(self, *, endpoint_name: str | None = None, memory_conversion: _tensor_MemoryConversionPolicy | None = None, schema: _payload_PayloadType | None = None, capture_time: _clocks_ClockKey | None = None, clock_health: ClockHealthOptions | None = None, attach: TopicAttachPolicy = ..., bounds: EndpointBounds = ..., tensor: _tensor_TensorBoundaryContract | None = None, tensor_fields: dict[str, _tensor_TensorBoundaryContract] = ...) -> None: ...

    @property
    def endpoint_name(self) -> str | None: ...

    @endpoint_name.setter
    def endpoint_name(self, arg: object | None, /) -> None: ...

    @property
    def memory_conversion(self) -> _tensor_MemoryConversionPolicy | None: ...

    @memory_conversion.setter
    def memory_conversion(self, arg: object | None, /) -> None: ...

    @property
    def schema(self) -> _payload_PayloadType | None: ...

    @schema.setter
    def schema(self, arg: object | None, /) -> None: ...

    @property
    def capture_time(self) -> _clocks_ClockKey | None: ...

    @capture_time.setter
    def capture_time(self, arg: object | None, /) -> None: ...

    @property
    def clock_health(self) -> ClockHealthOptions: ...

    @clock_health.setter
    def clock_health(self, arg: object | None, /) -> None: ...

    @property
    def attach(self) -> TopicAttachPolicy: ...

    @attach.setter
    def attach(self, arg: object | None, /) -> None: ...

    @property
    def bounds(self) -> EndpointBounds: ...

    @bounds.setter
    def bounds(self, arg: object | None, /) -> None: ...

    @property
    def tensor(self) -> _tensor_TensorBoundaryContract | None: ...

    @tensor.setter
    def tensor(self, arg: object | None, /) -> None: ...

    @property
    def tensor_fields(self) -> dict[str, _tensor_TensorBoundaryContract] | None: ...

    @tensor_fields.setter
    def tensor_fields(self, arg: object | None, /) -> None: ...

__all__: list[str] = ['ClockHealthCapability', 'ClockHealthOptions', 'ClockHealthRequirement', 'EndpointBounds', 'EndpointCatalog', 'EndpointCompatibilityReport', 'EndpointNameOrigin', 'EndpointSchemaIdentity', 'TensorBoundaryContractView', 'EndpointView', 'ExportedTopic', 'ImportedTopic', 'PeerRequirement', 'ProbeFailurePolicy', 'PublisherEndpointBounds', 'SubscriberEndpointBounds', 'TopicAttachPolicy', 'TopicDirection', 'TopicOptions']
