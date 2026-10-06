# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Compile-time parameter binding and run-scoped parameter control."""

from holoscan.config import Document as _config_Document
from holoscan.config import RawSourceDigest as _config_RawSourceDigest
from holoscan.config import ValueDecodeOptions as _config_ValueDecodeOptions
from holoscan.lifecycle import LifecycleOperatorRef as _lifecycle_LifecycleOperatorRef

from collections.abc import Iterable
import enum



class Binding:
    def __repr__(self) -> str: ...

class BindingMode(enum.Enum):
    PATCH = 0

class ConfigDocumentProvenance:
    @property
    def semantic_version(self) -> int: ...

    @property
    def semantic_digest(self) -> str: ...

    @property
    def source(self) -> _config_RawSourceDigest | None: ...

    @property
    def base_source(self) -> _config_RawSourceDigest | None: ...

    def __repr__(self) -> str: ...

class ConfigFieldProvenance:
    @property
    def document(self) -> ConfigDocumentProvenance: ...

    @property
    def source_path(self) -> str: ...

    @property
    def source_ordinal(self) -> int: ...

    @property
    def target_operator_path(self) -> str: ...

    @property
    def target_schema_id(self) -> str: ...

    @property
    def target_schema_version(self) -> int: ...

    @property
    def target_field_id(self) -> int: ...

    @property
    def target_full_id(self) -> str: ...

    @property
    def origin(self) -> FieldOrigin: ...

    def __repr__(self) -> str: ...

class DecodeOptions:
    def __init__(self) -> None: ...

    @property
    def values(self) -> _config_ValueDecodeOptions: ...

    @values.setter
    def values(self, arg: _config_ValueDecodeOptions, /) -> None: ...

    @property
    def max_bindings(self) -> int: ...

    @max_bindings.setter
    def max_bindings(self, arg: int, /) -> None: ...

    @property
    def max_fields(self) -> int: ...

    @max_fields.setter
    def max_fields(self, arg: int, /) -> None: ...

class FieldOrigin(enum.Enum):
    CAPTURED = 0

    ALIAS = 1

    MIGRATION = 2

    BASELINE = 3

    DEFAULT = 4

class ParamAccessPolicy:
    """Restrict one run-wide, local-application parameter-control capability.

    Python currently exposes permission and visibility filters but not Core scope narrowing.
    """

    def __init__(self, *, permissions: ParamPermission = ..., include_internal: bool = False, include_sensitive: bool = False, max_event_cursors: int = 0) -> None: ...

    @property
    def permissions(self) -> ParamPermission: ...

    @permissions.setter
    def permissions(self, arg: ParamPermission, /) -> None: ...

    @property
    def include_internal(self) -> bool: ...

    @include_internal.setter
    def include_internal(self, arg: bool, /) -> None: ...

    @property
    def include_sensitive(self) -> bool: ...

    @include_sensitive.setter
    def include_sensitive(self, arg: bool, /) -> None: ...

    @property
    def max_event_cursors(self) -> int: ...

    @max_event_cursors.setter
    def max_event_cursors(self, arg: int, /) -> None: ...

class ParamBatchUpdateResult:
    @property
    def status(self) -> ParamUpdateStatus: ...

    @property
    def committed(self) -> bool: ...

    @property
    def patch_count(self) -> int: ...

    @property
    def failed_patch(self) -> int | None: ...

    @property
    def update_sequence(self) -> int: ...

    @property
    def target_generation(self) -> int: ...

class ParamControlHandle:
    """Read, submit, await, and observe policy-authorized parameter updates.

    ``read`` supports fixed scalar and duration fields. Aggregate fields accepted by ``submit``
    are not scalar-projectable and raise ``NotSupportedError`` when read.
    """

    def read(self, path: str, view: ParamReadView = ParamReadView.PUBLISHED) -> ParamReadResult: ...

    def submit(self, patches: ParamPatch | Iterable[ParamPatch]) -> ParamBatchUpdateResult: ...

    def await_applied(self, update_sequence: int, timeout: float | None = None) -> ParamUpdateResult: ...

    def events(self) -> ParamEventCursor: ...

class ParamEventKind(enum.Enum):
    TRANSITION = 0

    GAP = 1

class ParamEvent:
    @property
    def kind(self) -> ParamEventKind: ...

    @property
    def event_sequence(self) -> int: ...

    @property
    def first_missing_sequence(self) -> int: ...

    @property
    def last_missing_sequence(self) -> int: ...

    @property
    def operator_instance(self) -> _lifecycle_LifecycleOperatorRef: ...

    @property
    def field(self) -> int: ...

    @property
    def update_sequence(self) -> int: ...

    @property
    def prior_generation(self) -> int: ...

    @property
    def target_generation(self) -> int: ...

    @property
    def state(self) -> ParamUpdateState: ...

    @property
    def status(self) -> ParamUpdateStatus: ...

    @property
    def actor_id(self) -> int: ...

    @property
    def reason_id(self) -> int: ...

class ParamEventCursor:
    """A nonblocking live event cursor; call ``drain`` repeatedly to poll for new events."""

    def drain(self, maximum: int = 64) -> list[ParamEvent]: ...

class ParameterCompileOptions:
    def __init__(self) -> None: ...

    @property
    def storage(self) -> ParamStorageMode: ...

    @storage.setter
    def storage(self, arg: ParamStorageMode, /) -> None: ...

    @property
    def runtime_updates(self) -> ParamRuntimeUpdatePolicy: ...

    @runtime_updates.setter
    def runtime_updates(self, arg: ParamRuntimeUpdatePolicy, /) -> None: ...

    @property
    def generation_slots_per_operator(self) -> int: ...

    @generation_slots_per_operator.setter
    def generation_slots_per_operator(self, arg: int, /) -> None: ...

    @property
    def max_pending_updates_per_operator(self) -> int: ...

    @max_pending_updates_per_operator.setter
    def max_pending_updates_per_operator(self, arg: int, /) -> None: ...

    @property
    def audit_capacity(self) -> int: ...

    @audit_capacity.setter
    def audit_capacity(self, arg: int, /) -> None: ...

    @property
    def max_event_cursors(self) -> int: ...

    @max_event_cursors.setter
    def max_event_cursors(self, arg: int, /) -> None: ...

    @property
    def max_parameter_arena_bytes(self) -> int | None: ...

    @max_parameter_arena_bytes.setter
    def max_parameter_arena_bytes(self, arg: int | None, /) -> None: ...

    @property
    def exhaustion(self) -> ParamExhaustionPolicy: ...

    @exhaustion.setter
    def exhaustion(self, arg: ParamExhaustionPolicy, /) -> None: ...

    @property
    def require_proven_schema_completeness(self) -> bool: ...

    @require_proven_schema_completeness.setter
    def require_proven_schema_completeness(self, arg: bool, /) -> None: ...

    @property
    def require_explicit_field_policies(self) -> bool: ...

    @require_explicit_field_policies.setter
    def require_explicit_field_policies(self, arg: bool, /) -> None: ...

class ParamExhaustionPolicy(enum.Enum):
    REJECT_UPDATE = 0

class ParamPermission(enum.IntFlag):
    NONE = 0

    READ = 1

    UPDATE = 2

    AWAIT = 4

    CAPTURE = 8

    OBSERVE = 16

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ParamReadView(enum.Enum):
    PUBLISHED = 0

    LATEST_ACCEPTED = 1

    PENDING = 2

class ParamRuntimeUpdatePolicy(enum.Enum):
    PROFILE_DEFAULT = 0

    DISABLED = 1

    ENABLED_IF_ADMITTED = 2

    REQUIRE_PREALLOCATED = 3

class ParamStorageMode(enum.Enum):
    PROFILE_DEFAULT = 0

    DYNAMIC_COLD_PATH = 1

    FREEZE_BEFORE_ARM = 2

    PREALLOCATED_BOUNDED = 3

    INLINE_ONLY = 4

class ParamUpdateState(enum.Enum):
    SUBMITTED = 0

    REJECTED = 1

    ACCEPTED = 2

    QUIESCING = 3

    RESTARTING = 4

    ACCEPTED_NO_CHANGE = 5

    PUBLISHED = 6

    OBSERVED = 7

    SUPERSEDED = 8

    CANCELED = 9

    ROLLED_BACK = 10

    FAILED_STOPPED = 11

class ParamUpdateStatus(enum.Enum):
    ACCEPTED = 0

    ACCEPTED_NO_CHANGE = 1

    SUBMITTED = 2

    REJECTED = 3

    DEFERRED = 4

    UPDATES_DISABLED = 5

    OUT_OF_BOUNDS = 6

    VALIDATION_FAILED = 7

    TYPE_MISMATCH = 8

    UNKNOWN_PARAMETER = 9

    STALE_OPERATOR_REF = 10

    EXPECTED_GENERATION_MISMATCH = 11

    REQUIRES_LIFECYCLE_BARRIER = 12

    REQUIRES_RECOMPILE = 13

    UNSUPPORTED_ATOMIC_SCOPE = 14

    PAST_EFFECTIVE_TIME = 15

    UNSUPPORTED_CLOCK = 16

    QUEUE_FULL = 17

    INVALID_CALLING_CONTEXT = 18

    UNSUPPORTED_IN_PROFILE = 19

    CLOSED = 20

class ParamPatch:
    """Own one path/value update request for a live session."""

    def __init__(self, path: str, value: object | None, *, expected_generation: int | None = None) -> None: ...

    @property
    def path(self) -> str: ...

    @property
    def value(self) -> object: ...

    @property
    def expected_generation(self) -> int | None: ...

class ParamReadResult:
    @property
    def value(self) -> object: ...

    @property
    def generation(self) -> int: ...

    @property
    def update_sequence(self) -> int: ...

    @property
    def state(self) -> ParamUpdateState: ...

class RebindMode(enum.Enum):
    SPARSE_CONTRIBUTIONS = 0

    COMPLETE_OBJECT = 1

class SetParamsProvenance:
    """Immutable provenance for one successful parameter attachment."""

    @property
    def mode(self) -> RebindMode: ...

    @property
    def config_fields(self) -> tuple[ConfigFieldProvenance, ...]: ...

    @property
    def config_field_count(self) -> int: ...

    def __repr__(self) -> str: ...

class ParamUpdateResult:
    @property
    def status(self) -> ParamUpdateStatus: ...

    @property
    def state(self) -> ParamUpdateState: ...

    @property
    def update_sequence(self) -> int: ...

    @property
    def target_generation(self) -> int: ...

    @property
    def lifecycle_transaction_id(self) -> int: ...

    @property
    def coalesced_into_transaction(self) -> int: ...

def bind(operator: object, path: object, mode: BindingMode = BindingMode.PATCH) -> Binding: ...

def set_params(graph: object, document: _config_Document, bindings: object, options: DecodeOptions = DecodeOptions()) -> SetParamsProvenance:  # noqa: B008 - default constructed by the native binding
    """Decode and attach configuration values to Graph operators."""

__all__: list[str] = ['Binding', 'BindingMode', 'ConfigDocumentProvenance', 'ConfigFieldProvenance', 'DecodeOptions', 'FieldOrigin', 'ParamAccessPolicy', 'ParamBatchUpdateResult', 'ParamControlHandle', 'ParamEventKind', 'ParamEvent', 'ParamEventCursor', 'ParameterCompileOptions', 'ParamExhaustionPolicy', 'ParamPermission', 'ParamReadView', 'ParamRuntimeUpdatePolicy', 'ParamStorageMode', 'ParamUpdateState', 'ParamUpdateStatus', 'ParamPatch', 'ParamReadResult', 'RebindMode', 'SetParamsProvenance', 'ParamUpdateResult', 'bind', 'set_params']
