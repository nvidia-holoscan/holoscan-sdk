# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Execution planning and deployment support values."""

from holoscan.operators import OperatorHandle as _operators_OperatorHandle
from holoscan.tensor import MemoryKind as _tensor_MemoryKind
from holoscan.topics import ProbeFailurePolicy as ProbeFailurePolicy

import enum
from typing import Self
from collections.abc import Iterator

import holoscan


class AutomaticQueueSizingOptions:
    """Options controlling automatic logical queue sizing."""

    def __init__(self, *, headroom_ratio: RateRatio = ...) -> None: ...

    @property
    def headroom_ratio(self) -> RateRatio: ...

    @headroom_ratio.setter
    def headroom_ratio(self, arg: RateRatio, /) -> None: ...

class CPUAffinity:
    """
    A normalized CPU-core selection whose cores property is an immutable tuple snapshot.
    """

    def __init__(self, values: object | None = None) -> None: ...

    @staticmethod
    def from_ranges(ranges: object) -> CPUAffinity: ...

    @property
    def cores(self) -> tuple[int, ...]: ...

    @property
    def empty(self) -> bool: ...

    def __repr__(self) -> str: ...

class CudaComputeStreamPolicy(enum.Enum):
    PER_EXECUTION_PARTITION = 0

    SHARED_PROCESS_DEVICE = 1

class CudaExecutionCompileOptions:
    """Options controlling immutable CUDA execution-resource lowering."""

    def __init__(self, *, compute_stream_policy: CudaComputeStreamPolicy = CudaComputeStreamPolicy.PER_EXECUTION_PARTITION) -> None: ...

    @property
    def compute_stream_policy(self) -> CudaComputeStreamPolicy: ...

    @compute_stream_policy.setter
    def compute_stream_policy(self, arg: CudaComputeStreamPolicy, /) -> None: ...

class CudaGreenContextCapability:
    """Read-only CUDA Green Context partitioning evidence for one device."""

    @property
    def device(self) -> DeviceId: ...

    @property
    def deployment_device_id(self) -> str: ...

    @property
    def total_sm_count(self) -> int: ...

    @property
    def min_sm_partition_size(self) -> int: ...

    @property
    def sm_coscheduled_alignment(self) -> int: ...

    @property
    def structured_sm_quantum(self) -> int: ...

    @property
    def max_structured_context_count(self) -> int: ...

class CudaGreenContextPlacement:
    """
    A requested CUDA Green Context device and streaming-multiprocessor count.
    """

    def __init__(self, *, device: DeviceId, sm_count: int) -> None: ...

    @property
    def device(self) -> DeviceId: ...

    @device.setter
    def device(self, arg: DeviceId, /) -> None: ...

    @property
    def sm_count(self) -> int: ...

    @sm_count.setter
    def sm_count(self, arg: int, /) -> None: ...

class DeadlineScheduling:
    """
    Linux deadline-scheduling runtime, deadline, and period values in nanoseconds.
    """

    def __init__(self, *, runtime_ns: int = 0, deadline_ns: int = 0, period_ns: int = 0) -> None: ...

    @property
    def runtime_ns(self) -> int: ...

    @runtime_ns.setter
    def runtime_ns(self, arg: int, /) -> None: ...

    @property
    def deadline_ns(self) -> int: ...

    @deadline_ns.setter
    def deadline_ns(self, arg: int, /) -> None: ...

    @property
    def period_ns(self) -> int: ...

    @period_ns.setter
    def period_ns(self, arg: int, /) -> None: ...

class DeploymentBinding:
    """Caller-owned deployment bindings applied while compiling a Graph."""

    def __init__(self) -> None: ...

    def bind_thread_placement(self, name: str, placement: ThreadPlacement) -> DeploymentBinding: ...

    def bind_support_thread_affinity(self, role: SupportThreadRole, affinity: CPUAffinity) -> DeploymentBinding:
        """
        Bind a Runtime support-thread role to a copied, nonempty CPU affinity request.
        """

    def bind_cuda_green_context(self, name: str, placement: CudaGreenContextPlacement) -> DeploymentBinding: ...

    def bind_tensor_output_device(self, *, operator_path: str, output_port: str, device: DeviceId) -> DeploymentBinding: ...

class DeviceId:
    """An explicit deployment-device ordinal."""

    def __init__(self, value: int) -> None: ...

    @property
    def value(self) -> int: ...

    @value.setter
    def value(self, arg: int, /) -> None: ...

    @property
    def valid(self) -> bool: ...

    def __repr__(self) -> str: ...

class ExecutionPartition:
    """An immutable view of one accepted execution partition."""

    @property
    def id(self) -> int: ...

    @property
    def origin(self) -> ExecutionPartitionOrigin: ...

    @property
    def authored_name(self) -> str | None: ...

    @property
    def members(self) -> list[int]: ...

    @property
    def operator_paths(self) -> list[str]: ...

    @property
    def execution_lane(self) -> int: ...

class ExecutionPartitionOrigin(enum.Enum):
    AUTHORED = 0

    AUTOMATIC = 1

class ExecutionPartitions:
    """
    A retained range and lookup view over the accepted execution partitions.
    """

    @property
    def mode(self) -> PartitioningMode: ...

    @property
    def authored_count(self) -> int: ...

    @property
    def automatic_count(self) -> int: ...

    def lookup_named(self, name: str) -> ExecutionPartition: ...

    def of(self, operator: object) -> ExecutionPartition: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[ExecutionPartition]: ...

class IRQIsolation(enum.Enum):
    NOT_REQUESTED = 0

    PREFER_ISOLATED = 1

    REQUIRE_ISOLATED = 2

class PartitionHandle:
    """A weak fluent handle for one named manual execution partition."""

    def add(self, *members: _operators_OperatorHandle) -> Self: ...

    def include_all(self) -> PartitionHandle: ...

class PartitioningMode(enum.Enum):
    AUTOMATIC = 0

    CONSTRAINED_AUTOMATIC = 1

    MANUAL = 2

class Profile(enum.Enum):
    STANDARD = 0

    ROBOTICS = 1

    REAL_TIME = 2

    REGULATED = 3

    RESEARCH = 4

class RateRatio:
    """A positive rational rate or sizing ratio."""

    def __init__(self, numerator: int = 1, denominator: int = 1) -> None: ...

    @property
    def numerator(self) -> int: ...

    @numerator.setter
    def numerator(self, arg: int, /) -> None: ...

    @property
    def denominator(self) -> int: ...

    @denominator.setter
    def denominator(self, arg: int, /) -> None: ...

class RegionConstraint:
    """A backend-neutral logical-region constraint value."""

    @staticmethod
    def static_topology() -> RegionConstraint: ...

    @staticmethod
    def bounded_memory() -> RegionConstraint: ...

    @staticmethod
    def device_compatible_ops() -> RegionConstraint: ...

    @staticmethod
    def cuda_graph() -> RegionConstraint: ...

    @staticmethod
    def latency_budget(budget: object) -> RegionConstraint: ...

    @property
    def kind(self) -> RegionConstraintKind: ...

    @property
    def latency_ns(self) -> int: ...

class RegionConstraintKind(enum.Enum):
    STATIC_TOPOLOGY = 0

    BOUNDED_MEMORY = 1

    DEVICE_COMPATIBLE_OPERATORS = 2

    LATENCY_BUDGET = 3

    CUDA_GRAPH = 4

class RegionHandle:
    """A weak fluent handle for one named Graph region."""

    def add(self, *members: _operators_OperatorHandle | holoscan.GraphHandle) -> Self: ...

    def constrain(self, constraint: RegionConstraint) -> RegionHandle: ...

    def profile(self, profile: Profile) -> RegionHandle: ...

class SchedulingPolicy(enum.Enum):
    OTHER = 0

    FIFO = 1

    ROUND_ROBIN = 2

    DEADLINE = 3

class SupportThreadRole(enum.Enum):
    """A Runtime-owned support-thread role that can receive CPU affinity."""

    CONTROLLER = 0

    CHANNEL_PUMP = 1

class TensorByteSpanDerivation:
    """Immutable compiler evidence for one derived Tensor output-pool slot capacity."""

    @property
    def output(self) -> int: ...

    @property
    def field_id(self) -> int | None: ...

    @property
    def tensor_index(self) -> int | None: ...

    @property
    def resolved_max_byte_span(self) -> int: ...

    @property
    def resolved_max_staging_byte_span(self) -> int | None: ...

    @property
    def update_disposition(self) -> TensorByteSpanDerivationUpdateDisposition: ...

class TensorByteSpanDerivationUpdateDisposition(enum.Enum):
    FROZEN_COMPILE_TIME = 0

    RECOMPILE_REQUIRED = 1

class TensorMemoryKindDerivation:
    """Immutable compiler evidence for one Parameter-derived Tensor output memory kind."""

    @property
    def output(self) -> int: ...

    @property
    def field_id(self) -> int | None: ...

    @property
    def tensor_index(self) -> int | None: ...

    @property
    def parameter_schema_id(self) -> str: ...

    @property
    def parameter_path(self) -> str: ...

    @property
    def resolved_memory_kind(self) -> _tensor_MemoryKind: ...

    @property
    def update_disposition(self) -> TensorMemoryKindDerivationUpdateDisposition: ...

class TensorMemoryKindDerivationUpdateDisposition(enum.Enum):
    FROZEN_COMPILE_TIME = 0

    RECOMPILE_REQUIRED = 1

class ThreadPlacement:
    """
    CPU affinity, scheduling, wake, interrupt-isolation, and probe policy for one execution lane.
    """

    def __init__(self, *, affinity: CPUAffinity | None = None, policy: SchedulingPolicy = SchedulingPolicy.OTHER, priority: int | None = None, deadline: DeadlineScheduling | None = None, wake_mode: WakeMode = WakeMode.SLEEPING_WAITSET, irq_isolation: IRQIsolation = IRQIsolation.NOT_REQUESTED, probe_failure_policy: ProbeFailurePolicy = ProbeFailurePolicy.WARN_AND_RECORD_FALLBACK) -> None: ...

    @property
    def affinity(self) -> CPUAffinity | None: ...

    @affinity.setter
    def affinity(self, arg: CPUAffinity | None, /) -> None: ...

    @property
    def scheduling(self) -> ThreadScheduling: ...

    @scheduling.setter
    def scheduling(self, arg: ThreadScheduling, /) -> None: ...

    @property
    def policy(self) -> SchedulingPolicy: ...

    @policy.setter
    def policy(self, arg: SchedulingPolicy, /) -> None: ...

    @property
    def priority(self) -> int | None: ...

    @priority.setter
    def priority(self, arg: int | None, /) -> None: ...

    @property
    def deadline(self) -> DeadlineScheduling: ...

    @deadline.setter
    def deadline(self, arg: DeadlineScheduling | None, /) -> None: ...

    @property
    def wake_mode(self) -> WakeMode: ...

    @wake_mode.setter
    def wake_mode(self, arg: WakeMode, /) -> None: ...

    @property
    def irq_isolation(self) -> IRQIsolation: ...

    @irq_isolation.setter
    def irq_isolation(self, arg: IRQIsolation, /) -> None: ...

    @property
    def probe_failure_policy(self) -> ProbeFailurePolicy: ...

    @probe_failure_policy.setter
    def probe_failure_policy(self, arg: ProbeFailurePolicy, /) -> None: ...

class ThreadScheduling:
    """A thread scheduling policy and its policy-specific parameters."""

    def __init__(self, *, policy: SchedulingPolicy = SchedulingPolicy.OTHER, priority: int | None = None, deadline: DeadlineScheduling | None = None) -> None: ...

    @property
    def policy(self) -> SchedulingPolicy: ...

    @policy.setter
    def policy(self, arg: SchedulingPolicy, /) -> None: ...

    @property
    def priority(self) -> int | None: ...

    @priority.setter
    def priority(self, arg: int | None, /) -> None: ...

    @property
    def deadline(self) -> DeadlineScheduling | None: ...

    @deadline.setter
    def deadline(self, arg: DeadlineScheduling | None, /) -> None: ...

class WakeMode(enum.Enum):
    SLEEPING_WAITSET = 0

    BUSY_POLL = 1

def query_cuda_green_context_capability(device: DeviceId) -> CudaGreenContextCapability:
    """
    Query CUDA Green Context partitioning evidence for one explicit device.
    """

__all__: list[str] = ['AutomaticQueueSizingOptions', 'CPUAffinity', 'CudaComputeStreamPolicy', 'CudaExecutionCompileOptions', 'CudaGreenContextCapability', 'CudaGreenContextPlacement', 'DeadlineScheduling', 'DeploymentBinding', 'DeviceId', 'ExecutionPartition', 'ExecutionPartitionOrigin', 'ExecutionPartitions', 'IRQIsolation', 'PartitionHandle', 'PartitioningMode', 'ProbeFailurePolicy', 'Profile', 'RateRatio', 'RegionConstraint', 'RegionConstraintKind', 'RegionHandle', 'SchedulingPolicy', 'SupportThreadRole', 'TensorByteSpanDerivation', 'TensorByteSpanDerivationUpdateDisposition', 'TensorMemoryKindDerivation', 'TensorMemoryKindDerivationUpdateDisposition', 'ThreadPlacement', 'ThreadScheduling', 'WakeMode', 'query_cuda_green_context_capability']
