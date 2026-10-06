# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Graph lifecycle authoring values."""

from holoscan import errors
from holoscan.clocks import ClockKey as _clocks_ClockKey
from holoscan.clocks import ClockRef as _clocks_ClockRef
from holoscan.operators import OperatorHandle as _operators_OperatorHandle

from collections.abc import Iterable
import enum



class GraphLifecycleBuilder:
    """A weak fluent builder for lifecycle ordering and restart permissions."""

    def before(self, before: _operators_OperatorHandle, after: _operators_OperatorHandle, stage: LifecycleStage) -> GraphLifecycleBuilder: ...

    def allow_self_restart(self, target: _operators_OperatorHandle) -> GraphLifecycleBuilder: ...

    def allow_external_restart(self, target: _operators_OperatorHandle) -> GraphLifecycleBuilder: ...

    def allow_restart_request(self, requester: _operators_OperatorHandle, target: _operators_OperatorHandle) -> GraphLifecycleBuilder: ...

class LifecycleAccessPolicy:
    """Restrict one session lifecycle-control capability to named targets and stages."""

    def __init__(self, targets: Iterable[_operators_OperatorHandle], from_stages: Iterable[LifecycleStage]) -> None: ...

class LifecycleControlHandle:
    """Submit and observe restart transactions within an immutable access policy."""

    def request_restart(self, target: _operators_OperatorHandle | LifecycleOperatorRef, options: LifecycleRestartOptions = ...) -> LifecycleRestartReceipt: ...

    def inspect(self, receipt: LifecycleRestartReceipt) -> LifecycleRestartReceiptView: ...

    def await_terminal(self, receipt: LifecycleRestartReceipt, timeout: float | None = None) -> LifecycleRestartReceiptView: ...

class LifecycleContext:
    """Callback-lifetime lifecycle state for one hosted Python operator.

    ``params`` returns a fresh dictionary containing hosted bool, int, float, and bounded integral
    sequence snapshot fields. Other Python ``Param`` kinds are not represented in that snapshot.
    """

    @property
    def params(self) -> dict[str, object]: ...

    @property
    def params_generation(self) -> int: ...

    @property
    def transition_phase(self) -> LifecycleTransitionPhase: ...

    @property
    def stop_requested(self) -> bool: ...

    def report_error(self, code: errors.ErrorCode, message: str) -> None:
        """Supply bounded detail for a returned failure status.

        The first report wins. A successful callback discards reported detail.
        """

    def restart_target(self, operator_path: str) -> LifecycleOperatorRef: ...

    def resolve_clock(self, clock: _clocks_ClockKey | None = None) -> _clocks_ClockRef:
        """Resolve the default or named clock in this callback Operator's owning Graph."""

class LifecycleOperatorRef:
    """Generation-stamped operator reference resolved by a lifecycle callback."""

    @property
    def id(self) -> int: ...

    @property
    def plan_generation(self) -> int: ...

    @property
    def run_generation(self) -> int: ...

class LifecycleStage(enum.Enum):
    CONFIGURE = 0

    DISCOVER = 1

    ALLOCATE = 2

    WARM_UP = 3

    ARM = 4

    START = 5

    STOP = 6

    RELEASE = 7

class LifecycleStatus(enum.Enum):
    OK = 0

    SKIPPED = 1

    RETRYABLE_FAILURE = 2

    FATAL_FAILURE = 3

class LifecycleTransactionState(enum.Enum):
    REJECTED = 0

    ACCEPTED_NO_CHANGE = 1

    QUEUED = 2

    QUIESCING = 3

    RESTARTING = 4

    PUBLISHED = 5

    SUPERSEDED = 6

    CANCELED = 7

    ROLLED_BACK = 8

    FAILED_STOPPED = 9

class LifecycleTransitionPhase(enum.Enum):
    """
    The lifecycle transition phase associated with supplemental diagnostic evidence.
    """

    NONE = 0

    STARTUP_FORWARD = 1

    PRE_RESTART_TEARDOWN = 2

    RESTART_FORWARD = 3

    FORWARD_ROLLBACK = 4

    RESTORE_OLD = 5

    SHUTDOWN_TEARDOWN = 6

class LifecycleTransactionOptions:
    """Explicit bounded capacities for lifecycle parameter transactions."""

    def __init__(self, *, max_pending_requests: int, max_pending_transactions: int, max_receipts: int, max_fenced_operators: int, max_candidate_generations: int, max_events: int, max_audit_records: int, max_quiesce_records: int, max_rollback_records: int) -> None: ...

    @property
    def max_pending_requests(self) -> int: ...

    @max_pending_requests.setter
    def max_pending_requests(self, arg: int, /) -> None: ...

    @property
    def max_pending_transactions(self) -> int: ...

    @max_pending_transactions.setter
    def max_pending_transactions(self, arg: int, /) -> None: ...

    @property
    def max_receipts(self) -> int: ...

    @max_receipts.setter
    def max_receipts(self, arg: int, /) -> None: ...

    @property
    def max_fenced_operators(self) -> int: ...

    @max_fenced_operators.setter
    def max_fenced_operators(self, arg: int, /) -> None: ...

    @property
    def max_candidate_generations(self) -> int: ...

    @max_candidate_generations.setter
    def max_candidate_generations(self, arg: int, /) -> None: ...

    @property
    def max_events(self) -> int: ...

    @max_events.setter
    def max_events(self, arg: int, /) -> None: ...

    @property
    def max_audit_records(self) -> int: ...

    @max_audit_records.setter
    def max_audit_records(self, arg: int, /) -> None: ...

    @property
    def max_quiesce_records(self) -> int: ...

    @max_quiesce_records.setter
    def max_quiesce_records(self, arg: int, /) -> None: ...

    @property
    def max_rollback_records(self) -> int: ...

    @max_rollback_records.setter
    def max_rollback_records(self, arg: int, /) -> None: ...

class LifecycleRestartOptions:
    def __init__(self, *, from_stage: LifecycleStage = LifecycleStage.CONFIGURE, reason_id: int = 0) -> None: ...

    @property
    def from_stage(self) -> LifecycleStage: ...

    @from_stage.setter
    def from_stage(self, arg: LifecycleStage, /) -> None: ...

    @property
    def reason_id(self) -> int: ...

    @reason_id.setter
    def reason_id(self, arg: int, /) -> None: ...

class LifecycleRestartReceipt:
    @property
    def receipt_id(self) -> int: ...

    @property
    def acceptance_sequence(self) -> int: ...

    @property
    def target(self) -> LifecycleOperatorRef: ...

    @property
    def requested_stage(self) -> LifecycleStage: ...

class LifecycleRestartReceiptView:
    @property
    def receipt(self) -> LifecycleRestartReceipt: ...

    @property
    def state(self) -> LifecycleTransactionState: ...

    @property
    def transaction_id(self) -> int: ...

    @property
    def coalesced_into_transaction(self) -> int: ...

    @property
    def related_transaction_id(self) -> int: ...

    @property
    def fence_id(self) -> int: ...

    @property
    def effective_stage(self) -> LifecycleStage: ...

    @property
    def candidate_generation(self) -> int: ...

class OperatorLifecycleBuilder:
    """Register hosted Python lifecycle callbacks by method name.

    START and STOP use the context-free ``Operator.start`` and ``Operator.stop`` hooks and cannot
    be registered here. The bound method object is captured during graph authoring, so replacing
    the class attribute later does not change the callback used by this operator instance.
    """

    def stage(self, stage: LifecycleStage, method: str) -> OperatorLifecycleBuilder: ...

    def optional(self, stage: LifecycleStage, method: str) -> OperatorLifecycleBuilder: ...

__all__: list[str] = ['GraphLifecycleBuilder', 'LifecycleAccessPolicy', 'LifecycleControlHandle', 'LifecycleContext', 'LifecycleOperatorRef', 'LifecycleStage', 'LifecycleStatus', 'LifecycleTransactionState', 'LifecycleTransitionPhase', 'LifecycleTransactionOptions', 'LifecycleRestartOptions', 'LifecycleRestartReceipt', 'LifecycleRestartReceiptView', 'OperatorLifecycleBuilder']
