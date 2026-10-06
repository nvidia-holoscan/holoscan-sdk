# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Holoscan error codes and exception types."""

import enum


class ErrorCode(enum.IntEnum):
    """Stable Holoscan SDK error category."""

    SUCCESS = 0

    FAILURE = 1

    RECEIVE_ERROR = 2

    CODEC_ERROR = 3

    INVALID_ARGUMENT = 4

    NOT_FOUND = 5

    DUPLICATE_NAME = 6

    NOT_IMPLEMENTED = 7

    BACKPRESSURED = 8

    NOT_READY = 9

    STALE_ENDPOINT = 10

    CLOSED = 11

    RESOURCE_EXHAUSTED = 12

    TIMEOUT = 13

    PERMISSION_DENIED = 14

    INTERRUPTED = 15

    PROTOCOL_ERROR = 16

    INCOMPATIBLE = 17

    INPUT_EMPTY = 18

    INPUT_NOT_SELECTED = 19

    INPUT_UNDECLARED = 20

    INPUT_SELECTION_CARDINALITY_MISMATCH = 21

    PUBLICATION_INDETERMINATE = 22

    NOT_SUPPORTED = 23

    LIFECYCLE_INVALID_RESTART_STAGE = 24

    LIFECYCLE_TRANSACTIONS_DISABLED = 25

    STALE_OPERATOR_REF = 26

    CROSS_SESSION_OPERATOR_REF = 27

    LIFECYCLE_RESTART_UNAUTHORIZED = 28

    LIFECYCLE_RESTART_QUEUE_FULL = 29

    LIFECYCLE_RESTART_UNSUPPORTED_IN_PROFILE = 30

    LIFECYCLE_INVALID_CALLING_CONTEXT = 31

    LIFECYCLE_RUN_STATE = 32

    LIFECYCLE_QUIESCE_FAILED = 33

    LIFECYCLE_STAGE_FAILED = 34

    LIFECYCLE_ROLLBACK_FAILED = 35

    PUBLICATION_PARTIAL = 36

    RATE_BINDING_INVALID = 37

    RATE_BINDING_ABOVE_MAXIMUM = 38

    RATE_BINDING_CONFLICT = 39

    ARITHMETIC_OVERFLOW = 40

class HoloscanError(Exception):
    """
    Base exception for Holoscan SDK failures.

    Every instance carries a stable ``code`` and ``token``. Branch on those fields
    instead of matching the human-readable message.
    """

class FailureError(HoloscanError):
    pass

class ReceiveError(HoloscanError):
    pass

class CodecError(HoloscanError):
    pass

class InvalidArgumentError(HoloscanError):
    pass

class NotFoundError(HoloscanError):
    pass

class DuplicateNameError(HoloscanError):
    pass

class HoloscanNotImplementedError(HoloscanError):
    pass

class BackpressuredError(HoloscanError):
    pass

class NotReadyError(HoloscanError):
    pass

class StaleEndpointError(HoloscanError):
    pass

class ClosedError(HoloscanError):
    pass

class ResourceExhaustedError(HoloscanError):
    pass

class HoloscanTimeoutError(HoloscanError):
    pass

class PermissionDeniedError(HoloscanError):
    pass

class HoloscanInterruptedError(HoloscanError):
    pass

class ProtocolError(HoloscanError):
    pass

class IncompatibleError(HoloscanError):
    pass

class InputEmptyError(HoloscanError):
    pass

class InputNotSelectedError(HoloscanError):
    pass

class InputUndeclaredError(HoloscanError):
    pass

class InputSelectionCardinalityMismatchError(HoloscanError):
    pass

class PublicationIndeterminateError(HoloscanError):
    pass

class NotSupportedError(HoloscanError):
    pass

class LifecycleInvalidRestartStageError(HoloscanError):
    pass

class LifecycleTransactionsDisabledError(HoloscanError):
    pass

class StaleOperatorRefError(HoloscanError):
    pass

class CrossSessionOperatorRefError(HoloscanError):
    pass

class LifecycleRestartUnauthorizedError(HoloscanError):
    pass

class LifecycleRestartQueueFullError(HoloscanError):
    pass

class LifecycleRestartUnsupportedInProfileError(HoloscanError):
    pass

class LifecycleInvalidCallingContextError(HoloscanError):
    pass

class LifecycleRunStateError(HoloscanError):
    pass

class LifecycleQuiesceFailedError(HoloscanError):
    pass

class LifecycleStageFailedError(HoloscanError):
    pass

class LifecycleRollbackFailedError(HoloscanError):
    pass

class PublicationPartialError(HoloscanError):
    pass

class RateBindingInvalidError(HoloscanError):
    pass

class RateBindingAboveMaximumError(HoloscanError):
    pass

class RateBindingConflictError(HoloscanError):
    pass

class ArithmeticOverflowError(HoloscanError):
    pass

class ConsumedLoanError(HoloscanError):
    """A single-use Tensor loan was accessed after commit or abandonment."""

    pass

class CompileError(HoloscanError):
    pass

__all__: list[str] = ['ErrorCode', 'HoloscanError', 'FailureError', 'ReceiveError', 'CodecError', 'InvalidArgumentError', 'NotFoundError', 'DuplicateNameError', 'HoloscanNotImplementedError', 'BackpressuredError', 'NotReadyError', 'StaleEndpointError', 'ClosedError', 'ResourceExhaustedError', 'HoloscanTimeoutError', 'PermissionDeniedError', 'HoloscanInterruptedError', 'ProtocolError', 'IncompatibleError', 'InputEmptyError', 'InputNotSelectedError', 'InputUndeclaredError', 'InputSelectionCardinalityMismatchError', 'PublicationIndeterminateError', 'NotSupportedError', 'LifecycleInvalidRestartStageError', 'LifecycleTransactionsDisabledError', 'StaleOperatorRefError', 'CrossSessionOperatorRefError', 'LifecycleRestartUnauthorizedError', 'LifecycleRestartQueueFullError', 'LifecycleRestartUnsupportedInProfileError', 'LifecycleInvalidCallingContextError', 'LifecycleRunStateError', 'LifecycleQuiesceFailedError', 'LifecycleStageFailedError', 'LifecycleRollbackFailedError', 'PublicationPartialError', 'RateBindingInvalidError', 'RateBindingAboveMaximumError', 'RateBindingConflictError', 'ArithmeticOverflowError', 'ConsumedLoanError', 'CompileError']
