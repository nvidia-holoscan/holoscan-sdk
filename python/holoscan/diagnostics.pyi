# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Compile and runtime diagnostic values."""

from holoscan.errors import ErrorCode as _errors_ErrorCode
from holoscan.lifecycle import LifecycleStage as _lifecycle_LifecycleStage
from holoscan.lifecycle import LifecycleTransitionPhase as _lifecycle_LifecycleTransitionPhase

import enum


class Diagnostic:
    """A stable structured compile or runtime diagnostic."""

    @property
    def id(self) -> int: ...

    @property
    def severity(self) -> DiagnosticSeverity: ...

    @property
    def token(self) -> str: ...

    @property
    def source(self) -> DiagnosticSource: ...

    @property
    def service(self) -> str: ...

    @property
    def endpoint(self) -> str: ...

    @property
    def json_pointer(self) -> str: ...

    @property
    def required(self) -> str: ...

    @property
    def provided(self) -> str: ...

    @property
    def cause_id(self) -> int | None: ...

    @property
    def detail(self) -> DiagnosticDetail | None: ...

    def __repr__(self) -> str: ...

class DiagnosticDetail:
    """Read-only supplemental failure evidence retained by a diagnostic."""

    @property
    def operation(self) -> DiagnosticOperation: ...

    @property
    def error(self) -> DiagnosticError: ...

    @property
    def stage(self) -> _lifecycle_LifecycleStage | None: ...

    @property
    def phase(self) -> _lifecycle_LifecycleTransitionPhase: ...

    @property
    def transaction_id(self) -> int: ...

    @property
    def role(self) -> DiagnosticFailureRole: ...

    @property
    def operator_path(self) -> str: ...

    @property
    def port_name(self) -> str: ...

    @property
    def related_source(self) -> DiagnosticSource: ...

    @property
    def related_operator_path(self) -> str: ...

    @property
    def related_port_name(self) -> str: ...

    @property
    def mismatches(self) -> list[DiagnosticMismatch]: ...

    @property
    def causes(self) -> list[DiagnosticError]: ...

    @property
    def primary_diagnostic_index(self) -> int | None: ...

    @property
    def truncated(self) -> bool: ...

class DiagnosticError:
    """
    A read-only error category and bounded explanatory message retained by a diagnostic.
    """

    @property
    def code(self) -> _errors_ErrorCode: ...

    @property
    def message(self) -> str: ...

class DiagnosticFailureRole(enum.Enum):
    """
    The failure's relationship to its initiating lifecycle or runtime operation.
    """

    PRIMARY = 0

    ROLLBACK = 1

    RESTORATION = 2

    TEARDOWN = 3

class DiagnosticMismatch:
    """
    One read-only provided-versus-required field mismatch retained by a diagnostic.
    """

    @property
    def field(self) -> str: ...

    @property
    def provided(self) -> str: ...

    @property
    def required(self) -> str: ...

class DiagnosticOperation(enum.Enum):
    """The Core operation that produced supplemental diagnostic evidence."""

    UNKNOWN = 0

    COMPOSE = 1

    CONSTRUCT = 2

    SETUP = 3

    CONTRACT = 4

    LIFECYCLE = 5

    COMPUTE = 6

    RUNTIME = 7

    COMPILE = 8

class DiagnosticSeverity(enum.Enum):
    WARNING = 0

    ERROR = 1

class DiagnosticSource:
    """A bounded plan-local location associated with a diagnostic."""

    @property
    def graph_generation(self) -> int: ...

    @property
    def graph(self) -> int: ...

    @property
    def vertex(self) -> int: ...

    @property
    def port(self) -> int: ...

__all__: list[str] = ['Diagnostic', 'DiagnosticDetail', 'DiagnosticError', 'DiagnosticFailureRole', 'DiagnosticMismatch', 'DiagnosticOperation', 'DiagnosticSeverity', 'DiagnosticSource']
