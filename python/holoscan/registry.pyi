# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, overload

import holoscan
from holoscan.operators import OperatorHandle as _OperatorHandle

@dataclass(frozen=True)
class GraphPortInfo:
    name: str
    direction: str
    payload_type: str

@dataclass(frozen=True)
class GraphParameterInfo:
    name: str
    python_type: str
    description: str
    alias: str | None
    unit: str
    optional: bool
    required: bool
    nested: bool
    nested_schema_id: str | None
    nested_parameters: Mapping[str, GraphParameterInfo] | None
    sensitive: bool
    value_range: tuple[object, object] | None
    choices: tuple[tuple[str | None, object], ...]

@dataclass(frozen=True)
class OperatorPortInfo:
    name: str
    direction: str
    payload_type: str

@dataclass(frozen=True)
class OperatorConstructorHandleInfo:
    name: str
    type_name: str
    description: str
    required: bool

@dataclass(frozen=True)
class OperatorParameterInfo:
    name: str
    python_type: str
    description: str
    alias: str | None
    unit: str
    optional: bool
    required: bool
    nested: bool
    nested_schema_id: str | None
    nested_parameters: Mapping[str, OperatorParameterInfo] | None
    sensitive: bool
    value_range: tuple[object, object] | None
    choices: tuple[tuple[str | None, object], ...]

class GraphType[HandleT]:
    description: str
    parameter_introspection: str
    parameter_summary: str
    ports: Mapping[str, GraphPortInfo]
    parameters: Mapping[str, GraphParameterInfo]
    def __init__(self, metadata: Mapping[str, Any]) -> None: ...
    @property
    def name(self) -> str: ...
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Any) -> HandleT: ...  # noqa: ANN401 - dynamically resolved type
    def __getattr__(self, name: str) -> GraphPortInfo | GraphParameterInfo: ...
    def __dir__(self) -> list[str]: ...
    def __repr__(self) -> str: ...

class OperatorType[HandleT]:
    description: str
    parameter_introspection: str
    parameter_summary: str
    ports: Mapping[str, OperatorPortInfo]
    parameters: Mapping[str, OperatorParameterInfo]
    required_parameters: frozenset[str]
    constructor_handles: Mapping[str, OperatorConstructorHandleInfo]
    def __init__(self, metadata: Mapping[str, Any]) -> None: ...
    @property
    def name(self) -> str: ...
    def __call__(self, graph: holoscan.Graph, instance_name: str, **params: Any) -> HandleT: ...  # noqa: ANN401 - dynamically resolved type
    def __getattr__(self, name: str) -> OperatorPortInfo | OperatorParameterInfo: ...
    def __dir__(self) -> list[str]: ...
    def __repr__(self) -> str: ...

def graph_types() -> tuple[GraphType[holoscan.GraphHandle], ...]: ...
def operators() -> tuple[OperatorType[_OperatorHandle], ...]: ...
@overload
def resolve(name: str, *, kind: Literal["graph"]) -> GraphType[holoscan.GraphHandle]: ...
@overload
def resolve(name: str, *, kind: Literal["operator"]) -> OperatorType[_OperatorHandle]: ...
@overload
def resolve(
    name: str, *, kind: None = None
) -> GraphType[holoscan.GraphHandle] | OperatorType[_OperatorHandle]: ...
def _importable_graph_aliases() -> dict[str, GraphType[holoscan.GraphHandle]]: ...
def _provider_operator_aliases(
    registry_prefix: str,
) -> dict[str, OperatorType[_OperatorHandle]]: ...
