# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover Graph and Operator types registered by loaded Holoscan modules."""

from __future__ import annotations as _annotations

import keyword
from dataclasses import dataclass
from difflib import get_close_matches
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from . import Graph

from ._public_api import public_api_names as _public_api_names

_core = cast("Any", import_module("holoscan._core"))


@dataclass(frozen=True)
class GraphPortInfo:
    """Static metadata for one exported reusable-Graph interface port."""

    name: str
    direction: str
    payload_type: str

    def __repr__(self) -> str:
        return f"<GraphPortInfo {self.direction} {self.name!r}: {self.payload_type}>"


@dataclass(frozen=True)
class GraphParameterInfo:
    """Static metadata for one registered reusable-Graph constructor parameter."""

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

    def __repr__(self) -> str:
        return f"<GraphParameterInfo {self.name!r}: {self.python_type}>"


@dataclass(frozen=True)
class OperatorPortInfo:
    """Static metadata for one registered Operator port."""

    name: str
    direction: str
    payload_type: str

    def __repr__(self) -> str:
        return f"<OperatorPortInfo {self.direction} {self.name!r}: {self.payload_type}>"


@dataclass(frozen=True)
class OperatorParameterInfo:
    """Static metadata for one declared Operator parameter."""

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

    def __repr__(self) -> str:
        return f"<OperatorParameterInfo {self.name!r}: {self.python_type}>"


def _parameter_values[ParameterInfoT: (GraphParameterInfo, OperatorParameterInfo)](
    metadata: Mapping[str, Any],
    info_type: type[ParameterInfoT],
) -> dict[str, ParameterInfoT]:
    values: dict[str, ParameterInfoT] = {}
    for item in metadata["parameters"]:
        nested = item["nested_parameters"]
        nested_parameters = (
            None
            if nested is None
            else MappingProxyType(_parameter_values({"parameters": nested}, info_type))
        )
        values[str(item["name"])] = info_type(
            name=str(item["name"]),
            python_type=str(item["python_type"]),
            description=str(item["description"]),
            alias=None if item["alias"] is None else str(item["alias"]),
            unit=str(item["unit"]),
            optional=bool(item["optional"]),
            required=bool(item["required"]),
            nested=bool(item["nested"]),
            nested_schema_id=(
                None if item["nested_schema_id"] is None else str(item["nested_schema_id"])
            ),
            nested_parameters=cast("Any", nested_parameters),
            sensitive=bool(item["sensitive"]),
            value_range=None if item["range"] is None else tuple(item["range"]),
            choices=tuple(
                (
                    None if choice["name"] is None else str(choice["name"]),
                    choice["value"],
                )
                for choice in item["choices"]
            ),
        )
    return values


def _metadata_attribute_name(name: str, members: Mapping[str, object]) -> str:
    """Map the PEP 8 trailing underscore back to a keyword registry name."""

    if name in members:
        return name
    if name.endswith("_") and keyword.iskeyword(name[:-1]) and name[:-1] in members:
        return name[:-1]
    return name


def _visible_metadata_name(name: str) -> str | None:
    if keyword.iskeyword(name):
        return f"{name}_"
    return name if name.isidentifier() else None


class GraphType:
    """A discoverable token for one registered C++ reusable Graph type."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self._registry_name = str(metadata["name"])
        self.description = str(metadata["description"])
        self.parameter_introspection = str(metadata["parameter_introspection"])
        self.parameter_summary = str(metadata["parameter_summary"])
        ports = {
            str(item["name"]): GraphPortInfo(
                name=str(item["name"]),
                direction=str(item["direction"]),
                payload_type=str(item["payload_type"]),
            )
            for item in metadata["ports"]
        }
        parameters = _parameter_values(metadata, GraphParameterInfo)
        self.ports: Mapping[str, GraphPortInfo] = MappingProxyType(ports)
        self.parameters: Mapping[str, GraphParameterInfo] = MappingProxyType(parameters)

    @property
    def name(self) -> str:
        """Return the stable registry name."""

        return self._registry_name

    def __call__(self, graph: Graph, instance_name: str, **params: object) -> object:
        """Embed this registered type into ``graph``."""

        return cast("Any", graph).embed(self, instance_name, **params)

    def __getattr__(self, name: str) -> GraphPortInfo | GraphParameterInfo:
        if name.startswith("__"):
            raise AttributeError(name)
        attributes = vars(self)
        ports = attributes.get("ports")
        if ports is not None:
            port_name = _metadata_attribute_name(name, ports)
            if port_name in ports:
                return ports[port_name]
        parameters = attributes.get("parameters")
        if parameters is not None:
            parameter_name = _metadata_attribute_name(name, parameters)
            if parameter_name in parameters:
                return parameters[parameter_name]
        raise AttributeError(f"{self.__class__.__name__!s} has no port or parameter {name!r}")

    def __dir__(self) -> list[str]:
        attributes = vars(self)
        dynamic = {
            visible
            for name in (*attributes.get("ports", ()), *attributes.get("parameters", ()))
            if (visible := _visible_metadata_name(name)) is not None
        }
        return sorted(set(super().__dir__()) | dynamic)

    def __repr__(self) -> str:
        return f"<GraphType {vars(self).get('_registry_name', '<uninitialized>')!r}>"


class OperatorType:
    """A discoverable token for one registered C++ Operator type."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        self._registry_name = str(metadata["name"])
        self.description = str(metadata["description"])
        self.parameter_introspection = str(metadata["parameter_introspection"])
        self.parameter_summary = str(metadata["parameter_summary"])
        ports = {
            str(item["name"]): OperatorPortInfo(
                name=str(item["name"]),
                direction=str(item["direction"]),
                payload_type=str(item["payload_type"]),
            )
            for item in metadata["ports"]
        }
        parameters = _parameter_values(metadata, OperatorParameterInfo)
        self.ports: Mapping[str, OperatorPortInfo] = MappingProxyType(ports)
        self.parameters: Mapping[str, OperatorParameterInfo] = MappingProxyType(parameters)
        self.required_parameters = frozenset(str(name) for name in metadata["required_parameters"])

    @property
    def name(self) -> str:
        """Return the stable registry name."""

        return self._registry_name

    def __call__(self, graph: Graph, instance_name: str, **params: object) -> object:
        """Create this registered type in ``graph``."""

        return cast("Any", graph).op(self, instance_name, **params)

    def __getattr__(self, name: str) -> OperatorPortInfo | OperatorParameterInfo:
        if name.startswith("__"):
            raise AttributeError(name)
        attributes = vars(self)
        ports = attributes.get("ports")
        if ports is not None:
            port_name = _metadata_attribute_name(name, ports)
            if port_name in ports:
                return ports[port_name]
        parameters = attributes.get("parameters")
        if parameters is not None:
            parameter_name = _metadata_attribute_name(name, parameters)
            if parameter_name in parameters:
                return parameters[parameter_name]
        raise AttributeError(f"{self.__class__.__name__!s} has no port or parameter {name!r}")

    def __dir__(self) -> list[str]:
        attributes = vars(self)
        dynamic = {
            visible
            for name in (*attributes.get("ports", ()), *attributes.get("parameters", ()))
            if (visible := _visible_metadata_name(name)) is not None
        }
        return sorted(set(super().__dir__()) | dynamic)

    def __repr__(self) -> str:
        return f"<OperatorType {vars(self).get('_registry_name', '<uninitialized>')!r}>"


def _type_doc(
    metadata: Mapping[str, Any], *, kind: str, ports_heading: str, parameters_heading: str
) -> str:
    lines = [str(metadata["description"]) or f"Registered {kind} {metadata['name']}."]
    lines.extend(("", f"Registry name: {metadata['name']}"))
    ports = metadata["ports"]
    if ports:
        lines.extend(("", f"{ports_heading}:"))
        lines.extend(
            (f"  {port['name']}: {str(port['direction']).title()}[{port['payload_type']}]")
            for port in ports
        )
    parameters = metadata["parameters"]
    if parameters:
        lines.extend(("", f"{parameters_heading}:"))
        for parameter in parameters:
            details = [str(parameter["python_type"])]
            if parameter["range"] is not None:
                details.append(f"range {tuple(parameter['range'])!r}")
            named_choices = [
                choice["name"] for choice in parameter["choices"] if choice["name"] is not None
            ]
            if named_choices:
                details.append("choices " + ", ".join(map(str, named_choices)))
            description = str(parameter["description"])
            suffix = f" — {description}" if description else ""
            lines.append(f"  {parameter['name']}: {', '.join(details)}{suffix}")
    elif metadata["parameter_introspection"] == "pass_through":
        lines.extend(("", f"{parameters_heading}: accepted by a hand-authored row."))
    return "\n".join(lines)


_graph_proxies: dict[str, GraphType] = {}
_operator_proxies: dict[str, OperatorType] = {}
_graph_registry_counts: tuple[int, int, int, int] | None = None
_operator_registry_counts: tuple[int, int, int, int] | None = None


def _refresh_graphs() -> None:
    global _graph_registry_counts
    registry_counts = _core._registered_registry_counts()
    if registry_counts == _graph_registry_counts:
        return
    for metadata in _core._registered_graph_descriptions():
        name = str(metadata["name"])
        if name in _graph_proxies:
            continue
        class_name = name.rsplit(".", 1)[-1]
        proxy_class = type(
            class_name,
            (GraphType,),
            {
                "__doc__": _type_doc(
                    metadata,
                    kind="reusable Graph",
                    ports_heading="Interface ports",
                    parameters_heading="Constructor parameters",
                ),
                "__module__": "holoscan.graphs",
                "Handle": _core.GraphHandle,
                "Params": dict,
            },
        )
        _graph_proxies[name] = proxy_class(metadata)
    _graph_registry_counts = registry_counts


def _refresh_operators() -> None:
    global _operator_registry_counts
    registry_counts = _core._registered_registry_counts()
    if registry_counts == _operator_registry_counts:
        return
    for metadata in _core._registered_operator_descriptions():
        name = str(metadata["name"])
        if name in _operator_proxies:
            continue
        class_name = name.rsplit(".", 1)[-1]
        proxy_class = type(
            class_name,
            (OperatorType,),
            {
                "__doc__": _type_doc(
                    metadata,
                    kind="Operator",
                    ports_heading="Ports",
                    parameters_heading="Parameters",
                ),
                "__module__": "holoscan.registry",
                "Handle": _core.OperatorHandle,
                "Params": dict,
            },
        )
        _operator_proxies[name] = proxy_class(metadata)
    _operator_registry_counts = registry_counts


def graph_types() -> tuple[GraphType, ...]:
    """Return registered reusable-Graph proxies in stable lexical-name order."""

    _refresh_graphs()
    return tuple(_graph_proxies[name] for name in sorted(_graph_proxies))


def operators() -> tuple[OperatorType, ...]:
    """Return registered Operator proxies in stable lexical-name order."""

    _refresh_operators()
    return tuple(_operator_proxies[name] for name in sorted(_operator_proxies))


def resolve(
    name: str, *, kind: Literal["graph", "operator"] | None = None
) -> GraphType | OperatorType:
    """Resolve one fully qualified Graph or Operator registry name."""

    if kind not in (None, "graph", "operator"):
        raise ValueError("kind must be 'graph', 'operator', or None")
    selected = []
    if kind in (None, "graph"):
        _refresh_graphs()
        selected.append(_graph_proxies)
    if kind in (None, "operator"):
        _refresh_operators()
        selected.append(_operator_proxies)
    matches = [proxies[name] for proxies in selected if name in proxies]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"ambiguous registered type {name!r}; pass kind='graph' or kind='operator'")
    known_names = sorted({proxy_name for proxies in selected for proxy_name in proxies})
    close_matches = get_close_matches(name, known_names, n=3, cutoff=0.35)
    suffix = f"; close matches: {', '.join(close_matches)}" if close_matches else ""
    raise KeyError(f"unknown registered type {name!r}; did you import its provider module?{suffix}")


def _importable_graph_aliases() -> dict[str, GraphType]:  # pyright: ignore[reportUnusedFunction]
    """Return unambiguous leaf-name aliases for :mod:`holoscan.graphs`."""

    return _importable_aliases(graph_types())


def _provider_operator_aliases(
    registry_prefix: str,
) -> dict[str, OperatorType]:  # pyright: ignore[reportUnusedFunction]
    """Return importable Operator aliases owned by one provider prefix."""

    aliases: dict[str, OperatorType] = {}
    for proxy in operators():
        if not proxy.name.startswith(registry_prefix):
            continue
        alias = proxy.name.removeprefix(registry_prefix)
        if not alias.isidentifier() or keyword.iskeyword(alias):
            raise ValueError(f"registry name {proxy.name!r} has no importable provider alias")
        if alias in aliases:
            raise ValueError(f"registry prefix {registry_prefix!r} has duplicate alias {alias!r}")
        aliases[alias] = proxy
    return aliases


def _importable_aliases[ProxyT: (GraphType, OperatorType)](
    proxies: tuple[ProxyT, ...],
) -> dict[str, ProxyT]:
    aliases: dict[str, ProxyT] = {}
    collisions: set[str] = set()
    for proxy in proxies:
        alias = proxy.name.rsplit(".", 1)[-1]
        if not alias.isidentifier() or keyword.iskeyword(alias) or alias in collisions:
            continue
        if alias in aliases:
            aliases.pop(alias)
            collisions.add(alias)
            continue
        aliases[alias] = proxy
    return aliases


__all__ = _public_api_names(__name__)  # noqa: PLE0605
