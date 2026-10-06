# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate deterministic type stubs for registry-backed operator proxies."""

from __future__ import annotations

import argparse
import keyword
import re
import sys
from collections.abc import Iterable, Mapping
from importlib import import_module
from pathlib import Path

# Do not let the script's sibling source package shadow the built extension.
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

from holoscan.registry import (
    OperatorParameterInfo,
    OperatorType,
    _provider_operator_aliases,  # pyright: ignore[reportPrivateUsage]
)

LICENSE_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# fmt: off\n\n"
)


def _member_name(name: str) -> str | None:
    """Return the supported attribute spelling for one registry name."""

    if keyword.iskeyword(name):
        return f"{name}_"
    return name if name.isidentifier() else None


def _runtime_visible_members(
    members: Mapping[str, str], *, hidden: set[str] | None = None
) -> list[tuple[str, str]]:
    """Project dynamic members using the runtime's attribute precedence."""

    visible_members: dict[str, tuple[str, str]] = {}
    for name, annotation in members.items():
        visible = _member_name(name)
        if visible is None or (hidden is not None and visible in hidden):
            continue
        existing = visible_members.get(visible)
        if existing is None or name == visible:
            # A literal trailing-underscore name wins over an escaped keyword,
            # matching _metadata_attribute_name() at runtime.
            visible_members[visible] = (name, annotation)
    return [(visible, annotation) for visible, (_, annotation) in sorted(visible_members.items())]


def _type_fragment(name: str) -> str:
    words = [word for word in re.split(r"[^A-Za-z0-9]+", name) if word]
    fragment = "".join(word[:1].upper() + word[1:] for word in words) or "Field"
    return f"Field{fragment}" if fragment[0].isdigit() else fragment


def _typed_dict_definition(name: str, fields: list[tuple[str, str, bool]]) -> list[str]:
    """Render a TypedDict, retaining non-identifier registry keys when necessary."""

    if all(key.isidentifier() and not keyword.iskeyword(key) for key, _, _ in fields):
        lines = [f"class {name}(TypedDict, total=False):"]
        lines.extend(
            f"    {key}: {'Required[' if required else ''}{annotation}{']' if required else ''}"
            for key, annotation, required in fields
        )
        if not fields:
            lines.append("    pass")
        return lines
    members = ", ".join(
        f"{key!r}: {'Required[' if required else ''}{annotation}{']' if required else ''}"
        for key, annotation, required in fields
    )
    return [f"{name} = TypedDict({name!r}, {{{members}}}, total=False)"]


def _parameter_types(
    alias: str, parameters: Iterable[OperatorParameterInfo]
) -> tuple[list[str], str]:
    """Render the root and recursively nested constructor TypedDicts."""

    definitions: list[str] = []

    def visit(type_name: str, values: Iterable[OperatorParameterInfo]) -> None:
        fields: list[tuple[str, str, bool]] = []
        for parameter in values:
            if parameter.nested_parameters is None:
                annotation = parameter.python_type.replace(
                    "holoscan.time.Duration", "_time_Duration"
                )
            else:
                nested_name = (
                    f"{type_name.removesuffix('Params')}{_type_fragment(parameter.name)}Params"
                )
                visit(nested_name, parameter.nested_parameters.values())
                annotation = nested_name + (" | None" if parameter.optional else "")
            fields.append((parameter.name, annotation, parameter.required))
            if parameter.alias is not None:
                fields.append((parameter.alias, annotation, False))
        definitions.extend((*_typed_dict_definition(type_name, fields), "", ""))

    root_name = f"_{alias}Params"
    visit(root_name, parameters)
    return definitions, root_name


def _module_static_stub(module_doc: str, aliases: dict[str, OperatorType]) -> str:
    """Render the static preamble for a provider-owned Operator facade."""

    exports = ", ".join(repr(alias) for alias in sorted(aliases))
    return "\n".join(
        (
            "# ruff: noqa: E501 - registry metadata intentionally remains on generated lines",
            "",
            f'"""{module_doc}"""',
            "",
            "import holoscan",
            "from holoscan.operators import OperatorHandle as _OperatorHandle",
            "from holoscan.registry import OperatorType as _OperatorType",
            "",
            f"__all__: list[str] = [{exports}]",
        )
    )


def render_operator_proxies(aliases: dict[str, OperatorType]) -> str:
    """Render selected provider-owned registry proxies in stable lexical order."""

    selected_aliases = aliases
    typing_names = ["Final", "Literal", "Never", "Required", "TypedDict", "Unpack"]
    if any(proxy.parameter_introspection == "pass_through" for proxy in selected_aliases.values()):
        typing_names.insert(0, "Any")

    lines = [
        "# Registry-backed operator proxy types.",
        "import datetime",
        f"from typing import {', '.join(typing_names)}",
        "",
        "import holoscan",
        "from holoscan.registry import OperatorParameterInfo as _OperatorParameterInfo",
        "from holoscan.registry import OperatorPortInfo as _OperatorPortInfo",
        "from holoscan.time import Duration as _time_Duration",
        "",
        "",
    ]
    for alias, proxy in sorted(selected_aliases.items()):
        proxy_type = f"_{alias}Type"
        handle_type = f"_{alias}Handle"
        parameter_definitions, params_type = _parameter_types(alias, proxy.parameters.values())
        lines.extend(parameter_definitions)
        lines.append(f"class {handle_type}(_OperatorHandle):")
        handle_members = set(dir(proxy.Handle))
        visible_ports = _runtime_visible_members(
            {name: "holoscan.Port" for name in proxy.ports}, hidden=handle_members
        )
        lines.extend(f"    {name}: {annotation}" for name, annotation in visible_ports)
        lines.extend(
            (
                "    def __getattr__(self, name: Never) -> Never: ...  "
                "# type: ignore[misc, override]",
                "",
                "",
            )
        )
        lines.extend(
            (f"class {proxy_type}(_OperatorType[{handle_type}]):", f"    {proxy.__doc__!r}")
        )
        lines.extend(
            (
                f"    Handle: Final[type[{handle_type}]]",
                f"    Params: Final[type[{params_type}]]",
            )
        )
        fixed_proxy_members = set(vars(proxy)) | set(dir(type(proxy)))
        visible_members = _runtime_visible_members(
            {name: "_OperatorPortInfo" for name in proxy.ports}, hidden=fixed_proxy_members
        )
        visible_names = {name for name, _ in visible_members}
        visible_members.extend(
            (name, annotation)
            for name, annotation in _runtime_visible_members(
                {name: "_OperatorParameterInfo" for name in proxy.parameters},
                hidden=fixed_proxy_members,
            )
            if name not in visible_names
        )
        visible_members.sort()
        lines.extend(f"    {name}: {annotation}" for name, annotation in visible_members)
        if proxy.parameter_introspection == "pass_through":
            lines.append(
                f"    def __call__(self, graph: holoscan.Graph, instance_name: str, "
                f"**params: Any) -> {handle_type}: ...  "
                "# type: ignore[override]  # noqa: ANN401 - hand-authored row"
            )
        else:
            lines.append(
                f"    def __call__(self, graph: holoscan.Graph, instance_name: str, "
                f"**params: Unpack[{params_type}]) -> {handle_type}: ...  "
                "# type: ignore[override]"
            )
        lines.extend(("", f"{alias}: Final[{proxy_type}]", "", ""))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Import a registry provider before projecting its rows",
    )
    parser.add_argument(
        "--provider-module",
        action="append",
        default=[],
        help="Alias for --provider used by component-package build rules",
    )
    parser.add_argument("--registry-prefix", required=True)
    parser.add_argument(
        "--module-doc", default="Provider-owned registered Holoscan Operator proxies."
    )
    args = parser.parse_args()
    for provider in (*args.provider, *args.provider_module):
        import_module(provider)
    aliases = _provider_operator_aliases(args.registry_prefix)
    if not aliases:
        raise ValueError(f"registry prefix {args.registry_prefix!r} selected no operators")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    static_stub = f"{LICENSE_HEADER}{_module_static_stub(args.module_doc, aliases)}"
    args.output.write_text(
        f"{static_stub}\n\n\n{render_operator_proxies(aliases)}", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
