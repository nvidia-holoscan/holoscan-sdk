# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate deterministic type stubs for registry-backed operator proxies.

Run it with the interpreter and import path of the Holoscan build it describes, either by path or
as ``python -m holoscan._tools.generate_operator_stubs``. CMake projects use
``holoscan_add_operator_stubs()`` from ``HoloscanStubGeneration.cmake``.
"""

from __future__ import annotations

import argparse
import ast
import keyword
import re
import unicodedata
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from holoscan.registry import (
    OperatorParameterInfo,
    OperatorType,
    _provider_operator_aliases,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

LICENSE_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)
# Generated stubs are byte-compared with committed copies, so formatters must leave them alone.
FORMATTER_DIRECTIVE = "# fmt: off\n\n"


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


def _is_literal_identifier(name: str) -> bool:
    """Return whether Python keeps ``name`` verbatim as a class-body field or parameter name.

    Besides keywords and non-identifiers, Python rejects `__debug__` as a target, NFKC-normalizes
    identifiers (so fullwidth or compatibility characters can change), and mangles class-private
    names such as `__channel` inside a class body. Any of these would describe a different key than
    the registry name that runtime routing uses.
    """

    return (
        name.isidentifier()
        and not keyword.iskeyword(name)
        and name != "__debug__"
        and unicodedata.normalize("NFKC", name) == name
        and not (name.startswith("__") and not name.endswith("__"))
    )


def _typed_dict_definition(name: str, fields: list[tuple[str, str, bool]]) -> list[str]:
    """Render a TypedDict, retaining registry keys that class syntax would change or reject."""

    if all(_is_literal_identifier(key) for key, _, _ in fields):
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
    alias: str,
    parameters: Iterable[OperatorParameterInfo],
    root_fields: Iterable[tuple[str, str, bool]] = (),
) -> tuple[list[str], str]:
    """Render the root and recursively nested constructor TypedDicts."""

    definitions: list[str] = []
    root_name = f"_{alias}Params"

    def visit(type_name: str, values: Iterable[OperatorParameterInfo]) -> None:
        fields: list[tuple[str, str, bool]] = list(root_fields) if type_name == root_name else []
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

    visit(root_name, parameters)
    return definitions, root_name


# Python annotations for SDK built-in constructor-handle types; providers map their own types.
_BUILTIN_HANDLE_ANNOTATIONS = {"holoscan.VoidCallback": "collections.abc.Callable[[], object]"}

# Parameter names that the generated pass-through ``__call__`` signature already uses.
_PASS_THROUGH_SIGNATURE_NAMES = frozenset({"self", "graph", "instance_name", "params"})


def _check_keyword_handle_name(alias: str, name: str) -> None:
    """Reject a handle name that cannot be a keyword-only parameter of a pass-through signature.

    Declared-schema rows place handles in a TypedDict, which accepts any string key. Pass-through
    rows spell handles as keyword-only parameters, so a name that Python would reject or rewrite
    (see `_is_literal_identifier`), or one the signature already uses, would produce invalid Python
    or describe a different keyword. Renaming it would describe a different runtime API, so
    generation fails instead.
    """

    if not _is_literal_identifier(name) or name in _PASS_THROUGH_SIGNATURE_NAMES:
        raise ValueError(
            f"constructor handle input {name!r} of {alias} cannot be a keyword parameter of the "
            "generated pass-through signature (Python would reject or rewrite it, or it is one "
            "of self, graph, instance_name, params); rename the input or give the row a declared "
            "schema"
        )


def _handle_fields(
    alias: str, proxy: OperatorType, handle_annotations: Mapping[str, str]
) -> list[tuple[str, str, bool]]:
    """Return typed keyword fields for a proxy's C++ constructor-handle arguments."""

    fields: list[tuple[str, str, bool]] = []
    for handle in proxy.constructor_handles.values():
        if proxy.parameter_introspection == "pass_through":
            _check_keyword_handle_name(alias, handle.name)
        annotation = handle_annotations.get(
            handle.type_name, _BUILTIN_HANDLE_ANNOTATIONS.get(handle.type_name)
        )
        if annotation is None:
            raise ValueError(
                f"no --handle-annotation for constructor handle type {handle.type_name!r} "
                f"used by {alias}.{handle.name}"
            )
        fields.append(
            (
                handle.name,
                annotation if handle.required else f"{annotation} | None",
                handle.required,
            )
        )
    return fields


def _module_static_stub(module_doc: str, aliases: dict[str, OperatorType]) -> str:
    """Render the static preamble for a provider-owned Operator facade."""

    exports = ", ".join(repr(alias) for alias in sorted(aliases))
    return "\n".join(
        (
            "# ruff: noqa: E501 - registry metadata intentionally remains on generated lines",
            "",
            f'"""{module_doc}"""',
            "",
            "from holoscan.operators import OperatorHandle as _OperatorHandle",
            "from holoscan.registry import OperatorType as _OperatorType",
            "",
            f"__all__: list[str] = [{exports}]",
        )
    )


def render_operator_proxies(
    aliases: dict[str, OperatorType], handle_annotations: Mapping[str, str] | None = None
) -> str:
    """Render selected provider-owned registry proxies in stable lexical order.

    ``handle_annotations`` maps each constructor-handle stable type name to the fully qualified
    Python class that provides it, for example ``{"holoscan.hsb.DataChannel":
    "hsb.DataChannel"}``. A handle type without a mapping is an error rather than ``Any``.
    """

    selected_aliases = aliases
    annotations = {**_BUILTIN_HANDLE_ANNOTATIONS, **(handle_annotations or {})}
    handle_fields = {
        alias: _handle_fields(alias, proxy, annotations)
        for alias, proxy in selected_aliases.items()
    }
    used_annotations = {
        annotations[handle.type_name]
        for proxy in selected_aliases.values()
        for handle in proxy.constructor_handles.values()
    }
    # The proxy section always imports holoscan itself for holoscan.Graph and holoscan.Port.
    handle_modules = sorted(
        {
            annotation.rpartition(".")[0]
            for annotation in used_annotations
            if "." in annotation and annotation.rpartition(".")[0] != "holoscan"
        }
    )

    lines: list[str] = []
    for alias, proxy in sorted(selected_aliases.items()):
        proxy_type = f"_{alias}Type"
        handle_type = f"_{alias}Handle"
        declared_handles = (
            handle_fields[alias] if proxy.parameter_introspection != "pass_through" else []
        )
        parameter_definitions, params_type = _parameter_types(
            alias, proxy.parameters.values(), declared_handles
        )
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
            keyword_handles = "".join(
                f"{name}: {annotation}{'' if required else ' = None'}, "
                for name, annotation, required in handle_fields[alias]
            )
            separator = "*, " if keyword_handles else ""
            lines.append(
                f"    def __call__(self, graph: holoscan.Graph, instance_name: str, "
                f"{separator}{keyword_handles}**params: Any) -> {handle_type}: ...  "
                "# type: ignore[override]  # noqa: ANN401 - hand-authored row"
            )
        else:
            lines.append(
                f"    def __call__(self, graph: holoscan.Graph, instance_name: str, "
                f"**params: Unpack[{params_type}]) -> {handle_type}: ...  "
                "# type: ignore[override]"
            )
        lines.extend(("", f"{alias}: Final[{proxy_type}]", "", ""))

    # Import only what the rendered declarations reference, so a committed copy is lint-clean.
    used = _referenced_names("\n".join(lines))
    typing_names = [
        name
        for name in ("Any", "Final", "Literal", "Never", "Required", "TypedDict", "Unpack")
        if name in used
    ]
    standard_imports = [
        *(["import datetime"] if "datetime" in used else []),
        *([f"from typing import {', '.join(typing_names)}"] if typing_names else []),
    ]
    holoscan_imports = [
        "import holoscan",
        *(f"import {module}" for module in handle_modules),
        *(
            f"from {module} import {name} as {alias}"
            for module, name, alias in (
                ("holoscan.registry", "OperatorParameterInfo", "_OperatorParameterInfo"),
                ("holoscan.registry", "OperatorPortInfo", "_OperatorPortInfo"),
                ("holoscan.time", "Duration", "_time_Duration"),
            )
            if alias in used
        ),
    ]
    header = [
        "# Registry-backed operator proxy types.",
        *standard_imports,
        *([""] if standard_imports else []),
        *holoscan_imports,
        "",
        "",
    ]
    return "\n".join([*header, *lines]).rstrip() + "\n"


def _referenced_names(source: str) -> set[str]:
    """Return names loaded by expressions in the rendered stub declarations."""

    return {
        node.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


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
        "--handle-annotation",
        action="append",
        default=[],
        metavar="TYPE_NAME=MODULE.CLASS",
        help="Map a constructor-handle stable type name to its fully qualified Python class",
    )
    parser.add_argument(
        "--module-doc", default="Provider-owned registered Holoscan Operator proxies."
    )
    parser.add_argument(
        "--header-file",
        type=Path,
        help="Use this file's comment lines as the stub's license header instead of the SDK's",
    )
    args = parser.parse_args()
    for provider in (*args.provider, *args.provider_module):
        import_module(provider)
    aliases = _provider_operator_aliases(args.registry_prefix)
    if not aliases:
        raise ValueError(f"registry prefix {args.registry_prefix!r} selected no operators")
    handle_annotations: dict[str, str] = {}
    for mapping in args.handle_annotation:
        type_name, separator, annotation = mapping.partition("=")
        # The stub imports MODULE and names MODULE.CLASS, so both must be a dotted Python path.
        module, _, class_name = annotation.rpartition(".")
        if (
            not separator
            or not type_name
            or not module
            or not all(part.isidentifier() for part in (*module.split("."), class_name))
        ):
            raise ValueError(f"--handle-annotation expects TYPE_NAME=MODULE.CLASS, got {mapping!r}")
        handle_annotations[type_name] = annotation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = LICENSE_HEADER
    if args.header_file is not None:
        header = args.header_file.read_text(encoding="utf-8").rstrip("\n") + "\n"
        if not all(line.startswith("#") for line in header.splitlines()):
            raise ValueError(f"--header-file must contain only comment lines: {args.header_file}")
    static_stub = f"{header}{FORMATTER_DIRECTIVE}{_module_static_stub(args.module_doc, aliases)}"
    args.output.write_text(
        f"{static_stub}\n\n\n{render_operator_proxies(aliases, handle_annotations)}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
