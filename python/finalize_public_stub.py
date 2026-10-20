# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add the repository header to one nanobind-generated public facade stub."""

from __future__ import annotations

import argparse
import io
import re
import tokenize
from collections.abc import Callable
from pathlib import Path

_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# fmt: off\n\n"
)

_PUBLIC_TYPE_REFERENCE = re.compile(
    r"(?<![\w.])(?:holoscan\.)?"
    r"(?P<module>clocks|config|diagnostics|errors|execution|graphs|lifecycle|operators|params|"
    r"payload|registry|tensor|time|topics)\."
    r"(?P<symbol>[A-Z][A-Za-z0-9_]*)"
)

_GRAPH_OP_SIGNATURE = (
    "    def op(self, type_or_name: _registry_OperatorType | type[Operator] | str, "
    "instance_name: str, "
    "**params: object) -> _operators_OperatorHandle:"
)
_GRAPH_EMBED_SIGNATURE = (
    "    def embed(self, graph_type: _registry_GraphType | type[Graph] | str, "
    "instance_name: str, **params: object) -> GraphHandle:"
)


def _expand_self_package_import(match: re.Match[str]) -> str:
    """Expose facade modules without making their private extension aliases public."""

    imports: list[str] = []
    for line in match.group("members").splitlines():
        module_name, alias = line.strip().removesuffix(",").split(" as ", maxsplit=1)
        imports.append(f"from . import {module_name} as {alias}")
    return "\n".join(imports) + "\n"


def _sub_outside_literals(
    pattern: re.Pattern[str], replacement: str | Callable[[re.Match[str]], str], source: str
) -> str:
    """Apply a substitution without modifying string literals or comments."""

    line_offsets = [0]
    for line in source.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))

    protected: list[tuple[int, int]] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        start = line_offsets[token.start[0] - 1] + token.start[1]
        end = line_offsets[token.end[0] - 1] + token.end[1]
        protected.append((start, end))

    chunks: list[str] = []
    cursor = 0
    for match in pattern.finditer(source):
        if any(start < match.end() and match.start() < end for start, end in protected):
            continue
        chunks.append(source[cursor : match.start()])
        chunks.append(replacement(match) if callable(replacement) else match.expand(replacement))
        cursor = match.end()
    chunks.append(source[cursor:])
    return "".join(chunks)


def _insert_direct_imports(raw_stub: str, direct_imports: str) -> str:
    """Insert direct facade imports after the module docstring when present."""

    docstring = re.match(r'(?P<doc>""".*?"""\n\n)', raw_stub, flags=re.DOTALL)
    if docstring is None:
        return direct_imports + "\n\n" + raw_stub
    end = docstring.end()
    return raw_stub[:end] + direct_imports + "\n\n" + raw_stub[end:]


def _normalize_type_references(raw_stub: str) -> str:
    """Use direct type imports so mypy can resolve mutually referring facades."""

    imports: set[tuple[str, str, str]] = set()

    def replace_reference(match: re.Match[str]) -> str:
        module_name = match.group("module")
        symbol = match.group("symbol")
        alias = f"_{module_name}_{symbol}"
        imports.add((module_name, symbol, alias))
        return alias

    raw_stub = _sub_outside_literals(_PUBLIC_TYPE_REFERENCE, replace_reference, raw_stub)
    raw_stub = re.sub(
        r"^import holoscan\.(?:clocks|config|diagnostics|errors|execution|graphs|lifecycle|"
        r"operators|params|payload|registry|tensor|time|topics)\n",
        "",
        raw_stub,
        flags=re.MULTILINE,
    )
    if not imports:
        return raw_stub
    direct_imports = "\n".join(
        f"from holoscan.{module_name} import {symbol} as {alias}"
        for module_name, symbol, alias in sorted(imports)
    )
    return _insert_direct_imports(raw_stub, direct_imports)


def _add_registry_overloads(raw_stub: str) -> str:
    """Give registry proxies precise handles while retaining string escape hatches."""

    if _GRAPH_OP_SIGNATURE not in raw_stub or _GRAPH_EMBED_SIGNATURE not in raw_stub:
        return raw_stub
    raw_stub = raw_stub.replace(
        "from typing import Self, overload", "from typing import Any, Self, overload"
    )
    raw_stub = raw_stub.replace(
        _GRAPH_OP_SIGNATURE,
        "    @overload\n"
        "    def op[HandleT](self, type_or_name: _registry_OperatorType[HandleT], "
        "instance_name: str, **params: object) -> HandleT: ...\n\n"
        "    @overload\n"
        "    def op(self, type_or_name: type[Operator], instance_name: str, "
        "**params: object) -> _operators_OperatorHandle: ...\n\n"
        "    @overload\n"
        "    def op(self, type_or_name: str, instance_name: str, **params: Any) -> "
        "_operators_OperatorHandle:  # noqa: ANN401 - dynamic registry name",
    )
    return raw_stub.replace(
        _GRAPH_EMBED_SIGNATURE,
        "    @overload\n"
        "    def embed[HandleT](self, graph_type: _registry_GraphType[HandleT], "
        "instance_name: str, **params: object) -> HandleT: ...\n\n"
        "    @overload\n"
        "    def embed(self, graph_type: type[Graph], instance_name: str, "
        "**params: object) -> GraphHandle: ...\n\n"
        "    @overload\n"
        "    def embed(self, graph_type: str, instance_name: str, "
        "**params: object) -> GraphHandle:",
    )


def finalize_stub(raw_stub: str) -> str:
    """Remove private-extension artifacts and add the checked-in header."""

    raw_stub = re.sub(
        r"^from holoscan\._core import \(\n.*?^\)\n",
        "",
        raw_stub,
        flags=re.MULTILINE | re.DOTALL,
    )
    raw_stub = re.sub(r"^from holoscan\._core import [^\n]+\n", "", raw_stub, flags=re.MULTILINE)
    if "class Graph:" in raw_stub:
        raw_stub = re.sub(
            r"^(    )def op\(self, type_or_name: object, instance_name: str, "
            r"\*\*params(?:: object)?\) -> ([^:\n]+):$",
            r"\1def op(self, type_or_name: _registry_OperatorType | type[Operator] | str, "
            r"instance_name: str, "
            r"**params: object) -> \2:",
            raw_stub,
            flags=re.MULTILINE,
        )
        raw_stub = re.sub(
            r"^(    )def embed\(self, graph_type: object, instance_name: str, "
            r"\*\*params(?:: object)?\) -> [^:\n]+:$",
            r"\1def embed(self, graph_type: _registry_GraphType | type[Graph] | str, "
            r"instance_name: str, **params: object) -> GraphHandle:",
            raw_stub,
            flags=re.MULTILINE,
        )
    raw_stub = re.sub(
        r"^from holoscan import \(\n(?P<members>(?:    \w+ as \w+,?\n)+)\)\n",
        _expand_self_package_import,
        raw_stub,
        flags=re.MULTILINE,
    )
    raw_stub = _normalize_type_references(raw_stub)
    if "class Graph:" in raw_stub:
        missing_registry_imports = [
            f"from holoscan.registry import {symbol} as {alias}"
            for symbol, alias in (
                ("GraphType", "_registry_GraphType"),
                ("OperatorType", "_registry_OperatorType"),
            )
            if alias in raw_stub
            and f"from holoscan.registry import {symbol} as {alias}" not in raw_stub
        ]
        if missing_registry_imports:
            raw_stub = _insert_direct_imports(raw_stub, "\n".join(missing_registry_imports))
    raw_stub = _add_registry_overloads(raw_stub)
    raw_stub = re.sub(
        r"^from typing_extensions import CapsuleType\n", "", raw_stub, flags=re.MULTILINE
    )
    raw_stub = re.sub(
        r"^(?P<indent>\s*)__dlpack_c_exchange_api__: PyCapsule(?: = \.\.\.)?$",
        r"\g<indent>__dlpack_c_exchange_api__: object = ...",
        raw_stub,
        flags=re.MULTILINE,
    )
    raw_stub = re.sub(
        r"^(?P<indent>\s*)def __eq__\(self, arg: object, /\) -> object:$",
        r"\g<indent>def __eq__(self, arg: object, /) -> bool:",
        raw_stub,
        flags=re.MULTILINE,
    )
    raw_stub = "\n".join(
        f"{line}  # noqa: B008 - default constructed by the native binding"
        if re.match(r"^\s*def ", line) and re.search(r"= [\w.]+\(\)", line)
        else line
        for line in raw_stub.split("\n")
    )
    raw_stub = re.sub(r"^__all__: list =", "__all__: list[str] =", raw_stub, flags=re.MULTILINE)
    if re.search(
        r'^__all__: list\[str\] = .*?(?:\'__version__\'|"__version__")',
        raw_stub,
        flags=re.MULTILINE,
    ):
        raw_stub = re.sub(
            r"^__all__: list\[str\] =",
            "__version__: str\n\n__all__: list[str] =",
            raw_stub,
            count=1,
            flags=re.MULTILINE,
        )
    return _HEADER + raw_stub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(finalize_stub(args.input.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
