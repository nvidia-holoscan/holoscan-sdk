# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that a refined public stub covers its generated baseline.

By default the committed stub must define every public name and class member of the baseline,
which lets refined stubs change annotations. ``--match-declarations`` also requires every
generated declaration, including private types such as an Operator proxy's parameter
``TypedDict``, to appear exactly as generated, while still allowing extra declarations. Its
``__all__`` may export more names than the generated one, so a single committed stub can cover
build configurations that register different subsets of a provider's Operators.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def _is_public(name: str) -> bool:
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def _defined_names(body: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.rsplit(".", 1)[-1] for alias in node.names)
    return {name for name in names if _is_public(name)}


_DATACLASS_GENERATED_MEMBERS = {
    "__dataclass_fields__",
    "__dataclass_params__",
    "__delattr__",
    "__eq__",
    "__hash__",
    "__init__",
    "__match_args__",
    "__repr__",
    "__setattr__",
}


def _is_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "dataclass":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "dataclass":
            return True
    return False


def _classes(tree: ast.Module) -> dict[str, set[str]]:
    return {
        node.name: _defined_names(node.body)
        | (_DATACLASS_GENERATED_MEMBERS if _is_dataclass(node) else set())
        for node in tree.body
        if isinstance(node, ast.ClassDef) and _is_public(node.name)
    }


def _explicit_exports(tree: ast.Module) -> set[str] | None:
    """Return the literal names in a module's ``__all__``, or ``None`` when it declares none."""

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple)):
            return {
                item.value
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
    return None


def _exported_names(tree: ast.Module) -> set[str]:
    exports = _explicit_exports(tree)
    return exports if exports is not None else _defined_names(tree.body)


def _declarations(body: list[ast.stmt]) -> dict[str, ast.stmt]:
    """Return the named declarations of a module or class body, excluding imports."""

    declarations: dict[str, ast.stmt] = {}
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            declarations[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declarations[node.target.id] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    declarations[target.id] = node
    return declarations


def _mismatched_declarations(
    generated: list[ast.stmt], committed: list[ast.stmt], *, module_scope: bool = False
) -> list[str]:
    """Return qualified names of generated declarations that the committed body lacks or changes.

    A class is compared by its header, docstring, and each generated member, so a committed class
    may declare more members than the generated one.
    """

    committed_declarations = _declarations(committed)
    mismatches: list[str] = []
    for name, node in _declarations(generated).items():
        if module_scope and name == "__all__":
            # Exports are compared as sets by _mismatched_exports.
            continue
        other = committed_declarations.get(name)
        if isinstance(node, ast.ClassDef) and isinstance(other, ast.ClassDef):
            if _class_header(node) != _class_header(other) or ast.get_docstring(
                node
            ) != ast.get_docstring(other):
                mismatches.append(name)
            mismatches.extend(
                f"{name}.{member}" for member in _mismatched_declarations(node.body, other.body)
            )
        elif other is None or ast.dump(node) != ast.dump(other):
            mismatches.append(name)
    return mismatches


def _mismatched_exports(generated: ast.Module, committed: ast.Module) -> list[str]:
    """Return ``__all__`` unless the committed exports include every generated export.

    A committed stub may export more names, so one stub can describe a provider whose build
    configurations register different subsets of its Operators.
    """

    generated_exports = _explicit_exports(generated)
    if generated_exports is None:
        return []
    committed_exports = _explicit_exports(committed)
    if committed_exports is None or not generated_exports <= committed_exports:
        return ["__all__"]
    return []


def _class_header(node: ast.ClassDef) -> list[str]:
    """Return the class's bases, keywords, decorators, and type parameters as AST dumps."""

    # Hosts may run this check on Python versions whose ast.ClassDef predates type_params (3.12).
    type_params: list[ast.AST] = getattr(node, "type_params", [])
    parts: list[ast.AST] = [
        *node.bases,
        *node.keywords,
        *node.decorator_list,
        *type_params,
    ]
    return [ast.dump(part) for part in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--committed", type=Path, required=True)
    parser.add_argument(
        "--match-declarations",
        action="store_true",
        help="Require every generated declaration, including private ones, to appear unchanged",
    )
    args = parser.parse_args()

    generated = ast.parse(args.generated.read_text(encoding="utf-8"), filename=str(args.generated))
    committed = ast.parse(args.committed.read_text(encoding="utf-8"), filename=str(args.committed))
    missing = sorted(_exported_names(generated) - _defined_names(committed.body))
    generated_classes = _classes(generated)
    committed_classes = _classes(committed)
    missing_members = {
        name: sorted(members - committed_classes.get(name, set()))
        for name, members in generated_classes.items()
        if members - committed_classes.get(name, set())
    }
    changed = (
        [
            *_mismatched_exports(generated, committed),
            *_mismatched_declarations(generated.body, committed.body, module_scope=True),
        ]
        if args.match_declarations
        else []
    )
    if missing or missing_members or changed:
        details: list[str] = []
        if missing:
            details.append("missing module names: " + ", ".join(missing))
        details.extend(
            f"{name} missing members: {', '.join(members)}"
            for name, members in sorted(missing_members.items())
        )
        if changed:
            details.append("missing or changed declarations: " + ", ".join(changed))
        raise SystemExit(f"{args.committed} does not cover {args.generated}: " + "; ".join(details))


if __name__ == "__main__":
    main()
