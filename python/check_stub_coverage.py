# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that a refined public stub covers its nanobind-generated baseline."""

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


def _exported_names(tree: ast.Module) -> set[str]:
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
    return _defined_names(tree.body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--committed", type=Path, required=True)
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
    if missing or missing_members:
        details = []
        if missing:
            details.append("missing module names: " + ", ".join(missing))
        details.extend(
            f"{name} missing members: {', '.join(members)}"
            for name, members in sorted(missing_members.items())
        )
        raise SystemExit(f"{args.committed} does not cover {args.generated}: " + "; ".join(details))


if __name__ == "__main__":
    main()
