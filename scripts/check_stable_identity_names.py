#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject alternate NVIDIA ownership prefixes in maintained SDK string literals.

This is source policy, not runtime namespace validation. It checks C++/CUDA and
Python literals, including constants and adjacent C++ strings. Portable schema
names and unrelated protocol vocabularies use exact, reasoned exceptions.
Schema, callback and POD declarations also check ownership and version placement,
including simple local constants and standard sv literals. Qualified Holoscan POD
traits are recognized. It does not interpret macros, resolve C++ scopes, evaluate
computed expressions or interpret custom literal suffixes. Bare hostnames are not identities.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

from _git_environment import git_environment
from check_include_style import TOKEN_RE

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".cpp", ".hpp", ".h", ".cc", ".cxx", ".cu", ".cuh", ".py"}
FORBIDDEN = ("com.nvidia.", "nvidia.")
STRING_VIEW_SUFFIX_RE = re.compile(r"sv\b")


def cpp_token_end(source: str, token: re.Match[str]) -> int:
    """Include the standard string-view suffix when consuming a C++ string token."""
    end = token.end()
    if token.group("comment") is None and token.group().endswith('"'):
        suffix = STRING_VIEW_SUFFIX_RE.match(source, end)
        if suffix:
            return suffix.end()
    return end


def literal_value(spelling: str) -> str | None:
    raw = re.fullmatch(r'(?:u8|[uUL])?R"([^()]*)\(([\s\S]*)\)\1"', spelling)
    if raw:
        return raw.group(2)
    try:
        value = ast.literal_eval(spelling[spelling.index('"') :])
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, str) else None


def declaration_findings(source: str) -> list[tuple[int, str]]:
    """Resolve direct literals and simple constants in schema/callback declarations.

    Comments and string contents cannot introduce declarations. This deliberately
    does not implement C++ scope lookup, macro expansion, or expression evaluation.
    """
    values = {}
    masked = []
    end = 0
    for index, token in enumerate(TOKEN_RE.finditer(source)):
        masked.append(source[end : token.start()])
        key = f"__identity_literal_{index}"
        if token.group("comment") is not None:
            masked.append(re.sub(r"[^\n]", " ", token.group()))
        else:
            value = literal_value(token.group())
            if isinstance(value, str):
                values[key] = (source.count("\n", 0, token.start()) + 1, value)
            masked.append(key)
        end = cpp_token_end(source, token)
    masked.append(source[end:])
    code = "".join(masked)

    def join_literals(match: re.Match[str]) -> str:
        keys = match.group().split()
        if all(key in values for key in keys):
            values[keys[0]] = (values[keys[0]][0], "".join(values[key][1] for key in keys))
        return keys[0]

    code = re.sub(r"__identity_literal_\d+(?:\s+__identity_literal_\d+)+", join_literals, code)
    for binding in re.finditer(r"\b(\w+)\s*=\s*(__identity_literal_\d+)\s*;", code):
        if binding[2] in values:
            values[binding[1]] = values[binding[2]]
    errors = []
    declarations = re.compile(
        r"\b(?P<kind>param_schema|validator|parameter_projected_byte_span)"
        r"\s*(?:<[^;{}]*?>)?\s*\(\s*(?P<name>\w+)"
    )
    identities = [(call["kind"], call["name"]) for call in declarations.finditer(code)]
    identities.extend(
        ("pod", pod[1])
        for pod in re.finditer(
            r"\bstruct\s+(?:(?:::)?\s*holoscan\s*::\s*)?ChannelTypeIdentity"
            r"\s*<[^;{}]+>\s*\{[^{}]*?"
            r"\bname\s*=\s*(\w+)",
            code,
        )
    )
    for kind, name in identities:
        if name not in values:
            continue
        line, value = values[name]
        owned = re.fullmatch(
            r"holoscan\.(?!ext(?:\.|$))[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*",
            value,
        )
        versioned = re.search(r"\.v[1-9][0-9]*$", value) is not None
        if not owned or (kind in {"param_schema", "pod"}) == versioned:
            errors.append((line, value))
    return errors


def identity_literals(source: str, *, python: bool = False) -> list[tuple[int, str]]:
    """Return alternate-prefix literals with physical line numbers."""
    literals = []
    if python:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.STRING:
                try:
                    value = ast.literal_eval(token.string)
                except (SyntaxError, ValueError):
                    continue  # Dynamic f-strings are outside this literal-only check.
                if isinstance(value, str):
                    literals.append((token.start[0], value))
    else:
        pending = ""
        pending_line = 1
        last_end = 0
        for match in TOKEN_RE.finditer(source):
            if match.group("comment") is not None:
                continue
            spelling = match.group()
            if '"' not in spelling:
                continue
            between = source[last_end : match.start()]
            between = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", between)
            if between.strip():
                literals.append((pending_line, pending))
                pending = ""
            if not pending:
                pending_line = source.count("\n", 0, match.start()) + 1
            value = literal_value(spelling) or ""
            pending += value
            last_end = cpp_token_end(source, match)
        literals.append((pending_line, pending))
    findings = [
        (line, value)
        for line, value in literals
        if value.startswith(FORBIDDEN) and value not in (*FORBIDDEN, "nvidia.com")
    ]
    return sorted(set(findings + ([] if python else declaration_findings(source))))


def check_paths(root: Path, paths: list[Path], exceptions: list[dict[str, str]]) -> list[str]:
    """Check sources and reject missing, duplicate, or stale exact exceptions."""
    errors = []
    allowed = set()
    for entry in exceptions:
        if set(entry) != {"path", "literal", "reason"} or not entry.get("reason", "").strip():
            errors.append("identity exception requires path, literal, and nonempty reason")
            continue
        key = (entry["path"], entry["literal"])
        if key in allowed:
            errors.append(f"duplicate identity exception: {key}")
        allowed.add(key)
    seen = set()
    scanned = set()
    for path in paths:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        if path.suffix not in SUFFIXES:
            continue
        scanned.add(relative)
        try:
            findings = identity_literals(
                path.read_text(encoding="utf-8"), python=path.suffix == ".py"
            )
        except (OSError, UnicodeError, tokenize.TokenError, IndentationError) as error:
            errors.append(f"{relative}: cannot check identities: {error}")
            continue
        for line, literal in findings:
            key = (relative, literal)
            if key in allowed:
                seen.add(key)
            else:
                errors.append(
                    f"{relative}:{line}: {literal}: "
                    "use holoscan.<area-or-provider>.* without .ext; "
                    "schemas and PODs use a separate version, callbacks require .vN"
                )
    errors.extend(
        f"stale identity exception: {key}"
        for key in sorted(allowed - seen)
        if key[0] in scanned or not (root / key[0]).is_file()
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths
    if not paths:
        # Hooks export repository paths that git -C does not override. Keep the hook's index: it
        # holds the files being committed, which differ from the default index under commit -a.
        environment = git_environment()
        if index_file := os.environ.get("GIT_INDEX_FILE"):
            environment["GIT_INDEX_FILE"] = index_file
        output = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "-z"], text=True, env=environment
        )
        paths = [ROOT / name for name in output.split("\0") if name]
    else:
        paths = [path.resolve() for path in paths]
        paths = [path for path in paths if path.is_relative_to(ROOT)]
    exceptions = json.loads((ROOT / "scripts/stable_identity_exceptions.json").read_text())
    errors = check_paths(ROOT, paths, exceptions)
    for error in errors:
        print(error, file=sys.stderr)
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
