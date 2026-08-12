#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import re
import sys
from pathlib import Path

HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".cuh"}
GENERATED_HEADER_SUFFIXES = ("_export.h",)
EXCLUDED_PATH_PARTS = {
    ".cache",
    ".git",
    "_deps",
    "build",
    "install",
    "thirdparty",
    "holoscan-sdk-4",
}
MACRO_GUARD_IFNDEF_RE = re.compile(r"^\s*#\s*ifndef\s+([A-Za-z_][A-Za-z0-9_]*)\b")
# An include guard defines its macro to nothing (`#define GUARD`). A `#define NAME value` is a
# configuration/feature macro (e.g. HOLOSCAN_LOG_ACTIVE_LEVEL), not an include guard, so require an
# empty replacement list (only optional trailing whitespace) to avoid misclassifying those.
MACRO_GUARD_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")


def is_header(path: Path) -> bool:
    return path.suffix in HEADER_SUFFIXES


def is_excluded(path: Path) -> bool:
    if path.name.endswith(GENERATED_HEADER_SUFFIXES):
        return True
    for part in path.parts:
        if part in EXCLUDED_PATH_PARTS or part.startswith(("build-", "install-")):
            return True
    return False


def discover_headers() -> list[Path]:
    roots = [
        Path("public/include"),
        Path("public/src"),
        Path("public/modules"),
        Path("include"),
        Path("src"),
        Path("modules"),
    ]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file() and is_header(path))
    return paths


def has_pragma_once_near_top(lines: list[str]) -> bool:
    return any(re.match(r"^\s*#\s*pragma\s+once\b", line) for line in lines[:40])


def macro_guard_pair(lines: list[str]) -> tuple[int, str] | None:
    for index, line in enumerate(lines):
        ifndef_match = MACRO_GUARD_IFNDEF_RE.match(line)
        if ifndef_match is None:
            continue

        guard_name = ifndef_match.group(1)
        for next_line in lines[index + 1 : index + 8]:
            define_match = MACRO_GUARD_DEFINE_RE.match(next_line)
            if define_match and define_match.group(1) == guard_name:
                return index + 1, guard_name
    return None


def check_header(path: Path) -> list[str]:
    if not path.exists() or not is_header(path) or is_excluded(path):
        return []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return [f"{path}: cannot decode as UTF-8"]

    errors: list[str] = []
    has_pragma_once = has_pragma_once_near_top(lines)
    guard_pair = macro_guard_pair(lines)

    if not has_pragma_once:
        errors.append(f"{path}: missing #pragma once near the top of the header")
    if guard_pair is not None:
        line_number, guard_name = guard_pair
        errors.append(f"{path}:{line_number}: macro include guard '{guard_name}' is not allowed")
    if has_pragma_once and guard_pair is not None:
        errors.append(f"{path}: do not combine #pragma once with a macro include guard")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require #pragma once and reject macro include guards in project headers."
    )
    parser.add_argument("paths", nargs="*", help="Header paths passed by pre-commit.")
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths] if args.paths else discover_headers()
    errors: list[str] = []
    for path in paths:
        errors.extend(check_header(path))

    if errors:
        print("Header guard policy failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
