#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
from pathlib import Path

HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".cuh"}
GENERATED_HEADER_SUFFIXES = ("_export.h",)
HEADER_ROOTS = ("include", "src", "tests", "examples")
EXCLUDED_PATH_PARTS = {".cache", ".git", "_deps", "build", "install", "thirdparty"}
MACRO_GUARD_IFNDEF_RE = re.compile(r"^\s*#\s*ifndef\s+([A-Za-z_][A-Za-z0-9_]*)\b")
MACRO_GUARD_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\b")
CONDITIONAL_OPEN_RE = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef)\b")
CONDITIONAL_END_RE = re.compile(r"^\s*#\s*endif\b")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT_RE = re.compile(r"//.*$", re.MULTILINE)
PRAGMA_ONCE_RE = re.compile(r"^\s*#\s*pragma\s+once\b.*$", re.MULTILINE)
MAX_MACRO_GUARD_DEFINE_LOOKAHEAD = 256


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
    paths: list[Path] = []
    for root_name in HEADER_ROOTS:
        root = Path(root_name)
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file() and is_header(path))
    return paths


def has_pragma_once_near_top(lines: list[str]) -> bool:
    return any(re.match(r"^\s*#\s*pragma\s+once\b", line) for line in lines[:40])


def has_substantive_content(lines: list[str], *, allow_pragma_once: bool = False) -> bool:
    text = "\n".join(lines)
    text = BLOCK_COMMENT_RE.sub("", text)
    text = LINE_COMMENT_RE.sub("", text)
    if allow_pragma_once:
        text = PRAGMA_ONCE_RE.sub("", text)
    return bool(text.strip())


def matching_endif(lines: list[str], opening_index: int) -> int | None:
    depth = 0
    for index in range(opening_index, len(lines)):
        if CONDITIONAL_OPEN_RE.match(lines[index]):
            depth += 1
        elif CONDITIONAL_END_RE.match(lines[index]):
            depth -= 1
            if depth == 0:
                return index
    return None


def is_blank_or_comment_line(line: str, in_block_comment: bool) -> tuple[bool, bool]:
    stripped = line.strip()
    if in_block_comment:
        end_index = stripped.find("*/")
        if end_index == -1:
            return True, True
        stripped = stripped[end_index + 2 :].strip()

    if not stripped or stripped.startswith("//"):
        return True, False
    if not stripped.startswith("/*"):
        return False, False

    end_index = stripped.find("*/", 2)
    if end_index == -1:
        return True, True
    stripped = stripped[end_index + 2 :].strip()
    return not stripped or stripped.startswith("//"), False


def macro_guard_pair(lines: list[str]) -> tuple[int, str] | None:
    for index, line in enumerate(lines):
        ifndef_match = MACRO_GUARD_IFNDEF_RE.match(line)
        if ifndef_match is None:
            continue

        guard_name = ifndef_match.group(1)
        has_matching_define = False
        in_block_comment = False
        lookahead_end = min(len(lines), index + 1 + MAX_MACRO_GUARD_DEFINE_LOOKAHEAD)
        for next_line in lines[index + 1 : lookahead_end]:
            is_comment, in_block_comment = is_blank_or_comment_line(next_line, in_block_comment)
            if is_comment:
                continue

            define_match = MACRO_GUARD_DEFINE_RE.match(next_line)
            if define_match is not None and define_match.group(1) == guard_name:
                has_matching_define = True
            break
        if not has_matching_define:
            continue

        closing_index = matching_endif(lines, index)
        if closing_index is None:
            continue

        if has_substantive_content(lines[:index], allow_pragma_once=True):
            continue
        if has_substantive_content(lines[closing_index + 1 :]):
            continue

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
        description="Require #pragma once and reject macro include guards in HoloMQ headers."
    )
    parser.add_argument("paths", nargs="*", help="Header paths passed by pre-commit.")
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths] if args.paths else discover_headers()
    errors: list[str] = []
    for path in paths:
        errors.extend(check_header(path))

    if errors:
        print("HoloMQ header guard policy failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
