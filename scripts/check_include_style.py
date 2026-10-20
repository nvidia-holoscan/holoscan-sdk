#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require angle brackets for established SDK and dependency include roots."""

from __future__ import annotations

import argparse
import os
import re
import sys
from bisect import bisect_right
from pathlib import Path

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx", ".cuh"}
EXCLUDED_PARTS = {"_deps", "build", "install", "generated", "thirdparty", "third_party", "vendor"}
# Keep this explicit: quoted private/local headers are valid. See DEVELOP.md.
ANGLE_BRACKET_PREFIXES = ("holoscan/", "holomq/", "fmt/", "iox2/")
SOURCE_ROOT = Path(__file__).resolve().parent.parent
SPLICE_RE = re.compile(r"\\\r?\n")
TOKEN_RE = re.compile(
    r"(?P<comment>//[^\n]*|/\*[\s\S]*?(?:\*/|\Z))"
    r'|(?:u8|[uUL])?R"(?P<delimiter>[^\s()\\]{0,16})\([\s\S]*?\)(?P=delimiter)"'
    r'|"(?:\\[\s\S]|[^"\\])*"'
    r"|(?<![\w'])(?:u8|[uUL])?'(?:\\[\s\S]|[^'\\\n])*'"
)
DIRECTIVE_RE = re.compile(r"^[^\S\n]*#[^\S\n]*include\b", re.MULTILINE)
QUOTED_HEADER_RE = re.compile(r'[^\S\n]*"([^"\n]+)"')


def quoted_dependency_includes(source: str) -> list[tuple[int, str]]:
    """Return physical directive line numbers and offending literal header names.

    Splice continued lines before tokenizing, as the preprocessor does. Mask comments and
    literals before recognizing directives, but retain header strings in the operand view.
    Conditional branches are all checked; macro-expanded header names are outside this rule.
    """
    splice_positions = []
    removed = 0
    for match in SPLICE_RE.finditer(source):
        splice_positions.append(match.start() - removed)
        removed += match.end() - match.start()
    text = SPLICE_RE.sub("", source)
    code = list(text)
    operands = list(text)
    for match in TOKEN_RE.finditer(text):
        if match.group("comment") is not None:
            # A block comment is whitespace even when it spans physical lines inside a directive.
            masked = " " * len(match.group())
            operands[match.start() : match.end()] = masked
        else:
            masked = re.sub(r"[^\n]", " ", match.group())
        code[match.start() : match.end()] = masked
    code_text = "".join(code)
    operand_text = "".join(operands)
    errors = []
    for directive in DIRECTIVE_RE.finditer(code_text):
        header = QUOTED_HEADER_RE.match(operand_text, directive.end())
        if header is not None and header.group(1).startswith(ANGLE_BRACKET_PREFIXES):
            offset = code_text.index("#", directive.start(), directive.end())
            line = 1 + text.count("\n", 0, offset) + bisect_right(splice_positions, offset)
            errors.append((line, header.group(1)))
    return errors


def excluded_part(part: str) -> bool:
    return part.startswith(".") or part in EXCLUDED_PARTS or part.startswith(("build-", "install-"))


def check_file(path: Path, root: Path = SOURCE_ROOT) -> list[str]:
    """Check maintained sources inside this script's internal or public source tree."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return []
    if (
        path.suffix not in SOURCE_SUFFIXES
        or any(excluded_part(part) for part in relative.parts)
        or path.name.endswith(("_generated.h", "_generated.hpp", "_export.h", ".pb.h", ".pb.cc"))
    ):
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [f"{path}: cannot read source: {error}"]
    return [
        f'{path}:{line}: use #include <{header}> instead of "{header}"'
        for line, header in quoted_dependency_includes(source)
    ]


def discover_sources(root: Path) -> list[Path]:
    paths = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(part for part in subdirs if not excluded_part(part))
        paths.extend(
            Path(directory) / name for name in sorted(files) if Path(name).suffix in SOURCE_SUFFIXES
        )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Sources passed by pre-commit.")
    args = parser.parse_args()
    paths = args.paths if args.paths else discover_sources(SOURCE_ROOT)
    errors = [error for path in paths for error in check_file(path)]
    if errors:
        print("Include style check failed:")
        print("\n".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
