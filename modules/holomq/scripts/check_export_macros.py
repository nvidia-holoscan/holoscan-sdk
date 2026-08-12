#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
import sys
from pathlib import Path

SOURCE_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".cuh", ".c", ".cc", ".cpp", ".cxx", ".cu"}
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}
TEXT_SUFFIXES = SOURCE_SUFFIXES | {".cmake", ".txt"}
GENERATED_EXPORT_HEADERS = {"holomq_export.h"}
EXPORT_MACROS = {"HOLOMQ_EXPORT", "HOLOMQ_NO_EXPORT"}
GENERATED_EXPORT_INCLUDE = "#include <holomq/holomq_export.h>"
HAND_ROLLED_EXPORT_RE = re.compile(
    r"^\s*#\s*define\s+(HOLOMQ_EXPORT|HOLOMQ_NO_EXPORT)\b",
    re.MULTILINE,
)
OLD_HOLOMQ_EXPORT_INCLUDE_RE = re.compile(
    r"^\s*#\s*include\s*[<\"]holomq/export\.hpp[>\"]", re.MULTILINE
)
DETAIL_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"][^>\"]*/detail/[^>\"]*[>\"]", re.MULTILINE)


def is_text_file(path: Path) -> bool:
    return path.name in {"CMakeLists.txt", "Dockerfile"} or path.suffix in TEXT_SUFFIXES


def is_generated_export_header(path: Path) -> bool:
    return path.name in GENERATED_EXPORT_HEADERS or path.name.endswith("_export.h")


def is_excluded(path: Path) -> bool:
    for part in path.parts:
        if part in {".cache", ".git", "_deps", "build", "install", "thirdparty"}:
            return True
        if part.startswith(("build-", "install-")):
            return True
    return False


def has_detail_segment(path: Path) -> bool:
    return "detail" in path.parts


def is_public_holomq_header(path: Path) -> bool:
    parts = path.parts
    return "include" in parts and "holomq" in parts and path.suffix in HEADER_SUFFIXES


def is_example_source(path: Path) -> bool:
    return "examples" in path.parts


def discover_files() -> list[Path]:
    roots = [
        Path("include"),
        Path("src"),
        Path("tests"),
        Path("examples"),
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and is_text_file(path))
    if Path("CMakeLists.txt").exists():
        files.append(Path("CMakeLists.txt"))
    return files


def check_file(path: Path) -> list[str]:
    if not path.exists() or not is_text_file(path) or is_excluded(path):
        return []

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path}: cannot decode as UTF-8"]

    errors: list[str] = []
    hand_rolled_match = HAND_ROLLED_EXPORT_RE.search(content)
    if hand_rolled_match and not is_generated_export_header(path):
        errors.append(
            f"{path}: hand-written {hand_rolled_match.group(1)} is forbidden; "
            "use CMake GenerateExportHeader"
        )

    if "WINDOWS_EXPORT_ALL_SYMBOLS" in content:
        errors.append(f"{path}: WINDOWS_EXPORT_ALL_SYMBOLS is forbidden")

    if OLD_HOLOMQ_EXPORT_INCLUDE_RE.search(content):
        errors.append(f"{path}: include <holomq/holomq_export.h>, not <holomq/export.hpp>")

    if DETAIL_INCLUDE_RE.search(content) and is_example_source(path):
        errors.append(f"{path}: examples must not include detail/ internals")

    if has_detail_segment(path):
        errors.extend(
            f"{path}: {macro} must not be used in detail/ headers"
            for macro in EXPORT_MACROS
            if macro in content
        )
        if GENERATED_EXPORT_INCLUDE in content:
            errors.append(f"{path}: generated export headers must not be included in detail/")

    if (
        is_public_holomq_header(path)
        and "HOLOMQ_EXPORT" in content
        and GENERATED_EXPORT_INCLUDE not in content
    ):
        errors.append(f"{path}: HOLOMQ_EXPORT requires <holomq/holomq_export.h>")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce HoloMQ generated export macro and detail-header policy."
    )
    parser.add_argument("paths", nargs="*", help="Paths passed by pre-commit.")
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths] if args.paths else discover_files()
    errors: list[str] = []
    for path in paths:
        errors.extend(check_file(path))

    if errors:
        print("HoloMQ export macro policy failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
