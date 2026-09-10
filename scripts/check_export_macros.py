#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import re
import sys
from pathlib import Path

SOURCE_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".cuh", ".c", ".cc", ".cpp", ".cxx", ".cu"}
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx"}
TEXT_SUFFIXES = SOURCE_SUFFIXES | {".cmake", ".txt"}
GENERATED_EXPORT_HEADERS = {
    "holomq_export.h",
    "holoviz_export.h",
    "holoscan_config_export.h",
    "holoscan_config_io_export.h",
    "holoscan_core_export.h",
    "holoscan_experimental_params_web_tuner_export.h",
    "holoscan_logger_export.h",
    "holoscan_params_export.h",
    "holoscan_params_io_export.h",
}
EXPORT_MACROS = {
    "HOLOMQ_EXPORT",
    "HOLOMQ_NO_EXPORT",
    "HOLOVIZ_EXPORT",
    "HOLOVIZ_NO_EXPORT",
    "HOLOSCAN_CORE_EXPORT",
    "HOLOSCAN_CORE_NO_EXPORT",
    "HOLOSCAN_CONFIG_EXPORT",
    "HOLOSCAN_CONFIG_NO_EXPORT",
    "HOLOSCAN_CONFIG_IO_EXPORT",
    "HOLOSCAN_CONFIG_IO_NO_EXPORT",
    "HOLOSCAN_PARAMS_EXPORT",
    "HOLOSCAN_PARAMS_NO_EXPORT",
    "HOLOSCAN_PARAMS_IO_EXPORT",
    "HOLOSCAN_PARAMS_IO_NO_EXPORT",
    "HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_EXPORT",
    "HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_NO_EXPORT",
    "HOLOSCAN_LOGGER_EXPORT",
    "HOLOSCAN_LOGGER_NO_EXPORT",
}
GENERATED_EXPORT_INCLUDES = {
    "holomq": "#include <holomq/holomq_export.h>",
    "holoviz": "#include <holoviz/holoviz_export.h>",
    "holoscan": "#include <holoscan/holoscan_core_export.h>",
}
HOLOSCAN_COMPONENT_EXPORT_INCLUDES = {
    "HOLOSCAN_CONFIG_EXPORT": "#include <holoscan/holoscan_config_export.h>",
    "HOLOSCAN_CONFIG_IO_EXPORT": "#include <holoscan/holoscan_config_io_export.h>",
    "HOLOSCAN_PARAMS_EXPORT": "#include <holoscan/holoscan_params_export.h>",
    "HOLOSCAN_PARAMS_IO_EXPORT": "#include <holoscan/holoscan_params_io_export.h>",
    "HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_EXPORT": (
        "#include <holoscan/holoscan_experimental_params_web_tuner_export.h>"
    ),
}
# The logger is a separate DSO under include/holoscan; its export macro requires its own header.
HOLOSCAN_LOGGER_EXPORT_INCLUDE = "#include <holoscan/holoscan_logger_export.h>"
HAND_ROLLED_EXPORT_RE = re.compile(
    r"^\s*#\s*define\s+"
    r"(HOLOMQ_EXPORT|HOLOMQ_NO_EXPORT|HOLOVIZ_EXPORT|HOLOVIZ_NO_EXPORT"
    r"|HOLOSCAN_CORE_EXPORT|HOLOSCAN_CORE_NO_EXPORT"
    r"|HOLOSCAN_CONFIG_EXPORT|HOLOSCAN_CONFIG_NO_EXPORT"
    r"|HOLOSCAN_CONFIG_IO_EXPORT|HOLOSCAN_CONFIG_IO_NO_EXPORT"
    r"|HOLOSCAN_PARAMS_EXPORT|HOLOSCAN_PARAMS_NO_EXPORT"
    r"|HOLOSCAN_PARAMS_IO_EXPORT|HOLOSCAN_PARAMS_IO_NO_EXPORT"
    r"|HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_EXPORT"
    r"|HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_NO_EXPORT"
    r"|HOLOSCAN_LOGGER_EXPORT|HOLOSCAN_LOGGER_NO_EXPORT)\b",
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


def has_path_sequence(path: Path, sequence: tuple[str, ...]) -> bool:
    parts = path.parts
    return any(parts[index : index + len(sequence)] == sequence for index in range(len(parts)))


def is_holoviz_module_path(path: Path) -> bool:
    return has_path_sequence(path, ("public", "modules", "holoviz")) or has_path_sequence(
        path, ("modules", "holoviz")
    )


def is_public_holoviz_header(path: Path) -> bool:
    parts = path.parts
    return "include" in parts and "holoviz" in parts and path.suffix in HEADER_SUFFIXES


def is_holomq_module_path(path: Path) -> bool:
    return has_path_sequence(path, ("public", "modules", "holomq")) or has_path_sequence(
        path, ("modules", "holomq")
    )


def is_holomq_test_source(path: Path) -> bool:
    return has_path_sequence(path, ("tests", "holomq"))


def is_public_holomq_header(path: Path) -> bool:
    parts = path.parts
    return "include" in parts and "holomq" in parts and path.suffix in HEADER_SUFFIXES


def is_public_holoscan_header(path: Path) -> bool:
    parts = path.parts
    return "include" in parts and "holoscan" in parts and path.suffix in HEADER_SUFFIXES


def is_application_facing_source(path: Path) -> bool:
    return any(part in {"examples", "tests"} for part in path.parts)


def is_example_source(path: Path) -> bool:
    return "examples" in path.parts


def discover_files() -> list[Path]:
    roots = [Path("public"), Path("include"), Path("src"), Path("modules")]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and is_text_file(path))
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

    if DETAIL_INCLUDE_RE.search(content):
        if (
            is_holomq_module_path(path)
            or is_public_holomq_header(path)
            or is_holomq_test_source(path)
        ):
            if is_example_source(path):
                errors.append(f"{path}: HoloMQ examples must not include detail/ internals")
        elif (
            is_public_holomq_header(path)
            or is_public_holoscan_header(path)
            or is_application_facing_source(path)
        ):
            errors.append(
                f"{path}: public headers, examples, and tests must not include detail/ internals"
            )

    if has_detail_segment(path):
        errors.extend(
            f"{path}: {macro} must not be used in detail/ headers"
            for macro in EXPORT_MACROS
            if macro in content
        )
        errors.extend(
            f"{path}: generated export headers must not be included in detail/"
            for include in GENERATED_EXPORT_INCLUDES.values()
            if include in content
        )

    if (
        is_public_holomq_header(path)
        and "HOLOMQ_EXPORT" in content
        and GENERATED_EXPORT_INCLUDES["holomq"] not in content
    ):
        errors.append(f"{path}: HOLOMQ_EXPORT requires <holomq/holomq_export.h>")

    if (
        is_public_holoviz_header(path)
        and "HOLOVIZ_EXPORT" in content
        and GENERATED_EXPORT_INCLUDES["holoviz"] not in content
    ):
        errors.append(f"{path}: HOLOVIZ_EXPORT requires <holoviz/holoviz_export.h>")

    if (
        is_public_holoscan_header(path)
        and "HOLOSCAN_CORE_EXPORT" in content
        and GENERATED_EXPORT_INCLUDES["holoscan"] not in content
    ):
        errors.append(f"{path}: HOLOSCAN_CORE_EXPORT requires <holoscan/holoscan_core_export.h>")

    if (
        is_public_holoscan_header(path)
        and "HOLOSCAN_LOGGER_EXPORT" in content
        and HOLOSCAN_LOGGER_EXPORT_INCLUDE not in content
    ):
        errors.append(
            f"{path}: HOLOSCAN_LOGGER_EXPORT requires <holoscan/holoscan_logger_export.h>"
        )

    if is_public_holoscan_header(path):
        for macro, include in HOLOSCAN_COMPONENT_EXPORT_INCLUDES.items():
            if macro in content and include not in content:
                errors.append(f"{path}: {macro} requires {include.removeprefix('#include ')}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce generated export macro and detail-header policy."
    )
    parser.add_argument("paths", nargs="*", help="Paths passed by pre-commit.")
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths] if args.paths else discover_files()
    errors: list[str] = []
    for path in paths:
        errors.extend(check_file(path))

    if errors:
        print("Export macro policy failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
