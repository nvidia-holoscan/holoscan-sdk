#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import argparse
import json
import re
import sys
from pathlib import Path

PUBLIC_HEADER_ROOTS = (
    # Component libraries are their own DSOs with their own export macros; list them before the
    # general Holoscan root so the most-specific root wins (see public_headers()).
    (
        "public/include/holoscan/experimental/params",
        "HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_EXPORT",
    ),
    ("include/holoscan/experimental/params", "HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_EXPORT"),
    ("public/include/holoscan/config", "HOLOSCAN_CONFIG_EXPORT"),
    ("include/holoscan/config", "HOLOSCAN_CONFIG_EXPORT"),
    ("public/include/holoscan/params", "HOLOSCAN_PARAMS_EXPORT"),
    ("include/holoscan/params", "HOLOSCAN_PARAMS_EXPORT"),
    ("public/include/holoscan/logger", "HOLOSCAN_LOGGER_EXPORT"),
    ("include/holoscan/logger", "HOLOSCAN_LOGGER_EXPORT"),
    ("public/include/holoscan/schema", "HOLOSCAN_SCHEMA_EXPORT"),
    ("include/holoscan/schema", "HOLOSCAN_SCHEMA_EXPORT"),
    ("public/include/holoscan", "HOLOSCAN_CORE_EXPORT"),
    ("include/holoscan", "HOLOSCAN_CORE_EXPORT"),
    ("public/include/holomq", "HOLOMQ_EXPORT"),
    ("include/holomq", "HOLOMQ_EXPORT"),
    ("public/modules/visualization/include/holoviz", "HOLOVIZ_EXPORT"),
    ("include/holoviz", "HOLOVIZ_EXPORT"),
)
PUBLIC_HEADER_FILE_MACROS = {
    "holoscan/config/io.hpp": "HOLOSCAN_CONFIG_IO_EXPORT",
    "holoscan/params/persistence.hpp": "HOLOSCAN_PARAMS_IO_EXPORT",
}
ALLOWLIST_SECTIONS = {
    "HOLOMQ_EXPORT": "holomq",
    "HOLOVIZ_EXPORT": "holoviz",
    "HOLOSCAN_CORE_EXPORT": "holoscan_core",
}
DEFAULT_ALLOWLIST = Path(__file__).resolve().parents[1] / "ci/abi/export-macro-allowlist.yml"
DECLARATION_RE = re.compile(
    r"^\s*(?!#)(?!using\b)(?!typedef\b)(?!return\b)"
    r"(?!inline\b)(?!constexpr\b)(?!static\b)(?!friend\b)"
    r"(?:[A-Za-z_][A-Za-z0-9_:<>~,\s*&]+)\s+"
    r"[A-Za-z_~][A-Za-z0-9_:~]*\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?;"
)


def load_allowlist(path: Path) -> dict[str, set[str]]:
    """Load the intentionally small exact-declaration export-policy allowlist."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot read export macro allowlist: {path}: {error}") from error

    entries: dict[str, set[str]] = {}
    current_section: str | None = None
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not line[0].isspace():
            match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*):(?:\s*(\[\]))?", stripped)
            if match is None:
                raise RuntimeError(f"invalid allowlist syntax: {path}:{line_number}")
            current_section = match.group(1)
            entries.setdefault(current_section, set())
            if match.group(2) == "[]":
                current_section = None
            continue

        if current_section is None or not stripped.startswith("- "):
            raise RuntimeError(f"invalid allowlist entry: {path}:{line_number}")
        declaration = stripped[2:].strip()
        if "::" not in declaration:
            raise RuntimeError(f"allowlist entry lacks path separator: {path}:{line_number}")
        header_path, declaration_text = declaration.split("::", maxsplit=1)
        if not header_path or not declaration_text:
            raise RuntimeError(f"empty allowlist entry: {path}:{line_number}")
        entries[current_section].add(f"{header_path}::{' '.join(declaration_text.split())}")

    return entries


def canonical_header_path(path: Path) -> str:
    """Return one stable source-tree path for both repository and public roots."""
    normalized = path.as_posix()
    marker_to_prefix = (
        ("/public/include/", "public/include/"),
        ("/public/modules/", "public/modules/"),
        ("/modules/", "public/modules/"),
        ("/include/", "public/include/"),
    )
    for marker, prefix in marker_to_prefix:
        if marker in normalized:
            return f"{prefix}{normalized.split(marker, maxsplit=1)[1]}"
    if normalized.startswith("public/"):
        return normalized
    if normalized.startswith("include/"):
        return f"public/{normalized}"
    return normalized


def declaration_key(path: Path, declaration: str) -> str:
    """Build an exact path-qualified key with insignificant whitespace collapsed."""
    return f"{canonical_header_path(path)}::{' '.join(declaration.split())}"


def code_without_comments_or_literals(line: str, in_block_comment: bool) -> tuple[str, bool]:
    """Remove text that cannot contribute C++ class-scope braces."""
    code: list[str] = []
    index = 0
    while index < len(line):
        if in_block_comment:
            end = line.find("*/", index)
            if end < 0:
                return "".join(code), True
            in_block_comment = False
            index = end + 2
            continue

        if line.startswith("//", index):
            break
        if line.startswith("/*", index):
            in_block_comment = True
            index += 2
            continue

        character = line[index]
        if character in {'"', "'"}:
            quote = character
            code.append(" ")
            index += 1
            while index < len(line):
                if line[index] == "\\":
                    index += 2
                    continue
                if line[index] == quote:
                    index += 1
                    break
                index += 1
            continue

        code.append(character)
        index += 1

    return "".join(code), in_block_comment


def load_compile_commands(path: Path) -> list[dict[str, object]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"cannot read compile commands: {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON compile commands: {path}: {error}") from error

    if not isinstance(data, list):
        raise RuntimeError(f"compile commands must be a JSON list: {path}")
    return data


def public_headers(root: Path) -> list[tuple[Path, str]]:
    # Assign each header to the most-specific matching root (longest relative path), so a header
    # under a nested DSO root (e.g. holoscan/logger -> HOLOSCAN_LOGGER_EXPORT) is not also checked
    # against a broader root (holoscan -> HOLOSCAN_CORE_EXPORT).
    ordered_roots = sorted(PUBLIC_HEADER_ROOTS, key=lambda entry: len(entry[0]), reverse=True)
    headers: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for relative_root, export_macro in ordered_roots:
        header_root = root / relative_root
        if not header_root.exists():
            continue
        for path in header_root.rglob("*"):
            if path.suffix not in {".h", ".hh", ".hpp", ".hxx"} or "detail" in path.parts:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            normalized = path.as_posix()
            selected_macro = next(
                (
                    macro
                    for suffix, macro in PUBLIC_HEADER_FILE_MACROS.items()
                    if normalized.endswith(suffix)
                ),
                export_macro,
            )
            headers.append((path, selected_macro))
    return headers


def check_header(
    path: Path, export_macro: str, allowed_declarations: set[str] | None = None
) -> list[str]:
    errors: list[str] = []
    allowed_declarations = allowed_declarations or set()
    in_preprocessor_continuation = False
    in_block_comment = False
    exported_class_pending = False
    exported_class_depth = 0
    function_body_depth = 0
    exported_class_re = re.compile(rf"\b(?:class|struct)\s+{re.escape(export_macro)}\b")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        code, in_block_comment = code_without_comments_or_literals(line, in_block_comment)
        stripped = code.strip()
        if in_preprocessor_continuation:
            in_preprocessor_continuation = line.rstrip().endswith("\\")
            continue
        if stripped.startswith("#"):
            in_preprocessor_continuation = line.rstrip().endswith("\\")
            continue
        if function_body_depth > 0:
            function_body_depth += code.count("{") - code.count("}")
            continue
        if exported_class_depth > 0:
            exported_class_depth += code.count("{") - code.count("}")
            continue

        if exported_class_pending:
            opening_brace = code.find("{")
            terminator = code.find(";")
            if terminator >= 0 and (opening_brace < 0 or terminator < opening_brace):
                exported_class_pending = False
                continue
            if opening_brace >= 0:
                exported_class_pending = False
                exported_class_depth = code.count("{") - code.count("}")
            continue

        if exported_class_re.search(code):
            opening_brace = code.find("{")
            terminator = code.find(";")
            if opening_brace >= 0 and (terminator < 0 or opening_brace < terminator):
                exported_class_depth = code.count("{") - code.count("}")
            elif terminator < 0:
                exported_class_pending = True
            continue

        if not stripped or export_macro in stripped:
            continue

        # Ignore declarations and calls inside inline/template function bodies. The lexical
        # declaration matcher deliberately remains simple, so a wrapped call such as
        # `package, header(), identity, generation, bytes());` can otherwise resemble a free
        # function declaration. Keep scanning unexported class scope itself so genuine member
        # declarations still require an export boundary.
        opening_brace = code.find("{")
        if opening_brace >= 0 and ")" in code[:opening_brace]:
            function_body_depth = code.count("{") - code.count("}")
            continue
        if DECLARATION_RE.match(line):
            if declaration_key(path, stripped) in allowed_declarations:
                continue
            errors.append(
                f"{path}:{line_number}: exported function declaration lacks {export_macro}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the staged public-header export macro semantic check. "
            "The compile database is validated so CI catches stale CMake configuration."
        )
    )
    parser.add_argument("--root", default=".", help="Repository or module root to scan.")
    parser.add_argument(
        "--compile-commands",
        help="Path to compile_commands.json generated by CMake.",
    )
    parser.add_argument(
        "--allowlist",
        default=str(DEFAULT_ALLOWLIST),
        help="Path to the exact path-qualified declaration allowlist.",
    )
    args = parser.parse_args()

    if args.compile_commands:
        compile_commands = Path(args.compile_commands)
        try:
            load_compile_commands(compile_commands)
        except RuntimeError as error:
            print(f"ERROR: {error}")
            return 1

    root = Path(args.root)
    try:
        allowlist = load_allowlist(Path(args.allowlist))
    except RuntimeError as error:
        print(f"ERROR: {error}")
        return 1

    errors: list[str] = []
    for path, export_macro in public_headers(root):
        section = ALLOWLIST_SECTIONS.get(export_macro)
        allowed_declarations = allowlist.get(section, set()) if section is not None else set()
        errors.extend(check_header(path, export_macro, allowed_declarations))

    if errors:
        print("Export macro semantic policy failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
