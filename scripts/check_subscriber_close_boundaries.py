#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Enforce the external subscriber local-close source boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RUNTIME_SUBSCRIBER_CLOSE_RE = re.compile(
    r"\b(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\.|->)\s*close\s*\(",
)
RUNTIME_SUBSCRIBER_DECLARATION_RE = re.compile(
    r"\b(?:std::optional\s*<\s*)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*::)*"
    r"(?:ChannelSubscriber|ChannelTensorSchemaSubscriber)\s*<[^>\n;{}]+>\s*"
    r">?\s*(?:[&*]\s*)?(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\b",
)
RUNTIME_SUBSCRIBER_ALIAS_RE = re.compile(
    r"\b(?:const\s+)?auto\s*[&*]+\s*(?P<alias>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*=\s*(?P<source>[A-Za-z_][A-Za-z0-9_]*)\b"
)
RUNTIME_SUBSCRIBER_FACTORY_ASSIGNMENT_RE = re.compile(
    r"\b(?:const\s+)?auto(?:\s+|\s*&&?\s*)(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"[^;{}]*?(?:\.|->)\s*(?:create_subscriber|create_tensor_schema_subscriber)"
    r"\s*(?:<[^;{}]*?>)?\s*\("
)
RUNTIME_PRIVATE_CLOSE_RE = re.compile(r"\bclose_local_subscriber\s*\(")
MESSAGE_ERROR_RE = re.compile(r"\bmake_channel_error\s*\(")


def mask_cpp_non_code(text: str) -> str:
    """Replace comments and quoted literals with spaces while preserving newlines."""
    result = list(text)
    index = 0
    state = "code"
    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code":
            if current == "/" and following == "/":
                result[index] = result[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if current == "/" and following == "*":
                result[index] = result[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if current in {'"', "'"}:
                result[index] = " "
                state = "string" if current == '"' else "character"
        elif state == "line_comment":
            if current == "\n":
                state = "code"
            else:
                result[index] = " "
        elif state == "block_comment":
            if current == "*" and following == "/":
                result[index] = result[index + 1] = " "
                index += 2
                state = "code"
                continue
            if current != "\n":
                result[index] = " "
        else:
            result[index] = " "
            if current == "\\" and following:
                if following != "\n":
                    result[index + 1] = " "
                index += 2
                continue
            if (state == "string" and current == '"') or (state == "character" and current == "'"):
                state = "code"
        index += 1
    return "".join(result)


def function_bodies(text: str, signature: re.Pattern[str]) -> list[str]:
    """Return balanced function bodies whose signatures match masked C++ source."""
    masked = mask_cpp_non_code(text)
    bodies: list[str] = []
    for match in signature.finditer(masked):
        opening = masked.find("{", match.end())
        if opening < 0:
            continue
        depth = 0
        for index in range(opening, len(masked)):
            if masked[index] == "{":
                depth += 1
            elif masked[index] == "}":
                depth -= 1
                if depth == 0:
                    bodies.append(masked[opening : index + 1])
                    break
    return bodies


def check_runtime_source(path: Path, text: str) -> list[str]:
    """Reject local subscriber retirement from recurring Runtime ownership."""
    masked = mask_cpp_non_code(text)
    errors: list[str] = []
    if RUNTIME_PRIVATE_CLOSE_RE.search(masked):
        errors.append(f"{path}: Runtime must not call close_local_subscriber()")
    subscriber_receivers = {
        match.group("receiver") for match in RUNTIME_SUBSCRIBER_DECLARATION_RE.finditer(masked)
    }
    subscriber_receivers.update(
        match.group("receiver")
        for match in RUNTIME_SUBSCRIBER_FACTORY_ASSIGNMENT_RE.finditer(masked)
    )
    aliases_changed = True
    while aliases_changed:
        aliases_changed = False
        for match in RUNTIME_SUBSCRIBER_ALIAS_RE.finditer(masked):
            if (
                match.group("source") in subscriber_receivers
                and match.group("alias") not in subscriber_receivers
            ):
                subscriber_receivers.add(match.group("alias"))
                aliases_changed = True
    if any(
        match.group("receiver") in subscriber_receivers
        for match in RUNTIME_SUBSCRIBER_CLOSE_RE.finditer(masked)
    ):
        errors.append(f"{path}: Runtime must not call public subscriber close()")
    return errors


def check_close_body(
    path: Path, text: str, signature: re.Pattern[str], *, require_delegate: bool
) -> list[str]:
    """Require one allocation-free close body and its common retirement delegate."""
    bodies = function_bodies(text, signature)
    if len(bodies) != 1:
        return [f"{path}: expected exactly one subscriber-close body, found {len(bodies)}"]
    body = bodies[0]
    errors: list[str] = []
    if MESSAGE_ERROR_RE.search(body):
        errors.append(f"{path}: subscriber close must not call make_channel_error()")
    if require_delegate and not RUNTIME_PRIVATE_CLOSE_RE.search(body):
        errors.append(f"{path}: subscriber close must delegate to close_local_subscriber()")
    return errors


def public_root(repository_root: Path) -> Path:
    """Resolve either the internal repository or exported public layout."""
    nested = repository_root / "public"
    return nested if (nested / "src" / "runtime").is_dir() else repository_root


def check_repository(repository_root: Path) -> list[str]:
    """Check all production sources that own the subscriber-close boundary."""
    root = public_root(repository_root)
    errors: list[str] = []
    runtime_root = root / "src" / "runtime"
    for path in sorted(runtime_root.rglob("*")):
        if path.is_file() and path.suffix in {".cpp", ".cu", ".cuh", ".hpp"}:
            errors.extend(check_runtime_source(path, path.read_text(encoding="utf-8")))

    targets = (
        (
            root / "include" / "holoscan" / "channel" / "endpoint_interfaces.hpp",
            re.compile(
                r"expected\s*<\s*void\s*,\s*Error\s*>\s+close_local_endpoint\s*\(\s*\)\s*noexcept"
            ),
            False,
        ),
        (
            root / "include" / "holoscan" / "channel" / "endpoint_interfaces.hpp",
            re.compile(r"expected\s*<\s*void\s*,\s*Error\s*>\s+close_local_subscriber\s*\("),
            False,
        ),
        (
            root / "include" / "holoscan" / "channel" / "inline_endpoints.hpp",
            re.compile(r"expected\s*<\s*void\s*,\s*Error\s*>\s+close\s*\(\s*\)\s*noexcept"),
            True,
        ),
        (
            root / "include" / "holoscan" / "channel" / "tensor_schema_endpoints.hpp",
            re.compile(r"expected\s*<\s*void\s*,\s*Error\s*>\s+close\s*\(\s*\)\s*noexcept"),
            True,
        ),
        (
            root / "src" / "channel" / "channel_tensor.cpp",
            re.compile(
                r"expected\s*<\s*void\s*,\s*Error\s*>\s+"
                r"ChannelSubscriber\s*<\s*Tensor\s*>::close\s*\(\s*\)\s*noexcept"
            ),
            True,
        ),
    )
    for path, signature, require_delegate in targets:
        if not path.is_file():
            errors.append(f"{path}: subscriber-close source is missing")
            continue
        errors.extend(
            check_close_body(
                path,
                path.read_text(encoding="utf-8"),
                signature,
                require_delegate=require_delegate,
            )
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    errors = check_repository(args.root.resolve())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
