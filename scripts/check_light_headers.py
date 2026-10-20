#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Check that light Core headers do not reach heavy Core headers.

The manifest lists light headers (vocabulary leaves and headers kept free of the authoring stack)
and heavy headers (the authoring stack and the Tensor storage and logging chain). A light header
fails the check when any header in its Holoscan include closure, direct or transitive, is heavy.
Headers the manifest does not list are not checked, so a new header needs no entry.
"""

import argparse
import re
import sys
from pathlib import Path

SECTIONS = ("light", "heavy")
INCLUDE_RE = re.compile(r"^\s*#\s*include\s*<holoscan/([^>]+)>", re.MULTILINE)
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "core_light_headers.txt"


def parse_manifest(text: str, name: str = "manifest") -> tuple[dict[str, list[str]], list[str]]:
    """Return the headers of each section and any parse errors."""
    sections: dict[str, list[str]] = {section: [] for section in SECTIONS}
    errors = []
    section = None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            if section not in SECTIONS:
                errors.append(f"{name}:{number}: unknown section [{section}]")
                section = None
            continue
        if section is None:
            errors.append(f"{name}:{number}: entry outside a section")
            continue
        if line in sections[section]:
            errors.append(f"{name}:{number}: {line} is listed more than once")
        sections[section].append(line)
    return sections, errors


def includes(include_root: Path, header: str) -> list[str]:
    """Return the Holoscan headers ``header`` includes that exist under ``include_root``."""
    text = (include_root / "holoscan" / header).read_text(encoding="utf-8")
    return [
        name for name in INCLUDE_RE.findall(text) if (include_root / "holoscan" / name).is_file()
    ]


def first_heavy_path(include_root: Path, header: str, heavy: set[str]) -> list[str] | None:
    """Return the shortest include chain from ``header`` to a heavy header, if any."""
    parents: dict[str, str] = {header: ""}
    queue = [header]
    while queue:
        current = queue.pop(0)
        for included in includes(include_root, current):
            if included in parents:
                continue
            parents[included] = current
            if included in heavy:
                chain = [included]
                while parents[chain[-1]]:
                    chain.append(parents[chain[-1]])
                return list(reversed(chain))
            queue.append(included)
    return None


def check(include_root: Path, manifest_text: str, manifest_name: str = "manifest") -> list[str]:
    sections, errors = parse_manifest(manifest_text, manifest_name)
    for section in SECTIONS:
        errors.extend(
            f"{manifest_name}: [{section}] {header} is not an installed header; remove it"
            for header in sections[section]
            if not (include_root / "holoscan" / header).is_file()
        )
    if errors:
        return errors
    heavy = set(sections["heavy"])
    for header in sections["light"]:
        chain = first_heavy_path(include_root, header, heavy)
        if chain:
            errors.append(
                f"{header} is listed as light but reaches heavy holoscan/{chain[-1]}: "
                + " -> ".join(chain)
            )
    return errors


def sdk_root(root: Path) -> Path:
    """Return the SDK source root: ``root`` itself, or its ``public/`` directory internally."""
    for candidate in (root, root / "public"):
        if (candidate / "include" / "holoscan").is_dir():
            return candidate
    raise SystemExit(f"{root}: no include/holoscan directory under it or its public/ directory")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[1])
    parser.add_argument("--root", type=Path, default=Path("."), help="repository or SDK root")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    errors = check(
        sdk_root(args.root) / "include",
        args.manifest.read_text(encoding="utf-8"),
        args.manifest.name,
    )
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
