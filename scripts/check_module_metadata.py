#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the metadata.json of every Core Module under modules/.

Each module directory (a direct child of modules/ with a CMakeLists.txt) must
contain a metadata.json that conforms to modules/metadata.schema.json
(urn:holoscan:module:v2), is named holoscan-<module>, and is versioned with
the SDK VERSION file.

Run from any directory:

  python3 scripts/check_module_metadata.py

Exit code 0 if every module passes; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SDK_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_RELATIVE_PATH = Path("modules") / "metadata.schema.json"


def module_dirs(sdk_root: Path) -> list[Path]:
    modules_root = sdk_root / "modules"
    return sorted(p for p in modules_root.iterdir() if (p / "CMakeLists.txt").is_file())


def check_module(module_dir: Path, validator: Draft202012Validator, sdk_version: str) -> list[str]:
    metadata_path = module_dir / "metadata.json"
    if not metadata_path.is_file():
        return [f"{metadata_path}: missing; every Core Module requires a metadata.json"]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{metadata_path}: invalid JSON: {exc}"]

    errors = [
        f"{metadata_path}: {'/'.join(map(str, err.absolute_path)) or '<root>'}: {err.message}"
        for err in sorted(validator.iter_errors(metadata), key=lambda e: list(e.absolute_path))
    ]
    if errors:
        return errors

    module = metadata["module"]
    expected_name = f"holoscan-{module_dir.name}"
    if module["name"] != expected_name:
        errors.append(
            f"{metadata_path}: module.name is {module['name']!r}, expected {expected_name!r}"
        )
    debian = module["binary_packages"]["debian"]
    if debian != expected_name:
        errors.append(
            f"{metadata_path}: module.binary_packages.debian is {debian!r}, "
            f"expected {expected_name!r}"
        )
    if module["version"] != sdk_version:
        errors.append(
            f"{metadata_path}: module.version is {module['version']!r}, "
            f"expected the SDK version {sdk_version!r}"
        )
    sdk = module["holoscan_sdk"]
    if sdk["minimum_required_version"] != sdk_version:
        errors.append(
            f"{metadata_path}: module.holoscan_sdk.minimum_required_version is "
            f"{sdk['minimum_required_version']!r}, expected {sdk_version!r}"
        )
    if sdk_version not in sdk["tested_versions"]:
        errors.append(
            f"{metadata_path}: module.holoscan_sdk.tested_versions does not include {sdk_version!r}"
        )
    readme = module_dir / module["documentation"]["readme"]
    if not readme.is_file():
        errors.append(f"{metadata_path}: module.documentation.readme {readme} does not exist")
    return errors


def check_all(sdk_root: Path) -> list[str]:
    schema = json.loads((sdk_root / SCHEMA_RELATIVE_PATH).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    sdk_version = (sdk_root / "VERSION").read_text(encoding="utf-8").strip()
    errors: list[str] = []
    for module_dir in module_dirs(sdk_root):
        errors.extend(check_module(module_dir, validator, sdk_version))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=SDK_ROOT,
        help=f"SDK source root containing VERSION and modules/ (default: {SDK_ROOT})",
    )
    args = parser.parse_args(argv)
    errors = check_all(args.root.resolve())
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
