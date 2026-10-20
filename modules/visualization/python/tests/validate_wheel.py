#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the separately owned Holoscan Visualization wheel."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path, PurePosixPath

from packaging.utils import parse_wheel_filename

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.check_python_wheel import (  # noqa: E402
    WheelValidationError,
    _installed_path,
    validate_structure,
    validate_wheel_native_loader,
    validate_wheel_structure,
)

PACKAGE = "holoscan_visualization"
EXTENSION = PurePosixPath(PACKAGE, "_visualization.abi3.so")
HOLOVIZ = PurePosixPath(PACKAGE, "lib", "libholoviz.so.5")
OP_HOLOVIZ = PurePosixPath(PACKAGE, "lib", "libholoscan_op_holoviz.so.5")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--core-wheel", required=True, type=Path)
    parser.add_argument("--require-typing", action="store_true")
    args = parser.parse_args()

    distribution, version, _, tags = parse_wheel_filename(args.wheel.name)
    core_distribution, core_version, _, core_tags = parse_wheel_filename(args.core_wheel.name)
    if distribution != "holoscan-visualization-cu13":
        raise WheelValidationError(f"unexpected distribution: {distribution}")
    if core_distribution != "holoscan-core-cu13" or core_version != version or core_tags != tags:
        raise WheelValidationError("Core and Visualization versions/platform tags must match")

    native_members = validate_wheel_structure(
        args.wheel,
        package=PACKAGE,
        expected_extension=EXTENSION,
        expected_entry_points={"visualization": PACKAGE},
        required_dependencies=("holoscan-core-cu13",),
        require_typing=args.require_typing,
    )
    with zipfile.ZipFile(args.wheel) as wheel:
        installed = {_installed_path(PurePosixPath(name)) for name in wheel.namelist()}
    if any(member.parts[:1] == ("holoscan",) for member in installed):
        raise WheelValidationError("Visualization wheel must not own holoscan files")
    actual_libraries = {
        member
        for member in installed
        if member.parent == PurePosixPath(PACKAGE, "lib") and ".so" in member.name
    }
    expected_libraries = {HOLOVIZ, OP_HOLOVIZ}
    if actual_libraries != expected_libraries:
        raise WheelValidationError(
            f"Visualization native ownership mismatch: {sorted(map(str, actual_libraries))}"
        )

    core_members = validate_structure(args.core_wheel)
    validate_wheel_native_loader(
        args.wheel,
        native_members,
        required_runpaths={
            EXTENSION: ("$ORIGIN/lib", "$ORIGIN/../holoscan/lib"),
            HOLOVIZ: ("$ORIGIN", "$ORIGIN/../../holoscan/lib"),
            OP_HOLOVIZ: ("$ORIGIN", "$ORIGIN/../../holoscan/lib"),
        },
        required_linkage={
            EXTENSION: ("libholoscan_op_holoviz.so",),
            OP_HOLOVIZ: ("libholoviz.so", "libholoscan_core.so"),
        },
        companion_members=core_members,
    )
    print(f"Validated {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
