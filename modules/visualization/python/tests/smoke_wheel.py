#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify an installed Visualization wheel without creating a display."""

import argparse
import importlib
import sys
import sysconfig
from pathlib import Path


def _check_loaded_runtime(package: Path, library_names: tuple[str, ...]) -> None:
    mapped_paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            mapped_paths.add(Path(fields[5]).resolve())
    library_directory = (package / "lib").resolve()
    for name in library_names:
        matches = {
            path
            for path in mapped_paths
            if path.name == name or path.name.startswith((f"{name}.", f"{name}-"))
        }
        assert len(matches) == 1, (name, matches)
        assert matches.pop().parent == library_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-typing", action="store_true")
    parser.add_argument("--import-order", choices=("canonical", "alias"), default="canonical")
    args = parser.parse_args()

    assert sys.implementation.name == "cpython"
    assert (3, 12) <= sys.version_info[:2] <= (3, 14)
    assert not sysconfig.get_config_var("Py_GIL_DISABLED")

    holoscan = importlib.import_module("holoscan")
    first = (
        "holoscan_visualization" if args.import_order == "canonical" else "holoscan.visualization"
    )
    importlib.import_module(first)
    provider = importlib.import_module("holoscan_visualization")
    alias = importlib.import_module("holoscan.visualization")
    extension = importlib.import_module("holoscan_visualization._visualization")
    operators = importlib.import_module("holoscan_visualization.operators")

    assert provider is alias is holoscan.visualization
    assert importlib.import_module("holoscan.visualization._visualization") is extension
    assert extension.__file__.endswith("/_visualization.abi3.so")
    assert operators.HolovizOp.name == "holoscan.visualization.HolovizOp"
    typed_layer = operators.ImageLayer(port_name="video", max_width=1, max_height=1)
    assert typed_layer.kind == "image"
    assert operators.ImageFormat.RGBA8_UNORM == "rgba8_unorm"
    assert "ImageFormat" in operators.__all__

    package = Path(provider.__file__).parent
    if args.require_typing:
        for relative in (
            "py.typed",
            "__init__.pyi",
            "_visualization.pyi",
            "operators/__init__.pyi",
        ):
            assert (package / relative).is_file(), relative
    _check_loaded_runtime(package, ("libholoviz.so", "libholoscan_op_holoviz.so"))
    print(f"Qualified Visualization ({args.import_order} first) on {sys.version}")


if __name__ == "__main__":
    main()
