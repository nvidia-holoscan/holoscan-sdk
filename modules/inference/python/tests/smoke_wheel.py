#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify an installed Inference wheel without running an inference model."""

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
    first = "holoscan_inference" if args.import_order == "canonical" else "holoscan.inference"
    importlib.import_module(first)
    provider = importlib.import_module("holoscan_inference")
    alias = importlib.import_module("holoscan.inference")
    extension = importlib.import_module("holoscan_inference._inference")
    operators = importlib.import_module("holoscan_inference.operators")

    assert provider is alias is holoscan.inference
    assert importlib.import_module("holoscan.inference._inference") is extension
    assert extension.__file__.endswith("/_inference.abi3.so")
    assert operators.InferenceOp.name == "holoscan.inference.InferenceOp"

    package = Path(provider.__file__).parent
    if args.require_typing:
        for relative in (
            "py.typed",
            "__init__.pyi",
            "_inference.pyi",
            "operators/__init__.pyi",
        ):
            assert (package / relative).is_file(), relative
    _check_loaded_runtime(
        package,
        ("libholoinfer_utils.so", "libholoscan_infer.so", "libholoscan_op_inference.so"),
    )
    print(f"Qualified Inference ({args.import_order} first) on {sys.version}")


if __name__ == "__main__":
    main()
