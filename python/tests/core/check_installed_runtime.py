# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test an installed Core package without depending on typing artifacts."""

import argparse
import importlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()

    holoscan = importlib.import_module("holoscan")
    core = importlib.import_module("holoscan._core")
    execution = importlib.import_module("holoscan.execution")
    package_root = args.package_root.resolve()

    assert Path(holoscan.__file__).resolve().is_relative_to(package_root)
    assert Path(core.__file__).resolve().is_relative_to(package_root)
    assert core.__file__.endswith("/_core.abi3.so")
    assert repr(holoscan.Graph("installed-runtime")) == (
        "<Graph 'installed-runtime' operators=0 flows=0>"
    )
    support_binding = execution.DeploymentBinding()
    assert (
        support_binding.bind_support_thread_affinity(
            execution.SupportThreadRole.CONTROLLER,
            execution.CPUAffinity([0]),
        )
        is support_binding
    )


if __name__ == "__main__":
    main()
