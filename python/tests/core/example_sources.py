# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load shipped Python examples for source-level tests."""

import importlib.util
import os
from pathlib import Path
from types import ModuleType


def load_shipped_example(name: str) -> ModuleType:
    """Load one example directly from the public source tree."""
    source_root = Path(os.environ["HOLOSCAN_SHIPPED_EXAMPLE_SOURCE_DIR"])
    source = source_root / name / "python" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"holoscan_example_{name}", source)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
