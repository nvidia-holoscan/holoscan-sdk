# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional package contracts also run against the installed wheel environment."""

import ast
from pathlib import Path

import pytest

import holoscan.testing as testing


# REQ: PY-API-026
def test_optional_package_exports_have_public_typing():
    package = Path(testing.__file__).parent
    if not (package / "py.typed").exists():
        pytest.skip("Python stubs disabled in this SDK build")
    stub = ast.parse((package / "__init__.pyi").read_text())
    declarations = {
        node.name for node in stub.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert set(testing.__all__) <= declarations
    assert all(getattr(testing, name).__module__ == "holoscan.testing" for name in testing.__all__)
