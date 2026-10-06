# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated Python typing artifact tests."""

import importlib.util
from pathlib import Path

import holoscan as hs
import holoscan.operators as operators
from holoscan._public_api import PUBLIC_API


# REQ: PY-API-016
def test_package_ships_typing_marker_and_public_facade_stubs():
    # Key setup: locate the installed package and its declared public facades.
    package_directory = Path(hs.__file__).parent

    # Key assertion: every public facade ships a stub alongside the typing marker.
    assert (package_directory / "py.typed").is_file()
    assert not (package_directory / "core").exists()
    assert importlib.util.find_spec("holoscan.core") is None
    for module_name in PUBLIC_API:
        parts = module_name.split(".")[1:]
        stub = (
            package_directory / "__init__.pyi"
            if not parts
            else package_directory.joinpath(*parts).with_suffix(".pyi")
        )
        if parts and (package_directory.joinpath(*parts)).is_dir():
            stub = package_directory.joinpath(*parts, "__init__.pyi")
        assert stub.is_file(), module_name

    root_text = (package_directory / "__init__.pyi").read_text(encoding="utf-8")
    assert "class Graph:" in root_text
    assert "def op[HandleT](" in root_text
    assert "_registry_OperatorType[HandleT]" in root_text
    assert "type_or_name: type[Operator]" in root_text
    assert "def embed[HandleT](" in root_text
    assert "_registry_GraphType[HandleT]" in root_text
    assert "class RunSession:" in root_text
    assert "holoscan._core" not in root_text
    assert "from . import registry as registry" in root_text

    payload_text = (package_directory / "payload/__init__.pyi").read_text(encoding="utf-8")
    assert "def __eq__(self, arg: object, /) -> bool" in payload_text


# REQ: PY-API-016
def test_operator_stub_exposes_only_the_core_handle_surface():
    # Key setup: Inspect the Core-owned Operator facade stub.
    operator_stub = Path(operators.__file__).with_suffix(".pyi")
    stub_text = operator_stub.read_text(encoding="utf-8")

    # Key assertion: Provider proxies are absent while the static handle API remains typed.
    assert "class OperatorHandle:" in stub_text
    assert "OperatorType" not in stub_text
    assert "def __getattr__(name: str)" not in stub_text
