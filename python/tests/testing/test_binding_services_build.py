# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual compatibility generator with isolated declaration fixtures."""

import shutil
import subprocess
from pathlib import Path

import pytest


# REQ: PY-API-026
def test_bridge_fingerprint_tracks_declarations_and_preserves_abi3(tmp_path):
    cmake = shutil.which("cmake")
    helper = (
        Path(__file__).resolve().parents[3]
        / "python/holoscan/cmake/BindingServicesCompatibility.cmake"
    )
    if cmake is None or not helper.is_file():
        pytest.skip("CMake and SDK source are needed for the isolated configure fixture")
    source = tmp_path / "source"
    for name, content in {
        "sdk/value.hpp": "struct Value { int value; };\n",
        "core/binding_services.hpp": "struct Bridge { int (*convert)(int); };\n",
        "nanobind/include/nanobind.h": "#define NB_VERSION_MAJOR 3\n",
        "nanobind/src/nb_internals.h": "#define NB_INTERNALS_VERSION 22\n",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(binding_services_fixture NONE)\n"
        f'include("{helper.as_posix()}")\n'
        "holoscan_configure_binding_services_compatibility(\n"
        '  "${CMAKE_CURRENT_SOURCE_DIR}/sdk" "${CMAKE_CURRENT_SOURCE_DIR}/core"\n'
        '  "${CMAKE_CURRENT_SOURCE_DIR}/nanobind" "${CMAKE_CURRENT_BINARY_DIR}/generated")\n'
    )

    def configure(directory, *options):
        build = tmp_path / (directory.name + "-build")
        result = subprocess.run(
            [cmake, "-S", str(directory), "-B", str(build), *options],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return (build / "generated/binding_services_compatibility.hpp").read_text()

    baseline = configure(source, "-DPython_VERSION=3.12.0")
    assert configure(source, "-DPython_VERSION=3.14.0") == baseline
    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    assert configure(relocated, "-DPython_VERSION=3.13.0") == baseline

    bridge = source / "core/binding_services.hpp"
    original = bridge.read_text()
    # The function-pointer field keeps its C++ size while its call signature changes.
    bridge.write_text(original.replace("(int)", "(char)"))
    assert configure(source) != baseline
    bridge.write_text(original)
    assert configure(source, "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0") != baseline
    assert configure(source, "-DCMAKE_CXX_FLAGS=") == baseline
    (source / "nanobind/src/nb_internals.h").write_text("#define NB_INTERNALS_VERSION 23\n")
    assert configure(source) != baseline
