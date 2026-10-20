# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify external tl-expected discovery without a native SDK build."""

import subprocess
from pathlib import Path

import pytest

SDK = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("bundled_fmt", [True, False])
@pytest.mark.parametrize("provider_version", ["1.1.0", "1.0.0", None])
def test_expected_consumer_only_allows_external_expected(tmp_path, bundled_fmt, provider_version):
    prefix = tmp_path / "sdk"
    external = tmp_path / "external"
    expected_dir = external / "lib/cmake/tl-expected"
    expected_dir.mkdir(parents=True)
    (expected_dir / "tl-expected-config.cmake").write_text(
        "if(NOT TARGET tl::expected)\n  add_library(tl::expected INTERFACE IMPORTED)\nendif()\n"
    )
    (expected_dir / "tl-expected-config-version.cmake").write_text(
        f'set(PACKAGE_VERSION "{provider_version}")\n'
        "if(PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)\n"
        "  set(PACKAGE_VERSION_COMPATIBLE FALSE)\n"
        "else()\n"
        "  set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
        "endif()\n"
    )
    if provider_version is None:
        (expected_dir / "tl-expected-config.cmake").unlink()
    external_fmt = external / "lib/cmake/fmt"
    external_fmt.mkdir(parents=True)
    (external_fmt / "fmt-config.cmake").write_text(
        'message(FATAL_ERROR "Bundled fmt was replaced by an external package")\n'
    )
    if bundled_fmt:
        fmt_dir = prefix / "lib/cmake/fmt"
        fmt_dir.mkdir(parents=True)
        (fmt_dir / "fmt-config.cmake").write_text("set(fmt_FOUND TRUE)\n")
    package_dir = prefix / "lib/cmake/holoscan"
    package_dir.mkdir(parents=True)
    (package_dir / "holoscan-config.cmake").write_text(
        "include(CMakeFindDependencyMacro)\n"
        "find_dependency(tl-expected)\n"
        "find_dependency(fmt CONFIG)\n"
        "add_library(holoscan::core INTERFACE IMPORTED)\n"
        "add_library(holoscan::logger INTERFACE IMPORTED)\n"
    )
    build = tmp_path / "build"
    # project() initializes platform search prefixes, so override them afterward. Keep the
    # synthetic provider discoverable only through system paths: a cached tl-expected_DIR
    # would hide a regression that removes discovery before the consumer restricts those paths.
    search_paths = tmp_path / "search_paths.cmake"
    search_paths.write_text(
        f'set(CMAKE_SYSTEM_PREFIX_PATH "{external}")\n'
        "set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)\n"
        "set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)\n"
        "set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)\n"
        "set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)\n"
        "set(CMAKE_FIND_USE_INSTALL_PREFIX FALSE)\n"
    )
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(SDK / "tests/system/expected_installed_consumer"),
            "-B",
            str(build),
            f"-DHOLOSCAN_TEST_INSTALL_PREFIX={prefix}",
            f"-DCMAKE_PROJECT_holoscan_expected_installed_consumer_INCLUDE={search_paths}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    if provider_version is None:
        assert result.returncode != 0
        assert 'provided by "tl-expected"' in output
    elif bundled_fmt:
        assert result.returncode == 0, output
        assert f"tl-expected_DIR:PATH={expected_dir}" in (build / "CMakeCache.txt").read_text()
    else:
        assert result.returncode != 0
        assert 'provided by "fmt"' in output
        assert "Bundled fmt was replaced" not in output
