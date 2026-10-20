# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify private dependency install layout without a native SDK build."""

import subprocess
from pathlib import Path

import pytest

SDK = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("includedir", "libdir", "bindir"),
    [("include", "lib", "bin"), ("headers", "lib/aarch64-linux-gnu", "tools")],
)
def test_predeclared_flatbuffers_installs_in_private_dirs(tmp_path, includedir, libdir, bindir):
    headers = tmp_path / "headers/flatbuffers"
    headers.mkdir(parents=True)
    (headers / "flatbuffers.h").write_text("// fixture\n")
    flatc = tmp_path / "flatc"
    flatc.write_text("fixture generator\n")
    build = tmp_path / "build"
    prefix = tmp_path / "install"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(SDK / "tests/system/flatbuffers_predeclared_provider"),
            "-B",
            str(build),
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            f"-DCMAKE_INSTALL_INCLUDEDIR={includedir}",
            f"-DCMAKE_INSTALL_LIBDIR={libdir}",
            f"-DCMAKE_INSTALL_BINDIR={bindir}",
            f"-DHOLOSCAN_FLATC_EXECUTABLE={flatc}",
            f"-DHOLOSCAN_FLATBUFFERS_INCLUDE_DIR={headers.parent}",
            f"-DHOLOSCAN_FLATBUFFERS_DEPENDENCY_FILE={SDK / 'cmake/deps/flatbuffers.cmake'}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(["cmake", "--install", str(build)], check=True, timeout=30)
    assert (prefix / includedir / "holoscan/third_party/flatbuffers/flatbuffers.h").is_file()
    assert (prefix / bindir / "holoscan/flatc").read_text() == flatc.read_text()
    assert not (prefix / includedir / "flatbuffers").exists()
    assert not (prefix / bindir / "flatc").exists()
