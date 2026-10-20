# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify CTest preparation ordering, focused selection, and failure isolation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def project(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    build = tmp_path / "build"
    helper = Path(__file__).with_name("consumer_test.cmake")
    (source / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 4.0)
project(consumer_ordering NONE)
enable_testing()
set(holoscan_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}")
set(holoscan_BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}")
include([==[{helper}]==])
set(CMAKE_CXX_COMPILER_LAUNCHER [==[tool path;--arg=two words;$<CONFIG>;$<TARGET_PROPERTY:NAME>]==])
set(CMAKE_C_COMPILER_LAUNCHER "")
set(HOLOSCAN_THIRD_PARTY_INSTALL_BIN_DIR "tools/holoscan")
add_subdirectory(a)
add_subdirectory(b)
holoscan_add_installed_consumer_test(command-shape
    SOURCE_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
    WORKING_DIRECTORY "${{CMAKE_CURRENT_BINARY_DIR}}/unused"
    CMAKE_ARGS "-DHOLOSCAN_EXTRA_BUILD_TARGETS=alpha;beta")
"""
    )
    (source / "record.py").write_text(
        "import pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(time.monotonic_ns()))\n"
        "sys.exit(1 if sys.argv[2] == 'ON' else 0)\n"
    )
    for case in ("a", "b"):
        directory = source / case
        directory.mkdir()
        (directory / "CMakeLists.txt").write_text(
            f"""add_test(NAME case-{case}.prepare
    COMMAND [==[{sys.executable}]==] "${{holoscan_SOURCE_DIR}}/record.py"
        "${{holoscan_BINARY_DIR}}/{case}.prepare" "${{FAIL_{case}}}")
set_tests_properties(case-{case}.prepare PROPERTIES
    RUN_SERIAL TRUE FIXTURES_SETUP prepared-{case})
add_test(NAME case-{case}.consumer
    COMMAND [==[{sys.executable}]==] "${{holoscan_SOURCE_DIR}}/record.py"
        "${{holoscan_BINARY_DIR}}/{case}.consumer" OFF)
set_tests_properties(case-{case}.consumer PROPERTIES
    FIXTURES_REQUIRED prepared-{case} PROCESSORS 2 COST 100)
holoscan_order_prepared_consumer(case-{case}.consumer case-{case}.prepare)
"""
        )
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], check=True
    )
    return source, build


def run_tests(build, pattern, jobs=4):
    return subprocess.run(
        ["ctest", "--test-dir", str(build), "-R", pattern, "-j", str(jobs), "--output-on-failure"],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_selected_preparations_finish_before_any_consumer(project):
    _, build = project
    result = run_tests(build, "^case-")
    assert result.returncode == 0, result.stdout + result.stderr
    prep = [int((build / f"{case}.prepare").read_text()) for case in ("a", "b")]
    runs = [int((build / f"{case}.consumer").read_text()) for case in ("a", "b")]
    assert max(prep) < min(runs)


def test_focused_selection_only_adds_its_own_fixture(project):
    _, build = project
    result = run_tests(build, "^case-a.consumer$", jobs=1)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build / "a.prepare").exists()
    assert (build / "a.consumer").exists()
    assert not (build / "b.prepare").exists()
    assert not (build / "b.consumer").exists()


def test_failed_preparation_only_skips_its_own_consumer(project):
    source, build = project
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-DFAIL_b=ON"], check=True)
    result = run_tests(build, "^case-")
    assert result.returncode != 0
    assert (build / "a.consumer").exists()
    assert not (build / "b.consumer").exists()


def test_preserves_list_arguments_and_reserves_compiler_slots(project):
    _, build = project
    inventory = json.loads(
        subprocess.check_output(
            ["ctest", "--test-dir", str(build), "--show-only=json-v1"], text=True
        )
    )
    test = next(test for test in inventory["tests"] if test["name"] == "command-shape")
    assert "-DHOLOSCAN_EXTRA_BUILD_TARGETS=alpha;beta" in test["command"]
    assert "-DHOLOSCAN_THIRD_PARTY_INSTALL_BIN_DIR=tools/holoscan" in test["command"]
    properties = {prop["name"]: prop["value"] for prop in test["properties"]}
    assert properties["PROCESSORS"] == 2
    assert not properties.get("RUN_SERIAL", False)


@pytest.mark.parametrize("override", [None, "explicit-launcher"])
def test_launcher_cache_preserves_arguments_and_overrides(project, override):
    _, build = project
    cache = build / "unused" / "compiler-launchers.cmake"
    probe = build / "launcher-probe-source"
    probe.mkdir()
    (probe / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 4.0)\n"
        "project(launcher_probe NONE)\n"
        "add_library(consumer_launcher_target INTERFACE)\n"
        'if(NOT DEFINED CMAKE_C_COMPILER_LAUNCHER OR NOT CMAKE_C_COMPILER_LAUNCHER STREQUAL "")\n'
        '  message(FATAL_ERROR "Explicitly disabled compiler launcher changed")\n'
        "endif()\n"
        'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/launcher.txt"\n'
        '  CONTENT "$<GENEX_EVAL:${CMAKE_CXX_COMPILER_LAUNCHER}>"\n'
        "  TARGET consumer_launcher_target)\n"
    )
    probe_build = build / "launcher-probe-build"
    command = [
        "cmake",
        "-S",
        str(probe),
        "-B",
        str(probe_build),
        "-C",
        str(cache),
        "-DCMAKE_BUILD_TYPE=Debug",
    ]
    if override is not None:
        command.append(f"-DCMAKE_CXX_COMPILER_LAUNCHER={override}")
    subprocess.run(command, check=True)
    expected = override or "tool path;--arg=two words;Debug;consumer_launcher_target"
    assert (probe_build / "launcher.txt").read_text() == expected
