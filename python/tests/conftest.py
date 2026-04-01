"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import os
import re
import subprocess
from xml.etree import ElementTree

import pytest

from holoscan.core import Application, Fragment
from holoscan.logger import LogLevel, set_log_level

# set log level to INFO during testing
set_log_level(LogLevel.INFO)


@pytest.fixture
def app():
    return Application()


@pytest.fixture
def fragment():
    return Fragment()


@pytest.fixture
def operators_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "operator_parameters.yaml")
    return config_file


@pytest.fixture
def data_loggers_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "data_logger_parameters.yaml")
    return config_file


@pytest.fixture
def subgraphs_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "subgraph_parameters.yaml")
    return config_file


@pytest.fixture
def ping_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "app_config_ping.yaml")
    return config_file


@pytest.fixture
def deprecated_extension_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "deprecated_stream_playback.yaml")
    return config_file


# Cached result for Green Context availability.
_green_context_available = None
_cached_nvidia_smi_output = None
_cached_nvidia_smi_loaded = False
# Holoscan SDK FAQ prerequisite documents that CUDA Green Context features
# require CUDA Driver API version >= 12.4.
# Ref: https://docs.nvidia.com/holoscan/sdk-user-guide/hsdk_faq.html
MIN_GREEN_CONTEXT_CUDA_DRIVER_API = 12040


def _nvidia_smi_output():
    global _cached_nvidia_smi_loaded, _cached_nvidia_smi_output
    if _cached_nvidia_smi_loaded:
        return _cached_nvidia_smi_output
    _cached_nvidia_smi_loaded = True
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        _cached_nvidia_smi_output = None
        return None
    _cached_nvidia_smi_output = proc.stdout
    return _cached_nvidia_smi_output


def _cuda_driver_api_version():
    output = _nvidia_smi_output()
    if not output:
        return None
    try:
        root = ElementTree.fromstring(output)
    except ElementTree.ParseError:
        return None
    cuda_version = root.findtext("cuda_version")
    if not cuda_version:
        return None
    match = re.search(r"(\d+)\.(\d+)", cuda_version)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    return major * 1000 + minor * 10


def green_context_available():
    """Return True if Green Context is supported in this environment.

    Tests that require it should skip when this returns False.
    """
    global _green_context_available
    if _green_context_available is not None:
        return _green_context_available
    version = _cuda_driver_api_version()
    _green_context_available = version is not None and version >= MIN_GREEN_CONTEXT_CUDA_DRIVER_API
    return _green_context_available


def pytest_configure(config):  # noqa: ARG001
    os.environ["HOLOSCAN_DISABLE_BACKTRACE"] = "1"


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        "--run-slow",
        action="store_true",
        help="include tests marked slow (--runslow and --run-slow are equivalent)",
    )
    parser.addoption(
        "--run-realtime",
        action="store_true",
        default=False,
        help="run tests marked as requiring real-time kernel config",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow (or --runslow) option to run")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip_slow)

    if not config.getoption("--run-realtime"):
        skip_realtime = pytest.mark.skip(reason="need --run-realtime option to run")
        for item in items:
            if item.get_closest_marker("realtime"):
                item.add_marker(skip_realtime)


# Note: see [tool.pytest.ini_options] in pyproject.toml for marker definitions
