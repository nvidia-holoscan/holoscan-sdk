"""
SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
def ping_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "app_config_ping.yaml")
    return config_file


@pytest.fixture
def deprecated_extension_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "deprecated_stream_playback.yaml")
    return config_file


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
