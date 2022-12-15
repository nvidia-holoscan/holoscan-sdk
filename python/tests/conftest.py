# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
def config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "operator_parameters.yaml")
    return config_file


@pytest.fixture
def ping_config_file():
    yaml_file_dir = os.path.dirname(__file__)
    config_file = os.path.join(yaml_file_dir, "app_config_ping.yaml")
    return config_file
