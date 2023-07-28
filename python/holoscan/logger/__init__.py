# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""This module provides a Python interface to the Holoscan SDK logger.

.. autosummary::

    holoscan.logger.LogLevel
    holoscan.logger.disable_backtrace
    holoscan.logger.dump_backtrace
    holoscan.logger.enable_backtrace
    holoscan.logger.flush
    holoscan.logger.flush_level
    holoscan.logger.flush_on
    holoscan.logger.log_level
    holoscan.logger.set_log_level
    holoscan.logger.set_log_pattern
    holoscan.logger.should_backtrace
"""

from ._logger import (
    LogLevel,
    disable_backtrace,
    dump_backtrace,
    enable_backtrace,
    flush,
    flush_level,
    flush_on,
    log_level,
    set_log_level,
    set_log_pattern,
    should_backtrace,
)

__all__ = [
    "LogLevel",
    "disable_backtrace",
    "dump_backtrace",
    "enable_backtrace",
    "flush",
    "flush_level",
    "flush_on",
    "log_level",
    "set_log_level",
    "set_log_pattern",
    "should_backtrace",
]
