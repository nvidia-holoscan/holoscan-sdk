"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import os

import pytest

from holoscan.logger import (
    LogLevel,
    log_level,
    set_log_level,
    set_log_pattern,
)


def test_set_log_pattern():
    set_log_pattern(r"[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v")


@pytest.mark.parametrize(
    "level",
    [
        LogLevel.TRACE,
        LogLevel.DEBUG,
        LogLevel.INFO,
        LogLevel.WARN,
        LogLevel.ERROR,
        LogLevel.OFF,
    ],
)
def test_set_log_level(level):
    # remember existing environment variable
    orig_env = os.environ.get("HOLOSCAN_LOG_LEVEL")
    # remember current log level
    orig_level = log_level()
    try:
        # remove the environment variable
        if "HOLOSCAN_LOG_LEVEL" in os.environ:
            del os.environ["HOLOSCAN_LOG_LEVEL"]
        set_log_level(level)
        assert log_level() == level
        # set INFO to the environment variable
        os.environ["HOLOSCAN_LOG_LEVEL"] = "INFO"
        # now set_log_level should not change the log level
        set_log_level(level)
        assert log_level() == LogLevel.INFO
    finally:
        # restore the environment variable
        if orig_env is None:
            if "HOLOSCAN_LOG_LEVEL" in os.environ:
                del os.environ["HOLOSCAN_LOG_LEVEL"]
        else:
            os.environ["HOLOSCAN_LOG_LEVEL"] = orig_env
        # restore the logging level prior to the test
        set_log_level(orig_level)
