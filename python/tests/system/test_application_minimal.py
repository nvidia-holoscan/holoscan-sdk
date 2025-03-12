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

import pytest
from env_wrapper import env_var_context

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingRxOp, PingTxOp
from holoscan.resources import ManualClock, RealtimeClock
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler


class MinimalOp(Operator):
    def __init__(self, *args, **kwargs):
        self.count = 1
        self.param_value = None
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def initialize(self):
        print("** initialize method called **")
        print(f"initialize(): param_value = {self.param_value}")

    def start(self):
        print("** start method called **")

    def setup(self, spec: OperatorSpec):
        spec.param("param_value", 500)

    def stop(self):
        print("** stop method called **")

    def compute(self, op_input, op_output, context):
        self.count += 1


class MinimalApp(Application):
    def compose(self):
        mx = MinimalOp(self, CountCondition(self, 10), name="mx")
        self.add_operator(mx)


@pytest.mark.parametrize("SchedulerClass", [None, GreedyScheduler, MultiThreadScheduler])
def test_minimal_app(ping_config_file, SchedulerClass, capfd):  # noqa: N803
    app = MinimalApp()
    app.config(ping_config_file)
    if SchedulerClass is not None:
        app.scheduler(SchedulerClass(app))
    app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()

    assert "error" not in captured.err
    assert "Exception occurred" not in captured.err

    # verify that the Python overrides of start and stop methods were called
    assert captured.out.count("** initialize method called **") == 1
    assert captured.out.count("** start method called **") == 1
    assert captured.out.count("** stop method called **") == 1

    # verity if parameter value is set
    assert "initialize(): param_value = 500" in captured.out


@pytest.mark.parametrize(
    "SchedulerClass", [EventBasedScheduler, GreedyScheduler, MultiThreadScheduler]
)
@pytest.mark.parametrize("ClockClass", [RealtimeClock, ManualClock])
def test_minimal_app_with_clock(ping_config_file, SchedulerClass, ClockClass):  # noqa: N803
    app = MinimalApp()
    app.config(ping_config_file)
    app.scheduler(SchedulerClass(app, clock=ClockClass(app)))
    app.run()


def test_app_ping_config_keys(ping_config_file):
    app = MinimalApp()
    app.config(ping_config_file)
    keys = app.config_keys()
    assert isinstance(keys, set)
    assert keys == {"mx", "mx.multiplier"}


def test_deprecated_extension(deprecated_extension_config_file, capfd):
    app = MinimalApp()
    app.config(deprecated_extension_config_file)

    app.run()

    captured_error = capfd.readouterr().err
    warning_msg = "no longer require specifying the libgxf_stream_playback.so extension"
    # deprecated extension is listed twice in the config file (once with full path)
    assert captured_error.count(warning_msg) == 2


def test_app_config_keys(config_file):
    app = MinimalApp()
    app.config(config_file)
    keys = app.config_keys()
    assert isinstance(keys, set)

    # verify that various expected keys are present
    assert "replayer.frame_rate" in keys
    assert "replayer.basename" in keys
    assert "source" in keys
    assert "visualizer" in keys
    assert "visualizer.in_tensor_names" in keys

    # other non-existent keys are not
    assert "abcdefg" not in keys


class MyPingApp(Application):
    def compose(self):
        # Define the tx and rx operators, allowing tx to execute 10 times
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        rx = PingRxOp(self, name="rx")

        # Define the workflow:  tx -> rx
        self.add_flow(tx, rx)


def test_app_log_function(capfd):
    """
    The following debug log messages are expected to be printed:

        Executing PyApplication::run()... (log_func_ptr=0x7ffff37dc660)
        Executing Application::run()... (log_func_ptr=0x7ffff37dc660)

    The addresses (log_func_ptr=0x<address>) should be the same for both log messages.
    """

    env_var_settings = {
        ("HOLOSCAN_LOG_LEVEL", "DEBUG"),
    }
    with env_var_context(env_var_settings):
        # Application class's constructor reads the environment variable HOLOSCAN_LOG_LEVEL so
        # wrap the app in the context manager to ensure the environment variables are set
        app = MyPingApp()
        app.run()

    captured = capfd.readouterr()
    # Extract text (log_func_ptr=0x<address>) from the log message and check if the addresses are
    # all same.
    import re

    addresses = re.findall(r"log_func_ptr=0x[0-9a-fA-F]+", captured.err)
    assert len(set(addresses)) == 1
