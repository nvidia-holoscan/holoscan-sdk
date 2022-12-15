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

from test_operators_native import MinimalOp, PingMiddleOp, PingRxOp, PingTxOp

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.logger import load_env_log_level


class MinimalApp(Application):
    def compose(self):
        mx = MinimalOp(self, CountCondition(self, 10), name="mx")
        self.add_operator(mx)


def test_minimal_app(ping_config_file):
    load_env_log_level()
    app = MinimalApp()
    app.config(ping_config_file)
    app.run()


class MyPingApp(Application):
    def compose(self):
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        mx = PingMiddleOp(self, self.from_config("mx"), name="mx")
        rx = PingRxOp(self, name="rx")
        self.add_flow(tx, mx, {("out1", "in1"), ("out2", "in2")})
        self.add_flow(mx, rx, {("out1", "receivers"), ("out2", "receivers")})


def test_my_ping_app(ping_config_file):
    load_env_log_level()
    app = MyPingApp()
    app.config(ping_config_file)
    app.run()
