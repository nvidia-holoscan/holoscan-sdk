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

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import load_env_log_level


class MinimalOp(Operator):
    def __init__(self, *args, **kwargs):
        self.count = 1
        self.param_value = None
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.param("param_value", 500)

    def compute(self, op_input, op_output, context):
        self.count += 1


class MinimalApp(Application):
    def compose(self):
        mx = MinimalOp(self, CountCondition(self, 10), name="mx")
        self.add_operator(mx)


def test_minimal_app(ping_config_file):
    load_env_log_level()
    app = MinimalApp()
    app.config(ping_config_file)
    app.run()
