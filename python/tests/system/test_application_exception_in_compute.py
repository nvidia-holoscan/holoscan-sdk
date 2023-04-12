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
from holoscan.core import Application, Operator
from holoscan.logger import load_env_log_level


class BrokenOp(Operator):
    def compute(self, op_input, op_output, context):
        # intentionally cause a ZeroDivisionError to test exception_handling
        1 / 0


class BadOperatorApp(Application):
    def compose(self):
        mx = BrokenOp(self, CountCondition(self, 1), name="mx")
        self.add_operator(mx)


def test_exception_handling(capfd):
    load_env_log_level()
    app = BadOperatorApp()
    app.run()

    # assert that the exception was logged
    captured = capfd.readouterr()
    assert "ZeroDivisionError: division by zero" in captured.err
    assert captured.err.count("Traceback") == 1
