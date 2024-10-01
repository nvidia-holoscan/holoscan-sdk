"""
SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator


class BrokenComputeOp(Operator):
    def compute(self, op_input, op_output, context):
        # intentionally cause a ZeroDivisionError to test exception_handling
        1 / 0  # noqa: B018


class BrokenInitializeOp(Operator):
    def initialize(self):
        # intentionally cause a ZeroDivisionError to test exception_handling
        1 / 0  # noqa: B018


class BrokenStartOp(Operator):
    def start(self):
        # intentionally cause a ZeroDivisionError to test exception_handling
        1 / 0  # noqa: B018


class BrokenStopOp(Operator):
    def stop(self):
        # intentionally cause a ZeroDivisionError to test exception_handling
        1 / 0  # noqa: B018


class BrokenAllOp(Operator):
    def initialize(self):
        # intentionally cause a ZeroDivisionError to test exception_handling
        1 / 0  # noqa: B018

    def start(self):
        # intentionally cause a ZeroDivisionError to test exception_handling
        1 / 0  # noqa: B018

    def compute(self, op_input, op_output, context):
        # intentionally cause a ZeroDivisionError to test exception_handling
        1 / 0  # noqa: B018

    def stop(self):
        # intentionally cause a ZeroDivisionError to test exception_handling
        1 / 0  # noqa: B018


class BadComputeOperatorApp(Application):
    def compose(self):
        mx = BrokenComputeOp(self, CountCondition(self, 1), name="mx")
        self.add_operator(mx)


class BadInitializeOperatorApp(Application):
    def compose(self):
        mx = BrokenInitializeOp(self, CountCondition(self, 1), name="mx")
        self.add_operator(mx)


class BadStartOperatorApp(Application):
    def compose(self):
        mx = BrokenStartOp(self, CountCondition(self, 1), name="mx")
        self.add_operator(mx)


class BadStopOperatorApp(Application):
    def compose(self):
        mx = BrokenStopOp(self, CountCondition(self, 1), name="mx")
        self.add_operator(mx)


class BadAllOperatorApp(Application):
    def compose(self):
        mx = BrokenAllOp(self, CountCondition(self, 1), name="mx")
        self.add_operator(mx)


@pytest.mark.parametrize("method", ["compute", "initialize", "start", "stop", "all"])
def test_exception_handling(method):
    if method == "compute":
        app = BadComputeOperatorApp()
    elif method == "initialize":
        app = BadInitializeOperatorApp()
    elif method == "start":
        app = BadStartOperatorApp()
    elif method == "stop":
        app = BadStopOperatorApp()
    elif method == "all":
        app = BadAllOperatorApp()
    else:
        raise ValueError("invalid method name")

    with pytest.raises(ZeroDivisionError):
        app.run()
