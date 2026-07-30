"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator


class MyOp(Operator):
    def compute(self, op_input, op_output, context):
        print(f"I am {self.name}")


class AppWithUnnamedOps(Application):
    def compose(self):
        self.add_operator(MyOp(self, CountCondition(self, 1)))
        self.add_operator(MyOp(self, CountCondition(self, 1)))


def test_operator_unnamed(capfd):
    AppWithUnnamedOps().run()

    # assert no error logged
    captured = capfd.readouterr()
    assert captured.err.count("[error]") == 0


class AppWithDuplicateNamedOps(Application):
    def compose(self):
        self.add_operator(MyOp(self, CountCondition(self, 1), name="myop"))
        self.add_operator(MyOp(self, CountCondition(self, 1), name="myop"))


def test_operator_duplicate_name():
    with pytest.raises(RuntimeError):
        AppWithDuplicateNamedOps().run()
