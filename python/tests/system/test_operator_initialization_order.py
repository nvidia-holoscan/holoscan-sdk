"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingRxOp, PingTxOp


class VerboseInitForwardOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def initialize(self):
        print(f"initialized {self.name}")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")
        op_output.emit(data, "out")


class VerboseInitRxOp(PingRxOp):
    def initialize(self):
        print(f"initialized {self.name}")


class VerboseInitTxOp(PingTxOp):
    def initialize(self):
        print(f"initialized {self.name}")


class InitOrderApp(Application):
    r"""App with the following forked operator layout

    A - B - C - D - E - F
         \           \
          G           H

    Initialization order is expected to be in topographic order
    (root -> branch). When there is a fork, the path taken first is determined
    by the order the operators were inserted into the graph.
    """

    def compose(self):
        op_a = VerboseInitTxOp(self, CountCondition(self, name="count", count=1), name="A")
        op_b = VerboseInitForwardOp(self, name="B")
        op_c = VerboseInitForwardOp(self, name="C")
        op_d = VerboseInitForwardOp(self, name="D")
        op_e = VerboseInitForwardOp(self, name="E")
        op_f = VerboseInitRxOp(self, name="F")
        op_h = VerboseInitRxOp(self, name="H")
        op_g = VerboseInitRxOp(self, name="G")

        # add D in non-topographic order
        self.add_operator(op_d)

        self.add_flow(op_a, op_b)
        # for fork at B add G before C
        self.add_flow(op_b, op_e)
        self.add_flow(op_b, op_g)
        self.add_flow(op_c, op_d)
        self.add_flow(op_d, op_e)
        # for fork at E add F before H
        self.add_flow(op_e, op_f)
        self.add_flow(op_e, op_h)


def test_operator_initialization_order(capfd):
    app = InitOrderApp()
    app.run()

    captured = capfd.readouterr()
    loc_a = captured.err.find("initialized A")
    loc_b = captured.err.find("initialized B")
    loc_c = captured.err.find("initialized C")
    loc_d = captured.err.find("initialized D")
    loc_e = captured.err.find("initialized E")
    loc_f = captured.err.find("initialized F")
    loc_g = captured.err.find("initialized G")
    loc_h = captured.err.find("initialized H")

    # in topographic order and at the forks:
    #   - after B expect G was initialized before C
    #   - after E expect F was initialized before H
    assert sorted([loc_a, loc_b, loc_c, loc_d, loc_e, loc_f, loc_g, loc_h]) == [
        loc_a,
        loc_b,
        loc_g,
        loc_c,
        loc_d,
        loc_e,
        loc_f,
        loc_h,
    ]
