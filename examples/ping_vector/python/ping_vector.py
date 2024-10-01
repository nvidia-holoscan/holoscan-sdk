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

from holoscan.conditions import CountCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec


class PingTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        value = self.index
        self.index += 1

        output = []
        for _ in range(0, 5):
            output.append(value)
            value += 1

        op_output.emit(output, "out")


class PingMxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out1")
        spec.output("out2")
        spec.output("out3")
        spec.param("multiplier", 2)

    def compute(self, op_input, op_output, context):
        values1 = op_input.receive("in")
        print(f"Middle message received (count: {self.count})")
        self.count += 1

        values2 = []
        values3 = []
        for i in range(0, len(values1)):
            print(f"Middle message value: {values1[i]}")
            values2.append(values1[i] * self.multiplier)
            values3.append(values1[i] * self.multiplier * self.multiplier)

        op_output.emit(values1, "out1")
        op_output.emit(values2, "out2")
        op_output.emit(values3, "out3")


class PingRxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.input("dup_in")
        # # Since Holoscan SDK v2.3, users can define a multi-receiver input port using
        # # 'spec.input()' with 'size=IOSpec.ANY_SIZE'.
        # # The old way is to use 'spec.param()' with 'kind="receivers"'.
        # spec.param("receivers", kind="receivers")
        spec.input("receivers", size=IOSpec.ANY_SIZE)

    def compute(self, op_input, op_output, context):
        receiver_vector = op_input.receive("receivers")
        input_vector = op_input.receive("in")
        dup_input_vector = op_input.receive("dup_in")
        print(
            f"Rx message received (count: {self.count}, input vector size: {len(input_vector)},"
            "duplicated input vector size: {len(dup_input_vector)}, receiver size: "
            "{len(receiver_vector)})"
        )
        self.count += 1

        for i in range(0, len(input_vector)):
            print(f"Rx message input value[{i}]: {input_vector[i]}")

        for i in range(0, len(dup_input_vector)):
            print(f"Rx message duplicated input value[{i}]: {dup_input_vector[i]}")

        for i in range(0, len(receiver_vector)):
            for j in range(0, len(receiver_vector[i])):
                print(f"Rx message receiver value[{i}][{j}]: {receiver_vector[i][j]}")


class MyPingApp(Application):
    def compose(self):
        # Define the tx, mx, rx operators, allowing the tx operator to execute 10 times
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        mx = PingMxOp(self, name="mx", multiplier=3)
        rx = PingRxOp(self, name="rx")

        # Define the workflow
        self.add_flow(tx, mx, {("out", "in")})
        self.add_flow(mx, rx, {("out1", "in"), ("out1", "dup_in")})
        self.add_flow(mx, rx, {("out2", "receivers"), ("out3", "receivers")})


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
