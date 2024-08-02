"""
 SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import sys

import numpy as np

from holoscan.conditions import CountCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec
from holoscan.logger import LogLevel, set_log_level

# define a custom class to represent data used in the app

REF_COUNT_RECORD = []


class ValueData:
    """Example of a custom Python class"""

    def __init__(self, value):
        self.data = value

    def __repr__(self):
        return f"ValueData({self.data})"

    def __eq__(self, other):
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def __del__(self):
        print("destructor called:", self.data)


class PingTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        global REF_COUNT_RECORD
        REF_COUNT_RECORD.append(self.index)
        value = ValueData(self.index)
        print(f"# TX {self.index}(refcount: {sys.getrefcount(value)})")
        REF_COUNT_RECORD.append(sys.getrefcount(value))
        del value
        value = ValueData(self.index)
        print(f"# TX {self.index} after recreate (refcount: {sys.getrefcount(value)})")
        REF_COUNT_RECORD.append(sys.getrefcount(value))

        op_output.emit(value, "out")

        print(f"- TX {self.index} after emit (refcount: {sys.getrefcount(value)})")
        REF_COUNT_RECORD.append(sys.getrefcount(value))

        self.index += 1


class PingRxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("receivers", size=IOSpec.ANY_SIZE)

    def compute(self, op_input, op_output, context):
        REF_COUNT_RECORD.append(self.count)
        values = op_input.receive("receivers")
        print(
            f"# RX({self.name}) {self.count} after receive (refcount: {sys.getrefcount(values[0])})"
        )
        REF_COUNT_RECORD.append(sys.getrefcount(values[0]))

        print(
            "Rx message received "
            f"(count: {self.count}, size: {len(values)}, value: {values[0].data})"
        )
        value = values[0]
        print(
            f"# RX({self.name}) {self.count} after value=values[0] "
            f"(refcount: {sys.getrefcount(value)})"
        )
        REF_COUNT_RECORD.append(sys.getrefcount(value))
        del values
        print(
            f"# RX({self.name}) {self.count} after del values (refcount: {sys.getrefcount(value)})"
        )
        REF_COUNT_RECORD.append(sys.getrefcount(value))
        del value
        print(f"# RX({self.name}) {self.count} after del value")

        self.count += 1
        gc.collect()
        print(f"# RX({self.name}) {self.count} after gc.collect()")


class MyPingApp(Application):
    def compose(self):
        tx1 = PingTxOp(self, CountCondition(self, 3), name="tx")
        rx1 = PingRxOp(self, name="rx1", multiplier=2)
        rx2 = PingRxOp(self, name="rx2", multiplier=3)
        rx3 = PingRxOp(self, name="rx3", multiplier=4)

        # Define the workflow
        self.add_flow(tx1, rx1, {("out", "receivers")})
        self.add_flow(tx1, rx2, {("out", "receivers")})
        self.add_flow(tx1, rx3, {("out", "receivers")})


def test_receivers_pyobject_refcount():
    global REF_COUNT_RECORD
    REF_COUNT_RECORD = []
    app = MyPingApp()
    app.run()
    # Check the ref count record. The correct ref count record can be obtained by running
    # this script with the standalone Python interpreter.
    # (python3 test_receivers_pyobject_refcount.py)
    # Or, you can copy this script to examples/ping_simple/python/ping_simple.py and run
    # the ping_simple example with the debugger.

    np.testing.assert_array_equal(
        REF_COUNT_RECORD,
        [
            1, 2, 2, 3, 1, 3, 4, 3, 1, 3, 4, 3, 1, 2, 3, 2, 2, 2, 2, 3, 2, 3, 4, 3, 2, 3, 4, 3, 2,
            2, 3, 2, 3, 2, 2, 3, 3, 3, 4, 3, 3, 3, 4, 3, 3, 2, 3, 2,
        ]
    )  # fmt:skip


if __name__ == "__main__":
    REF_COUNT_RECORD = []
    set_log_level(LogLevel.TRACE)
    app = MyPingApp()
    app.run()
    print(REF_COUNT_RECORD)
