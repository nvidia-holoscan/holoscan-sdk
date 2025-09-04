"""
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingRxOp
from holoscan.operators.test_ops import DataTypeRxTestOp, DataTypeTxTestOp

# Define a single fragment application designed to test emit of C++ types from a wrapped C++
# operator (DataTypeTxTestOp) and receive by a native Python operator.


# Define the types to test (these are C++ types registered with the EmitterReceiver registry by
# default. The registration is done in `python/holoscan/core/io_context.cpp`).
item_types = [
    "bool",
    "float",
    "double",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "std::complex<float>",
    "std::complex<double>",
    "std::string",
]
vector_types = [f"std::vector<{T}>" for T in item_types]
vector_types += [f"std::vector<std::vector<{T}>>" for T in item_types]
map_types = ["std::unordered_map<std::string, std::string>"]
shared_ptr_types = [f"std::shared_ptr<{T}>" for T in vector_types + map_types + item_types]
all_types = item_types + vector_types + shared_ptr_types + ["std::nullptr_t"]


class DataTypeCppToPythonTestApp(Application):
    """Test conversion of C++ types to Python objects."""

    def __init__(self, *args, data_type="tensor", count=10, **kwargs):
        self.data_type = data_type
        self.count = count

        super().__init__(*args, **kwargs)

    def compose(self):
        tx = DataTypeTxTestOp(
            self, CountCondition(self, count=self.count), data_type=self.data_type, name="tx"
        )
        rx = PingRxOp(self, name="rx")

        self.add_flow(tx, rx)


@pytest.mark.parametrize("data_type", all_types)
def test_cpp_type_to_python(data_type, capfd):
    # minimal app to test the fix for issue 5397595
    count = 5
    app = DataTypeCppToPythonTestApp(data_type=data_type, count=count)
    app.run()

    captured = capfd.readouterr()

    assert captured.out.count("Rx message value:") == count

    # assert that no errors were logged
    assert "error" not in captured.err
    assert "Exception occurred" not in captured.err


# Define a single fragment application designed to test emit of C++ types from a native Python
# operator. The types are received by a wrapped C++ operator that prints the type name.


class EmitterTxTestOp(Operator):
    """Test operator that emits a fixed value of a specified `data_type`."""

    def __init__(self, *args, data_type="double", count=10, **kwargs):
        self.data_type = data_type
        self.count = count
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def get_data_item(self, dt):
        if dt == "bool":
            data = True
        elif dt in ["double", "float"]:
            data = 1.0
        elif dt in ["int8_t", "int16_t", "int32_t", "int64_t"]:
            data = -1
        elif dt in ["uint8_t", "uint16_t", "uint32_t", "uint64_t"]:
            data = 1
        elif dt == "std::string":
            data = "test"
        elif dt == "std::nullptr_t":
            data = None
        elif dt in ["std::complex<float>", "std::complex<double>"]:
            data = 1.0 + 1.0j
        elif dt in ["std::unordered_map<std::string, std::string>"]:
            data = {"key1": "value1", "key2": "value2"}
        else:
            raise ValueError(f"Unsupported data type: {dt}")
        return data

    def compute(self, op_input, op_output, context):
        dt = self.data_type

        # extract T from std::shared_ptr<T> as there is no ptr on the Python side
        if dt.startswith("std::shared_ptr<") and dt.endswith(">"):
            dt = dt[len("std::shared_ptr<") : -1]

        if dt.startswith("std::vector<") and dt.endswith(">"):
            inner_type = dt[len("std::vector<") : -1]
            # Check for nested vector: std::vector<std::vector<T>>
            if inner_type.startswith("std::vector<") and inner_type.endswith(">"):
                element_type = inner_type[len("std::vector<") : -1]
                # Create nested list: [[item, item, item], [item, item, item]]
                data = [[self.get_data_item(element_type) for _ in range(3)] for _ in range(2)]
            else:
                # Create simple list: [item, item, item]
                data = [self.get_data_item(inner_type) for _ in range(3)]
        else:
            data = self.get_data_item(dt)
        op_output.emit(data, emitter_name=self.data_type, name="out")


class DataTypePythonToCppTestApp(Application):
    """Test emission of C++ types from a native Python operator.

    The emitted C++ type is received by a wrapped C++ operator that prints it's typeinfo
    """

    def __init__(self, *args, data_type="tensor", count=10, **kwargs):
        self.data_type = data_type
        self.count = count

        super().__init__(*args, **kwargs)

    def compose(self):
        tx = EmitterTxTestOp(
            self, CountCondition(self, count=self.count), data_type=self.data_type, name="tx"
        )
        rx = DataTypeRxTestOp(self, name="rx")

        # Connect the two fragments (tx.out -> rx.in)
        # We can skip the "out" and "in" suffixes, as they are the default
        self.add_flow(tx, rx)


@pytest.mark.parametrize("data_type", all_types)
def test_python_to_cpp_type(data_type, capfd):
    count = 5
    app = DataTypePythonToCppTestApp(data_type=data_type, count=count)
    app.run()

    captured = capfd.readouterr()

    # make sure message about received values were logged
    assert captured.err.count("Received message of type:") == count

    # assert that no errors were logged
    assert "error" not in captured.err
    assert "Exception occurred" not in captured.err
