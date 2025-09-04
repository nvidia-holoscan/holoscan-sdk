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
from holoscan.core import Application, Fragment
from holoscan.operators import PingRxOp
from holoscan.operators.test_ops import DataTypeTxTestOp

# Define a dual fragment application designed to test emit of C++ types from
# a wrapped C++ operator (DataTypeTxTestOp) and received by a native Python
# Operator (PingRxOp).


class TxFragment(Fragment):
    """Transmit Fragment

    Fragment containing a wrapped C++ Operator (DataTypeTxTestOp) that emits a
    user-specified `data_type`. It is connected within-fragment to receiver
    `rx1` and between fragments to receiver `RxFragment.rx2`.
    """

    def __init__(self, *args, data_type="double", count=10, **kwargs):
        self.data_type = data_type
        self.count = count

        super().__init__(*args, **kwargs)

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        tx = DataTypeTxTestOp(
            self, CountCondition(self, count=self.count), data_type=self.data_type, name="tx"
        )
        rx1 = PingRxOp(self, name="rx1")

        self.add_flow(tx, rx1)


class RxFragment(Fragment):
    """Fragment containing a native Python receiver that prints the Python
    object received."""

    def compose(self):
        rx2 = PingRxOp(self, name="rx2")

        # Add the operator (rx2) to the fragment
        self.add_operator(rx2)


class DataTypeCppToPythonDistributedTestApp(Application):
    """Test conversion of C++ types to Python objects for both intra-fragment
    and inter-fragment connections."""

    def __init__(self, *args, data_type="tensor", count=10, **kwargs):
        self.data_type = data_type
        self.count = count

        super().__init__(*args, **kwargs)

    def compose(self):
        tx_fragment = TxFragment(
            self, name="tx_fragment", data_type=self.data_type, count=self.count
        )
        rx_fragment = RxFragment(self, name="rx_fragment")

        # Connect the two fragments (tx.out -> rx.in)
        # We can skip the "out" and "in" suffixes, as they are the default
        self.add_flow(tx_fragment, rx_fragment, {("tx", "rx2")})


@pytest.mark.parametrize(
    # as distributed tests take multiple seconds each, only test a subset of supported types here
    # See more comprehensive type testing in non-distributed cases in
    # test_cpp_types_to_from_python.py
    "data_type",
    [
        "bool",
        "double",
        "int8_t",
        "uint64_t",
        "std::string",
        "std::vector<std::string>",
        "std::vector<std::vector<std::complex<float>>>",
        "std::unordered_map<std::string, std::string>",
        "std::shared_ptr<std::unordered_map<std::string, std::string>>",
    ],
)
def test_distributed_ucx_cpp_to_python(data_type, capfd):
    # minimal app to test the fix for issue 5397595
    count = 5
    app = DataTypeCppToPythonDistributedTestApp(data_type=data_type, count=count)
    app.run()

    captured = capfd.readouterr()

    # make sure message about received values were logged
    expected_num_messages = 2 * count  # two receivers: rx1 and rx2
    assert captured.out.count("Rx message value:") == expected_num_messages

    # assert that no errors were logged
    # avoid catching the expected error message
    # : "error handling callback was invoked with status -25 (Connection reset by remote peer)"
    captured_error = captured.err.replace("error handling callback", "ucx handling callback")
    assert "error" not in captured_error
    assert "Exception occurred" not in captured_error
