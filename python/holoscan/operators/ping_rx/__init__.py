"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.core import Operator, OperatorSpec


class PingRxOp(Operator):
    """Simple receiver operator.

    This is an example of a native operator with one input port.
    On each tick, it receives an integer from the "in" port.

    **==Named Inputs==**

        in : any
            A received value.
    """

    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        print(f"Rx message value: {value}")


__all__ = ["PingRxOp"]
