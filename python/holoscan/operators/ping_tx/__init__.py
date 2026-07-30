"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.core import Operator, OperatorSpec


class PingTxOp(Operator):
    """Simple transmitter operator.

    On each tick, it transmits an integer to the "out" port.

    **==Named Outputs==**

        out : int
            An index value that increments by one on each call to `compute`. The starting value is
            1.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        op_output.emit(self.index, "out")
        self.index += 1


__all__ = ["PingTxOp"]
