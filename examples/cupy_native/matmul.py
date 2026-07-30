"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

try:
    import cupy as cp
except ImportError:
    raise ImportError("cupy must be installed to run this example.") from None


class SourceOp(Operator):
    def __init__(self, *args, **kwargs):
        self.rng = cp.random.default_rng()
        self.static_out = self.rng.standard_normal((1000, 1000), dtype=cp.float32)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("static_out")
        spec.output("variable_out")

    def compute(self, op_input, op_output, context):
        op_output.emit(self.rng.standard_normal((1000, 1000), dtype=cp.float32), "variable_out")
        op_output.emit(self.static_out, "static_out")


class MatMulOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in_static")
        spec.input("in_variable")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        mat_static = op_input.receive("in_static")
        mat_dynamic = op_input.receive("in_variable")
        op_output.emit(cp.matmul(mat_static, mat_dynamic), "out")


class SinkOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        sig = op_input.receive("in")
        print(sig)


class MatMulApp(Application):
    """Demonstrates using CuPy tensors with Holoscan Python operators, performing matrix
    multiplication on GPU, and printing output tensors to the console. This example shows how
    CuPy native tensors can be emitted and received directly in a Holoscan workflow."""

    def compose(self):
        src = SourceOp(self, CountCondition(self, 1000), name="src_op")
        matmul = MatMulOp(self, name="matmul_op")
        sink = SinkOp(self, name="sink_op")

        # Connect the operators into the workflow:  src -> matmul -> sink
        self.add_flow(src, matmul, {("static_out", "in_static"), ("variable_out", "in_variable")})
        self.add_flow(matmul, sink)


def main():
    app = MatMulApp()
    app.run()


if __name__ == "__main__":
    main()
