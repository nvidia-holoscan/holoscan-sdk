"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import logging
import re
import warnings

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec

try:
    import torch
except ImportError:
    raise ImportError("torch must be installed to run this example.") from None


def get_torch_device() -> torch.device:
    """Determine the best available torch device, checking CUDA compatibility.

    Returns torch.device("cuda") if CUDA is available and SM-compatible,
    otherwise torch.device("cpu").
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        torch.cuda.init()
        for w in caught_warnings:
            if re.search(r"CUDA capability sm_[^ ]+ is not compatible", str(w.message)):
                logging.warning(str(w.message))
                return torch.device("cpu")

    return torch.device("cuda")


# Check CUDA availability and compatibility at import time
TORCH_DEVICE = get_torch_device()
print(f"Torch using device: {TORCH_DEVICE}")


class SourceOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = TORCH_DEVICE
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(1234)
        self.static_out = torch.randn(
            (256, 256), device=self.device, dtype=torch.float32, generator=self.rng
        )

    def setup(self, spec: OperatorSpec):
        spec.output("static_out")
        spec.output("variable_out")

    def compute(self, op_input, op_output, context):
        variable = torch.randn(
            (256, 256), device=self.device, dtype=torch.float32, generator=self.rng
        )
        op_output.emit(variable, "variable_out")
        op_output.emit(self.static_out, "static_out")


class MatMulOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in_static")
        spec.input("in_variable")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        a = op_input.receive("in_static")
        b = op_input.receive("in_variable")
        out = torch.matmul(a, b)
        op_output.emit(out, "out")


class SinkOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        t = op_input.receive("in")
        # Keep output small (avoid printing full matrices).
        print(f"result: shape={tuple(t.shape)} device={t.device} mean={t.mean().item():.6f}")


class MatMulApp(Application):
    """Demonstrates using PyTorch tensors with Holoscan Python operators, performing matrix
    multiplication on GPU, or CPU, and reporting summary statistics. This example shows how
    PyTorch native tensors can be emitted and received directly in a Holoscan workflow."""

    def compose(self):
        src = SourceOp(self, CountCondition(self, 100), name="src_op")
        matmul = MatMulOp(self, name="matmul_op")
        sink = SinkOp(self, name="sink_op")

        self.add_flow(src, matmul, {("static_out", "in_static"), ("variable_out", "in_variable")})
        self.add_flow(matmul, sink)


def main():
    app = MatMulApp()
    app.run()


if __name__ == "__main__":
    main()
