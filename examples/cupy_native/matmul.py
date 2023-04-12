# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import load_env_log_level

try:
    import cupy as cp
except ImportError:
    raise ImportError("cupy must be installed to run this example.")


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
    def compose(self):
        src = SourceOp(self, CountCondition(self, 1000), name="src_op")
        matmul = MatMulOp(self, name="matmul_op")
        sink = SinkOp(self, name="sink_op")

        # Connect the operators into the workflow:  src -> matmul -> sink
        self.add_flow(src, matmul, {("static_out", "in_static"), ("variable_out", "in_variable")})
        self.add_flow(matmul, sink)


if __name__ == "__main__":
    load_env_log_level()
    app = MatMulApp()
    app.run()
