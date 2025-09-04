"""
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import platform
from argparse import ArgumentParser

import cupy as cp

from holoscan.conditions import CountCondition, CudaStreamCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingTensorRxOp
from holoscan.resources import CudaGreenContext, CudaGreenContextPool, CudaStreamPool


class CuPySourceOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        width=3840,
        height=2160,
        use_default_stream=False,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.use_default_stream = use_default_stream
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        if self.use_default_stream:
            cp_tensor = cp.linspace(0, 10, self.height * self.width, dtype=cp.float32)
            cp_tensor = cp_tensor.reshape((self.height, self.width))

            cp_tensor = cp.exp(cp_tensor)
        else:
            # create a stream managed by Holoscan
            stream = context.allocate_cuda_stream("my_stream")

            # Create a CuPy array and launch some kernel on it on the external stream
            with cp.cuda.ExternalStream(stream):
                cp_tensor = cp.linspace(0, 10, self.height * self.width, dtype=cp.float32)
                cp_tensor = cp_tensor.reshape((self.height, self.width))

                cp_tensor = cp.exp(cp_tensor)

            # configure to emit a stream ID component when emitting from the "out" port
            op_output.set_cuda_stream(stream, "out")

        # Emit the CuPy array (the stream ID will be emitted as well)
        op_output.emit(cp_tensor, "out")


class CuPyProcessOp(Operator):
    def __init__(self, fragment, *args, use_default_stream=False, **kwargs):
        self.use_default_stream = use_default_stream
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.input("in2")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        tensor1 = op_input.receive("in1")
        tensor2 = op_input.receive("in2")

        if self.use_default_stream:
            # CuPySourceOp emits CuPy array objects, but it is good practice to call
            # cp.asarray to ensure any array-like object is converted to a CuPy array.
            in1 = cp.asarray(tensor1)
            in2 = cp.asarray(tensor2)
            # elementwise multiplication
            prod1 = in1 * in2
            # sum along the last axis
            partial_sum = prod1.sum(axis=-1)
        else:
            # stream1 and stream2 will both be using the operator's internal stream
            # Any stream on in1 or in2 will have been synchronized to the same internal stream.
            stream = op_input.receive_cuda_stream("in1")
            stream2 = op_input.receive_cuda_stream("in2")
            assert stream2 == stream

            # Create a CuPy array on the stream
            with cp.cuda.ExternalStream(stream):
                # CuPySourceOp emits CuPy array objects, but it is good practice to call
                # cp.asarray to ensure any array-like object is converted to a CuPy array.
                in1 = cp.asarray(tensor1)
                in2 = cp.asarray(tensor2)

                # note that while within the cp.cuda.ExternalStream context
                #   in1.__cuda_array_interface__["stream"] == stream
                #   in2.__cuda_array_interface__["stream"] == stream

                # elementwise multiplication
                prod1 = in1 * in2
                # sum along the last axis
                partial_sum = prod1.sum(axis=-1)

        # Send the CuPy array to the output port
        # Use emitter_name="holoscan::Tensor" to output as a C++ holoscan::Tensor instead of a
        # default CuPy array object so that the output is compatible with wrapped C++ operators
        # like PingTensorRxOp.
        op_output.emit(partial_sum, "out", emitter_name="holoscan::Tensor")


class CuPyExampleApp(Application):
    def __init__(
        self, *args, count=10, use_default_stream=False, use_green_context=False, **kwargs
    ):
        self.count = count
        self.use_default_stream = use_default_stream
        self.use_green_context = use_green_context
        if self.use_green_context and self.use_default_stream:
            raise ValueError("Green context is not supported with default stream")

        super().__init__(*args, **kwargs)

    def compose(self):
        # Default stream is not supported with green context
        if self.use_green_context and not self.use_default_stream:
            arch = platform.machine().lower()
            if arch in ["x86_64", "amd64"]:
                partitions = [8, 8]
            elif arch in ["aarch64", "arm64"]:
                partitions = [4, 4]
            else:
                raise ValueError(f"Unsupported platform architecture: {arch}")
            cuda_green_context_pool = CudaGreenContextPool(
                self,
                dev_id=0,
                flags=0,
                num_partitions=2,
                sms_per_partition=partitions,
                name="cuda_green_context_pool",
            )
            cuda_green_context = CudaGreenContext(
                self,
                cuda_green_context_pool=cuda_green_context_pool,
                index=1,
                name="cuda_green_context",
            )

        else:
            cuda_green_context = None
            cuda_green_context_pool = None

        if self.use_default_stream:
            source1 = CuPySourceOp(
                self,
                CountCondition(self, self.count, name="source1_count"),
                use_default_stream=True,
                name="source1",
            )
            source2 = CuPySourceOp(
                self,
                CountCondition(self, self.count, name="source2_count"),
                use_default_stream=True,
                name="source2",
            )
            proc_op = CuPyProcessOp(
                self,
                use_default_stream=True,
                name="combine_and_sum",
            )
            rx = PingTensorRxOp(
                self,
                name="rx",
            )
        else:
            # Creat a cuda stream pool with green context if enabled
            stream_pool = CudaStreamPool(
                self,
                name="stream_pool",
                dev_id=0,
                stream_flags=0,
                stream_priority=0,
                reserved_size=1,
                max_size=5,
                cuda_green_context=cuda_green_context,
            )
            source1 = CuPySourceOp(
                self,
                stream_pool,
                CountCondition(self, self.count, name="source1_count"),
                use_default_stream=False,
                name="source1",
            )
            source2 = CuPySourceOp(
                self,
                stream_pool,
                CountCondition(self, self.count, name="source2_count"),
                use_default_stream=False,
                name="source2",
            )
            proc_op = CuPyProcessOp(
                self,
                stream_pool,
                use_default_stream=False,
                name="combine_and_sum",
            )
            stream_cond = CudaStreamCondition(self, receiver="in", name="stream_sync")
            rx = PingTensorRxOp(
                self,
                stream_cond,
                stream_pool,
                name="rx",
            )
        self.add_flow(source1, proc_op, {("out", "in1")})
        self.add_flow(source2, proc_op, {("out", "in2")})
        self.add_flow(proc_op, rx)


if __name__ == "__main__":
    parser = ArgumentParser(description="Operator stream handling example")
    parser.add_argument(
        "-d",
        "--default_stream",
        action="store_true",
        help=(
            "Sets the application to disable dedicated operator streams and just use the default "
            "stream for all kernels."
        ),
    )
    parser.add_argument(
        "-g",
        "--green_context",
        action="store_true",
        help=("Sets the application to use green context when creating cuda stream pool."),
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=10,
        help="The number of messages to transmit.",
    )
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be a positive integer")
    if args.default_stream and args.green_context:
        parser.error("--default_stream and --green_context cannot be used together")

    app = CuPyExampleApp(
        count=args.count,
        use_default_stream=args.default_stream,
        use_green_context=args.green_context,
    )
    app.run()
