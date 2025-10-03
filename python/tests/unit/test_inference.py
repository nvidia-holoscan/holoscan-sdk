"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from argparse import ArgumentParser

import cupy as cp

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import InferenceOp
from holoscan.resources import (
    CudaGreenContext,
    CudaGreenContextPool,
    CudaStreamPool,
    UnboundedAllocator,
)


class TensorGeneratorOp(Operator):
    """Example make ActivationSpec for dynamically selecting models in run-time"""

    def __init__(self, fragment, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        # Generate a tensor where each element is its index, starting from 0
        tensor_data = cp.arange(1 * 256 * 256, dtype=cp.float32).reshape(1, 256, 256)
        tensormap = {
            "tensor": tensor_data,
        }
        op_output.emit(tensormap, "output")


class ResultCheckerOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("input")

        if isinstance(data, dict):
            data = next(iter(data.values()))
            array = cp.asarray(data)
            array = array.reshape(-1)
            for i in range(array.shape[0]):
                element = array[i]
                assert element == float(i), f"data validation failed at index {i}"
            print("Inference result verified")


class InferenceOpTestApp(Application):
    def __init__(self, green_context):
        super().__init__()
        self.green_context = green_context
        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
        )
        self.model_path_map = {
            "first": os.path.join(model_path, "identity_model.pt"),
        }

    def compose(self):
        allocator = UnboundedAllocator(self, name="allocator")

        tensor_generator_op = TensorGeneratorOp(
            self,
            CountCondition(self, 2),
            name="tensor_generator",
            allocator=allocator,
        )

        if self.green_context:
            partitions = [4, 4]
            cuda_green_context_pool = CudaGreenContextPool(
                self,
                dev_id=0,
                flags=0,
                num_partitions=len(partitions),
                sms_per_partition=partitions,
                name="cuda_green_context_pool",
            )
            cuda_green_context = CudaGreenContext(
                self,
                cuda_green_context_pool=cuda_green_context_pool,
                index=0,
                name="cuda_green_context",
            )
        else:
            cuda_green_context = None

        cuda_stream_pool = CudaStreamPool(
            self,
            name="infer_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
            cuda_green_context=cuda_green_context,
        )

        pre_processor_map = {
            "first": ["tensor"],
        }
        inference_map = {
            "first": ["tensor"],
        }

        in_tensor_names = ["tensor"]
        out_tensor_names = ["tensor"]
        parallel_inference = True
        infer_on_cpu = False
        enable_fp16 = False
        enable_cuda_graphs = True
        input_on_cuda = True
        output_on_cuda = True
        transmit_on_cuda = True

        infer_op = InferenceOp(
            self,
            name="inference",
            backend="torch",
            allocator=allocator,
            model_path_map=self.model_path_map,
            cuda_stream_pool=cuda_stream_pool,
            pre_processor_map=pre_processor_map,
            inference_map=inference_map,
            in_tensor_names=in_tensor_names,
            out_tensor_names=out_tensor_names,
            parallel_inference=parallel_inference,
            infer_on_cpu=infer_on_cpu,
            enable_fp16=enable_fp16,
            enable_cuda_graphs=enable_cuda_graphs,
            input_on_cuda=input_on_cuda,
            output_on_cuda=output_on_cuda,
            transmit_on_cuda=transmit_on_cuda,
            **self.kwargs("inference"),
        )
        result_checker_op = ResultCheckerOp(self, name="result_checker")

        self.add_flow(
            tensor_generator_op,
            infer_op,
            {("output", "receivers")},
        )
        self.add_flow(infer_op, result_checker_op, {("transmitter", "input")})


def test_inference_torch(green_context: bool = False):
    app = InferenceOpTestApp(green_context)
    app.run()


if __name__ == "__main__":
    flag_green_context = False
    parser = ArgumentParser(description="Inference op test application.")
    parser.add_argument(
        "-g",
        "--green_context",
        action="store_true",
        help="Use green context",
    )

    args = parser.parse_args()
    if args.green_context:
        flag_green_context = True

    test_inference_torch(green_context=flag_green_context)
