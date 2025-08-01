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
from holoscan.resources import UnboundedAllocator

TENSOR_SIZE = 16


class MakeTensorsOp(Operator):
    """Example make ActivationSpec for dynamically selecting models in run-time"""

    def __init__(self, fragment, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("tensors")
        spec.output("models")

    def compute(self, op_input, op_output, context):
        tensormap = {
            "first_preprocessed": cp.array([self.index] * TENSOR_SIZE, dtype=cp.float32),
            "second_preprocessed": cp.array([self.index] * TENSOR_SIZE, dtype=cp.float32),
            "third_preprocessed": cp.array([self.index] * TENSOR_SIZE, dtype=cp.float32),
        }

        act_models = [
            InferenceOp.ActivationSpec("first", 0),
            InferenceOp.ActivationSpec("second", 0),
            InferenceOp.ActivationSpec("third", 0),
        ]

        if self.index == 0:
            # first model
            act_models[0].set_active()
        elif self.index == 1:
            # second model
            act_models[1].set_active()
        elif self.index == 2:
            # third model
            act_models[2].set_active()
        elif self.index == 3:
            # first + second models
            act_models[0].set_active()
            act_models[1].set_active()
        elif self.index == 4:
            # second + third models
            act_models[1].set_active()
            act_models[2].set_active()
        elif self.index == 5:
            # third + first models
            act_models[2].set_active()
            act_models[0].set_active()
        else:
            act_models[0].set_active()
            act_models[1].set_active()
            act_models[2].set_active()

        op_output.emit(tensormap, "tensors")
        op_output.emit(act_models, "models")

        self.index += 1


class PrintResultInferOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("models")
        spec.input("models_result")

    def compute(self, op_input, op_output, context):
        selected_models_act_spec = op_input.receive("models")
        models_result = op_input.receive("models_result")

        selected_models = [act.model() for act in selected_models_act_spec if act.is_active()]
        print(f"Model {', '.join(selected_models)} was selected to infer")
        first_ret = cp.asarray(models_result.get("first_output"))[0]
        second_ret = cp.asarray(models_result.get("second_output"))[0]
        third_ret = cp.asarray(models_result.get("third_output"))[0]

        first_ret = [str(int(x)) for x in first_ret.tolist()]
        second_ret = [str(int(x)) for x in second_ret.tolist()]
        third_ret = [str(int(x)) for x in third_ret.tolist()]
        print("  First model result:", ", ".join(first_ret))
        print("  Second model result:", ", ".join(second_ret))
        print("  Third model result:", ", ".join(third_ret))


class ActivationMapDemoApp(Application):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(os.path.dirname(__file__), "../models")
        self.model_path_map = {
            "first": os.path.join(self.model_path, "dummy_addition_model_1.onnx"),
            "second": os.path.join(self.model_path, "dummy_addition_model_2.onnx"),
            "third": os.path.join(self.model_path, "dummy_addition_model_3.onnx"),
        }

    def compose(self):
        make_tensors_op = MakeTensorsOp(self, CountCondition(self, 7), name="make_tensors")
        infer_op = InferenceOp(
            self,
            name="infer",
            model_path_map=self.model_path_map,
            allocator=UnboundedAllocator(self, name="allocator"),
            **self.kwargs("infers"),
        )
        print_result_op = PrintResultInferOp(self, name="print_op")

        self.add_flow(
            make_tensors_op,
            infer_op,
            {("models", "model_activation_specs"), ("tensors", "receivers")},
        )
        self.add_flow(
            make_tensors_op,
            print_result_op,
            {
                ("models", "models"),
            },
        )
        self.add_flow(infer_op, print_result_op, {("transmitter", "models_result")})


def main(config_file):
    app = ActivationMapDemoApp()
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Activation map demo application.")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help=("Set the configuration file"),
    )

    args = parser.parse_args()

    config_file = args.config or os.path.join(os.path.dirname(__file__), "activation_map.yaml")
    main(config_file=config_file)
