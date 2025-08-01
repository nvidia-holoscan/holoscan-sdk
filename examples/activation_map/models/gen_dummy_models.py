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
import warnings

import numpy as np
import onnx
from onnx import TensorProto, helper


def main(output_dir: str = None, n: int = 3, tensor_size: int = 16):
    for s in range(1, n + 1):
        model_name = f"dummy_addition_model_{s}"
        shape = [1, tensor_size]
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)

        # create const tensor
        scalar_np = np.zeros(shape, dtype=np.float32) + s
        scalar_tensor = helper.make_tensor(
            name="scalar", data_type=TensorProto.FLOAT, dims=shape, vals=scalar_np
        )

        # create node for scalar tensor
        scalar_node = helper.make_node(
            "Constant", inputs=[], outputs=["scalar"], value=scalar_tensor
        )

        # create addition node
        add_node = helper.make_node("Add", inputs=["input", "scalar"], outputs=["output"])

        # create graph and save model
        graph = helper.make_graph(
            [scalar_node, add_node],
            model_name,
            [input_tensor],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, tensor_size])],
        )
        model = helper.make_model(graph)
        model.opset_import[0].version = 13
        save_path = os.path.join(output_dir, f"{model_name}.onnx")
        onnx.save_model(model, save_path)
        # Test model with onnxruntime
        try:
            import onnxruntime as ort  # noqa: PLC0415

            sess = ort.InferenceSession(save_path)
            input_ = np.asarray(
                [
                    [
                        0.0,
                    ]
                    * tensor_size
                ],
                dtype=np.float32,
            )
            output = sess.run(None, {"input": input_})[0]
            np.testing.assert_equal(scalar_np, output)
        except ModuleNotFoundError:
            warnings.warn("Onnxruntime not found, generating without test!", stacklevel=2)


if __name__ == "__main__":
    output_dir = os.path.dirname(__file__)
    main(output_dir=output_dir)
