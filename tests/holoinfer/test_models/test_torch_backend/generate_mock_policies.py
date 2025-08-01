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

import torch
import yaml


class MockPolicy(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.forward_function = func

    def forward(self, *args):
        return self.forward_function(*args)


def simple_policy(input1):
    return input1 + 10


def dict_input_policy(inputs):
    return inputs["input1"] + inputs["input2"]


def list_input_policy(inputs):
    return inputs[0] + inputs[1]


def tuple_output_policy(inputs):
    return (inputs * 2, inputs * 3, inputs * 4)


def nested_list_policy(inputs):
    return inputs[0][0] + inputs[0][1]


def nested_dict_policy(inputs):
    return inputs["input1"]["input2"] + inputs["input3"]["input4"]


def nested_list_and_dict_policy(inputs):
    return inputs[0]["input1"] + inputs[1]["input1"] + inputs[1]["input2"]


def heterogeneous_io_policy(input1, input2, input3):
    return input1["input"] + input2[0][0] + input3, [input3, input2[1][0] + input3]


def map_from_torch_dtype(dtype):
    if dtype == torch.float32:
        return "kFloat32"
    elif dtype == torch.float64:
        return "kFloat64"
    elif dtype == torch.int32:
        return "kInt32"
    elif dtype == torch.int64:
        return "kInt64"
    elif dtype == torch.uint8:
        return "kUInt8"
    elif dtype == torch.int8:
        return "kInt8"
    elif dtype == torch.bool:
        return "kBool"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def generate_input_description(inputs):
    tensor_descriptions = {}

    def process_input(inputs, prefix=""):
        if isinstance(inputs, torch.Tensor):
            tensor_name = prefix if prefix else f"tensor_{len(tensor_descriptions)}"
            tensor_descriptions[tensor_name] = {
                "dim": " ".join(str(dim) for dim in inputs.shape),
                "dtype": map_from_torch_dtype(inputs.dtype),
            }
            return tensor_name
        elif isinstance(inputs, dict):
            return {
                k: process_input(v, f"{prefix}_{k}" if prefix else k) for k, v in inputs.items()
            }
        elif isinstance(inputs, list | tuple):
            return [
                process_input(item, f"{prefix}_{i}" if prefix else str(i))
                for i, item in enumerate(inputs)
            ]
        else:
            return str(inputs)

    simplified_structure = process_input(inputs)
    if isinstance(simplified_structure, tuple):
        simplified_structure = list(simplified_structure)
    return simplified_structure, tensor_descriptions


def generate_yaml(inputs, output):
    yaml_dict = {"inference": {}}
    simplified_structure, tensor_descriptions = generate_input_description(inputs)
    simplified_structure_output, tensor_descriptions_output = generate_input_description(output)
    yaml_dict["inference"]["input_nodes"] = tensor_descriptions
    yaml_dict["inference"]["output_nodes"] = tensor_descriptions_output
    yaml_dict["inference"]["input_format"] = simplified_structure
    yaml_dict["inference"]["output_format"] = simplified_structure_output
    return yaml_dict


def get_copyright_header():
    """Generate copyright header for YAML files."""
    return (
        "# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "#\n"
        '# Licensed under the Apache License, Version 2.0 (the "License");\n'
        "# you may not use this file except in compliance with the License.\n"
        "# You may obtain a copy of the License at\n"
        "#\n"
        "# http://www.apache.org/licenses/LICENSE-2.0\n"
        "#\n"
        "# Unless required by applicable law or agreed to in writing, software\n"
        '# distributed under the License is distributed on an "AS IS" BASIS,\n'
        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
        "# See the License for the specific language governing permissions and\n"
        "# limitations under the License.\n"
        "\n"
    )


def write_yaml_with_header(yaml_dict, filename):
    """Write YAML file with copyright header."""
    with open(filename, "w") as f:
        f.write(get_copyright_header())
        yaml.dump(yaml_dict, f, default_flow_style=False)


def export_policy(policy, inputs, output_path):
    policy.eval()
    output = policy.forward(*inputs)
    traced_policy = torch.jit.trace(policy, inputs, strict=False)
    traced_policy.save(output_path)
    return output


def build_and_export_policy(policy_fn, inputs, filename_stem):
    """Helper function to build, export, and save a policy with its YAML config."""
    policy_module = MockPolicy(policy_fn)

    # Handle both single inputs and multiple inputs (for heterogeneous case)
    policy_inputs = inputs if isinstance(inputs, tuple) else (inputs,)

    # Export the policy
    output = export_policy(policy_module, policy_inputs, f"{filename_stem}.pt")

    # Generate and write YAML
    yaml_dict = generate_yaml(policy_inputs, output)
    write_yaml_with_header(yaml_dict, f"{filename_stem}.yaml")


if __name__ == "__main__":
    # Define all policies with their inputs and filenames
    policies = [
        # (policy_function, inputs, filename_stem)
        (simple_policy, torch.tensor([1, 2, 3]), "simple_policy"),
        (
            dict_input_policy,
            {"input1": torch.tensor([1, 2, 3]), "input2": torch.tensor([4, 5, 6])},
            "dict_input_policy",
        ),
        (
            list_input_policy,
            [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
            "list_input_policy",
        ),
        (tuple_output_policy, torch.tensor([1, 2, 3]), "tuple_output_policy"),
        (
            nested_list_policy,
            [[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]],
            "nested_list_policy",
        ),
        (
            nested_dict_policy,
            {
                "input1": {"input2": torch.tensor([1, 2, 3])},
                "input3": {"input4": torch.tensor([4, 5, 6])},
            },
            "nested_dict_policy",
        ),
        (
            nested_list_and_dict_policy,
            [
                {"input1": torch.tensor([1, 2, 3])},
                {"input1": torch.tensor([1, 2, 3]), "input2": torch.tensor([4, 5, 6])},
            ],
            "nested_list_and_dict_policy",
        ),
        # Special case: heterogeneous policy with multiple separate inputs
        (
            heterogeneous_io_policy,
            (
                {"input": torch.tensor([1, 2, 3])},
                [[torch.tensor([4, 5, 6])], [torch.tensor([7, 8, 9])]],
                torch.tensor([10, 11, 12]),
            ),
            "heterogeneous_io_policy",
        ),
    ]

    # Build and export all policies
    for policy_fn, inputs, filename_stem in policies:
        build_and_export_policy(policy_fn, inputs, filename_stem)
