"""
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from holoscan.core import Application, ConditionType, IOSpec, Operator
from holoscan.decorator import Input, Output, create_op
from holoscan.operators import PingRxOp, PingTxOp


class TestInput:
    def test_input(self):
        obj = Input("in")
        assert obj.name == "in"
        assert not obj.arg_map

    def test_input_string_arg_map(self):
        obj = Input("in", arg_map="x")
        assert obj.arg_map == "x"

    def test_input_dict_arg_map(self):
        obj = Input("in", arg_map={"image": "x", "mask": "m"})
        assert isinstance(obj.arg_map, dict)
        assert len(obj.arg_map.keys()) == 2

    def test_create_input_with_condition(self):
        app = Application()
        op = PingRxOp(fragment=app, name="rx")

        # initially PingRxOp has one input port named "in" with a default connector
        assert tuple(op.spec.inputs.keys()) == ("in",)
        assert op.spec.inputs["in"].connector_type == IOSpec.ConnectorType.DEFAULT

        # add a new port with NONE condition via Input.create_input
        obj = Input("in2", condition_type=ConditionType.NONE, condition_kwargs={})
        obj.create_input(op.spec)
        in_spec = op.spec.inputs["in2"]
        assert len(in_spec.conditions) == 1
        assert in_spec.conditions[0][0] == ConditionType.NONE

        # add a new port with MESSAGE_AVAILABLE condition via Input.create_input
        obj = Input(
            "in3", condition_type=ConditionType.MESSAGE_AVAILABLE, condition_kwargs=dict(min_size=2)
        )
        obj.create_input(op.spec)
        in_spec = op.spec.inputs["in3"]
        assert len(in_spec.conditions) == 1
        cond_type, cond_obj = in_spec.conditions[0]
        assert cond_type == ConditionType.MESSAGE_AVAILABLE
        assert len(cond_obj.args) == 1
        assert cond_obj.args[0].name == "min_size"
        assert "value: 2" in cond_obj.args[0].description

    def test_create_input_with_connector(self):
        app = Application()
        op = PingRxOp(fragment=app, name="rx")

        # initially PingRxOp has one input port named "in" with a default connector
        assert tuple(op.spec.inputs.keys()) == ("in",)
        assert op.spec.inputs["in"].connector_type == IOSpec.ConnectorType.DEFAULT

        obj = Input(
            "in2",
            connector_type=IOSpec.ConnectorType.DOUBLE_BUFFER,
            connector_kwargs=dict(capacity=2, policy=1),
        )
        obj.create_input(op.spec)
        in_spec = op.spec.inputs["in2"]
        assert in_spec.connector_type == IOSpec.ConnectorType.DOUBLE_BUFFER
        connector = in_spec.connector()
        assert len(connector.args) == 2
        assert connector.args[0].name == "capacity"
        assert "value: 2" in connector.args[0].description
        assert connector.args[1].name == "policy"
        assert "value: 1" in connector.args[1].description


class TestOutput:
    def test_output(self):
        obj = Output("out")
        assert obj.name == "out"
        assert obj.tensor_names == ()

    def test_output_tensor_names(self):
        tensor_names = ("x", "waveform")
        obj = Output("out", tensor_names=tensor_names)
        assert obj.tensor_names == tensor_names

    def test_create_output_with_condition(self):
        app = Application()
        op = PingTxOp(fragment=app, name="rx")

        # initially PingTxOp has one output port named "out" with a default connector
        assert tuple(op.spec.outputs.keys()) == ("out",)
        assert op.spec.outputs["out"].connector_type == IOSpec.ConnectorType.DEFAULT

        # add a new port with NONE condition via Output.create_output
        obj = Output("out2", condition_type=ConditionType.NONE, condition_kwargs={})
        obj.create_output(op.spec)
        in_spec = op.spec.outputs["out2"]
        assert len(in_spec.conditions) == 1
        assert in_spec.conditions[0][0] == ConditionType.NONE

        # add a new port with MESSAGE_AVAILABLE condition via Output.create_output
        obj = Output(
            "out3",
            condition_type=ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE,
            condition_kwargs=dict(min_size=2),
        )
        obj.create_output(op.spec)
        in_spec = op.spec.outputs["out3"]
        assert len(in_spec.conditions) == 1
        cond_type, cond_obj = in_spec.conditions[0]
        assert cond_type == ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE
        assert len(cond_obj.args) == 1
        assert cond_obj.args[0].name == "min_size"
        assert "value: 2" in cond_obj.args[0].description

    def test_create_output_with_connector(self):
        app = Application()
        op = PingTxOp(fragment=app, name="rx")

        # initially PingTxOp has one output port named "out" with a default connector
        assert tuple(op.spec.outputs.keys()) == ("out",)
        assert op.spec.outputs["out"].connector_type == IOSpec.ConnectorType.DEFAULT

        obj = Output(
            "out2",
            connector_type=IOSpec.ConnectorType.DOUBLE_BUFFER,
            connector_kwargs=dict(capacity=2, policy=1),
        )
        obj.create_output(op.spec)
        in_spec = op.spec.outputs["out2"]
        assert in_spec.connector_type == IOSpec.ConnectorType.DOUBLE_BUFFER
        connector = in_spec.connector()
        assert len(connector.args) == 2
        assert connector.args[0].name == "capacity"
        assert "value: 2" in connector.args[0].description
        assert connector.args[1].name == "policy"
        assert "value: 1" in connector.args[1].description


class TestCreateOp:
    def test_create_op_no_args_func(self):
        @create_op()
        def func_no_args():
            pass

        with pytest.raises(ValueError, match="fragment must be provided"):
            func_no_args()

        # pass fragment positionally
        app = Application()
        my_op = func_no_args(app)
        # __name__ will be a camelcase version of the function name
        assert my_op.__name__ == "FuncNoArgsOp"
        assert my_op.name == "func_no_args"

        # pass fragment and name via kwarg
        my_op2 = func_no_args(fragment=app, name="my-op")
        assert my_op2.__name__ == "FuncNoArgsOp"
        assert my_op2.name == "my-op"

    def test_create_op_input_not_specified(self):
        @create_op()
        def func_one_positional_arg(image):
            return image

        app = Application()
        with pytest.raises(ValueError, match="required argument, 'image', has not been specified"):
            func_one_positional_arg(app)

    @pytest.mark.parametrize(
        "inputs",
        [
            "image",
            ("image",),
            Input("image", arg_map="image"),
            Input("image", arg_map={"tensor": "image"}),
            (Input("image", arg_map={"tensor": "image"}),),
        ],
    )
    def test_create_op_inputs_specified(self, inputs):
        @create_op(inputs=inputs)
        def func_one_positional_arg(image):
            return image

        # pass fragment positionally
        app = Application()
        my_op = func_one_positional_arg(app)
        # __name__ will be a camelcase version of the function name
        assert my_op.__name__ == "FuncOnePositionalArgOp"
        assert my_op.name == "func_one_positional_arg"
        assert "image" in my_op.dynamic_kwargs
        assert "image" not in my_op.fixed_kwargs

    @pytest.mark.parametrize(
        "inputs, exception_type, expected_error_message",
        [
            ("tensor", KeyError, "Provided func does not have an arg or kwarg named 'tensor'"),
            (("tensor",), KeyError, "Provided func does not have an arg or kwarg named 'tensor'"),
            # image not in destinations for arg_map
            (
                Input("image", arg_map=()),
                ValueError,
                "required argument, 'image', has not been specified",
            ),
            (
                Input("image", arg_map={"tensor": "tensor"}),
                KeyError,
                "Provided func does not have an arg or kwarg named 'tensor'",
            ),
            (
                (Input("image", arg_map={"tensor": "tensor"}),),
                KeyError,
                "Provided func does not have an arg or kwarg named 'tensor'",
            ),
        ],
    )
    def test_create_op_invalid_input_name(self, inputs, exception_type, expected_error_message):
        @create_op(inputs=inputs)
        def func_one_positional_arg(image):
            return image

        app = Application()
        with pytest.raises(exception_type, match=expected_error_message):
            func_one_positional_arg(app)

    @pytest.mark.parametrize("input_as_tuple", [False, True])
    def test_create_op_inputs_specified_via_input_obj(self, input_as_tuple):
        port_name = "image_in"
        inputs = Input(port_name, {"image": "image"}, condition_type=ConditionType.NONE)
        if input_as_tuple:
            inputs = (inputs,)

        @create_op(inputs=inputs)
        def func_one_positional_arg(image):
            return image

        # pass fragment positionally
        app = Application()
        my_op = func_one_positional_arg(app)
        assert "image" in my_op.dynamic_kwargs
        assert "image" not in my_op.fixed_kwargs

        cond_type, cond_obj = my_op.spec.inputs[port_name].conditions[0]
        assert cond_obj is None
        assert cond_type == ConditionType.NONE

    def test_create_op_inputs_keyword_only_arg(self):
        @create_op(inputs="image", outputs="image")
        def func_one_positional_arg(image, x=5, *, y=7):
            return image

        # pass fragment positionally
        app = Application()
        my_op = func_one_positional_arg(app, y=12)
        assert "image" in my_op.dynamic_kwargs
        assert "x" in my_op.fixed_kwargs
        assert my_op.fixed_kwargs["x"] == 5
        assert "y" in my_op.fixed_kwargs
        assert my_op.fixed_kwargs["y"] == 12

    def test_create_op_generator_func(self):
        @create_op(inputs="image", outputs="image")
        def int_generator(image, *, count=10):
            yield from range(count)

        # pass fragment positionally
        app = Application()
        my_op = int_generator(app)
        assert my_op.fixed_kwargs["count"] == 10

        my_op = int_generator(app, count=1)
        assert my_op.fixed_kwargs["count"] == 1

    @pytest.mark.parametrize("explicit_inputs_and_outputs", [False, True])
    def test_create_op_from_class(self, explicit_inputs_and_outputs):
        if explicit_inputs_and_outputs:

            @create_op(inputs="image", outputs="out")
            class TensorGenerator:
                def __init__(self, start_index=5):
                    self.counter = start_index - 1

                def __call__(self, image, *, msg="hello"):
                    print(f"{msg}: {image.shape=}")
                    self.counter += 1
                    return self.counter
        else:

            @create_op
            class TensorGenerator:
                def __init__(self, start_index=5):
                    self.counter = start_index - 1

                def __call__(self, image, *, msg="hello"):
                    print(f"{msg}: {image.shape=}")
                    self.counter += 1
                    return self.counter

        # pass fragment positionally
        app = Application()
        start_index = 10
        my_op = TensorGenerator(start_index=start_index)(app, name="class_op")

        assert isinstance(my_op, Operator)
        # verify input port name
        inputs = my_op.spec.inputs
        assert len(inputs) == 1
        assert inputs["image"].name == "image"

        # verify output port name
        outputs = my_op.spec.outputs
        assert len(outputs) == 1
        if explicit_inputs_and_outputs:
            assert "out" in outputs
        else:
            assert "" in outputs

        # check internal state of the wrapped class
        assert my_op.class_obj.counter == start_index - 1
        assert my_op.class_obj.__class__.__name__ == "TensorGenerator"

        # check names
        assert my_op.name == "class_op"
        assert my_op.__name__ == "TensorGeneratorOp"

        # verify kwargs based on __call__ function signature
        assert len(my_op.dynamic_kwargs) == 1
        assert "image" in my_op.dynamic_kwargs
        assert len(my_op.fixed_kwargs) == 1
        assert "msg" in my_op.fixed_kwargs
