# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from holoscan.core import (
    Application,
    Arg,
    ArgContainerType,
    ArgElementType,
    ArgList,
    ArgType,
    Component,
    ComponentSpec,
    Condition,
    ConditionType,
    Config,
    Executor,
    Fragment,
    Graph,
    InputContext,
    IOSpec,
    Operator,
    OperatorSpec,
    OutputContext,
    Resource,
    _Fragment,
    py_object_to_arg,
)
from holoscan.executors import GXFExecutor
from holoscan.graphs import FlowGraph


class OpTx(Operator):
    def __init__(self, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("tensor")


class OpRx(Operator):
    def __init__(self, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.input("in2")


def get_tx_and_rx_ops(_fragment):
    op_tx = OpTx(_fragment, name="op_tx")
    op_rx = OpRx(_fragment, name="op_rx")

    return op_tx, op_rx


class TestArgType:
    def test_empty_init(self):
        t = ArgType()
        assert t.container_type == ArgContainerType.NATIVE
        assert t.element_type == ArgElementType.CUSTOM

    def test_init(self):
        t = ArgType(ArgElementType.FLOAT64, ArgContainerType.ARRAY)
        assert t.container_type == ArgContainerType.ARRAY
        assert t.element_type == ArgElementType.FLOAT64

    def test_repr(self):
        t = ArgType(ArgElementType.FLOAT64, ArgContainerType.ARRAY)
        assert repr(t) == "std::array<double,N>"

    def test_dynamic_attribute_not_allowed(self):
        obj = ArgType()
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestArg:
    def test_name(self):
        a = Arg("arg_1")
        assert a.name == "arg_1"

        with pytest.raises(TypeError):
            Arg()

    def test_arg_type(self):
        a = Arg("arg_1")
        isinstance(a.arg_type, ArgType)
        isinstance(a.arg_type.container_type, ArgContainerType)
        isinstance(a.arg_type.element_type, ArgElementType)

        assert a.arg_type.container_type == ArgContainerType.NATIVE
        assert a.arg_type.element_type == ArgElementType.CUSTOM

        # Note: have not implemented assignment operator of Arg as it is
        #       unlikely users need to create these directly from Python

    def test_has_value(self):
        a = Arg("my_arg")
        assert not a.has_value

    def test_repr(self):
        a = Arg("my_arg")
        s = a.__repr__()
        assert s == "name: my_arg\ntype: CustomType"

    def test_dynamic_attribute_not_allowed(self):
        obj = Arg("a")
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5

    def test_int_cast(self):
        arg = py_object_to_arg(5, name="num_iter")
        assert isinstance(arg, Arg)
        assert int(arg) == 5

    def test_bool_cast(self):
        arg = py_object_to_arg(True, name="verbose")
        assert isinstance(arg, Arg)
        assert bool(arg)

    def test_float_cast(self):
        arg = py_object_to_arg(5.5, name="beta")
        assert isinstance(arg, Arg)
        assert float(arg) == 5.5

    def test_str_cast(self):
        arg = py_object_to_arg("abc", name="name")
        assert isinstance(arg, Arg)
        assert str(arg) == "abc"


class TestArgList:
    def test_init(self):
        args = ArgList()
        assert args.name == "arglist"
        assert args.size == 0

    def test_add_and_clear(self):
        args = ArgList()
        args.add(Arg("alpha"))
        args.add(Arg("beta"))
        assert args.size == 2

        args2 = ArgList()
        args2.add(Arg("theta"))
        args2.add(args)
        assert args2.size == 3

        args.clear()
        assert args.size == 0
        assert args2.size == 3

    def test_dynamic_attribute_not_allowed(self):
        obj = ArgList()
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestComponentSpec:
    def test_init(self, fragment):
        c = ComponentSpec(fragment)
        assert c.params == {}
        assert c.fragment is fragment

    def test_dynamic_attribute_not_allowed(self, fragment):
        obj = ComponentSpec(fragment)
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestComponent:
    def test_init(self):
        c = Component()
        assert c.name == ""
        assert c.fragment is None
        assert c.id == -1

    def test_add_arg(self):
        c = Component()
        c.add_arg(Arg("a1"))

    def test_initialize(self):
        c = Component()
        c.initialize()

    def test_dynamic_attribute_allowed(self):
        obj = Component()
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestCondition:
    def test_init(self, fragment):
        c = Condition()
        assert c.name == ""
        assert c.fragment is None

    def test_init_with_kwargs(self):
        c = Condition(a=5, b=(13.7, 15.2), c="abcd")
        assert c.name == ""
        assert c.fragment is None
        assert len(c.args) == 3

    def test_init_with_name_and_kwargs(self):
        # name provided by kwarg
        c = Condition(name="c2", a=5, b=(13.7, 15.2), c="abcd")
        assert c.name == "c2"
        assert c.fragment is None
        assert len(c.args) == 3

    def test_name(self):
        c = Condition()
        c.name = "cond1"
        assert c.name == "cond1"

        c = Condition(name="cond3")
        assert c.name == "cond3"

    def test_fragment(self, fragment):
        c = Condition()
        assert c.fragment is None
        c.fragment = fragment
        assert c.fragment is fragment

    def test_add_arg(self):
        c = Condition()
        c.add_arg(Arg("a1"))

    def test_initialize(self):
        c = Condition()
        c.initialize()

    def test_setup(self, fragment):
        spec = ComponentSpec(fragment=fragment)
        c = Condition()
        c.setup(spec)

    def test_dynamic_attribute_not_allowed(self):
        obj = Condition()
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestResource:
    def test_init(self, fragment):
        r = Resource()
        assert r.name == ""
        assert r.fragment is None

    def test_init_with_kwargs(self):
        r = Resource(a=5, b=(13.7, 15.2), c="abcd")
        assert r.name == ""
        assert r.fragment is None
        assert len(r.args) == 3

    def test_init_with_name_and_kwargs(self):
        # name provided by kwarg
        r = Resource(name="r2", a=5, b=(13.7, 15.2), c="abcd")
        assert r.name == "r2"
        assert r.fragment is None
        assert len(r.args) == 3

    def test_name(self):
        r = Resource()
        r.name = "res1"
        assert r.name == "res1"

        r = Resource(name="res3")
        assert r.name == "res3"

    def test_fragment(self, fragment):
        r = Resource()
        assert r.fragment is None
        r.fragment = fragment
        assert r.fragment is fragment

    def test_add_arg(self):
        r = Resource()
        r.add_arg(Arg("a1"))

    def test_initialize(self):
        r = Resource()
        r.initialize()

    def test_setup(self, fragment):
        spec = ComponentSpec(fragment=fragment)
        r = Resource()
        r.setup(spec)

    def test_dynamic_attribute_not_allowed(self):
        obj = Resource()
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestOperatorSpec:
    def test_init(self, fragment):
        c = OperatorSpec(fragment)
        assert c.params == {}
        assert c.fragment is fragment

    def test_input(self, fragment, capfd):
        c = OperatorSpec(fragment)
        iospec = c.input()
        assert isinstance(iospec, IOSpec)
        assert iospec.name == "__iospec_input"
        assert iospec.io_type == IOSpec.IOType.INPUT

        iospec = c.input("input_no_condition").condition(ConditionType.NONE)
        assert isinstance(iospec, IOSpec)
        assert iospec.name == "input_no_condition"
        assert iospec.io_type == IOSpec.IOType.INPUT
        assert iospec.conditions == [(ConditionType.NONE, None)]

        iospec = c.input("input_message_available_condition").condition(
            ConditionType.MESSAGE_AVAILABLE, min_size=1
        )
        assert isinstance(iospec, IOSpec)
        assert iospec.name == "input_message_available_condition"
        assert iospec.io_type == IOSpec.IOType.INPUT
        assert len(iospec.conditions) == 1
        assert iospec.conditions[0][0] == ConditionType.MESSAGE_AVAILABLE
        assert iospec.conditions[0][1] is not None

        iospec2 = c.input("input2")
        assert iospec2.name == "input2"
        assert iospec.io_type == IOSpec.IOType.INPUT

        # Calling a second time with the same name will log an error to the
        # console. TODO: test this
        iospec2 = c.input("input2")
        captured = capfd.readouterr()
        assert "error" in captured.err
        assert "already exists" in captured.err

    def test_output(self, fragment, capfd):
        c = OperatorSpec(fragment)
        iospec = c.output()
        assert isinstance(iospec, IOSpec)
        assert iospec.name == "__iospec_output"
        assert iospec.io_type == IOSpec.IOType.OUTPUT

        iospec = c.output("output_no_condition").condition(ConditionType.NONE)
        assert isinstance(iospec, IOSpec)
        assert iospec.name == "output_no_condition"
        assert iospec.io_type == IOSpec.IOType.OUTPUT
        assert iospec.conditions == [(ConditionType.NONE, None)]

        iospec = c.output("output_downstream_message_affordable_condition").condition(
            ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE, min_size=1
        )
        assert isinstance(iospec, IOSpec)
        assert iospec.name == "output_downstream_message_affordable_condition"
        assert iospec.io_type == IOSpec.IOType.OUTPUT
        assert len(iospec.conditions) == 1
        assert iospec.conditions[0][0] == ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE
        assert iospec.conditions[0][1] is not None

        iospec2 = c.input("output2")
        assert iospec2.name == "output2"
        assert iospec.io_type == IOSpec.IOType.OUTPUT

        # Calling a second time with the same name will log an error
        iospec2 = c.input("output2")
        captured = capfd.readouterr()
        assert "error" in captured.err
        assert "already exists" in captured.err

    def test_dynamic_attribute_not_allowed(self, fragment):
        obj = OperatorSpec(fragment)
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestInputContext:
    def test_init_not_allowed(self):
        # abstract base class can't be initialized
        with pytest.raises(TypeError):
            InputContext()


class TestOutputContext:
    def test_init_not_allowed(self):
        # abstract base class can't be initialized
        with pytest.raises(TypeError):
            OutputContext()

    def test_output_type(self):
        assert hasattr(OutputContext, "OutputType")
        assert hasattr(OutputContext.OutputType, "GXF_ENTITY")
        assert hasattr(OutputContext.OutputType, "SHARED_POINTER")


def test_condition_type():
    # just verifies that the various enums exist
    (
        ConditionType.NONE,
        ConditionType.MESSAGE_AVAILABLE,
        ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE,
        ConditionType.COUNT,
        ConditionType.BOOLEAN,
    )


class TestConfig:
    def test_init_nonexistent(self, capfd):
        # The following will log a warning to the console
        conf = Config(config_file="nonexistent-file", prefix="")
        captured = capfd.readouterr()
        assert isinstance(conf, Config)
        assert "warning" in captured.err
        assert "Config file 'nonexistent-file' doesn't exist" in captured.err

    def test_init_from_config_file(self, config_file):
        conf = Config(config_file)
        assert conf.config_file == config_file
        assert conf.prefix == ""

    def test_init_from_config_file_and_prefix(self, config_file):
        conf = Config(config_file, prefix="abcd")
        assert conf.config_file == config_file
        assert conf.prefix == "abcd"

    def test_dynamic_attribute_not_allowed(self, config_file):
        obj = Config(config_file)
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestFragment:
    def test_init(self):
        f = Fragment()
        f.name == ""

    def test_name(self, fragment):
        fragment.name = "fragment_1"
        assert fragment.name == "fragment_1"

        with pytest.raises(TypeError):
            fragment.name = 5

    def test_application(self, fragment):
        app = Application()
        fragment.application(app)

    def test_graph(self, fragment):
        # first call to fragment.graph constructs a FlowGraph object
        graph = fragment.graph
        assert isinstance(graph, Graph)
        assert isinstance(graph, FlowGraph)

    def test_executor(self, fragment):
        # first call to fragment.graph constructs a FlowGraph object
        executor = fragment.executor
        assert isinstance(executor, Executor)
        assert isinstance(executor, GXFExecutor)
        assert executor.fragment is fragment

    def test_from_config(self, fragment, config_file):
        fragment.config(config_file)

        aja_kwargs = fragment.from_config("aja")
        assert isinstance(aja_kwargs, ArgList)
        assert aja_kwargs.size == len(aja_kwargs.args) == 5
        # all arguments in the ArgList are YAML nodes
        for arg in aja_kwargs.args:
            assert arg.arg_type.element_type == ArgElementType.YAML_NODE

    def test_from_config_nested_key(self, fragment, config_file):
        fragment.config(config_file)

        width = fragment.from_config("aja.width")
        assert isinstance(width, Arg)
        assert width.arg_type.element_type == ArgElementType.YAML_NODE

    def test_from_config_missing_key(self, fragment, config_file, capfd):
        fragment.config(config_file)
        # TODO: verify that this logs an error
        nonexistent_kwargs = fragment.from_config("nonexistent")
        assert nonexistent_kwargs.size == 0
        msg = "Unable to find the parameter item/map with key 'nonexistent'"
        captured = capfd.readouterr()
        assert "error" in captured.err
        assert msg in captured.err

    def test_uninitialized_config(self, fragment, config_file, capfd):
        assert fragment.config().config_file == ""

    def test_add_operator(self, fragment, config_file):
        fragment.config(config_file)

        op_tx, op_rx = get_tx_and_rx_ops(fragment)
        fragment.add_operator(op_tx)
        fragment.add_operator(op_rx)

    def test_add_flow(self, fragment, config_file, capfd):
        fragment.config(config_file)

        op_tx, op_rx = get_tx_and_rx_ops(fragment)

        # test add_flow with tuple of 2-tuple
        fragment.add_flow(op_tx, op_rx, {("tensor", "in1")})
        # test add_flow with set of 2-tuple
        fragment.add_flow(op_tx, op_rx, {("tensor", "in2")})

        # using non-existent names doesn't yet raise a Python exception...
        fragment.add_flow(op_tx, op_rx, {("nonexistent", "in2")})

        captured = capfd.readouterr()
        assert "error" in captured.err
        assert "nonexistent" in captured.err

    def test_dynamic_attribute_allowed(self, fragment):
        fragment.custom_attribute = 5


class TestApplication:
    def test_init(self):
        app = Application()
        assert isinstance(app, _Fragment)

    def test_name(self, app):
        app.name = "app_1"
        assert app.name == "app_1"

        with pytest.raises(TypeError):
            app.name = 5

    def test_graph(self, app):
        # first call to app.graph constructs a FlowGraph object
        graph = app.graph
        assert isinstance(graph, Graph)
        assert isinstance(graph, FlowGraph)

    def test_executor(self, app):
        # first call to app.graph constructs a FlowGraph object
        executor = app.executor
        assert isinstance(executor, Executor)
        assert isinstance(executor, GXFExecutor)
        assert executor.fragment is app

    def test_from_config(self, app, config_file):
        app.config(config_file)

        aja_kwargs = app.from_config("aja")
        assert isinstance(aja_kwargs, ArgList)
        assert aja_kwargs.size == 5

    def test_kwargs(self, app, config_file):
        app.config(config_file)

        replayer_kwargs = app.kwargs("aja")
        assert isinstance(replayer_kwargs, dict)
        assert "enable_overlay" in replayer_kwargs

    def test_from_config_missing_key(self, app, config_file, capfd):
        app.config(config_file)
        # TODO: verify that this logs an error
        nonexistent_kwargs = app.from_config("nonexistent")
        assert nonexistent_kwargs.size == 0
        msg = "Unable to find the parameter item/map with key 'nonexistent'"
        captured = capfd.readouterr()
        assert "error" in captured.err
        assert msg in captured.err

    def test_uninitialized_config(self, app, config_file, capfd):
        assert app.config().config_file == ""

    def test_add_operator(self, app, config_file):
        app.config(config_file)

        op_tx, op_rx = get_tx_and_rx_ops(app)
        app.add_operator(op_tx)
        app.add_operator(op_rx)

    def test_add_flow(self, app, config_file, capfd):
        app.config(config_file)

        op_tx, op_rx = get_tx_and_rx_ops(app)

        # list of 2-tuples
        app.add_flow(op_tx, op_rx, {("tensor", "in1")})
        # set of 2-tuples
        app.add_flow(op_tx, op_rx, {("tensor", "in2")})

        # using non-existent names doesn't yet raise a Python exception...
        app.add_flow(op_tx, op_rx, {("nonexistent", "in2")})

        captured = capfd.readouterr()
        assert "error" in captured.err
        assert "nonexistent" in captured.err

    def test_dynamic_attribute(self, app):
        # verify that attributes not in the underlying C++ class can be
        # dynamically added
        app.custom_attribute = 5


class TestIOSpec:
    @pytest.mark.parametrize(
        "name, io_type",
        [("input", IOSpec.IOType.INPUT), ("output", IOSpec.IOType.OUTPUT)],
    )
    def test_init(self, fragment, name, io_type):
        op_spec = OperatorSpec(fragment)
        io_spec = IOSpec(op_spec, name, io_type)
        assert io_spec.name == name
        assert io_spec.io_type == io_type
        assert io_spec.resource is None
        assert io_spec.conditions == []

        if name == "output":
            assert repr(io_spec) == "<IOSpec: name=output, io_type=OUTPUT>"
        elif name == "input":
            assert repr(io_spec) == "<IOSpec: name=input, io_type=INPUT>"

    @pytest.mark.parametrize(
        "name, io_type",
        [("input", IOSpec.IOType.INPUT), ("output", IOSpec.IOType.OUTPUT)],
    )
    def test_kwarg_init(self, fragment, name, io_type):
        op_spec = OperatorSpec(fragment)
        io_spec = IOSpec(op_spec=op_spec, name=name, io_type=io_type)
        assert io_spec.name == name
        assert io_spec.io_type == io_type

    def test_dynamic_attribute_not_allowed(self, fragment):
        op_spec = OperatorSpec(fragment)
        io_spec = IOSpec(op_spec=op_spec, name="in", io_type=IOSpec.IOType.INPUT)
        with pytest.raises(AttributeError):
            io_spec.custom_attribute = 5


# Various classes/enums below here have minimal Python functionality and may
# not all be needed.


class TestExecutor:
    def test_init(self, fragment):
        e = Executor(fragment)
        assert e.context is None

    def test_dynamic_attribute_not_allowed(self, fragment):
        obj = Executor(fragment)
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5
