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

from holoscan.core import Component, Condition, Resource, _Operator
from holoscan.gxf import (
    Entity,
    GXFComponent,
    GXFCondition,
    GXFExecutionContext,
    GXFInputContext,
    GXFOperator,
    GXFOutputContext,
    GXFResource,
)

# need any operator based on GXFOperator for testing here
# from holoscan.operators import BayerDemosaicOp


class TestEntity:
    def test_not_constructable(self):
        with pytest.raises(TypeError):
            Entity()


class TestGXFComponent:
    GXFClass = GXFComponent

    def test_init(self):
        c = self.GXFClass()
        assert c.gxf_cid == 0
        assert c.gxf_eid == 0
        assert c.gxf_cname == ""

    def test_cid(self):
        c = self.GXFClass()
        c.gxf_cid = 5
        assert c.gxf_cid == 5

        with pytest.raises(TypeError):
            c.gxf_cid = "abcd"

    def test_eid(self):
        c = self.GXFClass()
        c.gxf_eid = 5
        assert c.gxf_eid == 5

        with pytest.raises(TypeError):
            c.gxf_eid = "abcd"

    def test_cname(self):
        c = self.GXFClass()
        c.gxf_cname = "my_name"
        assert c.gxf_cname == "my_name"

        with pytest.raises(TypeError):
            c.gxf_cname = 5

    def test_type(self):
        c = self.GXFClass()
        assert isinstance(c, self.GXFClass)

    def test_dynamic_attribute_not_allowed(self):
        obj = GXFComponent()
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestGXFCondition(TestGXFComponent):
    GXFClass = GXFCondition

    def test_type(self):
        c = GXFCondition()
        assert isinstance(c, Component)
        assert isinstance(c, Condition)
        assert isinstance(c, GXFComponent)

    def test_dynamic_attribute_not_allowed(self):
        obj = GXFCondition()
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestGXFResource(TestGXFComponent):
    GXFClass = GXFResource

    def test_type(self):
        r = GXFResource()
        assert isinstance(r, Component)
        assert isinstance(r, Resource)
        assert isinstance(r, GXFComponent)

    def test_dynamic_attribute_not_allowed(self):
        obj = GXFResource()
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


class TestGXFInputContext:
    def test_init(self, app, config_file):
        app.config(config_file)
        context = app.executor.context
        op = GXFOperator()
        input_context = GXFInputContext(context, op)
        assert isinstance(input_context, GXFInputContext)


class TestGXFOutputContext:
    def test_init(self, app, config_file):
        app.config(config_file)
        context = app.executor.context
        op = GXFOperator()
        output_context = GXFOutputContext(context, op)
        assert isinstance(output_context, GXFOutputContext)


class TestGXFExecutionContext:
    def test_init(self, app, config_file):
        app.config(config_file)
        context = app.executor.context
        op = GXFOperator()
        output_context = GXFExecutionContext(context, op)
        assert isinstance(output_context, GXFExecutionContext)


class TestGXFOperator:
    GXFClass = GXFOperator

    def test_init(self):
        c = GXFOperator()
        assert c.gxf_cid == 0
        assert c.gxf_eid == 0

    def test_cid(self):
        c = GXFOperator()
        c.gxf_cid = 5
        assert c.gxf_cid == 5

        with pytest.raises(TypeError):
            c.gxf_cid = "abcd"

    def test_eid(self):
        c = GXFOperator()
        c.gxf_eid = 5
        assert c.gxf_eid == 5

        with pytest.raises(TypeError):
            c.gxf_eid = "abcd"

    def test_type(self):
        op = GXFOperator()
        assert isinstance(op, _Operator)

    def test_dynamic_attribute_allowed(self):
        obj = GXFOperator()
        obj.custom_attribute = 5

    # initialize causes a segfault
    # def test_initialize(self):
    #     op = GXFOperator()
    #     op.initialize()
