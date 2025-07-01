"""
SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import Component
from holoscan.core import _Condition as ConditionBase
from holoscan.core import _Operator as OperatorBase
from holoscan.core import _Resource as ResourceBase
from holoscan.gxf import (
    Entity,
    GXFComponent,
    GXFCondition,
    GXFExecutionContext,
    GXFInputContext,
    GXFNetworkContext,
    GXFOperator,
    GXFOutputContext,
    GXFResource,
    GXFScheduler,
)


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
        assert isinstance(c, ConditionBase)
        assert isinstance(c, GXFComponent)

    def test_dynamic_attribute_allowed(self):
        obj = GXFCondition()
        obj.custom_attribute = 5


class TestGXFResource(TestGXFComponent):
    GXFClass = GXFResource

    def test_type(self):
        r = GXFResource()
        assert isinstance(r, Component)
        assert isinstance(r, ResourceBase)
        assert isinstance(r, GXFComponent)

    def test_dynamic_attribute_allowed(self):
        # parent Resource class allows dynamic attributes
        obj = GXFResource()
        obj.custom_attribute = 5


class TestGXFInputContext:
    def test_init(self, app):
        context = app.executor.context
        op = GXFOperator()
        exec_context = GXFExecutionContext(context, op)
        input_context = GXFInputContext(exec_context, op)
        assert isinstance(input_context, GXFInputContext)


class TestGXFOutputContext:
    def test_init(self, app):
        context = app.executor.context
        op = GXFOperator()
        exec_context = GXFExecutionContext(context, op)
        output_context = GXFOutputContext(exec_context, op)
        assert isinstance(output_context, GXFOutputContext)


class TestGXFExecutionContext:
    def test_init(self, app):
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
        assert isinstance(op, OperatorBase)

    def test_dynamic_attribute_allowed(self):
        obj = GXFOperator()
        obj.custom_attribute = 5

    # initialize causes a segfault
    # def test_initialize(self):
    #     op = GXFOperator()
    #     op.initialize()


class TestGXFScheduler:
    def test_base_class_init(self):
        with pytest.raises(TypeError):
            GXFScheduler()


class TestGXFNetworkContext:
    def test_base_class_init(self):
        with pytest.raises(TypeError):
            GXFNetworkContext()
