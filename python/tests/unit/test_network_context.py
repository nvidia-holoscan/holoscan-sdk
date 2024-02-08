"""
 SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import ComponentSpec, NetworkContext
from holoscan.gxf import GXFNetworkContext
from holoscan.network_contexts import UcxContext
from holoscan.resources import UcxEntitySerializer


class TestUcxContext:
    def test_default_init(self, app):
        e = UcxContext(app)
        assert isinstance(e, GXFNetworkContext)
        assert isinstance(e, NetworkContext)
        assert isinstance(e.spec, ComponentSpec)

    def test_init_kwargs(self, app):
        entity_serializer = UcxEntitySerializer(
            fragment=app,
            verbose_warning=False,
            name="ucx_entity_serializer",
        )
        name = "net_context"
        context = UcxContext(
            app,
            serializer=entity_serializer,
            name=name,
        )
        assert isinstance(context, GXFNetworkContext)
        app.network_context(context)
        assert f"name: {name}" in repr(context)

    def test_init_defaults(self, app):
        context = UcxContext(app)
        app.network_context(context)
