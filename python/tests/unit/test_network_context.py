"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import holoscan.network_contexts as network_contexts
from holoscan.core import NetworkContext
from holoscan.core._core import ComponentSpec as ComponentSpecBase
from holoscan.gxf import GXFNetworkContext
from holoscan.network_contexts import UcxContext
from holoscan.resources import UcxEntitySerializer


@pytest.mark.skipif(
    not hasattr(network_contexts, "FastDdsPubSubNetworkContext"),
    reason="FastDDS pub/sub backend not enabled in this build",
)
class TestFastDdsPubSubNetworkContext:
    def test_default_init(self, app):
        context = network_contexts.FastDdsPubSubNetworkContext(app)
        assert isinstance(context, GXFNetworkContext)
        assert isinstance(context, NetworkContext)
        assert isinstance(context.spec, ComponentSpecBase)

    def test_init_kwargs(self, app):
        name = "fastdds_context"
        context = network_contexts.FastDdsPubSubNetworkContext(
            app,
            native_buffer_policy="required",
            native_buffer_acquire_timeout_ms=250,
            native_buffer_export_ttl_ms=7500,
            native_buffer_use_eager_acquire=True,
            name=name,
        )
        assert isinstance(context, GXFNetworkContext)
        app.network_context(context)
        assert f"name: {name}" in repr(context)

    def test_init_defaults(self, app):
        context = network_contexts.FastDdsPubSubNetworkContext(app)
        app.network_context(context)


class TestUcxContext:
    def test_default_init(self, app):
        e = UcxContext(app)
        assert isinstance(e, GXFNetworkContext)
        assert isinstance(e, NetworkContext)
        assert isinstance(e.spec, ComponentSpecBase)

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
