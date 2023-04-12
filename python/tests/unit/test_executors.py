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

from holoscan.core import Executor, IOSpec
from holoscan.executors import GXFExecutor, create_input_port, create_output_port
from holoscan.operators import BayerDemosaicOp
from holoscan.resources import CudaStreamPool, UnboundedAllocator


class TestGXFExecutor:
    def test_fragment(self, app):
        executor = GXFExecutor(app)
        assert executor.fragment is app

    def test_context(self, app):
        executor = GXFExecutor(app)
        assert type(executor.context).__name__ == "PyCapsule"

    def test_type(self, app):
        executor = GXFExecutor(app)
        assert isinstance(executor, Executor)

    def test_dynamic_attribute_not_allowed(self, app):
        obj = GXFExecutor(app)
        with pytest.raises(AttributeError):
            obj.custom_attribute = 5


@pytest.mark.parametrize("port_type", ("input", "output"))
def test_create_port_methods(port_type, app):
    # op here could be any operator that is a GXFOperator
    op = BayerDemosaicOp(
        app,
        pool=UnboundedAllocator(app),
        cuda_stream_pool=CudaStreamPool(
            app,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        ),
    )
    if port_type == "input":
        io_spec = IOSpec(op.spec, "my_" + port_type, IOSpec.IOType.INPUT)
        creation_func = create_input_port
    else:
        io_spec = IOSpec(op.spec, "my_" + port_type, IOSpec.IOType.OUTPUT)
        creation_func = create_output_port

    assert io_spec.resource is None

    creation_func(app, op.gxf_context, op.gxf_eid, io_spec)

    assert io_spec.resource is not None
    assert io_spec.resource.name == "my_" + port_type
    if port_type == "input":
        assert io_spec.resource.gxf_typename == "nvidia::gxf::DoubleBufferReceiver"
    elif port_type == "output":
        assert io_spec.resource.gxf_typename == "nvidia::gxf::DoubleBufferTransmitter"
