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

import os

from holoscan.core import _Operator
from holoscan.gxf import GXFOperator
from holoscan.operators import BayerDemosaicOp
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType

sample_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")


class TestBayerDemosaicOp:
    def test_kwarg_based_initialization(self, app, config_file, capfd):
        app.config(config_file)
        demosaic_stream_pool = CudaStreamPool(
            app,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        op = BayerDemosaicOp(
            app,
            name="demosaic",
            pool=BlockMemoryPool(
                fragment=app,
                name="device_alloc",
                storage_type=MemoryStorageType.DEVICE,
                block_size=16 * 1024**2,
                num_blocks=4,
            ),
            cuda_stream_pool=demosaic_stream_pool,
            **app.kwargs("demosaic"),
        )
        assert isinstance(op, GXFOperator)
        assert isinstance(op, _Operator)
        assert op.gxf_typename == "nvidia::holoscan::BayerDemosaic"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err
