# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Final, TypedDict

import holoscan
from holoscan.operators import OperatorHandle as _OperatorHandle
from holoscan.registry import OperatorParameterInfo as _OperatorParameterInfo
from holoscan.registry import OperatorPortInfo as _OperatorPortInfo

type _StringMap = dict[str, str]
type _StringSequenceMap = dict[str, list[str] | tuple[str, ...]]

class _CudaStreamPoolConfig(TypedDict, total=False):
    device_id: int
    stream_flags: int
    stream_priority: int
    reserved_size: int
    max_size: int
    nvtx_identifier: str

class _CudaGreenContextPoolConfig(TypedDict, total=False):
    device_id: int
    num_partitions: int
    sms_per_partition: list[int] | tuple[int, ...]
    default_context: int | None
    nvtx_identifier: str

class _InferenceOpType:
    @property
    def Handle(self) -> type[_OperatorHandle]: ...  # noqa: N802
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def parameter_introspection(self) -> str: ...
    @property
    def parameter_summary(self) -> str: ...
    @property
    def ports(self) -> Mapping[str, _OperatorPortInfo]: ...
    @property
    def parameters(self) -> Mapping[str, _OperatorParameterInfo]: ...
    @property
    def required_parameters(self) -> frozenset[str]: ...
    def __call__(
        self,
        graph: holoscan.Graph,
        instance_name: str,
        *,
        config_path: str | None = None,
        model_path_map: _StringMap | None = None,
        input_map: _StringSequenceMap | None = None,
        output_map: _StringSequenceMap | None = None,
        backend: str | None = None,
        backend_map: _StringMap | None = None,
        trt_opt_profile: _StringSequenceMap | None = None,
        is_engine_path: bool | None = None,
        input_on_cuda: bool | None = None,
        output_on_cuda: bool | None = None,
        transmit_on_cuda: bool | None = None,
        dynamic_input_dims: bool | None = None,
        output_dynamic_dims: bool | None = None,
        enable_fp16: bool | None = None,
        enable_cuda_graphs: bool | None = None,
        dla_core: int | None = None,
        dla_core_map: _StringMap | None = None,
        dla_gpu_fallback: bool | None = None,
        parallel_inference: bool | None = None,
        max_batch_size: int | None = None,
        max_memory: int | None = None,
        cuda_stream_pool: _CudaStreamPoolConfig | None = None,
        cuda_green_context_pool: _CudaGreenContextPoolConfig | None = None,
    ) -> _OperatorHandle: ...
    def __repr__(self) -> str: ...

InferenceOp: Final[_InferenceOpType]

__all__: list[str] = ["InferenceOp"]
