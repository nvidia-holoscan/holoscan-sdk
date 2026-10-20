# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holoscan Inference operator bindings."""

from typing import Any as _Any

from holoscan.registry import resolve as _resolve


class _InferenceOpType:
    """Callable module facade for the registered C++ InferenceOp."""

    def __init__(self) -> None:
        self._registered = _resolve("holoscan.inference.InferenceOp", kind="operator")

    @property
    def Handle(self) -> object:  # noqa: N802
        return self._registered.Handle

    @property
    def name(self) -> str:
        return self._registered.name

    @property
    def description(self) -> str:
        return self._registered.description

    @property
    def parameter_introspection(self) -> str:
        return self._registered.parameter_introspection

    @property
    def parameter_summary(self) -> str:
        return self._registered.parameter_summary

    @property
    def ports(self) -> object:
        return self._registered.ports

    @property
    def parameters(self) -> object:
        return self._registered.parameters

    @property
    def required_parameters(self) -> frozenset[str]:
        return self._registered.required_parameters

    def __call__(
        self,
        graph: _Any,
        instance_name: str,
        *,
        config_path: str | None = None,
        model_path_map: object | None = None,
        input_map: object | None = None,
        output_map: object | None = None,
        backend: str | None = None,
        backend_map: object | None = None,
        trt_opt_profile: object | None = None,
        is_engine_path: bool | None = None,
        input_on_cuda: bool | None = None,
        output_on_cuda: bool | None = None,
        transmit_on_cuda: bool | None = None,
        dynamic_input_dims: bool | None = None,
        output_dynamic_dims: bool | None = None,
        enable_fp16: bool | None = None,
        enable_cuda_graphs: bool | None = None,
        dla_core: int | None = None,
        dla_core_map: object | None = None,
        dla_gpu_fallback: bool | None = None,
        parallel_inference: bool | None = None,
        max_batch_size: int | None = None,
        max_memory: int | None = None,
        cuda_stream_pool: object | None = None,
        cuda_green_context_pool: object | None = None,
    ) -> object:
        """Add an InferenceOp from one YAML file or the three direct model maps.

        Relative model paths loaded through ``config_path`` resolve from that file's
        directory. Relative paths in ``model_path_map`` are retained for the runtime
        deployment target and therefore resolve from that process's working directory.
        """
        direct = {
            "model_path_map": model_path_map,
            "input_map": input_map,
            "output_map": output_map,
            "backend": backend,
            "backend_map": backend_map,
            "trt_opt_profile": trt_opt_profile,
            "is_engine_path": is_engine_path,
            "input_on_cuda": input_on_cuda,
            "output_on_cuda": output_on_cuda,
            "transmit_on_cuda": transmit_on_cuda,
            "dynamic_input_dims": dynamic_input_dims,
            "output_dynamic_dims": output_dynamic_dims,
            "enable_fp16": enable_fp16,
            "enable_cuda_graphs": enable_cuda_graphs,
            "dla_core": dla_core,
            "dla_core_map": dla_core_map,
            "dla_gpu_fallback": dla_gpu_fallback,
            "parallel_inference": parallel_inference,
            "max_batch_size": max_batch_size,
            "max_memory": max_memory,
            "cuda_stream_pool": cuda_stream_pool,
            "cuda_green_context_pool": cuda_green_context_pool,
        }
        supplied = {name: value for name, value in direct.items() if value is not None}
        if config_path is not None:
            if supplied:
                raise TypeError("config_path cannot be combined with direct InferenceOp parameters")
            return self._registered(graph, instance_name, config_path=config_path)

        missing = [
            name for name in ("model_path_map", "input_map", "output_map") if direct[name] is None
        ]
        if missing:
            names = ", ".join(missing)
            raise TypeError(f"direct InferenceOp configuration requires: {names}")
        return self._registered(graph, instance_name, **supplied)

    def __repr__(self) -> str:
        return repr(self._registered)


InferenceOp = _InferenceOpType()

__all__ = ["InferenceOp"]
