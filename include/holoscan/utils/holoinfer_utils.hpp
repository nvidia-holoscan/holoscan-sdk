/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan_core_export.h>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/port.hpp>

#include <holoinfer/holoinfer_buffer.hpp>

// Glue layer between InferenceOp's `holoscan::Input<Tensor>`/`holoscan::Output<Tensor>` ports and
// HoloInfer's DataBuffer-based backend. Each model input/output tensor is bound to its own stable
// port, declared dynamically in InferenceOp::setup() from the operator's compile-time-authored
// parameter aggregate (input_map / output_map). HoloInfer-side DataBuffer allocations are reused
// across calls (allocated once, resized only if a shape grows). `ExecutionContext::cuda_stream()`
// supplies the single Runtime-owned stream used per activation.
//
// Note: this header intentionally does not alias `holoscan::inference` to `HoloInfer` (that
// alias must not live in a public header — see inference.hpp). Callers that want the short form
// declare their own `namespace HoloInfer = holoscan::inference;` at file scope in their .cpp.
namespace holoscan::utils {

/**
 * @brief Convert a HoloInfer element datatype to its DLPack equivalent.
 * @return The matching DLDataType, or DLDataType{} (code/bits/lanes all zero) when unsupported.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT DLDataType
holoinfer_dtype_to_dlpack(holoscan::inference::holoinfer_datatype dtype) noexcept;

/**
 * @brief Convert a DLPack dtype to its closest HoloInfer element datatype.
 * @return The matching holoinfer_datatype, or h_Unsupported when no exact match exists.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT holoscan::inference::holoinfer_datatype
dlpack_dtype_to_holoinfer(DLDataType dtype) noexcept;

/**
 * @brief Extract input tensor data from a set of named Input<Tensor> ports into HoloInfer's
 * per-tensor device/host buffers.
 *
 * For each `(name, port)` pair in `ports`, the tensor is received and its data copied
 * (device-to-device, or device-to-host when `input_on_cuda` is false) into
 * `data_per_tensor[name]`. The backing HoloInfer DataBuffer is allocated on first use (or resized
 * if a later frame is larger) using the received tensor's own shape.
 *
 * @param ports Map of input tensor name to its bound Input<Tensor> port.
 * @param data_per_tensor HoloInfer input DataMap to populate (InferenceSpecs::data_per_tensor_).
 * @param dims_per_tensor Map updated with the received dimensions of each input tensor.
 * @param input_on_cuda Whether the extracted buffer should be device-resident.
 * @param module Caller name used in error messages.
 * @param stream CUDA stream used for the extraction copies.
 * @return Success, or a typed error.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> get_data_per_model(
    const std::map<std::string, Input<Tensor>*>& ports,
    holoscan::inference::DataMap& data_per_tensor,
    std::map<std::string, std::vector<int>>& dims_per_tensor, bool input_on_cuda,
    const std::string& module, cudaStream_t stream);

/**
 * @brief Transmit HoloInfer per-model output buffers through their bound Output<Tensor> ports.
 *
 * For each model in `model_to_tensor_map`, each of its output tensor names is looked up in
 * `output_per_tensor` (keyed by tensor name), loaned from the port's runtime pool, filled by
 * an async CUDA copy on the compute stream, committed (which records the readiness event so
 * downstream consumers are correctly ordered w.r.t. the copy), and emitted.
 *
 * The runtime-pool grant is declared by `InferenceOp::setup()`; this path never calls a
 * general-purpose allocator on the steady-state hot path (see performance_model.mdx).
 *
 * A tensor name with no entry in `ports` is treated as an internal/chained tensor (produced by
 * one model and consumed by another within this same InferenceOp) rather than an error: it is
 * silently skipped, since InferenceOp only declares an `Output<Tensor>` port for its *external*
 * outputs (see `InferenceOp::setup()`).
 *
 * @param model_to_tensor_map Map of model name to its ordered list of output tensor names.
 * @param ports Map of output tensor name to its bound Output<Tensor> port.
 * @param output_per_tensor HoloInfer output DataMap (InferenceSpecs::output_per_model_), keyed by
 * tensor name.
 * @param model_dims Output tensor dimensions, keyed by model name; each entry holds one dims
 * vector per tensor name of that model, in the same order as `model_to_tensor_map`.
 * @param output_on_cuda Whether HoloInfer's own output buffer is device-resident.
 * @param transmit_on_cuda Whether the emitted Tensor should be device-resident.
 * @param module Caller name used in error messages.
 * @param context ExecutionContext supplying the CUDA compute stream and activation-time stamp.
 * @return Success, or a typed error.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> transmit_data_per_model(
    const holoscan::inference::MultiMappings& model_to_tensor_map,
    const std::map<std::string, Output<Tensor>*>& ports,
    holoscan::inference::DataMap& output_per_tensor, const holoscan::inference::DimType& model_dims,
    bool output_on_cuda, bool transmit_on_cuda, const std::string& module,
    holoscan::ExecutionContext& context);

}  // namespace holoscan::utils
