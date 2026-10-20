/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/allocator.hpp>
#include <holoscan/memory/memory_buffer.hpp>
#include <holoscan/memory/memory_storage_type.hpp>
#include <holoscan/memory/tensor_shape.hpp>

namespace holoscan {

/**
 * @brief Allocate tensor storage with a native allocator and attach it to a Holoscan tensor.
 *
 * Shape is represented as a vector of int64_t dimensions, dtype is DLPack's DLDataType, and
 * optional strides are byte strides matching holoscan::Tensor::strides(). The
 * optional stream is passed to the allocator and becomes the initial deallocation stream recorded
 * by the backing MemoryBuffer. It is typed as void* so this header does not require CUDA headers.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> reshape_custom(
    Tensor& tensor, TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, const std::shared_ptr<Allocator>& allocator,
    void* stream = nullptr);

/**
 * @brief Allocate tensor storage with explicit MemoryBuffer options.
 *
 * Runtime setup can use this overload to pass graph-derived buffer-control
 * parameters, such as the maximum same-process fanout release-stream count,
 * before the backing storage is attached.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> reshape_custom(
    Tensor& tensor, TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, const std::shared_ptr<Allocator>& allocator,
    MemoryBuffer::Options memory_options, void* stream = nullptr);

/**
 * @brief Allocate tensor storage and return a Tensor value.
 *
 * This value-returning helper is the production-direction spelling for code
 * that wants an ordinary `holoscan::Tensor` handle while keeping allocation
 * and lifetime ownership hidden behind Tensor storage.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, const std::shared_ptr<Allocator>& allocator,
    void* stream = nullptr);

/**
 * @brief Allocate tensor storage with explicit MemoryBuffer options and return a Tensor value.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, const std::shared_ptr<Allocator>& allocator,
    MemoryBuffer::Options memory_options, void* stream = nullptr);

/**
 * @brief Allocate tensor storage on a CCCL CUDA stream and return a Tensor value.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, const std::shared_ptr<Allocator>& allocator,
    ::cuda::stream_ref stream);

/**
 * @brief Allocate tensor storage on a CCCL CUDA stream with explicit MemoryBuffer options.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, const std::shared_ptr<Allocator>& allocator,
    MemoryBuffer::Options memory_options, ::cuda::stream_ref stream);

/**
 * @brief Attach an existing allocation to a Holoscan tensor.
 *
 * The move-only allocation token is consumed by this call, but the underlying
 * buffer control block is retained by the Tensor context. Application-facing
 * Tensor handles therefore remain ordinary `holoscan::Tensor` values while
 * allocation lifetime remains hidden behind Tensor storage.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> attach_allocation(
    Tensor& tensor, TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    BufferAllocation allocation);

/**
 * @brief Consume an existing allocation and return a Tensor value.
 *
 * The returned Tensor retains the hidden buffer control block after the
 * move-only allocation token has been consumed.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    BufferAllocation allocation);

/**
 * @brief Wrap existing storage with an explicit Tensor memory reference.
 *
 * This is the value-returning helper for storage whose lifetime, readiness, or
 * release behavior is represented by a TensorMemoryReference rather than by a
 * MemoryBuffer. It is intended for sidecar loans, imported buffers, and runtime
 * source-slot pools that need custom stream/event behavior while still
 * presenting an ordinary `holoscan::Tensor` value to application code.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, void* pointer,
    std::shared_ptr<TensorMemoryReference> memory_reference);

/**
 * @brief Wrap existing storage with an explicit memory reference and known device ordinal.
 *
 * @param shape Tensor dimensions.
 * @param dtype DLPack element type.
 * @param byte_strides Optional byte strides matching @p shape.
 * @param storage_type Physical storage type for @p pointer.
 * @param pointer Existing storage address.
 * @param memory_reference Owner and stream-lifetime contract for @p pointer.
 * @param device_id CUDA device ordinal asserted by the caller for device storage, or
 * Allocator::kUnknownDeviceId to derive it from @p pointer.
 * @return A Tensor on success, or an Error for invalid shape, type, strides, storage, or inputs.
 * @throws std::bad_alloc if Tensor context or metadata allocation fails.
 * @note A nonnegative @p device_id is trusted provenance and avoids a CUDA pointer-attribute
 * query. Use Allocator::kUnknownDeviceId when the ordinal must be derived from @p pointer.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, void* pointer,
    std::shared_ptr<TensorMemoryReference> memory_reference, int32_t device_id);

/**
 * @brief Wrap existing tensor storage and attach it to a Holoscan tensor.
 *
 * The release function is optional; when provided, it is called when the tensor releases the
 * backing MemoryBuffer. A nonnegative device_id is a caller assertion that device storage belongs
 * to that CUDA ordinal; this skips the CUDA pointer-attribute query used for validation. Leave it
 * as Allocator::kUnknownDeviceId (the default) for unknown provenance so Tensor construction
 * derives and validates the ordinal from the backing pointer when the storage type requires it.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> wrap_memory(
    Tensor& tensor, TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, void* pointer,
    MemoryBuffer::release_function_t release_func = nullptr,
    int32_t device_id = Allocator::kUnknownDeviceId);

/**
 * @brief Wrap existing tensor storage and return a Tensor value.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Tensor, Error> make_tensor(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, void* pointer,
    MemoryBuffer::release_function_t release_func = nullptr);

}  // namespace holoscan
