/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

#include <cstdint>
#include <string_view>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan {

/// Options controlling how DLPack metadata is copied when wrapping or exporting tensors.
struct DLPackMetadataCopyOptions {
  /// When true, materialize explicit (element) strides even if the source uses implicit
  /// (nullptr) strides. Required for DLPack >= 1.2 versioned tensors with ndim != 0.
  bool materialize_implicit_strides = false;
};

/// DLPack 1.2+ requires explicit strides in DLManagedTensorVersioned when ndim != 0.
[[nodiscard]] constexpr bool dlpack_version_requires_explicit_strides(
    DLPackVersion version) noexcept {
  return version.major > 1 || (version.major == 1 && version.minor >= 2);
}

/**
 * @brief Return the number of bytes occupied by one DLPack element.
 *
 * Vectorized DLPack dtypes count all lanes as one logical element, so this helper rounds the
 * bit-lane product up to a whole number of bytes.
 *
 * @param dtype The DLPack dtype to inspect.
 * @return The number of bytes occupied by one element.
 */
[[nodiscard]] constexpr uint64_t dlpack_dtype_item_size_bytes(DLDataType dtype) noexcept {
  return (static_cast<uint64_t>(dtype.bits) * dtype.lanes + 7U) / 8U;
}

/**
 * @brief Copy a DLTensor while pointing the destination shape/stride arrays at owned storage.
 *
 * @param source The DLTensor to copy.
 * @param[out] destination The DLTensor to populate (shape/strides point into the owned vectors).
 * @param[out] owned_shape Owned shape storage.
 * @param[out] owned_strides Owned stride storage (in elements).
 * @param options Copy options (e.g. materialize implicit strides).
 */
HOLOSCAN_CORE_EXPORT void copy_dl_tensor_with_owned_metadata(
    const DLTensor& source, DLTensor& destination, std::vector<int64_t>& owned_shape,
    std::vector<int64_t>& owned_strides, DLPackMetadataCopyOptions options = {});

/// Copy an unversioned DLTensor into a Tensor context (treated as current version, flags cleared).
HOLOSCAN_CORE_EXPORT void copy_dl_tensor_with_owned_metadata(
    DLManagedTensorContext& context, const DLTensor& source,
    DLPackMetadataCopyOptions options = {});

/// Copy a versioned DLPack tensor into a Tensor context, preserving version and flags.
HOLOSCAN_CORE_EXPORT void copy_dl_tensor_with_owned_metadata(
    DLManagedTensorContext& context, const DLManagedTensorVersioned& source,
    DLPackMetadataCopyOptions options = {});

/**
 * @brief Detect the device information from the given pointer.
 *
 * @param ptr The pointer to the memory.
 * @return The device information, or an Error if the CUDA pointer query fails.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<DLDevice, Error> dldevice_from_pointer(void* ptr);

/**
 * @brief Fill strides from the given DLTensor object.
 *
 * The following fields are used to fill strides:
 *
 * - ndim
 * - shape
 * - dtype
 *
 * If tensor's strides is nullptr, `strides` argument is filled with the calculated strides of the
 * given DLTensor object. Otherwise, `strides` argument is filled with the given DLTensor object's
 * strides. `strides` is resized to `ndim`, or cleared (left empty) when `ndim <= 0` or when
 * implicit strides are requested but the shape pointer is null. Byte strides use the full element
 * size, accounting for vectorized (multi-lane) dtypes.
 *
 * @param tensor DLTensor object that holds information to fill strides.
 * @param[out] strides Strides to fill.
 * @param to_num_elements If true, the strides in `strides` argument are in number of elements, not
 * bytes (default: false).
 */
HOLOSCAN_CORE_EXPORT void calc_strides(const DLTensor& tensor, std::vector<int64_t>& strides,
                                       bool to_num_elements = false);

/**
 * @brief Return DLDataType object from the NumPy type string.
 *
 * @param typestr The NumPy type string.
 * @return The DLDataType object, or an Error for an unsupported/big-endian type string.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<DLDataType, Error> dldatatype_from_typestr(
    std::string_view typestr);

/**
 * @brief Return a string providing the basic type of the homogeneous array in NumPy.
 *
 * Note: This method assumes little-endian for now.
 *
 * @return A const character pointer that represents a string, or an Error for an unsupported
 * DLDataType.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<const char*, Error> numpy_dtype(DLDataType dtype);

/**
 * @brief Record an event on @p producer_stream and make @p consumer_stream wait on it.
 *
 * No-op when the two streams are identical. Used to order Tensor-producing work on a producer
 * stream before a consumer stream reads the result.
 *
 * @param producer_stream The stream that produced the data.
 * @param consumer_stream The stream that will consume the data.
 * @return Empty on success, or an Error describing the failed CUDA call.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> synchronize_cuda_streams(
    cudaStream_t producer_stream, cudaStream_t consumer_stream);

/**
 * @brief Block the host until @p producer_stream completes prior Tensor-producing work.
 *
 * @param producer_stream The stream to synchronize against.
 * @return Empty on success, or an Error describing the failed CUDA call.
 */
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> synchronize_cuda_stream_for_host(
    cudaStream_t producer_stream);

}  // namespace holoscan
