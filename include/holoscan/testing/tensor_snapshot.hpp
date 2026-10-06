/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <holoscan/holoscan_testing_export.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/sample.hpp>

namespace holoscan::testing {
namespace detail {
struct TensorSnapshotAccess;
}  // namespace detail

/// @brief Own copied Tensor content and original metadata, independently of Runtime pools.
/// @note This record is host content for inspection, not a native Tensor for publication.
class TensorSnapshot {
 public:
  /// @brief Return independently owned contiguous host bytes.
  [[nodiscard]] const std::vector<std::byte>& bytes() const noexcept { return bytes_; }
  /// @brief Return original logical extents.
  [[nodiscard]] const std::vector<std::int64_t>& shape() const noexcept { return shape_; }
  /// @brief Return original byte strides, including explicit contiguous strides.
  [[nodiscard]] const std::vector<std::int64_t>& source_byte_strides() const noexcept {
    return strides_;
  }
  /// @brief Return original allocation-relative logical byte offset.
  [[nodiscard]] std::uint64_t source_byte_offset() const noexcept { return offset_; }
  /// @brief Return original DLPack device, not the host destination location.
  [[nodiscard]] DLDevice source_device() const noexcept { return device_; }
  /// @brief Return the original element representation.
  [[nodiscard]] DLDataType dtype() const noexcept { return dtype_; }
  /// @brief Return semantic sample metadata copied at collection.
  [[nodiscard]] const SampleMetadata& sample_metadata() const noexcept { return metadata_; }

 private:
  std::vector<std::byte> bytes_;
  std::vector<std::int64_t> shape_;
  std::vector<std::int64_t> strides_;
  std::uint64_t offset_{};
  DLDevice device_{};
  DLDataType dtype_{};
  SampleMetadata metadata_{};
  friend struct detail::TensorSnapshotAccess;
};

namespace detail {
HOLOSCAN_TESTING_EXPORT expected<TensorSnapshot, Error> capture_tensor_snapshot(
    const Tensor& value, SampleMetadata metadata, std::size_t maximum_bytes, cudaStream_t stream);
}  // namespace detail
}  // namespace holoscan::testing
