/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/holoscan_schema_export.h"
#include "holoscan/schema/descriptor_flatbuffer_payload.hpp"

#include "image_generated.h"

namespace holoscan::schema {

/** @brief Options encoded into serialized Image metadata. */
struct ImageMetadataOptions {
  /// Image height in pixels.
  int32_t height = 0;
  /// Image width in pixels.
  int32_t width = 0;
  /// Image pixel encoding.
  ImageEncoding encoding = ImageEncoding_CUSTOM;
  /// Presentation or source timestamp in nanoseconds.
  int64_t time_ns = 0;
  /// Acquisition timestamp in nanoseconds.
  int64_t acqtime_ns = 0;
  /// Producer-defined frame identifier.
  std::string frame_id;
};

/** @brief Non-owning view of parsed Image metadata.
 * @warning The image pointer remains valid only while the source metadata bytes remain alive and
 * unmodified. */
struct ImageMetadataView {
  /// Parsed Image root pointing into caller-owned metadata storage.
  const Image* image = nullptr;
};

/** @brief Tensor view information extracted from serialized Image metadata. */
struct ImageTensorMetadata {
  /// Maximum Image Tensor rank represented inline.
  static constexpr std::size_t kMaxRank = 8;

  /// Tensor rank.
  std::uint32_t rank = 0;
  /// DLPack element type.
  ::DLDataType dtype{};
  /// DLPack device type and ordinal.
  ::DLDevice device{};
  /// Whether dlpack_strides contains explicit element strides.
  bool has_explicit_strides = false;
  /// Logical Tensor byte size.
  std::uint64_t byte_size = 0;
  /// Tensor shape; only entries below rank are valid.
  std::array<int64_t, kMaxRank> shape{};
  /// DLPack element strides; only entries below rank are valid when has_explicit_strides is true.
  std::array<int64_t, kMaxRank> dlpack_strides{};
};

/// @brief Serialize bounded descriptor-backed Image metadata for a Tensor.
/// @param tensor Tensor whose shape, type, device, and strides are encoded.
/// @param options Image dimensions, encoding, timestamps, and frame identifier.
/// @return Serialized FlatBuffer bytes on success, or Error for incompatible Tensor metadata.
/// @note Tensor data bytes are never serialized by this helper, including while a
/// \ref WithTensorMaterialization or \ref WithTensorGather scope is active. The returned metadata
/// is bounded by \c flatbuffer_default_descriptor_metadata_capacity_v<Image>; an oversized frame
/// identifier or future metadata returns \c ErrorCode::kResourceExhausted.
[[nodiscard]] HOLOSCAN_SCHEMA_EXPORT expected<std::vector<std::byte>, Error> build_image_metadata(
    const holoscan::Tensor& tensor, const ImageMetadataOptions& options);

/// @brief Parse serialized Image metadata without copying its storage.
/// @param metadata Serialized FlatBuffer metadata bytes.
/// @return Non-owning Image view on success, or Error for empty or invalid metadata.
[[nodiscard]] HOLOSCAN_SCHEMA_EXPORT expected<ImageMetadataView, Error> parse_image_metadata(
    std::span<const std::byte> metadata);

/// @brief Extract descriptor-backed Tensor metadata from a parsed Image view.
/// @param view Parsed Image metadata view.
/// @return Tensor metadata on success, or Error for missing, invalid, overflowing, or embedded-data
/// fields.
/// @note This helper accepts metadata-only Image buffers. A nonempty Tensor data vector is rejected
/// so descriptor-backed paths cannot silently carry a second materialized payload representation.
[[nodiscard]] HOLOSCAN_SCHEMA_EXPORT expected<ImageTensorMetadata, Error> image_tensor_metadata(
    const ImageMetadataView& view);

/// @brief Decode the generated Image Tensor member through the common descriptor adapter.
template <>
struct DescriptorFlatBufferTensorViewTraits<Image>
    : DescriptorFlatBufferTensorViewAdapter<Image, &Image::data> {};

}  // namespace holoscan::schema
