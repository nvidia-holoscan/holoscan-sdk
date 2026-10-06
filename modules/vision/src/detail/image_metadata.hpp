// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::vision::detail {

// Backend-independent description of scalar image storage, shared by adapters and metadata rules.
enum class ImageTensorLayout : std::uint8_t {
  kInterleaved,
  kPlanar,
};

struct ImageTensorView {
  static constexpr std::size_t kMaxRank = 4U;

  const void* data{};
  std::array<std::int64_t, kMaxRank> shape{};
  std::array<std::int64_t, kMaxRank> byte_strides{};
  DLDataType dtype{};
  DLDevice device{};
  std::uint32_t rank{};
  std::int64_t batch{};
  std::int64_t height{};
  std::int64_t width{};
  std::int64_t channels{};
  ImageTensorLayout image_layout{ImageTensorLayout::kInterleaved};

  [[nodiscard]] const char* layout() const noexcept {
    if (image_layout == ImageTensorLayout::kPlanar) {
      return rank == 3U ? "CHW" : "NCHW";
    }
    return rank == 3U ? "HWC" : "NHWC";
  }
};

// Describe geometric effects rather than backend operations. Value and layout effects compose
// independently, so a fused operation or a non-CV-CUDA backend can use the same metadata rules.
enum class ImageGeometryChange : std::uint8_t {
  kPreserve,
  kCrop,
  kPad,
  kResize,
  kWarp,
};

struct ImageMetadataTransform {
  ImageGeometryChange geometry{ImageGeometryChange::kPreserve};
  std::int32_t x{};  // crop origin or left padding, in input pixels
  std::int32_t y{};  // crop origin or top padding, in input pixels
  bool values_changed{};
  bool preserves_precision{true};
};

// Validate that the image encoding and plane descriptors agree with the backend's tensor view.
[[nodiscard]] expected<void, Error> validate_image_payload(const schema::ImageT& image,
                                                           const ImageTensorView& view);

// Update descriptor fields for a new tensor view without changing data or acquisition metadata.
// Both views must have positive geometry and nonoverlapping scalar strides validated by the
// backend adapter; dimensions and row strides must fit the Image schema integer fields.
// On success, the updated descriptor agrees with the output view under validate_image_payload().
// The caller publishes the descriptor with the committed output loan through emit_tensor().
[[nodiscard]] expected<void, Error> update_image_metadata(schema::ImageT& image,
                                                          const ImageTensorView& input,
                                                          const ImageTensorView& output,
                                                          ImageMetadataTransform transform = {});

}  // namespace holoscan::vision::detail
