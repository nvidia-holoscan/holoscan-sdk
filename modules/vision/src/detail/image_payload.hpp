// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "cvcuda_tensor.hpp"

namespace holoscan::vision::detail {

[[nodiscard]] inline bool zero_border(const std::vector<float>& values) {
  for (const float value : values) {
    if (value != 0.F) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline expected<ImageTensorView, Error> describe_image_payload(
    const schema::ImageT& image, TensorOperation operation, std::uint32_t max_batch_size,
    ImageTensorLayout layout = ImageTensorLayout::kInterleaved) {
  if (!image.data) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Vision Image data must contain a Tensor"});
  }
  auto view = describe_image_tensor(*image.data, operation, max_batch_size, layout);
  if (!view) {
    return make_unexpected(std::move(view).error());
  }
  auto valid = validate_image_payload(image, *view);
  if (!valid) {
    return make_unexpected(std::move(valid).error());
  }
  return view;
}

}  // namespace holoscan::vision::detail
