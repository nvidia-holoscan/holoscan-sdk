// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "holoviz_op/image_input.hpp"

namespace holoscan::visualization {
namespace {

// The imported Tensor snapshots shape/strides and borrows the backing until it is destroyed.
holoscan::expected<TensorImageLayout, holoscan::Error> inspect(ImageFormat format,
                                                               std::vector<int64_t> shape,
                                                               std::vector<int64_t> strides = {},
                                                               DLDataType dtype = {kDLUInt, 8, 1}) {
  std::array<uint8_t, 1024> bytes{};
  DLManagedTensor managed{};
  managed.dl_tensor = {bytes.data(),
                       {kDLCPU, 0},
                       static_cast<int32_t>(shape.size()),
                       dtype,
                       shape.data(),
                       strides.empty() ? nullptr : strides.data(),
                       0};
  auto tensor = make_tensor_from_dlpack(&managed);
  if (!tensor) {
    return make_unexpected(tensor.error());
  }
  return image_input_layout(*tensor, ImageLayerDesc{format, 16, 16});
}

TEST(ImageInput, NV12DerivesVisibleHeightAndTightPlaneOffsets) {
  const auto input = inspect(ImageFormat::kNV12, {6, 6}, {10, 1});
  ASSERT_TRUE(input) << input.error().message;
  EXPECT_EQ(input->width, 6);
  EXPECT_EQ(input->height, 4);
  EXPECT_EQ(input->pitch, 10);
  EXPECT_EQ(input->storage_rows, 6);
  EXPECT_EQ(input->layout.plane_count, 2);
  EXPECT_EQ(input->layout.offsets[1], 24);
  EXPECT_EQ(input->layout.bytes, 36);
}

TEST(ImageInput, AcceptsPackedAndPaddedRGBAndYUYV) {
  EXPECT_TRUE(inspect(ImageFormat::kR8G8B8Unorm, {3, 5, 3}));
  EXPECT_TRUE(inspect(ImageFormat::kR8G8B8Unorm, {3, 5, 3}, {19, 3, 1}));
  EXPECT_TRUE(inspect(ImageFormat::kYUYV422, {3, 6, 2}, {19, 2, 1}));
}

// A stride on an extent-1 axis never reaches an address, so Core-accepted tensors such as a
// grayscale {H, W, 1} with a zero channel stride must not be rejected as unpacked.
TEST(ImageInput, IgnoresStridesOnSingletonDimensions) {
  EXPECT_TRUE(inspect(ImageFormat::kR8Unorm, {3, 5, 1}, {5, 1, 0}));
  EXPECT_TRUE(inspect(ImageFormat::kR8Unorm, {3, 5, 1}, {5, 1, 7}));
  EXPECT_TRUE(
      inspect(ImageFormat::kR16Unorm, {3, 5, 1}, {5, 1, 0}, {kDLUInt, 16, 1}));  // element strides
  EXPECT_TRUE(inspect(ImageFormat::kR8G8B8Unorm, {3, 1, 3}, {3, 99, 1}));
  // Extent > 1 keeps the packed requirement.
  EXPECT_FALSE(inspect(ImageFormat::kR8G8B8Unorm, {3, 5, 3}, {15, 3, 0}));
  EXPECT_FALSE(inspect(ImageFormat::kR8Unorm, {3, 5, 1}, {5, 2, 0}));
}

TEST(ImageInput, PreservesTypedAndBytePackedSixteenBitInputs) {
  EXPECT_TRUE(inspect(ImageFormat::kR16Unorm, {3, 5}, {8, 1}, {kDLUInt, 16, 1}));
  EXPECT_TRUE(inspect(ImageFormat::kR16Unorm, {3, 5, 2}));
  EXPECT_TRUE(inspect(ImageFormat::kR16G16B16A16Unorm, {3, 5, 4}, {}, {kDLUInt, 16, 1}));
  EXPECT_TRUE(inspect(ImageFormat::kR16G16B16A16Unorm, {3, 5, 8}));
}

TEST(ImageInput, RejectsInvalidDimensionsTypesAndInnerStrides) {
  EXPECT_FALSE(inspect(ImageFormat::kNV12, {5, 6}));
  EXPECT_FALSE(inspect(ImageFormat::kNV12, {6, 5}));
  EXPECT_FALSE(inspect(ImageFormat::kNV12, {6, 6, 1}));
  EXPECT_FALSE(inspect(ImageFormat::kNV12, {27, 6}));
  EXPECT_FALSE(inspect(ImageFormat::kYUYV422, {3, 5, 2}));
  EXPECT_FALSE(inspect(ImageFormat::kR8G8B8Unorm, {0, 5, 3}));
  EXPECT_FALSE(inspect(ImageFormat::kR8G8B8Unorm, {3, 17, 3}));
  EXPECT_FALSE(inspect(ImageFormat::kR8G8B8Unorm, {3, 5, 4}));
  EXPECT_FALSE(inspect(ImageFormat::kR8G8B8Unorm, {3, 5, 3}, {}, {kDLInt, 8, 1}));
  EXPECT_FALSE(inspect(ImageFormat::kR8G8B8Unorm, {3, 5, 3}, {14, 3, 1}));
  EXPECT_FALSE(inspect(ImageFormat::kR8G8B8Unorm, {3, 5, 3}, {-15, 3, 1}));
  EXPECT_FALSE(inspect(ImageFormat::kR8G8B8Unorm, {3, 5, 3}, {30, 6, 2}));
  EXPECT_FALSE(inspect(ImageFormat::kUnknown, {3, 5, 3}));
}

TEST(ImageLayout, RejectsOverflowAndExcludesLastRowPadding) {
  using ::holoscan::visualization::backends::image_layout;
  using ::holoscan::visualization::backends::image_plane_span;
  EXPECT_EQ(image_plane_span(6, 4, 10), 36);
  EXPECT_FALSE(image_plane_span(6, 4, 5));
  EXPECT_FALSE(image_plane_span(6, 4, std::numeric_limits<size_t>::max()));
  EXPECT_FALSE(image_layout(ImageFormat::kR16G16B16A16Unorm,
                            std::numeric_limits<uint32_t>::max(),
                            std::numeric_limits<uint32_t>::max()));
}

}  // namespace
}  // namespace holoscan::visualization
