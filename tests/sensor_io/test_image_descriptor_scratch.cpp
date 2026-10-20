/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "common/allocation_counter.hpp"

namespace holoscan::sensor_io {
namespace {

// REQ: SENSOR-IO-011
static_assert(!std::is_copy_constructible_v<ImageDescriptorScratch>);
static_assert(!std::is_copy_assignable_v<ImageDescriptorScratch>);
static_assert(std::is_nothrow_move_constructible_v<ImageDescriptorScratch>);
static_assert(std::is_nothrow_move_assignable_v<ImageDescriptorScratch>);

schema::ImageT populated_image() {
  schema::ImageT source;
  source.header = std::make_shared<schema::HeaderT>();
  source.header->stamp = std::make_shared<schema::TimeT>();
  source.header->frame_id.assign(128U, 'c');
  source.header->capture_timestamp_ns = 123;
  source.header->device_sequence = 456;
  source.header->stamp->time_ns = 789;
  source.header->stamp->acqtime_ns = 1011;
  source.data = std::make_shared<Tensor>();
  source.width = 4;
  source.height = 8;
  source.encoding = schema::ImageEncoding_R8_G8_B8;
  source.color_space = schema::ColorSpace_REC2020_PQ;
  source.significant_bits = 7;
  source.roi_offset_x = 3;
  source.roi_offset_y = 5;
  source.exposure_time_ns = 12;
  source.gain = 2.5F;
  source.plane_layouts.emplace_back(0, 4, 32);
  source.plane_layouts.emplace_back(32, 4, 32);
  source.plane_layouts.emplace_back(64, 4, 32);
  return source;
}

// REQ: SENSOR-IO-011
TEST(ImageDescriptorScratch, CopiesEveryMetadataFieldWithoutAllocationOrTensorRetention) {
  auto scratch = ImageDescriptorScratch::create();
  ASSERT_TRUE(scratch);
  auto source = populated_image();
  const std::weak_ptr<Tensor> input_tensor = source.data;
  bool copied = true;
  std::size_t allocations{};
  {
    const test::ScopedAllocationCounter counter;
    for (int iteration = 0; iteration < 100; ++iteration) {
      copied = scratch->copy_metadata_from(source).has_value() && copied;
    }
    allocations = counter.count();
  }
  ASSERT_TRUE(copied);
  EXPECT_EQ(allocations, 0U);
  const auto& image = std::as_const(*scratch).image();
  EXPECT_EQ(image.data, nullptr);
  ASSERT_NE(image.header, nullptr);
  ASSERT_NE(image.header->stamp, nullptr);
  EXPECT_NE(image.header, source.header);
  EXPECT_NE(image.header->stamp, source.header->stamp);
  EXPECT_EQ(image.header->frame_id, source.header->frame_id);
  EXPECT_EQ(image.header->capture_timestamp_ns, 123);
  EXPECT_EQ(image.header->device_sequence, 456U);
  EXPECT_EQ(image.header->stamp->time_ns, 789);
  EXPECT_EQ(image.header->stamp->acqtime_ns, 1011);
  EXPECT_EQ(image.width, 4);
  EXPECT_EQ(image.height, 8);
  EXPECT_EQ(image.encoding, source.encoding);
  EXPECT_EQ(image.color_space, source.color_space);
  EXPECT_EQ(image.significant_bits, 7U);
  EXPECT_EQ(image.roi_offset_x, 3);
  EXPECT_EQ(image.roi_offset_y, 5);
  EXPECT_EQ(image.exposure_time_ns, 12);
  EXPECT_EQ(image.gain, 2.5F);
  ASSERT_EQ(image.plane_layouts.size(), 3U);
  for (std::size_t plane = 0; plane < 3; ++plane) {
    EXPECT_EQ(image.plane_layouts[plane].offset_bytes(), plane * 32);
    EXPECT_EQ(image.plane_layouts[plane].row_stride_bytes(), 4U);
    EXPECT_EQ(image.plane_layouts[plane].size_bytes(), 32U);
  }
  scratch->image().header->frame_id[0] = 'd';
  EXPECT_EQ(source.header->frame_id[0], 'c');
  source.data.reset();
  EXPECT_TRUE(input_tensor.expired());
}

// REQ: SENSOR-IO-011
TEST(ImageDescriptorScratch, OptionalTransitionsResetAndMovesPreservePreparedCapacity) {
  auto created = ImageDescriptorScratch::create();
  ASSERT_TRUE(created);
  auto scratch = std::move(*created);
  auto other = ImageDescriptorScratch::create();
  ASSERT_TRUE(other);
  *other = std::move(scratch);
  auto source = populated_image();
  auto absent_stamp = populated_image();
  absent_stamp.header->stamp.reset();
  const schema::ImageT absent_header;
  bool valid = true;
  std::size_t allocations{};
  {
    const test::ScopedAllocationCounter counter;
    valid = other->copy_metadata_from(source).has_value();
    const auto* header = other->image().header.get();
    const auto* stamp = other->image().header->stamp.get();
    for (int iteration = 0; iteration < 100; ++iteration) {
      valid = other->copy_metadata_from(absent_stamp).has_value() && valid;
      valid = other->image().header && !other->image().header->stamp && valid;
      valid = other->copy_metadata_from(absent_header).has_value() && valid;
      valid = !other->image().header && other->image().plane_layouts.empty() && valid;
      other->reset();
      valid = !other->image().header && other->image().width == 0 && valid;
      valid = other->copy_metadata_from(source).has_value() && valid;
      valid = other->image().header.get() == header &&
              other->image().header->stamp.get() == stamp && !other->image().data && valid;
    }
    allocations = counter.count();
  }
  EXPECT_TRUE(valid);
  EXPECT_EQ(allocations, 0U);
}

// REQ: SENSOR-IO-011
TEST(ImageDescriptorScratch, BoundsRejectBeforeMutationWithoutAllocation) {
  auto scratch = ImageDescriptorScratch::create();
  ASSERT_TRUE(scratch);
  auto source = populated_image();
  source.header->frame_id.assign(schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>,
                                 'x');
  source.plane_layouts.emplace_back(96, 4, 32);
  ASSERT_EQ(source.plane_layouts.size(), schema::kMaximumImagePlanes);
  ASSERT_TRUE(scratch->copy_metadata_from(source));
  auto too_many_planes = populated_image();
  too_many_planes.width = 99;
  too_many_planes.plane_layouts.resize(schema::kMaximumImagePlanes + 1);
  auto too_long_frame = populated_image();
  too_long_frame.width = 99;
  too_long_frame.header->frame_id.resize(source.header->frame_id.size() + 1);
  bool rejected{};
  std::size_t allocations{};
  {
    const test::ScopedAllocationCounter counter;
    const auto planes = scratch->copy_metadata_from(too_many_planes);
    const auto frame = scratch->copy_metadata_from(too_long_frame);
    rejected = !planes && !frame && planes.error().code == ErrorCode::kResourceExhausted &&
               frame.error().code == ErrorCode::kResourceExhausted;
    allocations = counter.count();
  }
  EXPECT_TRUE(rejected);
  EXPECT_EQ(allocations, 0U);
  EXPECT_EQ(scratch->image().width, source.width);
  EXPECT_EQ(scratch->image().header->frame_id, source.header->frame_id);
  EXPECT_EQ(scratch->image().plane_layouts.size(), source.plane_layouts.size());
}

// REQ: SENSOR-IO-011
TEST(ImageDescriptorScratch, CreationAllocationFailuresAreTypedAndRecoverable) {
  for (std::size_t allocation = 0; allocation < 4; ++allocation) {
    bool exhausted{};
    {
      const test::ScopedAllocationFailure failure{allocation};
      const auto scratch = ImageDescriptorScratch::create();
      exhausted = !scratch && scratch.error().code == ErrorCode::kResourceExhausted;
    }
    EXPECT_TRUE(exhausted) << allocation;
    EXPECT_TRUE(ImageDescriptorScratch::create());
  }
}

}  // namespace
}  // namespace holoscan::sensor_io
