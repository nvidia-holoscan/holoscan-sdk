/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "common/allocation_counter.hpp"

namespace holoscan {
namespace {

// Isolate the prepared codec operations; the public graph suite separately covers publication.
// REQ: SENSOR-IO-011
TEST(ImageScratchCodec, SeparateTensorAuthorityAndPreparedDecodeAreAllocationFree) {
  using Payload = schema::DescriptorFlatBufferPayload<
      schema::Image,
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>>;
  struct Envelope {
    SampleMetadata metadata{};
    Payload payload{};
  };
  std::array<std::uint64_t, 1> pixels{42};
  std::array<std::int64_t, 1> shape{1};
  DLManagedTensor managed{};
  managed.dl_tensor = {.data = pixels.data(),
                       .device = {kDLCPU, 0},
                       .ndim = 1,
                       .dtype = {kDLUInt, 64, 1},
                       .shape = shape.data()};
  managed.deleter = [](DLManagedTensor*) {};
  const Tensor tensor{&managed};
  auto scratch = sensor_io::ImageDescriptorScratch::create();
  ASSERT_TRUE(scratch);
  schema::ImageT source;
  source.width = 1;
  source.height = 1;
  source.header = std::make_shared<schema::HeaderT>();
  source.header->frame_id.assign(128, 'c');
  ASSERT_TRUE(scratch->copy_metadata_from(source));
  ASSERT_EQ(scratch->image().data, nullptr);

  const auto* ops = detail::runtime_port_type_ops<schema::ImageT>();
  alignas(std::max_align_t) std::array<std::byte, sizeof(Envelope)> wire{};
  struct Prepared {
    const detail::RuntimePortTypeOps* ops;
    alignas(Sample<schema::ImageT>) std::array<std::byte, sizeof(Sample<schema::ImageT>)> storage{};
    bool ready{};
    ~Prepared() {
      if (ready)
        ops->destroy_sample(storage.data());
    }
  } prepared{ops};
  ASSERT_TRUE(ops->prepare_sample_storage(prepared.storage.data()));
  prepared.ready = true;
  const SampleMetadata metadata{.sequence = 1, .byte_size = sizeof(pixels)};
  bool valid = true;
  std::size_t encode_allocations{};
  std::size_t decode_allocations{};
  for (int iteration = 0; iteration < 100; ++iteration) {
    {
      const test::ScopedAllocationCounter counter;
      const auto encoded =
          ops->encode_descriptor_tensor_loan(&scratch->image(), tensor, metadata, 1, wire);
      valid = encoded.has_value() && valid;
      encode_allocations += counter.count();
    }
    {
      const test::ScopedAllocationCounter counter;
      const auto assigned =
          ops->assign_descriptor_tensor_sample(prepared.storage.data(),
                                               wire,
                                               tensor,
                                               detail::DescriptorTensorWireMetadata{metadata},
                                               detail::DescriptorTensorSampleMetadata{metadata});
      valid = assigned.has_value() && valid;
      decode_allocations += counter.count();
    }
  }
  ASSERT_TRUE(valid);
  EXPECT_EQ(encode_allocations, 0U);
  EXPECT_EQ(decode_allocations, 0U);
  const auto& received = *reinterpret_cast<const Sample<schema::ImageT>*>(prepared.storage.data());
  ASSERT_NE(received.data.data, nullptr);
  EXPECT_EQ(received.data.data->data(), pixels.data());
  EXPECT_EQ(received.data.header->frame_id, source.header->frame_id);
  EXPECT_EQ(scratch->image().data, nullptr);

  // Each container fits, but the complete FlatBuffer exceeds its bounded metadata envelope.
  source.header->frame_id.resize(schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>);
  ASSERT_TRUE(scratch->copy_metadata_from(source));
  const auto too_large =
      ops->encode_descriptor_tensor_loan(&scratch->image(), tensor, metadata, 1, wire);
  ASSERT_FALSE(too_large);
  EXPECT_EQ(too_large.error().code, ErrorCode::kResourceExhausted);
}

}  // namespace
}  // namespace holoscan
