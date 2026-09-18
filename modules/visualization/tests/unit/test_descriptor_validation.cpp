// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Descriptor binds must enforce the pipeline's reflected binding contract before any
// native descriptor write: an unknown binding index, a bind method that disagrees with
// the reflected kind, and a resource owned by another device are all rejected with
// kInvalidArgument, while matching binds keep succeeding.

#include <string>
#include <utility>
#include <vector>

#include <holoviz/holoviz.hpp>
#include <holoviz/slang_compiler.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz {
namespace {

// One binding of each bindable kind, so every bind method has a matching slot and two
// mismatched ones.
constexpr const char* kShaderSource = R"(
StructuredBuffer<float> values;
Texture2D<float4> image;
SamplerState image_sampler;

struct VSInput {
  float3 position;
};

struct VSOutput {
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(VSInput input) {
  VSOutput output;
  output.position = float4(input.position, 1.0);
  return output;
}

[shader("fragment")]
float4 fragment_main(VSOutput input) : SV_Target {
  return image.Sample(image_sampler, float2(0.5, 0.5)) * values[0];
}
)";

class DescriptorValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto adapters = enumerate_adapters();
    if (!adapters || adapters->empty()) {
      HOLOVIZ_ENV_SKIP() << "No HoloViz adapter available";
    }
    auto device = (*adapters)[0].create_device({});
    if (!device) {
      HOLOVIZ_ENV_SKIP() << "Device creation failed: " << device.error().message;
    }
    device_ = std::move(*device);

    auto blob = compile_slang_shader(
        device_.backend().value(), "vertex_main", "fragment_main", kShaderSource);
    ASSERT_TRUE(blob.has_value()) << blob.error().message;
    for (const auto& binding : blob->bindings) {
      if (binding.name == "values") {
        buffer_binding_ = binding.index;
      } else if (binding.name == "image") {
        image_binding_ = binding.index;
      } else if (binding.name == "image_sampler") {
        sampler_binding_ = binding.index;
      }
    }

    auto pipeline = device_.create_pipeline(
        {.shader_blob = *blob, .topology = PrimitiveTopology::kTriangleList});
    ASSERT_TRUE(pipeline.has_value()) << pipeline.error().message;
    pipeline_ = std::move(*pipeline);

    auto descriptor = pipeline_.create_descriptor();
    ASSERT_TRUE(descriptor.has_value()) << descriptor.error().message;
    descriptor_ = std::move(*descriptor);

    auto buffer = device_.create_buffer({.size_bytes = 256, .share = ShareMode::kPrivate});
    ASSERT_TRUE(buffer.has_value()) << buffer.error().message;
    buffer_ = std::move(*buffer);

    auto image = device_.create_image({.extent = {4, 4},
                                       .format = Format::kR8G8B8A8Unorm,
                                       .usage = ImageUsage::kSampled,
                                       .share = ShareMode::kPrivate});
    ASSERT_TRUE(image.has_value()) << image.error().message;
    image_ = std::move(*image);
  }

  // Members are destroyed in reverse declaration order, so resources and the
  // descriptor/pipeline unwind before the device.
  Device device_;
  Pipeline pipeline_;
  Descriptor descriptor_;
  Buffer buffer_;
  Image image_;
  uint32_t buffer_binding_ = 0;
  uint32_t image_binding_ = 0;
  uint32_t sampler_binding_ = 0;
};

TEST_F(DescriptorValidationTest, MatchingBindsSucceed) {
  EXPECT_TRUE(descriptor_.bind_buffer(BufferSlice(buffer_, 0, 256), buffer_binding_).has_value());
  EXPECT_TRUE(descriptor_.bind_image(image_, image_binding_).has_value());
  EXPECT_TRUE(descriptor_.bind_sampler(device_, {}, sampler_binding_).has_value());
}

TEST_F(DescriptorValidationTest, UnknownBindingIndexIsRejected) {
  const auto bound = descriptor_.bind_buffer(BufferSlice(buffer_, 0, 256), 42);
  ASSERT_FALSE(bound.has_value());
  EXPECT_EQ(bound.error().code, ErrorCode::kInvalidArgument);
}

TEST_F(DescriptorValidationTest, KindMismatchIsRejected) {
  const auto buffer_at_image =
      descriptor_.bind_buffer(BufferSlice(buffer_, 0, 256), image_binding_);
  ASSERT_FALSE(buffer_at_image.has_value());
  EXPECT_EQ(buffer_at_image.error().code, ErrorCode::kInvalidArgument);

  const auto image_at_buffer = descriptor_.bind_image(image_, buffer_binding_);
  ASSERT_FALSE(image_at_buffer.has_value());
  EXPECT_EQ(image_at_buffer.error().code, ErrorCode::kInvalidArgument);

  const auto sampler_at_buffer = descriptor_.bind_sampler(device_, {}, buffer_binding_);
  ASSERT_FALSE(sampler_at_buffer.has_value());
  EXPECT_EQ(sampler_at_buffer.error().code, ErrorCode::kInvalidArgument);
}

TEST_F(DescriptorValidationTest, RenderTargetImageIsRejected) {
  auto rt_image = device_.create_image({.extent = {4, 4},
                                        .format = Format::kR8G8B8A8Unorm,
                                        .usage = ImageUsage::kRenderTarget,
                                        .share = ShareMode::kPrivate});
  ASSERT_TRUE(rt_image.has_value()) << rt_image.error().message;

  const auto bound = descriptor_.bind_image(*rt_image, image_binding_);
  ASSERT_FALSE(bound.has_value());
  EXPECT_EQ(bound.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(bound.error().message.find("kSampled"), std::string::npos) << bound.error().message;
}

TEST_F(DescriptorValidationTest, CrossDeviceResourceIsRejected) {
  auto adapters = enumerate_adapters();
  ASSERT_TRUE(adapters.has_value()) << adapters.error().message;
  auto other_device = (*adapters)[0].create_device({});
  if (!other_device) {
    HOLOVIZ_ENV_SKIP() << "second device creation failed: " << other_device.error().message;
  }

  auto other_buffer =
      other_device->create_buffer({.size_bytes = 256, .share = ShareMode::kPrivate});
  ASSERT_TRUE(other_buffer.has_value()) << other_buffer.error().message;
  const auto bound_buffer =
      descriptor_.bind_buffer(BufferSlice(*other_buffer, 0, 256), buffer_binding_);
  ASSERT_FALSE(bound_buffer.has_value());
  EXPECT_EQ(bound_buffer.error().code, ErrorCode::kInvalidArgument);

  auto other_image = other_device->create_image({.extent = {4, 4},
                                                 .format = Format::kR8G8B8A8Unorm,
                                                 .usage = ImageUsage::kSampled,
                                                 .share = ShareMode::kPrivate});
  ASSERT_TRUE(other_image.has_value()) << other_image.error().message;
  const auto bound_image = descriptor_.bind_image(*other_image, image_binding_);
  ASSERT_FALSE(bound_image.has_value());
  EXPECT_EQ(bound_image.error().code, ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace holoviz
