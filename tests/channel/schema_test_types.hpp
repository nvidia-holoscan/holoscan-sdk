/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/schema/inline_flatbuffer_payload.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "custom_control_generated.h"

namespace holoscan {

// Minimal native type with a deliberately incomplete DescriptorTensorSchemaNativeTraits
// specialization. It omits the required embedded-Tensor accessor and therefore must not
// satisfy HasDescriptorTensorSchemaNativeTraits.
struct UnsupportedDescriptorTensorSchemaNative {};

template <>
struct DescriptorTensorSchemaNativeTraits<UnsupportedDescriptorTensorSchemaNative> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity = 512U;
};

namespace channel_schema_test {

using StringPayload = schema::InlineFlatBufferPayloadFor<schema::String>;
using TimePayload = schema::InlineFlatBufferPayloadFor<schema::Time>;
using ImuPayload = schema::InlineFlatBufferPayloadFor<schema::Imu>;
using DescriptorBackedImagePayload = schema::DescriptorFlatBufferPayloadFor<schema::Image>;
using MultiTensorStagingPayload = schema::DescriptorFlatBufferPayload<schema::Image, 512U, 3U>;
using ResolvedImageMetadata = ChannelTensorSchemaMetadata<DescriptorBackedImagePayload>;
using CustomControlPayload = schema::InlineFlatBufferPayloadFor<tests::channel::CustomControl>;

}  // namespace channel_schema_test
}  // namespace holoscan
