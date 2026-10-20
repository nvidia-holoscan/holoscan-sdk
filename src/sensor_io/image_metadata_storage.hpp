/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::sensor_io::detail {

// Cold-only shared preparation. The receive slot also prepares a Tensor wrapper; scratch never
// does. Optional presence during decode and scratch assignment deliberately remain independent.
inline void prepare_image_metadata(schema::ImageT& image) {
  image.header = std::make_shared<schema::HeaderT>();
  image.header->stamp = std::make_shared<schema::TimeT>();
  image.header->frame_id.reserve(schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>);
  image.plane_layouts.reserve(schema::kMaximumImagePlanes);
}

inline void copy_image_scalar_metadata(schema::ImageT& destination,
                                       const schema::ImageT& source) noexcept {
  destination.height = source.height;
  destination.width = source.width;
  destination.encoding = source.encoding;
  destination.color_space = source.color_space;
  destination.significant_bits = source.significant_bits;
  destination.roi_offset_x = source.roi_offset_x;
  destination.roi_offset_y = source.roi_offset_y;
  destination.exposure_time_ns = source.exposure_time_ns;
  destination.gain = source.gain;
}

}  // namespace holoscan::sensor_io::detail
