/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <type_traits>
#include <utility>

#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

// REQ: SENSOR-IO-011
// REQ: OPERATOR-056
static_assert(std::is_trivially_copyable_v<holoscan::SelectedSampleView<holoscan::schema::ImageT>>);
static_assert(!std::is_copy_constructible_v<holoscan::sensor_io::ImageDescriptorScratch>);

int main() {
  auto created = holoscan::sensor_io::ImageDescriptorScratch::create();
  if (!created)
    return 1;
  auto scratch = std::move(*created);
  holoscan::schema::ImageT image;
  image.width = 4;
  if (!scratch.copy_metadata_from(image) || scratch.image().width != 4 || scratch.image().data) {
    return 2;
  }
  scratch.reset();
  return scratch.image().width == 0 ? 0 : 3;
}
