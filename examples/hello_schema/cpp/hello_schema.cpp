/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke test for installed-SDK consumers of holoscan::schema: round-trips a host Tensor's image
// metadata through the FlatBuffers-backed schema helpers (build_image_metadata ->
// parse_image_metadata). This exercises the install-tree story for the schema DSO: the static
// FlatBuffers runtime is linked PRIVATELY (absorbed into libholoscan_schema.so), but its headers
// are a public *compile* dependency -- image_metadata.hpp includes the flatc-generated
// image_generated.h, which in turn includes "flatbuffers/...". If the schema headers, the generated
// headers, or the flatbuffers headers are not installed where find_package(holoscan) can find them,
// this fails to configure/compile against the install tree -- which is what we want CI to catch.
// Uses host (kDLCPU) data only, so it needs no CUDA device.

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <utility>
#include <vector>

int main() {
  // A 3x2 (height x width) host tensor; only the image metadata is encoded, not the tensor bytes.
  std::vector<float> data(6, 0.0F);

  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = {3, 2};
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0;

  const holoscan::Tensor tensor(std::move(context));

  const holoscan::schema::ImageMetadataOptions options{
      .height = 3,
      .width = 2,
      .encoding = holoscan::schema::ImageEncoding_MONO32F,
      .capture_timestamp_ns = 987654321,
      .frame_id = "hello_schema_frame"};

  const auto metadata = holoscan::schema::build_image_metadata(tensor, options);
  if (!metadata) {
    std::cerr << "holoscan schema sample FAILED: build_image_metadata: " << metadata.error().message
              << "\n";
    return 1;
  }

  const auto parsed = holoscan::schema::parse_image_metadata(
      std::span<const std::byte>(metadata->data(), metadata->size()));
  if (!parsed) {
    std::cerr << "holoscan schema sample FAILED: parse_image_metadata: " << parsed.error().message
              << "\n";
    return 1;
  }

  const holoscan::schema::Image* image = parsed->image;
  if (image == nullptr || image->height() != 3 || image->width() != 2 ||
      image->encoding() != holoscan::schema::ImageEncoding_MONO32F) {
    std::cerr << "holoscan schema sample FAILED: unexpected metadata round-trip\n";
    return 1;
  }

  std::cout << "image metadata height=" << image->height() << " width=" << image->width()
            << " encoding=" << static_cast<int>(image->encoding()) << "\n";
  std::cout << "holoscan schema sample complete\n";
  return 0;
}
