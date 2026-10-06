/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace {
struct ResizeParams {
  std::uint32_t width{6U};
  std::uint32_t height{4U};
};
HOLOSCAN_PARAM_SCHEMA(ResizeParams,
                      holoscan::param_schema("com.nvidia.holoscan.example.TestResizeParams", 1),
                      holoscan::field<&ResizeParams::width>("width").range(1U, 16U),
                      holoscan::field<&ResizeParams::height>("height").range(1U, 16U));

// A small host RGB operator; its production callbacks contain no harness-specific code.
class ImageResizeOp final : public holoscan::Operator<ResizeParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    const std::array<std::int64_t, 3> maximum_shape{16, 16, 3};
    spec.input(input, "input")
        .expects_tensor(
            holoscan::TensorInputSpec{.representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                                         .dtype = DLDataType{kDLUInt, 8, 1},
                                                         .rank = 3U},
                                      .bounds = holoscan::tensor_bounds(maximum_shape)});
    spec.output(output, "output")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8, 1},
                               .rank = 3U},
            .bounds = holoscan::tensor_bounds(16U * 16U * 3U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = input.receive();
    if (!received)
      return holoscan::make_unexpected(std::move(received).error());
    const auto& image = received->data;
    if (!image.data || image.encoding != holoscan::schema::ImageEncoding_RGB8 ||
        image.width == 0U || image.height == 0U || image.width > 16U || image.height > 16U ||
        image.data->shape() != std::vector<std::int64_t>{image.height, image.width, 3}) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                       "expected a bounded interleaved RGB image"});
    }
    std::vector<std::uint8_t> source(image.width * image.height * 3U);
    auto copied = image.data->copy_to_host(source.data(), source.size());
    if (!copied)
      return copied;
    const std::array<std::int64_t, 3> shape{params().height, params().width, 3};
    auto loan = output.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = shape, .dtype = {kDLUInt, 8, 1}});
    if (!loan)
      return holoscan::make_unexpected(std::move(loan).error());
    auto filled = loan->write_host_and_commit([&](const auto& write) {
      auto* destination = write.template data_as<std::uint8_t>();
      for (std::uint32_t y = 0U; y < params().height; ++y) {
        for (std::uint32_t x = 0U; x < params().width; ++x) {
          const auto sy = y * image.height / params().height;
          const auto sx = x * image.width / params().width;
          for (std::uint32_t channel = 0U; channel < 3U; ++channel) {
            destination[(y * params().width + x) * 3U + channel] =
                source[(sy * image.width + sx) * 3U + channel];
          }
        }
      }
    });
    if (!filled)
      return filled;
    auto metadata = image;
    metadata.width = static_cast<std::int32_t>(params().width);
    metadata.height = static_cast<std::int32_t>(params().height);
    return output.emit_tensor(std::move(*loan), metadata);
  }
  holoscan::Input<holoscan::schema::ImageT> input;
  holoscan::Output<holoscan::schema::ImageT> output;
};

std::shared_ptr<holoscan::Tensor> make_pixels() {
  struct Storage {
    DLManagedTensor managed{};
    std::array<std::int64_t, 3> shape{2, 3, 3};
    std::array<std::uint8_t, 18> bytes{};
  };
  auto storage = std::make_unique<Storage>();
  for (std::size_t i = 0; i < storage->bytes.size(); ++i)
    storage->bytes[i] = static_cast<std::uint8_t>(i);
  storage->managed.dl_tensor = {.data = storage->bytes.data(),
                                .device = {kDLCPU, 0},
                                .ndim = 3,
                                .dtype = {kDLUInt, 8, 1},
                                .shape = storage->shape.data(),
                                .strides = nullptr,
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* value) {
    delete static_cast<Storage*>(value->manager_ctx);
  };
  auto* managed = &storage.release()->managed;
  return std::make_shared<holoscan::Tensor>(managed);
}
}  // namespace

int main() {
  using Image = holoscan::schema::ImageT;
  Image image;
  image.width = 3U;
  image.height = 2U;
  image.encoding = holoscan::schema::ImageEncoding_RGB8;
  image.header = std::make_shared<holoscan::schema::HeaderT>();
  image.header->frame_id = "camera-0";
  image.exposure_time_ns = 2'000'000;
  image.gain = 1.5F;
  image.data = make_pixels();
  auto h = holoscan::testing::create_operator_test<ImageResizeOp>();
  h.set_params(ResizeParams{.width = 6U, .height = 4U});
  h.push("input", image);
  h.capture_snapshot<Image>("output");
  const auto result = h.run();
  if (!result.ok()) {
    std::cerr << result.describe_failure() << '\n';
    return 1;
  }
  const auto snapshot = result.snapshot<Image>("output");
  const auto metadata = snapshot.metadata();
  if (metadata.data || metadata.width != 6U || metadata.height != 4U || !metadata.header ||
      metadata.header->frame_id != "camera-0" || metadata.exposure_time_ns != 2'000'000 ||
      metadata.gain != 1.5F)
    return 1;
  const auto& content = snapshot.tensor(holoscan::schema::ImageTensorFields::data);
  if (content.shape() != std::vector<std::int64_t>{4, 6, 3} || content.bytes().size() != 72U)
    return 1;
  for (std::size_t y = 0; y < 4; ++y) {
    for (std::size_t x = 0; x < 6; ++x) {
      for (std::size_t c = 0; c < 3; ++c) {
        const auto expected = static_cast<std::byte>(((y / 2) * 3 + x / 2) * 3 + c);
        if (content.bytes()[(y * 6 + x) * 3 + c] != expected)
          return 1;
      }
    }
  }
  std::cout << "image snapshot example complete: resize pixels and preserved metadata verified\n";
}
