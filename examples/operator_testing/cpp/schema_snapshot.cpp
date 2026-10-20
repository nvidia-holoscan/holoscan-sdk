/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include <cstddef>
#include <cstdint>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "paired_frame_schema_traits.hpp"

namespace {
using Frame = fixture_example::PairedFrameT;
using Fields = fixture_example::PairedFrameTensorFields;
class PairedFrameOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out")
        .tensor_fields(std::array{Fields::first, Fields::second},
                       holoscan::TensorOutputSpec{
                           .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                              .dtype = DLDataType{kDLUInt, 8, 1},
                                              .rank = 1U},
                           .bounds = holoscan::tensor_bounds(4U),
                           .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto frame = in.receive();
    if (!frame)
      return holoscan::make_unexpected(std::move(frame).error());
    return out.emit(std::move(*frame));
  }
  holoscan::Input<Frame> in;
  holoscan::Output<Frame> out;
};
std::shared_ptr<holoscan::Tensor> content(std::uint8_t first) {
  const std::array<std::int64_t, 1> shape{4};
  auto tensor = holoscan::make_tensor(shape,
                                      DLDataType{kDLUInt, 8, 1},
                                      std::nullopt,
                                      holoscan::MemoryStorageType::kSystem,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor)
    throw holoscan::RuntimeError{std::move(tensor).error()};
  auto destination = tensor->mutable_data();
  if (!destination)
    throw holoscan::RuntimeError{std::move(destination).error()};
  auto* bytes = static_cast<std::uint8_t*>(*destination);
  for (std::uint8_t i = 0; i < 4; ++i)
    bytes[i] = first + i;
  return std::make_shared<holoscan::Tensor>(std::move(*tensor));
}
}  // namespace

int main() {
  using namespace holoscan::testing;
  using namespace std::chrono_literals;
  try {
    auto h = create_operator_test<PairedFrameOp>();
    for (std::uint8_t i = 0; i < 2; ++i) {
      Frame frame;
      frame.header = std::make_shared<fixture_example::FrameHeaderT>();
      frame.header->sequence = i;
      frame.header->source = "sample";
      frame.first = content(static_cast<std::uint8_t>(10U + i));
      frame.second = content(static_cast<std::uint8_t>(20U + i));
      h.push_samples<Frame>("in", {sample_at(frame, 1s + i * 1ms)});
      frame.header->source = "caller changed after staging";
    }
    h.capture_snapshot<Frame>("out", 2U, 8U);
    const auto result = h.run();
    if (!result.ok())
      throw std::runtime_error{result.describe_failure()};
    const auto outputs = result.snapshots<Frame>("out");
    if (outputs.size() != 2U)
      throw std::runtime_error{"expected two schema snapshots"};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const auto metadata = outputs[i].metadata();
      if (!metadata.header || metadata.header->sequence != i ||
          metadata.header->source != "sample" || metadata.first || metadata.second)
        throw std::runtime_error{"metadata snapshot mismatch"};
      for (const auto field : {Fields::first, Fields::second}) {
        if (!outputs[i].has_tensor(field))
          throw std::runtime_error{"missing Tensor field"};
        const auto& tensor = outputs[i].tensor(field);
        const auto start = (field.field_id == Fields::first.field_id ? 10U : 20U) + i;
        std::vector<std::byte> expected;
        for (std::size_t j = 0; j < 4U; ++j)
          expected.push_back(std::byte{static_cast<std::uint8_t>(start + j)});
        if (tensor.bytes() != expected)
          throw std::runtime_error{"Tensor field content mismatch"};
      }
    }
    std::cout << "schema snapshots verified: independent metadata and both Tensor fields\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
