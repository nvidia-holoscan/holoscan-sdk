// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include <nanobind/nanobind.h>

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "byte_payload_schema_traits.hpp"

namespace {

holoscan::test::BytePayloadT configured_payload;

// Configure before graph construction; every source owns its native payload throughout the run.
class ByteSource final : public holoscan::Operator<> {
 public:
  ByteSource() : payload_(configured_payload) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(out_, "out"); }

  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::microseconds{10}});
    return result;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto result = out_.emit(payload_);
    if (!result && result.error().code != holoscan::ErrorCode::kBackpressured) {
      return result;
    }
    return {};
  }

 private:
  holoscan::test::BytePayloadT payload_;
  holoscan::Output<holoscan::test::BytePayloadT> out_;
};

}  // namespace

NB_MODULE(byte_schema_support, module) {
  auto registered = holoscan::authoring::authoring_registry().register_operator(
      holoscan::authoring::detail::make_registered_operator<ByteSource>(
          "holoscan.test.ByteSource"));
  if (!registered) {
    throw std::runtime_error("byte schema source registration failed");
  }
  module.def("configure", [](std::size_t size) {
    configured_payload = {};
    configured_payload.blob.resize(size);
    for (std::size_t index = 0; index < size; ++index) {
      configured_payload.blob[index] = static_cast<std::uint8_t>(index % 256U);
    }
    configured_payload.numbers = {1U, 128U, 65536U};
  });
}
