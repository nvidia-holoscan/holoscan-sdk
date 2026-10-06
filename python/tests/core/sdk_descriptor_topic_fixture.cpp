/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <exception>
#include <mutex>
#include <string>

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/schema/sdk_schemas.hpp>

namespace {

// Provides an SDK TensorPacket edge for
// test_sdk_descriptor_tensor_topic_records_named_field_contracts.
class SdkTensorPacketSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .tensor_field(holoscan::schema::TensorPacketTensorFields::data)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 2U,
                               .strides = holoscan::TensorStridePolicy::kContiguousOnly}});
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<holoscan::schema::TensorPacketT> input{"input"};
};

}  // namespace

#if defined(_WIN32)
#define HOLOSCAN_PYTHON_TEST_EXPORT __declspec(dllexport)
#else
#define HOLOSCAN_PYTHON_TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" HOLOSCAN_PYTHON_TEST_EXPORT const char*
holoscan_python_register_sdk_descriptor_topic_fixture() noexcept {
  static std::once_flag once;
  static std::string error;
  std::call_once(once, [] {
    try {
      auto row = holoscan::authoring::authoring_registry().register_operator(
          holoscan::authoring::detail::make_registered_operator<SdkTensorPacketSink>(
              "holoscan.test.SdkTensorPacketSink"));
      if (!row) {
        throw holoscan::RuntimeError{row.error()};
      }
    } catch (const std::exception& exception) {
      error = exception.what();
    }
  });
  return error.empty() ? nullptr : error.c_str();
}
