/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <exception>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "external_topic_common.hpp"

namespace {

using descriptor_tensor_external_topic::Fields;
using descriptor_tensor_external_topic::Frame;

// Relay descriptor payloads while validating both named Tensor fields for the Python peer test.
// Representative test: test_descriptor_tensor_topic_interoperates_with_cpp_peer.
class DescriptorTensorRelay final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& input_builder = spec.input(input, "input").queue_depth(2U);
    input_builder.tensor_field(Fields::pixels)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = descriptor_tensor_external_topic::kPixelsRepresentation});
    input_builder.tensor_field(Fields::calibration)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = descriptor_tensor_external_topic::kCalibrationRepresentation});

    auto& output_builder = spec.output(output, "output").max_emits_per_compute(1U);
    output_builder.tensor_field(
        Fields::pixels,
        holoscan::TensorOutputSpec{
            .representation = descriptor_tensor_external_topic::kPixelsRepresentation,
            .bounds = holoscan::tensor_bounds(descriptor_tensor_external_topic::kPixelsShape),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
            .transfer =
                holoscan::TensorTransferPolicy{
                    .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
    output_builder.tensor_field(
        Fields::calibration,
        holoscan::TensorOutputSpec{
            .representation = descriptor_tensor_external_topic::kCalibrationRepresentation,
            .bounds = holoscan::tensor_bounds(descriptor_tensor_external_topic::kCalibrationShape),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
            .transfer =
                holoscan::TensorTransferPolicy{
                    .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    if (sample->data.sequence == descriptor_tensor_external_topic::kAckSequence) {
      context.request_stop();
      return {};
    }
    if (sample->data.sequence != descriptor_tensor_external_topic::kDataSequence ||
        sample->data.source != descriptor_tensor_external_topic::kClientSource ||
        sample->data.pixels == nullptr || sample->data.calibration == nullptr) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kProtocolError, "invalid descriptor Tensor input"});
    }
    const std::array tensors{*sample->data.pixels, *sample->data.calibration};
    if (!descriptor_tensor_external_topic::tensors_match(tensors)) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kProtocolError, "invalid descriptor Tensor input"});
    }
    sample->data.source = descriptor_tensor_external_topic::kGraphSource;
    auto emitted = output.emit(std::move(sample->data));
    if (!emitted) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    return {};
  }

  holoscan::Input<Frame> input;
  holoscan::Output<Frame> output;
};

}  // namespace

#if defined(_WIN32)
#define HOLOSCAN_PYTHON_TEST_EXPORT __declspec(dllexport)
#else
#define HOLOSCAN_PYTHON_TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" HOLOSCAN_PYTHON_TEST_EXPORT const char*
holoscan_python_register_descriptor_topic_fixture() noexcept {
  static std::once_flag once;
  static std::string error;
  std::call_once(once, [] {
    try {
      auto& registry = holoscan::authoring::authoring_registry();
      constexpr auto identity = holoscan::schema_identity<Frame>();
      auto payload = registry.register_payload(
          holoscan::authoring::detail::make_registered_payload<Frame>(identity.canonical_name));
      if (!payload) {
        throw holoscan::RuntimeError{payload.error()};
      }
      auto relay = registry.register_operator(
          holoscan::authoring::detail::make_registered_operator<DescriptorTensorRelay>(
              "holoscan.test.DescriptorTensorRelay"));
      if (!relay) {
        throw holoscan::RuntimeError{relay.error()};
      }
    } catch (const std::exception& exception) {
      error = exception.what();
    }
  });
  return error.empty() ? nullptr : error.c_str();
}
