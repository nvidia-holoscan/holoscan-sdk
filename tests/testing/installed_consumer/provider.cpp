/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/core/execution_context.hpp>
#include "provider.hpp"

namespace {
struct ScaleParams {
  std::int32_t factor{1};
};
HOLOSCAN_PARAM_SCHEMA(ScaleParams, holoscan::param_schema("holoscan.test.InstalledScaleParams", 1),
                      holoscan::field<&ScaleParams::factor>("factor"));
class ScaleOp final : public holoscan::Operator<ScaleParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    sample->data *= params().factor;
    return output.emit(std::move(*sample));
  }
  holoscan::Input<std::int32_t> input;
  holoscan::Output<std::int32_t> output;
};

class ClockScaleOp final : public holoscan::Operator<ScaleParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}})
        .input(input, holoscan::LatestWithin{.max_age = std::chrono::milliseconds{100}})
        .on_missing(input, holoscan::Skip{});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    sample->data *= params().factor;
    return output.emit(std::move(*sample));
  }
  holoscan::Input<std::int32_t> input;
  holoscan::Output<std::int32_t> output;
};

class TensorScaleOp final : public holoscan::Operator<ScaleParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLInt, 32, 1},
                               .rank = 1U},
            .bounds = holoscan::tensor_bounds(sizeof(std::int32_t)),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    const std::array<std::int64_t, 1> shape{1};
    auto loan = output.allocate_tensor(shape);
    if (!loan)
      return holoscan::make_unexpected(std::move(loan).error());
    auto filled = loan->write_host_and_commit([&](const auto& write) {
      write.template data_as<std::int32_t>()[0] =
          sample->data.template data_as<std::int32_t>()[0] * params().factor;
    });
    if (!filled)
      return filled;
    return output.emit(std::move(*loan));
  }
  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;
};
// Image uses Sensor I/O's custom descriptor codec as well as independently generated metadata.
class ImageEchoOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(
            holoscan::TensorInputSpec{.representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                                         .dtype = DLDataType{kDLUInt, 8, 1},
                                                         .rank = 3U}});
    spec.output(output, "output")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8, 1},
                               .rank = 3U},
            .bounds = holoscan::tensor_bounds(6U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    return output.emit(std::move(*sample));
  }
  holoscan::Input<holoscan::schema::ImageT> input;
  holoscan::Output<holoscan::schema::ImageT> output;
};

template <holoscan::PublishedPayload T>
class EchoOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input");
    auto& builder = spec.output(output, "output");
    if constexpr (holoscan::testing::TensorCapturePayload<T>) {
      using Traits = holoscan::DescriptorTensorSchemaNativeTraits<T>;
      for (std::size_t index = 0U; index < Traits::tensor_count; ++index) {
        builder
            .tensor_field(holoscan::TensorFieldSelector<T>{Traits::tensor_field_ids[index],
                                                           static_cast<std::uint16_t>(index)})
            .produces_tensor(holoscan::TensorOutputSpec{
                .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                   .dtype = DLDataType{kDLInt, 32, 1},
                                   .rank = 1U},
                .bounds = holoscan::tensor_bounds(sizeof(std::int32_t)),
                .storage = holoscan::TensorOutputStorage::kRuntimePool});
      }
    }
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    return output.emit(std::move(*sample));
  }
  holoscan::Input<T> input;
  holoscan::Output<T> output;
};
}  // namespace
const holoscan::authoring::RegisteredOperatorType& test_operator_provider() {
  static const auto row = holoscan::authoring::detail::make_registered_operator<ScaleOp>(
      "holoscan.test.InstalledScale");
  return row;
}
const holoscan::authoring::RegisteredOperatorType& test_tensor_operator_provider() {
  static const auto row = holoscan::authoring::detail::make_registered_operator<TensorScaleOp>(
      "holoscan.test.InstalledTensorScale");
  return row;
}

holoscan::testing::StagedInput test_tensor_input(holoscan::Tensor value) {
  return holoscan::testing::staged_values<holoscan::Tensor>({std::move(value)});
}
holoscan::testing::CaptureRequest test_tensor_capture() {
  return holoscan::testing::snapshot_capture<holoscan::Tensor>(4U, 16U);
}

const holoscan::authoring::RegisteredOperatorType& test_clock_operator_provider() {
  static const auto row = holoscan::authoring::detail::make_registered_operator<ClockScaleOp>(
      "holoscan.test.InstalledClockScale");
  return row;
}

const holoscan::authoring::RegisteredOperatorType& test_pod_operator_provider() {
  static const auto row =
      holoscan::authoring::detail::make_registered_operator<EchoOp<InstalledTestPod>>(
          "holoscan.test.InstalledPodEcho");
  return row;
}
const holoscan::authoring::RegisteredOperatorType& test_schema_operator_provider() {
  static const auto row =
      holoscan::authoring::detail::make_registered_operator<EchoOp<holoscan::schema::StringT>>(
          "holoscan.test.InstalledStringEcho");
  return row;
}
const holoscan::authoring::RegisteredOperatorType& test_packet_operator_provider() {
  static const auto row = holoscan::authoring::detail::make_registered_operator<
      EchoOp<holoscan::schema::TensorPacketT>>("holoscan.test.InstalledPacketEcho");
  return row;
}
const holoscan::authoring::RegisteredOperatorType& test_image_operator_provider() {
  static const auto row = holoscan::authoring::detail::make_registered_operator<ImageEchoOp>(
      "holoscan.test.InstalledImageEcho");
  return row;
}
holoscan::testing::StagedInput test_pod_input(InstalledTestPod value) {
  return holoscan::testing::staged_values<InstalledTestPod>({value});
}
holoscan::testing::CaptureRequest test_pod_capture() {
  return holoscan::testing::value_capture<InstalledTestPod>(4U);
}
holoscan::testing::StagedInput test_schema_input(holoscan::schema::StringT value) {
  return holoscan::testing::staged_values<holoscan::schema::StringT>({std::move(value)});
}
holoscan::testing::CaptureRequest test_schema_capture() {
  return holoscan::testing::value_capture<holoscan::schema::StringT>(4U);
}
holoscan::testing::StagedInput test_packet_input(holoscan::schema::TensorPacketT value) {
  return holoscan::testing::staged_values<holoscan::schema::TensorPacketT>({std::move(value)});
}
holoscan::testing::CaptureRequest test_packet_capture() {
  return holoscan::testing::snapshot_capture<holoscan::schema::TensorPacketT>(4U, 16U);
}
holoscan::testing::StagedInput test_image_input(holoscan::schema::ImageT value) {
  return holoscan::testing::staged_values<holoscan::schema::ImageT>({std::move(value)});
}
holoscan::testing::CaptureRequest test_image_capture() {
  return holoscan::testing::snapshot_capture<holoscan::schema::ImageT>(4U, 6U);
}
const std::type_info& test_image_native_type() noexcept {
  return typeid(holoscan::schema::ImageT);
}
const std::type_info& test_pod_native_type() noexcept {
  return typeid(InstalledTestPod);
}
const std::type_info& test_schema_native_type() noexcept {
  return typeid(holoscan::schema::StringT);
}
const std::type_info& test_packet_native_type() noexcept {
  return typeid(holoscan::schema::TensorPacketT);
}

void test_allow_restart(holoscan::testing::NativeTestGraphView& view) {
  holoscan::testing::TestGraphView<ScaleOp> typed{view};
  typed.graph().lifecycle().allow_external_restart(typed.tested_operator());
  view.compile_options().lifecycle_transactions =
      holoscan::LifecycleTransactionOptions{.max_pending_requests = 4U,
                                            .max_pending_transactions = 4U,
                                            .max_receipts = 8U,
                                            .max_fenced_operators = 16U,
                                            .max_candidate_generations = 8U,
                                            .max_events = 64U,
                                            .max_audit_records = 32U,
                                            .max_quiesce_records = 64U,
                                            .max_rollback_records = 64U};
}
