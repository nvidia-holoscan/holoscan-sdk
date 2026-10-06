/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <memory>
#include <utility>
#include <vector>

namespace {
struct StageParams {
  std::int32_t factor{1};
};
HOLOSCAN_PARAM_SCHEMA(StageParams,
                      holoscan::param_schema("com.nvidia.holoscan.test.FixtureStageParams", 1),
                      holoscan::field<&StageParams::factor>("factor"));
struct StageObservation {
  std::array<int, 32> events{};
  std::size_t size{};
  std::int32_t configured_factor{};
  void record(int event) noexcept {
    if (size < events.size())
      events[size++] = event;
  }
  std::vector<int> values() const { return {events.begin(), events.begin() + size}; }
};
class StagedOperator final : public holoscan::Operator<StageParams> {
 public:
  explicit StagedOperator(std::shared_ptr<StageObservation> observation, bool fail = false)
      : observation_(std::move(observation)), fail_(fail) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kConfigure, &StagedOperator::configure)
        .optional(holoscan::LifecycleStage::kWarmUp, &StagedOperator::warm_up)
        .stage(holoscan::LifecycleStage::kStart, &StagedOperator::start_stage)
        .stage(holoscan::LifecycleStage::kStop, &StagedOperator::stop_stage)
        .stage(holoscan::LifecycleStage::kRelease, &StagedOperator::release);
  }
  holoscan::LifecycleStatus configure(holoscan::LifecycleContext&) noexcept {
    observation_->record(0);
    observation_->configured_factor = params().factor;
    return fail_ ? holoscan::LifecycleStatus::kFatalFailure : holoscan::LifecycleStatus::kOk;
  }
  holoscan::LifecycleStatus warm_up(holoscan::LifecycleContext&) noexcept {
    observation_->record(1);
    return holoscan::LifecycleStatus::kSkipped;
  }
  holoscan::LifecycleStatus start_stage(holoscan::LifecycleContext&) noexcept {
    observation_->record(2);
    return holoscan::LifecycleStatus::kOk;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    observation_->record(3);
    auto value = in.receive();
    if (!value)
      return holoscan::make_unexpected(std::move(value).error());
    return out.emit(value->data * params().factor + count_++);
  }
  holoscan::LifecycleStatus stop_stage(holoscan::LifecycleContext&) noexcept {
    observation_->record(4);
    return holoscan::LifecycleStatus::kOk;
  }
  holoscan::LifecycleStatus release(holoscan::LifecycleContext&) noexcept {
    observation_->record(5);
    return holoscan::LifecycleStatus::kOk;
  }
  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::int32_t> out;

 private:
  std::shared_ptr<StageObservation> observation_;
  bool fail_;
  std::int32_t count_{};
};
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestingLifecycle, RegisteredStagesSeeConfiguredParamsAndRunInNativeOrder) {
  auto observation = std::make_shared<StageObservation>();
  auto h = holoscan::testing::create_operator_test<StagedOperator>(observation);
  h.set_params(StageParams{.factor = 2});
  h.push_many<std::int32_t>("in", {10, 20});
  h.capture<std::int32_t>("out");
  const auto first = h.run();
  ASSERT_TRUE(first.ok()) << first.describe_failure();
  EXPECT_EQ(first.values<std::int32_t>("out"), (std::vector<std::int32_t>{20, 41}));
  EXPECT_EQ(observation->configured_factor, 2);
  EXPECT_EQ(observation->values(), (std::vector<int>{0, 1, 2, 3, 3, 4, 5}));
  observation->size = 0;
  h.set_params(StageParams{.factor = 3});
  h.push("in", std::int32_t{10});
  const auto second = h.run();
  ASSERT_TRUE(second.ok()) << second.describe_failure();
  EXPECT_EQ(second.value<std::int32_t>("out"), 30);
  EXPECT_EQ(observation->configured_factor, 3);
  EXPECT_EQ(observation->values(), (std::vector<int>{0, 1, 2, 3, 4, 5}));
}

// REQ: OPERATOR-064
TEST(OperatorTestingLifecycle, RequiredStageFailurePreventsSteadyStateExecution) {
  auto observation = std::make_shared<StageObservation>();
  auto h = holoscan::testing::create_operator_test<StagedOperator>(observation, true);
  h.push("in", std::int32_t{10});
  h.capture<std::int32_t>("out");
  const auto result = h.run();
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.termination(), holoscan::RunTermination::kFailed);
  EXPECT_FALSE(result.diagnostics().empty());
  EXPECT_TRUE(result.values<std::int32_t>("out").empty());
  const auto events = observation->values();
  EXPECT_EQ(std::count(events.begin(), events.end(), 0), 1);
  EXPECT_EQ(std::count(events.begin(), events.end(), 2), 0);
  EXPECT_EQ(std::count(events.begin(), events.end(), 3), 0);
  EXPECT_LE(std::count(events.begin(), events.end(), 4), 1);
  EXPECT_LE(std::count(events.begin(), events.end(), 5), 1);
}

namespace {
template <holoscan::PublishedPayload T>
class MetadataSink final : public holoscan::Operator<> {
 public:
  explicit MetadataSink(std::shared_ptr<std::vector<holoscan::SampleMetadata>> observed)
      : observed_(std::move(observed)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    observed_->push_back(sample->metadata);
    return {};
  }
  holoscan::Input<T> in;

 private:
  std::shared_ptr<std::vector<holoscan::SampleMetadata>> observed_;
};
}  // namespace
// REQ: OPERATOR-073
TEST(OperatorTestingMetadata, TensorStagingDoesNotRepairAnInvalidCaptureCoordinate) {
  const holoscan::SampleMetadata metadata{.capture_timestamp_ns = 7};
  const auto scalar_seen = std::make_shared<std::vector<holoscan::SampleMetadata>>();
  auto scalar = holoscan::testing::create_operator_test<MetadataSink<std::int32_t>>(scalar_seen);
  scalar.push_samples<std::int32_t>("in", {{.data = 1, .metadata = metadata}});
  const auto scalar_result = scalar.run();
  const auto tensor_seen = std::make_shared<std::vector<holoscan::SampleMetadata>>();
  const std::array<std::int64_t, 1> shape{1};
  auto tensor = holoscan::make_tensor(shape,
                                      DLDataType{kDLUInt, 8, 1},
                                      std::nullopt,
                                      holoscan::MemoryStorageType::kSystem,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  ASSERT_TRUE(tensor);
  auto destination = tensor->mutable_data();
  ASSERT_TRUE(destination);
  *static_cast<std::uint8_t*>(*destination) = 1U;
  auto fixture =
      holoscan::testing::create_operator_test<MetadataSink<holoscan::Tensor>>(tensor_seen);
  fixture.push_samples<holoscan::Tensor>("in",
                                         {{.data = std::move(*tensor), .metadata = metadata}});
  const auto tensor_result = fixture.run();
  EXPECT_EQ(tensor_result.ok(), scalar_result.ok());
  ASSERT_EQ(tensor_seen->size(), scalar_seen->size());
  if (!scalar_seen->empty()) {
    EXPECT_EQ(tensor_seen->front().capture_timestamp_ns, scalar_seen->front().capture_timestamp_ns);
    EXPECT_EQ(tensor_seen->front().source_clock_id, scalar_seen->front().source_clock_id);
  }
}
