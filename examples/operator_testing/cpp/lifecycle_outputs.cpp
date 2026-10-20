/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cstdint>
#include <holoscan/testing/operator_test_harness.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
class BurstOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out").max_emits_per_compute(2U);
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    if (sample->data <= 0)
      return {};
    auto emitted = out.emit(sample->data);
    if (!emitted)
      return emitted;
    return out.emit(-sample->data);
  }
  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::int32_t> out;
};
struct SinkObservation {
  std::vector<std::string> events;
  std::int32_t total{};
};
class RecordingSink final : public holoscan::Operator<> {
 public:
  explicit RecordingSink(std::shared_ptr<SinkObservation> observation)
      : observation_(std::move(observation)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  void start() override { observation_->events.emplace_back("start"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    observation_->total += sample->data;
    observation_->events.emplace_back("compute");
    return {};
  }
  void stop() override { observation_->events.emplace_back("stop"); }
  holoscan::Input<std::int32_t> in;

 private:
  std::shared_ptr<SinkObservation> observation_;
};
}  // namespace

int main() {
  using namespace holoscan::testing;
  try {
    auto burst = create_operator_test<BurstOp>();
    burst.push_many<std::int32_t>("in", {1, -1, 2});
    burst.capture<std::int32_t>("out", 4U);
    const auto first = burst.run();
    if (!first.ok())
      throw std::runtime_error{first.describe_failure()};
    if (first.values<std::int32_t>("out") != std::vector<std::int32_t>{1, -1, 2, -2})
      throw std::runtime_error{"burst cardinality mismatch"};
    burst.push("in", std::int32_t{-1});
    const auto empty = burst.run();
    if (!empty.ok() || !empty.values<std::int32_t>("out").empty())
      throw std::runtime_error{"expected successful run with no output"};
    if (first.values<std::int32_t>("out").size() != 4U)
      throw std::runtime_error{"later run changed an earlier result"};

    // An explicit observer can record sink side effects. Read it only after run() joins.
    auto observation = std::make_shared<SinkObservation>();
    auto sink = create_operator_test<RecordingSink>(observation);
    sink.push_many<std::int32_t>("in", {10, 20});
    const auto consumed = sink.run();
    if (!consumed.ok())
      throw std::runtime_error{consumed.describe_failure()};
    const std::vector<std::string> expected{"start", "compute", "compute", "stop"};
    if (observation->events != expected || observation->total != 30)
      throw std::runtime_error{"sink lifecycle or side-effect mismatch"};
    std::cout << "lifecycle and zero/multiple outputs verified after joined completion\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
