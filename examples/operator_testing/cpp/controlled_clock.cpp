/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <chrono>
#include <cstdint>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <iostream>
#include <utility>

namespace {
class FreshPairOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left");
    spec.input(right, "right");
    spec.output(out, "out");
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}})
        .input(left, holoscan::LatestWithin{.max_age = std::chrono::milliseconds{100}})
        .input(right, holoscan::LatestWithin{.max_age = std::chrono::milliseconds{100}})
        .on_missing(left, holoscan::Skip{})
        .on_missing(right, holoscan::Skip{});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto a = left.receive();
    auto b = right.receive();
    if (!a)
      return holoscan::make_unexpected(std::move(a).error());
    if (!b)
      return holoscan::make_unexpected(std::move(b).error());
    return out.emit(a->data + b->data);
  }
  holoscan::Input<std::int32_t> left;
  holoscan::Input<std::int32_t> right;
  holoscan::Output<std::int32_t> out;
};
}  // namespace

int main() {
  using namespace std::chrono_literals;
  using namespace holoscan::testing;
  // At a 1000 ms tick, 900 ms is the inclusive 100 ms freshness boundary.
  // 899 ms is stale, and 1001 ms is still in the future.
  for (const auto capture : {899ms, 900ms, 1001ms}) {
    auto h = create_operator_test<FreshPairOp>();
    h.push_samples<std::int32_t>("left", {sample_at(std::int32_t{1}, capture)});
    h.push_samples<std::int32_t>("right", {sample_at(std::int32_t{2}, capture)});
    h.capture<std::int32_t>("out");
    const auto result = h.run_controlled_clock(ControlledClockRun{.tick = 1000ms});
    if (!result.ok()) {
      std::cerr << result.describe_failure() << '\n';
      return 1;
    }
    const bool fresh = capture == 900ms;
    const auto expected = fresh ? ClockTestOutcome::kRetired : ClockTestOutcome::kSuppressed;
    if (result.clock_outcome() != expected ||
        result.termination() != holoscan::RunTermination::kStopped)
      return 2;
    if (fresh ? result.value<std::int32_t>("out") != 3
              : !result.values<std::int32_t>("out").empty())
      return 3;
    std::cout << "capture " << capture.count()
              << " ms: " << (fresh ? "retired with output 3" : "suppressed") << '\n';
  }
}
