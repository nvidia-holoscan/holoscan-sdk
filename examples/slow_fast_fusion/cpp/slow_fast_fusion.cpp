/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

struct FusionObservation {
  static void reset() noexcept {
    count.store(0U, std::memory_order_relaxed);
    last_value.store(0U, std::memory_order_relaxed);
  }

  [[nodiscard]] static std::size_t record(std::uint32_t value) noexcept {
    last_value.store(value, std::memory_order_relaxed);
    return count.fetch_add(1U, std::memory_order_release) + 1U;
  }

  static inline std::atomic<std::size_t> count{};
  static inline std::atomic<std::uint32_t> last_value{};
};

class PeriodicSource final : public holoscan::Operator<> {
 public:
  PeriodicSource(std::chrono::nanoseconds period, std::uint32_t scale) noexcept
      : period_(period), scale_(scale) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = period_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t value = (next_++ + 1U) * scale_;
    return output.emit(holoscan::Sample<std::uint32_t>{
        value,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::chrono::nanoseconds period_;
  std::uint32_t scale_{};
  std::uint32_t next_{};
};

class FuseLatest final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(slow, "slow").queue_depth(8U);
    spec.input(fast, "fast").queue_depth(16U);
    spec.output(fused, "fused");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{fast})
        .input(slow, holoscan::LatestWithin{.max_age = 100ms})
        .on_missing(slow, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto slow_sample = slow.receive();
    if (!slow_sample) {
      return holoscan::make_unexpected(std::move(slow_sample).error());
    }
    auto fast_sample = fast.receive();
    if (!fast_sample) {
      return holoscan::make_unexpected(std::move(fast_sample).error());
    }
    fast_sample->data += slow_sample->data;
    return fused.emit(std::move(*fast_sample));
  }

  holoscan::Input<std::uint32_t> slow;
  holoscan::Input<std::uint32_t> fast;
  holoscan::Output<std::uint32_t> fused;
};

class BoundedSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(8U); }

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
    if (FusionObservation::record(sample->data) == 6U) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

}  // namespace

// REQ: SYSTEM-004
int main(int argc, char** argv) {
  FusionObservation::reset();

  holoscan::Graph graph{"holoscan-slow-fast-fusion"};
  const auto slow = graph.op<PeriodicSource>("slow", 20ms, 100U);
  const auto fast = graph.op<PeriodicSource>("fast", 2ms, 1U);
  const auto fuse = graph.op<FuseLatest>("fuse");
  const auto sink = graph.op<BoundedSink>("sink");
  graph.add_flow(slow->output, fuse->slow, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(fast->output, fuse->fast, holoscan::ConnectionOptions{.queue_depth = 16U});
  graph.add_flow(fuse->fused, sink->input, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.partition("slow-source").add(slow);
  graph.partition("fast-fusion").add(fast).add(fuse).add(sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  if (argc != 1) {
    std::filesystem::path artifact_root;
    std::string deployment_id{"default"};
    bool artifact_mode = false;
    for (int index = 1; index < argc; ++index) {
      const std::string_view argument{argv[index]};
      if (argument == "--emit-artifacts") {
        artifact_mode = true;
      } else if ((argument == "--artifact-root" || argument == "--deployment-id") &&
                 index + 1 < argc) {
        if (argument == "--artifact-root") {
          artifact_root = argv[++index];
        } else {
          deployment_id = argv[++index];
        }
      } else {
        std::cerr << "invalid artifact-only arguments\n";
        return 2;
      }
    }
    if (!artifact_mode) {
      return 2;
    }
    auto emitted = holoscan::emit_artifacts(
        plan,
        holoscan::ArtifactEmitOptions{.root = std::move(artifact_root),
                                      .deployment_id = std::move(deployment_id)});
    if (!emitted) {
      std::cerr << "artifact publication failed: "
                << holoscan::error_code_name(emitted.error().code) << '\n';
      return 3;
    }
    std::cout << "artifact-only publication complete\n";
    return 0;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  session.wait();
  if (session.termination() != holoscan::RunTermination::kRequestedCompleted ||
      !session.diagnostics().empty()) {
    return 2;
  }

  std::cout << "trigger_records=" << plan.temporal_triggers().size()
            << " selection_records=" << plan.input_selections().size()
            << " fused_samples=" << FusionObservation::count.load(std::memory_order_relaxed)
            << " last_value=" << FusionObservation::last_value.load(std::memory_order_relaxed)
            << '\n';
  std::cout << "slow/fast fusion complete\n";
  return 0;
}
