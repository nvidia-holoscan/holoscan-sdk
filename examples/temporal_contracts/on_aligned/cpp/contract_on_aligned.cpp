/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// One dual-lane capture rig emits a left/right pair per 10 ms activation. The
// simulated shutter precedes the driver callback by a fixed delivery latency;
// the right lane adds an alternating inter-lane skew, so even-numbered pairs
// land inside the alignment window
// (+1 ms <= 2 ms) and odd-numbered pairs land outside it (+5 ms > 2 ms). Alignment
// compares capture timestamps, never wall-clock arrival, so the fired/skipped
// pattern below is deterministic regardless of scheduling jitter.
//
// Both physical timestamps come from one shutter and one clock domain. They may
// predate the callback that publishes them: capture coordinates are immutable
// sample provenance, while monotonic TagTime records when Runtime observes and
// dispatches the resulting work.
constexpr std::size_t kSampleCount = 8U;
constexpr std::size_t kExpectedFiredPairs = kSampleCount / 2U;
constexpr std::chrono::nanoseconds kEmitPeriod = 10ms;
constexpr std::chrono::nanoseconds kCaptureDeliveryLatency = 6ms;
constexpr std::chrono::nanoseconds kAlignedSkew = 1ms;
constexpr std::chrono::nanoseconds kMisalignedSkew = 5ms;
constexpr std::chrono::nanoseconds kAlignmentWindow = 2ms;
constexpr std::int64_t kNsPerMs = 1'000'000;

struct FiredPair {
  std::uint32_t left_sequence{};
  std::uint32_t right_sequence{};
  std::int64_t left_capture_ns{};
  std::int64_t right_capture_ns{};
};

struct AlignedObservation {
  static void reset() noexcept {
    const std::lock_guard lock(mutex);
    pairs.clear();
    fired.store(0U, std::memory_order_relaxed);
    left_emitted.store(0U, std::memory_order_relaxed);
    right_emitted.store(0U, std::memory_order_relaxed);
    base.store(kUnsetBase, std::memory_order_relaxed);
  }

  // The rig's first activation publishes the schedule origin so main() can print
  // every capture time as an offset from the start of the run.
  static void publish_base(std::int64_t candidate) {
    std::int64_t expected = kUnsetBase;
    base.compare_exchange_strong(expected, candidate, std::memory_order_acq_rel);
  }

  static void record_emit(bool right_lane) {
    (right_lane ? right_emitted : left_emitted).fetch_add(1U, std::memory_order_relaxed);
  }

  static void record_pair(const holoscan::Sample<std::uint32_t>& left,
                          const holoscan::Sample<std::uint32_t>& right) {
    {
      const std::lock_guard lock(mutex);
      pairs.push_back(FiredPair{left.data,
                                right.data,
                                left.metadata.capture_timestamp_ns,
                                right.metadata.capture_timestamp_ns});
      fired.fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(
        lock, 10s, [expected] { return fired.load(std::memory_order_acquire) >= expected; });
  }

  static constexpr std::int64_t kUnsetBase = -1;
  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<FiredPair> pairs;
  static inline std::atomic<std::size_t> fired{};
  static inline std::atomic<std::size_t> left_emitted{};
  static inline std::atomic<std::size_t> right_emitted{};
  static inline std::atomic<std::int64_t> base{kUnsetBase};
};

class StereoRigSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(left, "left");
    spec.output(right, "right");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kEmitPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_ >= kSampleCount) {
      return {};  // schedule exhausted; idle until main() requests the stop
    }
    // The raw shutter coordinates predate this callback. at_ns() labels them
    // with the graph clock without measuring or modifying that clock.
    const holoscan::TimePoint activation = context.activation_time();
    const std::int64_t left_capture_ns = activation.timestamp_ns - kCaptureDeliveryLatency.count();
    const auto sequence = static_cast<std::uint32_t>(++next_);
    const std::chrono::nanoseconds skew = (sequence % 2U == 0U) ? kAlignedSkew : kMisalignedSkew;
    const holoscan::TimePoint left_capture = context.clock().at_ns(left_capture_ns);
    const holoscan::TimePoint right_capture = context.clock().at_ns(left_capture_ns + skew.count());
    AlignedObservation::publish_base(left_capture.timestamp_ns);
    AlignedObservation::record_emit(false);
    auto emitted = left.emit(holoscan::Sample<std::uint32_t>{
        sequence,
        holoscan::SampleMetadata{.capture_timestamp_ns = left_capture.timestamp_ns,
                                 .source_clock_id = left_capture.clock_id},
    });
    if (!emitted) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    AlignedObservation::record_emit(true);
    emitted = right.emit(holoscan::Sample<std::uint32_t>{
        sequence,
        holoscan::SampleMetadata{.capture_timestamp_ns = right_capture.timestamp_ns,
                                 .source_clock_id = right_capture.clock_id},
    });
    if (!emitted) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    return {};
  }

  holoscan::Output<std::uint32_t> left;
  holoscan::Output<std::uint32_t> right;

 private:
  std::size_t next_{};
};

class AlignedJoin final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left").queue_depth(16U);
    spec.input(right, "right").queue_depth(16U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(
            holoscan::OnAligned{.ports = holoscan::ports(left, right), .window = kAlignmentWindow})
        .input(left, holoscan::Exact{})
        .input(right, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto left_sample = left.receive();
    if (!left_sample) {
      return holoscan::make_unexpected(std::move(left_sample).error());
    }
    auto right_sample = right.receive();
    if (!right_sample) {
      return holoscan::make_unexpected(std::move(right_sample).error());
    }
    AlignedObservation::record_pair(*left_sample, *right_sample);
    return {};
  }

  holoscan::Input<std::uint32_t> left;
  holoscan::Input<std::uint32_t> right;
};

}  // namespace

int main() {
  AlignedObservation::reset();

  holoscan::Graph graph{"holoscan-contract-on-aligned"};
  const auto rig = graph.op<StereoRigSource>("rig");
  const auto join = graph.op<AlignedJoin>("join");
  graph.add_flow(rig->left, join->left, holoscan::ConnectionOptions{.queue_depth = 16U});
  graph.add_flow(rig->right, join->right, holoscan::ConnectionOptions{.queue_depth = 16U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = AlignedObservation::wait_for(kExpectedFiredPairs);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "on_aligned example failed: observation incomplete or shutdown unclean\n";
    std::cerr << "  complete=" << complete
              << " termination=" << static_cast<int>(session.termination())
              << " fired=" << AlignedObservation::fired.load(std::memory_order_relaxed)
              << " emitted_left="
              << AlignedObservation::left_emitted.load(std::memory_order_relaxed)
              << " emitted_right="
              << AlignedObservation::right_emitted.load(std::memory_order_relaxed) << '\n';
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << "  diagnostic token=" << diagnostic.token << '\n';
    }
    return 2;
  }

  // All counters are stable here: the fourth fired pair requires the rig to have
  // emitted its full 8-pair schedule, and session.wait() has joined every operator,
  // so the summary below is deterministic.
  const std::int64_t base = AlignedObservation::base.load(std::memory_order_relaxed);
  const std::size_t fired = AlignedObservation::fired.load(std::memory_order_relaxed);
  const std::size_t emitted_left = AlignedObservation::left_emitted.load(std::memory_order_relaxed);
  const std::size_t emitted_right =
      AlignedObservation::right_emitted.load(std::memory_order_relaxed);
  std::uint32_t expected_sequence = 2U;
  for (const FiredPair& pair : AlignedObservation::pairs) {
    const std::int64_t spread_ns = pair.right_capture_ns - pair.left_capture_ns;
    if (pair.left_sequence != expected_sequence || pair.right_sequence != expected_sequence ||
        spread_ns < 0 || spread_ns > kAlignmentWindow.count()) {
      std::cerr << "on_aligned selected an unexpected pair or exceeded the alignment window\n";
      return 3;
    }
    expected_sequence += 2U;
  }
  for (const FiredPair& pair : AlignedObservation::pairs) {
    std::cout << "[on_aligned] fired left#" << pair.left_sequence << " (t=+"
              << (pair.left_capture_ns - base) / kNsPerMs << "ms) right#" << pair.right_sequence
              << " (t=+" << (pair.right_capture_ns - base) / kNsPerMs
              << "ms) spread=" << (pair.right_capture_ns - pair.left_capture_ns) / kNsPerMs
              << "ms\n";
  }
  std::cout << "[on_aligned] emitted_left=" << emitted_left << " emitted_right=" << emitted_right
            << '\n';
  std::cout << "[on_aligned] fired=" << fired << " skipped_left=" << (emitted_left - fired)
            << " skipped_right=" << (emitted_right - fired) << '\n';
  std::cout << "on_aligned example complete\n";
  return 0;
}
