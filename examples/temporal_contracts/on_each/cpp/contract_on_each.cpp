/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// OnEach{anchor} + side LatestWithin{50ms} + on_missing ReuseLast{50ms}.
// A 5 ms anchor stream drives every activation; a 20 ms side
// stream is sampled at the anchor rate. LatestWithin admits each queued side
// sample exactly once (admission is consuming); between side arrivals the
// ReuseLast policy re-serves the last admitted sample (InputState::kReused)
// while it is no older than 50 ms. The side source deliberately stalls for
// four ticks so the retained value ages past that bound; with nothing fresh
// and nothing reusable the side is absent and the whole activation is
// suppressed — visible afterwards as gaps in the recorded anchor sequence.

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

constexpr std::chrono::nanoseconds kAnchorPeriod = 5ms;
constexpr std::chrono::nanoseconds kSidePeriod = 20ms;
constexpr std::chrono::nanoseconds kSideMaxAge = 50ms;
constexpr std::uint32_t kStallStartTick = 5U;  // First silent side tick (inclusive).
constexpr std::uint32_t kStallEndTick = 8U;    // Last silent side tick: ~80 ms gap > 50 ms age.
constexpr std::uint32_t kSideValueScale = 100U;
constexpr std::uint32_t kResumeValueFloor = kStallEndTick * kSideValueScale + kSideValueScale;
constexpr std::size_t kPostResumeTarget = 3U;  // Activations to observe after the stall clears.

// Results are collected here and printed from main() after session.wait();
// per-activation stdout from operator threads would interleave non-atomically.
struct OnEachObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    anchor_values.clear();
    fresh_side_admissions = 0U;
    reused_side_activations = 0U;
    missing_in_compute = 0U;
    post_resume_activations = 0U;
  }

  static void record(std::uint32_t anchor_value, std::uint32_t side_value,
                     holoscan::InputState side_state, bool side_missing) {
    {
      const std::lock_guard lock(mutex);
      anchor_values.push_back(anchor_value);
      if (side_state == holoscan::InputState::kSelected) {
        ++fresh_side_admissions;
      } else if (side_state == holoscan::InputState::kReused) {
        ++reused_side_activations;
      }
      if (side_missing) {
        ++missing_in_compute;
      }
      if (side_value >= kResumeValueFloor) {
        ++post_resume_activations;
      }
    }
    ready.notify_all();
  }

  // Bounded wait: the run is considered demonstrated once the stall has ended,
  // a few post-resume activations were recorded, and at least one activation
  // re-served a retained side sample between side arrivals.
  [[nodiscard]] static bool wait_for_post_resume(std::size_t target) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [target] {
      return post_resume_activations >= target && reused_side_activations > 0U;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<std::uint32_t> anchor_values;  // Guarded by mutex.
  static inline std::size_t fresh_side_admissions{};       // Guarded by mutex.
  static inline std::size_t reused_side_activations{};     // Guarded by mutex.
  static inline std::size_t missing_in_compute{};          // Guarded by mutex.
  static inline std::size_t post_resume_activations{};     // Guarded by mutex.
};

// Emits 1, 2, 3, ... every kAnchorPeriod. Because the values are strictly
// sequential, any anchor sample whose activation was suppressed later shows up
// as a gap in the sequence the consumer recorded.
class AnchorSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kAnchorPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t value = ++next_;
    return output.emit(holoscan::Sample<std::uint32_t>{
        value,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::uint32_t next_{};
};

// Emits tick * 100 every kSidePeriod, except during the stall phase
// (ticks 5..8): four silent ticks make an ~80 ms gap, so the last pre-stall
// sample ages past the 50 ms reuse bound before the stream resumes.
class StallingSideSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kSidePeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t tick = ++tick_;
    if (tick >= kStallStartTick && tick <= kStallEndTick) {
      return {};  // stall phase: side stream goes silent
    }
    return output.emit(holoscan::Sample<std::uint32_t>{
        tick * kSideValueScale,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::uint32_t tick_{};
};

// The E05 pattern: the anchor input drives activation; the side input is
// sampled at the anchor rate. LatestWithin{50ms} admits the newest causal
// queued side sample once; on activations with no queued sample,
// ReuseLast{50ms} re-serves the last admitted sample (state kReused) while it
// is fresh enough. When neither policy can serve a value — before the first
// side sample, or once the retained sample ages out during the stall — the
// side is absent and the whole activation is suppressed: compute() never runs
// with a missing side.
class SampledConsumer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(anchor, "anchor").queue_depth(16U);
    spec.input(side, "side").queue_depth(8U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{anchor})
        .input(side, holoscan::LatestWithin{.max_age = kSideMaxAge})
        .on_missing(side, holoscan::ReuseLast{.max_age = kSideMaxAge});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto anchor_sample = anchor.receive();
    if (!anchor_sample) {
      return holoscan::make_unexpected(std::move(anchor_sample).error());
    }
    auto side_sample = side.receive();
    if (!side_sample) {
      return holoscan::make_unexpected(std::move(side_sample).error());
    }
    // The side must always be present here — a truly absent side suppresses
    // the activation before compute() runs. is_missing() is recorded to make
    // that guarantee checkable from main(); the frozen selection state
    // distinguishes a freshly admitted sample from a re-served one.
    OnEachObservation::record(
        anchor_sample->data, side_sample->data, side.selection().state, side.is_missing());
    return {};
  }

  holoscan::Input<std::uint32_t> anchor;
  holoscan::Input<std::uint32_t> side;
};

}  // namespace

int main() {
  OnEachObservation::reset();

  holoscan::Graph graph{"temporal-on-each"};
  const auto anchor_source = graph.op<AnchorSource>("anchor_source");
  const auto side_source = graph.op<StallingSideSource>("side_source");
  const auto sampler = graph.op<SampledConsumer>("sampler");
  graph.add_flow(
      anchor_source->output, sampler->anchor, holoscan::ConnectionOptions{.queue_depth = 16U});
  graph.add_flow(
      side_source->output, sampler->side, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.partition("anchor-sampling").add(anchor_source).add(sampler);
  graph.partition("side-input-source").add(side_source);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = OnEachObservation::wait_for_post_resume(kPostResumeTarget);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "run did not stop cleanly before observing the post-stall phase\n";
    return 2;
  }

  // All operator threads are joined; summarize the recorded activations.
  std::size_t activations = 0U;
  std::size_t anchor_gaps = 0U;
  std::size_t fresh_side_admissions = 0U;
  std::size_t reused_side_activations = 0U;
  std::size_t missing_in_compute = 0U;
  std::size_t post_resume_activations = 0U;
  {
    const std::lock_guard lock(OnEachObservation::mutex);
    const std::vector<std::uint32_t>& anchors = OnEachObservation::anchor_values;
    activations = anchors.size();
    if (!anchors.empty()) {
      anchor_gaps += anchors.front() - 1U;  // anchors suppressed before the first side sample
    }
    for (std::size_t index = 1U; index < anchors.size(); ++index) {
      anchor_gaps += anchors[index] - anchors[index - 1U] - 1U;  // suppressed during the stall
    }
    fresh_side_admissions = OnEachObservation::fresh_side_admissions;
    reused_side_activations = OnEachObservation::reused_side_activations;
    missing_in_compute = OnEachObservation::missing_in_compute;
    post_resume_activations = OnEachObservation::post_resume_activations;
  }

  std::cout << "[on_each] activations=" << activations << " anchor_gaps=" << anchor_gaps
            << " fresh_side_admissions=" << fresh_side_admissions
            << " reused_side_activations=" << reused_side_activations
            << " side_missing_in_compute=" << missing_in_compute
            << " post_resume_activations=" << post_resume_activations << '\n';

  // Semantic claims this example demonstrates:
  //  - the anchor drives: activations outnumber fresh side admissions because
  //    ReuseLast re-served retained samples between 20 ms side arrivals;
  //  - every activation saw either a fresh or a reused side sample;
  //  - at least one anchor was suppressed while no side value was fresh
  //    enough (before the first side sample and during the stall) — gaps;
  //  - compute() never observed a missing side.
  const bool anchor_drives = reused_side_activations > 0U && fresh_side_admissions > 0U &&
                             activations > fresh_side_admissions;
  const bool every_activation_served =
      activations == fresh_side_admissions + reused_side_activations;
  const bool absent_side_suppressed = anchor_gaps > 0U;
  const bool side_always_present = missing_in_compute == 0U;
  if (!anchor_drives || !every_activation_served || !absent_side_suppressed ||
      !side_always_present) {
    std::cerr << "on_each contract demo FAIL: anchor_drives=" << anchor_drives
              << " every_activation_served=" << every_activation_served
              << " absent_side_suppressed=" << absent_side_suppressed
              << " side_always_present=" << side_always_present << '\n';
    return 3;
  }

  std::cout << "on_each contract demo PASS: anchor-rate sampling with reuse, "
               "stale side suppressed\n";
  return 0;
}
