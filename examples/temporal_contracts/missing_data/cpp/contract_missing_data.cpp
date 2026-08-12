/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compare the three missing-data policies on otherwise identical consumers.
//
// One gated source emits bursts (4 ticks on, 4 ticks off). Three consumers
// share the same direct 10 ms OnClock contract and sample the gated stream
// with the same LatestWithin
// selector; they differ only in the on_missing policy: Skip{}, SafeDefault{},
// or ReuseLast{}. The per-policy counters printed after the run make the
// behavioral difference observable.
// The consumers use the direct public OnClock-with-input contract; no separate
// tick stream is needed to express their cadence.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
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

constexpr std::chrono::nanoseconds kTickPeriod = 10ms;
constexpr std::chrono::nanoseconds kMaxSampleAge = 15ms;  // staleness window on data
constexpr std::uint32_t kBurstLength = 4U;                // ticks emitting per burst
constexpr std::uint32_t kSilenceLength = 4U;              // silent ticks per burst cycle
constexpr std::uint32_t kConsumerActivations = 32U;
constexpr std::uint32_t kDataQueueDepth = 4U;

/// Per-policy counters recorded by each consumer and printed after the run.
/// Static storage: graph.op<> rejects raw-pointer captures by design, so
/// observation state follows the canonical static-struct example pattern.
struct PolicyCounters {
  std::atomic<std::uint32_t> activations{};
  std::atomic<std::uint32_t> fresh{};
  std::atomic<std::uint32_t> reused{};
  std::atomic<std::uint32_t> safe_default{};
  std::atomic<std::uint32_t> absent{};
};

enum class PolicySlot : std::uint32_t { kSkip = 0U, kSafeDefault = 1U, kReuseLast = 2U };

struct PolicyCountersBank {
  [[nodiscard]] static PolicyCounters& at(PolicySlot slot) {
    return slots[static_cast<std::size_t>(slot)];
  }
  static std::array<PolicyCounters, 3> slots;
};

inline std::array<PolicyCounters, 3> PolicyCountersBank::slots{};

struct Observation {
  static void record() {
    {
      const std::lock_guard lock(mutex);
      ++total;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::uint32_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] { return total >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::uint32_t total{};
};

/// Emits kBurstLength samples, stays silent kSilenceLength ticks, repeats.
class GatedSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(out, "out"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t phase = tick_++ % (kBurstLength + kSilenceLength);
    if (phase >= kBurstLength) {
      return {};  // silent tick: downstream LatestWithin windows go stale
    }
    const auto activation = context.activation_time();
    return out.emit(holoscan::Sample<std::uint32_t>{
        tick_,
        holoscan::SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                                 .source_clock_id = activation.clock_id}});
  }

  holoscan::Output<std::uint32_t> out;

 private:
  std::uint32_t tick_{};
};

/// One consumer shape for all three policies; the policy comes from contract().
template <typename MissingPolicyT>
class PolicyConsumer final : public holoscan::Operator<> {
 public:
  PolicyConsumer(PolicySlot slot, MissingPolicyT policy) noexcept : slot_(slot), policy_(policy) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(data, "data").queue_depth(kDataQueueDepth);
  }

  // LatestWithin sits outside the clock trigger's guarantee, so the
  // contract must also say what happens when the window is empty: that is
  // the on_missing policy under comparison.
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod})
        .input(data, holoscan::LatestWithin{.max_age = kMaxSampleAge})
        .on_missing(data, policy_);
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    PolicyCounters& counters = PolicyCountersBank::at(slot_);
    counters.activations.fetch_add(1U, std::memory_order_relaxed);
    switch (data.selection().state) {
      case holoscan::InputState::kSelected: {
        auto sample = data.receive();
        if (!sample) {
          return holoscan::make_unexpected(std::move(sample).error());
        }
        counters.fresh.fetch_add(1U, std::memory_order_relaxed);
        break;
      }
      case holoscan::InputState::kReused: {
        auto sample = data.receive();
        if (!sample) {
          return holoscan::make_unexpected(std::move(sample).error());
        }
        counters.reused.fetch_add(1U, std::memory_order_relaxed);
        break;
      }
      case holoscan::InputState::kSafeDefault:
        // The operator-authored degraded branch: no fabricated sample exists.
        counters.safe_default.fetch_add(1U, std::memory_order_relaxed);
        break;
      default:
        counters.absent.fetch_add(1U, std::memory_order_relaxed);
        break;
    }
    Observation::record();
    return {};
  }

  holoscan::Input<std::uint32_t> data;

 private:
  PolicySlot slot_;
  MissingPolicyT policy_;
};

void print_policy(std::string_view label, const PolicyCounters& counters) {
  std::cout << label << ": activations=" << counters.activations.load()
            << " fresh=" << counters.fresh.load() << " reused=" << counters.reused.load()
            << " safe_default=" << counters.safe_default.load()
            << " absent=" << counters.absent.load() << '\n';
}

}  // namespace

int main() {
  holoscan::Graph graph{"contract-missing-data"};
  const auto source = graph.op<GatedSource>("source");
  const auto skip_consumer =
      graph.op<PolicyConsumer<holoscan::Skip>>("skip", PolicySlot::kSkip, holoscan::Skip{});
  const auto safe_consumer = graph.op<PolicyConsumer<holoscan::SafeDefault>>(
      "safe-default", PolicySlot::kSafeDefault, holoscan::SafeDefault{});
  const auto reuse_consumer = graph.op<PolicyConsumer<holoscan::ReuseLast>>(
      "reuse-last", PolicySlot::kReuseLast, holoscan::ReuseLast{.max_age = 60ms});

  // Observational fan-out edges use bounded kDropOldest so each policy lane
  // retains recent candidates independently while a clock candidate is skipped
  // or served from the policy's fallback state.
  graph.add_flow(source->out,
                 skip_consumer->data,
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = kDataQueueDepth});
  graph.add_flow(source->out,
                 safe_consumer->data,
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = kDataQueueDepth});
  graph.add_flow(source->out,
                 reuse_consumer->data,
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = kDataQueueDepth});
  // Each policy consumer owns a clock cadence independent of the source.
  graph.partition("gated-source").add(source);
  graph.partition("skip-policy").add(skip_consumer);
  graph.partition("safe-default-policy").add(safe_consumer);
  graph.partition("reuse-last-policy").add(reuse_consumer);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  // Skip suppresses whole activations, so only the two always-activating
  // consumers gate the stop; the Skip consumer contributes what it fired.
  const bool observed = Observation::wait_for(2U * kConsumerActivations);
  session.request_stop();
  session.wait();

  if (!observed) {
    std::cerr << "timed out waiting for consumer activations\n";
    std::cerr << "termination=" << static_cast<int>(session.termination()) << '\n';
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << "diagnostic: " << diagnostic.token << '\n';
    }
    print_policy("skip", PolicyCountersBank::at(PolicySlot::kSkip));
    print_policy("safe_default", PolicyCountersBank::at(PolicySlot::kSafeDefault));
    print_policy("reuse_last", PolicyCountersBank::at(PolicySlot::kReuseLast));
    return 2;
  }
  if (!session.diagnostics().empty()) {
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << "diagnostic: " << diagnostic.token << '\n';
    }
    return 3;
  }

  print_policy("skip", PolicyCountersBank::at(PolicySlot::kSkip));
  print_policy("safe_default", PolicyCountersBank::at(PolicySlot::kSafeDefault));
  print_policy("reuse_last", PolicyCountersBank::at(PolicySlot::kReuseLast));

  // The teaching claims, asserted so CTest cannot pass a regression:
  // Skip never fabricates or reuses; SafeDefault always activates and takes
  // the degraded branch during silence; ReuseLast re-serves within max_age.
  const PolicyCounters& skip_counters = PolicyCountersBank::at(PolicySlot::kSkip);
  const PolicyCounters& safe_counters = PolicyCountersBank::at(PolicySlot::kSafeDefault);
  const PolicyCounters& reuse_counters = PolicyCountersBank::at(PolicySlot::kReuseLast);
  const bool skip_clean =
      skip_counters.safe_default.load() == 0U && skip_counters.reused.load() == 0U;
  const bool skip_suppressed_missing_ticks =
      skip_counters.activations.load() == skip_counters.fresh.load() &&
      skip_counters.activations.load() < safe_counters.activations.load();
  const bool safe_saw_default = safe_counters.safe_default.load() > 0U;
  const bool reuse_saw_reuse = reuse_counters.reused.load() > 0U;
  if (!skip_clean || !skip_suppressed_missing_ticks || !safe_saw_default || !reuse_saw_reuse) {
    std::cerr << "policy comparison did not demonstrate the expected states\n";
    return 4;
  }

  std::cout << "missing_data: policy comparison complete (skip/safe_default/reuse_last)\n";
  return 0;
}
