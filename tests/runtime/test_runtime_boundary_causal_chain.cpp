/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include "compiler/execution_plan_state.hpp"
#include "core/domain/tensor_access.hpp"
#include "runtime/run_session_internal.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"

namespace holoscan {

// REQ: RUNTIME-040
// REQ: RUNTIME-041

namespace {

using namespace std::chrono_literals;

// The tag time of each clock-root activation, by the value it published. A chain member continued
// from that activation takes exactly its logical coordinate and the next microstep; one frozen by
// the ordinary scan takes the coordinate of the scan that observed its trigger.
struct ChainRootLog {
  static inline std::mutex mutex;
  static inline std::vector<std::optional<TagTime>> tag_by_value;
  static void reset() {
    const std::lock_guard lock(mutex);
    tag_by_value.clear();
  }
  static void record(std::uint32_t value, TagTime tag_time) {
    const std::lock_guard lock(mutex);
    if (tag_by_value.size() <= value) {
      tag_by_value.resize(value + 1U);
    }
    tag_by_value[value] = tag_time;
  }
  [[nodiscard]] static std::optional<TagTime> lookup(std::uint32_t value) {
    const std::lock_guard lock(mutex);
    return value < tag_by_value.size() ? tag_by_value[value] : std::nullopt;
  }
};

// Ticks on a 1 ms clock and counts its publications; a chain root records its coordinates. A
// source feeding a LatestWithin input stamps its activation time as the capture time.
class BoundaryChainSource final : public Operator<> {
 public:
  explicit BoundaryChainSource(std::chrono::microseconds period, bool chain_root = false,
                               bool stamp_capture = false,
                               std::optional<TimePoint> start_at = std::nullopt)
      : period_(period),
        chain_root_(chain_root),
        stamp_capture_(stamp_capture),
        start_at_(start_at) {}

  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = period_, .start_at = start_at_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ++published_;
    if (chain_root_) {
      ChainRootLog::record(published_, context.tag_time());
    }
    if (stamp_capture_) {
      const TimePoint activation = context.activation_time();
      return output.emit(Sample<std::uint32_t>{
          published_,
          SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                         .source_clock_id = activation.clock_id},
      });
    }
    return output.emit(published_);
  }

  Output<std::uint32_t> output;

 private:
  std::chrono::microseconds period_;
  bool chain_root_{};
  bool stamp_capture_{};
  std::optional<TimePoint> start_at_;
  std::uint32_t published_{};
};

// Ticks on its clock only to publish the side samples a test grants, stamping each with its
// activation time, so a test can place every side sample between a chain root's ticks.
class ScheduledSideSource final : public Operator<> {
 public:
  ScheduledSideSource(std::chrono::microseconds period, TimePoint start_at)
      : period_(period), start_at_(start_at) {}

  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = period_, .start_at = start_at_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    expected<void, Error> emitted{};
    if (grants.load(std::memory_order_acquire) > 0U) {
      grants.fetch_sub(1U, std::memory_order_acq_rel);
      const TimePoint activation = context.activation_time();
      emitted = output.emit(Sample<std::uint32_t>{
          ++published_,
          SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                         .source_clock_id = activation.clock_id},
      });
    }
    ticks.fetch_add(1U, std::memory_order_release);
    return emitted;
  }

  Output<std::uint32_t> output;
  static inline std::atomic<std::uint32_t> grants{};
  static inline std::atomic<std::uint64_t> ticks{};

 private:
  std::chrono::microseconds period_;
  TimePoint start_at_;
  std::uint32_t published_{};
};

// One activation of a LatestWithin owner: the value it published or received, its side input
// state, and the side sample it selected or reused, if any.
struct SideSelection {
  std::uint32_t value{};
  InputState state{InputState::kNotSelected};
  std::optional<std::uint32_t> side;
};

// Records the side input state and the side sample behind it, if any, into `record`.
template <typename Record>
[[nodiscard]] expected<void, Error> record_side(Input<std::uint32_t>& side, Record& record) {
  record.state = side.selection().state;
  if (record.state == InputState::kSelected || record.state == InputState::kReused) {
    auto sample = side.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    record.side = sample->data;
  }
  return {};
}

// A side input missing policy: the degraded path, or reuse of the last sample within 1 s.
void on_missing_side(Contract& contract, const Input<std::uint32_t>& side, bool reuse_last) {
  if (reuse_last) {
    contract.on_missing(side, ReuseLast{.max_age = 1s});
  } else {
    contract.on_missing(side, SafeDefault{});
  }
}

// Ticks on a 1 ms clock, selects the latest side sample, records it, and counts its publications.
class BoundaryChainSideRoot final : public Operator<> {
 public:
  explicit BoundaryChainSideRoot(std::optional<TimePoint> start_at = std::nullopt,
                                 bool reuse_last = false)
      : start_at_(start_at), reuse_last_(reuse_last) {}

  void setup(OperatorSpec& spec) override {
    spec.input(side, "side").queue_depth(1U);
    spec.output(output, "output");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms, .start_at = start_at_})
        .input(side, LatestWithin{.max_age = 1s});
    on_missing_side(result, side, reuse_last_);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    SideSelection selection{.value = ++published_};
    if (auto recorded = record_side(side, selection); !recorded) {
      return recorded;
    }
    {
      const std::lock_guard lock(mutex);
      selections.push_back(selection);
    }
    return output.emit(selection.value);
  }

  Input<std::uint32_t> side;
  Output<std::uint32_t> output;
  static inline std::mutex mutex;
  static inline std::vector<SideSelection> selections;

 private:
  std::optional<TimePoint> start_at_;
  bool reuse_last_{};
  std::uint32_t published_{};
};

// One activation of a side member: its value, its logical coordinate, and the side samples it
// drained.
struct SideMemberActivation {
  std::uint32_t value{};
  TagTime tag_time{};
  std::vector<std::uint32_t> side;

  // Whether a causal chain continued into this activation from the root that published `value`.
  [[nodiscard]] bool continued() const {
    const std::optional<TagTime> root = ChainRootLog::lookup(value);
    return root && tag_time.logical_ns == root->logical_ns;
  }

  // Whether a continued activation took the microstep after its root's.
  [[nodiscard]] bool next_microstep() const {
    const std::optional<TagTime> root = ChainRootLog::lookup(value);
    return root && tag_time.logical_ns == root->logical_ns &&
           tag_time.microstep == root->microstep + 1U;
  }
};

// Runs on each value and also drains a side input, like the robot's MotionPlanner.
class BoundaryChainSideMember final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.input(side, "side").queue_depth(4U);
    spec.output(output, "output");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input}).input(side, DrainAll{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto value = input.receive();
    if (!value) {
      return make_unexpected(std::move(value).error());
    }
    auto batch = side.selected_batch();
    if (!batch) {
      return make_unexpected(std::move(batch).error());
    }
    SideMemberActivation activation{.value = value->data, .tag_time = context.tag_time()};
    for (const Sample<std::uint32_t>& sample : *batch) {
      activation.side.push_back(sample.data);
    }
    {
      const std::lock_guard lock(mutex);
      activations.push_back(std::move(activation));
    }
    return output.emit(value->data);
  }

  Input<std::uint32_t> input;
  Input<std::uint32_t> side;
  Output<std::uint32_t> output;
  static inline std::mutex mutex;
  static inline std::vector<SideMemberActivation> activations;
};

// One activation of a LatestWithin member: its value, its logical coordinate, its side input
// state, and the side sample it selected or reused, if any.
struct LatestMemberActivation {
  std::uint32_t value{};
  std::int64_t logical_ns{};
  InputState state{InputState::kNotSelected};
  std::optional<std::uint32_t> side;

  [[nodiscard]] bool continued() const {
    const std::optional<TagTime> root = ChainRootLog::lookup(value);
    return root && logical_ns == root->logical_ns;
  }
};

// Runs on each value and selects the latest side sample, like the robot's MotionPlanner with a
// LatestWithin target input.
class BoundaryChainLatestMember final : public Operator<> {
 public:
  explicit BoundaryChainLatestMember(bool reuse_last = false) : reuse_last_(reuse_last) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.input(side, "side").queue_depth(1U);
    spec.output(output, "output");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input}).input(side, LatestWithin{.max_age = 1s});
    on_missing_side(result, side, reuse_last_);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto value = input.receive();
    if (!value) {
      return make_unexpected(std::move(value).error());
    }
    LatestMemberActivation activation{.value = value->data,
                                      .logical_ns = context.tag_time().logical_ns};
    if (auto recorded = record_side(side, activation); !recorded) {
      return recorded;
    }
    {
      const std::lock_guard lock(mutex);
      activations.push_back(activation);
    }
    return output.emit(value->data);
  }

  Input<std::uint32_t> input;
  Input<std::uint32_t> side;
  Output<std::uint32_t> output;
  static inline std::mutex mutex;
  static inline std::vector<LatestMemberActivation> activations;

 private:
  bool reuse_last_{};
};

// Receives in another partition, like the robot's TargetProducer reading joint state.
class BoundaryChainRemoteSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto value = input.receive();
    if (!value) {
      return make_unexpected(std::move(value).error());
    }
    received.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  Input<std::uint32_t> input;
  static inline std::atomic<std::uint64_t> received{};
};

// Lets a side publication land while the clock root computes one frame. The root asks for it and
// waits; the side source publishes once when asked.
struct SideRaceGate {
  static constexpr std::uint32_t kRacedFrame = 5U;
  static constexpr std::uint32_t kSideValue = 1000U;
  static inline std::atomic<bool> requested{};
  static inline std::atomic<bool> published{};
  static void reset() noexcept {
    requested.store(false);
    published.store(false);
  }
};

// Ticks on a 1 ms clock; on SideRaceGate::kRacedFrame it waits for the side publication.
class RacingChainRoot final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ChainRootLog::record(++published_, context.tag_time());
    if (published_ == SideRaceGate::kRacedFrame) {
      SideRaceGate::requested.store(true, std::memory_order_release);
      const auto deadline = std::chrono::steady_clock::now() + 2s;
      while (!SideRaceGate::published.load(std::memory_order_acquire) &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(50us);
      }
    }
    return output.emit(published_);
  }

  Output<std::uint32_t> output;

 private:
  std::uint32_t published_{};
};

// Publishes one side sample once the racing root asks for it.
class RacingSideSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 100us});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (!SideRaceGate::requested.load(std::memory_order_acquire) ||
        SideRaceGate::published.load(std::memory_order_acquire)) {
      return {};
    }
    auto emitted = output.emit(SideRaceGate::kSideValue);
    SideRaceGate::published.store(true, std::memory_order_release);
    return emitted;
  }

  Output<std::uint32_t> output;
};

// Runs on each value and passes it on, or records it as the sink.
class BoundaryChainMember final : public Operator<> {
 public:
  explicit BoundaryChainMember(bool sink) : sink_(sink) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    if (!sink_) {
      spec.output(output, "output");
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto value = input.receive();
    if (!value) {
      return make_unexpected(std::move(value).error());
    }
    if (sink_) {
      const std::lock_guard lock(mutex);
      sunk.push_back(value->data);
      return {};
    }
    return output.emit(value->data);
  }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
  static inline std::mutex mutex;
  static inline std::vector<std::uint32_t> sunk;

 private:
  bool sink_{};
};

enum class BoundaryShape : std::uint8_t {
  kSideInput,                 // the boundary edge feeds a member's side input
  kTriggerInput,              // the boundary edge feeds a member's trigger input
  kFreshnessSideInput,        // the boundary edge feeds the clock root's drop-oldest LatestWithin
  kFreshnessMemberSideInput,  // the boundary edge feeds a member's drop-oldest LatestWithin
  kFanoutSideInput,  // kSideInput, with the root's output also fanned out across the boundary
  kRacingSideInput,  // kSideInput, with one side publication landing while the root computes
};

// Racing root -> side member -> relay -> sink, with the racing side source in another partition.
[[nodiscard]] ExecutionPlan make_racing_side_input_plan() {
  Graph graph{"racing-side-input-chain"};
  const ConnectionOptions depth_one{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U};
  const auto root = graph.op<RacingChainRoot>("root");
  const auto member = graph.op<BoundaryChainSideMember>("member");
  const auto relay = graph.op<BoundaryChainMember>("relay", false);
  const auto sink = graph.op<BoundaryChainMember>("sink", true);
  const auto racing_side = graph.op<RacingSideSource>("racing-side");
  graph.add_flow(root->output, member->input, depth_one);
  graph.add_flow(racing_side->output, member->side, depth_one);
  graph.add_flow(member->output, relay->input, depth_one);
  graph.add_flow(relay->output, sink->input, depth_one);
  graph.partition("control").add(root).add(member).add(relay).add(sink);
  graph.partition("side").add(racing_side);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  return compile(graph);
}

// Clock-rooted source -> side member -> relay -> sink in one partition, fed across a boundary by
// a side source in another partition. Mirrors the robot control loop's control partition.
[[nodiscard]] ExecutionPlan make_boundary_chain_plan(BoundaryShape shape) {
  if (shape == BoundaryShape::kRacingSideInput) {
    return make_racing_side_input_plan();
  }
  Graph graph{"boundary-causal-chain"};
  const auto relay = graph.op<BoundaryChainMember>("relay", false);
  const auto sink = graph.op<BoundaryChainMember>("sink", true);
  const bool freshness_shape = shape == BoundaryShape::kFreshnessSideInput ||
                               shape == BoundaryShape::kFreshnessMemberSideInput;
  const auto side_source = graph.op<BoundaryChainSource>(
      "side-source", std::chrono::microseconds{3000}, false, freshness_shape);
  const ConnectionOptions depth_one{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U};
  if (shape == BoundaryShape::kFreshnessSideInput) {
    // A drop-oldest LatestWithin side input of the clock root is freshness-reconciled.
    const auto root = graph.op<BoundaryChainSideRoot>("root");
    const auto first = graph.op<BoundaryChainMember>("first", false);
    graph.add_flow(side_source->output, root->side, depth_one);
    graph.add_flow(root->output, first->input, depth_one);
    graph.add_flow(first->output, relay->input, depth_one);
    graph.add_flow(relay->output, sink->input, depth_one);
    graph.partition("control").add(root).add(first).add(relay).add(sink);
    graph.partition("side").add(side_source);
    graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
    return compile(graph);
  }
  const auto source =
      graph.op<BoundaryChainSource>("source", std::chrono::microseconds{1000}, true);
  if (shape == BoundaryShape::kFreshnessMemberSideInput) {
    // The robot control partition with a LatestWithin target input on MotionPlanner.
    const auto member = graph.op<BoundaryChainLatestMember>("member");
    graph.add_flow(source->output, member->input, depth_one);
    graph.add_flow(side_source->output, member->side, depth_one);
    graph.add_flow(member->output, relay->input, depth_one);
    graph.add_flow(relay->output, sink->input, depth_one);
    graph.partition("control").add(source).add(member).add(relay).add(sink);
  } else if (shape == BoundaryShape::kTriggerInput) {
    // The other partition drives the member, so its arrivals activate a member of this partition;
    // the clock root feeds the member's side input instead. That side edge drops the oldest
    // sample: the 1 ms root outpaces the 3 ms member, and a blocked root would also hold the
    // partition's thread, so the member could not drain it within the graph blocking budget.
    // Chain admission does not depend on queue policy.
    const auto member = graph.op<BoundaryChainSideMember>("member");
    graph.add_flow(side_source->output, member->input, depth_one);
    graph.add_flow(source->output,
                   member->side,
                   ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 4U});
    graph.add_flow(member->output, relay->input, depth_one);
    graph.add_flow(relay->output, sink->input, depth_one);
    graph.partition("control").add(source).add(member).add(relay).add(sink);
  } else if (shape == BoundaryShape::kFanoutSideInput) {
    // The robot control partition's shape: the root's drop-oldest POD output reaches the member
    // and a recipient in the other partition as one direct fanout family (CHAN-160).
    const auto member = graph.op<BoundaryChainSideMember>("member");
    const auto remote = graph.op<BoundaryChainRemoteSink>("remote");
    graph.add_flow(source->output, member->input, depth_one);
    graph.add_flow(source->output, remote->input, depth_one);
    graph.add_flow(member->output, relay->input, depth_one);
    graph.add_flow(relay->output, sink->input, depth_one);
    graph.add_flow(side_source->output, member->side, depth_one);
    graph.partition("control").add(source).add(member).add(relay).add(sink);
    graph.partition("side").add(side_source);
    graph.partition("remote").add(remote);
    graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
    return compile(graph);
  } else {
    const auto member = graph.op<BoundaryChainSideMember>("member");
    graph.add_flow(member->output, relay->input, depth_one);
    graph.add_flow(relay->output, sink->input, depth_one);
    graph.add_flow(source->output, member->input, depth_one);
    graph.add_flow(side_source->output,
                   member->side,
                   ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 4U});
    graph.partition("control").add(source).add(member).add(relay).add(sink);
  }
  graph.partition("side").add(side_source);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  return compile(graph);
}

struct BoundaryChainRun {
  RunTermination termination{};
  std::size_t diagnostics{};
  std::vector<std::uint32_t> sunk;
  std::uint64_t attempts{};
  std::uint64_t accepted{};
  std::uint64_t fallbacks{};
  std::uint64_t boundary_fallbacks{};
  std::vector<SideMemberActivation> member_activations;
  std::vector<LatestMemberActivation> latest_activations;
  std::uint64_t remote_received{};
};

[[nodiscard]] BoundaryChainRun run_boundary_chain(const ExecutionPlan& compiled, bool disable_chain,
                                                  std::size_t ticks,
                                                  bool disable_side_input_continuation = false) {
  {
    const std::lock_guard lock(BoundaryChainMember::mutex);
    BoundaryChainMember::sunk.clear();
  }
  {
    const std::lock_guard lock(BoundaryChainSideMember::mutex);
    BoundaryChainSideMember::activations.clear();
  }
  {
    const std::lock_guard lock(BoundaryChainLatestMember::mutex);
    BoundaryChainLatestMember::activations.clear();
  }
  {
    const std::lock_guard lock(BoundaryChainSideRoot::mutex);
    BoundaryChainSideRoot::selections.clear();
  }
  BoundaryChainRemoteSink::received.store(0U);
  SideRaceGate::reset();
  ChainRootLog::reset();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  if (disable_chain) {
    plan = detail::RuntimeTestPlanAccess::disable_boundary_causal_chain_for_test(plan);
  }
  if (disable_side_input_continuation) {
    plan = detail::RuntimeTestPlanAccess::disable_side_input_continuation_for_test(plan);
  }
  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    {
      const std::lock_guard lock(BoundaryChainMember::mutex);
      if (BoundaryChainMember::sunk.size() >= ticks) {
        break;
      }
    }
    std::this_thread::sleep_for(1ms);
  }
  session.request_stop();
  session.wait();
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  std::vector<SideMemberActivation> member_activations;
  {
    const std::lock_guard lock(BoundaryChainSideMember::mutex);
    member_activations = BoundaryChainSideMember::activations;
  }
  std::vector<LatestMemberActivation> latest_activations;
  {
    const std::lock_guard lock(BoundaryChainLatestMember::mutex);
    latest_activations = BoundaryChainLatestMember::activations;
  }
  const std::lock_guard lock(BoundaryChainMember::mutex);
  return BoundaryChainRun{
      .termination = session.termination(),
      .diagnostics = session.diagnostics().size(),
      .sunk = BoundaryChainMember::sunk,
      .attempts = count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
      .accepted = count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
      .fallbacks = count(detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks),
      .boundary_fallbacks =
          count(detail::RuntimePhaseCounter::kPartitionCausalFrontierBoundaryFallbacks),
      .member_activations = std::move(member_activations),
      .latest_activations = std::move(latest_activations),
      .remote_received = BoundaryChainRemoteSink::received.load(),
  };
}

// A partition whose only boundary edges are side inputs and outputs keeps its causal chain: the
// side arrivals cannot activate a member, so the chain's successor order is still the whole frame.
// The two hops after the side-input member continue (its own hop is
// ChainContinuesIntoASideInputMember's). The control arm keeps the partition on the ordinary
// path.
// REQ: RUNTIME-041
// REQ: RUNTIME-070
TEST(RuntimeInlinePartitionExecution, BoundaryChainContinuesPastSideOnlyBoundaryInputs) {
  constexpr std::size_t kTicks = 30U;
  const ExecutionPlan compiled = make_boundary_chain_plan(BoundaryShape::kSideInput);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  for (const bool disable_chain : {false, true}) {
    SCOPED_TRACE(testing::Message() << "disable_chain=" << disable_chain);
    const BoundaryChainRun run = run_boundary_chain(compiled, disable_chain, kTicks);
    EXPECT_EQ(run.termination, RunTermination::kStopped);
    EXPECT_EQ(run.diagnostics, 0U);
    ASSERT_GE(run.sunk.size(), kTicks);
    EXPECT_TRUE(std::ranges::is_sorted(run.sunk));
    EXPECT_EQ(std::ranges::adjacent_find(run.sunk), run.sunk.end());
    if (disable_chain) {
      EXPECT_EQ(run.attempts, 0U);
      EXPECT_EQ(run.accepted, 0U);
    } else {
      // Key assertion: the relay and sink hops continue on most ticks; a hop whose boundary side
      // input changed falls back to the ordinary scan.
      EXPECT_GE(run.accepted, kTicks)
          << "attempts=" << run.attempts << " fallbacks=" << run.fallbacks
          << " boundary_fallbacks=" << run.boundary_fallbacks << " sunk=" << run.sunk.size();
      EXPECT_EQ(run.attempts, run.accepted + run.fallbacks);
      EXPECT_LE(run.boundary_fallbacks, run.fallbacks);
    }
  }
}

// A boundary edge into a member's trigger input lets another partition activate that member, so
// the partition keeps the ordinary path.
// REQ: RUNTIME-070
TEST(RuntimeInlinePartitionExecution, BoundaryTriggerInputKeepsTheOrdinaryPath) {
  constexpr std::size_t kTicks = 10U;
  const ExecutionPlan compiled = make_boundary_chain_plan(BoundaryShape::kTriggerInput);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const BoundaryChainRun run = run_boundary_chain(compiled, false, kTicks);
  EXPECT_EQ(run.termination, RunTermination::kStopped);
  EXPECT_EQ(run.diagnostics, 0U);
  ASSERT_GE(run.sunk.size(), kTicks);
  // Key assertion: no chain is installed, so no continuation is ever attempted.
  EXPECT_EQ(run.attempts, 0U);
}

// A freshness-reconciled LatestWithin side input of the clock root, or of a member, from another
// partition keeps the partition's causal chain. The reconcile still runs in the ordinary scan: the
// root is never continued into, and the hop into a LatestWithin member stays on the ordinary scan
// (RUNTIME-084). Every later hop continues. The control arm keeps the partition on the ordinary
// path, and the LatestWithin member selects in order under either.
// REQ: RUNTIME-086
TEST(RuntimeInlinePartitionExecution, BoundaryChainContinuesPastFreshnessSideInputs) {
  constexpr std::size_t kTicks = 30U;
  for (const BoundaryShape shape :
       {BoundaryShape::kFreshnessSideInput, BoundaryShape::kFreshnessMemberSideInput}) {
    const ExecutionPlan compiled = make_boundary_chain_plan(shape);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    for (const bool disable_chain : {false, true}) {
      SCOPED_TRACE(testing::Message() << "shape=" << static_cast<unsigned>(shape)
                                      << " disable_chain=" << disable_chain);
      const BoundaryChainRun run = run_boundary_chain(compiled, disable_chain, kTicks);
      EXPECT_EQ(run.termination, RunTermination::kStopped);
      EXPECT_EQ(run.diagnostics, 0U);
      ASSERT_GE(run.sunk.size(), kTicks);
      EXPECT_TRUE(std::ranges::is_sorted(run.sunk));
      EXPECT_EQ(std::ranges::adjacent_find(run.sunk), run.sunk.end());
      if (disable_chain) {
        EXPECT_EQ(run.attempts, 0U);
      } else {
        // Key assertion: the hops after the freshness owner continue on most ticks.
        EXPECT_GE(run.accepted, kTicks)
            << "attempts=" << run.attempts << " fallbacks=" << run.fallbacks
            << " boundary_fallbacks=" << run.boundary_fallbacks << " sunk=" << run.sunk.size();
        EXPECT_EQ(run.attempts, run.accepted + run.fallbacks);
      }
      if (shape == BoundaryShape::kFreshnessMemberSideInput) {
        ASSERT_GE(run.latest_activations.size(), kTicks);
        std::optional<std::uint32_t> previous;
        bool selected = false;
        for (const LatestMemberActivation& activation : run.latest_activations) {
          // The hop into the LatestWithin member is never continued.
          EXPECT_FALSE(activation.continued()) << "value " << activation.value;
          if (activation.side) {
            selected = true;
            if (previous) {
              EXPECT_GE(*activation.side, *previous);
            }
            previous = activation.side;
          }
        }
        EXPECT_TRUE(selected);
      }
    }
  }
}

// The freshness shapes of make_boundary_chain_plan on a manual clock: the chain root ticks every
// millisecond from 1 ms, and the side source in the other partition ticks at the quarter points
// between root ticks, publishing only what the test grants. With `reuse_last`, the LatestWithin
// owner reuses its last side sample when no new one is eligible.
[[nodiscard]] ExecutionPlan make_scheduled_freshness_plan(BoundaryShape shape, bool reuse_last,
                                                          std::shared_ptr<ManualClock> clock) {
  Graph graph{"scheduled-freshness-chain"};
  graph.set_default_clock(graph.bind_shared_clock("clock", std::move(clock)));
  // The graph's only clock.
  const ClockId clock_id{1U};
  const TimePoint root_start{.timestamp_ns = 1'000'000, .clock_id = clock_id};
  const TimePoint side_start{.timestamp_ns = 250'000, .clock_id = clock_id};
  const ConnectionOptions depth_one{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U};
  const auto side_source =
      graph.op<ScheduledSideSource>("side-source", std::chrono::microseconds{500}, side_start);
  const auto relay = graph.op<BoundaryChainMember>("relay", false);
  const auto sink = graph.op<BoundaryChainMember>("sink", true);
  if (shape == BoundaryShape::kFreshnessSideInput) {
    const auto root = graph.op<BoundaryChainSideRoot>("root", root_start, reuse_last);
    const auto first = graph.op<BoundaryChainMember>("first", false);
    graph.add_flow(side_source->output, root->side, depth_one);
    graph.add_flow(root->output, first->input, depth_one);
    graph.add_flow(first->output, relay->input, depth_one);
    graph.add_flow(relay->output, sink->input, depth_one);
    graph.partition("control").add(root).add(first).add(relay).add(sink);
  } else {
    const auto source = graph.op<BoundaryChainSource>(
        "source", std::chrono::microseconds{1000}, true, false, root_start);
    const auto member = graph.op<BoundaryChainLatestMember>("member", reuse_last);
    graph.add_flow(source->output, member->input, depth_one);
    graph.add_flow(side_source->output, member->side, depth_one);
    graph.add_flow(member->output, relay->input, depth_one);
    graph.add_flow(relay->output, sink->input, depth_one);
    graph.partition("control").add(source).add(member).add(relay).add(sink);
  }
  graph.partition("side").add(side_source);
  return compile(graph);
}

struct ScheduledFreshnessRun {
  bool stepped{};  // every scheduled activation finished in time
  RunTermination termination{};
  std::size_t diagnostics{};
  std::vector<std::uint32_t> sunk;
  std::uint64_t attempts{};
  std::uint64_t accepted{};
  std::vector<SideSelection> selections;  // one per activation of the LatestWithin owner
};

// Steps the manual clock through each side slot and root tick in order and waits for each
// activation to finish, so every root tick sees exactly the side samples granted before it.
// `grants[k]` holds the publications granted at the two side slots before root tick k + 1.
[[nodiscard]] ScheduledFreshnessRun run_scheduled_freshness(
    BoundaryShape shape, bool reuse_last, bool disable_chain,
    std::span<const std::array<bool, 2U>> grants) {
  {
    const std::lock_guard lock(BoundaryChainMember::mutex);
    BoundaryChainMember::sunk.clear();
  }
  {
    const std::lock_guard lock(BoundaryChainLatestMember::mutex);
    BoundaryChainLatestMember::activations.clear();
  }
  {
    const std::lock_guard lock(BoundaryChainSideRoot::mutex);
    BoundaryChainSideRoot::selections.clear();
  }
  ScheduledSideSource::grants.store(0U);
  ScheduledSideSource::ticks.store(0U);
  ChainRootLog::reset();
  const auto clock = std::make_shared<ManualClock>(0);
  const ExecutionPlan compiled = make_scheduled_freshness_plan(shape, reuse_last, clock);
  EXPECT_TRUE(compiled.ok()) << compiled.json();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_phase_counters(compiled, counters);
  if (disable_chain) {
    plan = detail::RuntimeTestPlanAccess::disable_boundary_causal_chain_for_test(plan);
  }
  const auto wait_until = [](const auto& done) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!done()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(100us);
    }
    return true;
  };
  const auto sunk_count = [] {
    const std::lock_guard lock(BoundaryChainMember::mutex);
    return BoundaryChainMember::sunk.size();
  };
  RunSession session = run_async(plan);
  ScheduledFreshnessRun run{.stepped = true};
  std::uint64_t side_ticks = 0U;
  for (std::size_t tick = 0U; tick < grants.size() && run.stepped; ++tick) {
    const std::int64_t tick_ns = static_cast<std::int64_t>(tick + 1U) * 1'000'000;
    for (std::size_t slot = 0U; slot < 2U && run.stepped; ++slot) {
      ScheduledSideSource::grants.store(grants[tick][slot] ? 1U : 0U);
      clock->advance_to(tick_ns - 750'000 + static_cast<std::int64_t>(slot) * 500'000);
      ++side_ticks;
      run.stepped = wait_until([&] { return ScheduledSideSource::ticks.load() >= side_ticks; });
    }
    if (run.stepped) {
      clock->advance_to(tick_ns);
      run.stepped = wait_until([&] { return sunk_count() >= tick + 1U; });
    }
  }
  session.request_stop();
  session.wait();
  const auto snapshot = counters->snapshot();
  run.termination = session.termination();
  run.diagnostics = session.diagnostics().size();
  run.attempts = snapshot[detail::RuntimePhaseCounters::index(
      detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts)];
  run.accepted = snapshot[detail::RuntimePhaseCounters::index(
      detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted)];
  {
    const std::lock_guard lock(BoundaryChainMember::mutex);
    run.sunk = BoundaryChainMember::sunk;
  }
  if (shape == BoundaryShape::kFreshnessSideInput) {
    const std::lock_guard lock(BoundaryChainSideRoot::mutex);
    run.selections = BoundaryChainSideRoot::selections;
  } else {
    const std::lock_guard lock(BoundaryChainLatestMember::mutex);
    for (const LatestMemberActivation& activation : BoundaryChainLatestMember::activations) {
      run.selections.push_back(
          {.value = activation.value, .state = activation.state, .side = activation.side});
    }
  }
  return run;
}

// The same freshness shapes on a manual clock, with every side publication placed between two
// chosen root ticks. LatestWithin consumes the sample it selects, so a tick without a new
// publication takes the missing policy: the degraded path, or the retained sample under
// ReuseLast. A tick after new publications must select the newest one over both the superseded
// and the retained sample. Every tick must make exactly the expected selection with the causal
// chain and without it, and the chain must continue the same hops on every tick.
// REQ: RUNTIME-086
TEST(RuntimeInlinePartitionExecution, BoundaryChainKeepsExactFreshnessSelections) {
  // The publications granted at the two side slots before each root tick.
  constexpr std::array<std::array<bool, 2U>, 8U> kGrants{{
      {true, false},   // tick 1: sample 1, so ReuseLast always has a sample to retain
      {false, false},  // tick 2: none; ReuseLast retains sample 1
      {true, true},    // tick 3: samples 2 and 3 behind the retained sample 1
      {false, true},   // tick 4: sample 4, late in the period
      {false, false},  // tick 5: none; ReuseLast retains sample 4
      {false, false},  // tick 6: none again
      {true, true},    // tick 7: samples 5 and 6
      {true, false},   // tick 8: sample 7, early in the period
  }};
  struct ExpectedSelection {
    InputState state{};
    std::optional<std::uint32_t> side;
  };
  for (const bool reuse_last : {false, true}) {
    // The newest sample published since the previous tick, else the retained sample under
    // ReuseLast, else the degraded path.
    std::vector<ExpectedSelection> expected;
    std::uint32_t published = 0U;
    std::optional<std::uint32_t> retained;
    for (const auto& slots : kGrants) {
      const auto added =
          static_cast<std::uint32_t>(slots[0]) + static_cast<std::uint32_t>(slots[1]);
      published += added;
      if (added > 0U) {
        retained = published;
        expected.push_back({.state = InputState::kSelected, .side = retained});
      } else if (reuse_last) {
        expected.push_back({.state = InputState::kReused, .side = retained});
      } else {
        expected.push_back({.state = InputState::kSafeDefault, .side = std::nullopt});
      }
    }
    for (const BoundaryShape shape :
         {BoundaryShape::kFreshnessSideInput, BoundaryShape::kFreshnessMemberSideInput}) {
      // Every hop after the LatestWithin owner: three after the root, two after the member.
      const std::uint64_t hops = shape == BoundaryShape::kFreshnessSideInput ? 3U : 2U;
      for (const bool disable_chain : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "reuse_last=" << reuse_last << " shape=" << static_cast<unsigned>(shape)
                     << " disable_chain=" << disable_chain);
        const ScheduledFreshnessRun run =
            run_scheduled_freshness(shape, reuse_last, disable_chain, kGrants);
        ASSERT_TRUE(run.stepped) << "sunk=" << run.sunk.size();
        EXPECT_EQ(run.termination, RunTermination::kStopped);
        EXPECT_EQ(run.diagnostics, 0U);
        ASSERT_EQ(run.selections.size(), kGrants.size());
        for (std::size_t tick = 0U; tick < kGrants.size(); ++tick) {
          SCOPED_TRACE(testing::Message() << "tick " << tick + 1U);
          EXPECT_EQ(run.selections[tick].value, tick + 1U);
          // Key assertion: each tick makes exactly the expected selection.
          EXPECT_EQ(run.selections[tick].state, expected[tick].state);
          EXPECT_EQ(run.selections[tick].side, expected[tick].side);
        }
        // Key assertion: the chain continues every hop after the owner on every tick, but never
        // the hop into the LatestWithin member. The control arm attempts none.
        const std::uint64_t continued = disable_chain ? 0U : hops * kGrants.size();
        EXPECT_EQ(run.attempts, continued);
        EXPECT_EQ(run.accepted, continued);
      }
    }
  }
}

// The decision helper uses the same outgoing-only topology and terminal/failure inputs as the
// receipt path. Input-terminal work is checked independently of completion work.
// REQ: RUNTIME-070
TEST(RuntimeInlinePartitionExecution, OutgoingOnlyBoundaryChainGuardsFailureAndTerminalWork) {
  detail::PartitionRunnerPlanRecord runner;
  runner.outgoing_boundary_edges.push_back(EdgeId{1U});
  std::atomic_bool failed{false};
  EXPECT_FALSE(detail::boundary_chain_requires_normal_scan(&runner, {}, &failed, false, false));
  EXPECT_TRUE(detail::boundary_chain_requires_normal_scan(&runner, {}, &failed, true, false));
  EXPECT_TRUE(detail::boundary_chain_requires_normal_scan(&runner, {}, &failed, false, true));
  failed.store(true, std::memory_order_release);
  EXPECT_TRUE(detail::boundary_chain_requires_normal_scan(&runner, {}, &failed, false, false));
  runner.outgoing_boundary_edges.clear();
  EXPECT_FALSE(detail::boundary_chain_requires_normal_scan(&runner, {}, &failed, true, true));
  EXPECT_FALSE(detail::boundary_chain_requires_normal_scan(nullptr, {}, &failed, true, true));
}

struct BoundaryGuardProbe {
  static inline std::atomic<bool> fail_peer{};
  static inline std::atomic<std::size_t> relay_entries{};
};

class BoundaryGuardRoot final : public Operator<> {
 public:
  explicit BoundaryGuardRoot(bool finite) : finite_(finite) {}
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    if (finite_) {
      result.complete_after(ComputeCalls{.limit = 1U});
    }
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override { return output.emit(++value_); }
  Output<std::uint32_t> output;

 private:
  bool finite_{};
  std::uint32_t value_{};
};

class BoundaryGuardRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    BoundaryGuardProbe::relay_entries.fetch_add(1U, std::memory_order_relaxed);
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    return output.emit(sample->data);
  }
  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

class BoundaryGuardFailurePeer final : public Operator<> {
 public:
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    return BoundaryGuardProbe::fail_peer.load(std::memory_order_acquire)
               ? expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})}
               : expected<void, Error>{};
  }
};

ExecutionPlan make_outgoing_boundary_guard_plan(bool finite) {
  BoundaryGuardProbe::fail_peer.store(false, std::memory_order_release);
  BoundaryGuardProbe::relay_entries.store(0U, std::memory_order_relaxed);
  {
    const std::lock_guard lock{BoundaryChainMember::mutex};
    BoundaryChainMember::sunk.clear();
  }
  Graph graph{"outgoing-boundary-causal-guard"};
  auto root = graph.op<BoundaryGuardRoot>("root", finite);
  auto first = graph.op<BoundaryGuardRelay>("first");
  auto tail = graph.op<BoundaryGuardRelay>("tail");
  auto sink = graph.op<BoundaryChainMember>("sink", true);
  const ConnectionOptions options{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U};
  graph.add_flow(root->output, first->input, options);
  graph.add_flow(first->output, tail->input, options);
  graph.add_flow(tail->output, sink->input, options);
  graph.partition("chain").add(root).add(first).add(tail);
  graph.partition("sink").add(sink);
  if (!finite) {
    auto peer = graph.op<BoundaryGuardFailurePeer>("failing-peer");
    graph.partition("peer").add(peer);
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  return compile(graph);
}

// REQ: RUNTIME-070
// REQ: RUNTIME-041
TEST(RuntimeInlinePartitionExecution, OutgoingOnlyChainFallsBackBeforeEnteringAfterPeerFailure) {
  ExecutionPlan plan = make_outgoing_boundary_guard_plan(false);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_TRUE(state);
  const auto runner = std::ranges::find_if(state->partition_runners, [](const auto& candidate) {
    return candidate.operators.size() == 3U;
  });
  ASSERT_NE(runner, state->partition_runners.end());
  ASSERT_TRUE(runner->incoming_boundary_edges.empty());
  ASSERT_EQ(runner->outgoing_boundary_edges.size(), 1U);
  const auto control = std::make_shared<detail::RuntimeDirectContinuationTestControl>();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  plan = detail::RuntimeTestPlanAccess::with_direct_continuation_test_control(plan, control);
  RunSession session = run_async(plan);
  const bool staged = control->wait_until_receipt_staged();
  if (staged) {
    BoundaryGuardProbe::fail_peer.store(true, std::memory_order_release);
  }
  const bool failure_published = staged && control->wait_until_group_failure_published();
  control->release();
  const bool terminal = session.wait_for(5s);
  if (!terminal) {
    session.request_stop();
  }
  session.wait();
  ASSERT_TRUE(staged);
  ASSERT_TRUE(failure_published);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(BoundaryGuardProbe::relay_entries.load(std::memory_order_relaxed), 0U);
  EXPECT_TRUE(std::ranges::any_of(session.diagnostics(), [](const auto& diagnostic) {
    return diagnostic.token == "RUNTIME_OPERATOR_CALLBACK_FAILED";
  }));
  const auto observation = control->boundary_fallback_observation();
  ASSERT_TRUE(observation.recorded);
  EXPECT_TRUE(observation.group_failed);
  const auto snapshot = counters->snapshot();
  EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(
                detail::RuntimePhaseCounter::kPartitionCausalFrontierBoundaryFallbacks)],
            1U);
  EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(
                detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted)],
            0U);
}

// REQ: RUNTIME-070
TEST(RuntimeInlinePartitionExecution, OutgoingOnlyChainPreservesReceiptForCompletionDrain) {
  ExecutionPlan plan = make_outgoing_boundary_guard_plan(true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto control = std::make_shared<detail::RuntimeDirectContinuationTestControl>();
  plan = detail::RuntimeTestPlanAccess::with_direct_continuation_test_control(plan, control);
  RunSession session = run_async(plan);
  const bool staged = control->wait_until_receipt_staged();
  control->release();
  const bool fallback = control->wait_until_boundary_fallback();
  const bool terminal = session.wait_for(5s);
  if (!terminal) {
    session.request_stop();
  }
  session.wait();
  ASSERT_TRUE(staged);
  ASSERT_TRUE(fallback);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto observation = control->boundary_fallback_observation();
  EXPECT_FALSE(observation.group_failed);
  EXPECT_TRUE(observation.completion_pending);
  EXPECT_EQ(BoundaryGuardProbe::relay_entries.load(std::memory_order_relaxed), 2U);
  const std::lock_guard lock{BoundaryChainMember::mutex};
  EXPECT_EQ(BoundaryChainMember::sunk, (std::vector<std::uint32_t>{1U}));
}

// A chain continues into a member whose other inputs are plain DrainAll side inputs, through a
// fanout-one edge or a direct fanout family whose other recipient is in another partition, as in
// the robot's ReadJointState -> MotionPlanner hop. The continued member takes its parent's logical
// coordinate and drains the side samples already witnessed. The control arm keeps
// that hop on the ordinary scan; the hops after the member continue either way.
// REQ: RUNTIME-084
TEST(RuntimeInlinePartitionExecution, ChainContinuesIntoASideInputMember) {
  constexpr std::size_t kTicks = 30U;
  for (const BoundaryShape shape : {BoundaryShape::kSideInput, BoundaryShape::kFanoutSideInput}) {
    const ExecutionPlan compiled = make_boundary_chain_plan(shape);
    ASSERT_TRUE(compiled.ok()) << compiled.json();
    for (const bool disable_side_input_continuation : {false, true}) {
      SCOPED_TRACE(testing::Message() << "shape=" << static_cast<unsigned>(shape)
                                      << " disable=" << disable_side_input_continuation);
      const BoundaryChainRun run =
          run_boundary_chain(compiled, false, kTicks, disable_side_input_continuation);
      EXPECT_EQ(run.termination, RunTermination::kStopped);
      EXPECT_EQ(run.diagnostics, 0U);
      ASSERT_GE(run.sunk.size(), kTicks);
      EXPECT_TRUE(std::ranges::is_sorted(run.sunk));
      EXPECT_EQ(std::ranges::adjacent_find(run.sunk), run.sunk.end());
      const auto continued = static_cast<std::size_t>(
          std::ranges::count_if(run.member_activations, &SideMemberActivation::continued));
      const std::size_t side_samples = std::accumulate(
          run.member_activations.begin(),
          run.member_activations.end(),
          std::size_t{},
          [](std::size_t total, const auto& activation) { return total + activation.side.size(); });
      // The side source publishes every 3 ms, so a 30-tick run drains several of its samples.
      EXPECT_GT(side_samples, 0U);
      if (shape == BoundaryShape::kFanoutSideInput) {
        EXPECT_GT(run.remote_received, 0U);
      }
      if (disable_side_input_continuation) {
        EXPECT_EQ(continued, 0U);
      } else {
        // Key assertion: most frames continue into the member, each at its root's logical
        // coordinate and the next microstep; a frame whose side input changed after the root was
        // armed takes the ordinary scan instead.
        EXPECT_GE(continued, kTicks / 2U) << "member activations=" << run.member_activations.size()
                                          << " fallbacks=" << run.fallbacks;
        for (const SideMemberActivation& activation : run.member_activations) {
          if (activation.continued()) {
            EXPECT_TRUE(activation.next_microstep())
                << "value " << activation.value << " microstep " << activation.tag_time.microstep;
          }
        }
      }
    }
  }
}

// A side publication that lands while the root computes leaves the member's side input with an
// unobserved change, so that frame's hop into the member takes the ordinary scan, which polls the
// side arrival with the trigger and drains it in the same activation.
// REQ: RUNTIME-084
TEST(RuntimeInlinePartitionExecution, SideArrivalDuringTheProducerKeepsTheMemberHopOrdinary) {
  constexpr std::size_t kTicks = SideRaceGate::kRacedFrame + 10U;
  const ExecutionPlan compiled = make_boundary_chain_plan(BoundaryShape::kRacingSideInput);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  const BoundaryChainRun run = run_boundary_chain(compiled, false, kTicks);
  EXPECT_EQ(run.termination, RunTermination::kStopped);
  EXPECT_EQ(run.diagnostics, 0U);
  ASSERT_GE(run.sunk.size(), kTicks);
  ASSERT_TRUE(SideRaceGate::published.load());
  const auto raced = std::ranges::find(
      run.member_activations, SideRaceGate::kRacedFrame, &SideMemberActivation::value);
  ASSERT_NE(raced, run.member_activations.end());
  // Key assertion: the raced frame was frozen by the ordinary scan and drained the side sample.
  EXPECT_FALSE(raced->continued());
  EXPECT_EQ(raced->side, std::vector<std::uint32_t>{SideRaceGate::kSideValue});
  EXPECT_TRUE(std::ranges::any_of(run.member_activations, &SideMemberActivation::continued));
}

}  // namespace
}  // namespace holoscan
