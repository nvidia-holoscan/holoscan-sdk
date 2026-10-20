/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/rate_generation.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// Reading guide
// -------------
// Each run_* function is one small lesson. The example deliberately separates
// four moments that are easy to conflate:
//
//   setup() authoring -> compile-time recipe -> published rate generation
//                     -> optional restart generation
//
// A nominal rate is declarative sizing evidence; contract() still controls
// when compute() may run. Lifecycle-selected rates remain pending until their
// global stage barrier closes. A restart publishes a new rate/binding
// generation but may retain compatible physical allocation.
//
// The README derives the queue and pool numbers and gives an API choice table.

constexpr DLDataType kScalarTensorType{kDLUInt, 64U, 1U};

[[nodiscard]] bool fail(std::string_view scenario, std::string_view message) {
  std::cerr << "rate_aware_ports " << scenario << ": " << message << '\n';
  return false;
}

[[nodiscard]] const holoscan::PortPlanRecord* find_port(const holoscan::ExecutionPlan& plan,
                                                        std::string_view owner,
                                                        std::string_view name,
                                                        holoscan::PortDirection direction) {
  const auto vertex =
      std::ranges::find(plan.graph().vertices(), owner, &holoscan::VertexRecord::name);
  if (vertex == plan.graph().vertices().end()) {
    return nullptr;
  }
  const auto port = std::ranges::find_if(plan.ports(), [&](const auto& candidate) {
    return candidate.owner == vertex->id && candidate.local_name == name &&
           candidate.direction == direction;
  });
  return port == plan.ports().end() ? nullptr : &*port;
}

[[nodiscard]] const holoscan::PayloadEdgePlanRecord* find_edge(const holoscan::ExecutionPlan& plan,
                                                               std::string_view source_owner,
                                                               std::string_view source_name,
                                                               std::string_view target_owner,
                                                               std::string_view target_name) {
  const auto* source = find_port(plan, source_owner, source_name, holoscan::PortDirection::kOutput);
  const auto* target = find_port(plan, target_owner, target_name, holoscan::PortDirection::kInput);
  if (source == nullptr || target == nullptr) {
    return nullptr;
  }
  const auto edge = std::ranges::find_if(plan.payload_edges(), [&](const auto& candidate) {
    return candidate.source == source->id && candidate.target == target->id;
  });
  return edge == plan.payload_edges().end() ? nullptr : &*edge;
}

[[nodiscard]] const holoscan::QueueSizingRecipePlanRecord* find_recipe(
    const holoscan::ExecutionPlan& plan, holoscan::EdgeId edge) {
  const auto recipe = std::ranges::find(
      plan.queue_sizing_recipes(), edge, &holoscan::QueueSizingRecipePlanRecord::edge);
  return recipe == plan.queue_sizing_recipes().end() ? nullptr : &*recipe;
}

[[nodiscard]] const holoscan::PortRateGenerationRecord* find_rate(
    const holoscan::RateGenerationSnapshot& snapshot,
    const holoscan::RateGenerationEvidenceRecord& generation, holoscan::PortId port) {
  const auto records = snapshot.port_rates(generation);
  const auto record = std::ranges::find(records, port, &holoscan::PortRateGenerationRecord::port);
  return record == records.end() ? nullptr : &*record;
}

[[nodiscard]] const holoscan::QueueSizingGenerationRecord* find_queue(
    const holoscan::RateGenerationSnapshot& snapshot,
    const holoscan::RateGenerationEvidenceRecord& generation,
    holoscan::QueueSizingRecipeId recipe) {
  const auto records = snapshot.queue_sizings(generation);
  const auto record =
      std::ranges::find(records, recipe, &holoscan::QueueSizingGenerationRecord::recipe);
  return record == records.end() ? nullptr : &*record;
}

[[nodiscard]] std::optional<holoscan::RateGenerationSnapshot> wait_for_published_snapshot(
    holoscan::RunSession& session) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto snapshot = session.capture_rate_generations();
    if (snapshot && snapshot->published() != nullptr) {
      return std::move(*snapshot);
    }
    std::this_thread::sleep_for(1ms);
  }
  return std::nullopt;
}

[[nodiscard]] bool wait_for(const std::atomic<bool>& value) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!value.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return value.load(std::memory_order_acquire);
}

class FixedSixtySource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    // Fixed producer: author the rate directly on the output declaration.
    spec.output(output, "output").nominal_rate(holoscan::Rate::hz(60U));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    // Scheduling remains separate: the 10 ms trigger does not establish the
    // nominal 60 Hz declaration above.
    result.trigger(holoscan::OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<std::uint32_t> output;
};

class FollowsRelay final : public holoscan::Operator<> {
 public:
  explicit FollowsRelay(holoscan::RateRatio ratio) noexcept : ratio_(ratio) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input");
    // Derived producer: publish an exact rational multiple of the input rate.
    spec.output(output, "output").follows(input, ratio_);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input;
  holoscan::Output<std::uint32_t> output;

 private:
  holoscan::RateRatio ratio_;
};

class SlowClockSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1s})
        .input(input, holoscan::Exact{})
        .on_missing(input, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

[[nodiscard]] bool run_declarations() {
  // Lesson 1: fixed and followed rates form compile-time relationships, while
  // the runtime publishes their resolved values as one coherent generation.
  holoscan::Graph graph{"rate-declarations"};
  const auto source = graph.op<FixedSixtySource>("fixed_source");
  const auto one_to_one = graph.op<FollowsRelay>("one_to_one", holoscan::RateRatio{1U, 1U});
  const auto one_to_four = graph.op<FollowsRelay>("one_to_four", holoscan::RateRatio{1U, 4U});
  const auto sink = graph.op<SlowClockSink>("sink");
  graph.add_flow(source->output, one_to_one->input);
  graph.add_flow(one_to_one->output, one_to_four->input);
  graph.add_flow(one_to_four->output, sink->input);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("rate-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    return fail("declarations", plan.json());
  }
  const auto* fixed = find_port(plan, "fixed_source", "output", holoscan::PortDirection::kOutput);
  const auto* follows = find_port(plan, "one_to_one", "output", holoscan::PortDirection::kOutput);
  const auto* quarter = find_port(plan, "one_to_four", "output", holoscan::PortDirection::kOutput);
  if (fixed == nullptr || follows == nullptr || quarter == nullptr) {
    return fail("declarations", "compiled ports were not found");
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  auto snapshot = wait_for_published_snapshot(session);
  session.request_stop();
  session.wait();
  if (!snapshot || snapshot->published() == nullptr ||
      session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    return fail("declarations", "published rate snapshot was unavailable");
  }
  const auto* fixed_rate = find_rate(*snapshot, *snapshot->published(), fixed->id);
  const auto* follows_rate = find_rate(*snapshot, *snapshot->published(), follows->id);
  const auto* quarter_rate = find_rate(*snapshot, *snapshot->published(), quarter->id);
  if (fixed_rate == nullptr || follows_rate == nullptr || quarter_rate == nullptr ||
      fixed_rate->resolved_rate != holoscan::Rate::hz(60U) ||
      fixed_rate->resolved_origin != holoscan::ResolvedRateOrigin::kOutputNominal ||
      follows_rate->resolved_rate != holoscan::Rate::hz(60U) ||
      follows_rate->resolved_origin != holoscan::ResolvedRateOrigin::kFollowsExpression ||
      quarter_rate->resolved_rate != holoscan::Rate::hz(15U) ||
      quarter_rate->resolved_origin != holoscan::ResolvedRateOrigin::kFollowsExpression) {
    return fail("declarations", "resolved fixed/follows rates did not match 60, 60, and 15 Hz");
  }

  std::cout << "rate_aware_ports declarations: PASS fixed=60/1 follows=60/1 quarter=15/1"
               " scheduling=contract-only\n";
  return true;
}

class KilohertzSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "sample").nominal_rate(holoscan::Rate::hz(1000U));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<std::uint32_t> output;
};

class CameraFusion final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(camera, "camera").expected_rate(holoscan::Rate::hz(60U));
    spec.input(imu, "imu").expected_rate(holoscan::Rate::hz(1000U));
    spec.output(fused, "fused").follows(camera);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    // One camera sample defines the activation horizon; every IMU sample in
    // that horizon is drained. These consumption semantics close queue sizing.
    result.trigger(holoscan::OnEach{camera}).input(imu, holoscan::DrainAll{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> camera;
  holoscan::Input<std::uint32_t> imu;
  holoscan::Output<std::uint32_t> fused;
};

[[nodiscard]] holoscan::ExecutionPlan compile_fusion() {
  holoscan::Graph graph{"camera-imu-automatic"};
  const auto camera = graph.op<FixedSixtySource>("camera_source");
  const auto imu = graph.op<KilohertzSource>("imu_source");
  const auto fusion = graph.op<CameraFusion>("fusion");
  graph.add_flow(camera->output, fusion->camera);
  graph.add_flow(imu->output, fusion->imu);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("fusion-clock"));
  return holoscan::compile(graph);
}

[[nodiscard]] holoscan::ExecutionPlan compile_explicit_override() {
  holoscan::Graph graph{"exact-depth-override"};
  const auto source = graph.op<KilohertzSource>("source");
  const auto sink = graph.op<SlowClockSink>("sink");
  graph.add_flow(source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 48U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("explicit-clock"));
  return holoscan::compile(graph);
}

[[nodiscard]] bool run_fusion() {
  // Lesson 2: omitted depth is rate-and-contract derived. An explicitly
  // authored depth remains exact and is never replaced by that recommendation.
  const holoscan::ExecutionPlan automatic = compile_fusion();
  const holoscan::ExecutionPlan explicit_plan = compile_explicit_override();
  if (!automatic.ok() || !explicit_plan.ok()) {
    return fail("fusion", automatic.ok() ? explicit_plan.json() : automatic.json());
  }
  const auto* automatic_edge = find_edge(automatic, "imu_source", "sample", "fusion", "imu");
  const auto* explicit_edge = find_edge(explicit_plan, "source", "sample", "sink", "input");
  const auto* automatic_recipe =
      automatic_edge == nullptr ? nullptr : find_recipe(automatic, automatic_edge->edge);
  const auto* explicit_recipe =
      explicit_edge == nullptr ? nullptr : find_recipe(explicit_plan, explicit_edge->edge);
  if (automatic_edge == nullptr || explicit_edge == nullptr || automatic_recipe == nullptr ||
      explicit_recipe == nullptr || automatic_edge->queue_depth != 34U ||
      automatic_recipe->origin != holoscan::QueueDepthOrigin::kRateAndContractDerived ||
      automatic_recipe->explicit_depth.has_value() || explicit_edge->queue_depth != 48U ||
      explicit_recipe->origin != holoscan::QueueDepthOrigin::kAddFlowOption ||
      explicit_recipe->explicit_depth != 48U) {
    return fail("fusion", "automatic depth 34 or explicit depth 48 evidence was incomplete");
  }

  std::cout << "rate_aware_ports fusion: PASS camera=60/1 imu=1000/1"
               " automatic_logical_depth=34 explicit_logical_depth=48"
               " physical_ring=provider-specific\n";
  return true;
}

class UndeclaredSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<std::uint32_t> output;
};

class ExpectedRateSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    // expected_rate() is a fallback expectation for unresolved ingress. It is
    // weaker than valid rate evidence propagated from a connected producer.
    spec.input(input, "input").expected_rate(holoscan::Rate::hz(60U));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1s})
        .input(input, holoscan::Exact{})
        .on_missing(input, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

[[nodiscard]] bool run_fallback() {
  // Lesson 3: this source genuinely omits rate metadata, so the receiver's
  // expectation supplies fallback provenance rather than overriding upstream.
  holoscan::Graph graph{"unresolved-ingress-fallback"};
  const auto source = graph.op<UndeclaredSource>("undeclared_ingress");
  const auto sink = graph.op<ExpectedRateSink>("receiver");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("fallback-clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  const auto* input = find_port(plan, "receiver", "input", holoscan::PortDirection::kInput);
  const auto* edge = find_edge(plan, "undeclared_ingress", "output", "receiver", "input");
  if (!plan.ok() || input == nullptr || edge == nullptr || edge->queue_depth != 120U) {
    return fail("fallback", plan.ok() ? "expected depth 120 was absent" : plan.json());
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  auto snapshot = wait_for_published_snapshot(session);
  session.request_stop();
  session.wait();
  if (!snapshot || snapshot->published() == nullptr ||
      session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    return fail("fallback", "published fallback snapshot was unavailable");
  }
  const auto* rate = find_rate(*snapshot, *snapshot->published(), input->id);
  if (rate == nullptr || rate->resolved_rate != holoscan::Rate::hz(60U) ||
      rate->resolved_origin != holoscan::ResolvedRateOrigin::kInputExpectationFallback) {
    return fail("fallback", "expected_rate did not retain fallback provenance");
  }

  std::cout << "rate_aware_ports fallback: PASS upstream=undeclared expected=60/1"
               " origin=input_expectation_fallback depth=120\n";
  return true;
}

struct DiscoverQueryState {
  static void reset() noexcept {
    source_pending.store(false, std::memory_order_release);
    source_compute_resolved.store(false, std::memory_order_release);
    relay_allocate_resolved.store(false, std::memory_order_release);
    relay_compute_resolved.store(false, std::memory_order_release);
    failed.store(false, std::memory_order_release);
  }

  static inline std::atomic<bool> source_pending{};
  static inline std::atomic<bool> source_compute_resolved{};
  static inline std::atomic<bool> relay_allocate_resolved{};
  static inline std::atomic<bool> relay_compute_resolved{};
  static inline std::atomic<bool> failed{};
};

[[nodiscard]] bool source_rate_is_resolved(const holoscan::PortRateView& view) noexcept {
  return view.state == holoscan::PortRateViewState::kResolved &&
         view.resolved_nominal_rate == holoscan::Rate::hz(60U) &&
         view.max_nominal_rate == holoscan::Rate::hz(120U) &&
         view.origin == holoscan::ResolvedRateOrigin::kLifecycleBinding;
}

class DiscoverSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    // The selected value is unknown at compose time, but the declaration fixes
    // both its binding barrier and the maximum admitted allocation domain.
    spec.output(output, "output")
        .lifecycle_nominal_rate(holoscan::LifecycleStage::kDiscover)
        .max_nominal_rate(holoscan::Rate::hz(120U));
    spec.lifecycle().stage(holoscan::LifecycleStage::kDiscover, &DiscoverSource::discover);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms})
        .complete_after(holoscan::ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const auto rate = context.rate(output);
    if (!rate || !source_rate_is_resolved(*rate)) {
      DiscoverQueryState::failed.store(true, std::memory_order_release);
      return holoscan::make_unexpected(rate ? holoscan::Error{holoscan::ErrorCode::kFailure}
                                            : rate.error());
    }
    DiscoverQueryState::source_compute_resolved.store(true, std::memory_order_release);
    return output.emit(7U);
  }

  holoscan::LifecycleStatus discover(holoscan::LifecycleContext& context) noexcept {
    const auto bound = context.bind_nominal_rate(output, holoscan::Rate::hz(60U));
    // Binding records candidate state. The same-stage query remains pending;
    // publication occurs atomically only after every Discover callback passes.
    const auto pending = context.rate(output);
    if (!bound || !pending || pending->state != holoscan::PortRateViewState::kPendingBarrier ||
        pending->resolved_nominal_rate.has_value()) {
      DiscoverQueryState::failed.store(true, std::memory_order_release);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    DiscoverQueryState::source_pending.store(true, std::memory_order_release);
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::Output<std::uint32_t> output;
};

[[nodiscard]] bool relay_rates_are_resolved(const holoscan::PortRateView& input,
                                            const holoscan::PortRateView& output) noexcept {
  return input.state == holoscan::PortRateViewState::kResolved &&
         input.resolved_nominal_rate == holoscan::Rate::hz(60U) &&
         input.max_nominal_rate == holoscan::Rate::hz(120U) &&
         input.origin == holoscan::ResolvedRateOrigin::kConnectedOutput &&
         output.state == holoscan::PortRateViewState::kResolved &&
         output.resolved_nominal_rate == holoscan::Rate::hz(60U) &&
         output.max_nominal_rate == holoscan::Rate::hz(120U) &&
         output.origin == holoscan::ResolvedRateOrigin::kFollowsExpression;
}

class DiscoverRelay final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").expected_rate(holoscan::Rate::hz(60U));
    spec.output(output, "output").follows(input);
    spec.lifecycle().stage(holoscan::LifecycleStage::kAllocate, &DiscoverRelay::allocate);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    const auto input_rate = context.rate(input);
    const auto output_rate = context.rate(output);
    if (!sample || !input_rate || !output_rate ||
        !relay_rates_are_resolved(*input_rate, *output_rate)) {
      DiscoverQueryState::failed.store(true, std::memory_order_release);
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    DiscoverQueryState::relay_compute_resolved.store(true, std::memory_order_release);
    return {};
  }

  holoscan::LifecycleStatus allocate(holoscan::LifecycleContext& context) noexcept {
    const auto input_rate = context.rate(input);
    const auto output_rate = context.rate(output);
    if (!input_rate || !output_rate || !relay_rates_are_resolved(*input_rate, *output_rate)) {
      DiscoverQueryState::failed.store(true, std::memory_order_release);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    DiscoverQueryState::relay_allocate_resolved.store(true, std::memory_order_release);
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::Input<std::uint32_t> input;
  holoscan::Output<std::uint32_t> output;
};

[[nodiscard]] bool run_discover() {
  // Lesson 4: observe pending state inside Discover, then resolved propagated
  // state in the later Allocate barrier and steady-state compute callback.
  DiscoverQueryState::reset();
  holoscan::Graph graph{"discover-rate-propagation"};
  const auto source = graph.op<DiscoverSource>("camera");
  const auto relay = graph.op<DiscoverRelay>("preprocess");
  graph.add_flow(source->output, relay->input);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("discover-clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    return fail("discover", plan.json());
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool observed = wait_for(DiscoverQueryState::relay_compute_resolved);
  session.request_stop();
  session.wait();
  if (!observed || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty() ||
      DiscoverQueryState::failed.load(std::memory_order_acquire) ||
      !DiscoverQueryState::source_pending.load(std::memory_order_acquire) ||
      !DiscoverQueryState::source_compute_resolved.load(std::memory_order_acquire) ||
      !DiscoverQueryState::relay_allocate_resolved.load(std::memory_order_acquire) ||
      !DiscoverQueryState::relay_compute_resolved.load(std::memory_order_acquire)) {
    return fail("discover", "barrier propagation or lifecycle/compute query failed");
  }

  std::cout << "rate_aware_ports discover: PASS selected=60/1 maximum=120/1"
               " same_stage=pending downstream_allocate=resolved compute_query=resolved\n";
  return true;
}

struct LateBindingState {
  static void reset() noexcept {
    allocate_pending.store(false, std::memory_order_release);
    binding_pending.store(false, std::memory_order_release);
    start_resolved.store(false, std::memory_order_release);
    above_maximum_rejected.store(false, std::memory_order_release);
    failed.store(false, std::memory_order_release);
  }

  static inline std::atomic<bool> allocate_pending{};
  static inline std::atomic<bool> binding_pending{};
  static inline std::atomic<bool> start_resolved{};
  static inline std::atomic<bool> above_maximum_rejected{};
  static inline std::atomic<bool> failed{};
};

class LateBindingTensorSource final : public holoscan::Operator<> {
 public:
  LateBindingTensorSource(holoscan::LifecycleStage stage, holoscan::Rate selected) noexcept
      : stage_(stage), selected_(selected) {}

  void setup(holoscan::OperatorSpec& spec) override {
    // WarmUp/Arm run after physical allocation. max_nominal_rate() therefore
    // supplies a finite ceiling that can be allocated before either callback.
    spec.output(output, "output")
        .lifecycle_nominal_rate(stage_)
        .max_nominal_rate(holoscan::Rate::hz(120U))
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = kScalarTensorType,
                               .rank = 0U},
            .bounds = holoscan::tensor_bounds(8U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kAllocate, &LateBindingTensorSource::allocate)
        .stage(stage_, &LateBindingTensorSource::bind)
        .stage(holoscan::LifecycleStage::kStart, &LateBindingTensorSource::observe_start);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::LifecycleStatus allocate(holoscan::LifecycleContext& context) noexcept {
    const auto rate = context.rate(output);
    if (!rate || rate->state != holoscan::PortRateViewState::kPendingBarrier ||
        rate->resolved_nominal_rate.has_value()) {
      LateBindingState::failed.store(true, std::memory_order_release);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    LateBindingState::allocate_pending.store(true, std::memory_order_release);
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus bind(holoscan::LifecycleContext& context) noexcept {
    // This changes only the active logical limit inside preallocated backing.
    // A value above the maximum is a typed, fail-closed candidate rejection.
    const auto bound = context.bind_nominal_rate(output, selected_);
    if (!bound) {
      LateBindingState::above_maximum_rejected.store(
          bound.error().code == holoscan::ErrorCode::kRateBindingAboveMaximum,
          std::memory_order_release);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    const auto rate = context.rate(output);
    if (!rate || rate->state != holoscan::PortRateViewState::kPendingBarrier ||
        rate->resolved_nominal_rate.has_value()) {
      LateBindingState::failed.store(true, std::memory_order_release);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    LateBindingState::binding_pending.store(true, std::memory_order_release);
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus observe_start(holoscan::LifecycleContext& context) noexcept {
    const auto rate = context.rate(output);
    if (!rate || rate->state != holoscan::PortRateViewState::kResolved ||
        rate->max_nominal_rate != holoscan::Rate::hz(120U)) {
      LateBindingState::failed.store(true, std::memory_order_release);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    const auto& nominal_rate = rate->resolved_nominal_rate;
    if (!nominal_rate.has_value() || !nominal_rate->equivalent_to(selected_) ||
        *nominal_rate != holoscan::Rate::hz(100U)) {
      LateBindingState::failed.store(true, std::memory_order_release);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    LateBindingState::start_resolved.store(true, std::memory_order_release);
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::Output<holoscan::Tensor> output;

 private:
  holoscan::LifecycleStage stage_;
  holoscan::Rate selected_;
};

class LateBindingTensorSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = kScalarTensorType,
                               .rank = 0U},
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 100ms}).input(input, holoscan::DrainAll{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
};

[[nodiscard]] holoscan::ExecutionPlan compile_late_binding(holoscan::LifecycleStage stage,
                                                           holoscan::Rate selected) {
  holoscan::Graph graph{stage == holoscan::LifecycleStage::kWarmUp ? "warmup-rate" : "arm-rate"};
  const auto source = graph.op<LateBindingTensorSource>("sensor", stage, selected);
  const auto sink = graph.op<LateBindingTensorSink>("sink");
  graph.add_flow(source->output, sink->input);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("late-rate-clock"));
  return holoscan::compile(graph);
}

[[nodiscard]] bool run_valid_late_binding(holoscan::LifecycleStage stage, std::string_view label) {
  LateBindingState::reset();
  // Deliberately select an unreduced representation. Binding canonicalizes it to 100/1, while
  // equivalent_to() compares the rational value rather than the original 200/2 representation.
  const holoscan::ExecutionPlan plan = compile_late_binding(stage, holoscan::Rate::hz(200U, 2U));
  const auto* edge = find_edge(plan, "sensor", "output", "sink", "input");
  const auto* recipe = edge == nullptr ? nullptr : find_recipe(plan, edge->edge);
  if (!plan.ok() || plan.tensor_allocations().size() != 1U ||
      plan.provider_allocation_plans().size() != 1U || edge == nullptr || recipe == nullptr ||
      recipe->geometry_mode != holoscan::AllocationGeometryMode::kCompileConcrete) {
    return fail(label, plan.ok() ? "finite-maximum preallocation recipe was absent" : plan.json());
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  auto snapshot = wait_for_published_snapshot(session);
  session.request_stop();
  session.wait();
  if (!snapshot || snapshot->published() == nullptr ||
      session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty() || LateBindingState::failed.load(std::memory_order_acquire) ||
      !LateBindingState::allocate_pending.load(std::memory_order_acquire) ||
      !LateBindingState::binding_pending.load(std::memory_order_acquire) ||
      !LateBindingState::start_resolved.load(std::memory_order_acquire)) {
    return fail(label, "late-stage binding did not publish within preallocated capacity");
  }

  const auto* queue = find_queue(*snapshot, *snapshot->published(), recipe->id);
  const auto providers = snapshot->provider_allocations(*snapshot->published());
  if (queue == nullptr || !queue->admitted_depth_ceiling ||
      queue->effective_logical_depth >= *queue->admitted_depth_ceiling || providers.size() != 1U ||
      providers.front().action != holoscan::AllocationAction::kFresh) {
    return fail(label, "active/admitted depth or provider evidence was incomplete");
  }
  return true;
}

[[nodiscard]] bool run_late_binding() {
  // Lesson 5: prove both valid late barriers and one deliberately invalid
  // selection. The invalid candidate must fail without publishing it.
  if (!run_valid_late_binding(holoscan::LifecycleStage::kWarmUp, "late-binding-warmup") ||
      !run_valid_late_binding(holoscan::LifecycleStage::kArm, "late-binding-arm")) {
    return false;
  }

  LateBindingState::reset();
  const holoscan::ExecutionPlan rejected =
      compile_late_binding(holoscan::LifecycleStage::kArm, holoscan::Rate::hz(144U));
  if (!rejected.ok()) {
    return fail("late-binding", rejected.json());
  }
  holoscan::RunSession session = holoscan::run_async(rejected);
  session.wait();
  const auto snapshot = session.capture_rate_generations();
  if (session.termination() != holoscan::RunTermination::kFailed ||
      !LateBindingState::above_maximum_rejected.load(std::memory_order_acquire) || !snapshot ||
      snapshot->last_rejected() == nullptr ||
      snapshot->last_rejected()->diagnostic.token != "PORT_RATE_BINDING_ABOVE_MAXIMUM") {
    return fail("late-binding", "144 Hz did not fail against the admitted 120 Hz maximum");
  }

  std::cout << "rate_aware_ports late-binding: PASS warmup=100/1 arm=100/1 maximum=120/1"
               " above_maximum=144/1-rejected geometry=compile-concrete\n";
  return true;
}

struct RestartState {
  static void reset() noexcept {
    discover_calls.store(0U, std::memory_order_release);
    allow_restart.store(false, std::memory_order_release);
    initial_observed.store(false, std::memory_order_release);
    restarted_observed.store(false, std::memory_order_release);
    failed.store(false, std::memory_order_release);
  }

  [[nodiscard]] static holoscan::Rate next_rate() noexcept {
    return discover_calls.fetch_add(1U, std::memory_order_acq_rel) == 0U ? holoscan::Rate::hz(60U)
                                                                         : holoscan::Rate::hz(30U);
  }

  static inline std::atomic<std::uint32_t> discover_calls{};
  static inline std::atomic<bool> allow_restart{};
  static inline std::atomic<bool> initial_observed{};
  static inline std::atomic<bool> restarted_observed{};
  static inline std::atomic<bool> failed{};
};

class RestartTensorSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    // Retained history contributes to the shared runtime-pool obligation. The
    // maximum admits backing for the largest allowed restart generation.
    spec.output(output, "output")
        .lifecycle_nominal_rate(holoscan::LifecycleStage::kDiscover)
        .max_nominal_rate(holoscan::Rate::hz(120U))
        .late_joiner_history_depth(4U)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = kScalarTensorType,
                               .rank = 0U},
            .bounds = holoscan::tensor_bounds(8U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
    spec.lifecycle().stage(holoscan::LifecycleStage::kDiscover, &RestartTensorSource::discover);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const auto rate = context.rate(output);
    if (!rate || rate->state != holoscan::PortRateViewState::kResolved) {
      RestartState::failed.store(true, std::memory_order_release);
      return holoscan::make_unexpected(rate ? holoscan::Error{holoscan::ErrorCode::kFailure}
                                            : rate.error());
    }
    if (phase_ == 0U) {
      if (rate->resolved_nominal_rate != holoscan::Rate::hz(60U)) {
        RestartState::failed.store(true, std::memory_order_release);
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
      }
      RestartState::initial_observed.store(true, std::memory_order_release);
      if (!RestartState::allow_restart.load(std::memory_order_acquire)) {
        return {};
      }
      // A self restart must be authorized by graph.lifecycle() and names the
      // forward barrier that will select the next candidate rate.
      auto receipt = context.request_restart(
          holoscan::LifecycleRestartOptions{.from_stage = holoscan::LifecycleStage::kDiscover});
      if (!receipt) {
        RestartState::failed.store(true, std::memory_order_release);
        return holoscan::make_unexpected(std::move(receipt.error()));
      }
      phase_ = 1U;
      return {};
    }
    if (phase_ == 1U) {
      if (rate->resolved_nominal_rate != holoscan::Rate::hz(30U)) {
        RestartState::failed.store(true, std::memory_order_release);
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
      }
      RestartState::restarted_observed.store(true, std::memory_order_release);
      phase_ = 2U;
    }
    return {};
  }

  holoscan::LifecycleStatus discover(holoscan::LifecycleContext& context) noexcept {
    // Startup selects 60 Hz; the admitted restart selects 30 Hz. The barrier
    // publishes either complete generation, never a partially updated view.
    const auto bound = context.bind_nominal_rate(output, RestartState::next_rate());
    if (!bound) {
      RestartState::failed.store(true, std::memory_order_release);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::Output<holoscan::Tensor> output;

 private:
  std::uint32_t phase_{};
};

class RestartTensorSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = kScalarTensorType,
                               .rank = 0U},
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 62500us});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
};

[[nodiscard]] const holoscan::ProviderAllocationGenerationRecord* find_provider(
    const holoscan::RateGenerationSnapshot& snapshot,
    const holoscan::RateGenerationEvidenceRecord& generation,
    holoscan::ProviderAllocationPlanRecordId recipe) {
  const auto records = snapshot.provider_allocations(generation);
  const auto record =
      std::ranges::find(records, recipe, &holoscan::ProviderAllocationGenerationRecord::recipe);
  return record == records.end() ? nullptr : &*record;
}

[[nodiscard]] const holoscan::PoolCapacityGenerationRecord* find_pool(
    const holoscan::RateGenerationSnapshot& snapshot,
    const holoscan::RateGenerationEvidenceRecord& generation,
    holoscan::ProviderAllocationPlanRecordId recipe) {
  const auto records = snapshot.pool_capacities(generation);
  const auto record =
      std::ranges::find(records, recipe, &holoscan::PoolCapacityGenerationRecord::recipe);
  return record == records.end() ? nullptr : &*record;
}

enum class RestartEvidenceStatus : std::uint8_t {
  kComplete,
  kMissingRecord,
  kUnexpectedValue,
};

struct RestartEvidenceView {
  const holoscan::ProviderAllocationGenerationRecord* initial_provider{};
  const holoscan::ProviderAllocationGenerationRecord* restarted_provider{};
  const holoscan::PoolCapacityGenerationRecord* initial_pool{};
  const holoscan::PoolCapacityGenerationRecord* restarted_pool{};
  const holoscan::RateGenerationEvidenceRecord* initial_generation{};
  const holoscan::RateGenerationEvidenceRecord* restarted_generation{};
};

// Keep validation pure so the missing-record path is deterministic and can be
// exercised without racing a RunSession. Pointer presence is proved before any
// field is read; failure reporting uses the same guarded view below.
[[nodiscard]] RestartEvidenceStatus classify_restart_evidence(
    const RestartEvidenceView& evidence) noexcept {
  if (evidence.initial_provider == nullptr || evidence.restarted_provider == nullptr ||
      evidence.initial_pool == nullptr || evidence.restarted_pool == nullptr ||
      evidence.initial_generation == nullptr || evidence.restarted_generation == nullptr) {
    return RestartEvidenceStatus::kMissingRecord;
  }

  const auto& initial_provider = *evidence.initial_provider;
  const auto& restarted_provider = *evidence.restarted_provider;
  const auto& initial_pool = *evidence.initial_pool;
  const auto& restarted_pool = *evidence.restarted_pool;
  const auto& initial_generation = *evidence.initial_generation;
  const auto& restarted_generation = *evidence.restarted_generation;
  if (initial_provider.action != holoscan::AllocationAction::kFresh ||
      restarted_provider.action != holoscan::AllocationAction::kReusedLarger ||
      !initial_provider.allocation_generation || !restarted_provider.allocation_generation ||
      initial_provider.allocation_generation != restarted_provider.allocation_generation ||
      !initial_provider.storage_binding_generation ||
      !restarted_provider.storage_binding_generation ||
      initial_provider.storage_binding_generation ==
          restarted_provider.storage_binding_generation ||
      initial_pool.required_pool_entries != 14U ||
      initial_pool.admitted_pool_entry_ceiling != 22U ||
      restarted_pool.required_pool_entries != 10U ||
      restarted_pool.admitted_pool_entry_ceiling != 22U ||
      initial_generation.run_generation != restarted_generation.run_generation ||
      initial_generation.rate_generation == restarted_generation.rate_generation) {
    return RestartEvidenceStatus::kUnexpectedValue;
  }
  return RestartEvidenceStatus::kComplete;
}

[[nodiscard]] std::string summarize_restart_evidence(const RestartEvidenceView& evidence) {
  std::ostringstream summary;
  summary << "initial_action=";
  if (evidence.initial_provider == nullptr) {
    summary << "missing";
  } else {
    summary << static_cast<std::uint32_t>(evidence.initial_provider->action);
  }
  summary << " restarted_action=";
  if (evidence.restarted_provider == nullptr) {
    summary << "missing";
  } else {
    summary << static_cast<std::uint32_t>(evidence.restarted_provider->action);
  }
  summary << " initial_entries=";
  if (evidence.initial_pool == nullptr) {
    summary << "missing";
  } else {
    summary << evidence.initial_pool->required_pool_entries;
  }
  summary << " admitted_entries=";
  if (evidence.initial_pool == nullptr) {
    summary << "missing";
  } else {
    summary << evidence.initial_pool->admitted_pool_entry_ceiling.value_or(0U);
  }
  summary << " restarted_entries=";
  if (evidence.restarted_pool == nullptr) {
    summary << "missing";
  } else {
    summary << evidence.restarted_pool->required_pool_entries;
  }
  summary << " initial_allocation=";
  if (evidence.initial_provider == nullptr || !evidence.initial_provider->allocation_generation) {
    summary << "missing";
  } else {
    summary << evidence.initial_provider->allocation_generation->value;
  }
  summary << " restarted_allocation=";
  if (evidence.restarted_provider == nullptr ||
      !evidence.restarted_provider->allocation_generation) {
    summary << "missing";
  } else {
    summary << evidence.restarted_provider->allocation_generation->value;
  }
  return summary.str();
}

[[nodiscard]] int run_restart_evidence_self_check(const std::string_view missing) {
  const holoscan::ProviderAllocationGenerationRecord provider;
  const holoscan::PoolCapacityGenerationRecord pool;
  const holoscan::RateGenerationEvidenceRecord generation;
  RestartEvidenceView deliberately_incomplete{
      .initial_provider = &provider,
      .restarted_provider = &provider,
      .initial_pool = &pool,
      .restarted_pool = &pool,
      .initial_generation = &generation,
      .restarted_generation = &generation,
  };
  std::string_view expected_summary;
  if (missing == "provider") {
    deliberately_incomplete.initial_provider = nullptr;
    expected_summary = "initial_action=missing";
  } else if (missing == "pool") {
    deliberately_incomplete.initial_pool = nullptr;
    expected_summary = "initial_entries=missing";
  } else {
    return 3;
  }

  const std::string summary = summarize_restart_evidence(deliberately_incomplete);
  if (classify_restart_evidence(deliberately_incomplete) != RestartEvidenceStatus::kMissingRecord ||
      summary.find(expected_summary) == std::string::npos) {
    return 3;
  }
  std::cerr << "rate_aware_ports restart-evidence-self-check: REJECTED missing=" << missing << ' '
            << summary << '\n';
  return 2;
}

[[nodiscard]] bool run_restart() {
  // Lesson 6: a compatible restart changes the rate and storage-binding
  // generations while retaining the run and physical-allocation generations.
  RestartState::reset();
  holoscan::Graph graph{"rate-restart-reuse"};
  const auto source = graph.op<RestartTensorSource>("camera");
  const auto sink = graph.op<RestartTensorSink>("recorder");
  graph.add_flow(source->output, sink->input);
  graph.lifecycle().allow_self_restart(source);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("restart-clock"));
  holoscan::CompileOptions options;
  options.lifecycle_transactions = holoscan::LifecycleTransactionOptions{
      .max_pending_requests = 2U,
      .max_pending_transactions = 2U,
      .max_receipts = 4U,
      .max_fenced_operators = 4U,
      .max_candidate_generations = 4U,
      .max_events = 32U,
      .max_audit_records = 8U,
      .max_quiesce_records = 8U,
      .max_rollback_records = 16U,
  };
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok() || plan.provider_allocation_plans().size() != 1U) {
    return fail("restart", plan.ok() ? "provider plan was absent" : plan.json());
  }
  const auto provider_recipe = plan.provider_allocation_plans().front().id;

  holoscan::RunSession session = holoscan::run_async(plan);
  if (!wait_for(RestartState::initial_observed)) {
    session.request_stop();
    session.wait();
    return fail("restart", "initial 60 Hz generation was not observed");
  }
  auto initial = session.capture_rate_generations();
  RestartState::allow_restart.store(true, std::memory_order_release);
  if (!wait_for(RestartState::restarted_observed)) {
    session.request_stop();
    session.wait();
    return fail("restart", "restarted 30 Hz generation was not observed");
  }
  auto restarted = session.capture_rate_generations();
  session.request_stop();
  session.wait();
  if (!initial || !restarted || initial->published() == nullptr ||
      restarted->published() == nullptr || !session.diagnostics().empty() ||
      RestartState::failed.load(std::memory_order_acquire)) {
    return fail("restart", "rate-generation evidence was unavailable");
  }

  const auto* initial_provider = find_provider(*initial, *initial->published(), provider_recipe);
  const auto* restarted_provider =
      find_provider(*restarted, *restarted->published(), provider_recipe);
  const auto* initial_pool = find_pool(*initial, *initial->published(), provider_recipe);
  const auto* restarted_pool = find_pool(*restarted, *restarted->published(), provider_recipe);
  const RestartEvidenceView evidence{
      .initial_provider = initial_provider,
      .restarted_provider = restarted_provider,
      .initial_pool = initial_pool,
      .restarted_pool = restarted_pool,
      .initial_generation = initial->published(),
      .restarted_generation = restarted->published(),
  };
  if (session.termination() != holoscan::RunTermination::kStopped) {
    return fail("restart", "session did not report stopped termination");
  }
  if (classify_restart_evidence(evidence) != RestartEvidenceStatus::kComplete) {
    std::cerr << "restart evidence: " << summarize_restart_evidence(evidence) << '\n';
    return fail("restart", "compatible reuse identities or 14/22-to-10 geometry did not match");
  }

  std::cout << "rate_aware_ports restart: PASS initial_rate=60/1 restart_rate=30/1"
               " entries=14->10 admitted=22 allocation=reused-larger binding=fresh\n";
  return true;
}

[[nodiscard]] bool run_mode(std::string_view mode) {
  if (mode == "declarations") {
    return run_declarations();
  }
  if (mode == "fusion") {
    return run_fusion();
  }
  if (mode == "fallback") {
    return run_fallback();
  }
  if (mode == "discover") {
    return run_discover();
  }
  if (mode == "late-binding") {
    return run_late_binding();
  }
  if (mode == "restart") {
    return run_restart();
  }
  return false;
}

[[nodiscard]] bool is_known_mode(std::string_view mode) noexcept {
  return mode == "declarations" || mode == "fusion" || mode == "fallback" || mode == "discover" ||
         mode == "late-binding" || mode == "restart";
}

}  // namespace

int main(int argc, char** argv) {
  std::string_view mode{"all"};
  if (argc == 3 && std::string_view(argv[1]) == "--self-check-restart-evidence") {
    return run_restart_evidence_self_check(argv[2]);
  }
  if (argc == 3 && std::string_view(argv[1]) == "--mode") {
    mode = argv[2];
  } else if (argc != 1) {
    std::cerr << "usage: " << argv[0]
              << " [--mode declarations|fusion|fallback|discover|late-binding|restart]\n"
                 "       "
              << argv[0] << " --self-check-restart-evidence provider|pool\n";
    return 1;
  }

  if (mode == "all") {
    for (const std::string_view scenario :
         {"declarations", "fusion", "fallback", "discover", "late-binding", "restart"}) {
      if (!run_mode(scenario)) {
        return 2;
      }
    }
  } else if (!is_known_mode(mode)) {
    std::cerr << "unknown mode: " << mode << '\n';
    return 1;
  } else if (!run_mode(mode)) {
    return 2;
  }

  std::cout << "rate_aware_ports example complete\n";
  return 0;
}
