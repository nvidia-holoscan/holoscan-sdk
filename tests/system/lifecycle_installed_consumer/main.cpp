/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/lifecycle_control.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/rate_generation.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

class InstalledLifecycleOperator final : public holoscan::Operator<> {
 public:
  explicit InstalledLifecycleOperator(bool producer) : producer_(producer) {}

  void setup(holoscan::OperatorSpec& spec) override {
    auto& lifecycle = spec.lifecycle();
    lifecycle.stage(holoscan::LifecycleStage::kConfigure, &InstalledLifecycleOperator::configure)
        .optional(holoscan::LifecycleStage::kStart, &InstalledLifecycleOperator::on_start);
    if (producer_) {
      lifecycle.produces(holoscan::LifecycleStage::kConfigure,
                         holoscan::LifecycleDependencyKey{"installed.configured"});
    } else {
      lifecycle.requires_dependency(holoscan::LifecycleStage::kConfigure,
                                    holoscan::LifecycleDependencyKey{"installed.configured"});
    }
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }

 private:
  holoscan::LifecycleStatus configure(holoscan::LifecycleContext&) noexcept {
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext&) noexcept {
    return holoscan::LifecycleStatus::kOk;
  }

  bool producer_{};
};

struct InstalledRateState {
  static void reset() noexcept {
    failure.store(0U, std::memory_order_release);
    published_generation.store(0U, std::memory_order_release);
    sink_value.store(0U, std::memory_order_release);
  }

  static void fail(std::uint32_t code) noexcept {
    std::uint32_t expected{};
    static_cast<void>(failure.compare_exchange_strong(expected, code, std::memory_order_acq_rel));
  }

  static bool record_published(const holoscan::PortRateView& view) noexcept {
    if (view.run_generation == 0U || view.rate_generation == 0U) {
      return false;
    }
    std::uint64_t expected{};
    if (published_generation.compare_exchange_strong(
            expected, view.rate_generation, std::memory_order_acq_rel)) {
      return true;
    }
    return expected == view.rate_generation;
  }

  static inline std::atomic<std::uint32_t> failure{};
  static inline std::atomic<std::uint64_t> published_generation{};
  static inline std::atomic<std::uint32_t> sink_value{};
};

class InstalledRateSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output")
        .lifecycle_nominal_rate(holoscan::LifecycleStage::kDiscover)
        .max_nominal_rate(holoscan::Rate::hz(120U));
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kDiscover, &InstalledRateSource::discover)
        .stage(holoscan::LifecycleStage::kAllocate, &InstalledRateSource::allocate);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(holoscan::ComputeCalls{.limit = 1U});
    return result;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto view = context.rate(output);
    if (!view || view->state != holoscan::PortRateViewState::kResolved ||
        view->origin != holoscan::ResolvedRateOrigin::kLifecycleBinding ||
        view->resolved_nominal_rate != holoscan::Rate::hz(60U) ||
        view->max_nominal_rate != holoscan::Rate::hz(120U) ||
        !InstalledRateState::record_published(*view)) {
      InstalledRateState::fail(5U);
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    return output.emit(kValue);
  }

  holoscan::LifecycleStatus discover(holoscan::LifecycleContext& context) noexcept {
    const auto bound = context.bind_nominal_rate(output, holoscan::Rate::hz(120U, 2U));
    const auto same_stage = context.rate(output);
    if (!bound || !same_stage ||
        same_stage->state != holoscan::PortRateViewState::kPendingBarrier ||
        same_stage->resolved_nominal_rate.has_value()) {
      InstalledRateState::fail(6U);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus allocate(holoscan::LifecycleContext& context) noexcept {
    const auto view = context.rate(output);
    if (!view || view->state != holoscan::PortRateViewState::kResolved ||
        view->origin != holoscan::ResolvedRateOrigin::kLifecycleBinding ||
        view->resolved_nominal_rate != holoscan::Rate::hz(60U) ||
        view->max_nominal_rate != holoscan::Rate::hz(120U) ||
        !InstalledRateState::record_published(*view)) {
      InstalledRateState::fail(7U);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::Output<std::uint32_t> output;

  static constexpr std::uint32_t kValue = 0x52415445U;
};

class InstalledRateRelay final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").expected_rate(holoscan::Rate::hz(60U));
    spec.output(output, "output").follows(input, holoscan::RateRatio{1U, 4U});
    spec.lifecycle().stage(holoscan::LifecycleStage::kAllocate, &InstalledRateRelay::allocate);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const auto input_view = context.rate(input);
    const auto output_view = context.rate(output);
    const auto sample = input.receive();
    if (!valid_views(input_view, output_view) || !sample) {
      InstalledRateState::fail(8U);
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    return output.emit(sample->data);
  }

  holoscan::LifecycleStatus allocate(holoscan::LifecycleContext& context) noexcept {
    if (!valid_views(context.rate(input), context.rate(output))) {
      InstalledRateState::fail(9U);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::Input<std::uint32_t> input;
  holoscan::Output<std::uint32_t> output;

 private:
  static bool valid_views(
      const holoscan::expected<holoscan::PortRateView, holoscan::Error>& input_view,
      const holoscan::expected<holoscan::PortRateView, holoscan::Error>& output_view) noexcept {
    return input_view && output_view &&
           input_view->state == holoscan::PortRateViewState::kResolved &&
           input_view->origin == holoscan::ResolvedRateOrigin::kConnectedOutput &&
           input_view->resolved_nominal_rate == holoscan::Rate::hz(60U) &&
           input_view->max_nominal_rate == holoscan::Rate::hz(120U) &&
           output_view->state == holoscan::PortRateViewState::kResolved &&
           output_view->origin == holoscan::ResolvedRateOrigin::kFollowsExpression &&
           output_view->resolved_nominal_rate == holoscan::Rate::hz(15U) &&
           output_view->max_nominal_rate == holoscan::Rate::hz(30U) &&
           InstalledRateState::record_published(*input_view) &&
           InstalledRateState::record_published(*output_view);
  }
};

class InstalledRateSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const auto view = context.rate(input);
    const auto sample = input.receive();
    if (!view || !sample || view->state != holoscan::PortRateViewState::kResolved ||
        view->origin != holoscan::ResolvedRateOrigin::kConnectedOutput ||
        view->resolved_nominal_rate != holoscan::Rate::hz(15U) ||
        view->max_nominal_rate != holoscan::Rate::hz(30U) ||
        !InstalledRateState::record_published(*view)) {
      InstalledRateState::fail(10U);
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    InstalledRateState::sink_value.store(sample->data, std::memory_order_release);
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

static_assert(std::is_standard_layout_v<holoscan::LifecycleRestartOptions>);
static_assert(std::is_trivially_copyable_v<holoscan::LifecycleRestartReceipt>);
static_assert(sizeof(holoscan::LifecycleContext) == sizeof(void*));
static_assert(sizeof(holoscan::LifecycleControlHandle) == sizeof(void*));
static_assert(std::is_copy_constructible_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_nothrow_copy_constructible_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_copy_assignable_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_nothrow_copy_assignable_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_move_constructible_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_nothrow_move_constructible_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_move_assignable_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_nothrow_move_assignable_v<holoscan::LifecycleControlHandle>);
static_assert(std::is_nothrow_destructible_v<holoscan::LifecycleControlHandle>);

[[nodiscard]] bool exercise_closed_lifecycle_control() {
  const holoscan::LifecycleControlHandle original;
  holoscan::LifecycleControlHandle copied{original};
  holoscan::LifecycleControlHandle moved{std::move(copied)};
  copied = original;
  moved = std::move(copied);

  const auto restricted = moved.restrict({});
  const auto requested = moved.request_restart({});
  const auto inspected = moved.inspect({});
  const auto waited = moved.await({}, std::chrono::nanoseconds{0});
  return !original && !copied && !moved && !restricted && !requested && !inspected && !waited &&
         restricted.error().code == holoscan::ErrorCode::kClosed &&
         requested.error().code == holoscan::ErrorCode::kClosed &&
         inspected.error().code == holoscan::ErrorCode::kClosed &&
         waited.error().code == holoscan::ErrorCode::kClosed;
}

}  // namespace

// REQ: LIFECYCLE-002
// REQ: LIFECYCLE-003
// REQ: LIFECYCLE-005
// REQ: LIFECYCLE-007
// REQ: LIFECYCLE-012
// REQ: RUNTIME-008
int main() {
  const auto cancellation = &holoscan::LifecycleContext::cancellation;
  const auto params_generation = &holoscan::LifecycleContext::params_generation;
  const auto transition_phase = &holoscan::LifecycleContext::transition_phase;
  const auto notification_sender = &holoscan::LifecycleContext::notification_sender;
  const auto restrict_control = &holoscan::LifecycleControlHandle::restrict;
  const auto request_restart = &holoscan::LifecycleControlHandle::request_restart;
  const auto inspect_restart = &holoscan::LifecycleControlHandle::inspect;
  const auto await_restart = &holoscan::LifecycleControlHandle::await;
  if (cancellation == nullptr || params_generation == nullptr || transition_phase == nullptr ||
      notification_sender == nullptr || restrict_control == nullptr || request_restart == nullptr ||
      inspect_restart == nullptr || await_restart == nullptr) {
    return 1;
  }

  if (!exercise_closed_lifecycle_control()) {
    return 2;
  }

  const holoscan::LifecycleRestartOptions defaults;
  if (defaults.from_stage != holoscan::LifecycleStage::kConfigure) {
    return 3;
  }

  holoscan::Graph graph{"installed-lifecycle"};
  const auto producer = graph.op<InstalledLifecycleOperator>("producer", true);
  const auto consumer = graph.op<InstalledLifecycleOperator>("consumer", false);
  graph.lifecycle()
      .before(producer, consumer, holoscan::LifecycleStage::kConfigure)
      .allow_self_restart(producer)
      .allow_external_restart(consumer)
      .allow_restart_request(producer, consumer);

  const holoscan::GraphView view = graph.view();
  if (view.lifecycle_dependencies().size() != 1U ||
      view.lifecycle_restart_permissions().size() != 3U) {
    return 4;
  }

  InstalledRateState::reset();
  holoscan::Graph rate_graph{"installed-rate-aware-ports"};
  const auto rate_source = rate_graph.op<InstalledRateSource>("source");
  const auto rate_relay = rate_graph.op<InstalledRateRelay>("relay");
  const auto rate_sink = rate_graph.op<InstalledRateSink>("sink");
  rate_graph.partition("source-worker").add(rate_source);
  rate_graph.partition("relay-worker").add(rate_relay);
  rate_graph.partition("sink-worker").add(rate_sink);
  rate_graph.add_flow(rate_source->output, rate_relay->input);
  rate_graph.add_flow(rate_relay->output, rate_sink->input);
  rate_graph.set_default_clock(rate_graph.add_clock<holoscan::RealtimeClock>("rate-clock"));

  const holoscan::ExecutionPlan rate_plan = holoscan::compile(rate_graph);
  if (!rate_plan.ok()) {
    return 11;
  }
  holoscan::RunSession rate_session = holoscan::run_async(rate_plan);
  rate_session.wait();
  if (rate_session.termination() != holoscan::RunTermination::kCompleted ||
      !rate_session.diagnostics().empty() ||
      InstalledRateState::failure.load(std::memory_order_acquire) != 0U ||
      InstalledRateState::published_generation.load(std::memory_order_acquire) == 0U ||
      InstalledRateState::sink_value.load(std::memory_order_acquire) !=
          InstalledRateSource::kValue) {
    return 12;
  }
  return 0;
}
