/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle_control.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/parameter_compute_control.hpp>
#include <holoscan/core/parameter_control.hpp>
#include <holoscan/core/parameter_snapshot.hpp>
#include <holoscan/core/parameter_update.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/time/clock.hpp>

namespace {

class InstalledRuntimeClock final : public holoscan::Clock {
 public:
  [[nodiscard]] double time() const override { return holoscan::timestamp_to_time(timestamp_); }
  [[nodiscard]] std::int64_t timestamp() const override { return timestamp_; }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> sleep_for(
      std::int64_t duration_ns) override {
    timestamp_ += duration_ns;
    return {};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> sleep_until(
      std::int64_t target_time_ns) override {
    timestamp_ = target_time_ns;
    return {};
  }

 private:
  std::int64_t timestamp_{};
};

class InstalledRuntimeOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnSpin{.event = spin_});
    return result;
  }

  void start() override { callbacks.fetch_add(1U, std::memory_order_release); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    callbacks.fetch_add(1U, std::memory_order_release);
    return {};
  }

  void stop() override { callbacks.fetch_add(1U, std::memory_order_release); }

  static inline std::atomic<std::uint32_t> callbacks{};

 private:
  holoscan::SpinSource spin_;
};

class InstalledInterprocessSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
    spec.spin_source(spin_, "activation").capacity(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnSpin{.event = spin_});
    return result;
  }

  void start() override { callbacks.fetch_add(1U, std::memory_order_release); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    callbacks.fetch_add(1U, std::memory_order_release);
    return {};
  }

  void stop() override { callbacks.fetch_add(1U, std::memory_order_release); }

  static inline std::atomic<std::uint32_t> callbacks{};
  holoscan::Output<std::uint32_t> output;

 private:
  holoscan::SpinSource spin_;
};

class InstalledInterprocessSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(2U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAnyInput{});
    return result;
  }

  void start() override { callbacks.fetch_add(1U, std::memory_order_release); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    callbacks.fetch_add(1U, std::memory_order_release);
    return {};
  }

  void stop() override { callbacks.fetch_add(1U, std::memory_order_release); }

  static inline std::atomic<std::uint32_t> callbacks{};
  holoscan::Input<std::uint32_t> input;
};

template <typename T>
concept HasRunMember = requires(T& value) { value.run(); };

template <typename T>
concept HasRunAsyncMember = requires(T& value) { value.run_async(); };

template <typename T>
concept HasRunOnceMember = requires(T& value) { value.run_once(); };

template <typename T>
concept HasBindProcessRoleMember = requires { &T::bind_process_role; };

template <typename T>
concept HasBindSupportThreadAffinityMember = requires { &T::bind_support_thread_affinity; };

template <typename T>
concept HasProcessRolesMember = requires(const T& value) { value.process_roles(); };

template <typename T>
concept HasProcessSelectorsMember = requires(const T& value) { value.process_selectors(); };

template <typename T>
concept HasVertexProcessPlacementsMember =
    requires(const T& value) { value.vertex_process_placements(); };

static_assert(!HasRunMember<holoscan::Graph>);
static_assert(!HasRunAsyncMember<holoscan::Graph>);
static_assert(!HasRunMember<InstalledRuntimeOperator>);
static_assert(!HasRunOnceMember<InstalledRuntimeOperator>);
// REQ: COMPILE-061
static_assert(!HasBindProcessRoleMember<holoscan::DeploymentBinding>);
// REQ: COMPILE-096
// REQ: COMPILE-099
// REQ: COMPILE-100
static_assert(HasBindSupportThreadAffinityMember<holoscan::DeploymentBinding>);
static_assert(!HasProcessRolesMember<holoscan::ExecutionPlan>);
static_assert(!HasProcessSelectorsMember<holoscan::ExecutionPlan>);
static_assert(!HasVertexProcessPlacementsMember<holoscan::ExecutionPlan>);
static_assert(sizeof(holoscan::RunSession) == sizeof(void*));
static_assert(std::move_constructible<holoscan::RunSession>);
static_assert(std::is_nothrow_move_constructible_v<holoscan::RunSession>);
static_assert(std::is_nothrow_destructible_v<holoscan::RunSession>);
static_assert(!std::copyable<holoscan::RunSession>);
// REQ: RUNTIME-047
// REQ: RUNTIME-049
static_assert(std::move_constructible<holoscan::ManagedShutdownSignals>);
static_assert(std::is_nothrow_move_constructible_v<holoscan::ManagedShutdownSignals>);
static_assert(std::is_nothrow_destructible_v<holoscan::ManagedShutdownSignals>);
static_assert(!std::copyable<holoscan::ManagedShutdownSignals>);
static_assert(!std::is_move_assignable_v<holoscan::ManagedShutdownSignals>);
static_assert(static_cast<std::uint8_t>(holoscan::RunSignalPolicy::kManaged) == 0U);
static_assert(static_cast<std::uint8_t>(holoscan::RunSignalPolicy::kCallerManaged) == 1U);

[[nodiscard]] bool has_diagnostic(const holoscan::RunSession& session, std::string_view token) {
  return std::any_of(session.diagnostics().begin(),
                     session.diagnostics().end(),
                     [token](const auto& diagnostic) { return diagnostic.token == token; });
}

}  // namespace

// REQ: RUNTIME-001
// REQ: RUNTIME-002
// REQ: RUNTIME-003
// REQ: RUNTIME-004
// REQ: RUNTIME-005
// REQ: RUNTIME-008
// REQ: RUNTIME-010
// REQ: RUNTIME-036
// REQ: CHAN-077
int main() {
  const auto run_sync = static_cast<int (*)(const holoscan::ExecutionPlan&)>(&holoscan::run);
  const auto run_sync_policy =
      static_cast<int (*)(const holoscan::ExecutionPlan&, holoscan::RunSignalPolicy)>(
          &holoscan::run);
  const auto run_async =
      static_cast<holoscan::RunSession (*)(const holoscan::ExecutionPlan&)>(&holoscan::run_async);
  using ManagedRunAsync =
      holoscan::RunSession (*)(const holoscan::ExecutionPlan&, holoscan::ManagedShutdownSignals&);
  const auto run_async_managed = static_cast<ManagedRunAsync>(&holoscan::run_async);
  const auto session_wait = &holoscan::RunSession::wait;
  const auto session_stop = &holoscan::RunSession::request_stop;
  const auto session_running = &holoscan::RunSession::is_running;
  const auto session_termination = &holoscan::RunSession::termination;
  const auto session_diagnostics = &holoscan::RunSession::diagnostics;
  const auto session_operator_ref = &holoscan::RunSession::operator_ref;
  const auto session_lifecycle_control = &holoscan::RunSession::lifecycle_control;
  const auto session_parameter_control = &holoscan::RunSession::parameter_control;
  const auto session_close_parameter_ingress =
      &holoscan::RunSession::close_parameter_update_ingress;
  const auto session_capture_params = &holoscan::RunSession::capture_params;
  const auto handle_catalog = &holoscan::ParamControlHandle::catalog;
  const auto handle_read = &holoscan::ParamControlHandle::read;
  const auto handle_restrict = &holoscan::ParamControlHandle::restrict;
  const auto handle_submit_one = static_cast<holoscan::ParamBatchUpdateResult (  // NOLINT
      holoscan::ParamControlHandle::*)(const holoscan::ParamPatch&, holoscan::ParamUpdateOptions)>(
      &holoscan::ParamControlHandle::submit);
  const auto handle_submit_many = static_cast<holoscan::ParamBatchUpdateResult (  // NOLINT
      holoscan::ParamControlHandle::*)(std::span<const holoscan::ParamPatch>,
                                       holoscan::ParamUpdateOptions)>(
      &holoscan::ParamControlHandle::submit);
  const auto handle_await = &holoscan::ParamControlHandle::await;
  const auto handle_capture = &holoscan::ParamControlHandle::capture;
  const auto handle_open_cursor = &holoscan::ParamControlHandle::open_event_cursor;
  const auto cursor_drain = &holoscan::ParamEventCursor::drain;
  const auto snapshot_slice_scope =
      static_cast<holoscan::expected<holoscan::ParamSnapshot, holoscan::Error> (
          holoscan::ParamSnapshot::*)(const holoscan::ParamSnapshotScope&) const>(
          &holoscan::ParamSnapshot::slice);
  if (run_sync == nullptr || run_sync_policy == nullptr || run_async == nullptr ||
      run_async_managed == nullptr || session_wait == nullptr || session_stop == nullptr ||
      session_running == nullptr || session_termination == nullptr ||
      session_diagnostics == nullptr || session_operator_ref == nullptr ||
      session_lifecycle_control == nullptr || session_parameter_control == nullptr ||
      session_close_parameter_ingress == nullptr || session_capture_params == nullptr ||
      handle_catalog == nullptr || handle_read == nullptr || handle_restrict == nullptr ||
      handle_submit_one == nullptr || handle_submit_many == nullptr || handle_await == nullptr ||
      handle_capture == nullptr || handle_open_cursor == nullptr || cursor_drain == nullptr ||
      snapshot_slice_scope == nullptr) {
    return 1;
  }

  auto managed_signals = holoscan::ManagedShutdownSignals::create();
#if defined(__linux__)
  if (!managed_signals) {
    return 9;
  }
  auto moved_managed_signals = std::move(*managed_signals);
  const auto managed_signals_close = moved_managed_signals.close();
  if (!managed_signals_close) {
    return 10;
  }
#else
  if (managed_signals || managed_signals.error().code != holoscan::ErrorCode::kNotSupported) {
    return 9;
  }
#endif

  // Exercise the installed support-thread authoring ABI without asking this package smoke test to
  // assume that any particular CPU is available on its compile host.
  holoscan::DeploymentBinding support_thread_binding;
  support_thread_binding.bind_support_thread_affinity(holoscan::SupportThreadRole::kController,
                                                      holoscan::CPUAffinity{{0U}});
  if (support_thread_binding.support_thread_affinities().size() != 1U ||
      support_thread_binding.support_thread_affinities().front().role !=
          holoscan::SupportThreadRole::kController) {
    return 11;
  }

  holoscan::Graph graph{"installed-runtime-consumer"};
  const auto runtime_operator = graph.op<InstalledRuntimeOperator>("operator");
  graph.partition("installed-runtime-worker").add(runtime_operator);
  const holoscan::ClockKey clock = graph.add_clock<InstalledRuntimeClock>("runtime-clock");
  graph.set_default_clock(clock);
  holoscan::CompileOptions options;
  options.deployment.bind_thread_placement(
      "installed-runtime-worker",
      holoscan::ThreadPlacement{.wake_mode = holoscan::WakeMode::kBusyPoll});
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, options);
  if (!plan.ok()) {
    return 2;
  }
  if (!plan.support_thread_affinities().empty()) {
    return 12;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  session.wait();
  if (session.is_running() || session.termination() != holoscan::RunTermination::kFailed ||
      !has_diagnostic(session, "RUNTIME_READINESS_BINDING_UNAVAILABLE") ||
      InstalledRuntimeOperator::callbacks.load(std::memory_order_acquire) != 0U) {
    return 3;
  }

  if (plan.lifecycle_checkpoints().empty()) {
    return 4;
  }
  const auto terminal_ref =
      session.operator_ref(plan.lifecycle_checkpoints().front().operator_instance);
  if (terminal_ref || terminal_ref.error().code != holoscan::ErrorCode::kStaleOperatorRef) {
    return 5;
  }
  const auto terminal_control = session.lifecycle_control({});
  if (terminal_control ||
      terminal_control.error().code != holoscan::ErrorCode::kLifecycleTransactionsDisabled) {
    return 6;
  }

  // Ordinary installed consumers compile every local Operator into the sole process group.
  // The public surface exposes neither process-placement authoring nor plan inspection.
  holoscan::Graph multi_operator_graph{"installed-multi-operator-package-consumer"};
  auto source = multi_operator_graph.op<InstalledInterprocessSource>("source");
  auto sink = multi_operator_graph.op<InstalledInterprocessSink>("sink");
  multi_operator_graph.partition("installed-multi-operator-worker").add(source).add(sink);
  multi_operator_graph.add_flow(
      source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 2U});
  const holoscan::ClockKey multi_operator_clock =
      multi_operator_graph.add_clock<InstalledRuntimeClock>("runtime-clock");
  multi_operator_graph.set_default_clock(multi_operator_clock);
  holoscan::CompileOptions multi_operator_options;
  multi_operator_options.deployment.bind_thread_placement(
      "installed-multi-operator-worker",
      holoscan::ThreadPlacement{.wake_mode = holoscan::WakeMode::kBusyPoll});
  const holoscan::ExecutionPlan multi_operator_plan =
      holoscan::compile(multi_operator_graph, std::move(multi_operator_options));
  if (!multi_operator_plan.ok()) {
    return 7;
  }
  if (multi_operator_plan.payload_edges().size() != 1U ||
      multi_operator_plan.payload_edges().front().locality !=
          holoscan::ChannelLocalityKind::kProcessLocal) {
    return 8;
  }
  return 0;
}
