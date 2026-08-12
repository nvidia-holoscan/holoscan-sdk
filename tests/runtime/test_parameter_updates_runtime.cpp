/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#include <holoscan/core/parameter_control.hpp>

#include "../common/allocation_counter.hpp"
#include "runtime/parameter_control_state.hpp"
#include "runtime/port_binding.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"

#include "parameter_update_test_support.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;
using namespace test::parameter_update;

[[nodiscard]] constexpr ParamAudit invalid_param_audit_for_test() noexcept {
  // This test-only helper deliberately forges a value outside the closed public enum so the
  // runtime validation path can be exercised without scattering invalid casts through tests.
  using Underlying = std::underlying_type_t<ParamAudit>;
  return std::bit_cast<ParamAudit>(std::numeric_limits<Underlying>::max());
}

// Releases a blocked probe callback on destruction so a failed test cannot strand the Runtime
// worker. Release remains idempotent and occurs before the guard marks itself inactive.
template <void (*ReleaseFunction)()>
class ScopedCallbackRelease final {
 public:
  ScopedCallbackRelease() = default;
  ScopedCallbackRelease(const ScopedCallbackRelease&) = delete;
  ScopedCallbackRelease& operator=(const ScopedCallbackRelease&) = delete;
  ScopedCallbackRelease(ScopedCallbackRelease&&) = delete;
  ScopedCallbackRelease& operator=(ScopedCallbackRelease&&) = delete;

  ~ScopedCallbackRelease() noexcept { release(); }

  void release() {
    if (!released_) {
      ReleaseFunction();
      released_ = true;
    }
  }

 private:
  bool released_{};
};

using FirstCallbackRelease =
    ScopedCallbackRelease<&test::RuntimeParamProbeState::release_first_callback>;
using SecondCallbackRelease =
    ScopedCallbackRelease<&test::RuntimeParamProbeState::release_second_callback>;
using StartCallbackRelease =
    ScopedCallbackRelease<&test::RuntimeParamProbeState::release_start_callback>;
using StopCallbackRelease =
    ScopedCallbackRelease<&test::RuntimeParamProbeState::release_stop_callback>;

// Synthetic port binding that emits exactly one kSpin event then sets the probe
// runtime clock 50 ns ahead of the start coordinate. Used by
// DelayedDeadlinePortBindingFactory for PinnedDeadlineGenerationControlsTheInvocationGate.
class DelayedDeadlinePortBinding final : public detail::PortBinding {
 public:
  DelayedDeadlinePortBinding(const detail::RuntimePortBindingRequest& request,
                             TimePoint coordinate) noexcept
      : source_(request.source),
        event_{.kind = request.kind,
               .source = request.source,
               .port = request.port,
               .source_decision_time = coordinate,
               .reference_time = coordinate,
               .source_generation = request.source_generation,
               .source_sequence = request.initial_source_sequence + 1U},
        capacity_(request.capacity) {}

  [[nodiscard]] SourceId source() const noexcept override { return source_; }

  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<detail::RuntimeEvent> destination) noexcept override {
    if (!pending_ || destination.empty()) {
      return 0U;
    }
    destination.front() = event_;
    pending_ = false;
    outstanding_ = true;
    return 1U;
  }

  [[nodiscard]] expected<void, Error> retire(std::uint64_t source_sequence) noexcept override {
    if (!outstanding_ || source_sequence != event_.source_sequence) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    outstanding_ = false;
    closed_ = true;
    return {};
  }

  void cancel() noexcept override {
    pending_ = false;
    outstanding_ = false;
    closed_ = true;
  }

  [[nodiscard]] std::size_t capacity() const noexcept override { return capacity_; }
  [[nodiscard]] bool closed() const noexcept override { return closed_; }

 private:
  SourceId source_{};
  detail::RuntimeEvent event_{};
  std::size_t capacity_{};
  bool pending_{true};
  bool outstanding_{};
  bool closed_{};
};

// Factory that creates a DelayedDeadlinePortBinding and atomically advances the probe
// clock. Injected via RuntimeTestPlanAccess::with_port_binding_factory so that
// PinnedDeadlineGenerationControlsTheInvocationGate drives the deadline scheduling
// path without a real timer source.
class DelayedDeadlinePortBindingFactory final : public detail::RuntimePortBindingFactory {
 public:
  [[nodiscard]] expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest& request,
      TimePoint start_coordinate) noexcept override {
    if (request.kind != detail::RuntimeEventKind::kSpin || !request.source || request.port ||
        request.capacity == 0U || request.source_generation == 0U ||
        request.initial_source_sequence == std::numeric_limits<std::uint64_t>::max() ||
        !start_coordinate.valid() || start_coordinate.clock_id != request.reference_clock ||
        request.source_clock != request.reference_clock) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      auto result = std::unique_ptr<detail::PortBinding>{
          std::make_unique<DelayedDeadlinePortBinding>(request, start_coordinate)};
      test::RuntimeProbeState::runtime_clock_ns.store(start_coordinate.timestamp_ns + 50,
                                                      std::memory_order_release);
      return result;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
#endif
  }
};

// REQ: PARAM-017
TEST(ParameterRuntimeIntegrationTest, GenerationZeroOnlyLegacyLifecycleUsesOperatorOwnedAggregate) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  // Graph: [parameterized]  (single operator, no edges, generation-zero compile)
  //   The operator-owned aggregate is used directly by lifecycle hooks; the coordinator
  //   allocates no update, event, or generation-slot storage.
  Graph graph{"generation-zero-only-legacy-lifecycle-test"};
  const auto parameterized = graph.op<test::RuntimeParamProbeOperator>("parameterized");
  graph.partition("runtime-spin-worker").add(parameterized);
  const ClockKey clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-spin-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  const ExecutionPlan compiled = compile(graph, options);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_TRUE(compiled.parameter_runtime_capacity());
  const auto capacity = checked_value(compiled.parameter_runtime_capacity());
  EXPECT_FALSE(capacity.runtime_updates_enabled);
  EXPECT_FALSE(capacity.lifecycle_updates_enabled);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(compiled);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto observations = test::RuntimeParamProbeState::snapshot();
  ASSERT_EQ(observations.size(), 1U);
  EXPECT_EQ(observations.front().before_value, 7);
  EXPECT_EQ(observations.front().before_generation, 0U);
  EXPECT_EQ(observations.front().after_value, 7);
  EXPECT_EQ(observations.front().after_generation, 0U);
  EXPECT_EQ(test::RuntimeParamProbeState::stop_observed_value.load(std::memory_order_acquire), 7);
  EXPECT_TRUE(test::RuntimeParamProbeState::stop_observed_params.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeProbeState::snapshot(), (std::vector<std::int32_t>{901, 902, 903}));
}

// REQ: PARAM-017
// REQ: PARAM-018
// REQ: PARAM-019
TEST(ParameterRuntimeIntegrationTest, PinnedDeadlineGenerationControlsTheInvocationGate) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  test::RuntimeDeadlineParamProbeState::reset();
  test::RuntimeParamProbeState::hold_start_callback();
  StartCallbackRelease release_start;

  // Graph: [deadline-parameterized]  (single op, driven by DelayedDeadlinePortBindingFactory)
  //   The factory injects a kSpin event and advances the clock 50 ns, so a generation-1
  //   deadline update submitted before start() returns controls the scheduling gate.
  const ExecutionPlan compiled = test::make_runtime_deadline_param_test_plan();
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.parameter_derived_terms().size(), 1U);
  EXPECT_EQ(compiled.parameter_derived_terms().front().kind,
            ParameterDerivedTermKind::kContractDeadline);
  EXPECT_EQ(compiled.parameter_derived_terms().front().update_disposition,
            ParameterDerivedTermUpdateDisposition::kRuntimeAtomic);
  const ParameterFieldPlanRecord field = parameter_field(compiled, "deadline");
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_port_binding_factory(
      compiled, std::make_shared<DelayedDeadlinePortBindingFactory>());

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_start_enters())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto owner = session.operator_ref(field.owner);
  ASSERT_TRUE(owner);
  auto control = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(control);
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = *owner, .field = field.field},
          .value = ParamValue::from(std::chrono::nanoseconds{100}),
      },
  };
  const ParamBatchUpdateResult accepted = control->submit(patch);
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.target_generation, 1U);

  release_start.release();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(test::RuntimeDeadlineParamProbeState::compute_count.load(std::memory_order_acquire),
            1U);
  EXPECT_EQ(
      test::RuntimeDeadlineParamProbeState::observed_deadline_ns.load(std::memory_order_acquire),
      100);
  EXPECT_EQ(
      test::RuntimeDeadlineParamProbeState::observed_generation.load(std::memory_order_acquire),
      1U);
  EXPECT_EQ(test::RuntimeDeadlineParamProbeState::observed_reason.load(std::memory_order_acquire),
            ActivationReason::kNominal);
  const auto result = control->await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->state, ParamUpdateState::kObserved);
}

// REQ: PARAM-018
TEST(ParameterUpdateTest, SameOperatorBatchRejectsAtomicallyBeforeAcceptingSequence) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true);
  const ExecutionPlan raw_plan = test::make_runtime_two_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  ASSERT_EQ(raw_plan.parameter_fields().size(), 2U);
  const ParameterFieldPlanRecord field = raw_plan.parameter_fields()[0];
  const ParameterFieldPlanRecord other_field = raw_plan.parameter_fields()[1];
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  FirstCallbackRelease release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));

  const auto run_owner = session.operator_ref(field.owner);
  ASSERT_TRUE(run_owner);
  const auto other_run_owner = session.operator_ref(other_field.owner);
  ASSERT_TRUE(other_run_owner);
  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(handle_result).value();
  const ParamAddress address{.operator_instance = *run_owner, .field = field.field};

  // Key setup: snapshot the published and accepted values before attempting any invalid
  // batch; all rejections must leave both views identical to these baselines.
  const auto before_published = handle.read(address, ParamReadView::kPublished);
  ASSERT_TRUE(before_published);
  const auto before_accepted = handle.read(address, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(before_accepted);

  const std::array one_patch{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };
  const ParamBatchUpdateResult invalid_audit =
      handle.submit(one_patch, ParamUpdateOptions{.audit = invalid_param_audit_for_test()});
  EXPECT_FALSE(invalid_audit.committed);
  EXPECT_EQ(invalid_audit.status, ParamUpdateStatus::kRejected);
  EXPECT_EQ(invalid_audit.update_sequence, 0U);

  const ParamBatchUpdateResult empty = handle.submit(std::span<const ParamPatch>{});
  EXPECT_FALSE(empty.committed);
  EXPECT_EQ(empty.update_sequence, 0U);

  const std::array cross_operator_patches{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
      ParamPatch{
          .target = ParamAddress{.operator_instance = *other_run_owner, .field = other_field.field},
          .value = ParamValue::from(std::int32_t{99})},
  };
  const ParamBatchUpdateResult cross_operator = handle.submit(cross_operator_patches);
  EXPECT_FALSE(cross_operator.committed);
  EXPECT_EQ(cross_operator.status, ParamUpdateStatus::kUnsupportedAtomicScope);
  EXPECT_EQ(cross_operator.update_sequence, 0U);

  ParamFieldId unknown_field = std::numeric_limits<ParamFieldId>::max();
  if (unknown_field == field.field) {
    --unknown_field;
  }
  const std::array patches{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
      ParamPatch{.target = ParamAddress{.operator_instance = *run_owner, .field = unknown_field},
                 .value = ParamValue::from(std::int32_t{99})},
  };
  const ParamBatchUpdateResult rejected = handle.submit(patches);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.status, ParamUpdateStatus::kUnknownParameter);
  EXPECT_EQ(rejected.failed_patch, 1U);
  EXPECT_EQ(rejected.update_sequence, 0U);
  EXPECT_EQ(rejected.target_generation, 0U);

  const auto after_published = handle.read(address, ParamReadView::kPublished);
  ASSERT_TRUE(after_published);
  const auto after_accepted = handle.read(address, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(after_accepted);
  // Key assertion: every rejected batch leaves the published and accepted views unchanged.
  EXPECT_EQ(after_published->value, before_published->value);
  EXPECT_EQ(after_published->generation, before_published->generation);
  EXPECT_EQ(after_published->update_sequence, before_published->update_sequence);
  EXPECT_EQ(after_accepted->value, before_accepted->value);
  EXPECT_EQ(after_accepted->generation, before_accepted->generation);
  EXPECT_EQ(after_accepted->update_sequence, before_accepted->update_sequence);

  release.release();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
}

// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterUpdateTest, RestrictedCapabilitiesEnforceRunOperatorFieldAndAwaitScope) {
  test::RuntimeProbeState::reset();
  // The two parameter owners execute on independent partition owners. Hold both first
  // activations so the field selected below is necessarily still inside compute when its update
  // is accepted, regardless of which owner reaches the shared probe first.
  test::RuntimeParamProbeState::reset(true, false, true);
  const ExecutionPlan raw_plan = test::make_runtime_two_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  ASSERT_EQ(raw_plan.parameter_fields().size(), 2U);
  const ParameterFieldPlanRecord first_field = raw_plan.parameter_fields()[0];
  const ParameterFieldPlanRecord second_field = raw_plan.parameter_fields()[1];
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  FirstCallbackRelease release;
  SecondCallbackRelease concurrent_release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(2U));
  const auto first_owner = session.operator_ref(first_field.owner);
  const auto second_owner = session.operator_ref(second_field.owner);
  ASSERT_TRUE(first_owner);
  ASSERT_TRUE(second_owner);
  const OperatorRef first_owner_value = checked_ref(first_owner);
  const OperatorRef second_owner_value = checked_ref(second_owner);

  // Key setup: derive a chain of narrowed handles — run-scope, operator-scope, field-scope —
  // then verify each lower handle cannot be widened back upward and cross-scope
  // reads/submits are rejected with kPermissionDenied.
  auto run_handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(run_handle_result);
  ParamControlHandle run_handle = checked_value(std::move(run_handle_result));
  ParamAccessPolicy read_only_policy = full_local_parameter_policy();
  read_only_policy.permissions = ParamPermission::kRead;
  read_only_policy.max_event_cursors = 0U;
  const auto read_only_result = run_handle.restrict(read_only_policy);
  ASSERT_TRUE(read_only_result);
  EXPECT_EQ(checked_ref(read_only_result).catalog().compatibility_digest(),
            run_handle.catalog().compatibility_digest());
  const ParamAddress first_address{
      .operator_instance = first_owner_value,
      .field = first_field.field,
  };
  const ParamAddress second_address{
      .operator_instance = second_owner_value,
      .field = second_field.field,
  };

  ParamAccessPolicy first_policy = full_local_parameter_policy();
  first_policy.allowed_scope = ParamScopeFilter::for_operator(first_owner_value);
  auto first_handle_result = run_handle.restrict(first_policy);
  ASSERT_TRUE(first_handle_result);
  const ParamControlHandle first_handle = checked_value(std::move(first_handle_result));
  EXPECT_NE(first_handle.catalog().compatibility_digest(),
            run_handle.catalog().compatibility_digest());
  EXPECT_TRUE(first_handle.read(first_address));
  const auto denied_read = first_handle.read(second_address);
  ASSERT_FALSE(denied_read);
  EXPECT_EQ(denied_read.error().code, ErrorCode::kPermissionDenied);

  ParamAccessPolicy field_policy = full_local_parameter_policy();
  field_policy.allowed_scope = ParamScopeFilter::for_field(first_address);
  auto field_handle_result = first_handle.restrict(field_policy);
  ASSERT_TRUE(field_handle_result);
  ParamControlHandle field_handle = checked_value(std::move(field_handle_result));
  const ParamControlHandle copied_field_handle = field_handle;
  EXPECT_TRUE(field_handle.read(first_address));
  EXPECT_TRUE(copied_field_handle.read(first_address));
  const auto copied_denied_read = copied_field_handle.read(second_address);
  ASSERT_FALSE(copied_denied_read);
  EXPECT_EQ(copied_denied_read.error().code, ErrorCode::kPermissionDenied);
  const auto field_snapshot = field_handle.capture(ParamCaptureOptions{
      .consistency = ParamCaptureConsistency::kLivePerOperator,
  });
  ASSERT_TRUE(field_snapshot);
  const auto& field_snapshot_value = checked_ref(field_snapshot);
  EXPECT_EQ(field_snapshot_value.scope().kind(), ParamSnapshotScopeKind::kOperator);
  EXPECT_EQ(field_snapshot_value.catalog_digest(), field_handle.catalog().compatibility_digest());
  EXPECT_EQ(field_handle.catalog().compatibility_digest(),
            first_handle.catalog().compatibility_digest());
  EXPECT_NE(field_snapshot_value.catalog_digest(), run_handle.catalog().compatibility_digest());
  ASSERT_EQ(field_snapshot_value.operators().size(), 1U);
  ASSERT_EQ(field_snapshot_value.operators().front().fields.size(), 1U);

  ParamAccessPolicy capture_only_policy = field_policy;
  capture_only_policy.permissions = ParamPermission::kCapture;
  capture_only_policy.max_event_cursors = 0U;
  auto capture_only_result = field_handle.restrict(capture_only_policy);
  ASSERT_TRUE(capture_only_result);
  const ParamControlHandle capture_only = checked_value(std::move(capture_only_result));
  EXPECT_TRUE(capture_only.catalog().operators().empty());
  EXPECT_TRUE(capture_only.catalog().fields().empty());
  EXPECT_NE(capture_only.catalog().compatibility_digest(), ParamCatalogDigest{});
  const auto capture_only_snapshot = capture_only.capture(ParamCaptureOptions{
      .consistency = ParamCaptureConsistency::kLivePerOperator,
  });
  ASSERT_TRUE(capture_only_snapshot);
  const auto& capture_only_snapshot_value = checked_ref(capture_only_snapshot);
  EXPECT_EQ(capture_only_snapshot_value.catalog_digest(), field_snapshot_value.catalog_digest());
  EXPECT_EQ(capture_only_snapshot_value.catalog_digest(),
            field_handle.catalog().compatibility_digest());
  EXPECT_NE(capture_only_snapshot_value.catalog_digest(),
            capture_only.catalog().compatibility_digest());
  ASSERT_EQ(capture_only_snapshot_value.operators().size(),
            field_snapshot_value.operators().size());
  EXPECT_EQ(capture_only_snapshot_value.operators().front().operator_instance,
            field_snapshot_value.operators().front().operator_instance);
  ASSERT_EQ(capture_only_snapshot_value.operators().front().fields.size(),
            field_snapshot_value.operators().front().fields.size());
  EXPECT_EQ(capture_only_snapshot_value.operators().front().fields.front().address.field,
            field_snapshot_value.operators().front().fields.front().address.field);

  const auto same_operator_slice = field_snapshot_value.slice(field_snapshot_value.scope());
  ASSERT_TRUE(same_operator_slice);
  const auto& same_operator_slice_value = checked_ref(same_operator_slice);
  EXPECT_EQ(same_operator_slice_value.catalog_digest(), field_snapshot_value.catalog_digest());
  ASSERT_EQ(same_operator_slice_value.operators().size(), 1U);
  ASSERT_EQ(same_operator_slice_value.operators().front().fields.size(), 1U);
  EXPECT_EQ(same_operator_slice_value.operators().front().fields.front().address.field,
            field_snapshot_value.operators().front().fields.front().address.field);
  // Key assertion: a field-scoped snapshot cannot be sliced to a wider (root) scope.
  const auto widened_snapshot = field_snapshot_value.slice(ParamSnapshotScope::root());
  ASSERT_FALSE(widened_snapshot);
  EXPECT_EQ(widened_snapshot.error().code, ErrorCode::kPermissionDenied);

  ParamAccessPolicy second_policy = full_local_parameter_policy();
  second_policy.allowed_scope = ParamScopeFilter::for_operator(second_owner_value);
  auto second_handle_result = run_handle.restrict(second_policy);
  ASSERT_TRUE(second_handle_result);
  ParamControlHandle second_handle = checked_value(std::move(second_handle_result));

  ParamAccessPolicy update_only_policy = full_local_parameter_policy();
  update_only_policy.permissions = ParamPermission::kUpdate;
  update_only_policy.max_event_cursors = 0U;
  auto update_only_result = run_handle.restrict(update_only_policy);
  ASSERT_TRUE(update_only_result);
  const ParamControlHandle update_only = checked_value(std::move(update_only_result));
  EXPECT_TRUE(update_only.catalog().operators().empty());
  EXPECT_TRUE(update_only.catalog().fields().empty());
  EXPECT_NE(update_only.catalog().compatibility_digest(), ParamCatalogDigest{});
  const ParamCatalogDigest first_empty_catalog_digest =
      update_only.catalog().compatibility_digest();
  const auto denied_capture = update_only.capture(ParamCaptureOptions{
      .consistency = ParamCaptureConsistency::kLivePerOperator,
  });
  ASSERT_FALSE(denied_capture);
  EXPECT_EQ(denied_capture.error().code, ErrorCode::kPermissionDenied);
  const auto denied_cursor = update_only.open_event_cursor();
  ASSERT_FALSE(denied_cursor);
  EXPECT_EQ(denied_cursor.error().code, ErrorCode::kPermissionDenied);

  const std::array first_patch{
      ParamPatch{.target = first_address, .value = ParamValue::from(std::int32_t{42})},
  };
  const ParamBatchUpdateResult accepted = field_handle.submit(first_patch.front());
  ASSERT_TRUE(accepted.committed);
  ASSERT_NE(accepted.update_sequence, 0U);
  const auto field_await = field_handle.await(accepted.update_sequence, 0ns);
  ASSERT_FALSE(field_await);
  EXPECT_EQ(field_await.error().code, ErrorCode::kTimeout);
  const auto denied_await = second_handle.await(accepted.update_sequence, 0ns);
  ASSERT_FALSE(denied_await);
  EXPECT_EQ(denied_await.error().code, ErrorCode::kPermissionDenied);
  const ParamBatchUpdateResult denied_submit = second_handle.submit(first_patch);
  EXPECT_FALSE(denied_submit.committed);
  EXPECT_EQ(denied_submit.status, ParamUpdateStatus::kRejected);

  const auto widening = first_handle.restrict(second_policy);
  ASSERT_FALSE(widening);
  EXPECT_EQ(widening.error().code, ErrorCode::kPermissionDenied);

  ParamAccessPolicy foreign_policy = full_local_parameter_policy();
  OperatorRef foreign_owner = first_owner_value;
  ++foreign_owner.run_generation;
  foreign_policy.allowed_scope = ParamScopeFilter::for_operator(foreign_owner);
  const auto foreign = run_handle.restrict(foreign_policy);
  ASSERT_FALSE(foreign);
  EXPECT_EQ(foreign.error().code, ErrorCode::kInvalidArgument);

  for (const ParamScopeFilter invalid_scope : {
           ParamScopeFilter::for_operator(OperatorRef{}),
           ParamScopeFilter::for_operator(OperatorRef{
               .id = first_owner_value.id,
               .plan_generation = first_owner_value.plan_generation,
           }),
           ParamScopeFilter::for_field(ParamAddress{}),
           ParamScopeFilter::for_field(ParamAddress{.operator_instance = first_owner_value}),
       }) {
    EXPECT_NE(invalid_scope, ParamScopeFilter::run());
    ParamAccessPolicy invalid_policy = full_local_parameter_policy();
    invalid_policy.allowed_scope = invalid_scope;
    const auto invalid = run_handle.restrict(invalid_policy);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error().code, ErrorCode::kInvalidArgument);
  }

  ParamAccessPolicy unknown_field_policy = full_local_parameter_policy();
  unknown_field_policy.allowed_scope = ParamScopeFilter::for_field(ParamAddress{
      .operator_instance = first_owner_value,
      .field = std::numeric_limits<ParamFieldId>::max(),
  });
  const auto unknown_field = run_handle.restrict(unknown_field_policy);
  ASSERT_FALSE(unknown_field);
  EXPECT_EQ(unknown_field.error().code, ErrorCode::kInvalidArgument);

  {
    const detail::ParamGeneralControlContextScope callback_scope;
    const auto callback_read = run_handle.read(first_address);
    ASSERT_FALSE(callback_read);
    EXPECT_EQ(callback_read.error().code, ErrorCode::kInvalidArgument);
    const auto callback_restrict = run_handle.restrict(first_policy);
    ASSERT_FALSE(callback_restrict);
    EXPECT_EQ(callback_restrict.error().code, ErrorCode::kInvalidArgument);
    const ParamBatchUpdateResult callback_submit = run_handle.submit(first_patch);
    EXPECT_EQ(callback_submit.status, ParamUpdateStatus::kInvalidCallingContext);
    const auto callback_await = run_handle.await(accepted.update_sequence, 0ns);
    ASSERT_FALSE(callback_await);
    EXPECT_EQ(callback_await.error().code, ErrorCode::kInvalidArgument);
  }

  release.release();
  concurrent_release.release();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto completed_after_timeout = copied_field_handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(completed_after_timeout);
  EXPECT_EQ(checked_ref(completed_after_timeout).state, ParamUpdateState::kPublished);

  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true);
  RunSession second_session = run_async(plan);
  FirstCallbackRelease second_release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));
  auto second_run_handle_result = second_session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(second_run_handle_result);
  auto second_run_handle = checked_value(std::move(second_run_handle_result));
  auto second_update_only_result = second_run_handle.restrict(update_only_policy);
  ASSERT_TRUE(second_update_only_result);
  const auto& second_update_only = checked_ref(second_update_only_result);
  EXPECT_TRUE(second_update_only.catalog().operators().empty());
  EXPECT_TRUE(second_update_only.catalog().fields().empty());
  EXPECT_NE(second_update_only.catalog().compatibility_digest(), ParamCatalogDigest{});
  EXPECT_NE(second_update_only.catalog().compatibility_digest(), first_empty_catalog_digest);
  second_release.release();
  second_session.wait();
  EXPECT_EQ(second_session.termination(), RunTermination::kCompleted);
}

// REQ: PARAM-022
TEST(ParameterControlDiscoveryTest, MintingPreparesCatalogAndReportsAllocationFailure) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true);
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  FirstCallbackRelease release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));
  const auto owner = session.operator_ref(field.owner);
  ASSERT_TRUE(owner);
  const OperatorRef owner_value = checked_ref(owner);

  {
    // The handle state allocation succeeds; the first catalog-owned allocation
    // then fails and must remain a typed minting error rather than an empty view.
    const test::ScopedAllocationFailure fail_catalog{1U};
    const auto failed = session.parameter_control(full_local_parameter_policy());
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().code, ErrorCode::kResourceExhausted);
  }

  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  const ParamControlHandle handle = checked_value(std::move(handle_result));
  std::size_t accessor_allocations = 1U;
  const ParameterCatalogView* first_catalog = nullptr;
  const ParameterCatalogView* second_catalog = nullptr;
  {
    const test::ScopedAllocationCounter allocations;
    first_catalog = &handle.catalog();
    second_catalog = &handle.catalog();
    accessor_allocations = allocations.count();
  }
  EXPECT_EQ(accessor_allocations, 0U);
  EXPECT_EQ(first_catalog, second_catalog);
  ASSERT_NE(first_catalog, nullptr);
  EXPECT_EQ(first_catalog->fields().size(), 1U);

  ParamAccessPolicy owner_policy = full_local_parameter_policy();
  owner_policy.allowed_scope = ParamScopeFilter::for_operator(owner_value);
  {
    const test::ScopedAllocationFailure fail_child_catalog{1U};
    const auto failed = handle.restrict(owner_policy);
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().code, ErrorCode::kResourceExhausted);
  }

  release.release();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(handle.catalog().fields().size(), 1U);
}

// REQ: PARAM-023
TEST(ParameterControlDiscoveryTest, StopAndHandleMintRaceWithStartupRetirementClosesSafely) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ExecutionPlan invalid_plan =
      detail::RuntimeTestPlanAccess::invalidate_parameter_runtime_for_test(raw_plan);

  RunSession session = run_async(invalid_plan);
  std::thread stopper{[&] { session.request_stop(); }};
  const auto handle = session.parameter_control(full_local_parameter_policy());
  stopper.join();
  ASSERT_FALSE(handle);
  EXPECT_EQ(handle.error().code, ErrorCode::kClosed);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  session.request_stop();
}

// REQ: PARAM-023
TEST(ParameterControlDiscoveryTest, StopAndHandleMintRaceWithStartupAttachmentPreparesCatalog) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  std::optional<expected<ParamControlHandle, Error>> handle;
  std::thread minter{
      [&] { handle.emplace(session.parameter_control(full_local_parameter_policy())); }};
  session.request_stop();
  minter.join();
  ASSERT_TRUE(handle.has_value());
  auto& handle_result = checked_ref(handle);
  ASSERT_TRUE(handle_result);
  auto& control_handle = checked_ref(handle_result);
  EXPECT_EQ(control_handle.catalog().fields().size(), 1U);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_EQ(control_handle.catalog().fields().size(), 1U);
}

// REQ: PARAM-017
// REQ: PARAM-019
// REQ: PARAM-022
TEST(ParameterControlDiscoveryTest, AttachesCatalogAndGenerationZeroBeforeStartReturns) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  test::RuntimeParamProbeState::hold_start_callback();
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  StartCallbackRelease release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_start_enters());

  const auto owner = session.operator_ref(field.owner);
  ASSERT_TRUE(owner);
  const OperatorRef owner_value = checked_ref(owner);
  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  const ParamControlHandle handle = checked_value(std::move(handle_result));
  ASSERT_EQ(handle.catalog().operators().size(), 1U);
  ASSERT_EQ(handle.catalog().fields().size(), 1U);
  const auto initial =
      handle.read(ParamAddress{.operator_instance = owner_value, .field = field.field},
                  ParamReadView::kPublished);
  ASSERT_TRUE(initial);
  const auto& initial_record = checked_ref(initial);
  EXPECT_EQ(initial_record.generation, 0U);
  const auto initial_value = initial_record.value.as<std::int32_t>();
  ASSERT_TRUE(initial_value);
  EXPECT_EQ(checked_ref(initial_value), 7);

  release.release();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto observations = test::RuntimeParamProbeState::snapshot();
  ASSERT_EQ(observations.size(), 1U);
  EXPECT_EQ(observations.front().before_generation, 0U);
  EXPECT_EQ(observations.front().before_value, 7);
}

// REQ: PARAM-017
// REQ: PARAM-019
// REQ: PARAM-023
TEST(ParameterUpdateTest, GenerationRemainsPinnedAcrossConcurrentAcceptedUpdate) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true, false, true);
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_count(raw_plan, 2U);

  RunSession session = run_async(plan);
  FirstCallbackRelease first_release;
  SecondCallbackRelease second_release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));

  const auto run_owner = session.operator_ref(field.owner);
  ASSERT_TRUE(run_owner);
  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = checked_value(std::move(handle_result));
  const ParamAddress address{.operator_instance = *run_owner, .field = field.field};
  const std::array patches{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };
  const ParamBatchUpdateResult accepted =
      handle.submit(patches, ParamUpdateOptions{.effective = NextCompute{}});
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.status, ParamUpdateStatus::kAccepted);
  EXPECT_NE(accepted.update_sequence, 0U);
  EXPECT_EQ(accepted.target_generation, 1U);

  const auto still_published = handle.read(address, ParamReadView::kPublished);
  ASSERT_TRUE(still_published);
  const auto published_value = still_published->value.as<std::int32_t>();
  ASSERT_TRUE(published_value);
  EXPECT_EQ(*published_value, 7);
  EXPECT_EQ(still_published->generation, 0U);
  const auto latest_accepted = handle.read(address, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(latest_accepted);
  const auto accepted_value = latest_accepted->value.as<std::int32_t>();
  ASSERT_TRUE(accepted_value);
  EXPECT_EQ(*accepted_value, 42);
  EXPECT_EQ(latest_accepted->generation, 1U);
  EXPECT_EQ(latest_accepted->update_sequence, accepted.update_sequence);

  const auto while_blocked = test::RuntimeParamProbeState::snapshot();
  ASSERT_EQ(while_blocked.size(), 1U);
  EXPECT_EQ(while_blocked[0].before_value, 7);
  EXPECT_EQ(while_blocked[0].before_generation, 0U);

  first_release.release();
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(2U));
  const auto observed_before_compute_return = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(observed_before_compute_return);
  EXPECT_EQ(observed_before_compute_return->state, ParamUpdateState::kObserved);
  second_release.release();
  const auto completed = handle.await(accepted.update_sequence, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, ParamUpdateState::kObserved);
  EXPECT_EQ(completed->target_generation, 1U);
  session.wait();
  ASSERT_EQ(session.termination(), RunTermination::kCompleted);
  const auto observed = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(observed);
  EXPECT_EQ(observed->state, ParamUpdateState::kObserved);

  const auto observations = test::RuntimeParamProbeState::snapshot();
  ASSERT_EQ(observations.size(), 2U);
  EXPECT_EQ(observations[0].before_value, 7);
  EXPECT_EQ(observations[0].after_value, 7);
  EXPECT_EQ(observations[0].before_generation, 0U);
  EXPECT_EQ(observations[0].after_generation, 0U);
  EXPECT_EQ(observations[0].before_address, observations[0].after_address);
  EXPECT_EQ(observations[0].before_tag, observations[0].after_tag);
  EXPECT_EQ(observations[0].before_tag_time, observations[0].after_tag_time);
  EXPECT_EQ(observations[0].before_activation_time, observations[0].after_activation_time);
  EXPECT_EQ(observations[0].before_activation.reason, observations[0].after_activation.reason);
  EXPECT_EQ(observations[0].before_activation.policy, observations[0].after_activation.policy);
  EXPECT_EQ(observations[1].before_value, 42);
  EXPECT_EQ(observations[1].after_value, 42);
  EXPECT_EQ(observations[1].before_generation, 1U);
  EXPECT_EQ(observations[1].after_generation, 1U);
  EXPECT_EQ(observations[1].before_address, observations[1].after_address);
  EXPECT_EQ(observations[1].before_tag, observations[1].after_tag);
  EXPECT_EQ(observations[1].before_tag_time, observations[1].after_tag_time);
  EXPECT_EQ(observations[1].before_activation_time, observations[1].after_activation_time);
  EXPECT_EQ(observations[1].before_activation.reason, observations[1].after_activation.reason);
  EXPECT_EQ(observations[1].before_activation.policy, observations[1].after_activation.policy);
  EXPECT_EQ(observations[1].before_tag.generation, observations[0].before_tag.generation);
  EXPECT_GT(observations[1].before_tag.sequence, observations[0].before_tag.sequence);
  EXPECT_TRUE(test::RuntimeParamProbeState::stop_observed_params.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeParamProbeState::stop_observed_value.load(std::memory_order_relaxed), 42);
}

// REQ: PARAM-019
// REQ: PARAM-023
TEST(ParameterUpdateTest, AcceptedBurstAdvancesOneObservableGenerationPerActivation) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true, false, true);
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_count(raw_plan, 4U);

  RunSession session = run_async(plan);
  FirstCallbackRelease first_release;
  SecondCallbackRelease second_release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));

  const auto owner = session.operator_ref(field.owner);
  ASSERT_TRUE(owner);
  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  const ParamAddress address{.operator_instance = *owner, .field = field.field};
  std::array<ParamBatchUpdateResult, 3> accepted{};
  for (std::size_t index = 0U; index < accepted.size(); ++index) {
    const std::array patch{
        ParamPatch{.target = address,
                   .value = ParamValue::from(static_cast<std::int32_t>(42 + index))},
    };
    accepted[index] = handle.submit(patch, ParamUpdateOptions{.effective = NextCompute{}});
    ASSERT_TRUE(accepted[index].committed);
    EXPECT_EQ(accepted[index].target_generation, index + 1U);
  }

  first_release.release();
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(2U));
  const auto published_during_second = handle.read(address, ParamReadView::kPublished);
  ASSERT_TRUE(published_during_second);
  EXPECT_EQ(published_during_second->generation, 1U);
  const auto published_value = published_during_second->value.as<std::int32_t>();
  ASSERT_TRUE(published_value);
  EXPECT_EQ(*published_value, 42);
  const auto first_observed = handle.await(accepted[0].update_sequence, 0ns);
  ASSERT_TRUE(first_observed);
  EXPECT_EQ(first_observed->state, ParamUpdateState::kObserved);
  const auto second_not_yet_observed = handle.await(accepted[1].update_sequence, 0ns);
  ASSERT_FALSE(second_not_yet_observed);
  EXPECT_EQ(second_not_yet_observed.error().code, ErrorCode::kTimeout);

  second_release.release();
  session.wait();
  ASSERT_EQ(session.termination(), RunTermination::kCompleted);

  const auto observations = test::RuntimeParamProbeState::snapshot();
  ASSERT_EQ(observations.size(), 4U);
  const std::array expected_values{7, 42, 43, 44};
  for (std::size_t index = 0U; index < observations.size(); ++index) {
    EXPECT_EQ(observations[index].before_generation, index);
    EXPECT_EQ(observations[index].after_generation, index);
    EXPECT_EQ(observations[index].before_value, expected_values[index]);
    EXPECT_EQ(observations[index].after_value, expected_values[index]);
  }
  for (const ParamBatchUpdateResult& update : accepted) {
    const auto observed = handle.await(update.update_sequence, 0ns);
    ASSERT_TRUE(observed);
    EXPECT_EQ(observed->state, ParamUpdateState::kObserved);
  }
  EXPECT_TRUE(test::RuntimeParamProbeState::stop_observed_params.load(std::memory_order_acquire));
  EXPECT_EQ(test::RuntimeParamProbeState::stop_observed_value.load(std::memory_order_relaxed), 44);
}

// REQ: PARAM-019
// REQ: PARAM-023
TEST(ParameterUpdateTest, NextComputePublishesAfterFinalCallbackWithoutAnotherActivation) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true);
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  FirstCallbackRelease release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));
  const auto owner = session.operator_ref(field.owner);
  ASSERT_TRUE(owner);
  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  const ParamAddress address{.operator_instance = *owner, .field = field.field};
  const std::array patch{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };
  const ParamBatchUpdateResult accepted = handle.submit(patch);
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.target_generation, 1U);

  release.release();
  session.wait();
  ASSERT_EQ(session.termination(), RunTermination::kCompleted);
  ASSERT_EQ(test::RuntimeParamProbeState::snapshot().size(), 1U);

  const auto terminal = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->state, ParamUpdateState::kPublished);
  const auto published = handle.read(address, ParamReadView::kPublished);
  ASSERT_TRUE(published);
  EXPECT_EQ(published->generation, 1U);
  EXPECT_EQ(published->update_sequence, accepted.update_sequence);
  EXPECT_EQ(published->state, ParamUpdateState::kPublished);
  const auto value = published->value.as<std::int32_t>();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 42);
}

// REQ: PARAM-017
// REQ: PARAM-019
// REQ: PARAM-022
TEST(ParameterControlDiscoveryTest,
     CatalogPendingCaptureAndIndependentEventCursorsPreserveRunLineage) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true);
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_count(raw_plan, 2U);

  RunSession session = run_async(plan);
  FirstCallbackRelease release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));
  const auto run_owner = session.operator_ref(field.owner);
  ASSERT_TRUE(run_owner);
  const OperatorRef run_owner_value = checked_ref(run_owner);
  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(handle_result).value();

  const ParameterCatalogView& catalog = handle.catalog();
  ASSERT_EQ(catalog.operators().size(), 1U);
  ASSERT_EQ(catalog.fields().size(), 1U);
  ASSERT_EQ(catalog.operators().front().path.segments().size(), 1U);
  EXPECT_EQ(catalog.operators().front().path.segments().front(), "parameterized");
  EXPECT_EQ(catalog.operators().front().metadata_hash,
            plan.operator_parameters().front().metadata_hash);
  const ParamFieldCatalogRecord& catalog_field = catalog.fields().front();
  EXPECT_EQ(catalog_field.address.operator_instance, run_owner_value);
  EXPECT_EQ(catalog_field.address.field, field.field);
  EXPECT_EQ(catalog_field.full_id, field.full_id);
  EXPECT_EQ(catalog_field.logical_type, ParamLogicalType::kSignedInteger);
  EXPECT_EQ(catalog_field.type_shape.type, field.type_shape.type);
  EXPECT_EQ(catalog_field.type_shape.bit_width, field.type_shape.bit_width);
  EXPECT_EQ(catalog_field.type_shape.has_stable_shape, field.type_shape.has_stable_shape);
  EXPECT_EQ(catalog_field.value_route, ParamValueRouteDisposition::kPortableScalar);
  EXPECT_EQ(catalog_field.update_disposition, ParameterFieldUpdateDisposition::kAdmitted);
  EXPECT_FALSE(catalog_field.storage.owns_dynamic_storage);
  EXPECT_TRUE(catalog_field.storage.supports_bounded_materialization);

  const auto operator_path = OperatorPath::parse("parameterized");
  const auto parameter_path = ParamPath::parse("value");
  ASSERT_TRUE(operator_path);
  ASSERT_TRUE(parameter_path);
  const auto resolved = catalog.resolve(checked_ref(operator_path), checked_ref(parameter_path));
  ASSERT_TRUE(resolved);
  const ParamAddress resolved_address = checked_ref(resolved);
  EXPECT_EQ(resolved_address, catalog_field.address);
  const auto found_field = catalog.field(resolved_address);
  ASSERT_TRUE(found_field);
  EXPECT_EQ(checked_ref(found_field).get()->address, resolved_address);

  const auto stopped_while_running = handle.capture();
  ASSERT_FALSE(stopped_while_running);
  EXPECT_EQ(stopped_while_running.error().code, ErrorCode::kNotReady);
  const auto live_before_update = handle.capture(ParamCaptureOptions{
      .consistency = ParamCaptureConsistency::kLivePerOperator,
  });
  ASSERT_TRUE(live_before_update);
  const auto& live_snapshot = checked_ref(live_before_update);
  EXPECT_FALSE(live_snapshot.accepted_through().has_value());
  EXPECT_FALSE(live_snapshot.published_through().has_value());
  EXPECT_EQ(live_snapshot.plan_digest(), plan.digest());
  EXPECT_EQ(live_snapshot.plan_generation(), field.owner.plan_generation);
  EXPECT_EQ(live_snapshot.run_generation(), run_owner_value.run_generation);
  EXPECT_EQ(live_snapshot.catalog_digest(), catalog.compatibility_digest());
  EXPECT_EQ(live_snapshot.view(), ParamReadView::kPublished);
  EXPECT_EQ(live_snapshot.scope().kind(), ParamSnapshotScopeKind::kRoot);
  ASSERT_EQ(live_snapshot.operators().size(), 1U);
  EXPECT_EQ(live_snapshot.operators().front().metadata_hash,
            plan.operator_parameters().front().metadata_hash);
  ASSERT_EQ(live_snapshot.operators().front().fields.size(), 1U);
  const ParamSnapshotFieldRecord& initial_field = live_snapshot.operators().front().fields.front();
  EXPECT_EQ(initial_field.full_id, field.full_id);
  EXPECT_EQ(initial_field.type.id, "holoscan.i32");
  EXPECT_EQ(initial_field.type.version, 1U);
  EXPECT_EQ(initial_field.type.shape.type, field.type_shape.type);
  EXPECT_EQ(initial_field.type.shape.bit_width, field.type_shape.bit_width);
  ASSERT_TRUE(initial_field.codec.has_value());
  const auto initial_codec = checked_value(initial_field.codec);
  EXPECT_EQ(initial_codec.id, "holoscan.config.i32");
  EXPECT_EQ(initial_codec.version, 1U);
  EXPECT_EQ(initial_field.projection, params::ParamProjectionStatus::kIncluded);
  ASSERT_TRUE(initial_field.value.has_value());
  const auto initial_value = checked_value(initial_field.value).as<std::int32_t>();
  ASSERT_TRUE(initial_value);
  EXPECT_EQ(checked_ref(initial_value), 7);

  const ParamAddress address{.operator_instance = run_owner_value, .field = field.field};
  const auto no_pending = handle.read(address, ParamReadView::kPending);
  ASSERT_FALSE(no_pending);
  EXPECT_EQ(no_pending.error().code, ErrorCode::kNotReady);

  auto cursor_a_result = handle.open_event_cursor();
  auto cursor_b_result = handle.open_event_cursor();
  ASSERT_TRUE(cursor_a_result);
  ASSERT_TRUE(cursor_b_result);
  ParamEventCursor cursor_a = checked_value(std::move(cursor_a_result));
  ParamEventCursor cursor_b = checked_value(std::move(cursor_b_result));
  const auto invalid_cursor = handle.open_event_cursor(ParamEventCursorOptions{
      .after_event_sequence = 1U,
  });
  ASSERT_FALSE(invalid_cursor);
  EXPECT_EQ(invalid_cursor.error().code, ErrorCode::kInvalidArgument);
  const auto cursor_exhausted = handle.open_event_cursor();
  ASSERT_FALSE(cursor_exhausted);
  EXPECT_EQ(cursor_exhausted.error().code, ErrorCode::kResourceExhausted);

  const std::array patches{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };

  // Generate two policy-hidden source-ledger records before the accepted update. Restricted
  // cursor sequences must remain contiguous rather than exposing those records as sequence holes.
  const std::array hidden_patches{
      ParamPatch{
          .target = ParamAddress{.operator_instance = run_owner_value, .field = field.field + 1U},
          .value = ParamValue::from(std::int32_t{41})},
  };
  const ParamBatchUpdateResult hidden = handle.submit(hidden_patches);
  ASSERT_FALSE(hidden.committed);
  EXPECT_EQ(hidden.status, ParamUpdateStatus::kUnknownParameter);
  const std::array visible_rejected_patches{
      ParamPatch{.target = address, .value = ParamValue::from(std::uint32_t{41})},
  };
  const ParamBatchUpdateResult visible_rejected = handle.submit(visible_rejected_patches);
  ASSERT_FALSE(visible_rejected.committed);
  EXPECT_EQ(visible_rejected.status, ParamUpdateStatus::kTypeMismatch);
  EXPECT_EQ(visible_rejected.update_sequence, 0U);
  const ParamBatchUpdateResult accepted = handle.submit(patches);
  ASSERT_TRUE(accepted.committed);
  const auto unsupported_live_frontier = handle.capture(ParamCaptureOptions{
      .consistency = ParamCaptureConsistency::kLivePerOperator,
      .after_update_sequence = accepted.update_sequence,
  });
  ASSERT_FALSE(unsupported_live_frontier);
  EXPECT_EQ(unsupported_live_frontier.error().code, ErrorCode::kNotSupported);
  const auto pending = handle.read(address, ParamReadView::kPending);
  ASSERT_TRUE(pending);
  const auto& pending_record = checked_ref(pending);
  EXPECT_EQ(pending_record.state, ParamUpdateState::kAccepted);
  EXPECT_EQ(pending_record.update_sequence, accepted.update_sequence);
  const auto pending_value = pending_record.value.as<std::int32_t>();
  ASSERT_TRUE(pending_value);
  EXPECT_EQ(checked_ref(pending_value), 42);

  release.release();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto completed = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(completed);
  EXPECT_EQ(checked_ref(completed).state, ParamUpdateState::kObserved);

  const auto stopped = handle.capture(ParamCaptureOptions{
      .after_update_sequence = accepted.update_sequence,
  });
  ASSERT_TRUE(stopped);
  const auto& stopped_snapshot = checked_ref(stopped);
  ASSERT_TRUE(stopped_snapshot.accepted_through().has_value());
  ASSERT_TRUE(stopped_snapshot.published_through().has_value());
  EXPECT_EQ(checked_value(stopped_snapshot.accepted_through()), accepted.update_sequence);
  EXPECT_EQ(checked_value(stopped_snapshot.published_through()), accepted.update_sequence);
  ASSERT_EQ(stopped_snapshot.operators().size(), 1U);
  EXPECT_EQ(stopped_snapshot.operators().front().accepted_update_sequence,
            accepted.update_sequence);
  EXPECT_EQ(stopped_snapshot.operators().front().published_update_sequence,
            accepted.update_sequence);
  EXPECT_EQ(stopped_snapshot.operators().front().schema_id,
            "nvidia.holoscan.test.RuntimeProbeParams");
  EXPECT_EQ(stopped_snapshot.operators().front().state, ParamUpdateState::kObserved);
  ASSERT_EQ(stopped_snapshot.operators().front().fields.size(), 1U);
  const auto captured_value =
      checked_value(stopped_snapshot.operators().front().fields.front().value).as<std::int32_t>();
  ASSERT_TRUE(captured_value);
  EXPECT_EQ(checked_ref(captured_value), 42);
  const auto sliced = stopped_snapshot.slice("parameterized");
  ASSERT_TRUE(sliced);
  ASSERT_EQ(checked_ref(sliced).operators().size(), 1U);

  std::array<ParamEvent, 16> events_a{};
  std::array<ParamEvent, 16> events_b{};
  const auto drained_a = cursor_a.drain(events_a);
  const auto drained_b = cursor_b.drain(events_b);
  ASSERT_TRUE(drained_a);
  ASSERT_TRUE(drained_b);
  const std::size_t drained_a_count = checked_ref(drained_a);
  const std::size_t drained_b_count = checked_ref(drained_b);
  ASSERT_EQ(drained_a_count, drained_b_count);
  ASSERT_GE(drained_a_count, 6U);
  for (std::size_t index = 0U; index < drained_a_count; ++index) {
    EXPECT_EQ(events_a[index].event_sequence, index + 1U);
    EXPECT_EQ(events_a[index].event_sequence, events_b[index].event_sequence);
    EXPECT_EQ(events_a[index].state, events_b[index].state);
  }
  const auto has_state = [&](ParamUpdateState state) {
    return std::any_of(events_a.begin(),
                       events_a.begin() + drained_a_count,
                       [&](const ParamEvent& event) { return event.state == state; });
  };
  EXPECT_TRUE(has_state(ParamUpdateState::kSubmitted));
  EXPECT_TRUE(has_state(ParamUpdateState::kRejected));
  EXPECT_TRUE(has_state(ParamUpdateState::kAccepted));
  EXPECT_TRUE(has_state(ParamUpdateState::kPublished));
  EXPECT_TRUE(has_state(ParamUpdateState::kObserved));
}

// REQ: PARAM-017
// REQ: PARAM-019
// REQ: PARAM-021
TEST(ParameterRuntimeIntegrationTest,
     LifecycleUpdateUsesCandidateOnlyForForwardAndPublishesAfterStart) {
  test::RuntimeProbeState::reset();
  test::LifecycleParamProbeState::reset(true, false);
  const ExecutionPlan raw_plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(raw_plan.ok()) << (raw_plan.diagnostics().empty()
                                     ? ""
                                     : raw_plan.diagnostics().front().token);
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::LifecycleParamProbeState::wait_until_compute_count(1U));
  const auto owner = session.operator_ref(field.owner);
  ASSERT_TRUE(owner);
  auto control_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(control_result);
  ParamControlHandle control = std::move(*control_result);
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = *owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult accepted = control.submit(patch);
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.status, ParamUpdateStatus::kAccepted);

  test::LifecycleParamProbeState::release_first_callback();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto terminal = control.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(terminal);
  EXPECT_TRUE(terminal->state == ParamUpdateState::kPublished ||
              terminal->state == ParamUpdateState::kObserved);
  EXPECT_EQ(terminal->target_generation, 1U);
  EXPECT_NE(terminal->lifecycle_transaction_id, 0U);

  const auto lifecycle = test::LifecycleParamProbeState::lifecycle_snapshot();
  const auto contains = [&](LifecycleStage stage,
                            LifecycleTransitionPhase phase,
                            std::int32_t value,
                            ParamGeneration generation) {
    return std::any_of(lifecycle.begin(), lifecycle.end(), [&](const auto& observation) {
      return observation.stage == stage && observation.phase == phase &&
             observation.value == value && observation.generation == generation;
    });
  };
  EXPECT_TRUE(
      contains(LifecycleStage::kStop, LifecycleTransitionPhase::kPreRestartTeardown, 7, 0U));
  EXPECT_TRUE(
      contains(LifecycleStage::kRelease, LifecycleTransitionPhase::kPreRestartTeardown, 7, 0U));
  for (const LifecycleStage stage : {LifecycleStage::kAllocate,
                                     LifecycleStage::kWarmUp,
                                     LifecycleStage::kArm,
                                     LifecycleStage::kStart}) {
    EXPECT_TRUE(contains(stage, LifecycleTransitionPhase::kRestartForward, 42, 1U));
  }
  EXPECT_TRUE(contains(LifecycleStage::kStop, LifecycleTransitionPhase::kShutdownTeardown, 42, 1U));
  EXPECT_TRUE(
      contains(LifecycleStage::kRelease, LifecycleTransitionPhase::kShutdownTeardown, 42, 1U));
  const auto compute = test::LifecycleParamProbeState::compute_snapshot();
  ASSERT_FALSE(compute.empty());
  EXPECT_EQ(compute.front().operator_identity, 1);
  EXPECT_EQ(compute.front().value, 7);
  EXPECT_EQ(compute.front().generation, 0U);
}

// REQ: PARAM-017
// REQ: PARAM-018
// REQ: PARAM-021
TEST(ParameterRuntimeIntegrationTest,
     LifecycleUpdatesForTwoOwnersExecuteOneUnionRestartAndPublishBothCandidates) {
  test::RuntimeProbeState::reset();
  test::LifecycleParamProbeState::reset(true, false);
  const ExecutionPlan raw_plan = test::make_two_lifecycle_param_test_plan();
  ASSERT_TRUE(raw_plan.ok()) << raw_plan.json();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::LifecycleParamProbeState::wait_until_compute_count(1U));
  auto control_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(control_result);
  ParamControlHandle control = std::move(*control_result);

  std::array<ParamBatchUpdateResult, 2> accepted{};
  std::array<OperatorRef, 2> run_owners{};
  std::array<ParamFieldId, 2> field_ids{};
  const std::array values{std::int32_t{42}, std::int32_t{52}};
  for (std::size_t index = 0U; index < run_owners.size(); ++index) {
    const auto field =
        std::find_if(raw_plan.parameter_fields().begin(),
                     raw_plan.parameter_fields().end(),
                     [&](const auto& record) { return record.owner.id == index + 1U; });
    ASSERT_NE(field, raw_plan.parameter_fields().end());
    const auto owner = session.operator_ref(field->owner);
    ASSERT_TRUE(owner);
    run_owners[index] = *owner;
    field_ids[index] = field->field;
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = *owner, .field = field->field},
            .value = ParamValue::from(values[index]),
        },
    };
    accepted[index] = control.submit(patch);
    ASSERT_TRUE(accepted[index].committed);
  }
  test::LifecycleParamProbeState::release_first_callback();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  std::array<ParamUpdateResult, 2> terminal{};
  for (std::size_t index = 0U; index < accepted.size(); ++index) {
    const auto result = control.await(accepted[index].update_sequence, 0ns);
    ASSERT_TRUE(result);
    terminal[index] = *result;
    EXPECT_TRUE(result->state == ParamUpdateState::kPublished ||
                result->state == ParamUpdateState::kObserved);
  }
  ASSERT_NE(terminal[0].lifecycle_transaction_id, 0U);
  EXPECT_EQ(terminal[1].lifecycle_transaction_id, terminal[0].lifecycle_transaction_id);
  EXPECT_EQ(terminal[1].coalesced_into_transaction, terminal[0].lifecycle_transaction_id);

  const auto compute = test::LifecycleParamProbeState::compute_snapshot();
  ASSERT_EQ(compute.size(), 2U);
  std::array<bool, 2> observed{};
  for (const auto& observation : compute) {
    ASSERT_GE(observation.operator_identity, 1);
    ASSERT_LE(observation.operator_identity, 2);
    observed[static_cast<std::size_t>(observation.operator_identity - 1)] = true;
    EXPECT_EQ(observation.value, 7);
    EXPECT_EQ(observation.generation, 0U);
  }
  EXPECT_TRUE(std::ranges::all_of(observed, [](bool value) { return value; }));

  const auto lifecycle = test::LifecycleParamProbeState::lifecycle_snapshot();
  for (std::size_t index = 0U; index < values.size(); ++index) {
    for (const LifecycleStage stage : {LifecycleStage::kAllocate,
                                       LifecycleStage::kWarmUp,
                                       LifecycleStage::kArm,
                                       LifecycleStage::kStart}) {
      EXPECT_TRUE(std::any_of(lifecycle.begin(), lifecycle.end(), [&](const auto& observation) {
        return observation.operator_identity == static_cast<std::int32_t>(index + 1U) &&
               observation.stage == stage &&
               observation.phase == LifecycleTransitionPhase::kRestartForward &&
               observation.value == values[index] && observation.generation == 1U;
      }));
    }
    const auto published = control.read(
        ParamAddress{.operator_instance = run_owners[index], .field = field_ids[index]},
        ParamReadView::kPublished);
    ASSERT_TRUE(published);
    const auto value = published->value.as<std::int32_t>();
    ASSERT_TRUE(value);
    EXPECT_EQ(*value, values[index]);
    EXPECT_EQ(published->generation, 1U);
  }
}

// The startup-prepared union graph contains all three Operators, while the transaction selects
// only two. The inactive member must remain outside every restart callback and Parameter update.
// REQ: PARAM-018
// REQ: PARAM-021
TEST(ParameterRuntimeIntegrationTest,
     LifecycleUpdatesForSubsetUseUnionGraphAndSkipUnselectedMember) {
  test::RuntimeProbeState::reset();
  test::LifecycleParamProbeState::reset(true, false);
  const ExecutionPlan raw_plan = test::make_three_lifecycle_param_test_plan();
  ASSERT_TRUE(raw_plan.ok()) << raw_plan.json();
  ASSERT_EQ(raw_plan.lifecycle_restart_closures().size(), 3U);
  EXPECT_TRUE(std::all_of(raw_plan.lifecycle_restart_closures().begin(),
                          raw_plan.lifecycle_restart_closures().end(),
                          [](const auto& closure) { return closure.closure.size() == 1U; }));
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::LifecycleParamProbeState::wait_until_compute_count(1U));
  auto control_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(control_result);
  ParamControlHandle control = std::move(*control_result);

  std::array<ParamBatchUpdateResult, 2> accepted{};
  std::array<OperatorRef, 3> run_owners{};
  std::array<ParamFieldId, 3> field_ids{};
  for (std::size_t index = 0U; index < run_owners.size(); ++index) {
    const auto field =
        std::find_if(raw_plan.parameter_fields().begin(),
                     raw_plan.parameter_fields().end(),
                     [&](const auto& record) { return record.owner.id == index + 1U; });
    ASSERT_NE(field, raw_plan.parameter_fields().end());
    const auto owner = session.operator_ref(field->owner);
    ASSERT_TRUE(owner);
    run_owners[index] = *owner;
    field_ids[index] = field->field;
    if (index < accepted.size()) {
      const std::array patch{
          ParamPatch{
              .target = ParamAddress{.operator_instance = *owner, .field = field->field},
              .value = ParamValue::from(std::int32_t{42 + static_cast<std::int32_t>(index) * 10}),
          },
      };
      accepted[index] = control.submit(patch);
      ASSERT_TRUE(accepted[index].committed);
    }
  }

  test::LifecycleParamProbeState::release_first_callback();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  std::array<ParamUpdateResult, 2> terminal{};
  for (std::size_t index = 0U; index < accepted.size(); ++index) {
    const auto result = control.await(accepted[index].update_sequence, 0ns);
    ASSERT_TRUE(result);
    terminal[index] = *result;
    EXPECT_TRUE(result->state == ParamUpdateState::kPublished ||
                result->state == ParamUpdateState::kObserved);
  }
  ASSERT_NE(terminal[0].lifecycle_transaction_id, 0U);
  EXPECT_EQ(terminal[1].lifecycle_transaction_id, terminal[0].lifecycle_transaction_id);
  EXPECT_EQ(terminal[1].coalesced_into_transaction, terminal[0].lifecycle_transaction_id);

  const auto lifecycle = test::LifecycleParamProbeState::lifecycle_snapshot();
  EXPECT_FALSE(std::any_of(lifecycle.begin(), lifecycle.end(), [](const auto& observation) {
    return observation.operator_identity == 3 &&
           (observation.phase == LifecycleTransitionPhase::kPreRestartTeardown ||
            observation.phase == LifecycleTransitionPhase::kRestartForward);
  }));
  for (const LifecycleStage stage : {LifecycleStage::kAllocate,
                                     LifecycleStage::kWarmUp,
                                     LifecycleStage::kArm,
                                     LifecycleStage::kStart}) {
    for (const std::int32_t identity : {1, 2}) {
      EXPECT_EQ(std::count_if(lifecycle.begin(),
                              lifecycle.end(),
                              [&](const auto& observation) {
                                return observation.operator_identity == identity &&
                                       observation.stage == stage &&
                                       observation.phase ==
                                           LifecycleTransitionPhase::kRestartForward;
                              }),
                1);
    }
    EXPECT_EQ(std::count_if(lifecycle.begin(),
                            lifecycle.end(),
                            [&](const auto& observation) {
                              return observation.operator_identity == 3 &&
                                     observation.stage == stage &&
                                     observation.phase == LifecycleTransitionPhase::kRestartForward;
                            }),
              0);
  }

  for (std::size_t index = 0U; index < run_owners.size(); ++index) {
    const auto published = control.read(
        ParamAddress{.operator_instance = run_owners[index], .field = field_ids[index]},
        ParamReadView::kPublished);
    ASSERT_TRUE(published);
    const auto value = published->value.as<std::int32_t>();
    ASSERT_TRUE(value);
    EXPECT_EQ(*value, index < accepted.size() ? 42 + static_cast<std::int32_t>(index) * 10 : 7);
    EXPECT_EQ(published->generation, index < accepted.size() ? 1U : 0U);
  }
}

// REQ: PARAM-021
// REQ: PARAM-023
TEST(ParameterRuntimeIntegrationTest,
     StopDuringLifecycleRestartNeverPublishesTheCandidatePastTheTerminalFence) {
  test::RuntimeProbeState::reset();
  test::LifecycleParamProbeState::reset(true, false, 1, true);
  const ExecutionPlan raw_plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(raw_plan.ok()) << (raw_plan.diagnostics().empty()
                                     ? ""
                                     : raw_plan.diagnostics().front().token);
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::LifecycleParamProbeState::wait_until_compute_count(1U));
  const auto owner = session.operator_ref(field.owner);
  ASSERT_TRUE(owner);
  auto control_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(control_result);
  ParamControlHandle control = std::move(*control_result);
  const ParamPatch patch{
      .target = ParamAddress{.operator_instance = *owner, .field = field.field},
      .value = ParamValue::from(std::int32_t{42}),
  };
  const ParamBatchUpdateResult accepted = control.submit(std::span{&patch, 1U});
  ASSERT_TRUE(accepted.committed);

  test::LifecycleParamProbeState::release_first_callback();
  ASSERT_TRUE(test::LifecycleParamProbeState::wait_until_restart_start());
  session.request_stop();
  test::LifecycleParamProbeState::release_restart_start_callback();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(test::LifecycleParamProbeState::restart_start_saw_cancellation());
  const auto terminal = control.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->state, ParamUpdateState::kFailedStopped);
  auto published = control.read(ParamAddress{.operator_instance = *owner, .field = field.field},
                                ParamReadView::kPublished);
  ASSERT_TRUE(published);
  const auto value = published->value.as<std::int32_t>();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 7);
  EXPECT_EQ(published->generation, 0U);
}

// REQ: PARAM-017
// REQ: PARAM-021
TEST(ParameterRuntimeIntegrationTest, LifecycleFailureRollsBackCandidateAndRestoresPublishedView) {
  test::RuntimeProbeState::reset();
  test::LifecycleParamProbeState::reset(true, true);
  const ExecutionPlan raw_plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(raw_plan.ok()) << (raw_plan.diagnostics().empty()
                                     ? ""
                                     : raw_plan.diagnostics().front().token);
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::LifecycleParamProbeState::wait_until_compute_count(1U));
  const auto owner = session.operator_ref(field.owner);
  ASSERT_TRUE(owner);
  auto control_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(control_result);
  ParamControlHandle control = std::move(*control_result);
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = *owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult accepted = control.submit(patch);
  ASSERT_TRUE(accepted.committed);

  test::LifecycleParamProbeState::release_first_callback();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto terminal = control.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->state, ParamUpdateState::kRolledBack);
  auto published = control.read(ParamAddress{.operator_instance = *owner, .field = field.field},
                                ParamReadView::kPublished);
  ASSERT_TRUE(published);
  const auto restored = published->value.as<std::int32_t>();
  ASSERT_TRUE(restored);
  EXPECT_EQ(*restored, 7);
  EXPECT_EQ(published->generation, 0U);

  const auto lifecycle = test::LifecycleParamProbeState::lifecycle_snapshot();
  const auto candidate_rollback =
      std::find_if(lifecycle.begin(), lifecycle.end(), [](const auto& row) {
        return row.phase == LifecycleTransitionPhase::kForwardRollback && row.value == 42 &&
               row.generation == 1U;
      });
  EXPECT_NE(candidate_rollback, lifecycle.end());
  const auto restored_old = std::find_if(lifecycle.begin(), lifecycle.end(), [](const auto& row) {
    return row.phase == LifecycleTransitionPhase::kRestoreOld && row.value == 7 &&
           row.generation == 0U;
  });
  EXPECT_NE(restored_old, lifecycle.end());
}

// REQ: PARAM-017
// REQ: PARAM-021
TEST(ParameterRuntimeIntegrationTest,
     LifecycleDependencyRetainsPublishedViewThroughForwardAndRollback) {
  test::RuntimeProbeState::reset();
  test::LifecycleParamProbeState::reset(true, true, 2);
  const ExecutionPlan raw_plan = test::make_lifecycle_param_dependency_test_plan();
  ASSERT_TRUE(raw_plan.ok()) << (raw_plan.diagnostics().empty()
                                     ? ""
                                     : raw_plan.diagnostics().front().token);
  const auto field = std::find_if(raw_plan.parameter_fields().begin(),
                                  raw_plan.parameter_fields().end(),
                                  [](const auto& row) { return row.owner.id == 1U; });
  ASSERT_NE(field, raw_plan.parameter_fields().end());
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::LifecycleParamProbeState::wait_until_compute_count(1U));
  const auto owner = session.operator_ref(field->owner);
  ASSERT_TRUE(owner);
  auto control_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(control_result);
  ParamControlHandle control = std::move(*control_result);
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = *owner, .field = field->field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult accepted = control.submit(patch);
  ASSERT_TRUE(accepted.committed);

  test::LifecycleParamProbeState::release_first_callback();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  const auto terminal = control.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->state, ParamUpdateState::kRolledBack);

  const auto lifecycle = test::LifecycleParamProbeState::lifecycle_snapshot();
  const auto contains = [&](std::int32_t identity,
                            LifecycleTransitionPhase phase,
                            std::int32_t value,
                            ParamGeneration generation) {
    return std::any_of(lifecycle.begin(), lifecycle.end(), [&](const auto& observation) {
      return observation.operator_identity == identity && observation.phase == phase &&
             observation.value == value && observation.generation == generation;
    });
  };
  EXPECT_TRUE(contains(1, LifecycleTransitionPhase::kRestartForward, 42, 1U));
  EXPECT_TRUE(contains(1, LifecycleTransitionPhase::kForwardRollback, 42, 1U));
  EXPECT_TRUE(contains(2, LifecycleTransitionPhase::kRestartForward, 7, 0U));
  EXPECT_TRUE(contains(2, LifecycleTransitionPhase::kForwardRollback, 7, 0U));
  EXPECT_TRUE(contains(1, LifecycleTransitionPhase::kRestoreOld, 7, 0U));
  EXPECT_TRUE(contains(2, LifecycleTransitionPhase::kRestoreOld, 7, 0U));
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterRuntimeIntegrationTest, GeneralControlRejectsEveryRuntimeCallbackIdentity) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true);
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  FirstCallbackRelease release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U))
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto owner = session.operator_ref(field.owner);
  ASSERT_TRUE(owner);
  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  auto event_cursor_result = handle.open_event_cursor();
  ASSERT_TRUE(event_cursor_result);
  ParamEventCursor event_cursor = std::move(*event_cursor_result);
  const ParamAddress address{.operator_instance = *owner, .field = field.field};
  const std::array patch{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };

  for (const detail::RuntimeCallbackContext callback :
       {detail::RuntimeCallbackContext::kCompute,
        detail::RuntimeCallbackContext::kLifecycle,
        detail::RuntimeCallbackContext::kDispatcher}) {
    const detail::RuntimeCallbackContextScope scope{callback};
    EXPECT_TRUE(handle.catalog().operators().empty());
    const auto read = handle.read(address);
    ASSERT_FALSE(read);
    EXPECT_EQ(read.error().code, ErrorCode::kInvalidArgument);
    const ParamBatchUpdateResult submitted = handle.submit(patch);
    EXPECT_EQ(submitted.status, ParamUpdateStatus::kInvalidCallingContext);
    const auto awaited = handle.await(1U, 0ns);
    ASSERT_FALSE(awaited);
    EXPECT_EQ(awaited.error().code, ErrorCode::kInvalidArgument);
    const auto restricted = handle.restrict(full_local_parameter_policy());
    ASSERT_FALSE(restricted);
    EXPECT_EQ(restricted.error().code, ErrorCode::kInvalidArgument);
    const auto captured = handle.capture();
    ASSERT_FALSE(captured);
    EXPECT_EQ(captured.error().code, ErrorCode::kInvalidArgument);
    const auto cursor = handle.open_event_cursor();
    ASSERT_FALSE(cursor);
    EXPECT_EQ(cursor.error().code, ErrorCode::kInvalidArgument);
    std::array<ParamEvent, 1U> events{};
    const auto drained = event_cursor.drain(events);
    ASSERT_FALSE(drained);
    EXPECT_EQ(drained.error().code, ErrorCode::kInvalidArgument);
    const auto minted = session.parameter_control(full_local_parameter_policy());
    ASSERT_FALSE(minted);
    EXPECT_EQ(minted.error().code, ErrorCode::kInvalidArgument);
    const auto closed = session.close_parameter_update_ingress();
    ASSERT_FALSE(closed);
    EXPECT_EQ(closed.error().code, ErrorCode::kInvalidArgument);
    const auto session_capture = session.capture_params();
    ASSERT_FALSE(session_capture);
    EXPECT_EQ(session_capture.error().code, ErrorCode::kInvalidArgument);
  }

  release.release();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateTest, CloseFenceIsIdempotentAndRetainedHandleClosesSafely) {
  std::optional<ParamControlHandle> retained;
  ParamAddress retained_address{};
  {
    test::RuntimeProbeState::reset();
    test::RuntimeParamProbeState::reset(true);
    const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
    ASSERT_TRUE(raw_plan.ok());
    const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_count(raw_plan, 2U);

    RunSession session = run_async(plan);
    FirstCallbackRelease release;
    ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));
    const auto run_owner = session.operator_ref(field.owner);
    ASSERT_TRUE(run_owner);
    retained_address = ParamAddress{.operator_instance = *run_owner, .field = field.field};

    auto handle_result = session.parameter_control(full_local_parameter_policy());
    ASSERT_TRUE(handle_result);
    ParamControlHandle handle = std::move(handle_result).value();
    retained.emplace(handle);
    ASSERT_EQ(handle.catalog().fields().size(), 1U);
    const std::array patches{
        ParamPatch{.target = retained_address, .value = ParamValue::from(std::int32_t{42})},
    };
    const ParamBatchUpdateResult accepted = handle.submit(patches);
    ASSERT_TRUE(accepted.committed);

    const auto first_close = session.close_parameter_update_ingress();
    ASSERT_TRUE(first_close);
    EXPECT_FALSE(first_close->already_closed);
    EXPECT_EQ(first_close->terminal_update_sequence, accepted.update_sequence);
    const auto repeated_close = session.close_parameter_update_ingress();
    ASSERT_TRUE(repeated_close);
    EXPECT_TRUE(repeated_close->already_closed);
    EXPECT_EQ(repeated_close->terminal_update_sequence, first_close->terminal_update_sequence);

    const ParamBatchUpdateResult after_close = handle.submit(patches);
    EXPECT_FALSE(after_close.committed);
    EXPECT_EQ(after_close.status, ParamUpdateStatus::kClosed);
    EXPECT_EQ(after_close.update_sequence, 0U);

    release.release();
    const auto completed = handle.await(accepted.update_sequence, 5s);
    ASSERT_TRUE(completed);
    // Publication is terminal for await; the dispatcher may refine the same
    // result to Observed after the next activation pins it.
    EXPECT_TRUE(completed->state == ParamUpdateState::kPublished ||
                completed->state == ParamUpdateState::kObserved);
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  }

  ASSERT_TRUE(retained.has_value());
  EXPECT_EQ(retained->catalog().fields().size(), 1U);
  const auto child_after_session = retained->restrict(full_local_parameter_policy());
  ASSERT_FALSE(child_after_session);
  EXPECT_EQ(child_after_session.error().code, ErrorCode::kClosed);
  const auto after_session = retained->read(retained_address, ParamReadView::kPublished);
  ASSERT_FALSE(after_session);
  EXPECT_EQ(after_session.error().code, ErrorCode::kClosed);
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateTest, LateSubmissionAfterFinalCallbackIsCanceledAtRunEnd) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(false, true);
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  StopCallbackRelease release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_stop_enters());

  const auto run_owner = session.operator_ref(field.owner);
  ASSERT_TRUE(run_owner);
  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(handle_result).value();
  const ParamAddress address{.operator_instance = *run_owner, .field = field.field};
  const std::array patches{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };
  const ParamBatchUpdateResult accepted = handle.submit(patches);
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.status, ParamUpdateStatus::kAccepted);

  const auto still_published = handle.read(address, ParamReadView::kPublished);
  ASSERT_TRUE(still_published);
  const auto published_value = still_published->value.as<std::int32_t>();
  ASSERT_TRUE(published_value);
  EXPECT_EQ(*published_value, 7);
  const auto still_pending = handle.await(accepted.update_sequence, 0ns);
  ASSERT_FALSE(still_pending);
  EXPECT_EQ(still_pending.error().code, ErrorCode::kTimeout);

  release.release();
  session.wait();
  ASSERT_EQ(session.termination(), RunTermination::kCompleted);

  const auto terminal = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->status, ParamUpdateStatus::kClosed);
  EXPECT_EQ(terminal->state, ParamUpdateState::kCanceled);
  EXPECT_EQ(terminal->target_generation, accepted.target_generation);

  const auto stopped = handle.capture();
  ASSERT_TRUE(stopped);
  ASSERT_TRUE(stopped->accepted_through().has_value());
  ASSERT_TRUE(stopped->published_through().has_value());
  EXPECT_EQ(checked_value(stopped->accepted_through()), accepted.update_sequence);
  EXPECT_EQ(checked_value(stopped->published_through()), 0U);
  ASSERT_EQ(stopped->operators().size(), 1U);
  EXPECT_EQ(stopped->operators().front().accepted_update_sequence, accepted.update_sequence);
  EXPECT_EQ(stopped->operators().front().published_update_sequence, 0U);

  const ParamBatchUpdateResult after_run = handle.submit(patches);
  EXPECT_FALSE(after_run.committed);
  EXPECT_EQ(after_run.status, ParamUpdateStatus::kClosed);
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateTest, StopDuringCallbackTerminalizesPendingUpdateBeforeWaitReturns) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true);
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  FirstCallbackRelease release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));

  const auto run_owner = session.operator_ref(field.owner);
  ASSERT_TRUE(run_owner);
  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(handle_result).value();
  const ParamAddress address{.operator_instance = *run_owner, .field = field.field};
  const std::array patches{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };
  const ParamBatchUpdateResult accepted = handle.submit(patches);
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.status, ParamUpdateStatus::kAccepted);

  session.request_stop();
  const auto still_pending = handle.await(accepted.update_sequence, 0ns);
  ASSERT_FALSE(still_pending);
  EXPECT_EQ(still_pending.error().code, ErrorCode::kTimeout);
  release.release();
  session.wait();
  ASSERT_EQ(session.termination(), RunTermination::kStopped);

  const auto terminal = handle.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->status, ParamUpdateStatus::kClosed);
  EXPECT_EQ(terminal->state, ParamUpdateState::kCanceled);
  EXPECT_EQ(terminal->target_generation, accepted.target_generation);
}

// REQ: PARAM-020
// REQ: PARAM-022
// REQ: PARAM-023
TEST(ParameterUpdateTest, CloseFenceTerminalizesEverySequenceThroughNWithBlockedWaiter) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true);
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(raw_plan);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  FirstCallbackRelease release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));
  const auto owner = session.operator_ref(field.owner);
  ASSERT_TRUE(owner);
  auto handle_result = session.parameter_control(full_local_parameter_policy());
  ASSERT_TRUE(handle_result);
  ParamControlHandle handle = std::move(*handle_result);
  auto cursor_result = handle.open_event_cursor();
  ASSERT_TRUE(cursor_result);
  ParamEventCursor cursor = std::move(*cursor_result);
  const ParamAddress address{.operator_instance = *owner, .field = field.field};
  const auto submit_value = [&](std::int32_t value) {
    const std::array patch{
        ParamPatch{.target = address, .value = ParamValue::from(value)},
    };
    return handle.submit(patch);
  };

  const std::array accepted{
      submit_value(42),
      submit_value(43),
      submit_value(44),
  };
  for (const auto& result : accepted) {
    ASSERT_TRUE(result.committed);
    EXPECT_EQ(result.status, ParamUpdateStatus::kAccepted);
  }
  ASSERT_LT(accepted[0].update_sequence, accepted[1].update_sequence);
  ASSERT_LT(accepted[1].update_sequence, accepted[2].update_sequence);

  const auto fence = session.close_parameter_update_ingress();
  ASSERT_TRUE(fence);
  EXPECT_EQ(fence->terminal_update_sequence, accepted.back().update_sequence);
  const ParamBatchUpdateResult rejected = submit_value(45);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.status, ParamUpdateStatus::kClosed);
  EXPECT_EQ(rejected.update_sequence, 0U);

  std::atomic_bool waiter_entered{};
  auto waiter = std::async(std::launch::async, [&] {
    waiter_entered.store(true, std::memory_order_release);
    return handle.await(accepted.back().update_sequence, 5s);
  });
  while (!waiter_entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  EXPECT_EQ(waiter.wait_for(20ms), std::future_status::timeout);

  session.request_stop();
  release.release();
  session.wait();
  ASSERT_EQ(session.termination(), RunTermination::kStopped);

  const auto waiter_result = waiter.get();
  ASSERT_TRUE(waiter_result);
  EXPECT_EQ(waiter_result->state, ParamUpdateState::kCanceled);
  EXPECT_EQ(waiter_result->update_sequence, accepted.back().update_sequence);
  for (const auto& expected : accepted) {
    const auto terminal = handle.await(expected.update_sequence, 0ns);
    ASSERT_TRUE(terminal);
    EXPECT_EQ(terminal->state, ParamUpdateState::kCanceled);
    EXPECT_EQ(terminal->update_sequence, expected.update_sequence);
  }

  std::array<ParamEvent, 16U> events{};
  const auto drained = cursor.drain(events);
  ASSERT_TRUE(drained);
  for (const auto& expected : accepted) {
    EXPECT_TRUE(
        std::any_of(events.begin(), events.begin() + *drained, [&](const ParamEvent& event) {
          return event.update_sequence == expected.update_sequence &&
                 event.state == ParamUpdateState::kCanceled;
        }));
  }
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateTest, ConcurrentStopRequestsRemainSafeAcrossWorkerCompletion) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset(true);
  const ExecutionPlan raw_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(raw_plan.ok());
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_once(raw_plan);

  RunSession session = run_async(plan);
  FirstCallbackRelease release;
  ASSERT_TRUE(test::RuntimeParamProbeState::wait_until_callback_enters(1U));

  constexpr std::size_t kStopperCount = 4U;
  std::barrier start_requests{static_cast<std::ptrdiff_t>(kStopperCount + 1U)};
  std::barrier finish_requests{static_cast<std::ptrdiff_t>(kStopperCount + 1U)};
  std::array<std::thread, kStopperCount> stoppers;
  for (auto& stopper : stoppers) {
    stopper = std::thread{[&] {
      start_requests.arrive_and_wait();
      session.request_stop();
      finish_requests.arrive_and_wait();
    }};
  }

  // Keep the callback in flight until every concurrent caller completes one bounded stop request.
  // An unbounded request loop can starve worker terminalization on an unfair mutex.
  start_requests.arrive_and_wait();
  finish_requests.arrive_and_wait();
  release.release();
  for (auto& stopper : stoppers) {
    stopper.join();
  }
  session.wait();

  // Keep a post-completion request idempotent while GraphExec retains the coordinator attachment.
  session.request_stop();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
}

}  // namespace
}  // namespace holoscan
