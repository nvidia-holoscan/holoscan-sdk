/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <ratio>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/parameter_control.hpp>
#include <holoscan/logger/logger.hpp>

#include "../common/allocation_counter.hpp"
#include "../core/operator_test_fixture.hpp"
#include "core/operator_runtime_access.hpp"
#include "runtime/lifecycle_control_state.hpp"
#include "runtime/parameter_catalog_projection.hpp"
#include "runtime/parameter_control_state.hpp"
#include "runtime/parameter_update_coordinator.hpp"
#include "runtime_test_graph.hpp"

#include "parameter_update_test_support.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;
using namespace test::parameter_update;

// Measures repeated params<T>() access overhead within a single compute() call.
// Verifies that the header address and generation are stable and that zero heap
// allocations occur. Used by RepeatedTypedReadsAreAllocationFreeAndUseOnePinnedHeader.
class ParameterAccessorCostProbe final : public Operator<test::RuntimeProbeParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const test::ScopedAllocationCounter allocations;
    const auto* const first = &context.params<test::RuntimeProbeParams>();
    const ParamGeneration first_generation = context.params_generation();
    const auto* const second = &context.params<test::RuntimeProbeParams>();
    allocation_count = allocations.count();
    stable_address = first == second;
    stable_generation = first_generation == context.params_generation();
    observed_value = second->value;
    return {};
  }

  std::size_t allocation_count{std::numeric_limits<std::size_t>::max()};
  bool stable_address{};
  bool stable_generation{};
  std::int32_t observed_value{};
};

// Bundles an operator and coordinator for tests that bypass RunSession and drive
// ParameterUpdateCoordinator directly with a RuntimeParamProbeOperator.
struct DirectParameterCoordinator {
  std::unique_ptr<test::RuntimeParamProbeOperator> op;
  std::unique_ptr<detail::ParameterUpdateCoordinator> coordinator;
};

// Same pattern as DirectParameterCoordinator but using EventPolicyParamOperator
// to cover field-visibility and per-origin access-policy validation paths.
struct DirectEventPolicyCoordinator {
  std::unique_ptr<test::EventPolicyParamOperator> op;
  std::unique_ptr<detail::ParameterUpdateCoordinator> coordinator;
};

// Same pattern as DirectParameterCoordinator but using LifecycleParamProbeOperator
// to exercise the lifecycle-restart parameter update path.
struct DirectLifecycleParameterCoordinator {
  std::unique_ptr<test::LifecycleParamProbeOperator> op;
  std::unique_ptr<detail::ParameterUpdateCoordinator> coordinator;
};

// Minimal single-operator subgraph used as an embedded child in snapshot scope tests.
// Provides the "embedded/parameterized" operator path verified by
// GraphScopeSelectsDescendantOperatorsWithoutWidening.
class ParameterScopeGraph final : public Graph {
 protected:
  void compose() override { (void)op<test::RuntimeParamProbeOperator>("parameterized"); }
};

// Assembles a ParameterRuntimeOperatorBinding from the plan and operator with a synthetic
// run_generation. Shared entry point for all make_direct_*_coordinator helpers.
template <typename OperatorType>
[[nodiscard]] detail::ParameterRuntimeOperatorBinding make_direct_parameter_binding(
    const ExecutionPlan& plan, const OperatorType& op, RunGenerationId run_generation = 101U,
    OperatorInstanceId operator_id = 1U) {
  const auto operator_plan =
      std::find_if(plan.operator_parameters().begin(),
                   plan.operator_parameters().end(),
                   [operator_id](const auto& record) { return record.owner.id == operator_id; });
  if (operator_plan == plan.operator_parameters().end()) {
    throw std::runtime_error{"runtime parameter test plan is incomplete"};
  }
  const auto field_begin =
      std::find_if(plan.parameter_fields().begin(),
                   plan.parameter_fields().end(),
                   [operator_id](const auto& field) { return field.owner.id == operator_id; });
  const auto field_end =
      std::find_if(field_begin, plan.parameter_fields().end(), [operator_id](const auto& field) {
        return field.owner.id != operator_id;
      });
  return detail::ParameterRuntimeOperatorBinding{
      .run_bound_owner =
          OperatorRef{
              .id = operator_plan->owner.id,
              .plan_generation = operator_plan->owner.plan_generation,
              .run_generation = run_generation,
          },
      .plan_digest = plan.digest(),
      .plan = &*operator_plan,
      .fields =
          std::span<const ParameterFieldPlanRecord>{
              field_begin,
              static_cast<std::size_t>(field_end - field_begin),
          },
      .type = &detail::OperatorRuntimeAccess::parameter_type(op),
      .schema = detail::OperatorRuntimeAccess::parameter_schema(op),
      .initial_value = detail::OperatorRuntimeAccess::parameter_address(op),
  };
}

// Constructs a DirectParameterCoordinator from the plan; throws if the plan lacks
// runtime capacity. Covers tests that need a live coordinator without a RunSession.
[[nodiscard]] DirectParameterCoordinator make_direct_parameter_coordinator(
    const ExecutionPlan& plan, RunGenerationId run_generation = 101U) {
  if (!plan.parameter_runtime_capacity()) {
    throw std::runtime_error{"runtime parameter test plan is incomplete"};
  }
  auto op = std::make_unique<test::RuntimeParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op, run_generation);
  auto coordinator = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U},
      checked_value(plan.parameter_runtime_capacity()));
  if (!coordinator) {
    throw std::runtime_error{"runtime parameter coordinator construction failed"};
  }
  return DirectParameterCoordinator{
      .op = std::move(op),
      .coordinator = checked_value(std::move(coordinator)),
  };
}

// Constructs a DirectEventPolicyCoordinator with event_capacity=2 to test cursor
// overwrite and policy-filtered event visibility paths.
[[nodiscard]] DirectEventPolicyCoordinator make_direct_event_policy_coordinator(
    const ExecutionPlan& plan) {
  if (!plan.parameter_runtime_capacity()) {
    throw std::runtime_error{"event-policy parameter test plan is incomplete"};
  }
  auto op = std::make_unique<test::EventPolicyParamOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto capacity = checked_value(plan.parameter_runtime_capacity());
  capacity.event_capacity = 2U;
  auto coordinator = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  if (!coordinator) {
    throw std::runtime_error{"event-policy parameter coordinator construction failed"};
  }
  return DirectEventPolicyCoordinator{
      .op = std::move(op),
      .coordinator = checked_value(std::move(coordinator)),
  };
}

// Constructs a DirectLifecycleParameterCoordinator for lifecycle-restart parameter
// tests; a FakeLifecycleSubmitter must be bound before submitting updates.
[[nodiscard]] DirectLifecycleParameterCoordinator make_direct_lifecycle_parameter_coordinator(
    const ExecutionPlan& plan) {
  if (!plan.parameter_runtime_capacity()) {
    throw std::runtime_error{"lifecycle parameter test plan is incomplete"};
  }
  auto op = std::make_unique<test::LifecycleParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto coordinator = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U},
      checked_value(plan.parameter_runtime_capacity()));
  if (!coordinator) {
    throw std::runtime_error{"lifecycle parameter coordinator construction failed"};
  }
  return DirectLifecycleParameterCoordinator{
      .op = std::move(op),
      .coordinator = checked_value(std::move(coordinator)),
  };
}

// Test double for the lifecycle restart submitter. Records the last submit request,
// returns a configurable outcome (including rejection and coalescing scenarios), and
// counts cancel calls. Used by all direct lifecycle coordinator tests.
struct FakeLifecycleSubmitter {
  detail::ParameterLifecycleSubmitOutcome outcome{};
  Error error{ErrorCode::kResourceExhausted};
  detail::ParameterLifecycleSubmitRequest last_request{};
  std::uint32_t submit_calls{};
  std::uint32_t cancel_calls{};
  bool reject{};

  [[nodiscard]] static expected<detail::ParameterLifecycleSubmitOutcome, Error> submit(
      void* state, const detail::ParameterLifecycleSubmitRequest& request) noexcept {
    auto& self = *static_cast<FakeLifecycleSubmitter*>(state);
    self.last_request = request;
    ++self.submit_calls;
    if (self.reject) {
      return make_unexpected(self.error);
    }
    self.outcome.receipt.target = request.owner;
    return self.outcome;
  }

  static void cancel(void* state, const detail::ParameterLifecycleSubmitOutcome&) noexcept {
    ++static_cast<FakeLifecycleSubmitter*>(state)->cancel_calls;
  }

  [[nodiscard]] detail::ParameterLifecycleSubmitBinding binding() noexcept {
    return detail::ParameterLifecycleSubmitBinding{
        .state = this,
        .submit = &submit,
        .cancel_accepted = &cancel,
    };
  }
};

// Synthetic ParamControlOperations provider whose begin_detach hook blocks until
// allow_hook_return is set, allowing tests to synchronize close-vs-detach races.
// Used by DetachWaitsForOutOfLockCloseHookBeforeCoordinatorRetirement and
// CursorMoveAndClosedLinkCleanupReclaimCoordinatorSlots.
struct BlockingDetachProbe {
  std::atomic_bool hook_entered{};
  std::atomic_bool allow_hook_return{};
  std::atomic_bool coordinator_retired{};
  std::atomic_bool hook_observed_retirement{};
  std::atomic<std::uint32_t> next_cursor{};
  std::atomic<std::uint32_t> active_cursors{};
  std::atomic<std::uint32_t> unregistered_cursors{};
  std::uint32_t cursor_capacity{};

  [[nodiscard]] static const ParameterCatalogView& catalog(void*) noexcept {
    static const ParameterCatalogView kEmpty;
    return kEmpty;
  }

  [[nodiscard]] static std::uint32_t max_event_cursors(void* state) noexcept {
    return static_cast<BlockingDetachProbe*>(state)->cursor_capacity;
  }

  [[nodiscard]] static expected<ParamReadResult, Error> read(void*, const ParamAccessPolicy&,
                                                             ParamAddress, ParamReadView) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }

  [[nodiscard]] static ParamBatchUpdateResult submit(void*, const ParamAccessPolicy&,
                                                     std::span<const ParamPatch>,
                                                     ParamUpdateOptions) {
    return ParamBatchUpdateResult{.status = ParamUpdateStatus::kClosed};
  }

  [[nodiscard]] static expected<ParamUpdateResult, Error> await(void*, const ParamAccessPolicy&,
                                                                ParamUpdateSequence,
                                                                std::chrono::nanoseconds) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }

  [[nodiscard]] static expected<ParamSnapshot, Error> capture(void*, const ParamAccessPolicy&,
                                                              ParamCatalogDigest,
                                                              ParamCaptureOptions) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }

  [[nodiscard]] static expected<std::uint32_t, Error> register_event_cursor(
      void* state, const ParamAccessPolicy&) noexcept {
    auto& self = *static_cast<BlockingDetachProbe*>(state);
    self.active_cursors.fetch_add(1U, std::memory_order_acq_rel);
    return self.next_cursor.fetch_add(1U, std::memory_order_acq_rel);
  }

  static void unregister_event_cursor(void* state, std::uint32_t) noexcept {
    auto& self = *static_cast<BlockingDetachProbe*>(state);
    self.active_cursors.fetch_sub(1U, std::memory_order_acq_rel);
    self.unregistered_cursors.fetch_add(1U, std::memory_order_acq_rel);
  }

  [[nodiscard]] static expected<std::size_t, Error> drain_events(void*, std::uint32_t,
                                                                 std::span<ParamEvent>) noexcept {
    return make_unexpected(Error{ErrorCode::kClosed});
  }

  static void begin_detach(void* state) noexcept {
    auto& self = *static_cast<BlockingDetachProbe*>(state);
    self.hook_entered.store(true, std::memory_order_release);
    self.hook_entered.notify_all();
    self.allow_hook_return.wait(false, std::memory_order_acquire);
    self.hook_observed_retirement.store(self.coordinator_retired.load(std::memory_order_acquire),
                                        std::memory_order_release);
  }

  [[nodiscard]] static const detail::ParamControlOperations& operations() noexcept {
    static const detail::ParamControlOperations kOperations{
        .catalog = &catalog,
        .max_event_cursors = &max_event_cursors,
        .read = &read,
        .submit = &submit,
        .await = &await,
        .capture = &capture,
        .register_event_cursor = &register_event_cursor,
        .unregister_event_cursor = &unregister_event_cursor,
        .drain_events = &drain_events,
        .begin_detach = &begin_detach,
    };
    return kOperations;
  }
};

// Constructs a kQueued lifecycle submit outcome with the given transaction id and
// receipt. The most common FakeLifecycleSubmitter outcome in coordinator unit tests.
[[nodiscard]] detail::ParameterLifecycleSubmitOutcome queued_lifecycle_outcome(
    LifecycleTransactionId transaction, std::uint64_t receipt) noexcept {
  return detail::ParameterLifecycleSubmitOutcome{
      .receipt = LifecycleRestartReceipt{.receipt_id = receipt},
      .state = LifecycleTransactionState::kQueued,
      .transaction = transaction,
  };
}

// Builds a ParameterDerivedTermPlanRecord for a duration field with a [1 ms, 10 ms]
// range and 3 ms initial value. Used in derived-term coordinator tests that verify
// range rejection and atomic publication for kLatestWithinMaxAge / kContractDeadline.
[[nodiscard]] ParameterDerivedTermPlanRecord runtime_duration_term(
    const OperatorParameterPlanRecord& owner, const ParameterFieldPlanRecord& field,
    ParameterDerivedTermKind kind,
    ParameterDerivedTermUpdateDisposition disposition =
        ParameterDerivedTermUpdateDisposition::kRuntimeAtomic) {
  const bool per_input = kind == ParameterDerivedTermKind::kLatestWithinMaxAge;
  return ParameterDerivedTermPlanRecord{
      .term_id =
          per_input ? "operator:1/input:1/latest_within.max_age" : "operator:1/contract.deadline",
      .owner = field.owner,
      .kind = kind,
      .input = per_input ? PortId{1U} : PortId{},
      .field = field.field,
      .field_full_id = field.full_id,
      .field_path = field.canonical_name,
      .schema_id = owner.schema_id,
      .schema_version = owner.schema_version,
      .mutability = field.mutability,
      .update_disposition = disposition,
      .initial_value_ns = 3'000'000,
      .minimum_value_ns = 1'000'000,
      .maximum_value_ns = 10'000'000,
  };
}

// REQ: PARAM-017
// REQ: PARAM-019
TEST(ParameterUpdateCoordinatorTest, ReconstructionRejectsInconsistentComputeIngressBounds) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  ASSERT_TRUE(plan.parameter_runtime_capacity());
  const auto capacity = checked_value(plan.parameter_runtime_capacity());
  // OperatorRuntimeAccess exposes the mutable aggregate address required by reconstruction.
  // NOLINTNEXTLINE(misc-const-correctness)
  test::RuntimeParamProbeOperator op;
  const auto binding = make_direct_parameter_binding(plan, op);
  const auto create_with = [&](ParameterRuntimeCapacityPlanRecord capacity) {
    return detail::ParameterUpdateCoordinator::create(
        std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  };

  auto missing_ingress = capacity;
  missing_ingress.compute_ingress_capacity = 0U;
  const auto missing = create_with(missing_ingress);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, ErrorCode::kIncompatible);

  auto inconsistent_patch_bound = capacity;
  ++inconsistent_patch_bound.max_compute_patches_per_request;
  const auto inconsistent = create_with(inconsistent_patch_bound);
  ASSERT_FALSE(inconsistent);
  EXPECT_EQ(inconsistent.error().code, ErrorCode::kIncompatible);

  auto overflow_like_ingress = capacity;
  overflow_like_ingress.compute_ingress_capacity = std::numeric_limits<std::uint32_t>::max();
  const auto overflow_like = create_with(overflow_like_ingress);
  ASSERT_FALSE(overflow_like);
  EXPECT_EQ(overflow_like.error().code, ErrorCode::kIncompatible);
}

// REQ: PARAM-017
TEST(ParameterUpdateCoordinatorTest,
     ArenaAllocationFailureReturnsResourceExhaustedBeforeCoordinatorExposure) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  ASSERT_TRUE(plan.parameter_runtime_capacity());
  const auto capacity = checked_value(plan.parameter_runtime_capacity());
  // OperatorRuntimeAccess exposes the mutable aggregate address required by reconstruction.
  // NOLINTNEXTLINE(misc-const-correctness)
  test::RuntimeParamProbeOperator op;
  const auto binding = make_direct_parameter_binding(plan, op);

  expected<std::unique_ptr<detail::ParameterUpdateCoordinator>, Error> failed;
  {
    // Generation storage is the coordinator's first explicitly aligned allocation, so this
    // targets the payload arena without coupling the test to preceding bookkeeping allocations.
    const test::ScopedAlignedAllocationFailure fail_arena;
    failed = detail::ParameterUpdateCoordinator::create(
        std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  }
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code, ErrorCode::kResourceExhausted);

  // Construction failure exposes no coordinator or generation. A fresh construction starts from
  // the original configured value at generation zero rather than retaining partial failed state.
  auto recovered = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(recovered);
  auto recovered_coordinator = checked_value(std::move(recovered));
  auto published = recovered_coordinator->pin_published(binding.run_bound_owner);
  ASSERT_TRUE(published);
  ASSERT_NE(published->header(), nullptr);
  EXPECT_EQ(published->header()->generation, 0U);
  ASSERT_NE(published->header()->value, nullptr);
  EXPECT_NE(published->header()->value, binding.initial_value);
  EXPECT_EQ(static_cast<const test::RuntimeProbeParams*>(published->header()->value)->value,
            static_cast<const test::RuntimeProbeParams*>(binding.initial_value)->value);
  recovered_coordinator->release(*published);
}

// REQ: PARAM-017
// REQ: PARAM-022
TEST(ParameterUpdateCoordinatorTest,
     GenerationZeroOnlyPlanMaterializesWithoutUpdateOrObserverStorage) {
  // Graph: [parameterized]  (single operator, no edges, compiled without update capacity)
  Graph graph{"generation-zero-only-parameter-test"};
  const auto parameterized = graph.op<test::RuntimeParamProbeOperator>("parameterized");
  graph.partition("runtime-spin-worker").add(parameterized);
  const ClockKey clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-spin-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.parameter_runtime_capacity());
  const auto capacity = checked_value(plan.parameter_runtime_capacity());
  EXPECT_FALSE(capacity.runtime_updates_enabled);
  EXPECT_FALSE(capacity.lifecycle_updates_enabled);
  EXPECT_EQ(capacity.enabled_operator_count, 0U);
  EXPECT_EQ(capacity.max_pending_updates_per_operator, 0U);
  EXPECT_EQ(capacity.generation_slots_per_operator, 0U);
  EXPECT_EQ(capacity.compute_ingress_capacity, 0U);
  EXPECT_EQ(capacity.max_compute_patches_per_request, 0U);
  EXPECT_EQ(capacity.audit_capacity, 0U);
  EXPECT_EQ(capacity.event_capacity, 0U);
  EXPECT_EQ(capacity.max_event_cursors, 0U);
  EXPECT_EQ(capacity.parameter_arena_bytes, 0U);

  // Key setup: reconstruct an inactive coordinator from one mutable Operator-owned aggregate,
  // then change that source after reconstruction.
  // NOLINTNEXTLINE(misc-const-correctness)
  test::RuntimeParamProbeOperator op;
  const auto binding = make_direct_parameter_binding(plan, op);
  auto coordinator = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(coordinator) << static_cast<int>(coordinator.error().code);
  auto* const source_value = const_cast<test::RuntimeProbeParams*>(
      static_cast<const test::RuntimeProbeParams*>(binding.initial_value));
  source_value->value = 19;

  const OperatorRef owner = binding.run_bound_owner;
  auto direct_coordinator = checked_value(std::move(coordinator));
  auto pin = direct_coordinator->pin_published(owner);
  ASSERT_TRUE(pin);
  ASSERT_NE(pin->header(), nullptr);
  // Key assertion: generation zero is immutable and independent of the reconstructed Operator,
  // without being counted as an update-generation arena or observer allocation.
  EXPECT_EQ(pin->header()->generation, 0U);
  EXPECT_NE(pin->header()->value, binding.initial_value);
  EXPECT_EQ(static_cast<const test::RuntimeProbeParams*>(pin->header()->value)->value, 7);
  EXPECT_EQ(direct_coordinator->max_event_cursors(), 0U);
  const auto cursor = direct_coordinator->register_event_cursor(full_local_parameter_policy());
  ASSERT_FALSE(cursor);
  EXPECT_EQ(cursor.error().code, ErrorCode::kResourceExhausted);
}

// REQ: PARAM-022
TEST(ParameterCatalogTest, CompatibilityDigestBindsExactPlanAndRunIdentity) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  auto first = make_direct_parameter_coordinator(plan, 101U);
  auto same = make_direct_parameter_coordinator(plan, 101U);
  auto second = make_direct_parameter_coordinator(plan, 202U);
  const ExecutionPlan rebuilt_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(rebuilt_plan.ok());
  auto rebuilt = make_direct_parameter_coordinator(rebuilt_plan, 303U);

  ASSERT_EQ(first.coordinator->catalog().operators().size(), 1U);
  ASSERT_EQ(second.coordinator->catalog().operators().size(), 1U);
  EXPECT_NE(first.coordinator->catalog().operators().front().operator_instance.run_generation,
            second.coordinator->catalog().operators().front().operator_instance.run_generation);
  EXPECT_EQ(first.coordinator->catalog().compatibility_digest(),
            same.coordinator->catalog().compatibility_digest());
  EXPECT_NE(first.coordinator->catalog().compatibility_digest(),
            second.coordinator->catalog().compatibility_digest());
  EXPECT_NE(first.coordinator->catalog().compatibility_digest(),
            rebuilt.coordinator->catalog().compatibility_digest());
}

// REQ: PARAM-018
TEST(ParameterUpdateValue, FixedScalarVocabularyIsExactAndAllocationFree) {
  static_assert(std::is_standard_layout_v<ParamValue>);
  static_assert(std::is_trivially_copyable_v<ParamValue>);
  static_assert(sizeof(ParamValue) == sizeof(ParamScalarValue));
  constexpr ParamStorageTraitsSummary duration_storage =
      ParamStorageTraits<std::chrono::milliseconds>::summary();
  static_assert(!duration_storage.owns_dynamic_storage);
  static_assert(duration_storage.fixed_object_extent);
  static_assert(duration_storage.supports_bounded_materialization);
  static_assert(duration_storage.inline_bytes == sizeof(std::chrono::milliseconds));

  constexpr auto boolean = ParamValue::from(true);
  constexpr auto signed_value = ParamValue::from(std::int32_t{-42});
  constexpr auto unsigned_value = ParamValue::from(std::uint64_t{42});
  constexpr auto floating_value = ParamValue::from(2.5F);
  constexpr auto duration_value = ParamValue::from(3ms);

  const auto decoded_boolean = boolean.as<bool>();
  ASSERT_TRUE(decoded_boolean);
  EXPECT_TRUE(*decoded_boolean);
  const auto decoded_signed = signed_value.as<std::int32_t>();
  ASSERT_TRUE(decoded_signed);
  EXPECT_EQ(*decoded_signed, -42);
  const auto decoded_unsigned = unsigned_value.as<std::uint64_t>();
  ASSERT_TRUE(decoded_unsigned);
  EXPECT_EQ(*decoded_unsigned, 42U);
  const auto decoded_floating = floating_value.as<float>();
  ASSERT_TRUE(decoded_floating);
  EXPECT_EQ(*decoded_floating, 2.5F);
  const auto decoded_duration = duration_value.as<std::chrono::microseconds>();
  ASSERT_TRUE(decoded_duration);
  EXPECT_EQ(*decoded_duration, 3000us);
  const auto decoded_negative_duration = ParamValue::from(-3ms).as<std::chrono::microseconds>();
  ASSERT_TRUE(decoded_negative_duration);
  EXPECT_EQ(*decoded_negative_duration, -3000us);

  const auto wrong_family = signed_value.as<std::uint32_t>();
  ASSERT_FALSE(wrong_family);
  EXPECT_EQ(wrong_family.error().code, ErrorCode::kIncompatible);
  const auto wrong_width = signed_value.as<std::int64_t>();
  ASSERT_FALSE(wrong_width);
  EXPECT_EQ(wrong_width.error().code, ErrorCode::kIncompatible);

  constexpr auto subnanosecond =
      ParamValue::from(std::chrono::duration<std::int64_t, std::pico>{1});
  EXPECT_EQ(subnanosecond.kind(), ParamScalarKind::kNone);
  constexpr auto duration_overflow = ParamValue::from(
      std::chrono::duration<std::uint64_t>{std::numeric_limits<std::uint64_t>::max()});
  EXPECT_EQ(duration_overflow.kind(), ParamScalarKind::kNone);
  constexpr auto negative_duration_overflow = ParamValue::from(
      std::chrono::duration<std::int64_t>{std::numeric_limits<std::int64_t>::min()});
  EXPECT_EQ(negative_duration_overflow.kind(), ParamScalarKind::kNone);
  using LargePeriod =
      std::ratio<std::numeric_limits<std::intmax_t>::max(), std::intmax_t{999999937}>;
  constexpr auto type_level_ratio_overflow_is_avoided =
      ParamValue::from(std::chrono::duration<std::int64_t, LargePeriod>{1});
  EXPECT_EQ(type_level_ratio_overflow_is_avoided.kind(), ParamScalarKind::kNone);
  const auto nonintegral_target = ParamValue::from(1500ns).as<std::chrono::microseconds>();
  ASSERT_FALSE(nonintegral_target);
  EXPECT_EQ(nonintegral_target.error().code, ErrorCode::kIncompatible);
  const auto target_overflow =
      ParamValue::from(128ms).as<std::chrono::duration<std::int8_t, std::milli>>();
  ASSERT_FALSE(target_overflow);
  EXPECT_EQ(target_overflow.error().code, ErrorCode::kIncompatible);
  const auto negative_target_overflow =
      ParamValue::from(-129ms).as<std::chrono::duration<std::int8_t, std::milli>>();
  ASSERT_FALSE(negative_target_overflow);
  EXPECT_EQ(negative_target_overflow.error().code, ErrorCode::kIncompatible);
}

// REQ: PARAM-018
TEST(ParameterUpdateValue, OwningPatchValuePreservesNestedAggregateStorage) {
  // Key setup: Build and copy a cold-path patch value containing scalar, string, sequence, and
  // nested-object representations.
  const auto label = ParamPatchValue::from_string("alpha");
  ASSERT_TRUE(label);
  std::vector<ParamPatchValue> elements;
  elements.emplace_back(ParamValue::from(std::int32_t{7}));
  elements.push_back(*label);
  const auto sequence = ParamPatchValue::from_sequence(std::move(elements));
  ASSERT_TRUE(sequence);
  std::vector<ParamPatchAssignment> assignments;
  assignments.push_back(ParamPatchAssignment{.name = "values", .value = *sequence});
  const auto object = ParamPatchValue::from_object(std::move(assignments));
  ASSERT_TRUE(object);
  // Copy construction is the behavior under test; a reference would not exercise owned storage.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const ParamPatchValue copied = *object;

  // Key assertion: Every owning view remains valid through a copied top-level carrier.
  ASSERT_EQ(copied.kind(), ParamPatchValueKind::kObject);
  ASSERT_EQ(copied.object().size(), 1U);
  EXPECT_EQ(copied.object().front().name, "values");
  ASSERT_EQ(copied.object().front().value.sequence().size(), 2U);
  const auto scalar = copied.object().front().value.sequence().front().scalar();
  std::int32_t decoded{};
  ASSERT_TRUE(scalar.try_as(decoded));
  EXPECT_EQ(decoded, 7);
  EXPECT_EQ(copied.object().front().value.sequence().back().string(), "alpha");
  EXPECT_EQ(ParamPatchValue{}.kind(), ParamPatchValueKind::kUnset);
  EXPECT_EQ(ParamPatchValue::none().kind(), ParamPatchValueKind::kNone);
}

// REQ: PARAM-017
TEST(ParameterGenerationAccessTest, RepeatedTypedReadsAreAllocationFreeAndUseOnePinnedHeader) {
  ParameterAccessorCostProbe op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_initial_params(test::RuntimeProbeParams{.value = 42}));
  ASSERT_TRUE(fixture.invoke_once());
  EXPECT_EQ(op.allocation_count, 0U);
  EXPECT_TRUE(op.stable_address);
  EXPECT_TRUE(op.stable_generation);
  EXPECT_EQ(op.observed_value, 42);
}

// REQ: PARAM-017
// REQ: PARAM-018
TEST(ParameterUpdateCoordinatorTest, PublishesDerivedDurationTermWithSameGenerationAsAggregate) {
  const ExecutionPlan plan = test::make_runtime_duration_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.parameter_runtime_capacity());
  const auto capacity = checked_value(plan.parameter_runtime_capacity());
  ASSERT_EQ(plan.operator_parameters().size(), 1U);
  ASSERT_EQ(plan.parameter_fields().size(), 1U);
  EXPECT_TRUE(plan.operator_parameters().front().runtime_updates_enabled);
  EXPECT_EQ(plan.parameter_fields().front().logical_type, ParamLogicalType::kDuration);
  EXPECT_EQ(plan.parameter_fields().front().mutability, ParamMutability::kRuntime);
  EXPECT_EQ(plan.parameter_fields().front().update_disposition,
            ParameterFieldUpdateDisposition::kAdmitted);
  auto op = std::make_unique<test::RuntimeDurationParamProbeOperator>();
  auto binding = make_direct_parameter_binding(plan, *op);
  const std::array terms{runtime_duration_term(plan.operator_parameters().front(),
                                               plan.parameter_fields().front(),
                                               ParameterDerivedTermKind::kLatestWithinMaxAge)};
  binding.derived_terms = terms;
  auto created = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(created) << static_cast<std::uint32_t>(created.error().code);
  auto coordinator = checked_value(std::move(created));
  const OperatorRef owner = binding.run_bound_owner;

  auto generation_zero = coordinator->pin_published(owner);
  ASSERT_TRUE(generation_zero);
  auto generation_zero_pin = checked_value(std::move(generation_zero));
  EXPECT_EQ(generation_zero_pin.header()->generation, 0U);
  EXPECT_EQ(
      generation_zero_pin.duration_term(ParameterDerivedTermKind::kLatestWithinMaxAge, PortId{1U}),
      3'000'000);
  coordinator->release(generation_zero_pin);

  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner,
                                 .field = plan.parameter_fields().front().field},
          .value = ParamValue::from(5ms),
      },
  };
  const ParamBatchUpdateResult submitted = coordinator->submit(patch);
  ASSERT_TRUE(submitted.committed);
  EXPECT_EQ(submitted.target_generation, 1U);

  auto before_publication = coordinator->pin_published(owner);
  ASSERT_TRUE(before_publication);
  auto before_publication_pin = checked_value(std::move(before_publication));
  EXPECT_EQ(before_publication_pin.header()->generation, 0U);
  EXPECT_EQ(before_publication_pin.duration_term(ParameterDerivedTermKind::kLatestWithinMaxAge,
                                                 PortId{1U}),
            3'000'000);
  coordinator->release(before_publication_pin);

  ASSERT_TRUE(
      coordinator->publish_next_at_callback_boundary(owner, coordinator->publication_frontier())
          .has_value());
  auto published = coordinator->pin_published(owner);
  ASSERT_TRUE(published);
  auto published_pin = checked_value(std::move(published));
  EXPECT_EQ(published_pin.header()->generation, 1U);
  EXPECT_EQ(static_cast<const test::RuntimeDurationParamProbeParams*>(published_pin.header()->value)
                ->deadline,
            5ms);
  EXPECT_EQ(published_pin.duration_term(ParameterDerivedTermKind::kLatestWithinMaxAge, PortId{1U}),
            5'000'000);
  coordinator->release(published_pin);
}

// REQ: PARAM-017
// REQ: PARAM-018
TEST(ParameterUpdateCoordinatorTest, RejectsNonAtomicDerivedTermBeforeGenerationAcceptance) {
  const ExecutionPlan plan = test::make_runtime_duration_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.parameter_runtime_capacity());
  const auto capacity = checked_value(plan.parameter_runtime_capacity());
  auto op = std::make_unique<test::RuntimeDurationParamProbeOperator>();
  auto binding = make_direct_parameter_binding(plan, *op);
  const std::array terms{
      runtime_duration_term(plan.operator_parameters().front(),
                            plan.parameter_fields().front(),
                            ParameterDerivedTermKind::kContractDeadline,
                            ParameterDerivedTermUpdateDisposition::kRecompileRequired)};
  binding.derived_terms = terms;
  auto created = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(created) << static_cast<std::uint32_t>(created.error().code);
  auto coordinator = checked_value(std::move(created));
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = binding.run_bound_owner,
                                 .field = plan.parameter_fields().front().field},
          .value = ParamValue::from(5ms),
      },
  };
  const ParamBatchUpdateResult rejected = coordinator->submit(patch);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.status, ParamUpdateStatus::kRequiresRecompile);
  EXPECT_EQ(rejected.update_sequence, 0U);
  auto published = coordinator->pin_published(binding.run_bound_owner);
  ASSERT_TRUE(published);
  auto published_pin = checked_value(std::move(published));
  EXPECT_EQ(published_pin.header()->generation, 0U);
  EXPECT_EQ(published_pin.duration_term(ParameterDerivedTermKind::kContractDeadline), 3'000'000);
  coordinator->release(published_pin);
  const auto latest =
      coordinator->read(ParamAddress{.operator_instance = binding.run_bound_owner,
                                     .field = plan.parameter_fields().front().field},
                        ParamReadView::kLatestAccepted);
  ASSERT_TRUE(latest);
  const auto latest_record = checked_value(latest);
  EXPECT_EQ(latest_record.generation, 0U);
  EXPECT_EQ(latest_record.update_sequence, 0U);
  const auto latest_value = latest_record.value.as<std::chrono::milliseconds>();
  ASSERT_TRUE(latest_value);
  EXPECT_EQ(checked_value(latest_value), 3ms);
}

// REQ: PARAM-018
TEST(ParameterUpdateCoordinatorTest, RejectsDerivedTermRangeBeforeGenerationAcceptance) {
  const ExecutionPlan plan = test::make_runtime_duration_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.parameter_runtime_capacity());
  const auto capacity = checked_value(plan.parameter_runtime_capacity());
  auto op = std::make_unique<test::RuntimeDurationParamProbeOperator>();
  auto binding = make_direct_parameter_binding(plan, *op);
  auto term = runtime_duration_term(plan.operator_parameters().front(),
                                    plan.parameter_fields().front(),
                                    ParameterDerivedTermKind::kLatestWithinMaxAge);
  term.maximum_value_ns = 4'000'000;
  const std::array terms{term};
  binding.derived_terms = terms;
  auto created = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(created) << static_cast<std::uint32_t>(created.error().code);
  auto coordinator = checked_value(std::move(created));
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = binding.run_bound_owner,
                                 .field = plan.parameter_fields().front().field},
          .value = ParamValue::from(5ms),
      },
  };
  const ParamBatchUpdateResult rejected = coordinator->submit(patch);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.status, ParamUpdateStatus::kOutOfBounds);
  EXPECT_EQ(rejected.update_sequence, 0U);
  auto published = coordinator->pin_published(binding.run_bound_owner);
  ASSERT_TRUE(published);
  auto published_pin = checked_value(std::move(published));
  EXPECT_EQ(published_pin.header()->generation, 0U);
  EXPECT_EQ(published_pin.duration_term(ParameterDerivedTermKind::kLatestWithinMaxAge, PortId{1U}),
            3'000'000);
  coordinator->release(published_pin);
  const auto latest =
      coordinator->read(ParamAddress{.operator_instance = binding.run_bound_owner,
                                     .field = plan.parameter_fields().front().field},
                        ParamReadView::kLatestAccepted);
  ASSERT_TRUE(latest);
  const auto latest_record = checked_value(latest);
  EXPECT_EQ(latest_record.generation, 0U);
  EXPECT_EQ(latest_record.update_sequence, 0U);
}

TEST(ParameterUpdateValue, NoChangeEqualityUsesScalarSemantics) {
  const ParamScalarValue positive_zero = ParamScalarValue::from(0.0F);
  const ParamScalarValue negative_zero = ParamScalarValue::from(-0.0F);
  EXPECT_TRUE(detail::ParameterUpdateCoordinatorTestAccess::scalar_semantically_equal(
      positive_zero, negative_zero));

  const ParamScalarValue nan = ParamScalarValue::from(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(detail::ParameterUpdateCoordinatorTestAccess::scalar_semantically_equal(nan, nan));
  ParamScalarValue distinct_nan = nan;
  distinct_nan.raw_bits ^= 1U;
  EXPECT_FALSE(
      detail::ParameterUpdateCoordinatorTestAccess::scalar_semantically_equal(nan, distinct_nan));

  ParamScalarValue malformed_float = positive_zero;
  malformed_float.raw_bits = std::uint64_t{1} << 40U;
  EXPECT_FALSE(detail::ParameterUpdateCoordinatorTestAccess::scalar_semantically_equal(
      malformed_float, positive_zero));
}

// REQ: PARAM-018
TEST(ParameterUpdateCoordinatorTest, IdenticalUnconstrainedNanPatchIsAcceptedWithoutChange) {
  const ExecutionPlan plan = test::make_runtime_float_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.parameter_runtime_capacity());
  const auto capacity = checked_value(plan.parameter_runtime_capacity());
  ASSERT_EQ(plan.parameter_fields().size(), 1U);
  auto op = std::make_unique<test::RuntimeFloatParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(coordinator_result);
  auto coordinator = checked_value(std::move(coordinator_result));
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = binding.run_bound_owner,
                                 .field = plan.parameter_fields().front().field},
          .value = ParamValue::from(std::numeric_limits<float>::quiet_NaN()),
      },
  };

  const ParamBatchUpdateResult result = coordinator->submit(patch);
  EXPECT_TRUE(result.committed);
  EXPECT_EQ(result.status, ParamUpdateStatus::kAcceptedNoChange);
  EXPECT_EQ(result.target_generation, 0U);
  EXPECT_EQ(detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*coordinator),
            0U);
}

// REQ: PARAM-022
TEST(ParameterPathTest, ParsesOwningDecodedSegmentsAndRejectsAmbiguousGrammar) {
  const auto parsed = OperatorPath::parse("pipeline/op%2Fleaf");
  ASSERT_TRUE(parsed);
  const auto parsed_path = checked_value(parsed);
  ASSERT_EQ(parsed_path.segments().size(), 2U);
  EXPECT_EQ(parsed_path.segments()[0], "pipeline");
  EXPECT_EQ(parsed_path.segments()[1], "op/leaf");

  const std::array<std::string, 2> decoded_segments{"pipeline", "op/leaf"};
  const auto copied = OperatorPath::from_segments(decoded_segments);
  ASSERT_TRUE(copied);
  EXPECT_EQ(checked_value(copied), parsed_path);
  const auto nested_field = ParamPath::parse("calibration/gain%2Ffine");
  ASSERT_TRUE(nested_field);
  EXPECT_EQ(checked_value(nested_field).segments()[1], "gain/fine");

  constexpr std::array<std::string_view, 11> invalid_paths{"",
                                                           "/operator",
                                                           "operator/",
                                                           "operator//field",
                                                           ".",
                                                           "..",
                                                           "operator/%2E%2E",
                                                           "operator/%",
                                                           "operator/%GG",
                                                           "operator/%00",
                                                           "op:operator"};
  for (const std::string_view invalid : invalid_paths) {
    EXPECT_FALSE(OperatorPath::parse(invalid)) << invalid;
  }
  std::string invalid_utf8{"operator/"};
  invalid_utf8.push_back(static_cast<char>(0xC0U));
  invalid_utf8.push_back(static_cast<char>(0xAFU));
  EXPECT_FALSE(OperatorPath::parse(invalid_utf8));
}

// REQ: PARAM-022
TEST(ParameterSnapshotTest, GraphScopeSelectsDescendantOperatorsWithoutWidening) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  ASSERT_TRUE(plan.parameter_runtime_capacity());
  const auto capacity = checked_value(plan.parameter_runtime_capacity());
  // OperatorRuntimeAccess exposes the mutable aggregate address required by reconstruction.
  // NOLINTNEXTLINE(misc-const-correctness)
  test::RuntimeParamProbeOperator op;
  auto binding = make_direct_parameter_binding(plan, op);
  binding.operator_path = {"embedded", "parameterized"};
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(coordinator_result);
  auto coordinator = checked_value(std::move(coordinator_result));

  const auto snapshot = coordinator->capture(
      full_local_parameter_policy(),
      ParamCaptureOptions{.consistency = ParamCaptureConsistency::kLivePerOperator});
  ASSERT_TRUE(snapshot);
  const auto snapshot_value = checked_value(snapshot);

  const auto empty_operator_scope =
      snapshot_value.slice(ParamSnapshotScope::operator_(OperatorPath{}));
  ASSERT_FALSE(empty_operator_scope);
  EXPECT_EQ(empty_operator_scope.error().code, ErrorCode::kInvalidArgument);
  const auto empty_operator_path = snapshot_value.slice(OperatorPath{});
  ASSERT_FALSE(empty_operator_path);
  EXPECT_EQ(empty_operator_path.error().code, ErrorCode::kInvalidArgument);

  // Topology for scope resolution:
  //   root {"scope-root"} --> embed("embedded") --> [parameterized]
  //   Narrowing from root to graph-scope is allowed; widening back is kPermissionDenied.
  Graph root{"scope-root"};
  const auto embedded = root.embed<ParameterScopeGraph>("embedded");
  ASSERT_TRUE(embedded);
  const auto graph_path = embedded.path();
  ASSERT_TRUE(graph_path);
  const auto graph_path_value = checked_value(graph_path);
  const auto sliced = snapshot_value.slice(ParamSnapshotScope::graph(graph_path_value));
  ASSERT_TRUE(sliced);
  const auto sliced_value = checked_value(sliced);
  EXPECT_EQ(sliced_value.catalog_digest(), snapshot_value.catalog_digest());
  EXPECT_EQ(sliced_value.plan_digest(), snapshot_value.plan_digest());
  EXPECT_EQ(sliced_value.plan_generation(), snapshot_value.plan_generation());
  EXPECT_EQ(sliced_value.run_generation(), snapshot_value.run_generation());
  EXPECT_EQ(sliced_value.scope().kind(), ParamSnapshotScopeKind::kGraph);
  ASSERT_EQ(sliced_value.operators().size(), 1U);
  ASSERT_EQ(sliced_value.operators().front().path.segments().size(), 2U);
  EXPECT_EQ(sliced_value.operators().front().path.segments()[0], "embedded");
  EXPECT_EQ(sliced_value.operators().front().path.segments()[1], "parameterized");

  const auto operator_path =
      OperatorPath::from_segments(std::span<const std::string>{binding.operator_path});
  ASSERT_TRUE(operator_path);
  const auto operator_slice = sliced_value.slice(checked_value(operator_path));
  ASSERT_TRUE(operator_slice);
  const auto operator_slice_value = checked_value(operator_slice);
  EXPECT_EQ(operator_slice_value.catalog_digest(), snapshot_value.catalog_digest());
  EXPECT_EQ(operator_slice_value.plan_digest(), snapshot_value.plan_digest());
  EXPECT_EQ(operator_slice_value.plan_generation(), snapshot_value.plan_generation());
  EXPECT_EQ(operator_slice_value.run_generation(), snapshot_value.run_generation());
  EXPECT_EQ(operator_slice_value.scope().kind(), ParamSnapshotScopeKind::kOperator);
  ASSERT_EQ(operator_slice_value.operators().size(), 1U);

  const auto graph_to_root = sliced_value.slice(ParamSnapshotScope::root());
  ASSERT_FALSE(graph_to_root);
  EXPECT_EQ(graph_to_root.error().code, ErrorCode::kPermissionDenied);
  const auto operator_to_graph =
      operator_slice_value.slice(ParamSnapshotScope::graph(graph_path_value));
  ASSERT_FALSE(operator_to_graph);
  EXPECT_EQ(operator_to_graph.error().code, ErrorCode::kPermissionDenied);

  {
    const test::ScopedAllocationFailure fail_root_copy{0U};
    const auto failed = snapshot_value.slice(ParamSnapshotScope::root());
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().code, ErrorCode::kResourceExhausted);
  }
  const ParamSnapshotScope graph_scope = ParamSnapshotScope::graph(graph_path_value);
  {
    const test::ScopedAllocationFailure fail_scope_copy{0U};
    const auto failed = snapshot_value.slice(graph_scope);
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error().code, ErrorCode::kResourceExhausted);
  }
}

// REQ: PARAM-017
// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterSnapshotTest, LivePerOperatorCaptureRetainsAtomicPublishedLineagePerOperator) {
  const ExecutionPlan plan = test::make_runtime_two_param_test_plan();
  ASSERT_TRUE(plan.ok());
  ASSERT_EQ(plan.parameter_fields().size(), 2U);
  auto first_op = std::make_unique<test::RuntimeParamProbeOperator>();
  auto second_op = std::make_unique<test::RuntimeParamProbeOperator>();
  auto first_binding = make_direct_parameter_binding(plan, *first_op, 101U, 1U);
  auto second_binding = make_direct_parameter_binding(plan, *second_op, 101U, 2U);
  first_binding.operator_path = {"a-parameterized"};
  second_binding.operator_path = {"z-parameterized"};
  const std::array bindings{first_binding, second_binding};
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      bindings, checked_value(plan.parameter_runtime_capacity()));
  ASSERT_TRUE(coordinator_result);
  auto coordinator = checked_value(std::move(coordinator_result));

  const auto field_for = [&](OperatorInstanceId id) -> const ParameterFieldPlanRecord& {
    const auto field =
        std::find_if(plan.parameter_fields().begin(),
                     plan.parameter_fields().end(),
                     [id](const auto& candidate) { return candidate.owner.id == id; });
    if (field == plan.parameter_fields().end()) {
      throw std::runtime_error{"live capture field missing"};
    }
    return *field;
  };
  const auto submit = [&](OperatorInstanceId id, std::int32_t value) {
    const auto& field = field_for(id);
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = run_owner(field), .field = field.field},
            .value = ParamValue::from(value),
        },
    };
    return coordinator->submit(patch);
  };

  const ParamBatchUpdateResult first = submit(1U, 42);
  ASSERT_TRUE(first.committed);
  ASSERT_TRUE(coordinator
                  ->publish_next_at_callback_boundary(run_owner(field_for(1U)),
                                                      coordinator->publication_frontier())
                  .has_value());
  const ParamBatchUpdateResult second = submit(2U, 52);
  ASSERT_TRUE(second.committed);

  const auto snapshot = coordinator->capture(
      full_local_parameter_policy(),
      ParamCaptureOptions{.consistency = ParamCaptureConsistency::kLivePerOperator});
  ASSERT_TRUE(snapshot);
  const auto snapshot_value = checked_value(snapshot);
  EXPECT_FALSE(snapshot_value.accepted_through().has_value());
  EXPECT_FALSE(snapshot_value.published_through().has_value());
  ASSERT_EQ(snapshot_value.operators().size(), 2U);
  const auto& first_record = snapshot_value.operators()[0];
  const auto& second_record = snapshot_value.operators()[1];
  EXPECT_EQ(first_record.generation, 1U);
  EXPECT_EQ(first_record.published_update_sequence, first.update_sequence);
  EXPECT_EQ(first_record.accepted_update_sequence, first.update_sequence);
  ASSERT_EQ(first_record.fields.size(), 1U);
  ASSERT_TRUE(first_record.fields.front().value.has_value());
  const auto first_value = checked_value(first_record.fields.front().value).as<std::int32_t>();
  ASSERT_TRUE(first_value);
  EXPECT_EQ(checked_value(first_value), 42);

  EXPECT_EQ(second_record.generation, 0U);
  EXPECT_EQ(second_record.published_update_sequence, 0U);
  EXPECT_EQ(second_record.accepted_update_sequence, second.update_sequence);
  ASSERT_EQ(second_record.fields.size(), 1U);
  ASSERT_TRUE(second_record.fields.front().value.has_value());
  const auto second_value = checked_value(second_record.fields.front().value).as<std::int32_t>();
  ASSERT_TRUE(second_value);
  EXPECT_EQ(checked_value(second_value), 7);
}

// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterSnapshotTest, SafeProjectionUsesFrozenFirstMatchPolicyBeforeValueAccess) {
  const ExecutionPlan plan = test::make_event_policy_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto direct = make_direct_event_policy_coordinator(plan);

  const auto snapshot = direct.coordinator->capture(
      full_local_parameter_policy(),
      ParamCaptureOptions{
          .consistency = ParamCaptureConsistency::kLivePerOperator,
          .unavailable_values = ParamUnavailableValuePolicy::kRecordUnavailable,
      });
  ASSERT_TRUE(snapshot);
  ASSERT_EQ(snapshot->operators().size(), 1U);
  const auto& fields = snapshot->operators().front().fields;
  ASSERT_EQ(fields.size(), 3U);

  const auto projected = [&](std::string_view name) -> const ParamSnapshotFieldRecord& {
    const auto found = std::find_if(fields.begin(), fields.end(), [&](const auto& field) {
      return field.path.segments().size() == 1U && field.path.segments().front() == name;
    });
    if (found == fields.end()) {
      throw std::runtime_error{"projected snapshot field missing"};
    }
    return *found;
  };
  const auto& public_field = projected("public_value");
  const auto& sensitive_field = projected("sensitive_value");
  const auto& external_field = projected("external_value");

  EXPECT_EQ(public_field.projection, params::ParamProjectionStatus::kIncluded);
  EXPECT_TRUE(public_field.value.has_value());
  EXPECT_EQ(std::count_if(fields.begin(),
                          fields.end(),
                          [](const auto& field) {
                            return field.path.segments().size() == 1U &&
                                   field.path.segments().front() == "internal_value";
                          }),
            0U);
  EXPECT_EQ(sensitive_field.projection, params::ParamProjectionStatus::kRedactedSensitive);
  EXPECT_FALSE(sensitive_field.value.has_value());
  EXPECT_EQ(external_field.projection, params::ParamProjectionStatus::kIncluded);
  EXPECT_TRUE(external_field.value.has_value());
  EXPECT_EQ(snapshot->operators().front().value_coverage,
            ParamSnapshotValueCoverage::kPolicyFiltered);
}

// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterSnapshotTest, RejectsForgedSourceCatalogDigestBeforeValueAccess) {
  const ExecutionPlan plan = test::make_event_policy_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto direct = make_direct_event_policy_coordinator(plan);
  ParamCatalogDigest forged = direct.coordinator->catalog().compatibility_digest();
  forged.value.front() ^= std::byte{1U};

  const auto snapshot = direct.coordinator->capture(
      full_local_parameter_policy(),
      forged,
      ParamCaptureOptions{
          .consistency = ParamCaptureConsistency::kLivePerOperator,
          .unavailable_values = ParamUnavailableValuePolicy::kRecordUnavailable,
      });
  ASSERT_FALSE(snapshot);
  EXPECT_EQ(snapshot.error().code, ErrorCode::kIncompatible);
}

// REQ: PARAM-017
// REQ: PARAM-022
TEST(ParameterSnapshotTest, ConcurrentPublicationCaptureNeverTearsGenerationAndValue) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  auto direct = make_direct_parameter_coordinator(plan);
  auto& coordinator = *direct.coordinator;
  const OperatorRef owner = run_owner(field);
  // Key setup: a publisher thread cycles through three parameter generations while the
  // main thread repeatedly calls capture(); no snapshot may observe a torn pair where
  // the generation integer and the value integer belong to different generations.
  std::atomic_bool start{};
  std::atomic_bool done{};
  std::jthread publisher{[&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::int32_t generation = 1; generation <= 3; ++generation) {
      const std::array patch{
          ParamPatch{
              .target = ParamAddress{.operator_instance = owner, .field = field.field},
              .value = ParamValue::from(std::int32_t{40 + generation}),
          },
      };
      const ParamBatchUpdateResult accepted = coordinator.submit(patch);
      if (!accepted.committed ||
          !coordinator.publish_next_at_callback_boundary(owner, coordinator.publication_frontier())
               .has_value()) {
        done.store(true, std::memory_order_release);
        return;
      }
      auto pin = coordinator.pin_published(owner);
      if (!pin) {
        done.store(true, std::memory_order_release);
        return;
      }
      auto pin_value = checked_value(std::move(pin));
      if (!coordinator.commit_activation_observation(owner, pin_value)) {
        done.store(true, std::memory_order_release);
        return;
      }
      coordinator.release(pin_value);
      std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);
  }};

  start.store(true, std::memory_order_release);
  do {
    const auto snapshot = coordinator.capture(
        full_local_parameter_policy(),
        ParamCaptureOptions{.consistency = ParamCaptureConsistency::kLivePerOperator});
    ASSERT_TRUE(snapshot);
    const auto snapshot_value = checked_value(snapshot);
    ASSERT_EQ(snapshot_value.operators().size(), 1U);
    const ParamSnapshotOperatorRecord& record = snapshot_value.operators().front();
    ASSERT_EQ(record.fields.size(), 1U);
    ASSERT_TRUE(record.fields.front().value.has_value());
    const auto value = checked_value(record.fields.front().value).as<std::int32_t>();
    ASSERT_TRUE(value);
    // Key assertion: value must equal the initial 7 (gen 0) or 40+gen — never a mix.
    EXPECT_EQ(checked_value(value), record.generation == 0U ? 7 : 40 + record.generation);
    std::this_thread::yield();
  } while (!done.load(std::memory_order_acquire));
  publisher.join();

  const auto final = coordinator.capture(
      full_local_parameter_policy(),
      ParamCaptureOptions{.consistency = ParamCaptureConsistency::kLivePerOperator});
  ASSERT_TRUE(final);
  const auto final_snapshot = checked_value(final);
  ASSERT_EQ(final_snapshot.operators().size(), 1U);
  EXPECT_EQ(final_snapshot.operators().front().generation, 3U);
  const auto final_value =
      checked_value(final_snapshot.operators().front().fields.front().value).as<std::int32_t>();
  ASSERT_TRUE(final_value);
  EXPECT_EQ(checked_value(final_value), 43);
}

// REQ: PARAM-017
// REQ: PARAM-018
TEST(ParameterUpdateCoordinatorTest,
     OwningBatchesCopyCallerStorageAndDeriveFromLatestAcceptedGeneration) {
  const ExecutionPlan plan = test::make_event_policy_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.parameter_runtime_capacity().has_value());
  const ParameterFieldPlanRecord& public_field = parameter_field(plan, "public_value");
  const ParameterFieldPlanRecord& external_field = parameter_field(plan, "external_value");
  const OperatorRef owner = run_owner(public_field);

  auto op = std::make_unique<test::EventPolicyParamOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U},
      checked_value(plan.parameter_runtime_capacity()));
  ASSERT_TRUE(coordinator_result);
  auto coordinator = std::move(coordinator_result).value();

  std::array first_patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = public_field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult first = coordinator->submit(first_patch);
  ASSERT_TRUE(first.committed);
  EXPECT_EQ(first.target_generation, 1U);
  first_patch.front().value = ParamValue::from(std::int32_t{99});

  const std::array second_patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = external_field.field},
          .value = ParamValue::from(std::int32_t{52}),
      },
  };
  const ParamBatchUpdateResult second = coordinator->submit(second_patch);
  ASSERT_TRUE(second.committed);
  EXPECT_EQ(second.target_generation, 2U);

  const auto latest_public =
      coordinator->read(ParamAddress{.operator_instance = owner, .field = public_field.field},
                        ParamReadView::kLatestAccepted);
  const auto latest_external =
      coordinator->read(ParamAddress{.operator_instance = owner, .field = external_field.field},
                        ParamReadView::kLatestAccepted);
  ASSERT_TRUE(latest_public);
  ASSERT_TRUE(latest_external);
  EXPECT_EQ(latest_public->generation, 2U);
  EXPECT_EQ(latest_external->generation, 2U);
  EXPECT_EQ(latest_public->update_sequence, second.update_sequence);
  EXPECT_EQ(latest_external->update_sequence, second.update_sequence);
  const auto public_value = latest_public->value.as<std::int32_t>();
  const auto external_value = latest_external->value.as<std::int32_t>();
  ASSERT_TRUE(public_value);
  ASSERT_TRUE(external_value);
  EXPECT_EQ(*public_value, 42);
  EXPECT_EQ(*external_value, 52);
}

// REQ: PARAM-018
TEST(ParameterUpdateCoordinatorTest,
     TypeRangeMutabilityAndExpectedGenerationRejectBeforeAcceptance) {
  const ExecutionPlan plan = test::make_runtime_validation_param_test_plan();
  ASSERT_TRUE(plan.ok());
  ASSERT_TRUE(plan.parameter_runtime_capacity().has_value());
  const auto capacity = checked_value(plan.parameter_runtime_capacity());
  const ParameterFieldPlanRecord& bounded = parameter_field(plan, "bounded_value");
  const ParameterFieldPlanRecord& frozen = parameter_field(plan, "frozen_value");
  EXPECT_EQ(bounded.update_disposition, ParameterFieldUpdateDisposition::kAdmitted);
  EXPECT_EQ(frozen.update_disposition, ParameterFieldUpdateDisposition::kCompileTime);

  auto op = std::make_unique<test::RuntimeValidationParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(coordinator_result);
  auto coordinator = checked_value(std::move(coordinator_result));
  const OperatorRef owner = run_owner(bounded);
  const ParamAddress bounded_address{
      .operator_instance = owner,
      .field = bounded.field,
  };
  const ParamAddress frozen_address{
      .operator_instance = owner,
      .field = frozen.field,
  };
  const auto submit = [&](ParamPatch patch) {
    const std::array patches{std::move(patch)};
    return coordinator->submit(patches);
  };

  const ParamBatchUpdateResult wrong_type = submit(ParamPatch{
      .target = bounded_address,
      .value = ParamValue::from(std::uint32_t{42}),
  });
  EXPECT_FALSE(wrong_type.committed);
  EXPECT_EQ(wrong_type.status, ParamUpdateStatus::kTypeMismatch);
  EXPECT_EQ(wrong_type.update_sequence, 0U);

  const ParamBatchUpdateResult out_of_range = submit(ParamPatch{
      .target = bounded_address,
      .value = ParamValue::from(std::int32_t{101}),
  });
  EXPECT_FALSE(out_of_range.committed);
  EXPECT_EQ(out_of_range.status, ParamUpdateStatus::kOutOfBounds);
  EXPECT_EQ(out_of_range.update_sequence, 0U);

  const ParamBatchUpdateResult immutable = submit(ParamPatch{
      .target = frozen_address,
      .value = ParamValue::from(std::int32_t{9}),
  });
  EXPECT_FALSE(immutable.committed);
  EXPECT_EQ(immutable.status, ParamUpdateStatus::kRequiresRecompile);
  EXPECT_EQ(immutable.update_sequence, 0U);

  const ParamBatchUpdateResult stale = submit(ParamPatch{
      .target = bounded_address,
      .value = ParamValue::from(std::int32_t{42}),
      .expected_generation = ParamGeneration{1U},
  });
  EXPECT_FALSE(stale.committed);
  EXPECT_EQ(stale.status, ParamUpdateStatus::kExpectedGenerationMismatch);
  EXPECT_EQ(stale.update_sequence, 0U);

  const auto published = coordinator->read(bounded_address, ParamReadView::kPublished);
  const auto latest = coordinator->read(bounded_address, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(published);
  ASSERT_TRUE(latest);
  const auto& published_record = checked_ref(published);
  const auto& latest_record = checked_ref(latest);
  EXPECT_EQ(published_record.generation, 0U);
  EXPECT_EQ(latest_record.generation, 0U);
  const auto published_value = published_record.value.as<std::int32_t>();
  const auto latest_value = latest_record.value.as<std::int32_t>();
  ASSERT_TRUE(published_value);
  ASSERT_TRUE(latest_value);
  EXPECT_EQ(checked_ref(published_value), 7);
  EXPECT_EQ(checked_ref(latest_value), 7);
}

// REQ: PARAM-022
TEST(ParameterCatalogProjectionTest, DigestBindsExactVisibleProjectionAndPlanRun) {
  const ExecutionPlan plan = test::make_event_policy_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord& public_field = parameter_field(plan, "public_value");
  const ParameterFieldPlanRecord& internal_field = parameter_field(plan, "internal_value");
  const OperatorRef owner = run_owner(public_field);
  auto direct = make_direct_event_policy_coordinator(plan);
  const ParameterCatalogView& source = direct.coordinator->catalog();

  const ParamAccessPolicy public_policy = full_local_parameter_policy();
  const auto first = detail::filter_runtime_parameter_catalog(
      source, public_policy, public_field.owner.plan_generation, owner.run_generation);
  ASSERT_TRUE(first);
  const auto& first_view = checked_ref(first);
  EXPECT_FALSE(first_view.operators().empty());
  EXPECT_FALSE(first_view.fields().empty());
  EXPECT_NE(first_view.compatibility_digest(), ParamCatalogDigest{});

  ParamAccessPolicy equivalent_policy = public_policy;
  equivalent_policy.permissions = ParamPermission::kRead;
  equivalent_policy.max_event_cursors = 0U;
  const auto equivalent = detail::filter_runtime_parameter_catalog(
      source, equivalent_policy, public_field.owner.plan_generation, owner.run_generation);
  ASSERT_TRUE(equivalent);
  EXPECT_EQ(checked_ref(equivalent).compatibility_digest(), first_view.compatibility_digest());

  ParamAccessPolicy sensitive_metadata_policy = public_policy;
  sensitive_metadata_policy.include_sensitive = true;
  const auto sensitive_metadata = detail::filter_runtime_parameter_catalog(
      source, sensitive_metadata_policy, public_field.owner.plan_generation, owner.run_generation);
  ASSERT_TRUE(sensitive_metadata);
  EXPECT_EQ(checked_ref(sensitive_metadata).compatibility_digest(),
            first_view.compatibility_digest());

  const auto next_run = detail::filter_runtime_parameter_catalog(
      source, public_policy, public_field.owner.plan_generation, owner.run_generation + 1U);
  ASSERT_TRUE(next_run);
  EXPECT_NE(checked_ref(next_run).compatibility_digest(), first_view.compatibility_digest());

  ParamAccessPolicy hidden_field_policy = public_policy;
  hidden_field_policy.allowed_scope = ParamScopeFilter::for_field(ParamAddress{
      .operator_instance = owner,
      .field = internal_field.field,
  });
  hidden_field_policy.include_internal = false;
  const auto empty = detail::filter_runtime_parameter_catalog(
      source, hidden_field_policy, public_field.owner.plan_generation, owner.run_generation);
  ASSERT_TRUE(empty);
  const auto& empty_view = checked_ref(empty);
  EXPECT_TRUE(empty_view.operators().empty());
  EXPECT_TRUE(empty_view.fields().empty());
  EXPECT_NE(empty_view.compatibility_digest(), ParamCatalogDigest{});

  const auto empty_next_run = detail::filter_runtime_parameter_catalog(
      source, hidden_field_policy, public_field.owner.plan_generation, owner.run_generation + 1U);
  ASSERT_TRUE(empty_next_run);
  const auto& empty_next_run_view = checked_ref(empty_next_run);
  EXPECT_TRUE(empty_next_run_view.operators().empty());
  EXPECT_TRUE(empty_next_run_view.fields().empty());
  EXPECT_NE(empty_next_run_view.compatibility_digest(), empty_view.compatibility_digest());
}

// REQ: PARAM-023
TEST(ParameterControlLifetimeTest, DetachWaitsForOutOfLockCloseHookBeforeCoordinatorRetirement) {
  BlockingDetachProbe probe;
  const auto created = detail::ParamControlAccess::create_unattached_link(
      1U, 1U, 0U, std::span<const ParamAddress>{});
  ASSERT_TRUE(created);
  detail::ParamControlLink* const link = checked_ref(created);
  ASSERT_TRUE(detail::ParamControlAccess::attach(link, &probe, &BlockingDetachProbe::operations()));
  const auto retained = detail::ParamControlAccess::retain_for_handle(link, ParamAccessPolicy{});
  ASSERT_TRUE(retained);
  auto tombstone = detail::ParamControlAccess::make_handle_from_retained(checked_ref(retained),
                                                                         ParamAccessPolicy{});
  ASSERT_TRUE(tombstone);
  const auto& tombstone_handle = checked_ref(tombstone);
  EXPECT_TRUE(tombstone_handle.catalog().operators().empty());
  EXPECT_TRUE(tombstone_handle.catalog().fields().empty());
  EXPECT_NE(tombstone_handle.catalog().compatibility_digest(), ParamCatalogDigest{});

  std::thread closer{[&] { detail::ParamControlAccess::close(link); }};
  probe.hook_entered.wait(false, std::memory_order_acquire);
  std::thread detacher{[&] {
    detail::ParamControlAccess::detach_and_release(link);
    probe.coordinator_retired.store(true, std::memory_order_release);
  }};

  detail::ParamControlTestAccess::wait_until_detach_waits(link);
  EXPECT_FALSE(probe.coordinator_retired.load(std::memory_order_acquire));
  probe.allow_hook_return.store(true, std::memory_order_release);
  probe.allow_hook_return.notify_all();
  closer.join();
  detacher.join();
  EXPECT_TRUE(probe.coordinator_retired.load(std::memory_order_acquire));
  EXPECT_FALSE(probe.hook_observed_retirement.load(std::memory_order_acquire));
}

// REQ: PARAM-020
TEST(ParameterControlLifetimeTest, CursorMoveAndClosedLinkCleanupReclaimCoordinatorSlots) {
  BlockingDetachProbe probe;
  probe.cursor_capacity = 2U;
  const auto created = detail::ParamControlAccess::create_unattached_link(
      1U, 1U, 0U, std::span<const ParamAddress>{});
  ASSERT_TRUE(created);
  detail::ParamControlLink* const link = checked_ref(created);
  ASSERT_TRUE(detail::ParamControlAccess::attach(link, &probe, &BlockingDetachProbe::operations()));
  const ParamAccessPolicy policy = full_local_parameter_policy();
  const auto retained = detail::ParamControlAccess::retain_for_handle(link, policy);
  ASSERT_TRUE(retained);
  auto handle =
      detail::ParamControlAccess::make_handle_from_retained(checked_ref(retained), policy);
  ASSERT_TRUE(handle);
  auto& control_handle = checked_ref(handle);

  {
    auto first_result = control_handle.open_event_cursor();
    auto second_result = control_handle.open_event_cursor();
    ASSERT_TRUE(first_result);
    ASSERT_TRUE(second_result);
    ParamEventCursor first = checked_value(std::move(first_result));
    ParamEventCursor second = checked_value(std::move(second_result));
    EXPECT_EQ(probe.active_cursors.load(std::memory_order_acquire), 2U);

    first = std::move(second);
    EXPECT_EQ(probe.active_cursors.load(std::memory_order_acquire), 1U);
    EXPECT_EQ(probe.unregistered_cursors.load(std::memory_order_acquire), 1U);
    auto replacement = control_handle.open_event_cursor();
    ASSERT_TRUE(replacement);
    EXPECT_EQ(probe.active_cursors.load(std::memory_order_acquire), 2U);

    probe.allow_hook_return.store(true, std::memory_order_release);
    detail::ParamControlAccess::close(link);
  }

  EXPECT_EQ(probe.active_cursors.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(probe.unregistered_cursors.load(std::memory_order_acquire), 3U);
  detail::ParamControlAccess::detach_and_release(link);
}

// REQ: PARAM-017
TEST(ParameterUpdateCoordinatorTest, RetiredSlotsCannotBeReusedWhileAnyGenerationPinSurvives) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto op = std::make_unique<test::RuntimeParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto capacity = checked_value(plan.parameter_runtime_capacity());
  ASSERT_EQ(capacity.parameter_arena_bytes % capacity.generation_slots_per_operator, 0U);
  const std::uint64_t arena_bytes_per_slot =
      capacity.parameter_arena_bytes / capacity.generation_slots_per_operator;
  capacity.generation_slots_per_operator = 3U;
  capacity.max_pending_updates_per_operator = 1U;
  capacity.compute_ingress_capacity = 1U;
  capacity.parameter_arena_bytes = arena_bytes_per_slot * capacity.generation_slots_per_operator;
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(coordinator_result);
  auto coordinator = checked_value(std::move(coordinator_result));
  const ParamAddress address{.operator_instance = owner, .field = field.field};
  const auto submit = [&](std::int32_t value) {
    const std::array patch{ParamPatch{.target = address, .value = ParamValue::from(value)}};
    return coordinator->submit(patch);
  };
  const auto publish = [&] {
    return coordinator->publish_next_at_callback_boundary(owner,
                                                          coordinator->publication_frontier());
  };

  auto generation_zero = coordinator->pin_published(owner);
  ASSERT_TRUE(generation_zero);
  auto generation_zero_pin = checked_value(std::move(generation_zero));
  const ParamBatchUpdateResult first = submit(42);
  ASSERT_TRUE(first.committed);
  ASSERT_TRUE(publish().has_value());
  auto generation_one = coordinator->pin_published(owner);
  ASSERT_TRUE(generation_one);
  auto generation_one_pin = checked_value(std::move(generation_one));
  ASSERT_TRUE(coordinator->commit_activation_observation(owner, generation_one_pin));
  const ParamBatchUpdateResult second = submit(43);
  ASSERT_TRUE(second.committed);
  ASSERT_TRUE(publish().has_value());
  auto generation_two = coordinator->pin_published(owner);
  ASSERT_TRUE(generation_two);
  auto generation_two_pin = checked_value(std::move(generation_two));
  ASSERT_TRUE(coordinator->commit_activation_observation(owner, generation_two_pin));
  coordinator->release(generation_two_pin);

  const ParamBatchUpdateResult exhausted = submit(44);
  EXPECT_FALSE(exhausted.committed);
  EXPECT_EQ(exhausted.status, ParamUpdateStatus::kQueueFull);
  EXPECT_EQ(exhausted.update_sequence, 0U);
  const auto unchanged = coordinator->read(address, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(unchanged);
  const auto& unchanged_record = checked_ref(unchanged);
  EXPECT_EQ(unchanged_record.generation, 2U);
  const auto unchanged_value = unchanged_record.value.as<std::int32_t>();
  ASSERT_TRUE(unchanged_value);
  EXPECT_EQ(checked_ref(unchanged_value), 43);

  coordinator->release(generation_zero_pin);
  const ParamBatchUpdateResult third = submit(44);
  ASSERT_TRUE(third.committed);
  EXPECT_EQ(third.target_generation, 3U);
  ASSERT_TRUE(publish().has_value());
  EXPECT_EQ(generation_one_pin.header()->generation, 1U);
  const auto* const retained =
      static_cast<const test::RuntimeProbeParams*>(generation_one_pin.header()->value);
  EXPECT_EQ(retained->value, 42);
  const auto published = coordinator->read(address, ParamReadView::kPublished);
  ASSERT_TRUE(published);
  const auto& published_record = checked_ref(published);
  EXPECT_EQ(published_record.generation, 3U);
  const auto published_value = published_record.value.as<std::int32_t>();
  ASSERT_TRUE(published_value);
  EXPECT_EQ(checked_ref(published_value), 44);
  coordinator->release(generation_one_pin);
}

// REQ: PARAM-018
// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterUpdateCoordinatorTest,
     DetailedAuditIsProfileAuthorizedAndCarriesBoundedActorReasonEvidence) {
  constexpr std::uint64_t kActorId = 0xA11D;
  constexpr ParamReasonId kReasonId = 1001U;
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_parameter_coordinator(plan);
  const auto cursor = direct.coordinator->register_event_cursor(full_local_parameter_policy());
  ASSERT_TRUE(cursor);

  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult accepted =
      direct.coordinator->submit(patch,
                                 ParamUpdateOptions{
                                     .audit = ParamAudit::kDetailed,
                                     .actor_id = kActorId,
                                     .reason = ParamUpdateReason::interned(kReasonId),
                                 });
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.status, ParamUpdateStatus::kAccepted);

  std::array<ParamEvent, 4U> events{};
  const auto drained = direct.coordinator->drain_events(*cursor, events);
  ASSERT_TRUE(drained);
  ASSERT_EQ(*drained, 2U);
  EXPECT_EQ(events[0].state, ParamUpdateState::kSubmitted);
  EXPECT_EQ(events[1].state, ParamUpdateState::kAccepted);
  EXPECT_EQ(events[1].update_sequence, accepted.update_sequence);
  for (std::size_t index = 0U; index < *drained; ++index) {
    EXPECT_EQ(events[index].actor_id, kActorId);
    EXPECT_EQ(events[index].reason_id, kReasonId);
  }
  direct.coordinator->unregister_event_cursor(*cursor);
}

// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterUpdateCoordinatorTest, MultiFieldBatchSubmissionEvidenceUsesOperatorScope) {
  const ExecutionPlan plan = test::make_mixed_mutability_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const ParameterFieldPlanRecord& runtime_field = parameter_field(plan, "runtime_value");
  const ParameterFieldPlanRecord& lifecycle_field = parameter_field(plan, "lifecycle_value");
  const OperatorRef owner = run_owner(runtime_field);
  auto op = std::make_unique<test::MixedMutabilityParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U},
      checked_value(plan.parameter_runtime_capacity()));
  ASSERT_TRUE(coordinator_result);
  auto coordinator = std::move(coordinator_result).value();
  FakeLifecycleSubmitter lifecycle{.outcome = queued_lifecycle_outcome(1U, 11U)};
  ASSERT_TRUE(coordinator->bind_lifecycle_submitter(lifecycle.binding()));
  const auto cursor = coordinator->register_event_cursor(full_local_parameter_policy());
  ASSERT_TRUE(cursor);

  const std::array patches{
      ParamPatch{.target = ParamAddress{.operator_instance = owner, .field = runtime_field.field},
                 .value = ParamValue::from(std::int32_t{42})},
      ParamPatch{.target = ParamAddress{.operator_instance = owner, .field = lifecycle_field.field},
                 .value = ParamValue::from(std::int32_t{84})},
  };
  static_cast<void>(coordinator->submit(patches));

  std::array<ParamEvent, 8U> events{};
  const auto drained = coordinator->drain_events(*cursor, events);
  ASSERT_TRUE(drained);
  const auto submitted = std::find_if(events.begin(), events.begin() + *drained, [](const auto& e) {
    return e.state == ParamUpdateState::kSubmitted;
  });
  ASSERT_NE(submitted, events.begin() + *drained);
  EXPECT_EQ(submitted->operator_instance, owner);
  EXPECT_EQ(submitted->field, ParamFieldId{});
  coordinator->unregister_event_cursor(*cursor);
}

// REQ: PARAM-020
// REQ: PARAM-022
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, EventCursorsReportEveryTerminalParameterOutcome) {
  const auto drain_has_state = [](detail::ParameterUpdateCoordinator& coordinator,
                                  std::uint32_t cursor,
                                  ParamUpdateState state) {
    std::array<ParamEvent, 16U> events{};
    const auto drained = coordinator.drain_events(cursor, events);
    EXPECT_TRUE(drained);
    if (!drained) {
      return false;
    }
    return std::any_of(events.begin(), events.begin() + *drained, [state](const ParamEvent& event) {
      return event.state == state;
    });
  };

  const ExecutionPlan runtime_plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(runtime_plan.ok());
  const ParameterFieldPlanRecord runtime_field = runtime_value_field(runtime_plan);
  const OperatorRef runtime_owner = run_owner(runtime_field);
  const ParamAddress runtime_address{.operator_instance = runtime_owner,
                                     .field = runtime_field.field};
  {
    auto direct = make_direct_parameter_coordinator(runtime_plan);
    const auto cursor = direct.coordinator->register_event_cursor(full_local_parameter_policy());
    ASSERT_TRUE(cursor);
    const std::array patch{
        ParamPatch{.target = runtime_address, .value = ParamValue::from(std::int32_t{7})},
    };
    const ParamBatchUpdateResult no_change = direct.coordinator->submit(patch);
    ASSERT_TRUE(no_change.committed);
    EXPECT_EQ(no_change.status, ParamUpdateStatus::kAcceptedNoChange);
    EXPECT_TRUE(drain_has_state(*direct.coordinator, *cursor, ParamUpdateState::kAcceptedNoChange));
    direct.coordinator->unregister_event_cursor(*cursor);
  }
  {
    auto direct = make_direct_parameter_coordinator(runtime_plan);
    const auto cursor = direct.coordinator->register_event_cursor(full_local_parameter_policy());
    ASSERT_TRUE(cursor);
    const std::array patch{
        ParamPatch{.target = runtime_address, .value = ParamValue::from(std::int32_t{42})},
    };
    ASSERT_TRUE(direct.coordinator->submit(patch).committed);
    direct.coordinator->terminalize_unpublished_at_run_end();
    EXPECT_TRUE(drain_has_state(*direct.coordinator, *cursor, ParamUpdateState::kCanceled));
    direct.coordinator->unregister_event_cursor(*cursor);
  }

  const ExecutionPlan lifecycle_plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(lifecycle_plan.ok());
  const ParameterFieldPlanRecord lifecycle_field = runtime_value_field(lifecycle_plan);
  const OperatorRef lifecycle_owner = run_owner(lifecycle_field);
  const auto lifecycle_patch = [&](std::int32_t value) {
    return std::array{
        ParamPatch{
            .target =
                ParamAddress{.operator_instance = lifecycle_owner, .field = lifecycle_field.field},
            .value = ParamValue::from(value),
        },
    };
  };
  {
    auto direct = make_direct_lifecycle_parameter_coordinator(lifecycle_plan);
    FakeLifecycleSubmitter lifecycle{.outcome = queued_lifecycle_outcome(1U, 11U)};
    ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
    const auto cursor = direct.coordinator->register_event_cursor(full_local_parameter_policy());
    ASSERT_TRUE(cursor);
    ASSERT_TRUE(direct.coordinator->submit(lifecycle_patch(42)).committed);
    lifecycle.outcome = queued_lifecycle_outcome(1U, 12U);
    lifecycle.outcome.coalesced_into = 1U;
    lifecycle.outcome.superseded_transaction = 1U;
    ASSERT_TRUE(direct.coordinator->submit(lifecycle_patch(43)).committed);
    EXPECT_TRUE(drain_has_state(*direct.coordinator, *cursor, ParamUpdateState::kSuperseded));
    direct.coordinator->unregister_event_cursor(*cursor);
  }
  {
    auto direct = make_direct_lifecycle_parameter_coordinator(lifecycle_plan);
    FakeLifecycleSubmitter lifecycle{.outcome = queued_lifecycle_outcome(1U, 11U)};
    ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
    const auto cursor = direct.coordinator->register_event_cursor(full_local_parameter_policy());
    ASSERT_TRUE(cursor);
    const ParamBatchUpdateResult accepted = direct.coordinator->submit(lifecycle_patch(42));
    ASSERT_TRUE(accepted.committed);
    ASSERT_TRUE(direct.coordinator->transition_lifecycle_candidate(
        lifecycle_owner, accepted.target_generation, 1U, LifecycleTransactionState::kRolledBack));
    EXPECT_TRUE(drain_has_state(*direct.coordinator, *cursor, ParamUpdateState::kRolledBack));
    direct.coordinator->unregister_event_cursor(*cursor);
  }
  {
    auto direct = make_direct_lifecycle_parameter_coordinator(lifecycle_plan);
    FakeLifecycleSubmitter lifecycle{.outcome = queued_lifecycle_outcome(1U, 11U)};
    ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
    const auto cursor = direct.coordinator->register_event_cursor(full_local_parameter_policy());
    ASSERT_TRUE(cursor);
    const ParamBatchUpdateResult accepted = direct.coordinator->submit(lifecycle_patch(42));
    ASSERT_TRUE(accepted.committed);
    ASSERT_TRUE(direct.coordinator->transition_lifecycle_candidate(
        lifecycle_owner,
        accepted.target_generation,
        1U,
        LifecycleTransactionState::kFailedStopped));
    EXPECT_TRUE(drain_has_state(*direct.coordinator, *cursor, ParamUpdateState::kFailedStopped));
    direct.coordinator->unregister_event_cursor(*cursor);
  }
}

// REQ: PARAM-019
TEST(ParameterUpdateCoordinatorTest, ComputeIngressIsFiniteSubmittedOnlyAndAllocationFree) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  auto direct = make_direct_parameter_coordinator(plan);
  auto& coordinator = *direct.coordinator;
  const ParamInlinePatch patch{
      .target = ParamAddress{.operator_instance = run_owner(field), .field = field.field},
      .value = ParamValue::from(std::int32_t{42}),
  };

  ParamSubmitResult submitted;
  {
    const test::ScopedAllocationCounter allocations;
    submitted = coordinator.try_submit_from_compute(&patch, 1U);
    EXPECT_EQ(allocations.count(), 0U);
  }
  EXPECT_EQ(submitted.status, ParamUpdateStatus::kSubmitted);
  EXPECT_EQ(submitted.state, ParamUpdateState::kSubmitted);
  EXPECT_EQ(submitted.update_sequence, 0U);

  const std::array over_bound{patch, patch};
  const ParamSubmitResult too_many =
      coordinator.try_submit_from_compute(over_bound.data(), over_bound.size());
  EXPECT_EQ(too_many.status, ParamUpdateStatus::kOutOfBounds);
  EXPECT_EQ(too_many.state, ParamUpdateState::kRejected);

  const auto capacity = checked_value(plan.parameter_runtime_capacity());
  ASSERT_EQ(capacity.compute_ingress_capacity, 4U);
  ASSERT_EQ(capacity.max_compute_patches_per_request, 1U);
  for (std::uint32_t index = 1U; index < capacity.compute_ingress_capacity; ++index) {
    const ParamSubmitResult next = coordinator.try_submit_from_compute(&patch, 1U);
    EXPECT_EQ(next.status, ParamUpdateStatus::kSubmitted);
    EXPECT_EQ(next.state, ParamUpdateState::kSubmitted);
    EXPECT_EQ(next.update_sequence, 0U);
  }
  const ParamSubmitResult full = coordinator.try_submit_from_compute(&patch, 1U);
  EXPECT_EQ(full.status, ParamUpdateStatus::kQueueFull);
  EXPECT_EQ(full.state, ParamUpdateState::kRejected);

  EXPECT_EQ(coordinator.drain_compute_ingress(), capacity.compute_ingress_capacity);
  const auto accepted = coordinator.read(patch.target, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(accepted);
  const auto accepted_value = accepted->value.as<std::int32_t>();
  ASSERT_TRUE(accepted_value);
  EXPECT_EQ(*accepted_value, 42);
}

// REQ: PARAM-019
// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, QueuedComputeIngressIsRejectedOnceAfterCloseRequest) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  auto direct = make_direct_parameter_coordinator(plan);
  auto& coordinator = *direct.coordinator;
  const ParamAddress address{.operator_instance = run_owner(field), .field = field.field};
  const ParamInlinePatch patch{
      .target = address,
      .value = ParamValue::from(std::int32_t{42}),
  };
  const auto cursor = coordinator.register_event_cursor(full_local_parameter_policy());
  ASSERT_TRUE(cursor);

  const ParamSubmitResult submitted = coordinator.try_submit_from_compute(&patch, 1U);
  ASSERT_EQ(submitted.status, ParamUpdateStatus::kSubmitted);
  EXPECT_EQ(coordinator.terminal_sequence(), 0U);

  coordinator.request_close_ingress();
  EXPECT_EQ(coordinator.drain_compute_ingress(), 1U);
  EXPECT_EQ(coordinator.drain_compute_ingress(), 0U);
  EXPECT_EQ(coordinator.terminal_sequence(), 0U);

  const auto latest = coordinator.read(address, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(latest);
  const auto latest_value = latest->value.as<std::int32_t>();
  ASSERT_TRUE(latest_value);
  EXPECT_EQ(*latest_value, 7);
  EXPECT_EQ(latest->generation, 0U);

  std::array<ParamEvent, 2U> events{};
  const auto drained = coordinator.drain_events(*cursor, events);
  ASSERT_TRUE(drained);
  ASSERT_EQ(*drained, events.size());
  EXPECT_EQ(events[0].state, ParamUpdateState::kSubmitted);
  EXPECT_EQ(events[1].state, ParamUpdateState::kRejected);
  EXPECT_EQ(events[1].status, ParamUpdateStatus::kClosed);
  EXPECT_EQ(events[1].update_sequence, 0U);
  coordinator.unregister_event_cursor(*cursor);

  const ParamSubmitResult after_close = coordinator.try_submit_from_compute(&patch, 1U);
  EXPECT_EQ(after_close.status, ParamUpdateStatus::kClosed);
  const ParamStopFenceResult close = coordinator.close_ingress();
  EXPECT_TRUE(close.already_closed);
  EXPECT_EQ(close.terminal_update_sequence, 0U);
}

// REQ: PARAM-019
TEST(ParameterUpdateCoordinatorTest, PublicationBoundaryFreezesConcurrentAcceptedFrontier) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  auto direct = make_direct_parameter_coordinator(plan);
  auto& coordinator = *direct.coordinator;
  const OperatorRef owner = run_owner(field);
  const ParamAddress address{.operator_instance = owner, .field = field.field};
  const std::array first_patch{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };
  const ParamBatchUpdateResult first = coordinator.submit(first_patch);
  ASSERT_TRUE(first.committed);
  const ParamUpdateSequence accepted_cutoff = coordinator.publication_frontier();

  ParamBatchUpdateResult concurrent;
  std::thread submitter{[&] {
    const std::array second_patch{
        ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{43})},
    };
    concurrent = coordinator.submit(second_patch);
  }};
  submitter.join();
  ASSERT_TRUE(concurrent.committed);
  ASSERT_GT(concurrent.update_sequence, accepted_cutoff);

  ASSERT_TRUE(coordinator.publish_next_at_callback_boundary(owner, accepted_cutoff).has_value());
  const auto published_at_cutoff = coordinator.read(address, ParamReadView::kPublished);
  ASSERT_TRUE(published_at_cutoff);
  const auto cutoff_value = published_at_cutoff->value.as<std::int32_t>();
  ASSERT_TRUE(cutoff_value);
  EXPECT_EQ(*cutoff_value, 42);
  const auto still_accepted = coordinator.read(address, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(still_accepted);
  const auto accepted_value = still_accepted->value.as<std::int32_t>();
  ASSERT_TRUE(accepted_value);
  EXPECT_EQ(*accepted_value, 43);

  const ParamUpdateSequence next_cutoff = coordinator.publication_frontier();
  EXPECT_FALSE(coordinator.publish_next_at_callback_boundary(owner, next_cutoff).has_value());
  auto pin = coordinator.pin_published(owner);
  ASSERT_TRUE(pin);
  ASSERT_TRUE(coordinator.commit_activation_observation(owner, *pin));
  coordinator.release(*pin);
  ASSERT_TRUE(coordinator.publish_next_at_callback_boundary(owner, next_cutoff).has_value());
  const auto published_after = coordinator.read(address, ParamReadView::kPublished);
  ASSERT_TRUE(published_after);
  const auto published_value = published_after->value.as<std::int32_t>();
  ASSERT_TRUE(published_value);
  EXPECT_EQ(*published_value, 43);
}

// REQ: PARAM-019
// REQ: PARAM-022
TEST(ParameterUpdateCoordinatorTest, EventOverwriteReportsAnExplicitCursorGap) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  ASSERT_TRUE(plan.parameter_runtime_capacity());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  auto op = std::make_unique<test::RuntimeParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto capacity = checked_value(plan.parameter_runtime_capacity());
  capacity.event_capacity = 2U;
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(coordinator_result);
  auto coordinator = std::move(coordinator_result).value();
  const auto cursor = coordinator->register_event_cursor(full_local_parameter_policy());
  ASSERT_TRUE(cursor);

  const OperatorRef owner = run_owner(field);
  const ParamAddress address{.operator_instance = owner, .field = field.field};
  const std::array patches{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };
  const ParamBatchUpdateResult accepted = coordinator->submit(patches);
  ASSERT_TRUE(accepted.committed);
  const auto published =
      coordinator->publish_next_at_callback_boundary(owner, coordinator->publication_frontier());
  ASSERT_TRUE(published.has_value());
  auto pin = coordinator->pin_published(owner);
  ASSERT_TRUE(pin);
  ASSERT_TRUE(coordinator->commit_activation_observation(owner, *pin));
  coordinator->release(*pin);

  std::array<ParamEvent, 4> events{};
  const auto drained = coordinator->drain_events(*cursor, events);
  ASSERT_TRUE(drained);
  ASSERT_EQ(*drained, 3U);
  // Submitted and accepted were both visible and both overwritten. The single
  // marker is intentionally non-counting: it consumes one projected identity
  // but does not disclose that two source records were lost.
  EXPECT_EQ(events[0].kind, ParamEventKind::kGap);
  EXPECT_EQ(events[0].event_sequence, 1U);
  EXPECT_EQ(events[0].first_missing_sequence, 1U);
  EXPECT_EQ(events[0].last_missing_sequence, 1U);
  EXPECT_EQ(events[1].kind, ParamEventKind::kTransition);
  EXPECT_EQ(events[1].event_sequence, 2U);
  EXPECT_EQ(events[1].state, ParamUpdateState::kPublished);
  EXPECT_EQ(events[2].kind, ParamEventKind::kTransition);
  EXPECT_EQ(events[2].event_sequence, 3U);
  EXPECT_EQ(events[2].state, ParamUpdateState::kObserved);
  coordinator->unregister_event_cursor(*cursor);
}

// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterUpdateCoordinatorTest, EventOverwriteDoesNotRevealPolicyHiddenActivity) {
  const ExecutionPlan plan = test::make_event_policy_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord& public_field = parameter_field(plan, "public_value");
  const ParameterFieldPlanRecord& internal_field = parameter_field(plan, "internal_value");
  const ParameterFieldPlanRecord& sensitive_field = parameter_field(plan, "sensitive_value");
  const ParameterFieldPlanRecord& external_field = parameter_field(plan, "external_value");
  const OperatorRef owner = run_owner(public_field);

  const auto submit_value = [&](detail::ParameterUpdateCoordinator& coordinator,
                                const ParameterFieldPlanRecord& field,
                                std::int32_t value) {
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = owner, .field = field.field},
            .value = ParamValue::from(value),
        },
    };
    return coordinator.submit(patch);
  };
  const auto expect_no_visible_records = [](detail::ParameterUpdateCoordinator& coordinator,
                                            const ParamAccessPolicy& policy,
                                            const auto& produce_hidden_events) {
    const auto cursor = coordinator.register_event_cursor(policy);
    ASSERT_TRUE(cursor);
    produce_hidden_events();
    std::array<ParamEvent, 4> events{};
    const auto drained = coordinator.drain_events(*cursor, events);
    ASSERT_TRUE(drained);
    EXPECT_EQ(*drained, 0U);
    coordinator.unregister_event_cursor(*cursor);
  };

  {
    auto direct = make_direct_event_policy_coordinator(plan);
    ParamAccessPolicy restricted = full_local_parameter_policy();
    restricted.include_internal = false;
    restricted.include_sensitive = false;
    expect_no_visible_records(*direct.coordinator, restricted, [&] {
      EXPECT_TRUE(submit_value(*direct.coordinator, internal_field, 81).committed);
      EXPECT_TRUE(submit_value(*direct.coordinator, sensitive_field, 91).committed);
    });
  }

  {
    auto direct = make_direct_event_policy_coordinator(plan);
    ParamAccessPolicy field_scoped = full_local_parameter_policy();
    field_scoped.allowed_scope = ParamScopeFilter::for_field(
        ParamAddress{.operator_instance = owner, .field = external_field.field});
    expect_no_visible_records(*direct.coordinator, field_scoped, [&] {
      EXPECT_TRUE(submit_value(*direct.coordinator, public_field, 71).committed);
      EXPECT_TRUE(submit_value(*direct.coordinator, internal_field, 81).committed);
    });
  }

  {
    auto direct = make_direct_event_policy_coordinator(plan);
    ParamAccessPolicy external = full_local_parameter_policy();
    external.origin = ParamControlOrigin::kExternalAdapter;
    expect_no_visible_records(*direct.coordinator, external, [&] {
      EXPECT_TRUE(submit_value(*direct.coordinator, public_field, 71).committed);
      EXPECT_TRUE(submit_value(*direct.coordinator, internal_field, 81).committed);
    });
  }

  {
    auto direct = make_direct_event_policy_coordinator(plan);
    const ParamAccessPolicy restricted = full_local_parameter_policy();
    expect_no_visible_records(*direct.coordinator, restricted, [&] {
      const ParameterFieldPlanRecord unknown{
          .owner = public_field.owner,
          .field = public_field.field + internal_field.field + sensitive_field.field +
                   external_field.field,
      };
      EXPECT_FALSE(submit_value(*direct.coordinator, unknown, 1).committed);
      EXPECT_FALSE(submit_value(*direct.coordinator, unknown, 2).committed);
    });
  }
}

// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterUpdateCoordinatorTest, EventOverwriteReportsOnlyLostVisibleProjection) {
  const ExecutionPlan plan = test::make_event_policy_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord& public_field = parameter_field(plan, "public_value");
  const ParameterFieldPlanRecord& internal_field = parameter_field(plan, "internal_value");
  const OperatorRef owner = run_owner(public_field);
  auto direct = make_direct_event_policy_coordinator(plan);

  const auto cursor = direct.coordinator->register_event_cursor(full_local_parameter_policy());
  ASSERT_TRUE(cursor);
  const auto submit = [&](const ParameterFieldPlanRecord& field, std::int32_t value) {
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = owner, .field = field.field},
            .value = ParamValue::from(value),
        },
    };
    return direct.coordinator->submit(patch);
  };
  EXPECT_TRUE(submit(public_field, 71).committed);
  EXPECT_TRUE(submit(internal_field, 81).committed);

  std::array<ParamEvent, 4> events{};
  const auto drained = direct.coordinator->drain_events(*cursor, events);
  ASSERT_TRUE(drained);
  ASSERT_EQ(*drained, 1U);
  // The later internal-only source records remain filtered. They neither add
  // projected records nor reveal how many visible records the opaque gap lost.
  EXPECT_EQ(events[0].kind, ParamEventKind::kGap);
  EXPECT_EQ(events[0].event_sequence, 1U);
  EXPECT_EQ(events[0].first_missing_sequence, events[0].event_sequence);
  EXPECT_EQ(events[0].last_missing_sequence, events[0].event_sequence);
  direct.coordinator->unregister_event_cursor(*cursor);
}

// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterUpdateCoordinatorTest, NewCursorStartsAtOldestRetainedPolicyProjection) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  auto op = std::make_unique<test::RuntimeParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto capacity = checked_value(plan.parameter_runtime_capacity());
  capacity.event_capacity = 2U;
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(coordinator_result);
  auto coordinator = std::move(coordinator_result).value();

  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = run_owner(field), .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  ASSERT_TRUE(coordinator->submit(patch).committed);
  const auto cursor = coordinator->register_event_cursor(full_local_parameter_policy());
  ASSERT_TRUE(cursor);
  std::array<ParamEvent, 4> events{};
  const auto drained = coordinator->drain_events(*cursor, events);
  ASSERT_TRUE(drained);
  ASSERT_EQ(*drained, 2U);
  EXPECT_EQ(events[0].kind, ParamEventKind::kTransition);
  EXPECT_EQ(events[0].event_sequence, 1U);
  EXPECT_EQ(events[0].state, ParamUpdateState::kSubmitted);
  EXPECT_EQ(events[1].kind, ParamEventKind::kTransition);
  EXPECT_EQ(events[1].event_sequence, 2U);
  EXPECT_EQ(events[1].state, ParamUpdateState::kAccepted);
  coordinator->unregister_event_cursor(*cursor);
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, RuntimeAcceptedPathsOwnAndReleaseExactEventReservations) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_parameter_coordinator(plan);
  auto& coordinator = *direct.coordinator;
  const auto submit = [&](std::int32_t value) {
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = owner, .field = field.field},
            .value = ParamValue::from(value),
        },
    };
    return coordinator.submit(patch);
  };
  const auto reserved = [&] {
    return detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(coordinator);
  };

  EXPECT_EQ(reserved(), 0U);
  auto generation_zero = coordinator.pin_published(owner);
  ASSERT_TRUE(generation_zero);
  ASSERT_EQ(generation_zero->header()->generation, 0U);
  ASSERT_TRUE(coordinator.commit_activation_observation(owner, *generation_zero));
  ASSERT_TRUE(coordinator.commit_activation_observation(owner, *generation_zero));
  EXPECT_EQ(reserved(), 0U);
  coordinator.release(*generation_zero);

  const ParamBatchUpdateResult accepted = submit(42);
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(reserved(), 7U);
  ASSERT_TRUE(
      coordinator.publish_next_at_callback_boundary(owner, coordinator.publication_frontier())
          .has_value());
  EXPECT_EQ(reserved(), 6U);
  auto pin = coordinator.pin_published(owner);
  ASSERT_TRUE(pin);
  ASSERT_TRUE(coordinator.commit_activation_observation(owner, *pin));
  EXPECT_EQ(reserved(), 0U);
  ASSERT_TRUE(coordinator.commit_activation_observation(owner, *pin));
  EXPECT_EQ(reserved(), 0U);
  const auto observed = coordinator.result(accepted.update_sequence);
  ASSERT_TRUE(observed);
  EXPECT_EQ(observed->state, ParamUpdateState::kObserved);
  coordinator.release(*pin);

  const ParamBatchUpdateResult no_change = submit(42);
  ASSERT_TRUE(no_change.committed);
  EXPECT_EQ(no_change.status, ParamUpdateStatus::kAcceptedNoChange);
  EXPECT_EQ(reserved(), 0U);
  const ParamAddress address{.operator_instance = owner, .field = field.field};
  const auto latest_no_change = coordinator.read(address, ParamReadView::kLatestAccepted);
  ASSERT_TRUE(latest_no_change);
  EXPECT_EQ(latest_no_change->generation, 1U);
  EXPECT_EQ(latest_no_change->update_sequence, no_change.update_sequence);
  EXPECT_EQ(latest_no_change->state, ParamUpdateState::kAcceptedNoChange);
  const auto latest_no_change_value = latest_no_change->value.as<std::int32_t>();
  ASSERT_TRUE(latest_no_change_value);
  EXPECT_EQ(*latest_no_change_value, 42);

  const ParamBatchUpdateResult canceled = submit(43);
  ASSERT_TRUE(canceled.committed);
  EXPECT_EQ(reserved(), 7U);
  coordinator.terminalize_unpublished_at_run_end();
  EXPECT_EQ(reserved(), 0U);
  const auto canceled_result = coordinator.result(canceled.update_sequence);
  ASSERT_TRUE(canceled_result);
  EXPECT_EQ(canceled_result->state, ParamUpdateState::kCanceled);
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest,
     LifecyclePublicationTerminalizesDisplacedUnobservedRuntimeEvidence) {
  const ExecutionPlan plan = test::make_mixed_mutability_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const ParameterFieldPlanRecord& runtime_field = parameter_field(plan, "runtime_value");
  const ParameterFieldPlanRecord& lifecycle_field = parameter_field(plan, "lifecycle_value");
  const OperatorRef owner = run_owner(runtime_field);
  auto op = std::make_unique<test::MixedMutabilityParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U},
      checked_value(plan.parameter_runtime_capacity()));
  ASSERT_TRUE(coordinator_result);
  auto coordinator = std::move(coordinator_result).value();
  FakeLifecycleSubmitter lifecycle{.outcome = queued_lifecycle_outcome(1U, 11U)};
  ASSERT_TRUE(coordinator->bind_lifecycle_submitter(lifecycle.binding()));

  const std::array runtime_patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = runtime_field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult runtime_update = coordinator->submit(runtime_patch);
  ASSERT_TRUE(runtime_update.committed);
  ASSERT_TRUE(
      coordinator->publish_next_at_callback_boundary(owner, coordinator->publication_frontier())
          .has_value());
  ASSERT_EQ(coordinator->result(runtime_update.update_sequence)->state,
            ParamUpdateState::kPublished);

  const std::array lifecycle_patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = lifecycle_field.field},
          .value = ParamValue::from(std::int32_t{84}),
      },
  };
  const ParamBatchUpdateResult lifecycle_update = coordinator->submit(lifecycle_patch);
  ASSERT_TRUE(lifecycle_update.committed);
  ASSERT_TRUE(coordinator->transition_lifecycle_candidate(
      owner, lifecycle_update.target_generation, 1U, LifecycleTransactionState::kQuiescing));
  ASSERT_TRUE(coordinator->transition_lifecycle_candidate(
      owner, lifecycle_update.target_generation, 1U, LifecycleTransactionState::kRestarting));
  ASSERT_TRUE(coordinator->transition_lifecycle_candidate(
      owner, lifecycle_update.target_generation, 1U, LifecycleTransactionState::kPublished));
  const auto displaced = coordinator->result(runtime_update.update_sequence);
  ASSERT_TRUE(displaced);
  EXPECT_EQ(displaced->state, ParamUpdateState::kSuperseded);

  auto published = coordinator->pin_published(owner);
  ASSERT_TRUE(published);
  ASSERT_TRUE(coordinator->commit_activation_observation(owner, *published));
  coordinator->release(*published);
  EXPECT_EQ(detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*coordinator),
            0U);
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, PublishedResultCannotEvictItsPendingObservationEvidence) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto op = std::make_unique<test::RuntimeParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto capacity = checked_value(plan.parameter_runtime_capacity());
  capacity.audit_capacity = 1U;
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(coordinator_result);
  auto coordinator = std::move(coordinator_result).value();
  const auto submit = [&](std::int32_t value) {
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = owner, .field = field.field},
            .value = ParamValue::from(value),
        },
    };
    return coordinator->submit(patch);
  };

  const ParamBatchUpdateResult first = submit(42);
  ASSERT_TRUE(first.committed);
  ASSERT_TRUE(
      coordinator->publish_next_at_callback_boundary(owner, coordinator->publication_frontier())
          .has_value());
  const auto published = coordinator->result(first.update_sequence);
  ASSERT_TRUE(published);
  EXPECT_EQ(published->state, ParamUpdateState::kPublished);
  const ParamBatchUpdateResult before_observation = submit(43);
  EXPECT_FALSE(before_observation.committed);
  EXPECT_EQ(before_observation.status, ParamUpdateStatus::kQueueFull);

  auto pin = coordinator->pin_published(owner);
  ASSERT_TRUE(pin);
  ASSERT_TRUE(coordinator->commit_activation_observation(owner, *pin));
  EXPECT_EQ(detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*coordinator),
            0U);
  ASSERT_TRUE(coordinator->commit_activation_observation(owner, *pin));
  coordinator->release(*pin);
  const auto observed = coordinator->result(first.update_sequence);
  ASSERT_TRUE(observed);
  EXPECT_EQ(observed->state, ParamUpdateState::kObserved);
  const ParamBatchUpdateResult after_observation = submit(43);
  EXPECT_TRUE(after_observation.committed);
}

// REQ: PARAM-020
// REQ: PARAM-022
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, EventSequenceExhaustionRejectsBeforeLifecycleAdmission) {
  const ExecutionPlan plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  auto direct = make_direct_lifecycle_parameter_coordinator(plan);
  FakeLifecycleSubmitter lifecycle{
      .outcome = queued_lifecycle_outcome(1U, 11U),
  };
  ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
  detail::ParameterUpdateCoordinatorTestAccess::set_event_sequence(
      *direct.coordinator, std::numeric_limits<ParamEventSequence>::max() - 1U);

  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = run_owner(field), .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult rejected = direct.coordinator->submit(patch);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.status, ParamUpdateStatus::kQueueFull);
  EXPECT_EQ(rejected.update_sequence, 0U);
  EXPECT_EQ(lifecycle.submit_calls, 0U);
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      0U);
  auto published = direct.coordinator->pin_published(run_owner(field));
  ASSERT_TRUE(published);
  EXPECT_EQ(static_cast<const test::LifecycleParamProbeParams*>(published->header()->value)->value,
            7);
  direct.coordinator->release(*published);
}

// REQ: PARAM-018
// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, UpdateSequenceExhaustionRejectsBeforeLifecycleAdmission) {
  const ExecutionPlan plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_lifecycle_parameter_coordinator(plan);
  FakeLifecycleSubmitter lifecycle{
      .outcome = queued_lifecycle_outcome(1U, 11U),
  };
  ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
  detail::ParameterUpdateCoordinatorTestAccess::set_update_sequence(
      *direct.coordinator, std::numeric_limits<ParamUpdateSequence>::max());

  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult rejected = direct.coordinator->submit(patch);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.status, ParamUpdateStatus::kQueueFull);
  EXPECT_EQ(rejected.update_sequence, 0U);
  EXPECT_EQ(lifecycle.submit_calls, 0U);
  const auto published = direct.coordinator->read(
      ParamAddress{.operator_instance = owner, .field = field.field}, ParamReadView::kPublished);
  ASSERT_TRUE(published);
  EXPECT_EQ(published->generation, 0U);
  EXPECT_EQ(published->update_sequence, 0U);
  const auto value = published->value.as<std::int32_t>();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 7);
}

// REQ: PARAM-018
// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, NoChangeLineagePinSaturationRejectsBeforeSequenceAssignment) {
  constexpr std::uint64_t kMaximumPinCount = std::numeric_limits<std::uint64_t>::max() >> 1U;
  constexpr std::uint64_t kRetiredBit = std::uint64_t{1} << 63U;
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_parameter_coordinator(plan);
  detail::ParameterUpdateCoordinatorTestAccess::set_published_pin_count(
      *direct.coordinator, owner, kMaximumPinCount);

  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{7}),
      },
  };
  const ParamBatchUpdateResult rejected = direct.coordinator->submit(patch);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.status, ParamUpdateStatus::kQueueFull);
  EXPECT_EQ(rejected.update_sequence, 0U);
  EXPECT_EQ(direct.coordinator->terminal_sequence(), 0U);

  const std::uint64_t claims = detail::ParameterUpdateCoordinatorTestAccess::published_claim_word(
      *direct.coordinator, owner);
  EXPECT_EQ(claims, kMaximumPinCount);
  EXPECT_EQ(claims & kRetiredBit, 0U);
  const auto latest =
      direct.coordinator->read(ParamAddress{.operator_instance = owner, .field = field.field},
                               ParamReadView::kLatestAccepted);
  ASSERT_TRUE(latest);
  EXPECT_EQ(latest->generation, 0U);
  EXPECT_EQ(latest->update_sequence, 0U);
  EXPECT_EQ(latest->state, ParamUpdateState::kObserved);

  // The synthetic saturation owns no real pins; restore the fixture before teardown.
  detail::ParameterUpdateCoordinatorTestAccess::set_published_pin_count(
      *direct.coordinator, owner, 0U);
}

// REQ: PARAM-018
// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, GenerationExhaustionRejectsBeforeLifecycleAdmission) {
  const ExecutionPlan plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_lifecycle_parameter_coordinator(plan);
  FakeLifecycleSubmitter lifecycle{
      .outcome = queued_lifecycle_outcome(1U, 11U),
  };
  ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
  detail::ParameterUpdateCoordinatorTestAccess::set_published_generation(
      *direct.coordinator, owner, std::numeric_limits<ParamGeneration>::max());

  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult rejected = direct.coordinator->submit(patch);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.status, ParamUpdateStatus::kQueueFull);
  EXPECT_EQ(rejected.update_sequence, 0U);
  EXPECT_EQ(lifecycle.submit_calls, 0U);
  const auto published = direct.coordinator->read(
      ParamAddress{.operator_instance = owner, .field = field.field}, ParamReadView::kPublished);
  ASSERT_TRUE(published);
  EXPECT_EQ(published->generation, std::numeric_limits<ParamGeneration>::max());
  EXPECT_EQ(published->update_sequence, 0U);
  const auto value = published->value.as<std::int32_t>();
  ASSERT_TRUE(value);
  EXPECT_EQ(*value, 7);
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, CursorRegisteredAfterEmptyExhaustedLedgerStaysAtEnd) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  auto direct = make_direct_parameter_coordinator(plan);
  detail::ParameterUpdateCoordinatorTestAccess::set_event_sequence(
      *direct.coordinator, std::numeric_limits<ParamEventSequence>::max());

  const auto cursor = direct.coordinator->register_event_cursor(full_local_parameter_policy());
  ASSERT_TRUE(cursor);
  std::array<ParamEvent, 2U> events{};
  const auto first = direct.coordinator->drain_events(*cursor, events);
  ASSERT_TRUE(first);
  EXPECT_EQ(*first, 0U);
  const auto second = direct.coordinator->drain_events(*cursor, events);
  ASSERT_TRUE(second);
  EXPECT_EQ(*second, 0U);
  direct.coordinator->unregister_event_cursor(*cursor);
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, CursorDrainsRetainedFinalSourceEventWithoutWrapping) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  auto direct = make_direct_parameter_coordinator(plan);
  detail::ParameterUpdateCoordinatorTestAccess::set_event_sequence(
      *direct.coordinator, std::numeric_limits<ParamEventSequence>::max() - 2U);

  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = run_owner(field), .field = field.field},
          .value = ParamValue::from(true),
      },
  };
  const ParamBatchUpdateResult rejected = direct.coordinator->submit(patch);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.status, ParamUpdateStatus::kTypeMismatch);

  const auto cursor = direct.coordinator->register_event_cursor(full_local_parameter_policy());
  ASSERT_TRUE(cursor);
  std::array<ParamEvent, 3U> events{};
  const auto first = direct.coordinator->drain_events(*cursor, events);
  ASSERT_TRUE(first);
  ASSERT_EQ(*first, 2U);
  EXPECT_EQ(events[0].event_sequence, 1U);
  EXPECT_EQ(events[0].state, ParamUpdateState::kSubmitted);
  EXPECT_EQ(events[1].event_sequence, 2U);
  EXPECT_EQ(events[1].state, ParamUpdateState::kRejected);
  EXPECT_EQ(events[1].status, ParamUpdateStatus::kTypeMismatch);
  const auto second = direct.coordinator->drain_events(*cursor, events);
  ASSERT_TRUE(second);
  EXPECT_EQ(*second, 0U);
  direct.coordinator->unregister_event_cursor(*cursor);
}

// REQ: PARAM-017
// REQ: PARAM-018
// REQ: PARAM-021
TEST(ParameterUpdateCoordinatorTest, LifecycleCandidatePinsExactPhaseViewsAndPublishesAfterStart) {
  const ExecutionPlan plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_lifecycle_parameter_coordinator(plan);
  FakeLifecycleSubmitter lifecycle{
      .outcome = queued_lifecycle_outcome(1U, 11U),
  };
  ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));

  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult accepted = direct.coordinator->submit(patch);
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.status, ParamUpdateStatus::kAccepted);
  EXPECT_EQ(accepted.target_generation, 1U);
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      7U);
  EXPECT_EQ(lifecycle.last_request.earliest_stage, LifecycleStage::kAllocate);
  EXPECT_FALSE(lifecycle.last_request.equivalent_to_published);

  auto published_before = direct.coordinator->pin_published(owner);
  ASSERT_TRUE(published_before);
  EXPECT_EQ(
      static_cast<const test::LifecycleParamProbeParams*>(published_before->header()->value)->value,
      7);
  direct.coordinator->release(*published_before);

  // Key setup: cycle through all four forward candidate stages pinning generation 1 at
  // each stage to verify the candidate value (42) is visible throughout the restart.
  constexpr std::array kForwardCandidateStages{
      LifecycleStage::kAllocate,
      LifecycleStage::kWarmUp,
      LifecycleStage::kArm,
      LifecycleStage::kStart,
  };
  for (const LifecycleStage stage : kForwardCandidateStages) {
    static_cast<void>(stage);
    auto candidate = direct.coordinator->pin_lifecycle_generation(owner, 1U);
    ASSERT_TRUE(candidate);
    EXPECT_EQ(
        static_cast<const test::LifecycleParamProbeParams*>(candidate->header()->value)->value, 42);
    direct.coordinator->release(*candidate);
  }

  ASSERT_TRUE(direct.coordinator->transition_lifecycle_candidate(
      owner, 1U, 1U, LifecycleTransactionState::kQuiescing));
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      6U);
  ASSERT_TRUE(direct.coordinator->transition_lifecycle_candidate(
      owner, 1U, 1U, LifecycleTransactionState::kRestarting));
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      5U);
  ASSERT_TRUE(direct.coordinator->transition_lifecycle_candidate(
      owner, 1U, 1U, LifecycleTransactionState::kPublished));
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      4U);

  // Key assertion: after the kPublished transition the pinned published view carries the
  // candidate value (42) at generation 1.
  auto published_after = direct.coordinator->pin_published(owner);
  ASSERT_TRUE(published_after);
  ASSERT_TRUE(direct.coordinator->commit_activation_observation(owner, *published_after));
  EXPECT_EQ(
      static_cast<const test::LifecycleParamProbeParams*>(published_after->header()->value)->value,
      42);
  EXPECT_EQ(published_after->header()->generation, 1U);
  direct.coordinator->release(*published_after);
  const auto result = direct.coordinator->result(accepted.update_sequence);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->state, ParamUpdateState::kObserved);
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      0U);
}

// REQ: PARAM-017
// REQ: PARAM-018
// REQ: PARAM-021
TEST(ParameterUpdateCoordinatorTest, LifecycleRollbackRetainsCandidateUntilFinalPinReleases) {
  const ExecutionPlan plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_lifecycle_parameter_coordinator(plan);
  FakeLifecycleSubmitter lifecycle{
      .outcome = queued_lifecycle_outcome(1U, 11U),
  };
  ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));

  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult accepted = direct.coordinator->submit(patch);
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      7U);
  auto rollback_pin = direct.coordinator->pin_lifecycle_generation(owner, 1U);
  ASSERT_TRUE(rollback_pin);
  ASSERT_TRUE(direct.coordinator->transition_lifecycle_candidate(
      owner, 1U, 1U, LifecycleTransactionState::kRestarting));
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      6U);
  ASSERT_TRUE(direct.coordinator->transition_lifecycle_candidate(
      owner, 1U, 1U, LifecycleTransactionState::kRolledBack));
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      0U);

  EXPECT_EQ(
      static_cast<const test::LifecycleParamProbeParams*>(rollback_pin->header()->value)->value,
      42);
  auto restored = direct.coordinator->pin_published(owner);
  ASSERT_TRUE(restored);
  EXPECT_EQ(static_cast<const test::LifecycleParamProbeParams*>(restored->header()->value)->value,
            7);
  direct.coordinator->release(*restored);
  direct.coordinator->release(*rollback_pin);
  const auto retired = direct.coordinator->pin_lifecycle_generation(owner, 1U);
  ASSERT_FALSE(retired);
  EXPECT_EQ(retired.error().code, ErrorCode::kNotFound);
  const auto result = direct.coordinator->result(accepted.update_sequence);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->state, ParamUpdateState::kRolledBack);
}

// REQ: PARAM-018
// REQ: PARAM-021
TEST(ParameterUpdateCoordinatorTest, LifecycleNoChangeAndReplacementPreserveLineage) {
  const ExecutionPlan plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_lifecycle_parameter_coordinator(plan);
  FakeLifecycleSubmitter lifecycle{
      .outcome = queued_lifecycle_outcome(1U, 11U),
  };
  ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
  const auto submit = [&](std::int32_t value) {
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = owner, .field = field.field},
            .value = ParamValue::from(value),
        },
    };
    return direct.coordinator->submit(patch);
  };

  const ParamBatchUpdateResult first = submit(42);
  ASSERT_TRUE(first.committed);
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      7U);
  lifecycle.outcome = queued_lifecycle_outcome(1U, 12U);
  lifecycle.outcome.coalesced_into = 1U;
  lifecycle.outcome.superseded_transaction = 1U;
  const ParamBatchUpdateResult replacement = submit(43);
  ASSERT_TRUE(replacement.committed);
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      7U);
  EXPECT_EQ(replacement.target_generation, 2U);
  const auto first_result = direct.coordinator->result(first.update_sequence);
  ASSERT_TRUE(first_result);
  EXPECT_EQ(first_result->state, ParamUpdateState::kSuperseded);
  auto latest = direct.coordinator->pin_lifecycle_generation(owner, 2U);
  ASSERT_TRUE(latest);
  EXPECT_EQ(static_cast<const test::LifecycleParamProbeParams*>(latest->header()->value)->value,
            43);
  direct.coordinator->release(*latest);

  lifecycle.outcome = detail::ParameterLifecycleSubmitOutcome{
      .receipt = LifecycleRestartReceipt{.receipt_id = 13U},
      .state = LifecycleTransactionState::kAcceptedNoChange,
      .superseded_transaction = 1U,
  };
  const ParamBatchUpdateResult no_change = submit(7);
  ASSERT_TRUE(no_change.committed);
  EXPECT_EQ(no_change.status, ParamUpdateStatus::kAcceptedNoChange);
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      0U);
  EXPECT_TRUE(lifecycle.last_request.equivalent_to_published);
  const auto no_candidate = direct.coordinator->pin_lifecycle_generation(owner, 3U);
  EXPECT_FALSE(no_candidate);
  const auto latest_no_change =
      direct.coordinator->read(ParamAddress{.operator_instance = owner, .field = field.field},
                               ParamReadView::kLatestAccepted);
  ASSERT_TRUE(latest_no_change);
  EXPECT_EQ(latest_no_change->generation, 0U);
  EXPECT_EQ(latest_no_change->update_sequence, no_change.update_sequence);
  EXPECT_EQ(latest_no_change->state, ParamUpdateState::kAcceptedNoChange);
  const auto no_change_value = latest_no_change->value.as<std::int32_t>();
  ASSERT_TRUE(no_change_value);
  EXPECT_EQ(*no_change_value, 7);
  auto published = direct.coordinator->pin_published(owner);
  ASSERT_TRUE(published);
  EXPECT_EQ(static_cast<const test::LifecycleParamProbeParams*>(published->header()->value)->value,
            7);
  direct.coordinator->release(*published);
}

// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterUpdateCoordinatorTest, StoppedCaptureFrontierRetainsEveryUnpublishedLifecycleGap) {
  const ExecutionPlan plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  const ParamAccessPolicy policy = full_local_parameter_policy();
  const auto submit = [&](detail::ParameterUpdateCoordinator& coordinator, std::int32_t value) {
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = owner, .field = field.field},
            .value = ParamValue::from(value),
        },
    };
    return coordinator.submit(patch);
  };

  for (const LifecycleTransactionState terminal_state :
       {LifecycleTransactionState::kCanceled,
        LifecycleTransactionState::kRolledBack,
        LifecycleTransactionState::kFailedStopped}) {
    SCOPED_TRACE(static_cast<int>(terminal_state));
    auto direct = make_direct_lifecycle_parameter_coordinator(plan);
    FakeLifecycleSubmitter lifecycle{.outcome = queued_lifecycle_outcome(1U, 11U)};
    ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
    const ParamBatchUpdateResult accepted = submit(*direct.coordinator, 42);
    ASSERT_TRUE(accepted.committed);
    ASSERT_TRUE(direct.coordinator->transition_lifecycle_candidate(
        owner, accepted.target_generation, 1U, terminal_state));
    direct.coordinator->terminalize_unpublished_at_run_end();
    const auto stopped = direct.coordinator->capture(policy, ParamCaptureOptions{});
    ASSERT_TRUE(stopped);
    ASSERT_TRUE(stopped->accepted_through().has_value());
    ASSERT_TRUE(stopped->published_through().has_value());
    EXPECT_EQ(checked_value(stopped->accepted_through()), accepted.update_sequence);
    EXPECT_EQ(checked_value(stopped->published_through()), 0U);
  }

  auto direct = make_direct_lifecycle_parameter_coordinator(plan);
  FakeLifecycleSubmitter lifecycle{.outcome = queued_lifecycle_outcome(1U, 21U)};
  ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
  const ParamBatchUpdateResult superseded = submit(*direct.coordinator, 42);
  ASSERT_TRUE(superseded.committed);
  lifecycle.outcome = queued_lifecycle_outcome(1U, 22U);
  lifecycle.outcome.coalesced_into = 1U;
  lifecycle.outcome.superseded_transaction = 1U;
  const ParamBatchUpdateResult published = submit(*direct.coordinator, 43);
  ASSERT_TRUE(published.committed);
  const auto superseded_result = direct.coordinator->result(superseded.update_sequence);
  ASSERT_TRUE(superseded_result);
  EXPECT_EQ(superseded_result->state, ParamUpdateState::kSuperseded);
  ASSERT_TRUE(direct.coordinator->transition_lifecycle_candidate(
      owner, published.target_generation, 1U, LifecycleTransactionState::kPublished));
  direct.coordinator->terminalize_unpublished_at_run_end();
  const auto stopped = direct.coordinator->capture(policy, ParamCaptureOptions{});
  ASSERT_TRUE(stopped);
  ASSERT_TRUE(stopped->accepted_through().has_value());
  ASSERT_TRUE(stopped->published_through().has_value());
  EXPECT_EQ(checked_value(stopped->accepted_through()), published.update_sequence);
  EXPECT_EQ(checked_value(stopped->published_through()), 0U);
  ASSERT_EQ(stopped->operators().size(), 1U);
  EXPECT_EQ(stopped->operators().front().published_update_sequence, published.update_sequence);
}

// REQ: PARAM-020
// REQ: PARAM-022
TEST(ParameterUpdateCoordinatorTest, StoppedCaptureFrontierTreatsAcceptedNoChangeAsEffective) {
  const ExecutionPlan plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_lifecycle_parameter_coordinator(plan);
  FakeLifecycleSubmitter lifecycle{
      .outcome =
          detail::ParameterLifecycleSubmitOutcome{
              .receipt = LifecycleRestartReceipt{.receipt_id = 31U},
              .state = LifecycleTransactionState::kAcceptedNoChange,
          },
  };
  ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{7}),
      },
  };
  const ParamBatchUpdateResult accepted = direct.coordinator->submit(patch);
  ASSERT_TRUE(accepted.committed);
  EXPECT_EQ(accepted.status, ParamUpdateStatus::kAcceptedNoChange);
  const auto result = direct.coordinator->result(accepted.update_sequence);
  ASSERT_TRUE(result);
  EXPECT_EQ(result->state, ParamUpdateState::kAcceptedNoChange);
  direct.coordinator->terminalize_unpublished_at_run_end();
  const auto stopped =
      direct.coordinator->capture(full_local_parameter_policy(), ParamCaptureOptions{});
  ASSERT_TRUE(stopped);
  ASSERT_TRUE(stopped->accepted_through().has_value());
  ASSERT_TRUE(stopped->published_through().has_value());
  EXPECT_EQ(checked_value(stopped->accepted_through()), accepted.update_sequence);
  EXPECT_EQ(checked_value(stopped->published_through()), accepted.update_sequence);
}

// Result lookup and terminal waiting intentionally expose different zero/absent errors, while a
// restricted capability must never distinguish an absent sequence from a hidden one. Detach
// preserves retained terminal evidence but changes unrestricted missing-result errors to closed.
// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, ResultWaitAndDisclosureMatrixRemainsExactAfterDetach) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_parameter_coordinator(plan);
  auto& coordinator = *direct.coordinator;
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult accepted = coordinator.submit(patch);
  ASSERT_TRUE(accepted.committed);

  ParamAccessPolicy hidden_policy = full_local_parameter_policy();
  OperatorRef hidden_owner = owner;
  ++hidden_owner.id;
  hidden_policy.allowed_scope = ParamScopeFilter::for_operator(hidden_owner);
  constexpr ParamUpdateSequence kUnknownSequence = 999U;

  // Key setup: inspect zero, absent, hidden, and accepted-but-nonterminal identities before
  // terminalization, then repeat the retained/absent projections after detach.
  const auto zero_result = coordinator.result(0U);
  ASSERT_FALSE(zero_result);
  EXPECT_EQ(zero_result.error().code, ErrorCode::kNotFound);
  const auto zero_wait = coordinator.await(0U, 0ns);
  ASSERT_FALSE(zero_wait);
  EXPECT_EQ(zero_wait.error().code, ErrorCode::kInvalidArgument);
  const auto absent = coordinator.await(kUnknownSequence, 0ns);
  ASSERT_FALSE(absent);
  EXPECT_EQ(absent.error().code, ErrorCode::kNotFound);
  const auto hidden_absent = coordinator.await_authorized(kUnknownSequence, 0ns, hidden_policy);
  ASSERT_FALSE(hidden_absent);
  EXPECT_EQ(hidden_absent.error().code, ErrorCode::kPermissionDenied);
  const auto hidden_present =
      coordinator.await_authorized(accepted.update_sequence, 0ns, hidden_policy);
  ASSERT_FALSE(hidden_present);
  EXPECT_EQ(hidden_present.error().code, ErrorCode::kPermissionDenied);
  const auto pending = coordinator.await(accepted.update_sequence, 0ns);
  ASSERT_FALSE(pending);
  EXPECT_EQ(pending.error().code, ErrorCode::kTimeout);

  coordinator.terminalize_unpublished_at_run_end();
  const auto terminal = coordinator.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(terminal);
  EXPECT_EQ(terminal->state, ParamUpdateState::kCanceled);
  coordinator.begin_detach();

  // Key assertion: retained terminal evidence remains observable after detach, unrestricted
  // absent queries become kClosed, and restricted absent queries remain kPermissionDenied so the
  // capability cannot use guessed sequence numbers as an existence oracle.
  const auto retained = coordinator.await(accepted.update_sequence, 0ns);
  ASSERT_TRUE(retained);
  EXPECT_EQ(retained->state, ParamUpdateState::kCanceled);
  const auto closed_result = coordinator.result(kUnknownSequence);
  ASSERT_FALSE(closed_result);
  EXPECT_EQ(closed_result.error().code, ErrorCode::kClosed);
  const auto closed_wait = coordinator.await(kUnknownSequence, 0ns);
  ASSERT_FALSE(closed_wait);
  EXPECT_EQ(closed_wait.error().code, ErrorCode::kClosed);
  const auto hidden_after_detach =
      coordinator.await_authorized(kUnknownSequence, 0ns, hidden_policy);
  ASSERT_FALSE(hidden_after_detach);
  EXPECT_EQ(hidden_after_detach.error().code, ErrorCode::kPermissionDenied);
}

// Capacity preflight and reservation must agree on the exact reusable ledger population. Empty
// slots remain preferable to acknowledged terminal evidence; once full, the oldest acknowledged
// non-Published sequence is recycled first.
// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, ResultLedgerReusesOldestAcknowledgedTerminalSlot) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto op = std::make_unique<test::RuntimeParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto capacity = checked_value(plan.parameter_runtime_capacity());
  capacity.audit_capacity = 2U;
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U}, capacity);
  ASSERT_TRUE(coordinator_result);
  auto coordinator = std::move(coordinator_result).value();
  const auto submit_no_change = [&] {
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = owner, .field = field.field},
            .value = ParamValue::from(std::int32_t{7}),
        },
    };
    return coordinator->submit(patch);
  };

  // Key setup: fill both ledger slots with terminal no-change results and acknowledge them in
  // sequence order, leaving no empty slot for the third submission.
  const ParamBatchUpdateResult first = submit_no_change();
  const ParamBatchUpdateResult second = submit_no_change();
  ASSERT_TRUE(first.committed);
  ASSERT_TRUE(second.committed);
  ASSERT_TRUE(coordinator->result(first.update_sequence));
  ASSERT_TRUE(coordinator->result(second.update_sequence));
  const ParamBatchUpdateResult third = submit_no_change();
  ASSERT_TRUE(third.committed);

  // Key assertion: successful admission proves preflight and reservation selected the same
  // eligible population; only the oldest acknowledged sequence was evicted.
  const auto evicted = coordinator->result(first.update_sequence);
  ASSERT_FALSE(evicted);
  EXPECT_EQ(evicted.error().code, ErrorCode::kNotFound);
  EXPECT_TRUE(coordinator->result(second.update_sequence));
  EXPECT_TRUE(coordinator->result(third.update_sequence));
}

// REQ: PARAM-018
// REQ: PARAM-021
TEST(ParameterUpdateCoordinatorTest, LifecycleCandidateLedgerTracksInjectedCoalescingOutcomes) {
  const ExecutionPlan plan = test::make_two_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_parameters().size(), 2U);
  auto first_op = std::make_unique<test::LifecycleParamProbeOperator>();
  auto second_op = std::make_unique<test::LifecycleParamProbeOperator>();
  const std::array bindings{
      make_direct_parameter_binding(plan, *first_op, 101U, 1U),
      make_direct_parameter_binding(plan, *second_op, 101U, 2U),
  };
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      bindings, checked_value(plan.parameter_runtime_capacity()));
  ASSERT_TRUE(coordinator_result);
  auto coordinator = std::move(coordinator_result).value();
  FakeLifecycleSubmitter lifecycle{
      .outcome = queued_lifecycle_outcome(1U, 11U),
  };
  ASSERT_TRUE(coordinator->bind_lifecycle_submitter(lifecycle.binding()));

  const auto field_for = [&](OperatorInstanceId owner) -> const ParameterFieldPlanRecord& {
    const auto found = std::find_if(plan.parameter_fields().begin(),
                                    plan.parameter_fields().end(),
                                    [owner](const auto& field) { return field.owner.id == owner; });
    if (found == plan.parameter_fields().end()) {
      throw std::runtime_error{"lifecycle field missing"};
    }
    return *found;
  };
  const auto submit = [&](OperatorInstanceId id, std::int32_t value) {
    const auto& field = field_for(id);
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = run_owner(field), .field = field.field},
            .value = ParamValue::from(value),
        },
    };
    return coordinator->submit(patch);
  };

  const ParamBatchUpdateResult first = submit(1U, 42);
  ASSERT_TRUE(first.committed);
  lifecycle.outcome = queued_lifecycle_outcome(1U, 12U);
  lifecycle.outcome.coalesced_into = 1U;
  const ParamBatchUpdateResult second = submit(2U, 52);
  ASSERT_TRUE(second.committed);
  lifecycle.outcome = queued_lifecycle_outcome(1U, 13U);
  lifecycle.outcome.coalesced_into = 1U;
  lifecycle.outcome.superseded_transaction = 1U;
  const ParamBatchUpdateResult replacement = submit(1U, 43);
  ASSERT_TRUE(replacement.committed);
  EXPECT_EQ(detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*coordinator),
            14U);

  auto first_latest = coordinator->pin_lifecycle_generation(run_owner(field_for(1U)), 2U);
  ASSERT_TRUE(first_latest);
  EXPECT_EQ(
      static_cast<const test::LifecycleParamProbeParams*>(first_latest->header()->value)->value,
      43);
  auto second_latest = coordinator->pin_lifecycle_generation(run_owner(field_for(2U)), 1U);
  ASSERT_TRUE(second_latest);
  EXPECT_EQ(
      static_cast<const test::LifecycleParamProbeParams*>(second_latest->header()->value)->value,
      52);
  coordinator->release(*first_latest);
  coordinator->release(*second_latest);

  lifecycle.reject = true;
  lifecycle.error = Error{ErrorCode::kResourceExhausted};
  const ParamBatchUpdateResult rejected_delta = submit(2U, 53);
  EXPECT_FALSE(rejected_delta.committed);
  EXPECT_EQ(rejected_delta.status, ParamUpdateStatus::kQueueFull);
  EXPECT_EQ(detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*coordinator),
            14U);
  auto preserved = coordinator->pin_lifecycle_generation(run_owner(field_for(2U)), 1U);
  ASSERT_TRUE(preserved);
  EXPECT_EQ(static_cast<const test::LifecycleParamProbeParams*>(preserved->header()->value)->value,
            52);
  coordinator->release(*preserved);

  lifecycle.reject = false;
  lifecycle.outcome = queued_lifecycle_outcome(2U, 14U);
  const ParamBatchUpdateResult successor = submit(1U, 44);
  ASSERT_TRUE(successor.committed);
  EXPECT_EQ(successor.target_generation, 3U);
  EXPECT_EQ(detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*coordinator),
            21U);
  auto active = coordinator->pin_lifecycle_generation(run_owner(field_for(1U)), 2U);
  auto queued_successor = coordinator->pin_lifecycle_generation(run_owner(field_for(1U)), 3U);
  ASSERT_TRUE(active);
  ASSERT_TRUE(queued_successor);
  EXPECT_EQ(static_cast<const test::LifecycleParamProbeParams*>(active->header()->value)->value,
            43);
  EXPECT_EQ(
      static_cast<const test::LifecycleParamProbeParams*>(queued_successor->header()->value)->value,
      44);
  coordinator->release(*active);
  coordinator->release(*queued_successor);
  coordinator->terminalize_unpublished_at_run_end();
  EXPECT_EQ(detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*coordinator),
            0U);
}

// Exercise the real control-state compatibility domain and transaction kernel rather than the
// fake submitter used by the coordinator-only ledger test above.
// REQ: PARAM-018
// REQ: PARAM-021
TEST(ParameterUpdateCoordinatorTest, LifecycleControlCoalescesOwnersAndExecutesUnionClosure) {
  const ExecutionPlan plan = test::make_two_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto first_op = std::make_unique<test::LifecycleParamProbeOperator>();
  auto second_op = std::make_unique<test::LifecycleParamProbeOperator>();
  const std::array bindings{
      make_direct_parameter_binding(plan, *first_op, 101U, 1U),
      make_direct_parameter_binding(plan, *second_op, 101U, 2U),
  };
  auto coordinator_result = detail::ParameterUpdateCoordinator::create(
      bindings, checked_value(plan.parameter_runtime_capacity()));
  ASSERT_TRUE(coordinator_result);
  auto coordinator = std::move(coordinator_result).value();
  auto control = std::make_shared<detail::LifecycleControlState>(
      detail::ExecutionPlanAccess::retain(plan), 101U);
  ASSERT_TRUE(control->enabled());
  ASSERT_TRUE(control->open_ingress());
  ASSERT_TRUE(coordinator->bind_lifecycle_submitter(control->parameter_submit_binding()));

  const auto field_for = [&](OperatorInstanceId owner) -> const ParameterFieldPlanRecord& {
    const auto found = std::find_if(plan.parameter_fields().begin(),
                                    plan.parameter_fields().end(),
                                    [owner](const auto& field) { return field.owner.id == owner; });
    if (found == plan.parameter_fields().end()) {
      throw std::runtime_error{"lifecycle field missing"};
    }
    return *found;
  };
  const auto submit = [&](OperatorInstanceId id, std::int32_t value) {
    const auto& field = field_for(id);
    const std::array patch{
        ParamPatch{
            .target = ParamAddress{.operator_instance = run_owner(field), .field = field.field},
            .value = ParamValue::from(value),
        },
    };
    return coordinator->submit(patch);
  };

  // Key setup: submit one update per operator so the real LifecycleControlState coalesces
  // them into a single restart transaction that covers both owners.
  const ParamBatchUpdateResult first = submit(1U, 42);
  const ParamBatchUpdateResult second = submit(2U, 52);
  ASSERT_TRUE(first.committed);
  ASSERT_TRUE(second.committed);
  const auto first_accepted = coordinator->result(first.update_sequence);
  const auto second_accepted = coordinator->result(second.update_sequence);
  ASSERT_TRUE(first_accepted);
  ASSERT_TRUE(second_accepted);
  // Key assertion: both updates share the same lifecycle_transaction_id, confirming
  // the union-closure restart was coalesced into a single transaction.
  ASSERT_NE(first_accepted->lifecycle_transaction_id, 0U);
  EXPECT_EQ(second_accepted->lifecycle_transaction_id, first_accepted->lifecycle_transaction_id);
  EXPECT_EQ(second_accepted->coalesced_into_transaction, first_accepted->lifecycle_transaction_id);

  struct ExecutionCapture {
    detail::ParameterUpdateCoordinator* coordinator{};
    std::array<OperatorInstanceId, 2> members{};
    std::array<detail::LifecycleRestartParameterCandidate, 2> candidates{};
    std::size_t member_count{};
    std::size_t candidate_count{};
  } capture{.coordinator = coordinator.get()};
  const std::atomic_bool stop_requested{false};
  ASSERT_TRUE(control->process_pending(
      [](void* state, const detail::LifecycleRestartExecutionRequest& request) noexcept {
        auto& capture = *static_cast<ExecutionCapture*>(state);
        capture.member_count = request.members.size();
        capture.candidate_count = request.parameter_candidates.size();
        if (request.members.size() > capture.members.size() ||
            request.parameter_candidates.size() > capture.candidates.size()) {
          return detail::LifecycleRestartExecutionOutcome{};
        }
        std::copy(request.members.begin(), request.members.end(), capture.members.begin());
        std::copy(request.parameter_candidates.begin(),
                  request.parameter_candidates.end(),
                  capture.candidates.begin());
        for (const LifecycleTransactionState lifecycle_state : {
                 LifecycleTransactionState::kQuiescing,
                 LifecycleTransactionState::kRestarting,
                 LifecycleTransactionState::kPublished,
             }) {
          for (const auto& candidate : request.parameter_candidates) {
            if (!capture.coordinator->transition_lifecycle_candidate(
                    candidate.owner, candidate.generation, request.transaction, lifecycle_state)) {
              return detail::LifecycleRestartExecutionOutcome{};
            }
          }
        }
        return detail::LifecycleRestartExecutionOutcome{
            .disposition = detail::LifecycleRestartExecutionDisposition::kPublished,
        };
      },
      &capture,
      stop_requested));

  ASSERT_EQ(capture.member_count, 2U);
  EXPECT_EQ(capture.members, (std::array<OperatorInstanceId, 2>{1U, 2U}));
  ASSERT_EQ(capture.candidate_count, 2U);
  EXPECT_EQ(capture.candidates[0].owner.id, 1U);
  EXPECT_EQ(capture.candidates[1].owner.id, 2U);
  const auto first_published = coordinator->result(first.update_sequence);
  const auto second_published = coordinator->result(second.update_sequence);
  ASSERT_TRUE(first_published);
  ASSERT_TRUE(second_published);
  EXPECT_EQ(first_published->state, ParamUpdateState::kPublished);
  EXPECT_EQ(second_published->state, ParamUpdateState::kPublished);
  auto first_value = coordinator->pin_published(run_owner(field_for(1U)));
  auto second_value = coordinator->pin_published(run_owner(field_for(2U)));
  ASSERT_TRUE(first_value);
  ASSERT_TRUE(second_value);
  EXPECT_EQ(
      static_cast<const test::LifecycleParamProbeParams*>(first_value->header()->value)->value, 42);
  EXPECT_EQ(
      static_cast<const test::LifecycleParamProbeParams*>(second_value->header()->value)->value,
      52);
  coordinator->release(*first_value);
  coordinator->release(*second_value);
  control->finalize_close();
}

// REQ: PARAM-018
// REQ: PARAM-021
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, LifecycleKernelRejectionAbortsEveryLocalReservation) {
  const ExecutionPlan plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_lifecycle_parameter_coordinator(plan);
  FakeLifecycleSubmitter lifecycle{
      .error = Error{ErrorCode::kResourceExhausted},
      .reject = true,
  };
  ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult rejected = direct.coordinator->submit(patch);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.status, ParamUpdateStatus::kQueueFull);
  EXPECT_EQ(rejected.update_sequence, 0U);
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      0U);
  auto published = direct.coordinator->pin_published(owner);
  ASSERT_TRUE(published);
  EXPECT_EQ(static_cast<const test::LifecycleParamProbeParams*>(published->header()->value)->value,
            7);
  direct.coordinator->release(*published);

  lifecycle.reject = false;
  lifecycle.outcome = queued_lifecycle_outcome(1U, 11U);
  const std::array accepted_patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{99}),
      },
  };
  EXPECT_TRUE(direct.coordinator->submit(accepted_patch).committed);
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      7U);
  direct.coordinator->terminalize_unpublished_at_run_end();
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      0U);
}

// REQ: PARAM-018
// REQ: PARAM-021
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, MalformedLifecycleSuccessIsCanceledBeforeLocalAbort) {
  const ExecutionPlan plan = test::make_lifecycle_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  const OperatorRef owner = run_owner(field);
  auto direct = make_direct_lifecycle_parameter_coordinator(plan);
  FakeLifecycleSubmitter lifecycle{
      .outcome =
          detail::ParameterLifecycleSubmitOutcome{
              .receipt = LifecycleRestartReceipt{.receipt_id = 11U},
              .state = LifecycleTransactionState::kQueued,
              .transaction = 0U,
          },
  };
  ASSERT_TRUE(direct.coordinator->bind_lifecycle_submitter(lifecycle.binding()));
  const std::array patch{
      ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = field.field},
          .value = ParamValue::from(std::int32_t{42}),
      },
  };
  const ParamBatchUpdateResult rejected = direct.coordinator->submit(patch);
  EXPECT_FALSE(rejected.committed);
  EXPECT_EQ(rejected.update_sequence, 0U);
  EXPECT_EQ(lifecycle.cancel_calls, 1U);
  EXPECT_EQ(
      detail::ParameterUpdateCoordinatorTestAccess::reserved_event_sequences(*direct.coordinator),
      0U);
  auto published = direct.coordinator->pin_published(owner);
  ASSERT_TRUE(published);
  EXPECT_EQ(static_cast<const test::LifecycleParamProbeParams*>(published->header()->value)->value,
            7);
  direct.coordinator->release(*published);
}

// REQ: PARAM-020
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, StopIngressRequestDoesNotWaitForMutationLock) {
  test::RuntimeProbeState::reset();
  test::RuntimeParamProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const ParameterFieldPlanRecord field = runtime_value_field(plan);
  auto direct = make_direct_parameter_coordinator(plan);
  auto& coordinator = *direct.coordinator;
  const ParamAddress address{.operator_instance = run_owner(field), .field = field.field};
  const std::array patches{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };
  const ParamBatchUpdateResult accepted = coordinator.submit(patches);
  ASSERT_TRUE(accepted.committed);

  std::atomic_bool entered{};
  std::atomic_bool release{};
  std::thread holder{[&] {
    detail::ParameterUpdateCoordinatorTestAccess::hold_mutation_lock(coordinator, entered, release);
  }};
  while (!entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  coordinator.request_close_ingress();
  EXPECT_FALSE(release.load(std::memory_order_relaxed));
  const ParamInlinePatch compute_patch{
      .target = address,
      .value = ParamValue::from(std::int32_t{43}),
  };
  const ParamSubmitResult after_stop = coordinator.try_submit_from_compute(&compute_patch, 1U);
  EXPECT_EQ(after_stop.status, ParamUpdateStatus::kClosed);
  release.store(true, std::memory_order_release);
  holder.join();

  const ParamStopFenceResult fence = coordinator.close_ingress();
  EXPECT_FALSE(fence.already_closed);
  EXPECT_EQ(fence.terminal_update_sequence, accepted.update_sequence);
  const ParamStopFenceResult repeated = coordinator.close_ingress();
  EXPECT_TRUE(repeated.already_closed);
  EXPECT_EQ(repeated.terminal_update_sequence, fence.terminal_update_sequence);
}

// REQ: PARAM-019
// REQ: PARAM-023
TEST(ParameterUpdateCoordinatorTest, EmptyActivationMaintenanceDoesNotWaitForMutationLock) {
  const ExecutionPlan plan = test::make_runtime_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const OperatorRef owner = run_owner(runtime_value_field(plan));
  auto direct = make_direct_parameter_coordinator(plan);
  auto& coordinator = *direct.coordinator;
  auto generation_zero = coordinator.pin_published(owner);
  ASSERT_TRUE(generation_zero);

  std::atomic_bool entered{};
  std::atomic_bool release{};
  std::thread holder{[&] {
    detail::ParameterUpdateCoordinatorTestAccess::hold_mutation_lock(coordinator, entered, release);
  }};
  while (!entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  auto maintenance = std::async(std::launch::async, [&] {
    const auto published =
        coordinator.publish_next_at_callback_boundary(owner, coordinator.publication_frontier());
    const std::size_t drained = coordinator.drain_compute_ingress();
    const auto observed = coordinator.commit_activation_observation(owner, *generation_zero);
    return !published.has_value() && drained == 0U && observed.has_value();
  });
  const std::future_status status = maintenance.wait_for(1s);
  release.store(true, std::memory_order_release);
  holder.join();

  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(maintenance.get());
  coordinator.release(*generation_zero);
}

#if GTEST_HAS_DEATH_TEST
namespace {
enum class FatalLedgerGuard {
  kReplace,
  kRelease,
  kMissingResult,
  kExhaustedEvent,
  kLostPublication
};

// Prepare valid accepted transitions first, then corrupt only the evidence checked by each guard.
void exercise_fatal_ledger_guard(FatalLedgerGuard guard, bool fail_formatting) {
  using Access = detail::ParameterUpdateCoordinatorTestAccess;
  const ExecutionPlan plan = test::make_mixed_mutability_param_test_plan();
  ASSERT_TRUE(plan.ok());
  const auto& runtime_field = parameter_field(plan, "runtime_value");
  const auto& lifecycle_field = parameter_field(plan, "lifecycle_value");
  const auto owner = run_owner(runtime_field);
  auto op = std::make_unique<test::MixedMutabilityParamProbeOperator>();
  const auto binding = make_direct_parameter_binding(plan, *op);
  auto coordinator = checked_value(detail::ParameterUpdateCoordinator::create(
      std::span<const detail::ParameterRuntimeOperatorBinding>{&binding, 1U},
      checked_value(plan.parameter_runtime_capacity())));
  FakeLifecycleSubmitter lifecycle{.outcome = queued_lifecycle_outcome(1U, 11U)};
  ASSERT_TRUE(coordinator->bind_lifecycle_submitter(lifecycle.binding()));
  ParamGeneration lifecycle_generation{};
  if (guard != FatalLedgerGuard::kReplace && guard != FatalLedgerGuard::kRelease) {
    const std::array patch{
        ParamPatch{.target = ParamAddress{.operator_instance = owner, .field = runtime_field.field},
                   .value = ParamValue::from(std::int32_t{42})}};
    const auto update = coordinator->submit(patch);
    ASSERT_TRUE(update.committed);
    if (guard == FatalLedgerGuard::kMissingResult) {
      Access::set_result_event_reservations(*coordinator, update.update_sequence, 0U);
    } else if (guard == FatalLedgerGuard::kExhaustedEvent) {
      Access::set_event_sequence(*coordinator, std::numeric_limits<ParamEventSequence>::max());
    } else {
      ASSERT_TRUE(coordinator->publish_next_at_callback_boundary(
          owner, coordinator->publication_frontier()));
      const std::array lifecycle_patch{ParamPatch{
          .target = ParamAddress{.operator_instance = owner, .field = lifecycle_field.field},
          .value = ParamValue::from(std::int32_t{84})}};
      const auto next = coordinator->submit(lifecycle_patch);
      ASSERT_TRUE(next.committed);
      lifecycle_generation = next.target_generation;
      ASSERT_TRUE(coordinator->transition_lifecycle_candidate(
          owner, lifecycle_generation, 1U, LifecycleTransactionState::kQuiescing));
      ASSERT_TRUE(coordinator->transition_lifecycle_candidate(
          owner, lifecycle_generation, 1U, LifecycleTransactionState::kRestarting));
      Access::invalidate_result(*coordinator, update.update_sequence);
    }
  }
  std::set_terminate([] { std::_Exit(std::current_exception() ? 91 : 90); });
  std::optional<test::ScopedAllocationFailure> failure;
  if (fail_formatting)
    failure.emplace(0U);
  switch (guard) {
    case FatalLedgerGuard::kReplace:
      Access::replace_event_reservation(*coordinator, 1U);
      break;
    case FatalLedgerGuard::kRelease:
      Access::release_event_reservations(*coordinator, 1U);
      break;
    case FatalLedgerGuard::kMissingResult:
    case FatalLedgerGuard::kExhaustedEvent:
      static_cast<void>(coordinator->publish_next_at_callback_boundary(
          owner, coordinator->publication_frontier()));
      break;
    case FatalLedgerGuard::kLostPublication:
      static_cast<void>(coordinator->transition_lifecycle_candidate(
          owner, lifecycle_generation, 1U, LifecycleTransactionState::kPublished));
      break;
  }
  std::_Exit(92);
}

constexpr std::array kFatalLedgerGuards{
    std::pair{FatalLedgerGuard::kReplace, "Parameter event reservation replacement exceeded"},
    std::pair{FatalLedgerGuard::kRelease, "Parameter event reservation release exceeded"},
    std::pair{FatalLedgerGuard::kMissingResult, "Parameter transition lost its preaccepted"},
    std::pair{FatalLedgerGuard::kExhaustedEvent, "Parameter transition exhausted a preaccepted"},
    std::pair{FatalLedgerGuard::kLostPublication, "Unobserved parameter publication lost"}};
}  // namespace

// REQ: EXPECTED-032
TEST(ParameterDiagnosticDeathTest, FatalLedgerMessagesBypassAutomaticPolicy) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const auto& [guard, message] : kFatalLedgerGuards) {
    for (const int policy : {0, 1, 2}) {
      EXPECT_EXIT(
          {
            unsetenv("HOLOSCAN_LOG_LEVEL");
            unsetenv("HOLOSCAN_LOG_FORMAT");
            unsetenv("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING");
            if (policy == 2)
              setenv("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING", "OFF", 1);
            set_automatic_diagnostic_logging_enabled(policy != 1);
            set_log_level(LogLevel::ERROR);
            exercise_fatal_ledger_guard(guard, false);
            std::_Exit(93);
          },
          testing::ExitedWithCode(90),
          message);
    }
  }
}

// REQ: EXPECTED-032
TEST(ParameterDiagnosticDeathTest, FatalLedgerTerminationSurvivesFilteringAndFormattingFailure) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  for (const auto& [guard, message] : kFatalLedgerGuards) {
    static_cast<void>(message);
    for (const bool fail_formatting : {false, true}) {
      EXPECT_EXIT(
          {
            unsetenv("HOLOSCAN_LOG_LEVEL");
            unsetenv("HOLOSCAN_LOG_FORMAT");
            unsetenv("HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING");
            set_automatic_diagnostic_logging_enabled(true);
            set_log_level(fail_formatting ? LogLevel::ERROR : LogLevel::OFF);
            exercise_fatal_ledger_guard(guard, fail_formatting);
            std::_Exit(93);
          },
          testing::ExitedWithCode(90),
          "^$");
    }
  }
}
#endif

}  // namespace
}  // namespace holoscan
