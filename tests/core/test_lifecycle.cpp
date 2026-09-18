/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/port.hpp>

#include <core/lifecycle_context_internal.hpp>
#include <core/lifecycle_context_test_access.hpp>
#include <core/lifecycle_kernel.hpp>
#include <core/lifecycle_parameter_view.hpp>
#include <core/readiness_source_internal.hpp>
#include <core/source_id_internal.hpp>

namespace holoscan {
namespace {

// Trivial parameter struct used to verify typed param access through LifecycleContext.
// The context must surface the exact address of the params object, not a copy.
struct LifecycleTestParams {
  std::uint32_t value{};
};

template <typename T>
concept HasLifecycleRestartSubmission = requires(T& value) { value.request_restart(); };

template <typename T>
concept HasLifecycleAwait = requires(T& value) { value.await(); };

// Service callback that stamps run_generation on a plan-bound OperatorRef.
// Simulates the runtime binding step that the real scheduler performs when
// a lifecycle callback requests a restart target reference.
expected<OperatorRef, Error> bind_target(void* opaque, OperatorRef target) noexcept {
  const auto run_generation = *static_cast<const RunGenerationId*>(opaque);
  if (!target.plan_bound()) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  target.run_generation = run_generation;
  return target;
}

// Stub service callbacks that always fail; used to verify that LifecycleContext
// correctly propagates service-layer errors to the caller without crashing.
expected<ClockRef, Error> reject_clock(void*, const ClockKey&) noexcept {
  return make_unexpected(Error{ErrorCode::kNotFound});
}

expected<NotificationSender, Error> reject_sender(void*, NotificationSource&) noexcept {
  return make_unexpected(Error{ErrorCode::kNotReady});
}

struct LifecycleRateServiceProbe {
  bool bind_called{};
  bool resolve_called{};
};

expected<void, Error> record_rate_bind(void* opaque, const void*, Rate) noexcept {
  static_cast<LifecycleRateServiceProbe*>(opaque)->bind_called = true;
  return {};
}

expected<PortRateView, Error> record_rate_resolve(void* opaque, const void*, bool) noexcept {
  static_cast<LifecycleRateServiceProbe*>(opaque)->resolve_called = true;
  return PortRateView{};
}

// Tracks retain/release/post invocations on a NotificationSender to verify
// that the sender's copy semantics call retain once and release once per copy lifetime.
struct NotificationControl {
  std::uint32_t retains{};
  std::uint32_t releases{};
  std::uint32_t posts{};
};

void retain_notification(void* state) noexcept {
  ++static_cast<NotificationControl*>(state)->retains;
}

void release_notification(void* state) noexcept {
  ++static_cast<NotificationControl*>(state)->releases;
}

expected<void, Error> post_notification(void* state) noexcept {
  ++static_cast<NotificationControl*>(state)->posts;
  return {};
}

// Associates a specific NotificationSource* with a NotificationControl* so that
// bind_sender can validate source identity before handing out a live sender.
struct NotificationService {
  NotificationSource* source{};
  NotificationControl* control{};
};

// Validates that source is the expected bound source, then constructs a NotificationSender
// backed by the control's retain/release/post callbacks.
expected<NotificationSender, Error> bind_sender(void* state, NotificationSource& source) noexcept {
  auto& service = *static_cast<NotificationService*>(state);
  if (service.source != &source || !static_cast<NotificationSourceRef>(source)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return detail::ReadinessSourceAccess::make_sender(
      service.control, &retain_notification, &release_notification, &post_notification);
}

// Log sink that unconditionally throws; used to verify that emit_lifecycle_log_noexcept
// swallows all exceptions and does not propagate them to the caller.
void throw_from_log_sink(void*, const detail::LifecycleLogRecord&) {
  throw std::runtime_error("injected lifecycle log failure");
}

// Collects up to four LifecycleLogRecord structs for field-level assertion in
// LoggingSinkPreservesStructuralEvidenceAndCannotAffectControlFlow.
struct LifecycleLogProbe {
  std::array<detail::LifecycleLogRecord, 4> records{};
  std::size_t size{};
};

void record_lifecycle_log(void* state, const detail::LifecycleLogRecord& record) {
  auto& probe = *static_cast<LifecycleLogProbe*>(state);
  if (probe.size < probe.records.size()) {
    probe.records[probe.size++] = record;
  }
}

// REQ: LIFECYCLE-001
TEST(LifecycleValues, StagesAndBarriersAreClosedAndPortable) {
  constexpr std::array stages{
      LifecycleStage::kConfigure,
      LifecycleStage::kDiscover,
      LifecycleStage::kAllocate,
      LifecycleStage::kWarmUp,
      LifecycleStage::kArm,
      LifecycleStage::kStart,
      LifecycleStage::kStop,
      LifecycleStage::kRelease,
  };
  for (std::size_t index = 0; index < stages.size(); ++index) {
    EXPECT_EQ(static_cast<std::uint8_t>(stages[index]), index);
    EXPECT_TRUE(is_valid_lifecycle_stage(stages[index]));
    EXPECT_EQ(is_lifecycle_forward_stage(stages[index]), index <= 5U);
  }
  const auto unknown_stage = std::bit_cast<LifecycleStage>(std::uint8_t{255U});
  EXPECT_FALSE(is_valid_lifecycle_stage(unknown_stage));
  EXPECT_FALSE(is_lifecycle_forward_stage(unknown_stage));
  constexpr std::array phases{
      LifecycleTransitionPhase::kNone,
      LifecycleTransitionPhase::kStartupForward,
      LifecycleTransitionPhase::kPreRestartTeardown,
      LifecycleTransitionPhase::kRestartForward,
      LifecycleTransitionPhase::kForwardRollback,
      LifecycleTransitionPhase::kRestoreOld,
      LifecycleTransitionPhase::kShutdownTeardown,
  };
  for (std::size_t index = 0; index < phases.size(); ++index) {
    EXPECT_EQ(static_cast<std::uint8_t>(phases[index]), index);
  }
  static_assert(std::is_trivially_copyable_v<LifecycleStageDecision>);
  static_assert(std::is_trivially_copyable_v<LifecycleRollbackRecord>);
  static_assert(sizeof(LifecycleContext) == sizeof(void*));
}

// REQ: LIFECYCLE-005
TEST(LifecycleContext, BindsExactPhaseGenerationAndColdServices) {
  // Key setup: context state wires a real bind_target, a clock rejector, and a sender
  // rejector so each service path can be independently exercised in the same test.
  LifecycleTestParams params{.value = 42U};
  RunGenerationId run_generation = 91U;
  detail::LifecycleContextState state{
      .params = &params,
      .params_type = param_type_descriptor<LifecycleTestParams>().local_type_token,
      .params_generation = 17U,
      .phase = LifecycleTransitionPhase::kRestartForward,
      .service_state = &run_generation,
      .resolve_clock = &reject_clock,
      .notification_sender = &reject_sender,
      .restart_target = &bind_target,
  };
  const LifecycleContext context = detail::LifecycleContextTestAccess::context(state);

  EXPECT_EQ(&context.params<LifecycleTestParams>(), &params);
  EXPECT_EQ(context.params<LifecycleTestParams>().value, 42U);
  EXPECT_EQ(context.params_generation(), 17U);
  EXPECT_EQ(context.transition_phase(), LifecycleTransitionPhase::kRestartForward);
  EXPECT_FALSE(context.cancellation().stop_requested());
  state.cancellation_requested.store(true, std::memory_order_release);
  EXPECT_TRUE(context.cancellation().stop_requested());

  const auto target =
      context.restart_target(OperatorRef{.id = 3U, .plan_generation = 11U, .run_generation = 0U});
  ASSERT_TRUE(target);
  EXPECT_EQ(target->run_generation, run_generation);
  const auto clock = context.resolve_clock(ClockKey::default_clock());
  ASSERT_FALSE(clock);
  EXPECT_EQ(clock.error().code, ErrorCode::kNotFound);
  NotificationSource source;
  const auto sender = context.notification_sender(source);
  ASSERT_FALSE(sender);
  EXPECT_EQ(sender.error().code, ErrorCode::kNotReady);

  // Key assertion: a copied NotificationSender increments retains once; both the
  // original and the copy must post successfully; destroy both → releases == 2.
  NotificationControl notification_control;
  NotificationSource bound_source;
  detail::ReadinessSourceAccess::bind(
      bound_source,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(
          detail::make_source_id(7U), 19U));
  NotificationService notification_service{
      .source = &bound_source,
      .control = &notification_control,
  };
  detail::LifecycleContextState sender_state{
      .phase = LifecycleTransitionPhase::kRestartForward,
      .service_state = &notification_service,
      .notification_sender = &bind_sender,
  };
  const LifecycleContext sender_context = detail::LifecycleContextTestAccess::context(sender_state);
  {
    const auto bound_sender = sender_context.notification_sender(bound_source);
    ASSERT_TRUE(bound_sender);
    EXPECT_TRUE(bound_sender->post());
    // Exercise the admitted sender-copy retention contract explicitly.
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    const NotificationSender copied_sender = *bound_sender;
    EXPECT_TRUE(copied_sender.post());
    EXPECT_EQ(notification_control.retains, 1U);
    NotificationSource foreign_source;
    const auto foreign_sender = sender_context.notification_sender(foreign_source);
    ASSERT_FALSE(foreign_sender);
    EXPECT_EQ(foreign_sender.error().code, ErrorCode::kInvalidArgument);
  }
  EXPECT_EQ(notification_control.posts, 2U);
  EXPECT_EQ(notification_control.releases, 2U);

  detail::LifecycleContextState inert;
  LifecycleContext inert_context = detail::LifecycleContextTestAccess::context(inert);
  EXPECT_EQ(inert_context.transition_phase(), LifecycleTransitionPhase::kNone);
  const auto inert_clock = inert_context.resolve_clock(ClockKey::default_clock());
  ASSERT_FALSE(inert_clock);
  EXPECT_EQ(inert_clock.error().code, ErrorCode::kLifecycleInvalidCallingContext);

  Output<std::uint32_t> inert_output{"output"};
  const auto inert_bind = inert_context.bind_nominal_rate(inert_output, Rate::hz(60U));
  ASSERT_FALSE(inert_bind);
  EXPECT_EQ(inert_bind.error().code, ErrorCode::kLifecycleInvalidCallingContext);
  EXPECT_EQ(inert_bind.error().message, "PORT_RATE_BIND_NOT_AUTHORIZED");
  const auto inert_rate = inert_context.rate(inert_output);
  ASSERT_FALSE(inert_rate);
  EXPECT_EQ(inert_rate.error().code, ErrorCode::kLifecycleInvalidCallingContext);
  EXPECT_EQ(inert_rate.error().message, "PORT_RATE_VIEW_INVALID");

  LifecycleRateServiceProbe rate_service;
  detail::LifecycleContextState rate_state{
      .rate_state = &rate_service,
      .bind_nominal_rate = &record_rate_bind,
      .resolve_rate = &record_rate_resolve,
  };
  LifecycleContext rate_context = detail::LifecycleContextTestAccess::context(rate_state);
  const auto null_bind = detail::LifecycleContextTestAccess::bind_nominal_rate_port(
      rate_context, nullptr, Rate::hz(60U));
  ASSERT_FALSE(null_bind);
  EXPECT_EQ(null_bind.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(null_bind.error().message, "PORT_RATE_BIND_NOT_AUTHORIZED");
  const auto null_rate = detail::LifecycleContextTestAccess::rate_port(rate_context, nullptr, true);
  ASSERT_FALSE(null_rate);
  EXPECT_EQ(null_rate.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(null_rate.error().message, "PORT_RATE_VIEW_INVALID");
  EXPECT_FALSE(rate_service.bind_called);
  EXPECT_FALSE(rate_service.resolve_called);
}

// REQ: LIFECYCLE-005
TEST(LifecycleContext, PhaseSelectionUsesOneCompleteGeneration) {
  // Key setup: owner_candidate has an old_published (11), current_published (14),
  // private_candidate (12), and a creating_view in progress (generation 13).
  // Each transition phase must select the appropriate generation from this set.
  const detail::LifecycleParameterViewInputs owner_candidate{
      .old_published = 11U,
      .current_published = 14U,
      .private_candidate = 12U,
      .creating_view =
          detail::LifecycleParameterViewSelection{
              .kind = detail::LifecycleParameterViewKind::kPrivateCandidate,
              .generation = 13U,
          },
  };
  const auto expect_selection = [&](LifecycleTransitionPhase phase,
                                    detail::LifecycleParameterViewKind kind,
                                    ParamGeneration generation) {
    const auto selected = detail::select_lifecycle_parameter_view(phase, owner_candidate);
    ASSERT_TRUE(selected);
    EXPECT_EQ(*selected,
              (detail::LifecycleParameterViewSelection{.kind = kind, .generation = generation}));
  };
  expect_selection(LifecycleTransitionPhase::kStartupForward,
                   detail::LifecycleParameterViewKind::kStartupGenerationZero,
                   0U);
  expect_selection(LifecycleTransitionPhase::kPreRestartTeardown,
                   detail::LifecycleParameterViewKind::kOldPublished,
                   11U);
  expect_selection(LifecycleTransitionPhase::kRestartForward,
                   detail::LifecycleParameterViewKind::kPrivateCandidate,
                   12U);
  expect_selection(LifecycleTransitionPhase::kForwardRollback,
                   detail::LifecycleParameterViewKind::kPrivateCandidate,
                   13U);
  expect_selection(LifecycleTransitionPhase::kRestoreOld,
                   detail::LifecycleParameterViewKind::kOldPublished,
                   11U);
  expect_selection(LifecycleTransitionPhase::kShutdownTeardown,
                   detail::LifecycleParameterViewKind::kCurrentPublished,
                   14U);

  const auto outside =
      detail::select_lifecycle_parameter_view(LifecycleTransitionPhase::kNone, owner_candidate);
  ASSERT_FALSE(outside);
  EXPECT_EQ(outside.error(), detail::LifecycleParameterViewSelectionError::kOutsideCallback);

  // Key assertion: edge cases — missing private_candidate falls back to kRetainedCurrent;
  // missing creating_view during rollback returns kMissingCreatingView; out-of-range
  // phase byte returns kInvalidTransitionPhase.
  auto dependency_only = owner_candidate;
  dependency_only.private_candidate.reset();
  dependency_only.current_published = 12U;
  const auto retained = detail::select_lifecycle_parameter_view(
      LifecycleTransitionPhase::kRestartForward, dependency_only);
  ASSERT_TRUE(retained);
  EXPECT_EQ(retained->kind, detail::LifecycleParameterViewKind::kRetainedCurrent);
  EXPECT_EQ(retained->generation, 12U);

  dependency_only.creating_view.reset();
  const auto missing_creating = detail::select_lifecycle_parameter_view(
      LifecycleTransitionPhase::kForwardRollback, dependency_only);
  ASSERT_FALSE(missing_creating);
  EXPECT_EQ(missing_creating.error(),
            detail::LifecycleParameterViewSelectionError::kMissingCreatingView);

  const auto invalid = detail::select_lifecycle_parameter_view(
      std::bit_cast<LifecycleTransitionPhase>(std::uint8_t{255U}), owner_candidate);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error(), detail::LifecycleParameterViewSelectionError::kInvalidTransitionPhase);
}

// REQ: LIFECYCLE-007
TEST(LifecycleSurface, BoundsTransactionsAndExcludesForbiddenRuntimeAuthority) {
  static_assert(!HasLifecycleRestartSubmission<LifecycleContext>);
  static_assert(!HasLifecycleAwait<LifecycleContext>);
  EXPECT_EQ(static_cast<std::uint8_t>(LifecycleTransactionState::kRejected), 0U);
  EXPECT_EQ(static_cast<std::uint8_t>(LifecycleTransactionState::kAcceptedNoChange), 1U);
  EXPECT_EQ(static_cast<std::uint8_t>(LifecycleTransactionState::kFailedStopped), 9U);

  const LifecycleStageDecision decision{
      .transaction = 7U,
      .requested = LifecycleStage::kArm,
      .effective = LifecycleStage::kAllocate,
      .widening_reason = 1001U,
  };
  EXPECT_LT(static_cast<std::uint8_t>(decision.effective),
            static_cast<std::uint8_t>(decision.requested));
}

// REQ: LIFECYCLE-009
// REQ: LIFECYCLE-013
TEST(LifecycleSurface, LoggingSinkPreservesStructuralEvidenceAndCannotAffectControlFlow) {
  LifecycleLogProbe probe;
  const std::array records{
      detail::LifecycleLogRecord{.kind = detail::LifecycleLogEventKind::kTransitionBegin,
                                 .operation = detail::LifecycleLogOperation::kRestart},
      detail::LifecycleLogRecord{.kind = detail::LifecycleLogEventKind::kCallbackBegin,
                                 .operation = detail::LifecycleLogOperation::kRestart,
                                 .node_id = 17U,
                                 .stage = LifecycleStage::kWarmUp,
                                 .phase = LifecycleTransitionPhase::kRestartForward},
      detail::LifecycleLogRecord{.kind = detail::LifecycleLogEventKind::kCallbackEnd,
                                 .operation = detail::LifecycleLogOperation::kRestart,
                                 .node_id = 17U,
                                 .stage = LifecycleStage::kWarmUp,
                                 .phase = LifecycleTransitionPhase::kRestartForward,
                                 .status = LifecycleStatus::kRetryableFailure},
      detail::LifecycleLogRecord{
          .kind = detail::LifecycleLogEventKind::kTransitionEnd,
          .operation = detail::LifecycleLogOperation::kRestart,
          .disposition = detail::LifecycleExecutionDisposition::kRollbackFailed,
          .node_id = 17U,
          .stage = LifecycleStage::kWarmUp,
          .phase = LifecycleTransitionPhase::kRestartForward,
          .status = LifecycleStatus::kRetryableFailure},
  };
  for (const auto& record : records) {
    detail::emit_lifecycle_log_noexcept(&record_lifecycle_log, &probe, record);
  }
  ASSERT_EQ(probe.size, records.size());
  for (std::size_t index = 0U; index < records.size(); ++index) {
    EXPECT_EQ(probe.records[index].kind, records[index].kind);
    EXPECT_EQ(probe.records[index].operation, records[index].operation);
    EXPECT_EQ(probe.records[index].disposition, records[index].disposition);
    EXPECT_EQ(probe.records[index].node_id, records[index].node_id);
    EXPECT_EQ(probe.records[index].stage, records[index].stage);
    EXPECT_EQ(probe.records[index].phase, records[index].phase);
    EXPECT_EQ(probe.records[index].status, records[index].status);
  }

  EXPECT_NO_THROW(detail::emit_lifecycle_log_noexcept(
      &throw_from_log_sink,
      nullptr,
      {.kind = detail::LifecycleLogEventKind::kTransitionEnd,
       .operation = detail::LifecycleLogOperation::kRestart,
       .disposition = detail::LifecycleExecutionDisposition::kRollbackFailed}));
  EXPECT_NO_THROW(detail::emit_lifecycle_log_noexcept(nullptr, nullptr, records.front()));
}

}  // namespace
}  // namespace holoscan
