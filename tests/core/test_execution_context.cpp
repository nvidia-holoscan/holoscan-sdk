/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <type_traits>
#include <utility>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/tag.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/time_point.hpp>

#include "../../src/core/execution_context_internal.hpp"
#include "operator_test_fixture.hpp"

namespace holoscan {
namespace {

// Placeholder parameter struct for the type-safe params<T>() accessor on ExecutionContext.
// ContextParamsAlias verifies that a typedef resolves to the same type for the exact-match check.
struct ContextParams {
  std::uint32_t value{};
};

// Intentionally wrong parameter type; used by WrongParameterProbeOperator to trigger the terminal
// core handler when params<WrongContextParams>() is called on a ContextParams operator.
struct WrongContextParams {
  std::uint32_t value{};
};

using ContextParamsAlias = ContextParams;

template <typename Context, typename Params>
concept HasExactParams = requires(const Context& context) {
  { context.template params<Params>() } -> std::same_as<const Params&>;
};

template <typename Context>
concept HasTryParams =
    requires(const Context& context) { context.template try_params<ContextParams>(); };

template <typename Context>
concept HasReferenceTime = requires(const Context& context) { context.reference_time(); };

template <typename Context>
concept HasClockRegistry = requires(const Context& context) { context.clocks(); };

static_assert(HasExactParams<ExecutionContext, ContextParams>);
static_assert(HasExactParams<ExecutionContext, ContextParamsAlias>);
static_assert(!HasExactParams<ExecutionContext, const ContextParams>);
static_assert(!HasExactParams<ExecutionContext, ContextParams&>);
static_assert(!HasExactParams<ExecutionContext, ContextParams&&>);
static_assert(!HasTryParams<ExecutionContext>);
static_assert(!HasReferenceTime<ExecutionContext>);
static_assert(!HasClockRegistry<ExecutionContext>);

static_assert(std::same_as<decltype(std::declval<const ExecutionContext&>().tag()), Tag>);
static_assert(noexcept(std::declval<const ExecutionContext&>().tag()));
static_assert(std::same_as<decltype(std::declval<const ExecutionContext&>().tag_time()), TagTime>);
static_assert(noexcept(std::declval<const ExecutionContext&>().tag_time()));
static_assert(
    std::same_as<decltype(std::declval<const ExecutionContext&>().activation_time()), TimePoint>);
static_assert(noexcept(std::declval<const ExecutionContext&>().activation_time()));
static_assert(std::same_as<decltype(std::declval<const ExecutionContext&>().clock()), ClockRef>);
static_assert(noexcept(std::declval<const ExecutionContext&>().clock()));
static_assert(std::same_as<decltype(std::declval<const ExecutionContext&>().cancellation()),
                           CancellationView>);
static_assert(noexcept(std::declval<const ExecutionContext&>().cancellation()));
static_assert(
    std::same_as<decltype(std::declval<const ExecutionContext&>().activation()), ActivationStatus>);
static_assert(noexcept(std::declval<const ExecutionContext&>().activation()));
static_assert(std::same_as<decltype(std::declval<const ExecutionContext&>().params_generation()),
                           ParamGeneration>);
static_assert(noexcept(std::declval<const ExecutionContext&>().params_generation()));

static_assert(std::copy_constructible<CancellationView>);
static_assert(std::is_nothrow_copy_constructible_v<CancellationView>);
static_assert(
    std::same_as<decltype(std::declval<const CancellationView&>().stop_requested()), bool>);
static_assert(noexcept(std::declval<const CancellationView&>().stop_requested()));

using RestartResult = expected<LifecycleRestartReceipt, Error>;
static_assert(
    std::same_as<decltype(std::declval<ExecutionContext&>().request_restart()), RestartResult>);
static_assert(noexcept(std::declval<ExecutionContext&>().request_restart()));
static_assert(std::same_as<decltype(std::declval<ExecutionContext&>().request_restart(
                               std::declval<OperatorRef>())),
                           RestartResult>);
static_assert(
    noexcept(std::declval<ExecutionContext&>().request_restart(std::declval<OperatorRef>())));
// REQ: OPERATOR-038
static_assert(std::same_as<decltype(std::declval<ExecutionContext&>().request_completion()), void>);
static_assert(noexcept(std::declval<ExecutionContext&>().request_completion()));
static_assert(std::same_as<decltype(std::declval<ExecutionContext&>().request_stop()), void>);
static_assert(noexcept(std::declval<ExecutionContext&>().request_stop()));

static_assert(std::is_trivially_copyable_v<OperatorRef>);
static_assert(std::is_trivially_copyable_v<LifecycleRestartOptions>);
static_assert(std::is_trivially_copyable_v<LifecycleRestartReceipt>);

// REQ: OPERATOR-005
// REQ: LIFECYCLE-006
TEST(ExecutionContextValues, RestartDefaultsToConfigureAndCarriesNoText) {
  constexpr LifecycleRestartOptions options;
  EXPECT_EQ(options.from_stage, LifecycleStage::kConfigure);
  EXPECT_EQ(options.reason.stable_id, 0);
  EXPECT_TRUE((std::is_same_v<decltype(LifecycleRestartReason::stable_id), ReasonId>));
}

// REQ: OPERATOR-005
TEST(ExecutionContextValues, OperatorRefSeparatesPlanAndRunBinding) {
  const OperatorRef invalid;
  const OperatorRef plan_bound{
      .id = 4,
      .plan_generation = 9,
      .run_generation = 0,
  };
  const OperatorRef run_bound{
      .id = 4,
      .plan_generation = 9,
      .run_generation = 2,
  };

  EXPECT_FALSE(invalid.valid());
  EXPECT_TRUE(plan_bound.valid());
  EXPECT_TRUE(plan_bound.plan_bound());
  EXPECT_FALSE(plan_bound.run_bound());
  EXPECT_TRUE(run_bound.valid());
  EXPECT_FALSE(run_bound.plan_bound());
  EXPECT_TRUE(run_bound.run_bound());
}

// REQ: OPERATOR-022
TEST(ExecutionContextValues, ActivationStatusDefaultsToNominalEmitWitness) {
  constexpr ActivationStatus default_status{};
  constexpr ActivationStatus nominal{
      .reason = ActivationReason::kNominal,
      .policy = OnViolation::kEmitWitness,
  };
  EXPECT_EQ(nominal.reason, ActivationReason::kNominal);
  EXPECT_EQ(default_status.reason, ActivationReason::kNominal);
  EXPECT_EQ(default_status.policy, OnViolation::kEmitWitness);
}

// Operator that calls request_restart() from compute(), recording the outcome in 'receipt' and
// 'observed'. Supports both self-restart (target empty) and targeted restart (target set).
// Representative test: RestartRequestsValidateThroughThePublicContextBeforeCallingRuntime.
class RestartProbeOperator final : public Operator<> {
 public:
  LifecycleRestartOptions options{};
  std::optional<OperatorRef> target{};
  std::optional<LifecycleRestartReceipt> receipt{};
  ErrorCode observed{ErrorCode::kSuccess};

  expected<void, Error> compute(ExecutionContext& context) override {
    const auto result = target.has_value() ? context.request_restart(*target, options)
                                           : context.request_restart(options);
    if (result) {
      observed = ErrorCode::kSuccess;
      receipt = *result;
    } else {
      observed = result.error().code;
      receipt.reset();
    }
    return {};
  }
};

// Mock restart-request callback that counts invocations and records the most recent target and
// options. Returns a synthetic LifecycleRestartReceipt with monotonically increasing receipt_id
// and acceptance_sequence, enabling precise assertions on what the runtime received.
struct RestartRequestSpy {
  std::uint32_t calls{};
  OperatorRef target{};
  LifecycleRestartOptions options{};

  static expected<LifecycleRestartReceipt, Error> request(
      void* opaque, OperatorRef target, LifecycleRestartOptions options) noexcept {
    auto& spy = *static_cast<RestartRequestSpy*>(opaque);
    ++spy.calls;
    spy.target = target;
    spy.options = options;
    return LifecycleRestartReceipt{
        .receipt_id = spy.calls,
        .acceptance_sequence = spy.calls,
        .target = target,
        .requested_stage = options.from_stage,
    };
  }
};

// REQ: OPERATOR-017
TEST(ExecutionContextValues, RestartRequestsValidateThroughThePublicContextBeforeCallingRuntime) {
  // Key setup: RestartProbeOperator + RestartRequestSpy wired via SyntheticActivationOptions; a
  //   series of invoke() calls exercise invalid self refs, stale targets, cross-session targets,
  //   and one valid same-session target.
  // Key assertion: all invalid cases return an error and do not call the spy; the valid
  //   explicit-target call and a subsequent self-restart each reach the spy exactly once, with
  //   correct target id, plan/run generation, stage, and receipt fields.
  constexpr OperatorRef self{.id = 1, .plan_generation = 7, .run_generation = 11};
  constexpr OperatorRef same_session_target{.id = 2, .plan_generation = 7, .run_generation = 11};
  RestartProbeOperator op;
  RestartRequestSpy spy;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());

  const auto invoke =
      [&](OperatorRef invoking_self, std::optional<OperatorRef> target, ErrorCode expected) {
        op.target = target;
        const std::uint32_t calls_before = spy.calls;
        const detail::SyntheticActivationOptions activation{
            .restart_state = &spy,
            .request_restart = &RestartRequestSpy::request,
            .self = invoking_self,
        };
        EXPECT_TRUE(fixture.invoke_once(activation));
        EXPECT_EQ(op.observed, expected);
        if (expected == ErrorCode::kSuccess) {
          EXPECT_EQ(spy.calls, calls_before + 1);
        } else {
          EXPECT_EQ(spy.calls, calls_before);
          EXPECT_FALSE(op.receipt.has_value());
        }
      };

  invoke(OperatorRef{}, std::nullopt, ErrorCode::kStaleOperatorRef);
  invoke(OperatorRef{.id = 1, .plan_generation = 7, .run_generation = 0},
         std::nullopt,
         ErrorCode::kStaleOperatorRef);
  invoke(self, OperatorRef{}, ErrorCode::kStaleOperatorRef);
  invoke(self,
         OperatorRef{.id = 2, .plan_generation = 7, .run_generation = 0},
         ErrorCode::kStaleOperatorRef);
  invoke(self,
         OperatorRef{.id = 2, .plan_generation = 8, .run_generation = 11},
         ErrorCode::kStaleOperatorRef);
  invoke(self,
         OperatorRef{.id = 2, .plan_generation = 7, .run_generation = 12},
         ErrorCode::kCrossSessionOperatorRef);
  invoke(self, same_session_target, ErrorCode::kSuccess);

  ASSERT_TRUE(op.receipt.has_value());
  const LifecycleRestartReceipt explicit_receipt = op.receipt.value_or(LifecycleRestartReceipt{});
  EXPECT_EQ(spy.calls, 1);
  EXPECT_EQ(spy.target.id, same_session_target.id);
  EXPECT_EQ(spy.target.plan_generation, same_session_target.plan_generation);
  EXPECT_EQ(spy.target.run_generation, same_session_target.run_generation);
  EXPECT_EQ(spy.options.from_stage, LifecycleStage::kConfigure);
  EXPECT_EQ(explicit_receipt.receipt_id, 1);
  EXPECT_EQ(explicit_receipt.acceptance_sequence, 1);
  EXPECT_EQ(explicit_receipt.target.id, same_session_target.id);
  EXPECT_EQ(explicit_receipt.requested_stage, LifecycleStage::kConfigure);

  op.options.from_stage = LifecycleStage::kArm;
  invoke(self, std::nullopt, ErrorCode::kSuccess);
  ASSERT_TRUE(op.receipt.has_value());
  const LifecycleRestartReceipt self_receipt = op.receipt.value_or(LifecycleRestartReceipt{});
  EXPECT_EQ(spy.calls, 2);
  EXPECT_EQ(spy.target.id, self.id);
  EXPECT_EQ(spy.target.plan_generation, self.plan_generation);
  EXPECT_EQ(spy.target.run_generation, self.run_generation);
  EXPECT_EQ(spy.options.from_stage, LifecycleStage::kArm);
  EXPECT_EQ(self_receipt.receipt_id, 2);
  EXPECT_EQ(self_receipt.acceptance_sequence, 2);
  EXPECT_EQ(self_receipt.target.id, self.id);
  EXPECT_EQ(self_receipt.requested_stage, LifecycleStage::kArm);
}

// REQ: OPERATOR-005
TEST(ExecutionContextValues, InvalidRestartStagePrecedesDisabledRuntimeBinding) {
  // Key setup: RestartProbeOperator exercised with self-restart and targeted variants; the loop
  //   covers kStop, kRelease, and an out-of-range enum cast as invalid restart stages.
  // Key assertion: invalid stage -> kLifecycleInvalidRestartStage before any runtime check; a
  //   valid stage with no spy binding -> kLifecycleTransactionsDisabled.
  RestartProbeOperator op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());

  // The final value deliberately exercises a wire-corrupt/forward-unknown enum representation.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  const auto unknown_stage = static_cast<LifecycleStage>(255);
  for (const LifecycleStage stage :
       {LifecycleStage::kStop, LifecycleStage::kRelease, unknown_stage}) {
    op.options.from_stage = stage;
    ASSERT_TRUE(fixture.invoke_once());
    EXPECT_EQ(op.observed, ErrorCode::kLifecycleInvalidRestartStage);

    op.target = OperatorRef{.id = 1, .plan_generation = 1, .run_generation = 1};
    ASSERT_TRUE(fixture.invoke_once());
    EXPECT_EQ(op.observed, ErrorCode::kLifecycleInvalidRestartStage);
    op.target.reset();
  }

  op.options.from_stage = LifecycleStage::kConfigure;
  ASSERT_TRUE(fixture.invoke_once());
  EXPECT_EQ(op.observed, ErrorCode::kLifecycleTransactionsDisabled);
}

class TerminationRequestProbeOperator final : public Operator<> {
 public:
  bool request_stop{};
  bool returned{};

  expected<void, Error> compute(ExecutionContext& context) override {
    if (request_stop) {
      context.request_stop();
    } else {
      context.request_completion();
    }
    returned = true;
    return {};
  }
};

struct TerminationRequestSpy {
  std::uint32_t calls{};
  detail::TerminationRequest request{};

  static void notify(void* opaque, detail::TerminationRequest request) noexcept {
    auto& spy = *static_cast<TerminationRequestSpy*>(opaque);
    ++spy.calls;
    spy.request = request;
  }
};

// REQ: OPERATOR-038
TEST(ExecutionContextValues, TerminationRequestsForwardOnlyThroughTheActiveCallbackBinding) {
  TerminationRequestProbeOperator op;
  TerminationRequestSpy spy;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());

  ASSERT_TRUE(fixture.invoke_once());
  EXPECT_TRUE(op.returned);
  EXPECT_EQ(spy.calls, 0);

  const detail::SyntheticActivationOptions activation{
      .termination_request_state = &spy,
      .request_termination = &TerminationRequestSpy::notify,
  };
  ASSERT_TRUE(fixture.invoke_once(activation));
  EXPECT_TRUE(op.returned);
  EXPECT_EQ(spy.calls, 1);
  EXPECT_EQ(spy.request, detail::TerminationRequest::kCompletion);

  op.request_stop = true;
  ASSERT_TRUE(fixture.invoke_once(activation));
  EXPECT_TRUE(op.returned);
  EXPECT_EQ(spy.calls, 2);
  EXPECT_EQ(spy.request, detail::TerminationRequest::kStop);
}

// Operator<ContextParams> that deliberately calls params<WrongContextParams>() in compute().
// Used by the death test to confirm that a type mismatch routes through the terminal core handler.
class WrongParameterProbeOperator final : public Operator<ContextParams> {
 public:
  expected<void, Error> compute(ExecutionContext& context) override {
    static_cast<void>(context.params<WrongContextParams>());
    return {};
  }
};

// REQ: OPERATOR-005
// REQ: PARAM-001
TEST(ExecutionContextDeathTest, WrongExactParameterTypeUsesTerminalCoreHandler) {
  EXPECT_DEATH_IF_SUPPORTED(
      {
        WrongParameterProbeOperator op;
        test::SyntheticOperatorFixture fixture{op};
        if (!fixture.declare_ports()) {
          std::abort();
        }
        static_cast<void>(fixture.invoke_once());
      },
      "");
}

}  // namespace
}  // namespace holoscan
