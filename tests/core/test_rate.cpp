/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/rate_generation.hpp>

#include "../common/allocation_counter.hpp"
#include "core/execution_context_internal.hpp"
#include "core/operator_runtime_access.hpp"
#include "core/operator_spec_compiler_access.hpp"

namespace holoscan {
namespace {

using ScalarInput = Input<std::uint32_t>;
using ScalarOutput = Output<std::uint32_t>;
using TensorOutput = Output<Tensor>;

static_assert(std::same_as<decltype(Rate::hz(60U)), Rate>);
static_assert(
    std::same_as<decltype(std::declval<const Rate&>().normalized()), std::optional<Rate>>);
static_assert(std::same_as<decltype(std::declval<const RateRatio&>().normalized()),
                           std::optional<RateRatio>>);
static_assert(std::same_as<decltype(std::declval<const Rate&>().equivalent_to(Rate{})), bool>);
static_assert(
    std::same_as<decltype(std::declval<const RateRatio&>().equivalent_to(RateRatio{})), bool>);
static_assert(noexcept(std::declval<const Rate&>().normalized()));
static_assert(noexcept(std::declval<const RateRatio&>().normalized()));
static_assert(noexcept(std::declval<const Rate&>().equivalent_to(Rate{})));
static_assert(noexcept(std::declval<const RateRatio&>().equivalent_to(RateRatio{})));
static_assert(std::same_as<RateGenerationId, std::uint64_t>);
static_assert(std::is_copy_constructible_v<RateGenerationSnapshot>);
static_assert(std::is_move_constructible_v<RateGenerationSnapshot>);
static_assert(!std::is_copy_assignable_v<RateGenerationSnapshot>);
static_assert(!std::is_move_assignable_v<RateGenerationSnapshot>);
static_assert(std::same_as<decltype(std::declval<const RateGenerationSnapshot&>().port_rates()),
                           std::span<const PortRateGenerationRecord>>);
static_assert(
    std::same_as<decltype(std::declval<const RateGenerationSnapshot&>().provider_allocations()),
                 std::span<const ProviderAllocationGenerationRecord>>);
static_assert(std::same_as<decltype(std::declval<InputSpecBuilder&>().expected_rate(Rate{})),
                           InputSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<OutputSpecBuilder&>().nominal_rate(Rate{})),
                           OutputSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<OutputSpecBuilder&>().lifecycle_nominal_rate(
                               LifecycleStage::kDiscover)),
                           OutputSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<OutputSpecBuilder&>().max_nominal_rate(Rate{})),
                           OutputSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<OutputSpecBuilder&>().follows(
                               std::declval<const ScalarInput&>(), RateRatio{})),
                           OutputSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<TensorOutputSpecBuilder&>().nominal_rate(Rate{})),
                           TensorOutputSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<TensorOutputSpecBuilder&>().follows(
                               std::declval<const ScalarInput&>(), RateRatio{})),
                           TensorOutputSpecBuilder&>);

// Public declaration mutators reject malformed/conflicting authoring. They must not promise
// noexcept and then terminate in exception-enabled builds.
static_assert(!noexcept(std::declval<InputSpecBuilder&>().expected_rate(Rate{})));
static_assert(!noexcept(std::declval<OutputSpecBuilder&>().nominal_rate(Rate{})));
static_assert(!noexcept(
    std::declval<OutputSpecBuilder&>().lifecycle_nominal_rate(LifecycleStage::kDiscover)));
static_assert(!noexcept(std::declval<OutputSpecBuilder&>().max_nominal_rate(Rate{})));
static_assert(!noexcept(
    std::declval<OutputSpecBuilder&>().follows(std::declval<const ScalarInput&>(), RateRatio{})));
static_assert(CompileOptions{}.automatic_queue_sizing.headroom_ratio == (RateRatio{2U, 1U}));
static_assert(std::same_as<decltype(std::declval<LifecycleContext&>().bind_nominal_rate(
                               std::declval<ScalarOutput&>(), Rate{})),
                           expected<void, Error>>);
static_assert(std::same_as<decltype(std::declval<const LifecycleContext&>().rate(
                               std::declval<const ScalarInput&>())),
                           expected<PortRateView, Error>>);
static_assert(std::same_as<decltype(std::declval<const LifecycleContext&>().rate(
                               std::declval<const ScalarOutput&>())),
                           expected<PortRateView, Error>>);
static_assert(std::same_as<decltype(std::declval<const ExecutionContext&>().rate(
                               std::declval<const ScalarInput&>())),
                           expected<PortRateView, Error>>);
static_assert(std::same_as<decltype(std::declval<const ExecutionContext&>().rate(
                               std::declval<const ScalarOutput&>())),
                           expected<PortRateView, Error>>);

class InvalidExecutionRateQueryProbe final : public Operator<> {
 public:
  void setup(OperatorSpec&) override {}

  [[nodiscard]] Contract contract() const override { return {}; }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const test::ScopedAllocationCounter allocation_counter;
    const auto result = context.rate(output);
    allocations = allocation_counter.count();
    if (result) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    error_code = result.error().code;
    error_message_empty = result.error().message.empty();
    return {};
  }

  ScalarOutput output{"output"};
  std::size_t allocations{};
  ErrorCode error_code{ErrorCode::kSuccess};
  bool error_message_empty{};
};

// Representation equality remains useful for exact serialization and replay. Semantic helpers
// make rational comparisons explicit without changing that representation-level operator.
// REQ: OPERATOR-048
TEST(RateValueContract, NormalizesAndComparesEquivalentRationalsWithoutAllocating) {
  const Rate unreduced_rate{120U, 2U};
  const Rate canonical_rate{60U, 1U};
  const RateRatio unreduced_ratio{6U, 8U};
  const RateRatio canonical_ratio{3U, 4U};

  EXPECT_NE(unreduced_rate, canonical_rate);
  EXPECT_NE(unreduced_ratio, canonical_ratio);

  std::optional<Rate> normalized_rate;
  std::optional<RateRatio> normalized_ratio;
  bool rates_equivalent = false;
  bool ratios_equivalent = false;
  std::size_t allocation_count = 0U;
  {
    const test::ScopedAllocationCounter allocations;
    normalized_rate = unreduced_rate.normalized();
    normalized_ratio = unreduced_ratio.normalized();
    rates_equivalent = unreduced_rate.equivalent_to(canonical_rate);
    ratios_equivalent = unreduced_ratio.equivalent_to(canonical_ratio);
    allocation_count = allocations.count();
  }

  EXPECT_EQ(normalized_rate, canonical_rate);
  EXPECT_EQ(normalized_ratio, canonical_ratio);
  EXPECT_TRUE(rates_equivalent);
  EXPECT_TRUE(ratios_equivalent);
  EXPECT_EQ(allocation_count, 0U);
  EXPECT_FALSE((Rate{0U, 1U}.normalized()));
  EXPECT_FALSE((Rate{1U, 0U}.normalized()));
  EXPECT_FALSE((RateRatio{0U, 1U}.normalized()));
  EXPECT_FALSE((RateRatio{1U, 0U}.normalized()));
  EXPECT_FALSE((Rate{0U, 1U}.equivalent_to(Rate{0U, 1U})));
  EXPECT_FALSE((RateRatio{0U, 1U}.equivalent_to(RateRatio{0U, 1U})));
}

// ExecutionContext guards are part of the compute-time query surface and therefore return a
// code-only error even when the facade has no bound resolver.
// REQ: OPERATOR-048
TEST(RateValueContract, InvalidExecutionQueryIsAllocationFreeAndCodeOnly) {
  InvalidExecutionRateQueryProbe probe;
  detail::ExecutionContextState state;

  const auto invoked = detail::OperatorRuntimeAccess::invoke(probe, state);

  ASSERT_TRUE(invoked);
  EXPECT_EQ(probe.allocations, 0U);
  EXPECT_EQ(probe.error_code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(probe.error_message_empty);
}

// REQ: OPERATOR-048
TEST(PortRateViewContract, SeparatesStableRunIdentityFromRateGenerationIdentity) {
  const PortRateView first{
      .resolved_nominal_rate = Rate::hz(60U),
      .sizing_rate = Rate::hz(60U),
      .max_nominal_rate = Rate::hz(120U),
      .origin = ResolvedRateOrigin::kLifecycleBinding,
      .state = PortRateViewState::kResolved,
      .run_generation = 7U,
      .rate_generation = 11U,
  };
  auto restarted = first;
  restarted.rate_generation = 12U;

  EXPECT_EQ(first.run_generation, restarted.run_generation);
  EXPECT_NE(first.rate_generation, restarted.rate_generation);
  EXPECT_NE(first, restarted);
}

// REQ: OPERATOR-048
TEST(RateGenerationSnapshotContract, DefaultSnapshotIsEmptyAndOwnsValueRecords) {
  const RateGenerationSnapshot snapshot;
  EXPECT_EQ(snapshot.plan_generation(), 0U);
  EXPECT_EQ(snapshot.run_generation(), 0U);
  EXPECT_EQ(snapshot.published(), nullptr);
  EXPECT_EQ(snapshot.last_rejected(), nullptr);
  EXPECT_TRUE(snapshot.generations().empty());
  EXPECT_TRUE(snapshot.port_rates().empty());
  EXPECT_TRUE(snapshot.queue_sizings().empty());
  EXPECT_TRUE(snapshot.pool_contributions().empty());
  EXPECT_TRUE(snapshot.pool_capacities().empty());
  EXPECT_TRUE(snapshot.provider_allocations().empty());
}

// REQ: OPERATOR-048
TEST(PortRateAuthoring, CapturesNormalizedDeclarationsAndIdempotentRepeats) {
  OperatorSpec spec;
  ScalarInput input{"input"};
  ScalarOutput fixed{"fixed"};
  TensorOutput derived{"derived"};
  ScalarOutput selectable{"selectable"};

  spec.input(input).expected_rate(Rate::hz(60U)).expected_rate(Rate::hz(120U, 2U));
  spec.output(fixed).nominal_rate(Rate::hz(1'000U)).nominal_rate(Rate::hz(2'000U, 2U));
  spec.output(derived).follows(input, RateRatio{1U, 4U}).follows(input, RateRatio{2U, 8U});
  spec.output(selectable)
      .max_nominal_rate(Rate::hz(120U))
      .lifecycle_nominal_rate(LifecycleStage::kDiscover)
      .max_nominal_rate(Rate::hz(240U, 2U));

  const auto input_metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(input_metadata.rate_kind, PortRateKind::kExpectedInput);
  EXPECT_EQ(input_metadata.expected_rate, Rate::hz(60U));
  EXPECT_EQ(input_metadata.rate_error.code, ErrorCode::kSuccess);

  const auto fixed_metadata = detail::OperatorSpecCompilerAccess::port(spec, 1U);
  EXPECT_EQ(fixed_metadata.rate_kind, PortRateKind::kFixedNominal);
  EXPECT_EQ(fixed_metadata.nominal_rate, Rate::hz(1'000U));
  EXPECT_EQ(fixed_metadata.rate_error.code, ErrorCode::kSuccess);

  const auto derived_metadata = detail::OperatorSpecCompilerAccess::port(spec, 2U);
  EXPECT_EQ(derived_metadata.rate_kind, PortRateKind::kFollowsInput);
  EXPECT_EQ(derived_metadata.follows_input_declaration, &input);
  EXPECT_EQ(derived_metadata.follows_ratio, (RateRatio{1U, 4U}));
  EXPECT_EQ(derived_metadata.rate_error.code, ErrorCode::kSuccess);

  const auto selectable_metadata = detail::OperatorSpecCompilerAccess::port(spec, 3U);
  EXPECT_EQ(selectable_metadata.rate_kind, PortRateKind::kLifecycleNominal);
  EXPECT_EQ(selectable_metadata.lifecycle_binding_stage, LifecycleStage::kDiscover);
  EXPECT_EQ(selectable_metadata.max_nominal_rate, Rate::hz(120U));
  EXPECT_EQ(selectable_metadata.rate_error.code, ErrorCode::kSuccess);
}

// REQ: OPERATOR-048
TEST(PortRateAuthoring, RetainsFirstFailureForMalformedAndConflictingDeclarations) {
  OperatorSpec spec;
  ScalarInput input{"input"};
  ScalarOutput conflict{"conflict"};
  ScalarOutput invalid{"invalid"};
  ScalarOutput invalid_stage{"invalid_stage"};

  spec.input(input).expected_rate(Rate::hz(0U));
  spec.output(conflict).nominal_rate(Rate::hz(60U)).follows(input);
  spec.output(invalid).nominal_rate(Rate::hz(60U, 0U));
  spec.output(invalid_stage).lifecycle_nominal_rate(LifecycleStage::kAllocate);

  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(spec, 0U).rate_error.message,
            "PORT_RATE_INVALID");
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(spec, 1U).rate_error.message,
            "PORT_RATE_CONFLICT");
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(spec, 2U).rate_error.message,
            "PORT_RATE_INVALID");
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(spec, 3U).rate_error.message,
            "PORT_RATE_BIND_STAGE_INVALID");
}

// REQ: OPERATOR-048
TEST(PortRateAuthoring, StickyFailureSurvivesDiagnosticAllocationFailure) {
  OperatorSpec spec;
  ScalarOutput output{"output"};
  auto& builder = spec.output(output);

  {
    // The failure remains armed if the declaration path allocates.  A correct rate builder records
    // its static reason without allocating and therefore cannot lose the sticky failure state.
    const test::ScopedAllocationFailure fail_next_allocation{0U};
    EXPECT_NO_THROW(builder.nominal_rate(Rate::hz(0U)));
  }

  const auto metadata = detail::OperatorSpecCompilerAccess::port(spec, 0U);
  EXPECT_EQ(metadata.rate_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(metadata.rate_error.message, "PORT_RATE_INVALID");
  EXPECT_EQ(metadata.rate_kind, PortRateKind::kUnspecified);
}

// REQ: OPERATOR-048
TEST(PortRateAuthoring, ValidatesRatiosStagesMaximumsAndEveryConflictingRepeat) {
  const auto error_for = [](const auto& declare) {
    OperatorSpec spec;
    ScalarInput input{"input"};
    ScalarInput other{"other"};
    ScalarOutput output{"output"};
    spec.input(input);
    spec.input(other);
    auto& builder = spec.output(output);
    declare(builder, input, other);
    return detail::OperatorSpecCompilerAccess::port(spec, 2U).rate_error;
  };

  EXPECT_EQ(error_for([](auto& output, const auto& input, const auto&) {
              output.follows(input, RateRatio{0U, 1U});
            }).message,
            "PORT_RATE_INVALID");
  EXPECT_EQ(error_for([](auto& output, const auto& input, const auto&) {
              output.follows(input, RateRatio{1U, 0U});
            }).message,
            "PORT_RATE_INVALID");
  EXPECT_EQ(error_for([](auto& output, const auto& input, const auto& other) {
              output.follows(input).follows(other);
            }).message,
            "PORT_RATE_CONFLICT");
  EXPECT_EQ(error_for([](auto& output, const auto& input, const auto&) {
              output.follows(input).follows(input, RateRatio{2U, 1U});
            }).message,
            "PORT_RATE_CONFLICT");
  EXPECT_EQ(error_for([](auto& output, const auto&, const auto&) {
              output.lifecycle_nominal_rate(LifecycleStage::kConfigure)
                  .lifecycle_nominal_rate(LifecycleStage::kDiscover);
            }).message,
            "PORT_RATE_CONFLICT");
  EXPECT_EQ(error_for([](auto& output, const auto&, const auto&) {
              output.lifecycle_nominal_rate(LifecycleStage::kDiscover)
                  .max_nominal_rate(Rate::hz(120U))
                  .max_nominal_rate(Rate::hz(60U));
            }).message,
            "PORT_RATE_CONFLICT");
  EXPECT_EQ(error_for([](auto& output, const auto& input, const auto&) {
              output.nominal_rate(Rate::hz(60U)).follows(input);
            }).message,
            "PORT_RATE_CONFLICT");
  EXPECT_EQ(error_for([](auto& output, const auto&, const auto&) {
              output.lifecycle_nominal_rate(LifecycleStage::kDiscover).nominal_rate(Rate::hz(60U));
            }).message,
            "PORT_RATE_CONFLICT");
  EXPECT_EQ(error_for([](auto& output, const auto&, const auto&) {
              output.nominal_rate(Rate::hz(0U)).nominal_rate(Rate::hz(60U));
            }).message,
            "PORT_RATE_INVALID");

  for (const auto stage : {LifecycleStage::kConfigure,
                           LifecycleStage::kDiscover,
                           LifecycleStage::kWarmUp,
                           LifecycleStage::kArm}) {
    EXPECT_EQ(error_for([stage](auto& output, const auto&, const auto&) {
                output.lifecycle_nominal_rate(stage).lifecycle_nominal_rate(stage);
              }).code,
              ErrorCode::kSuccess);
  }
  for (const auto stage : {LifecycleStage::kAllocate,
                           LifecycleStage::kStart,
                           LifecycleStage::kStop,
                           LifecycleStage::kRelease}) {
    EXPECT_EQ(error_for([stage](auto& output, const auto&, const auto&) {
                output.lifecycle_nominal_rate(stage);
              }).message,
              "PORT_RATE_BIND_STAGE_INVALID");
  }

  OperatorSpec input_spec;
  ScalarInput input{"input"};
  input_spec.input(input).expected_rate(Rate::hz(60U)).expected_rate(Rate::hz(30U));
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(input_spec, 0U).rate_error.message,
            "PORT_RATE_CONFLICT");
}

// REQ: GRAPH-033
TEST(PortRateAuthoring, DistinguishesOmittedQueueDepthFromExplicitOne) {
  OperatorSpec spec;
  ScalarInput omitted{"omitted"};
  ScalarInput explicit_one{"explicit_one"};

  spec.input(omitted);
  spec.input(explicit_one).queue_depth(1U);

  EXPECT_FALSE(detail::OperatorSpecCompilerAccess::port(spec, 0U).queue_depth.has_value());
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(spec, 1U).queue_depth, 1U);
}

// REQ: OPERATOR-048
TEST(PortRateAuthoring, TensorBuilderPreservesItsSpecializedFluentType) {
  OperatorSpec spec;
  ScalarInput input{"input"};
  TensorOutput output{"output"};

  spec.input(input).expected_rate(Rate::hz(240U));
  auto& builder = spec.output(output);
  auto& chained = builder.follows(input, RateRatio{1U, 4U})
                      .max_emits_per_compute(2U)
                      .late_joiner_history_depth(3U);

  static_assert(std::same_as<decltype(chained), TensorOutputSpecBuilder&>);
  EXPECT_EQ(&chained, &builder);
  EXPECT_EQ(detail::OperatorSpecCompilerAccess::port(spec, 1U).rate_kind,
            PortRateKind::kFollowsInput);
}

}  // namespace
}  // namespace holoscan
