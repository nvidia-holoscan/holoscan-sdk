/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>

#include <holoscan/core/predicate_input_view.hpp>

#include <core/predicate_input_view_internal.hpp>

namespace holoscan {
namespace {

static_assert(
    std::is_same_v<decltype(std::declval<const PredicateInputView<std::uint32_t>&>().sample()),
                   const Sample<std::uint32_t>&>);
static_assert(
    std::is_same_v<decltype(std::declval<const PredicateInputView<std::uint32_t>&>().selection()),
                   InputSelection>);
static_assert(
    std::is_same_v<decltype(std::declval<const PredicateEvaluationContext&>().selections()),
                   std::span<const PredicateInputSelection>>);

// REQ: OPERATOR-008
// Verifies that a valid input view reflects the sample value and selection state,
// and that the evaluation context exposes selections in declaration order.
TEST(PredicateInputView, PreservesPinnedSampleAndDeclarationOrderedSelections) {
  // Key setup: selected input + a context with two selections in a fixed port order.
  const Sample<std::uint32_t> sample{17, SampleMetadata{.sequence = 4}};
  const std::uint64_t generation = 3;
  const auto view = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
      .active_generation = &generation,
      .expected_generation = generation,
  });
  ASSERT_TRUE(view);

  EXPECT_EQ(view->sample().data, 17);
  EXPECT_EQ(view->selection().state, InputState::kSelected);

  const std::array selections{
      PredicateInputSelection{PortId{9},
                              InputSelection{InputState::kSelected, SelectionReason::kNone}},
      PredicateInputSelection{PortId{2},
                              InputSelection{InputState::kReused, SelectionReason::kTooStale}},
  };
  const auto context = detail::PredicateBindingAccess::context({
      .reference_time = TimePoint{.timestamp_ns = 50, .clock_id = ClockId{.value = 1}},
      .tag = Tag{.sequence = 8, .generation = 1},
      .tag_time = TagTime{.logical_ns = 51},
      .selections = selections,
      .active_generation = &generation,
      .expected_generation = generation,
  });

  // Key assertion: both the individual view and the context preserve field values
  // without copying — the context selection span points into the caller's array.
  ASSERT_TRUE(context);
  ASSERT_EQ(context->selections().size(), 2);
  EXPECT_EQ(context->selections()[0].port, PortId{9});
  EXPECT_EQ(context->selections()[1].selection.state, InputState::kReused);
}

// REQ: OPERATOR-008
// A kReused state (holdover from a prior activation) is a valid input view state;
// predicates may choose to act on it or skip based on selection reason.
TEST(PredicateInputView, AdmitsReusedPinnedScalarSample) {
  const Sample<std::uint32_t> sample{23, SampleMetadata{.sequence = 5}};
  const std::uint64_t generation = 7;
  const auto view = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = InputSelection{InputState::kReused, SelectionReason::kTooStale},
      .active_generation = &generation,
      .expected_generation = generation,
  });

  ASSERT_TRUE(view);
  EXPECT_EQ(view->sample().data, 23);
  EXPECT_EQ(view->selection().state, InputState::kReused);
  EXPECT_EQ(view->selection().reason, SelectionReason::kTooStale);
}

// REQ: OPERATOR-008
// REQ: OPERATOR-018
// Every combination of bad binding arguments must fail with kInvalidArgument:
// null sample, absent/not-selected state, inconsistent reason flags, stale/zero generation.
TEST(PredicateInputView, RejectsNullOrUnselectedPrivateBindings) {
  // Key setup: eight distinct malformed bindings, each exercising a different
  // validation branch in PredicateBindingAccess::input<T>.
  const Sample<std::uint32_t> sample{31, SampleMetadata{}};
  const std::uint64_t active_generation = 4;
  const auto null_sample = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = nullptr,
      .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
      .active_generation = &active_generation,
      .expected_generation = active_generation,
  });
  const auto absent_sample = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen},
      .active_generation = &active_generation,
      .expected_generation = active_generation,
  });
  const auto not_selected_sample = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection =
          InputSelection{InputState::kNotSelected, SelectionReason::kNotSelectedForActivation},
      .active_generation = &active_generation,
      .expected_generation = active_generation,
  });
  const auto fresh_with_failure_reason = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = InputSelection{InputState::kSelected, SelectionReason::kTooStale},
      .active_generation = &active_generation,
      .expected_generation = active_generation,
  });
  const auto reused_without_failure_reason = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = InputSelection{InputState::kReused, SelectionReason::kNone},
      .active_generation = &active_generation,
      .expected_generation = active_generation,
  });
  const auto stale_generation = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
      .active_generation = &active_generation,
      .expected_generation = 3,
  });
  const auto missing_generation_pointer = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
      .expected_generation = 3,
  });
  const std::uint64_t zero_generation = 0;
  const auto zero_generation_witness = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
      .active_generation = &zero_generation,
      .expected_generation = zero_generation,
  });

  ASSERT_FALSE(null_sample);
  EXPECT_EQ(null_sample.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(absent_sample);
  EXPECT_EQ(absent_sample.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(not_selected_sample);
  EXPECT_EQ(not_selected_sample.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(fresh_with_failure_reason);
  EXPECT_EQ(fresh_with_failure_reason.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(reused_without_failure_reason);
  EXPECT_EQ(reused_without_failure_reason.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(stale_generation);
  EXPECT_EQ(stale_generation.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(missing_generation_pointer);
  EXPECT_EQ(missing_generation_pointer.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(zero_generation_witness);
  EXPECT_EQ(zero_generation_witness.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-018
// A predicate evaluation context must carry live generation evidence; missing pointer,
// zero generation, and stale generation are all rejected to prevent use-after-reset bugs.
TEST(PredicateInputView, RejectsMissingOrStaleContextGenerationEvidence) {
  const std::uint64_t active_generation = 9;
  const std::uint64_t zero_generation = 0;
  const auto missing_generation = detail::PredicateBindingAccess::context({});
  const auto zero_generation_witness = detail::PredicateBindingAccess::context({
      .active_generation = &zero_generation,
      .expected_generation = zero_generation,
  });
  const auto stale_generation = detail::PredicateBindingAccess::context({
      .active_generation = &active_generation,
      .expected_generation = active_generation + 1U,
  });

  ASSERT_FALSE(missing_generation);
  EXPECT_EQ(missing_generation.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(zero_generation_witness);
  EXPECT_EQ(zero_generation_witness.error().code, ErrorCode::kInvalidArgument);
  ASSERT_FALSE(stale_generation);
  EXPECT_EQ(stale_generation.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-018
// After the active generation counter is incremented (simulating the end of an activation
// window), any checked accessor on the stale view or context must assert-fail in debug builds.
TEST(PredicateInputView, CheckedAccessRejectsExpiredEvaluationGeneration) {
  // Key setup: valid view + context captured at generation 11; then generation is bumped
  // so that all subsequent accesses are detected as use-after-reset.
  const Sample<std::uint32_t> sample{37, SampleMetadata{.sequence = 6}};
  std::uint64_t generation = 11;
  const auto view = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = InputSelection{InputState::kSelected, SelectionReason::kNone},
      .active_generation = &generation,
      .expected_generation = generation,
  });
  ASSERT_TRUE(view);

  const std::array selections{
      PredicateInputSelection{PortId{4},
                              InputSelection{InputState::kSelected, SelectionReason::kNone}},
  };
  const auto context = detail::PredicateBindingAccess::context({
      .selections = selections,
      .active_generation = &generation,
      .expected_generation = generation,
  });
  ASSERT_TRUE(context);

  // Key assertion: reads succeed before the generation bump; all six accessors
  // trigger a debug death after ++generation signals the window has closed.
  EXPECT_EQ(view->sample().data, 37);
  EXPECT_EQ(context->selections().size(), 1);
  ++generation;

#ifndef NDEBUG
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(view->sample()), "");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(view->selection()), "");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(context->reference_time()), "");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(context->tag()), "");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(context->tag_time()), "");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(context->selections()), "");
#endif
}

}  // namespace
}  // namespace holoscan
