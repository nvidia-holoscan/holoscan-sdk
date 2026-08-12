/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "../common/allocation_counter.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_termination.hpp"

namespace holoscan::detail {
namespace {

// REQ: RUNTIME-004
TEST(RuntimeDiagnosticBufferTest, PublishesExplicitExhaustionAndKeepsDetailsParallel) {
  // Key setup: buffer filled to kRuntimeStructuralDiagnosticCapacity-1 with detail records,
  //            then one more append triggers the sentinel exhaustion slot.
  constexpr GraphGeneration kGraphGeneration = 17U;
  RuntimeDiagnosticBuffer buffer{
      DiagnosticId{901U},
      DiagnosticSource{.graph_generation = kGraphGeneration},
  };

  for (std::size_t index = 0U; index + 1U < kRuntimeStructuralDiagnosticCapacity; ++index) {
    const RuntimeDiagnosticDetail detail{
        .present = true,
        .error = Error{ErrorCode::kFailure, "callback failure"},
        .predicate_id = "predicate",
        .operator_id = static_cast<OperatorInstanceId>(index + 1U),
        .source_sequence = index + 1U,
        .tag = Tag{index + 1U, 1U, 0U},
        .tag_time = TagTime{static_cast<std::int64_t>(index), 0U, 0U},
        .reference_time = TimePoint{static_cast<std::int64_t>(index), ClockId{3U}},
        .state_generation = 1U,
    };
    buffer.append(DiagnosticId{static_cast<std::uint32_t>(index + 1U)},
                  "RUNTIME_OPERATOR_CALLBACK_FAILED",
                  DiagnosticSource{
                      .graph_generation = kGraphGeneration,
                      .vertex = VertexId{index + 1U},
                  },
                  "operator_compute",
                  &detail);
  }

  ASSERT_EQ(buffer.diagnostics().size(), kRuntimeStructuralDiagnosticCapacity - 1U);
  ASSERT_EQ(buffer.details().size(), buffer.diagnostics().size());
  EXPECT_TRUE(buffer.details().back().present);
  EXPECT_EQ(buffer.details().back().operator_id,
            static_cast<OperatorInstanceId>(kRuntimeStructuralDiagnosticCapacity - 1U));

  buffer.append(DiagnosticId{999U},
                "RUNTIME_OPERATOR_CALLBACK_FAILED",
                DiagnosticSource{.graph_generation = kGraphGeneration},
                "operator_compute");

  // Key assertion: the sentinel uses the buffer's own DiagnosticId and the exhaustion token;
  //               subsequent appends are silently dropped and size stays at capacity.
  ASSERT_EQ(buffer.diagnostics().size(), kRuntimeStructuralDiagnosticCapacity);
  ASSERT_EQ(buffer.details().size(), buffer.diagnostics().size());
  EXPECT_EQ(buffer.diagnostics().back().id, DiagnosticId{901U});
  EXPECT_EQ(buffer.diagnostics().back().token, "RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED");
  EXPECT_EQ(buffer.diagnostics().back().source.graph_generation, kGraphGeneration);
  EXPECT_FALSE(buffer.details().back().present);

  buffer.append(DiagnosticId{1000U},
                "RUNTIME_STARTUP_FAILED",
                DiagnosticSource{.graph_generation = kGraphGeneration});
  EXPECT_EQ(buffer.diagnostics().size(), kRuntimeStructuralDiagnosticCapacity);
  EXPECT_EQ(buffer.diagnostics().back().token, "RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED");
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: RUNTIME-004
TEST(RuntimeDiagnosticBufferTest, AllocationFailureLeavesTheBoundedSlotReusable) {
  RuntimeDiagnosticBuffer buffer{DiagnosticId{901U}, DiagnosticSource{}};
  constexpr std::string_view kLongToken =
      "RUNTIME_DIAGNOSTIC_WITH_A_TOKEN_LONG_ENOUGH_TO_REQUIRE_OWNED_STORAGE_ALLOCATION";
  // Key setup: ScopedAllocationFailure fails the very next allocation, which the long-token
  //            string copy triggers; the slot must be rolled back so the buffer stays empty.
  {
    const test::ScopedAllocationFailure fail_next_allocation{0U};
    buffer.append(DiagnosticId{1U}, kLongToken, DiagnosticSource{});
  }
  // Key assertion: buffer is empty after the allocation failure; a subsequent append succeeds.
  EXPECT_TRUE(buffer.diagnostics().empty());
  EXPECT_TRUE(buffer.details().empty());

  buffer.append(DiagnosticId{2U}, "RUNTIME_STARTUP_FAILED", DiagnosticSource{});
  ASSERT_EQ(buffer.diagnostics().size(), 1U);
  ASSERT_EQ(buffer.details().size(), 1U);
  EXPECT_EQ(buffer.diagnostics().front().id, DiagnosticId{2U});
  EXPECT_EQ(buffer.diagnostics().front().token, "RUNTIME_STARTUP_FAILED");
}
#endif

// REQ: RUNTIME-007
// REQ: RUNTIME-028
TEST(RuntimeDiagnosticBufferTest, PreservesCallerSelectedDeadlineSeverity) {
  RuntimeDiagnosticBuffer buffer{DiagnosticId{901U}, DiagnosticSource{}};
  buffer.append(DiagnosticId{241U},
                "DEADLINE_MISS",
                DiagnosticSource{.vertex = VertexId{1U}},
                "latest_start",
                nullptr,
                {},
                {},
                ::holoscan::DiagnosticSeverity::kWarning);

  ASSERT_EQ(buffer.diagnostics().size(), 1U);
  EXPECT_EQ(buffer.diagnostics().front().token, "DEADLINE_MISS");
  EXPECT_EQ(buffer.diagnostics().front().severity, ::holoscan::DiagnosticSeverity::kWarning);
}

#if GTEST_HAS_DEATH_TEST && (defined(__cpp_exceptions) || defined(_CPPUNWIND))
// REQ: RUNTIME-003
TEST(RuntimeFatalLogging, ThrowingDiagnosticCannotWeakenMandatoryTermination) {
  EXPECT_DEATH(
      {
        runtime_log_and_terminate_noexcept(
            [] { throw std::runtime_error{"injected Runtime logger failure"}; });
      },
      ".*");
}
#endif

}  // namespace
}  // namespace holoscan::detail
