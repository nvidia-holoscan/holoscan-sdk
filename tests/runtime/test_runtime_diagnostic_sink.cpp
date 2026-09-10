/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
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

// REQ: RUNTIME-047
// REQ: RUNTIME-049
TEST(RuntimeDiagnosticBufferTest, RetainsBoundedManagedSignalEvidenceInParallelDetail) {
  constexpr GraphGeneration kGraphGeneration = 43U;
  RuntimeDiagnosticBuffer buffer{
      DiagnosticId{225U},
      DiagnosticSource{.graph_generation = kGraphGeneration},
  };
  const RuntimeDiagnosticDetail detail{
      .managed_signal =
          RuntimeManagedSignalDiagnostic{
              .present = true,
              .phase = RuntimeManagedSignalPhase::kAcceptSignal,
              .signal = RuntimeManagedSignal::kSigInt,
              .occurrence = RuntimeManagedSignalOccurrence::kFirst,
              .prior_disposition = RuntimeManagedSignalDisposition::kCustom,
              .census_reason = RuntimeManagedSignalCensusReason::kNone,
              .originally_blocked = true,
              .domain_generation = 7U,
              .binding_generation = 11U,
              .observed_thread_count = 1U,
              .os_error = 0,
          },
  };

  buffer.append(DiagnosticId{290U},
                "RUNTIME_MANAGED_SIGNAL_RECEIVED",
                DiagnosticSource{.graph_generation = kGraphGeneration},
                {},
                &detail,
                {},
                {},
                ::holoscan::DiagnosticSeverity::kWarning);

  ASSERT_EQ(buffer.diagnostics().size(), 1U);
  ASSERT_EQ(buffer.details().size(), 1U);
  EXPECT_EQ(buffer.diagnostics().front().id, DiagnosticId{290U});
  EXPECT_EQ(buffer.diagnostics().front().token, "RUNTIME_MANAGED_SIGNAL_RECEIVED");
  EXPECT_EQ(buffer.diagnostics().front().severity, ::holoscan::DiagnosticSeverity::kWarning);
  EXPECT_EQ(buffer.diagnostics().front().source.graph_generation, kGraphGeneration);
  const RuntimeManagedSignalDiagnostic& observed = buffer.details().front().managed_signal;
  EXPECT_TRUE(observed.present);
  EXPECT_EQ(observed.phase, RuntimeManagedSignalPhase::kAcceptSignal);
  EXPECT_EQ(observed.signal, RuntimeManagedSignal::kSigInt);
  EXPECT_EQ(observed.occurrence, RuntimeManagedSignalOccurrence::kFirst);
  EXPECT_EQ(observed.prior_disposition, RuntimeManagedSignalDisposition::kCustom);
  EXPECT_TRUE(observed.originally_blocked);
  EXPECT_EQ(observed.domain_generation, 7U);
  EXPECT_EQ(observed.binding_generation, 11U);
  EXPECT_EQ(observed.observed_thread_count, 1U);
  EXPECT_EQ(observed.os_error, 0);
}

// REQ: RUNTIME-047
// REQ: RUNTIME-049
TEST(RuntimeDiagnosticBufferTest, RetainsAllManagedSignalIdentifiersWithTypedDetails) {
  struct Case {
    DiagnosticId id;
    const char* token;
    ::holoscan::DiagnosticSeverity severity;
    RuntimeManagedSignalDiagnostic detail;
  };
  const std::array cases{
      Case{DiagnosticId{290U},
           "RUNTIME_MANAGED_SIGNAL_RECEIVED",
           ::holoscan::DiagnosticSeverity::kWarning,
           {.present = true,
            .phase = RuntimeManagedSignalPhase::kAcceptSignal,
            .signal = RuntimeManagedSignal::kSigInt,
            .occurrence = RuntimeManagedSignalOccurrence::kFirst,
            .prior_disposition = RuntimeManagedSignalDisposition::kCustom,
            .domain_generation = 3U,
            .binding_generation = 4U}},
      Case{DiagnosticId{291U},
           "RUNTIME_MANAGED_SIGNAL_OWNER_BUSY",
           ::holoscan::DiagnosticSeverity::kError,
           {.present = true,
            .phase = RuntimeManagedSignalPhase::kAcquireBinding,
            .domain_generation = 3U,
            .binding_generation = 4U}},
      Case{DiagnosticId{292U},
           "RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED",
           ::holoscan::DiagnosticSeverity::kError,
           {.present = true,
            .phase = RuntimeManagedSignalPhase::kThreadCensus,
            .census_reason = RuntimeManagedSignalCensusReason::kMultipleThreads,
            .observed_thread_count = 2U}},
      Case{DiagnosticId{293U},
           "RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
           ::holoscan::DiagnosticSeverity::kError,
           {.present = true, .phase = RuntimeManagedSignalPhase::kSnapshotMask, .os_error = 5}},
      Case{DiagnosticId{294U},
           "RUNTIME_MANAGED_SIGNAL_WAIT_FAILED",
           ::holoscan::DiagnosticSeverity::kError,
           {.present = true,
            .phase = RuntimeManagedSignalPhase::kPoll,
            .domain_generation = 3U,
            .binding_generation = 4U,
            .os_error = 5}},
      Case{DiagnosticId{295U},
           "RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED",
           ::holoscan::DiagnosticSeverity::kError,
           {.present = true,
            .phase = RuntimeManagedSignalPhase::kRestoreMask,
            .domain_generation = 3U,
            .binding_generation = 4U,
            .os_error = 5}},
      Case{DiagnosticId{296U},
           "RUNTIME_MANAGED_SIGNAL_ESCALATED",
           ::holoscan::DiagnosticSeverity::kWarning,
           {.present = true,
            .phase = RuntimeManagedSignalPhase::kDelegateSignal,
            .signal = RuntimeManagedSignal::kSigTerm,
            .occurrence = RuntimeManagedSignalOccurrence::kSecond,
            .prior_disposition = RuntimeManagedSignalDisposition::kIgnored,
            .domain_generation = 3U,
            .binding_generation = 4U}},
  };
  RuntimeDiagnosticBuffer buffer{DiagnosticId{225U}, {}};
  for (const Case& test_case : cases) {
    const RuntimeDiagnosticDetail detail{
        .present = true,
        .managed_signal = test_case.detail,
    };
    buffer.append(test_case.id, test_case.token, {}, {}, &detail, {}, {}, test_case.severity);
  }
  ASSERT_EQ(buffer.diagnostics().size(), cases.size());
  ASSERT_EQ(buffer.details().size(), cases.size());
  for (std::size_t index = 0U; index < cases.size(); ++index) {
    EXPECT_EQ(buffer.diagnostics()[index].id, cases[index].id);
    EXPECT_EQ(buffer.diagnostics()[index].token, cases[index].token);
    EXPECT_EQ(buffer.diagnostics()[index].severity, cases[index].severity);
    EXPECT_EQ(buffer.details()[index].managed_signal.phase, cases[index].detail.phase);
    EXPECT_EQ(buffer.details()[index].managed_signal.signal, cases[index].detail.signal);
    EXPECT_EQ(buffer.details()[index].managed_signal.occurrence, cases[index].detail.occurrence);
    EXPECT_EQ(buffer.details()[index].managed_signal.census_reason,
              cases[index].detail.census_reason);
    EXPECT_EQ(buffer.details()[index].managed_signal.os_error, cases[index].detail.os_error);
  }
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
