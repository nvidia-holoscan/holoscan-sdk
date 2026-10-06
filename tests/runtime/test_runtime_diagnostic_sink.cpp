/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
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
// REQ: RUNTIME-054
// REQ: RUNTIME-060
// REQ: EXPECTED-030
TEST(RuntimeDiagnosticBufferTest, AllocationFailurePreservesPrimaryEvidence) {
  RuntimeDiagnosticBuffer buffer{DiagnosticId{901U}, DiagnosticSource{}};
  RuntimeDiagnosticDetail detail{.present = true};
  detail.capture.emplace();
  detail.capture->append(ErrorCode::kInvalidArgument, "original image format error");
  detail.capture->append(ErrorCode::kFailure, "nested backend error");
  // Key setup: ScopedAllocationFailure fails the nested cause's optional allocation after the
  //            pre-reserved primary message and structural identity have already been captured.
  {
    const test::ScopedAllocationFailure fail_next_allocation{0U};
    buffer.append(DiagnosticId{1U},
                  "RUNTIME_LIFECYCLE_CALLBACK_FAILED",
                  DiagnosticSource{.vertex = VertexId{7U}},
                  {},
                  &detail);
  }
  // Key assertion: the primary failure remains inspectable when secondary detail cannot allocate.
  ASSERT_EQ(buffer.diagnostics().size(), 1U);
  const auto& retained_detail = buffer.diagnostics().front().detail;
  if (!retained_detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(retained_detail->error.message, "original image format error");
  EXPECT_EQ(retained_detail->error.code, ErrorCode::kInvalidArgument);
  EXPECT_TRUE(retained_detail->truncated);
  EXPECT_EQ(buffer.diagnostics().front().source.vertex, VertexId{7U});

  buffer.append(DiagnosticId{2U}, "RUNTIME_STARTUP_FAILED", DiagnosticSource{});
  ASSERT_EQ(buffer.diagnostics().size(), 2U);
  ASSERT_EQ(buffer.details().size(), 2U);
  EXPECT_EQ(buffer.diagnostics().back().id, DiagnosticId{2U});
  EXPECT_EQ(buffer.diagnostics().back().token, "RUNTIME_STARTUP_FAILED");
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

TEST(RuntimeDiagnosticBufferTest, WarningsCannotConsumePrimaryFailureCapacity) {
  RuntimeDiagnosticBuffer buffer{DiagnosticId{901U}, DiagnosticSource{}};
  for (std::size_t index = 0U; index < 100U; ++index) {
    buffer.append(
        DiagnosticId{1U}, "warning", {}, {}, nullptr, {}, {}, DiagnosticSeverity::kWarning);
  }
  const auto prefix = buffer.diagnostics();
  EXPECT_EQ(prefix.size(), kRuntimeStructuralDiagnosticCapacity - 2U);
  const RuntimeDiagnosticDetail failure{.present = true,
                                        .error = Error{ErrorCode::kFailure, "root cause"}};
  buffer.append(DiagnosticId{2U}, "RUNTIME_OPERATOR_CALLBACK_FAILED", {}, {}, &failure);
  buffer.finalize();
  ASSERT_EQ(buffer.diagnostics().size(), kRuntimeStructuralDiagnosticCapacity);
  const auto& failure_detail = buffer.diagnostics()[prefix.size()].detail;
  if (!failure_detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(failure_detail->error.message, "root cause");
  EXPECT_EQ(buffer.diagnostics().back().token, "RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED");
  EXPECT_EQ(prefix.front().token, "warning");
  EXPECT_EQ(buffer.diagnostics().back().severity, DiagnosticSeverity::kWarning);
}

TEST(RuntimeDiagnosticBufferTest, WarningOnlyOverflowRemainsAdvisoryAndImmutable) {
  for (const std::size_t warnings : {0U, 14U, 15U, 100U}) {
    SCOPED_TRACE(warnings);
    RuntimeDiagnosticBuffer buffer{DiagnosticId{901U}, DiagnosticSource{}};
    for (std::size_t index = 0U; index < warnings; ++index) {
      buffer.append(
          DiagnosticId{1U}, "warning", {}, {}, nullptr, {}, {}, DiagnosticSeverity::kWarning);
    }
    const auto prefix = buffer.diagnostics();
    buffer.finalize();
    const auto size = buffer.diagnostics().size();
    EXPECT_EQ(size, std::min(warnings, std::size_t{15U}));
    for (const auto& diagnostic : buffer.diagnostics()) {
      EXPECT_EQ(diagnostic.severity, DiagnosticSeverity::kWarning);
    }
    for (const auto& diagnostic : prefix) {
      EXPECT_EQ(diagnostic.token, "warning");
    }
    buffer.finalize();
    buffer.append(DiagnosticId{2U}, "late error", {});
    EXPECT_EQ(buffer.diagnostics().size(), size);
  }
}

TEST(RuntimeDiagnosticBufferTest, ErrorLossAfterWarningsRemainsAnError) {
  RuntimeDiagnosticBuffer buffer{DiagnosticId{901U}, DiagnosticSource{}};
  for (std::size_t index = 0U; index < 15U; ++index) {
    buffer.append(
        DiagnosticId{1U}, "warning", {}, {}, nullptr, {}, {}, DiagnosticSeverity::kWarning);
  }
  buffer.append(DiagnosticId{2U}, "first failure", {});
  buffer.append(DiagnosticId{3U}, "lost failure", {});
  buffer.finalize();
  ASSERT_EQ(buffer.diagnostics().size(), kRuntimeStructuralDiagnosticCapacity);
  EXPECT_EQ(buffer.diagnostics()[14U].token, "first failure");
  EXPECT_EQ(buffer.diagnostics().back().severity, DiagnosticSeverity::kError);
}

TEST(RuntimeDiagnosticBufferTest, ExplicitCleanupRolesLinkOnlyToRetainedFailures) {
  for (const bool primary : {false, true}) {
    RuntimeDiagnosticBuffer buffer{DiagnosticId{901U}, DiagnosticSource{}};
    buffer.append(
        DiagnosticId{1U}, "warning", {}, {}, nullptr, {}, {}, DiagnosticSeverity::kWarning);
    if (primary) {
      buffer.append(DiagnosticId{2U}, "primary", {});
    }
    RuntimeDiagnosticDetail cleanup{.present = true,
                                    .operation = DiagnosticOperation::kRuntime,
                                    .phase = LifecycleTransitionPhase::kForwardRollback,
                                    .role = DiagnosticFailureRole::kTeardown};
    const auto first_cleanup = buffer.diagnostics().size();
    buffer.append(DiagnosticId{3U}, "cleanup", {}, {}, &cleanup);
    const auto& first_detail = buffer.diagnostics().back().detail;
    if (!first_detail.has_value()) {
      FAIL() << "Expected diagnostic detail";
    }
    const auto& first = *first_detail;
    EXPECT_EQ(first.role, DiagnosticFailureRole::kTeardown);
    EXPECT_EQ(first.error.code, ErrorCode::kSuccess);
    EXPECT_EQ(first.primary_diagnostic_index,
              primary ? std::optional<std::uint32_t>{1U} : std::nullopt);
    buffer.append(DiagnosticId{4U}, "another cleanup", {}, {}, &cleanup);
    const auto& second_detail = buffer.diagnostics().back().detail;
    if (!second_detail.has_value()) {
      FAIL() << "Expected diagnostic detail";
    }
    EXPECT_EQ(second_detail->primary_diagnostic_index, primary ? 1U : first_cleanup);
    cleanup.transaction_id = 7U;
    buffer.append(DiagnosticId{5U}, "unrelated transaction", {}, {}, &cleanup);
    const auto& transaction_detail = buffer.diagnostics().back().detail;
    if (!transaction_detail.has_value()) {
      FAIL() << "Expected diagnostic detail";
    }
    EXPECT_FALSE(transaction_detail->primary_diagnostic_index);
  }
}

// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RuntimeDiagnosticBufferTest, ComputeCleanupLinksItsOwnerDespiteInterleavedFailures) {
  RuntimeDiagnosticBuffer buffer{DiagnosticId{901U}, DiagnosticSource{}};
  const RuntimeDiagnosticDetail callback{
      .present = true,
      .error = Error{ErrorCode::kFailure, "compute failed"},
      .operation = DiagnosticOperation::kCompute,
  };
  // Owner A fails before B. B cleanup must follow B, even if A cleanup interleaves.
  buffer.append(DiagnosticId{1U}, "callback", {.vertex = VertexId{1U}}, {}, &callback);
  buffer.append(DiagnosticId{1U}, "callback", {.vertex = VertexId{2U}}, {}, &callback);
  const RuntimeDiagnosticDetail cleanup{
      .present = true,
      .error = Error{ErrorCode::kNotSupported},
      .operation = DiagnosticOperation::kRuntime,
      .role = DiagnosticFailureRole::kTeardown,
      .follows_compute_failure = true,
  };
  buffer.append(DiagnosticId{2U}, "restore", {.vertex = VertexId{1U}}, {}, &cleanup);
  buffer.append(DiagnosticId{2U}, "restore", {.vertex = VertexId{2U}}, {}, &cleanup);
  buffer.append(DiagnosticId{2U}, "restore", {.vertex = VertexId{3U}}, {}, &cleanup);
  const auto diagnostics = buffer.diagnostics();
  ASSERT_EQ(diagnostics.size(), 5U);
  const auto& cleanup_a = diagnostics[2].detail;
  if (!cleanup_a.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  const auto& cleanup_b = diagnostics[3].detail;
  if (!cleanup_b.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  const auto& cleanup_without_primary = diagnostics[4].detail;
  if (!cleanup_without_primary.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(cleanup_a->primary_diagnostic_index, 0U);
  EXPECT_EQ(cleanup_b->primary_diagnostic_index, 1U);
  EXPECT_FALSE(cleanup_without_primary->primary_diagnostic_index);
}

// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(RuntimeDiagnosticBufferTest, LinksRepeatedTokensWithinTheirOwnRestartTransaction) {
  RuntimeDiagnosticBuffer buffer{DiagnosticId{901U}, DiagnosticSource{}};
  for (const LifecycleTransactionId transaction : {7U, 9U}) {
    RuntimeDiagnosticDetail failure{
        .present = true,
        .error = Error{ErrorCode::kLifecycleStageFailed, "candidate failed"},
        .operation = DiagnosticOperation::kLifecycle,
        .stage = LifecycleStage::kStart,
        .phase = LifecycleTransitionPhase::kRestartForward,
        .transaction_id = transaction,
    };
    buffer.append(DiagnosticId{181U},
                  "RUNTIME_LIFECYCLE_CALLBACK_FAILED",
                  {},
                  {},
                  &failure,
                  {},
                  {},
                  DiagnosticSeverity::kWarning);
    failure.phase = LifecycleTransitionPhase::kForwardRollback;
    buffer.append(DiagnosticId{181U},
                  "RUNTIME_LIFECYCLE_CALLBACK_FAILED",
                  {},
                  {},
                  &failure,
                  {},
                  {},
                  DiagnosticSeverity::kWarning);
  }
  const auto diagnostics = buffer.diagnostics();
  ASSERT_EQ(diagnostics.size(), 4U);
  const auto& first_recovery = diagnostics[1].detail;
  if (!first_recovery.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  const auto& cleanup_b = diagnostics[3].detail;
  if (!cleanup_b.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(first_recovery->primary_diagnostic_index, 0U);
  EXPECT_EQ(cleanup_b->primary_diagnostic_index, 2U);
  EXPECT_EQ(first_recovery->transaction_id, 7U);
  EXPECT_EQ(cleanup_b->transaction_id, 9U);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
TEST(DiagnosticCaptureTest, CapturesNestedCauseWithoutAllocation) {
  DiagnosticCapture capture;
  try {
    try {
      throw RuntimeError{ErrorCode::kInvalidArgument, "RGB is not RGBA"};
    } catch (...) {
      std::throw_with_nested(std::runtime_error{"visualization initialization failed"});
    }
  } catch (...) {
    const test::ScopedAllocationFailure fail_allocation{0U};
    capture.capture_exception();
  }
  ASSERT_EQ(capture.count, 2U);
  const auto detail = capture.detail(DiagnosticOperation::kLifecycle);
  EXPECT_EQ(detail.error.message, "visualization initialization failed");
  ASSERT_EQ(detail.causes.size(), 1U);
  EXPECT_EQ(detail.causes.front().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(detail.causes.front().message, "RGB is not RGBA");
  EXPECT_FALSE(detail.truncated);
}

TEST(DiagnosticCaptureTest, UnknownExceptionAndTruncationAreExplicit) {
  DiagnosticCapture capture;
  try {
    throw 42;
  } catch (...) {
    capture.capture_exception();
  }
  EXPECT_EQ(capture.detail(DiagnosticOperation::kCompute).error.message, "non-standard exception");
  capture.clear();
  capture.append(ErrorCode::kFailure, std::string(8192U, 'x'));
  EXPECT_EQ(capture.size, DiagnosticCapture::kTextCapacity);
  EXPECT_TRUE(capture.truncated);
  capture.clear();
  EXPECT_EQ(capture.count, 0U);
  EXPECT_FALSE(capture.truncated);
}
#endif

}  // namespace
}  // namespace holoscan::detail
