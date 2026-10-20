/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>

#include <gtest/gtest.h>

#include <holoscan/core/diagnostic.hpp>

#include "run_session_test_support.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/parameter_update_coordinator.hpp"
#include "runtime/physical_readiness_binding.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan::test::run_session {

void expect_diagnostic_sequences_equal(std::span<const Diagnostic> expected,
                                       std::span<const Diagnostic> actual) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(actual[index].id, expected[index].id);
    EXPECT_EQ(actual[index].severity, expected[index].severity);
    EXPECT_EQ(actual[index].token, expected[index].token);
    EXPECT_EQ(actual[index].source.graph_generation, expected[index].source.graph_generation);
    EXPECT_EQ(actual[index].source.graph, expected[index].source.graph);
    EXPECT_EQ(actual[index].source.vertex, expected[index].source.vertex);
    EXPECT_EQ(actual[index].source.port, expected[index].source.port);
    EXPECT_EQ(actual[index].service, expected[index].service);
    EXPECT_EQ(actual[index].endpoint, expected[index].endpoint);
    EXPECT_EQ(actual[index].json_pointer, expected[index].json_pointer);
    EXPECT_EQ(actual[index].required, expected[index].required);
    EXPECT_EQ(actual[index].provided, expected[index].provided);
    EXPECT_EQ(actual[index].cause_id, expected[index].cause_id);
    const auto& actual_detail = actual[index].detail;
    const auto& expected_detail = expected[index].detail;
    ASSERT_EQ(actual_detail.has_value(), expected_detail.has_value());
    if (!actual_detail.has_value() || !expected_detail.has_value()) {
      continue;
    }
    const auto& observed = *actual_detail;
    const auto& wanted = *expected_detail;
    EXPECT_EQ(observed.operation, wanted.operation);
    EXPECT_EQ(observed.error.code, wanted.error.code);
    EXPECT_EQ(observed.error.message, wanted.error.message);
    EXPECT_EQ(observed.stage, wanted.stage);
    EXPECT_EQ(observed.phase, wanted.phase);
    EXPECT_EQ(observed.role, wanted.role);
    EXPECT_EQ(observed.transaction_id, wanted.transaction_id);
    EXPECT_EQ(observed.operator_path, wanted.operator_path);
    EXPECT_EQ(observed.port_name, wanted.port_name);
    EXPECT_EQ(observed.primary_diagnostic_index, wanted.primary_diagnostic_index);
    EXPECT_EQ(observed.truncated, wanted.truncated);
    ASSERT_EQ(observed.causes.size(), wanted.causes.size());
    for (std::size_t cause = 0U; cause < wanted.causes.size(); ++cause) {
      EXPECT_EQ(observed.causes[cause].code, wanted.causes[cause].code);
      EXPECT_EQ(observed.causes[cause].message, wanted.causes[cause].message);
    }
  }
}

void expect_terminal_compute_call_progress(const RunSession& session, const ExecutionPlan& plan,
                                           std::uint64_t configured_limit,
                                           std::uint64_t successful_compute_calls,
                                           bool limit_reached) {
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  const auto& rule = plan.operator_completion_rules().front();
  const auto& progress = session.operator_completion_progress().front();
  EXPECT_TRUE(rule.owner.plan_bound());
  EXPECT_TRUE(progress.owner.run_bound());
  EXPECT_EQ(progress.owner.id, rule.owner.id);
  EXPECT_EQ(progress.owner.plan_generation, rule.owner.plan_generation);
  EXPECT_EQ(progress.kind, OperatorCompletionRuleKind::kComputeCalls);
  EXPECT_EQ(progress.configured_limit, configured_limit);
  EXPECT_EQ(progress.successful_compute_calls, successful_compute_calls);
  EXPECT_EQ(progress.limit_reached, limit_reached);
}

}  // namespace holoscan::test::run_session
