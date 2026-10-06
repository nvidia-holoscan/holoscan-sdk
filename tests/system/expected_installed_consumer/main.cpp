/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Imported Holoscan targets must leave the compile-time logging threshold under caller control.
// Check before logger.hpp supplies its documented default.
#if defined(HOLOSCAN_LOG_ACTIVE_LEVEL)
#error "installed Holoscan targets must not define HOLOSCAN_LOG_ACTIVE_LEVEL for consumers"
#endif

#include <string>
#include <string_view>

#include <holoscan/core/diagnostic.hpp>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/logger/logger.hpp>
#include "diagnostic_policy_peer_export.h"

static_assert(HOLOSCAN_LOG_ACTIVE_LEVEL == HOLOSCAN_LOG_LEVEL_TRACE,
              "logger.hpp must retain its caller-owned default threshold");

extern "C" DIAGNOSTIC_POLICY_PEER_EXPORT bool diagnostic_policy_peer_get();
extern "C" DIAGNOSTIC_POLICY_PEER_EXPORT void diagnostic_policy_peer_set(bool enabled);
extern "C" DIAGNOSTIC_POLICY_PEER_EXPORT bool diagnostic_policy_peer_memory_failures();

// REQ: EXPECTED-032
static_assert(noexcept(holoscan::set_automatic_diagnostic_logging_enabled(false)));
static_assert(noexcept(holoscan::automatic_diagnostic_logging_enabled()));

namespace {

holoscan::expected<int, holoscan::Error> make_value(const bool fail) {
  if (fail) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotFound, "installed value failure"});
  }
  return 21;
}

holoscan::expected<void, holoscan::Error> validate_value(const int value, const bool fail) {
  if (fail || value != 21) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "installed void failure"});
  }
  return {};
}

holoscan::expected<int, holoscan::Error> propagate_try_forms(const bool value_failure,
                                                             const bool void_failure) {
  HOLOSCAN_TRY(auto value, make_value(value_failure));
  HOLOSCAN_TRY(validate_value(value, void_failure));
  return value * 2;
}

holoscan::expected<int, holoscan::Error> propagate_explicit_forms(const bool value_failure,
                                                                  const bool void_failure) {
  int value = 0;
  HOLOSCAN_ASSIGN_OR_RETURN(value, make_value(value_failure));
  HOLOSCAN_RETURN_IF_ERROR(validate_value(value, void_failure));
  return value * 2;
}

bool has_expected_failure(const holoscan::expected<int, holoscan::Error>& result,
                          const holoscan::ErrorCode code) {
  return !result && result.error().code == code;
}

}  // namespace

// REQ: EXPECTED-018
// REQ: EXPECTED-021
// REQ: EXPECTED-030
// REQ: EXPECTED-033
// REQ: EXPECTED-032
int main(int argc, char** argv) {
  if (argc == 2) {
    const bool preference = std::string_view(argv[1]) == "ON";
    holoscan::set_automatic_diagnostic_logging_enabled(preference);
    holoscan::set_log_level(holoscan::LogLevel::ERROR);
    return diagnostic_policy_peer_memory_failures() ? 0 : 3;
  }
  holoscan::set_log_level(holoscan::LogLevel::OFF);
  HOLOSCAN_LOG_INFO("installed package consumer linked the public logger surface");

  const holoscan::Diagnostic diagnostic{
      .token = "RUNTIME_OPERATOR_CALLBACK_FAILED",
      .detail =
          holoscan::DiagnosticDetail{
              .operation = holoscan::DiagnosticOperation::kCompute,
              .error = holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                       "installed diagnostic cause"},
              .operator_path = "installed/consumer",
              .related_source = {.graph_generation = 3U,
                                 .graph = holoscan::GraphId{2U},
                                 .vertex = holoscan::VertexId{7U},
                                 .port = holoscan::PortId{5U}},
          },
  };
  holoscan::set_automatic_diagnostic_logging_enabled(false);
  if (diagnostic_policy_peer_get()) {
    return 3;
  }
  diagnostic_policy_peer_set(true);
  if (!holoscan::automatic_diagnostic_logging_enabled()) {
    return 4;
  }
  diagnostic_policy_peer_set(false);
  if (holoscan::automatic_diagnostic_logging_enabled()) {
    return 5;
  }
  holoscan::log_diagnostics(holoscan::ExecutionPlan{});
  holoscan::log_diagnostics(std::span<const holoscan::Diagnostic>{&diagnostic, 1U},
                            holoscan::ExecutionGraphView{});
  holoscan::set_automatic_diagnostic_logging_enabled(true);
  const auto report = holoscan::format_diagnostic(diagnostic, holoscan::ExecutionGraphView{});
  const auto full_report = holoscan::format_diagnostics(
      std::span<const holoscan::Diagnostic>{&diagnostic, 1U}, holoscan::ExecutionGraphView{});
  if (full_report != report + " [diagnostic_index=0]" ||
      report.find("installed diagnostic cause") == std::string::npos ||
      report.find("installed/consumer") == std::string::npos ||
      report.find("related_vertex_id=7") == std::string::npos ||
      report.find("related_port_id=5") == std::string::npos) {
    return 2;
  }

  const auto try_success = propagate_try_forms(false, false);
  const auto try_value_failure = propagate_try_forms(true, false);
  const auto try_void_failure = propagate_try_forms(false, true);
  const auto explicit_success = propagate_explicit_forms(false, false);
  const auto explicit_value_failure = propagate_explicit_forms(true, false);
  const auto explicit_void_failure = propagate_explicit_forms(false, true);

  return try_success && *try_success == 42 &&
                 has_expected_failure(try_value_failure, holoscan::ErrorCode::kNotFound) &&
                 has_expected_failure(try_void_failure, holoscan::ErrorCode::kInvalidArgument) &&
                 explicit_success && *explicit_success == 42 &&
                 has_expected_failure(explicit_value_failure, holoscan::ErrorCode::kNotFound) &&
                 has_expected_failure(explicit_void_failure, holoscan::ErrorCode::kInvalidArgument)
             ? 0
             : 1;
}
