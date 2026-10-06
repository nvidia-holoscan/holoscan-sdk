/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>

#include "diagnostic_context.hpp"

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan {
namespace {

std::string_view severity_name(DiagnosticSeverity severity) noexcept {
  switch (severity) {
    case DiagnosticSeverity::kWarning:
      return "warning";
    case DiagnosticSeverity::kError:
      return "error";
  }
  return "unknown";
}

std::string source_vertex_name(const ExecutionGraphView& graph, VertexId id,
                               std::string_view fallback = "unknown") {
  const auto vertices = graph.vertices();
  const auto vertex = std::ranges::find(vertices, id, &VertexRecord::id);
  if (vertex == vertices.end()) {
    return std::string{fallback};
  }
  return vertex->name.empty() ? "unnamed" : detail::diagnostic_operator_path(graph, id);
}

std::optional<std::string_view> source_port_name(const ExecutionGraphView& graph, PortId id) {
  if (!id) {
    return std::nullopt;
  }
  const auto ports = graph.ports();
  const auto port = std::ranges::find_if(
      ports, [id](const auto& candidate) { return candidate.id.value == id.value(); });
  if (port == ports.end()) {
    return std::nullopt;
  }
  return port->name;
}

template <typename AppendField>
void append_source_fields(const Diagnostic& diagnostic, const ExecutionGraphView& graph,
                          const AppendField& append_field) {
  if (diagnostic.id) {
    append_field("id", diagnostic.id.value);
  }
  if (diagnostic.source.graph_generation != 0U) {
    append_field("graph_generation", diagnostic.source.graph_generation);
  }
  if (diagnostic.source.graph) {
    append_field("graph", diagnostic.source.graph.value);
  }
  if (diagnostic.source.vertex) {
    const std::string_view fallback = diagnostic.detail && !diagnostic.detail->operator_path.empty()
                                          ? std::string_view{diagnostic.detail->operator_path}
                                          : std::string_view{"unknown"};
    append_field("vertex", source_vertex_name(graph, diagnostic.source.vertex, fallback));
    append_field("vertex_id", diagnostic.source.vertex.value);
  }
  if (diagnostic.source.port) {
    append_field("port", diagnostic.source.port.value());
    if (const auto name = source_port_name(graph, diagnostic.source.port)) {
      append_field("port_name", *name);
    }
  }
}

template <typename AppendField>
void append_detail_context(const Diagnostic& diagnostic, const ExecutionGraphView& graph,
                           const DiagnosticDetail& evidence, const AppendField& append_field) {
  if (!evidence.operator_path.empty()) {
    append_field(diagnostic.source.vertex ? "operator" : "component", evidence.operator_path);
  } else if (const auto path = detail::diagnostic_operator_path(graph, diagnostic.source.vertex);
             !path.empty()) {
    append_field("operator", path);
  }
  if (evidence.operation != DiagnosticOperation::kUnknown) {
    append_field("operation", magic_enum::enum_name(evidence.operation));
  }
  if (evidence.stage) {
    append_field("stage", magic_enum::enum_name(*evidence.stage));
  }
  if (evidence.phase != LifecycleTransitionPhase::kNone) {
    append_field("phase", magic_enum::enum_name(evidence.phase));
  }
  if (evidence.transaction_id != 0U) {
    append_field("transaction", evidence.transaction_id);
  }
  const bool has_cause =
      evidence.error.code != ErrorCode::kSuccess || !evidence.error.message.empty();
  if (has_cause || evidence.stage || evidence.role != DiagnosticFailureRole::kPrimary) {
    append_field("role", magic_enum::enum_name(evidence.role));
  }
  if (has_cause) {
    append_field("cause", runtime_error_message(evidence.error));
  }
  for (std::size_t index = 0U; index < std::min(evidence.causes.size(), std::size_t{3U}); ++index) {
    append_field("caused_by", runtime_error_message(evidence.causes[index]));
  }
}

template <typename AppendField>
void append_related_source(const DiagnosticDetail& evidence, const ExecutionGraphView& graph,
                           const AppendField& append_field) {
  const auto& source = evidence.related_source;
  if (source.graph_generation != 0U) {
    append_field("related_graph_generation", source.graph_generation);
  }
  if (source.graph) {
    append_field("related_graph", source.graph.value);
  }
  if (source.vertex) {
    append_field("related_vertex_id", source.vertex.value);
  }
  if (source.port) {
    append_field("related_port_id", source.port.value());
  }
  if (!evidence.related_operator_path.empty()) {
    append_field("related_operator", evidence.related_operator_path);
  } else if (source.vertex) {
    append_field("related_operator", source_vertex_name(graph, source.vertex));
  }
  if (!evidence.related_port_name.empty()) {
    append_field("related_port", evidence.related_port_name);
  } else if (const auto name = source_port_name(graph, source.port)) {
    append_field("related_port", *name);
  }
}

template <typename AppendField>
void append_detail_contract(const Diagnostic& diagnostic, const ExecutionGraphView& graph,
                            const DiagnosticDetail& evidence, const AppendField& append_field) {
  if (!evidence.port_name.empty()) {
    const auto compiled_name = source_port_name(graph, diagnostic.source.port);
    // Compare owned evidence before display escaping or truncation can make names look alike.
    if (!compiled_name || *compiled_name != evidence.port_name) {
      append_field(compiled_name && !compiled_name->empty() ? "captured_port_name" : "port_name",
                   evidence.port_name);
    }
  }
  append_related_source(evidence, graph, append_field);
  for (std::size_t index = 0U; index < std::min(evidence.mismatches.size(), std::size_t{16U});
       ++index) {
    const auto& mismatch = evidence.mismatches[index];
    append_field("field", mismatch.field);
    append_field("provided", mismatch.provided);
    append_field("required", mismatch.required);
  }
  if (evidence.primary_diagnostic_index) {
    append_field("primary_diagnostic_index", *evidence.primary_diagnostic_index);
  }
  if (evidence.truncated || evidence.causes.size() > 3U || evidence.mismatches.size() > 16U) {
    append_field("detail_truncated", true);
  }
}

}  // namespace

std::string format_diagnostic(const Diagnostic& diagnostic, const ExecutionGraphView& graph) {
  fmt::memory_buffer output;
  fmt::format_to(std::back_inserter(output),
                 "{} {}",
                 severity_name(diagnostic.severity),
                 detail::diagnostic_display_text(diagnostic.token, 256U));

  bool first_field = true;
  const auto append_field = [&output, &first_field](std::string_view name, const auto& value) {
    if constexpr (std::is_convertible_v<decltype(value), std::string_view>) {
      fmt::format_to(std::back_inserter(output),
                     "{}{}={}",
                     first_field ? " [" : ", ",
                     name,
                     detail::diagnostic_display_text(value));
    } else {
      fmt::format_to(std::back_inserter(output), "{}{}={}", first_field ? " [" : ", ", name, value);
    }
    first_field = false;
  };

  append_source_fields(diagnostic, graph, append_field);
  if (diagnostic.detail) {
    append_detail_context(diagnostic, graph, *diagnostic.detail, append_field);
    append_detail_contract(diagnostic, graph, *diagnostic.detail, append_field);
  }
  if (!diagnostic.service.empty()) {
    append_field("service", diagnostic.service);
  }
  if (!diagnostic.endpoint.empty()) {
    append_field("endpoint", diagnostic.endpoint);
  }
  if (!diagnostic.json_pointer.empty()) {
    append_field("json_pointer", diagnostic.json_pointer);
  }
  if (!diagnostic.required.empty()) {
    append_field("required", diagnostic.required);
  }
  if (!diagnostic.provided.empty()) {
    append_field("provided", diagnostic.provided);
  }
  if (diagnostic.cause_id && *diagnostic.cause_id) {
    append_field("cause_id", diagnostic.cause_id->value);
  }
  if (!first_field) {
    output.push_back(']');
  }
  return fmt::to_string(output);
}

namespace {

template <typename Emit>
void render_diagnostic_reports(std::span<const Diagnostic> diagnostics,
                               const ExecutionGraphView& graph, bool errors_enabled,
                               bool warnings_enabled, const Emit& emit) {
  constexpr std::size_t limit = 16384U;
  constexpr std::size_t marker_reserve = 96U;
  const auto selected = [=](const Diagnostic& diagnostic) {
    return diagnostic.severity == DiagnosticSeverity::kError ? errors_enabled : warnings_enabled;
  };
  const auto selected_count =
      static_cast<std::size_t>(std::ranges::count_if(diagnostics, selected));
  std::size_t text_size = 0U;
  std::size_t rendered = 0U;
  for (const bool errors_first : {true, false}) {
    if (!(errors_first ? errors_enabled : warnings_enabled)) {
      continue;
    }
    std::string report;
    for (std::size_t index = 0U; index < diagnostics.size(); ++index) {
      const auto& diagnostic = diagnostics[index];
      if ((diagnostic.severity == DiagnosticSeverity::kError) != errors_first) {
        continue;
      }
      // Preserve original span indices, including references into the other severity group.
      const auto line =
          fmt::format("{} [diagnostic_index={}]", format_diagnostic(diagnostic, graph), index);
      if (text_size != 0U && text_size < limit - marker_reserve) {
        ++text_size;
        if (!report.empty()) {
          report.push_back('\n');
        }
      }
      const auto remaining = limit - marker_reserve - text_size;
      const auto appended = std::min(line.size(), remaining);
      report.append(line, 0U, appended);
      text_size += appended;
      if (line.size() > remaining) {
        report += fmt::format("\n[diagnostic report truncated; {} records not fully rendered]",
                              selected_count - rendered);
        emit(errors_first, report);
        return;
      }
      ++rendered;
    }
    if (!report.empty()) {
      emit(errors_first, report);
    }
  }
}

}  // namespace

std::string format_diagnostics(std::span<const Diagnostic> diagnostics,
                               const ExecutionGraphView& graph) {
  std::string result;
  render_diagnostic_reports(diagnostics, graph, true, true, [&](bool, const std::string& report) {
    if (!result.empty() && result.size() < 16384U - 96U) {
      result.push_back('\n');
    }
    result += report;
  });
  return result;
}

void log_diagnostics(std::span<const Diagnostic> diagnostics,
                     const ExecutionGraphView& graph) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    if (diagnostics.empty()) {
      return;
    }
    const auto level = log_level();
    const bool errors_enabled =
        HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_ERROR && level <= LogLevel::ERROR;
    const bool warnings_enabled =
        HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_WARN && level <= LogLevel::WARN;
    render_diagnostic_reports(diagnostics,
                              graph,
                              errors_enabled,
                              warnings_enabled,
                              [](bool errors, const std::string& report) {
                                // The shared formatter uses this separator between severity
                                // groups; independent logger records already have that boundary.
                                std::string_view payload{report};
                                if (payload.starts_with('\n')) {
                                  payload.remove_prefix(1U);
                                }
                                if (errors) {
                                  HOLOSCAN_LOG_ERROR("{}", payload);
                                } else {
                                  HOLOSCAN_LOG_WARN("{}", payload);
                                }
                              });
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Best-effort presentation never changes the caller's retained result or failure.
    return;
  }
#endif
}

void log_diagnostics(const ExecutionPlan& plan) noexcept {
  log_diagnostics(plan.diagnostics(), plan.graph());
}

}  // namespace holoscan
