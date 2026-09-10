/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <iterator>
#include <span>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_plan.hpp>

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

}  // namespace

std::string format_diagnostic(const Diagnostic& diagnostic, const ExecutionGraphView& graph) {
  fmt::memory_buffer output;
  fmt::format_to(
      std::back_inserter(output), "{} {}", severity_name(diagnostic.severity), diagnostic.token);

  bool first_field = true;
  const auto append_field = [&output, &first_field](std::string_view name, const auto& value) {
    fmt::format_to(std::back_inserter(output), "{}{}={}", first_field ? " [" : ", ", name, value);
    first_field = false;
  };

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
    const auto vertices = graph.vertices();
    const auto vertex = std::find_if(vertices.begin(), vertices.end(), [&](const auto& candidate) {
      return candidate.id == diagnostic.source.vertex;
    });
    const std::string_view name =
        vertex == vertices.end()
            ? std::string_view{"unknown"}
            : (vertex->name.empty() ? std::string_view{"unnamed"} : std::string_view{vertex->name});
    append_field("vertex", name);
    append_field("vertex_id", diagnostic.source.vertex.value);
  }
  if (diagnostic.source.port) {
    append_field("port", diagnostic.source.port.value());
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

}  // namespace holoscan
