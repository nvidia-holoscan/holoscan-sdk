/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler_diagnostic.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

#include "../core/diagnostic_context.hpp"
#include "../core/diagnostic_registry.hpp"
#include "diagnostic_detail.hpp"
#include "execution_plan_state.hpp"
#include "plan_lowering_diagnostic.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] const StructuralDiagnosticDescriptor* diagnostic_for_token(
    std::string_view token) noexcept {
  for (const auto& descriptor : kStructuralDiagnosticRegistry) {
    if (descriptor.token == token) {
      return &descriptor;
    }
  }
  return nullptr;
}

[[nodiscard]] DiagnosticSource complete_diagnostic_source(const ExecutionPlanState& state,
                                                          DiagnosticSource source) noexcept {
  if (source.graph_generation == 0U) {
    source.graph_generation = state.graph.generation();
  }
  if (!source.graph && source.vertex) {
    const auto owner = std::find_if(state.graph.vertices().begin(),
                                    state.graph.vertices().end(),
                                    [&](const auto& vertex) { return vertex.id == source.vertex; });
    if (owner != state.graph.vertices().end()) {
      source.graph = owner->graph;
    }
  }
  if (!source.graph && source.port) {
    const auto port = std::find_if(
        state.graph.ports().begin(), state.graph.ports().end(), [&](const auto& candidate) {
          return candidate.id.value == source.port.value();
        });
    if (port != state.graph.ports().end()) {
      source.vertex = port->owner;
      const auto owner = std::find_if(
          state.graph.vertices().begin(), state.graph.vertices().end(), [&](const auto& vertex) {
            return vertex.id == source.vertex;
          });
      if (owner != state.graph.vertices().end()) {
        source.graph = owner->graph;
      }
    }
  }
  if (!source.graph) {
    const auto root = std::find_if(state.graph.graphs().begin(),
                                   state.graph.graphs().end(),
                                   [](const auto& graph) { return !graph.parent.has_value(); });
    if (root != state.graph.graphs().end()) {
      source.graph = root->id;
    }
  }
  return source;
}

}  // namespace

Diagnostic& append_compiler_diagnostic(
    ExecutionPlanState& state, std::string_view token, DiagnosticSource source,
    std::optional<holoscan::DiagnosticSeverity> severity_override, std::string_view json_pointer,
    std::string_view required, std::string_view provided, const DiagnosticDetail* evidence) {
  source = complete_diagnostic_source(state, source);
  const auto* descriptor = diagnostic_for_token(token);
  if (descriptor == nullptr) {
    descriptor = diagnostic_for_token("COMPILER_DIAGNOSTIC_TOKEN_UNREGISTERED");
    token = descriptor->token;
  }
  const auto duplicate =
      std::find_if(state.diagnostics.begin(), state.diagnostics.end(), [&](const auto& item) {
        return item.token == token && item.source.graph_generation == source.graph_generation &&
               item.source.graph == source.graph && item.source.vertex == source.vertex &&
               item.source.port == source.port && item.json_pointer == json_pointer &&
               item.required == required && item.provided == provided;
      });
  if (duplicate != state.diagnostics.end()) {
    enrich_diagnostic_detail(duplicate->detail, evidence);
    return *duplicate;
  }

  const auto severity =
      severity_override.value_or(descriptor->severity == detail::DiagnosticSeverity::kWarning
                                     ? holoscan::DiagnosticSeverity::kWarning
                                     : holoscan::DiagnosticSeverity::kError);
  state.diagnostics.push_back(Diagnostic{
      .id = DiagnosticId{static_cast<std::uint32_t>(descriptor->id)},
      .severity = severity,
      .token = std::string(token),
      .source = source,
      .json_pointer = std::string(json_pointer),
      .required = std::string(required),
      .provided = std::string(provided),
  });
  auto& retained = state.diagnostics.back();
  enrich_diagnostic_detail(retained.detail, evidence);
  return retained;
}

void publish_derivation_diagnostic(ExecutionPlanState& state,
                                   const PlanLoweringDiagnostic& diagnostic) {
  auto& retained = append_compiler_diagnostic(state,
                                              diagnostic.token,
                                              diagnostic.source,
                                              std::nullopt,
                                              diagnostic.json_pointer,
                                              diagnostic.required,
                                              diagnostic.provided,
                                              diagnostic.detail ? &*diagnostic.detail : nullptr);
  if (!retained.detail) {
    retained.detail.emplace(DiagnosticDetail{.operation = DiagnosticOperation::kCompile,
                                             .error = Error{ErrorCode::kSuccess}});
  }
  // Reserve half the existing Core context bound for the operator and 255 bytes for the port.
  // Names are copied on the cold rejection path and remain usable after the Graph retires.
  constexpr std::size_t kOperatorPathLimit = DiagnosticCapture::kTextCapacity / 2U;
  constexpr std::size_t kPortNameLimit = 255U;
  auto& detail = *retained.detail;
  try {
    if (detail.operator_path.empty()) {
      copy_diagnostic_context(detail.operator_path,
                              diagnostic_operator_path(state.graph, retained.source.vertex),
                              detail,
                              kOperatorPathLimit);
    }
    if (detail.port_name.empty() && retained.source.port) {
      const auto ports = state.graph.ports();
      const auto port = std::find_if(ports.begin(), ports.end(), [&](const auto& candidate) {
        return candidate.id.value == retained.source.port.value() &&
               candidate.owner == retained.source.vertex;
      });
      if (port != ports.end()) {
        copy_diagnostic_context(detail.port_name, port->name, detail, kPortNameLimit);
      }
    }
  } catch (...) {
    detail.truncated = true;
  }
}

}  // namespace holoscan::detail
