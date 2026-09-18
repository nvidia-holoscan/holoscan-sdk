/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/diagnostic.hpp>

#include "diagnostic_detail.hpp"

namespace holoscan::detail {

/// Preserve one stable lowering diagnostic and its bounded canonical evidence.
struct PlanLoweringDiagnostic {
  std::string token;
  DiagnosticSource source{};
  std::string json_pointer;
  std::string required;
  std::string provided;
  std::optional<DiagnosticDetail> detail;
};

/// Return whether two lowering diagnostics contain the same complete value.
[[nodiscard]] inline bool plan_lowering_diagnostic_exactly_equal(
    const PlanLoweringDiagnostic& left, const PlanLoweringDiagnostic& right) noexcept {
  return left.token == right.token &&
         left.source.graph_generation == right.source.graph_generation &&
         left.source.graph == right.source.graph && left.source.vertex == right.source.vertex &&
         left.source.port == right.source.port && left.json_pointer == right.json_pointer &&
         left.required == right.required && left.provided == right.provided;
}

/// Return whether one lowering diagnostic precedes another in canonical plan order.
[[nodiscard]] inline bool plan_lowering_diagnostic_less(const PlanLoweringDiagnostic& left,
                                                        const PlanLoweringDiagnostic& right) {
  const auto key = [](const PlanLoweringDiagnostic& diagnostic) {
    return std::tuple{std::string_view{diagnostic.token},
                      diagnostic.source.graph_generation,
                      diagnostic.source.graph.value,
                      diagnostic.source.vertex.value,
                      diagnostic.source.port.value(),
                      std::string_view{diagnostic.json_pointer},
                      std::string_view{diagnostic.required},
                      std::string_view{diagnostic.provided}};
  };
  return key(left) < key(right);
}

/// Sort lowering diagnostics canonically and remove complete duplicates.
inline void canonicalize_plan_lowering_diagnostics(
    std::vector<PlanLoweringDiagnostic>& diagnostics) {
  std::stable_sort(diagnostics.begin(), diagnostics.end(), plan_lowering_diagnostic_less);
  auto retained = diagnostics.begin();
  for (auto current = diagnostics.begin(); current != diagnostics.end(); ++current) {
    if (retained != diagnostics.begin() &&
        plan_lowering_diagnostic_exactly_equal(*(retained - 1), *current)) {
      const auto& incoming = current->detail;
      enrich_diagnostic_detail((retained - 1)->detail, incoming ? &*incoming : nullptr);
    } else {
      if (retained != current) {
        *retained = std::move(*current);
      }
      ++retained;
    }
  }
  diagnostics.erase(retained, diagnostics.end());
}

/// Append one lowering diagnostic unless the same complete value is already present.
[[nodiscard]] inline bool append_unique_plan_lowering_diagnostic(
    std::vector<PlanLoweringDiagnostic>& diagnostics, PlanLoweringDiagnostic diagnostic) {
  const auto found =
      std::find_if(diagnostics.begin(), diagnostics.end(), [&](const auto& existing) {
        return plan_lowering_diagnostic_exactly_equal(existing, diagnostic);
      });
  if (found != diagnostics.end()) {
    enrich_diagnostic_detail(found->detail, diagnostic.detail ? &*diagnostic.detail : nullptr);
    return false;
  }
  diagnostics.push_back(std::move(diagnostic));
  return true;
}

}  // namespace holoscan::detail
