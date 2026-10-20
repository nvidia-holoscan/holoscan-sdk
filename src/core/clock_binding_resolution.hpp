/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

#include <holoscan/core/graph_ids.hpp>
#include <holoscan/core/graph_records.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan::detail {

enum class AuthoredClockResolutionStatus : std::uint8_t {
  kResolved,
  kNotFound,
  kImportUnbound,
  kBindingInvalid,
};

struct AuthoredClockResolution {
  AuthoredClockResolutionStatus status{AuthoredClockResolutionStatus::kNotFound};
  const ClockDeclarationRecord* declaration{};

  [[nodiscard]] bool resolved() const noexcept {
    return status == AuthoredClockResolutionStatus::kResolved && declaration != nullptr;
  }
};

/// Resolve one Graph-scoped authored clock key through explicit import records.
///
/// This helper runs only while compiling or reconstructing immutable state. It
/// deliberately performs no ancestor-name lookup: every scope transition must
/// be represented by one exact immediate-parent binding.
[[nodiscard]] inline AuthoredClockResolution resolve_authored_clock(
    std::span<const GraphRecord> graphs, std::span<const ClockDeclarationRecord> clocks,
    std::span<const ClockImportAuthoringRecord> imports,
    std::span<const EmbeddedClockBindingAuthoringRecord> bindings, GraphId initial_graph,
    const ClockKey& initial_key) noexcept {
  GraphId graph = initial_graph;
  const ClockKey* key = &initial_key;
  for (std::size_t depth = 0U; depth <= imports.size(); ++depth) {
    const auto matches_key = [&](const auto& candidate) {
      return candidate.graph == graph &&
             (key->is_default() ? candidate.is_default : candidate.key == *key);
    };
    const auto concrete_count = static_cast<std::size_t>(
        std::count_if(clocks.begin(), clocks.end(), [&](const auto& candidate) {
          return matches_key(candidate);
        }));
    const auto import_count = static_cast<std::size_t>(
        std::count_if(imports.begin(), imports.end(), [&](const auto& candidate) {
          return matches_key(candidate);
        }));
    if (concrete_count + import_count > 1U) {
      return {.status = AuthoredClockResolutionStatus::kBindingInvalid};
    }
    if (concrete_count == 1U) {
      const auto concrete = std::find_if(clocks.begin(), clocks.end(), [&](const auto& candidate) {
        return matches_key(candidate);
      });
      return {
          .status = AuthoredClockResolutionStatus::kResolved,
          .declaration = &*concrete,
      };
    }
    if (import_count == 0U) {
      return {.status = AuthoredClockResolutionStatus::kNotFound};
    }

    const auto imported = std::find_if(imports.begin(), imports.end(), [&](const auto& candidate) {
      return matches_key(candidate);
    });
    const auto binding_count = static_cast<std::size_t>(
        std::count_if(bindings.begin(), bindings.end(), [&](const auto& candidate) {
          return candidate.child_graph == imported->graph && candidate.child_clock == imported->key;
        }));
    if (binding_count == 0U) {
      return {.status = AuthoredClockResolutionStatus::kImportUnbound};
    }
    if (binding_count != 1U) {
      return {.status = AuthoredClockResolutionStatus::kBindingInvalid};
    }
    const auto binding = std::find_if(bindings.begin(), bindings.end(), [&](const auto& candidate) {
      return candidate.child_graph == imported->graph && candidate.child_clock == imported->key;
    });
    const auto child = std::find_if(
        graphs.begin(), graphs.end(), [&](const auto& row) { return row.id == graph; });
    if (child == graphs.end()) {
      return {.status = AuthoredClockResolutionStatus::kBindingInvalid};
    }
    const auto& parent = child->parent;
    if (!parent || *parent != binding->parent_graph || binding->parent_clock.is_default() ||
        binding->parent_clock.name().empty()) {
      return {.status = AuthoredClockResolutionStatus::kBindingInvalid};
    }
    graph = binding->parent_graph;
    key = &binding->parent_clock;
  }
  return {.status = AuthoredClockResolutionStatus::kBindingInvalid};
}

}  // namespace holoscan::detail
