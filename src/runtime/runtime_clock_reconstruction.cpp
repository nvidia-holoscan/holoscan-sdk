/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_clock_reconstruction.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../compiler/execution_plan_lookup.hpp"
#include "../core/clock_binding_resolution.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] Error startup_error(ErrorCode code, const char* token) {
  return Error{code, token};
}

[[nodiscard]] expected<void, Error> validate_clock_import_inventory(
    const ExecutionPlanState& plan) {
  const auto authored_imports = plan.graph.clock_imports();
  const auto authored_bindings = plan.graph.clock_bindings();
  if (authored_imports.size() != authored_bindings.size() ||
      authored_imports.size() != plan.clock_import_resolutions.size()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }
  std::vector<GraphId> defaults;
  defaults.reserve(plan.graph.clocks().size() + authored_imports.size());
  for (const auto& clock : plan.graph.clocks()) {
    if (clock.is_default) {
      defaults.push_back(clock.graph);
    }
  }
  for (const auto& clock_import : authored_imports) {
    if (clock_import.is_default) {
      defaults.push_back(clock_import.graph);
    }
  }
  std::ranges::sort(defaults, {}, &GraphId::value);
  if (std::ranges::adjacent_find(defaults) != defaults.end()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }
  for (std::size_t index = 0U; index < authored_imports.size(); ++index) {
    const auto& authored_import = authored_imports[index];
    const auto matches_import = [&](const auto& candidate) {
      return candidate.child_graph == authored_import.graph &&
             candidate.child_clock == authored_import.key;
    };
    const auto binding =
        std::find_if(authored_bindings.begin(), authored_bindings.end(), matches_import);
    const auto binding_count = static_cast<std::size_t>(
        std::count_if(authored_bindings.begin(), authored_bindings.end(), matches_import));
    const ClockId resolved =
        resolve_runtime_clock_id(plan, authored_import.graph, authored_import.key);
    if (binding_count != 1U || binding == authored_bindings.end() || !resolved.valid()) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    const auto& frozen = plan.clock_import_resolutions[index];
    if (frozen.child_graph != authored_import.graph || frozen.child_clock != authored_import.key ||
        frozen.is_default != authored_import.is_default ||
        frozen.parent_graph != binding->parent_graph ||
        frozen.parent_clock != binding->parent_clock || frozen.resolved_clock != resolved) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
  }
  return {};
}

[[nodiscard]] ClockKey runtime_clock_key(ClockId id) {
  // ClockRegistry assigns ids by sorted key. A fixed-width private key keeps
  // the immutable plan's dense numeric order even when independently embedded
  // Graphs reuse the same authored clock name.
  constexpr std::string_view kPrefix{"__holoscan_runtime_clock/"};
  std::array<char, 20U> digits{};
  const auto converted = std::to_chars(digits.data(), digits.data() + digits.size(), id.value);
  const std::size_t count = static_cast<std::size_t>(converted.ptr - digits.data());
  std::string name{kPrefix};
  name.append(digits.size() - count, '0');
  name.append(digits.data(), count);
  return ClockKey::named(name);
}

[[nodiscard]] expected<std::shared_ptr<Clock>, Error> reconstruct_clock(
    const ExecutableClockFactoryRecord& executable) {
  if (static_cast<bool>(executable.factory) == static_cast<bool>(executable.shared)) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_FACTORY_INVALID"));
  }
  if (executable.factory == nullptr) {
    return executable.shared;
  }
  auto created = executable.factory->create();
  if (!created) {
    return make_unexpected(
        startup_error(created.error().code, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
  }
  if (created.value() == nullptr) {
    return make_unexpected(
        startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
  }
  return std::shared_ptr<Clock>{std::move(created).value()};
}

// Reconstruct and authenticate the complete plan-local clock catalog as one cold-start unit.
}  // namespace

ClockId resolve_runtime_clock_id(const ExecutionPlanState& plan, GraphId graph,
                                 const ClockKey& key) noexcept {
  const AuthoredClockResolution resolved = resolve_authored_clock(plan.graph.graphs(),
                                                                  plan.graph.clocks(),
                                                                  plan.graph.clock_imports(),
                                                                  plan.graph.clock_bindings(),
                                                                  graph,
                                                                  key);
  if (!resolved.resolved()) {
    return {};
  }
  const auto found = std::find_if(plan.clocks.begin(), plan.clocks.end(), [&](const auto& clock) {
    return clock.graph == resolved.declaration->graph && clock.key == resolved.declaration->key;
  });
  return found == plan.clocks.end() ? ClockId{} : found->id;
}

// Runtime clock authentication intentionally validates the catalog, graph ownership, defaults,
// and every conversion in one cold startup pass so no partially trusted inventory escapes.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> validate_runtime_clock_reconstruction(
    const ExecutionPlanState& plan) {
  const auto authored_clocks = plan.graph.clocks();
  const auto authored_conversions = plan.graph.clock_conversions();
  if (plan.clocks.size() != authored_clocks.size() ||
      plan.executable_factories.clocks.size() != authored_clocks.size()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }

  const auto graph_records = plan.graph.graphs();
  const auto root_graph = std::find_if(graph_records.begin(),
                                       graph_records.end(),
                                       [](const auto& graph) { return !graph.parent.has_value(); });
  if (root_graph == graph_records.end()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }

  auto imports_valid = validate_clock_import_inventory(plan);
  if (!imports_valid) {
    return imports_valid;
  }

  ClockId expected_root_default;
  for (std::size_t index = 0U; index < authored_clocks.size(); ++index) {
    const auto& authored = authored_clocks[index];
    const auto& planned = plan.clocks[index];
    const auto& executable = plan.executable_factories.clocks[index];
    const bool has_factory = executable.factory != nullptr;
    const bool has_shared = executable.shared != nullptr;
    if (planned.id.value != index + 1U || planned.graph != authored.graph ||
        planned.key != authored.key || planned.shared != authored.shared ||
        planned.is_default != authored.is_default || executable.graph != authored.graph ||
        executable.key != authored.key || has_factory == has_shared ||
        planned.shared != has_shared) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    if (authored.graph == root_graph->id && authored.is_default) {
      if (expected_root_default.valid()) {
        return make_unexpected(
            startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
      }
      expected_root_default = planned.id;
    }
  }
  if (plan.root_default_clock != expected_root_default) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }

  struct ResolvedClockConversionPair {
    ClockId from;
    ClockId to;
  };
  std::vector<ResolvedClockConversionPair> resolved_pairs;
  resolved_pairs.reserve(authored_conversions.size());
  for (const auto& authored : authored_conversions) {
    resolved_pairs.push_back(ResolvedClockConversionPair{
        .from = resolve_runtime_clock_id(plan, authored.graph, authored.from),
        .to = resolve_runtime_clock_id(plan, authored.graph, authored.to),
    });
  }

  std::size_t planned_index = 0U;
  for (std::size_t authored_index = 0U; authored_index < authored_conversions.size();
       ++authored_index) {
    const auto& authored = authored_conversions[authored_index];
    const auto& resolved = resolved_pairs[authored_index];
    bool duplicate = false;
    for (std::size_t prior_index = 0U; prior_index < authored_index; ++prior_index) {
      const auto& prior = authored_conversions[prior_index];
      const auto& prior_resolved = resolved_pairs[prior_index];
      if (resolved.from == prior_resolved.from && resolved.to == prior_resolved.to) {
        if (authored.conversion.scale != prior.conversion.scale ||
            authored.conversion.offset_ns != prior.conversion.offset_ns) {
          return make_unexpected(
              startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
        }
        duplicate = true;
        break;
      }
    }
    if (duplicate) {
      continue;
    }
    if (planned_index >= plan.clock_conversions.size()) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    const auto& planned = plan.clock_conversions[planned_index];
    if (planned.id != planned_index + 1U || planned.plan_generation != plan.plan_generation ||
        planned.graph != authored.graph || planned.from != resolved.from ||
        planned.to != resolved.to || planned.scale != authored.conversion.scale ||
        planned.offset_ns != authored.conversion.offset_ns ||
        planned.rounding != ClockRoundingMode::kNearestTiesToEven ||
        planned.overflow != ClockOverflowPolicy::kError) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    ++planned_index;
  }
  if (planned_index != plan.clock_conversions.size()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }
  return {};
}

// NOLINTBEGIN(readability-function-cognitive-complexity)
[[nodiscard]] expected<RuntimeClockReconstruction, Error> reconstruct_runtime_clocks(
    const ExecutionPlanState& plan) {
  if (plan.executable_factories.clocks.size() != plan.clocks.size()) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }
  if (plan.clocks.empty()) {
    if (plan.root_default_clock.valid() || !plan.clock_conversions.empty()) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    return RuntimeClockReconstruction{};
  }

  const ClockPlanRecord* const root_default = find_clock_by_id(plan, plan.root_default_clock);
  if (root_default == nullptr || !root_default->is_default) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }

  ClockRegistryBuilder builder;
  std::vector<ClockKey> runtime_keys;
  runtime_keys.reserve(plan.clocks.size());
  for (std::size_t index = 0U; index < plan.clocks.size(); ++index) {
    const ClockPlanRecord& planned = plan.clocks[index];
    const ExecutableClockFactoryRecord& executable = plan.executable_factories.clocks[index];
    if (planned.id.value != index + 1U || planned.graph != executable.graph ||
        planned.key != executable.key || planned.shared != static_cast<bool>(executable.shared)) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    auto clock = reconstruct_clock(executable);
    if (!clock) {
      return make_unexpected(std::move(clock.error()));
    }
    runtime_keys.push_back(runtime_clock_key(planned.id));
    auto added = builder.add_clock(runtime_keys.back(),
                                   std::move(clock).value(),
                                   ClockCapabilityOptions{
                                       .can_read_now = planned.can_read_now,
                                       .can_wait_until = planned.can_wait_until,
                                       .steady = planned.steady,
                                   });
    if (!added) {
      return make_unexpected(
          startup_error(added.error().code, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
    }
  }

  auto selected_default = builder.set_default(
      runtime_keys[static_cast<std::size_t>(plan.root_default_clock.value - 1U)]);
  if (!selected_default) {
    return make_unexpected(
        startup_error(selected_default.error().code, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
  }

  for (std::size_t index = 0U; index < plan.clock_conversions.size(); ++index) {
    const ClockConversionRecord& conversion = plan.clock_conversions[index];
    const ClockPlanRecord* const from = find_clock_by_id(plan, conversion.from);
    const ClockPlanRecord* const to = find_clock_by_id(plan, conversion.to);
    if (conversion.id != index + 1U || conversion.plan_generation != plan.plan_generation ||
        from == nullptr || to == nullptr ||
        conversion.rounding != ClockRoundingMode::kNearestTiesToEven ||
        conversion.overflow != ClockOverflowPolicy::kError) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
    auto added =
        builder.add_conversion(runtime_keys[static_cast<std::size_t>(conversion.from.value - 1U)],
                               runtime_keys[static_cast<std::size_t>(conversion.to.value - 1U)],
                               AffineClockConversion{
                                   .scale = conversion.scale,
                                   .offset_ns = conversion.offset_ns,
                               });
    if (!added) {
      return make_unexpected(
          startup_error(added.error().code, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
    }
  }

  auto registry = builder.build();
  if (!registry) {
    return make_unexpected(
        startup_error(registry.error().code, "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"));
  }
  for (const ClockPlanRecord& planned : plan.clocks) {
    auto resolved =
        registry->resolve(runtime_keys[static_cast<std::size_t>(planned.id.value - 1U)]);
    if (!resolved || resolved->id() != planned.id) {
      return make_unexpected(
          startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
    }
  }
  auto resolved_default = registry->resolve(ClockKey::default_clock());
  if (!resolved_default || resolved_default->id() != plan.root_default_clock) {
    return make_unexpected(startup_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH"));
  }
  return RuntimeClockReconstruction{
      .registry = std::move(registry).value(),
      .default_clock = std::move(resolved_default).value(),
  };
}
// NOLINTEND(readability-function-cognitive-complexity)

}  // namespace holoscan::detail
