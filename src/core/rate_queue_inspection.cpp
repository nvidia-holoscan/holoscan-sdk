/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/rate_generation.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "canonical_text_encoding.hpp"

namespace holoscan {
namespace {

constexpr std::string_view kInspectionSchema{"holoscan.rate_queue_inspection.canonical_json.v1"};

[[nodiscard]] Error inspection_error(ErrorCode code, const char* token) {
  return Error{code, token};
}

template <typename Range, typename Predicate>
[[nodiscard]] const typename Range::value_type* find_exactly_one(const Range& range,
                                                                 Predicate predicate) noexcept {
  const typename Range::value_type* found = nullptr;
  for (const auto& item : range) {
    if (!predicate(item)) {
      continue;
    }
    if (found != nullptr) {
      return nullptr;
    }
    found = &item;
  }
  return found;
}

template <typename Value>
void append_integer(std::string& output, Value value) {
  std::array<char, 32U> buffer{};
  const auto converted = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
  output.append(buffer.data(), converted.ptr);
}

void append_json_key(std::string& output, std::string_view key) {
  detail::append_json_string(output, key);
  output.push_back(':');
}

template <typename Value>
void append_json_integer(std::string& output, std::string_view key, Value value) {
  append_json_key(output, key);
  append_integer(output, value);
}

template <typename Value>
void append_json_optional_integer(std::string& output, std::string_view key,
                                  const std::optional<Value>& value) {
  append_json_key(output, key);
  if (value) {
    append_integer(output, *value);
  } else {
    output += "null";
  }
}

void append_json_text(std::string& output, std::string_view key, std::string_view value) {
  append_json_key(output, key);
  detail::append_json_string(output, value);
}

[[nodiscard]] std::optional<std::string_view> queue_policy_name(QueuePolicy value) noexcept {
  switch (value) {
    case QueuePolicy::kBlocking:
      return "blocking";
    case QueuePolicy::kDropOldest:
      return "drop_oldest";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> queue_depth_origin_name(
    QueueDepthOrigin value) noexcept {
  switch (value) {
    case QueueDepthOrigin::kGraphEdgeOverride:
      return "graph_edge_override";
    case QueueDepthOrigin::kAddFlowOption:
      return "add_flow_option";
    case QueueDepthOrigin::kGraphInputOverride:
      return "graph_input_override";
    case QueueDepthOrigin::kOperatorSetup:
      return "operator_setup";
    case QueueDepthOrigin::kRateAndContractDerived:
      return "rate_and_contract_derived";
    case QueueDepthOrigin::kContractDerived:
      return "contract_derived";
    case QueueDepthOrigin::kFrameworkFallback:
      return "framework_fallback";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> stability_name(StabilityClass value) noexcept {
  switch (value) {
    case StabilityClass::kStableNominalEstimate:
      return "stable_nominal_estimate";
    case StabilityClass::kPredictedSustainedBlocking:
      return "predicted_sustained_blocking";
    case StabilityClass::kPredictedReplacementOrDrop:
      return "predicted_replacement_or_drop";
    case StabilityClass::kUnresolvedBoundedEvidence:
      return "unresolved_bounded_evidence";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> resolved_rate_origin_name(
    ResolvedRateOrigin value) noexcept {
  switch (value) {
    case ResolvedRateOrigin::kUnresolved:
      return "unresolved";
    case ResolvedRateOrigin::kOutputNominal:
      return "output_nominal";
    case ResolvedRateOrigin::kConnectedOutput:
      return "connected_output";
    case ResolvedRateOrigin::kFollowsExpression:
      return "follows_expression";
    case ResolvedRateOrigin::kInputExpectationFallback:
      return "input_expectation_fallback";
    case ResolvedRateOrigin::kLifecycleBinding:
      return "lifecycle_binding";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> allocation_action_name(
    AllocationAction value) noexcept {
  switch (value) {
    case AllocationAction::kFresh:
      return "fresh";
    case AllocationAction::kReusedExact:
      return "reused_exact";
    case AllocationAction::kReusedLarger:
      return "reused_larger";
    case AllocationAction::kReplaced:
      return "replaced";
    case AllocationAction::kRejected:
      return "rejected";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> generation_disposition_name(
    RateGenerationDisposition value) noexcept {
  switch (value) {
    case RateGenerationDisposition::kPublished:
      return "published";
    case RateGenerationDisposition::kRejected:
      return "rejected";
  }
  return std::nullopt;
}

[[nodiscard]] std::string rate_text(Rate rate) {
  std::string result;
  result.reserve(48U);
  append_integer(result, rate.numerator_hz);
  result.push_back('/');
  append_integer(result, rate.denominator_hz);
  result += " Hz";
  return result;
}

[[nodiscard]] const PortPlanRecord* plan_port(const ExecutionPlan& plan, PortId id) noexcept {
  if (!id || id.value() > plan.ports().size()) {
    return nullptr;
  }
  const auto& result = plan.ports()[static_cast<std::size_t>(id.value() - 1U)];
  return result.id == id ? &result : nullptr;
}

[[nodiscard]] std::optional<std::string> port_name(const ExecutionPlan& plan, PortId id) {
  const auto* const port = plan_port(plan, id);
  if (port == nullptr) {
    return std::nullopt;
  }
  const ExecutionGraphView graph = plan.graph();
  const auto* const vertex =
      find_exactly_one(graph.vertices(), [&](const auto& item) { return item.id == port->owner; });
  if (vertex == nullptr) {
    return std::nullopt;
  }
  const auto* const owner_graph =
      find_exactly_one(graph.graphs(), [&](const auto& item) { return item.id == vertex->graph; });
  if (owner_graph == nullptr) {
    return std::nullopt;
  }
  std::string result;
  if (owner_graph->path) {
    for (const auto& segment : owner_graph->path->segments()) {
      detail::append_percent_encoded_uri_path_segment(result, segment);
    }
    result.push_back('/');
  }
  detail::append_percent_encoded_uri_component(result, vertex->name);
  result.push_back('.');
  detail::append_percent_encoded_uri_component(result, port->local_name);
  return result;
}

template <typename Record, typename Id>
[[nodiscard]] std::span<const Record> generation_range(std::span<const Record> records, Id first,
                                                       std::uint32_t count) noexcept {
  if (count == 0U) {
    return {};
  }
  const auto begin = std::ranges::find(records, first, &Record::id);
  if (begin == records.end() ||
      static_cast<std::size_t>(std::distance(begin, records.end())) < count) {
    return {};
  }
  return {&*begin, count};
}

struct ProviderJoin {
  const PoolCapacityGenerationRecord* pool{};
  const ProviderAllocationPlanRecord* recipe{};
  const ProviderAllocationGenerationRecord* decision{};
};

struct InspectionJoin {
  const RateGenerationEvidenceRecord* generation{};
  const PayloadEdgePlanRecord* edge{};
  const QueueSizingRecipePlanRecord* recipe{};
  const QueueSizingGenerationRecord* sizing{};
  const PortRateGenerationRecord* producer{};
  const AccumulationHorizonPlanRecord* horizon{};
  std::string source_name;
  std::string target_name;
  std::vector<ProviderJoin> providers;
};

// This cold validator deliberately keeps every cross-record join in one fail-closed transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<InspectionJoin, Error> build_join(const ExecutionPlan& plan,
                                                         const RateGenerationSnapshot& snapshot,
                                                         RateGenerationId rate_generation,
                                                         EdgeId edge_id) {
  if (!plan.ok() || snapshot.plan_digest() != plan.digest() || snapshot.run_generation() == 0U) {
    return make_unexpected(
        inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_PLAN_MISMATCH"));
  }
  const auto* const generation = find_exactly_one(snapshot.generations(), [&](const auto& item) {
    return item.rate_generation == rate_generation;
  });
  if (generation == nullptr || generation->run_generation != snapshot.run_generation() ||
      !generation_disposition_name(generation->disposition)) {
    return make_unexpected(
        inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_GENERATION_UNKNOWN"));
  }
  const auto* const edge = find_exactly_one(plan.payload_edges(),
                                            [&](const auto& item) { return item.edge == edge_id; });
  if (edge == nullptr) {
    return make_unexpected(
        inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_EDGE_UNKNOWN"));
  }
  const auto* const recipe = find_exactly_one(
      plan.queue_sizing_recipes(), [&](const auto& item) { return item.edge == edge_id; });
  const auto queue_records = snapshot.queue_sizings(*generation);
  const auto port_records = snapshot.port_rates(*generation);
  if (queue_records.size() != generation->queue_sizing_count ||
      port_records.size() != generation->port_rate_count) {
    return make_unexpected(
        inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
  }
  const auto* const sizing =
      recipe == nullptr ? nullptr : find_exactly_one(queue_records, [&](const auto& item) {
        return item.recipe == recipe->id && item.run_generation == snapshot.run_generation() &&
               item.rate_generation == rate_generation;
      });
  if (recipe == nullptr || sizing == nullptr || !queue_depth_origin_name(recipe->origin) ||
      !stability_name(sizing->stability) ||
      (recipe->explicit_depth && *recipe->explicit_depth != sizing->effective_logical_depth)) {
    return make_unexpected(
        inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
  }

  const PortRateGenerationRecord* producer = nullptr;
  if (sizing->producer_rate) {
    producer = find_exactly_one(port_records, [&](const auto& item) {
      return item.id == *sizing->producer_rate &&
             item.run_generation == snapshot.run_generation() &&
             item.rate_generation == rate_generation &&
             (item.port == edge->source || item.port == edge->target);
    });
    if (producer == nullptr || !resolved_rate_origin_name(producer->resolved_origin)) {
      return make_unexpected(
          inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
    }
  }

  const AccumulationHorizonPlanRecord* horizon = nullptr;
  if (sizing->accumulation_horizon) {
    horizon = find_exactly_one(plan.accumulation_horizons(), [&](const auto& item) {
      return item.id == *sizing->accumulation_horizon && item.edge == edge_id;
    });
    if (horizon == nullptr ||
        horizon->horizon.has_value() == horizon->inverse_rate_port.has_value() ||
        (horizon->horizon &&
         (horizon->horizon->numerator_ns == 0U || horizon->horizon->denominator == 0U)) ||
        (horizon->inverse_rate_port && !*horizon->inverse_rate_port)) {
      return make_unexpected(
          inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
    }
  }

  auto source_name = port_name(plan, edge->source);
  auto target_name = port_name(plan, edge->target);
  if (!source_name || !target_name || !queue_policy_name(edge->queue_policy)) {
    return make_unexpected(
        inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
  }

  InspectionJoin result{
      .generation = generation,
      .edge = edge,
      .recipe = recipe,
      .sizing = sizing,
      .producer = producer,
      .horizon = horizon,
      .source_name = std::move(*source_name),
      .target_name = std::move(*target_name),
  };
  const auto generation_contributions = snapshot.pool_contributions(*generation);
  const auto generation_pools = snapshot.pool_capacities(*generation);
  const auto generation_providers = snapshot.provider_allocations(*generation);
  if (generation_contributions.size() != generation->pool_contribution_count ||
      generation_pools.size() != generation->pool_capacity_count ||
      generation_providers.size() != generation->provider_allocation_count) {
    return make_unexpected(
        inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
  }
  for (const auto& pool : generation_pools) {
    if (pool.run_generation != snapshot.run_generation() ||
        pool.rate_generation != rate_generation) {
      return make_unexpected(
          inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
    }
    const auto contributions = generation_range(
        generation_contributions, pool.first_contribution, pool.contribution_count);
    if (pool.contribution_count != 0U && contributions.size() != pool.contribution_count) {
      return make_unexpected(
          inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
    }
    bool joins_edge{};
    for (const auto& contribution : contributions) {
      if (contribution.edge_sizing_generation != sizing->id) {
        continue;
      }
      const auto* const dependency =
          find_exactly_one(plan.pool_capacity_dependencies(),
                           [&](const auto& item) { return item.id == contribution.dependency; });
      if (dependency == nullptr ||
          dependency->source != PoolCapacityDependencySource::kQueueSizingRecipe ||
          dependency->queue_sizing_recipe != recipe->id || dependency->pool_id != pool.pool_id) {
        return make_unexpected(
            inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
      }
      joins_edge = true;
    }
    if (!joins_edge) {
      continue;
    }
    const auto* const provider_recipe = find_exactly_one(
        plan.provider_allocation_plans(),
        [&](const auto& item) { return item.id == pool.recipe && item.pool_id == pool.pool_id; });
    if (provider_recipe == nullptr) {
      return make_unexpected(
          inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
    }
    const ProviderAllocationGenerationRecord* provider_decision = nullptr;
    for (const auto& item : generation_providers) {
      if (item.recipe != provider_recipe->id || item.pool_capacity_generation != pool.id ||
          item.run_generation != snapshot.run_generation() ||
          item.rate_generation != rate_generation) {
        continue;
      }
      if (provider_decision != nullptr) {
        return make_unexpected(
            inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
      }
      provider_decision = &item;
    }
    if (provider_decision != nullptr && !allocation_action_name(provider_decision->action)) {
      return make_unexpected(
          inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
    }
    result.providers.push_back(ProviderJoin{
        .pool = &pool,
        .recipe = provider_recipe,
        .decision = provider_decision,
    });
  }
  std::ranges::sort(result.providers, {}, [](const auto& item) { return item.recipe->id.value; });
  return result;
}

[[nodiscard]] std::string paste_ready_pin(const InspectionJoin& join) {
  if (join.recipe->origin != QueueDepthOrigin::kRateAndContractDerived ||
      join.recipe->explicit_depth) {
    return {};
  }
  std::string result{"holoscan::ConnectionOptions{.queue_depth = "};
  append_integer(result, join.sizing->effective_logical_depth);
  result += "U}";
  return result;
}

[[nodiscard]] std::string pin_reason(const InspectionJoin& join) {
  if (join.recipe->explicit_depth) {
    return "already_explicit";
  }
  if (join.recipe->origin == QueueDepthOrigin::kRateAndContractDerived) {
    return "copying this pin disables automatic resizing";
  }
  if (join.recipe->origin == QueueDepthOrigin::kContractDerived) {
    return "contract_derived";
  }
  return "framework_fallback";
}

void append_horizon_json(std::string& output, const InspectionJoin& join) {
  append_json_key(output, "accumulation_horizon");
  if (join.horizon == nullptr) {
    output += "null";
  } else if (join.horizon->horizon) {
    output += "{\"kind\":\"exact_duration\",\"numerator_ns\":";
    append_integer(output, join.horizon->horizon->numerator_ns);
    output += ",\"denominator\":";
    append_integer(output, join.horizon->horizon->denominator);
    output.push_back('}');
  } else {
    output += "{\"kind\":\"inverse_rate_port\",\"port\":";
    append_integer(output, join.horizon->inverse_rate_port.value_or(PortId{}).value());
    output.push_back('}');
  }
}

// Canonical field emission intentionally mirrors the complete validated join schema.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::string render_json(const ExecutionPlan& plan,
                                      const RateGenerationSnapshot& snapshot,
                                      const InspectionJoin& join) {
  std::string output;
  output.reserve(2048U + join.providers.size() * 512U);
  output.push_back('{');
  append_json_text(output, "schema", kInspectionSchema);
  output += ",\"plan_digest\":";
  detail::append_json_string(
      output, detail::lowercase_hex_text(plan.digest().value, detail::HexPrefix::kSha256));
  output += ",";
  append_json_integer(output, "plan_generation", snapshot.plan_generation());
  output += ",";
  append_json_integer(output, "run_generation", snapshot.run_generation());
  output += ",";
  append_json_integer(output, "rate_generation", join.generation->rate_generation);
  output += ",";
  append_json_text(output,
                   "generation_disposition",
                   join.generation->disposition == RateGenerationDisposition::kPublished
                       ? "published"
                       : "rejected");
  output += ",\"edge\":{";
  append_json_integer(output, "id", join.edge->edge.value);
  output += ",";
  append_json_text(output, "source", join.source_name);
  output += ",";
  append_json_integer(output, "source_port", join.edge->source.value());
  output += ",";
  append_json_text(output, "target", join.target_name);
  output += ",";
  append_json_integer(output, "target_port", join.edge->target.value());
  output += ",";
  append_json_text(
      output, "queue_policy", queue_policy_name(join.edge->queue_policy).value_or("invalid"));
  output += ",";
  append_json_integer(output, "admitted_backing_depth", join.edge->queue_depth);
  output += "},\"queue_sizing\":{";
  append_json_integer(output, "recipe", join.recipe->id.value);
  output += ",";
  append_json_text(
      output, "origin", queue_depth_origin_name(join.recipe->origin).value_or("invalid"));
  output += ",";
  append_json_optional_integer(output, "explicit_depth", join.recipe->explicit_depth);
  output += ",";
  append_json_integer(output, "hard_floor", join.sizing->hard_floor);
  output += ",";
  append_json_optional_integer(output, "nominal_arrivals", join.sizing->nominal_arrivals);
  output += ",";
  append_json_optional_integer(output, "rate_recommendation", join.sizing->rate_recommendation);
  output += ",";
  append_json_optional_integer(
      output, "admitted_depth_ceiling", join.sizing->admitted_depth_ceiling);
  output += ",";
  append_json_integer(output, "effective_logical_depth", join.sizing->effective_logical_depth);
  output += ",";
  append_json_text(output, "stability", stability_name(join.sizing->stability).value_or("invalid"));
  output += "},\"producer_rate\":";
  if (join.producer == nullptr) {
    output += "null";
  } else {
    output += "{\"port\":";
    append_integer(output, join.producer->port.value());
    output += ",\"resolved_rate\":";
    if (join.producer->resolved_rate) {
      detail::append_json_string(output, rate_text(*join.producer->resolved_rate));
    } else {
      output += "null";
    }
    output += ",";
    append_json_text(output,
                     "resolved_origin",
                     resolved_rate_origin_name(join.producer->resolved_origin).value_or("invalid"));
    output.push_back('}');
  }
  output += ",";
  append_horizon_json(output, join);
  output += ",\"provider_allocations\":[";
  for (std::size_t index = 0U; index < join.providers.size(); ++index) {
    if (index != 0U) {
      output.push_back(',');
    }
    const auto& provider = join.providers[index];
    output += "{\"pool_id\":";
    detail::append_json_string(output, provider.pool->pool_id);
    output += ",\"provider_recipe\":";
    append_integer(output, provider.recipe->id.value);
    output += ",";
    append_json_integer(output, "pool_capacity_generation", provider.pool->id.value);
    output += ",";
    append_json_integer(output, "required_pool_entries", provider.pool->required_pool_entries);
    output += ",";
    append_json_optional_integer(
        output, "admitted_pool_entry_ceiling", provider.pool->admitted_pool_entry_ceiling);
    output += ",";
    append_json_optional_integer(
        output, "max_payload_byte_span_per_slot", provider.pool->max_payload_byte_span_per_slot);
    output += ",";
    append_json_optional_integer(
        output, "logical_payload_capacity_bytes", provider.pool->logical_payload_capacity_bytes);
    output += ",";
    append_json_optional_integer(output,
                                 "admitted_logical_payload_capacity_ceiling_bytes",
                                 provider.pool->admitted_logical_payload_capacity_ceiling_bytes);
    output += ",\"provider\":";
    if (provider.decision == nullptr) {
      output += "null";
    } else {
      output += "{\"action\":";
      detail::append_json_string(
          output, allocation_action_name(provider.decision->action).value_or("invalid"));
      output += ",";
      append_json_optional_integer(
          output,
          "allocation_generation",
          provider.decision->allocation_generation
              ? std::optional<std::uint64_t>{provider.decision->allocation_generation->value}
              : std::nullopt);
      output += ",";
      append_json_optional_integer(
          output,
          "storage_binding_generation",
          provider.decision->storage_binding_generation
              ? std::optional<std::uint64_t>{provider.decision->storage_binding_generation->value}
              : std::nullopt);
      output += ",";
      append_json_optional_integer(
          output,
          "previous_decision",
          provider.decision->previous_decision
              ? std::optional<std::uint64_t>{provider.decision->previous_decision->value}
              : std::nullopt);
      output += ",";
      append_json_optional_integer(
          output,
          "previous_allocation_generation",
          provider.decision->previous_allocation_generation
              ? std::optional<std::uint64_t>{provider.decision->previous_allocation_generation
                                                 ->value}
              : std::nullopt);
      output += ",";
      append_json_optional_integer(
          output, "materialized_entries", provider.decision->provider_materialized_entries);
      output += ",";
      append_json_optional_integer(
          output, "stride_bytes_per_entry", provider.decision->provider_stride_bytes_per_entry);
      output += ",";
      append_json_optional_integer(
          output, "reserved_bytes_total", provider.decision->provider_reserved_bytes_total);
      output.push_back('}');
    }
    output.push_back('}');
  }
  output += "],\"paste_ready_exact_pin\":";
  const std::string pin = paste_ready_pin(join);
  if (pin.empty()) {
    output += "null";
  } else {
    detail::append_json_string(output, pin);
  }
  output += ",";
  append_json_text(output, "paste_ready_exact_pin_effect", pin_reason(join));
  output.push_back('}');
  return output;
}

// The human renderer follows the same validated join as the canonical serializer.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::string render_human(const RateGenerationSnapshot& snapshot,
                                       const InspectionJoin& join) {
  std::string output{"Rate-aware queue inspection\nplan digest: "};
  output += detail::lowercase_hex_text(snapshot.plan_digest().value, detail::HexPrefix::kSha256);
  output += "\nedge: ";
  output += join.source_name;
  output += " -> ";
  output += join.target_name;
  output += " (";
  append_integer(output, join.edge->edge.value);
  output.push_back(')');
  output += "\nrun generation: ";
  append_integer(output, snapshot.run_generation());
  output += "\nrate generation: ";
  append_integer(output, join.generation->rate_generation);
  output += " (";
  output += generation_disposition_name(join.generation->disposition).value_or("invalid");
  output.push_back(')');
  output += "\nqueue origin: ";
  output += queue_depth_origin_name(join.recipe->origin).value_or("invalid");
  output += "\nexplicit depth: ";
  if (join.recipe->explicit_depth) {
    append_integer(output, *join.recipe->explicit_depth);
  } else {
    output += "unavailable";
  }
  output += "\nadmitted backing depth: ";
  append_integer(output, join.edge->queue_depth);
  output += "\nproducer rate: ";
  if (join.producer == nullptr || !join.producer->resolved_rate) {
    output += "unavailable";
  } else {
    output += rate_text(*join.producer->resolved_rate);
    output += " (";
    output += resolved_rate_origin_name(join.producer->resolved_origin).value_or("invalid");
    output.push_back(')');
  }
  output += "\nhard floor: ";
  append_integer(output, join.sizing->hard_floor);
  output += "\naccumulation horizon: ";
  if (join.horizon == nullptr) {
    output += "unavailable";
  } else if (join.horizon->horizon) {
    append_integer(output, join.horizon->horizon->numerator_ns);
    output.push_back('/');
    append_integer(output, join.horizon->horizon->denominator);
    output += " ns";
  } else {
    output += "inverse rate of port ";
    append_integer(output, join.horizon->inverse_rate_port.value_or(PortId{}).value());
  }
  output += "\nrate recommendation: ";
  if (join.sizing->rate_recommendation) {
    append_integer(output, *join.sizing->rate_recommendation);
  } else {
    output += "unavailable";
  }
  output += "\neffective logical depth: ";
  append_integer(output, join.sizing->effective_logical_depth);
  output += "\nstability: ";
  output += stability_name(join.sizing->stability).value_or("invalid");
  if (join.providers.empty()) {
    output += "\nprovider allocations: none";
  } else {
    output += "\nprovider allocations:";
    for (const auto& provider : join.providers) {
      output += "\n  - pool ";
      output += provider.pool->pool_id;
      output += ": required entries ";
      append_integer(output, provider.pool->required_pool_entries);
      output += ", logical payload bytes ";
      if (provider.pool->logical_payload_capacity_bytes) {
        append_integer(output, *provider.pool->logical_payload_capacity_bytes);
      } else {
        output += "unavailable";
      }
      output += ", action ";
      if (provider.decision == nullptr) {
        output += "unavailable";
      } else {
        output += allocation_action_name(provider.decision->action).value_or("invalid");
      }
      output += ", allocation generation ";
      if (provider.decision != nullptr && provider.decision->allocation_generation) {
        append_integer(output, provider.decision->allocation_generation->value);
      } else {
        output += "unavailable";
      }
      output += ", storage binding generation ";
      if (provider.decision != nullptr && provider.decision->storage_binding_generation) {
        append_integer(output, provider.decision->storage_binding_generation->value);
      } else {
        output += "unavailable";
      }
      output += ", reserved bytes ";
      if (provider.decision != nullptr && provider.decision->provider_reserved_bytes_total) {
        append_integer(output, *provider.decision->provider_reserved_bytes_total);
      } else {
        output += "unavailable";
      }
    }
  }
  const std::string pin = paste_ready_pin(join);
  output += "\npaste-ready exact pin: ";
  output += pin.empty() ? "not applicable" : pin;
  output += "\npin effect: ";
  output += pin_reason(join);
  output.push_back('\n');
  return output;
}

}  // namespace

expected<std::string, Error> render_rate_queue_inspection(const ExecutionPlan& plan,
                                                          const RateGenerationSnapshot& snapshot,
                                                          RateGenerationId rate_generation,
                                                          EdgeId edge,
                                                          RateQueueInspectionFormat format) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto joined = build_join(plan, snapshot, rate_generation, edge);
    if (!joined) {
      return make_unexpected(std::move(joined.error()));
    }
    switch (format) {
      case RateQueueInspectionFormat::kHumanText:
        return render_human(snapshot, *joined);
      case RateQueueInspectionFormat::kCanonicalJson:
        return render_json(plan, snapshot, *joined);
    }
    return make_unexpected(
        inspection_error(ErrorCode::kInvalidArgument, "RATE_QUEUE_INSPECTION_JOIN_INVALID"));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        inspection_error(ErrorCode::kResourceExhausted, "RATE_QUEUE_INSPECTION_ALLOCATION_FAILED"));
  } catch (...) {
    return make_unexpected(inspection_error(ErrorCode::kFailure, "RATE_QUEUE_INSPECTION_FAILED"));
  }
#endif
}

}  // namespace holoscan
