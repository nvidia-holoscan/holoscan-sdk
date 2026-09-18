/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/plan_generation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "compiler/execution_plan_identity.hpp"
#include "compiler/parameter_plan_lowering.hpp"
#include "core/canonical_text_encoding.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kPlanGenerationDomain = "holoscan.plan_generation.v1";

}  // namespace

PlanGenerationId derive_plan_generation_id(const ExecutionPlanState& canonical_state) noexcept {
  StableHashBuilder hash;
  hash.add(kPlanGenerationDomain);
  // Authoring origins and source records do not define runtime behavior. Advanced transaction
  // masks do, so each field retains only whether the transaction contributed its final value.
  for (const std::byte byte : canonical_state.digest.value) {
    hash.add_byte(std::to_integer<std::uint8_t>(byte));
  }
  hash.add_canonical_size(canonical_state.parameter_generation_zero_operators.size());
  for (const auto& record : canonical_state.parameter_generation_zero_operators) {
    hash.add(record.owner.id);
    hash.add(record.value_digest.disposition);
    for (const std::byte byte : record.value_digest.digest.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
  }
  hash.add_canonical_size(canonical_state.parameter_generation_zero_fields.size());
  for (const auto& record : canonical_state.parameter_generation_zero_fields) {
    hash.add(record.instance);
    hash.add(record.field);
    for (const std::byte byte : record.full_id.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
    hash.add(record.transaction_contributed);
    hash.add(record.value_digest.disposition);
    for (const std::byte byte : record.value_digest.digest.value) {
      hash.add_byte(std::to_integer<std::uint8_t>(byte));
    }
  }
  const auto digest = hash.finish();
  std::uint64_t generation{};
  for (std::size_t index = 0U; index < sizeof(generation); ++index) {
    generation = (generation << 8U) | std::to_integer<std::uint8_t>(digest.value[index]);
  }
  return generation == 0U ? 1U : generation;
}

// Rebind every plan-local identity family together to preserve generation consistency.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void rebind_plan_generation(ExecutionPlanState& state, PlanGenerationId plan_generation) {
  const auto rebind_operator = [plan_generation](OperatorRef& reference) {
    if (reference.id != 0U) {
      reference.plan_generation = plan_generation;
    }
  };
  const auto rebind_owner_records = [&](auto& records) {
    for (auto& record : records) {
      rebind_operator(record.owner);
    }
  };
  std::vector<std::pair<std::string, std::string>> rebound_logical_ids;
  const auto rebind_logical_id = [&](std::string& identity, std::string replacement) {
    rebound_logical_ids.emplace_back(identity, replacement);
    identity = std::move(replacement);
  };

  state.plan_generation = plan_generation;
  for (auto& record : state.clock_conversions) {
    record.plan_generation = plan_generation;
  }
  for (auto& record : state.endpoint_clock_health) {
    record.plan_generation = plan_generation;
    if (record.conversion_generation != 0U) {
      record.conversion_generation = plan_generation;
    }
  }
  for (auto& record : state.readiness) {
    record.plan_generation = plan_generation;
  }
  for (auto& record : state.readable_triggers) {
    rebind_operator(record.owner);
    record.plan_generation = plan_generation;
  }
  std::vector<std::pair<std::string, std::string>> rebound_process_local_services;
  rebound_process_local_services.reserve(state.process_local_channel_services.size());
  for (auto& record : state.process_local_channel_services) {
    std::string replacement =
        process_local_channel_service_id(plan_generation, record.source, record.queue_policy);
    rebound_process_local_services.emplace_back(record.service_id, replacement);
    record.service_id = std::move(replacement);
  }
  const auto rebind_process_local_service = [&](std::string& service_id) {
    const auto found =
        std::find_if(rebound_process_local_services.begin(),
                     rebound_process_local_services.end(),
                     [&](const auto& identity) { return identity.first == service_id; });
    if (found != rebound_process_local_services.end()) {
      service_id = found->second;
    }
  };
  for (auto& record : state.process_local_channel_publishers) {
    rebind_process_local_service(record.service_id);
  }
  for (auto& record : state.process_local_channel_subscribers) {
    rebind_process_local_service(record.service_id);
  }
  std::vector<std::pair<std::string, std::string>> rebound_cross_process_services;
  rebound_cross_process_services.reserve(state.cross_process_channel_services.size());
  for (auto& record : state.cross_process_channel_services) {
    std::string replacement = cross_process_channel_service_id(
        plan_generation, record.source, record.queue_policy, record.edge);
    rebound_cross_process_services.emplace_back(record.service_id, replacement);
    record.service_id = std::move(replacement);
  }
  const auto rebind_cross_process_service = [&](std::string& service_id) {
    const auto found =
        std::find_if(rebound_cross_process_services.begin(),
                     rebound_cross_process_services.end(),
                     [&](const auto& identity) { return identity.first == service_id; });
    if (found != rebound_cross_process_services.end()) {
      service_id = found->second;
    }
  };
  for (auto& record : state.cross_process_channel_publishers) {
    rebind_cross_process_service(record.service_id);
  }
  for (auto& record : state.cross_process_channel_subscribers) {
    rebind_cross_process_service(record.service_id);
  }
  for (auto& record : state.generic_output_loans) {
    record.plan_generation = plan_generation;
    rebind_logical_id(record.pool_id, generic_output_pool_id(plan_generation, record.output));
  }
  for (auto& record : state.tensor_allocations) {
    record.plan_generation = plan_generation;
    rebind_logical_id(record.pool_id, tensor_output_pool_id(plan_generation, record.output));
    record.allocator_root_id =
        tensor_allocator_root_id(plan_generation, record.allocator_process_group, record.device);
  }
  for (auto& provider : state.provider_allocation_plans) {
    if (provider.geometry_mode != AllocationGeometryMode::kLifecycleCandidate ||
        provider.allocation_kind != ProviderAllocationKind::kTensorOutput || !provider.output ||
        !provider.tensor_template) {
      continue;
    }
    rebind_logical_id(provider.pool_id, tensor_output_pool_id(plan_generation, *provider.output));
    provider.tensor_template->allocator_root_id =
        tensor_allocator_root_id(plan_generation,
                                 provider.tensor_template->allocator_process_group,
                                 provider.tensor_template->device);
  }
  for (auto& record : state.tensor_byte_span_derivations) {
    record.plan_generation = plan_generation;
  }
  for (auto& record : state.tensor_memory_kind_derivations) {
    record.plan_generation = plan_generation;
  }
  for (auto& record : state.tensor_field_allocations) {
    record.plan_generation = plan_generation;
    rebind_logical_id(record.pool_id,
                      tensor_field_output_pool_id(plan_generation, record.output, record.field_id));
    record.allocator_root_id =
        tensor_allocator_root_id(plan_generation, record.allocator_process_group, record.device);
  }
  for (auto& record : state.tensor_interface_pools) {
    record.plan_generation = plan_generation;
    rebind_logical_id(record.pool_id,
                      record.field_id ? tensor_interface_field_pool_id(
                                            plan_generation, record.topic, *record.field_id)
                                      : tensor_interface_pool_id(plan_generation, record.topic));
    record.allocator_root_id =
        tensor_allocator_root_id(plan_generation, record.allocator_process_group, record.device);
    record.exporter_subpool_id = fmt::format("{}.exporter-generation", record.pool_id);
  }
  for (auto& record : state.tensor_device_resources) {
    record.plan_generation = plan_generation;
    rebind_logical_id(
        record.allocator_root_id,
        tensor_allocator_root_id(plan_generation, record.process_group, record.device));
  }
  std::vector<std::pair<std::string, std::string>> rebound_cuda_pools;
  rebound_cuda_pools.reserve(state.cuda_green_context_pools.size());
  for (auto& record : state.cuda_green_context_pools) {
    record.plan_generation = plan_generation;
    std::string replacement =
        cuda_green_context_pool_id(plan_generation, record.process_group, record.device);
    rebound_cuda_pools.emplace_back(record.pool_id, replacement);
    rebound_logical_ids.emplace_back(record.pool_id, replacement);
    record.pool_id = std::move(replacement);
  }
  const auto rebind_cuda_pool = [&](std::string& identity) {
    const auto found = std::find_if(rebound_cuda_pools.begin(),
                                    rebound_cuda_pools.end(),
                                    [&](const auto& row) { return row.first == identity; });
    if (found != rebound_cuda_pools.end()) {
      identity = found->second;
    }
  };
  const auto rebind_cuda_descendant = [&](std::string& identity) {
    const auto found =
        std::find_if(rebound_cuda_pools.begin(), rebound_cuda_pools.end(), [&](const auto& row) {
          return identity.starts_with(row.first + ".");
        });
    if (found != rebound_cuda_pools.end()) {
      identity = found->second + identity.substr(found->first.size());
    }
  };
  for (auto& record : state.cuda_green_context_resources) {
    record.plan_generation = plan_generation;
    rebind_cuda_pool(record.pool_id);
    const std::string prior_context_id = record.green_context_id;
    rebind_cuda_descendant(record.green_context_id);
    rebound_logical_ids.emplace_back(prior_context_id, record.green_context_id);
  }
  for (auto& record : state.cuda_green_context_requests) {
    record.plan_generation = plan_generation;
  }
  for (auto& record : state.cuda_green_context_assignments) {
    record.plan_generation = plan_generation;
    rebind_cuda_pool(record.pool_id);
    rebind_cuda_descendant(record.green_context_id);
  }
  for (auto& record : state.cuda_compute_streams) {
    record.plan_generation = plan_generation;
    if (record.green_context_id) {
      rebind_cuda_descendant(*record.green_context_id);
      const std::string prior_stream_id = record.stream_id;
      rebind_cuda_descendant(record.stream_id);
      rebound_logical_ids.emplace_back(prior_stream_id, record.stream_id);
    } else {
      record.owner_id =
          tensor_allocator_root_id(plan_generation, record.process_group, record.device);
      rebind_logical_id(
          record.stream_id,
          tensor_compute_stream_id(plan_generation, record.process_group, record.device));
    }
  }
  for (auto& capture_plan : state.cuda_graph_capture_plans) {
    rebind_operator(capture_plan.source);
    rebind_operator(capture_plan.sink);
    for (auto& member : capture_plan.members) {
      rebind_operator(member.operator_ref);
    }
    for (auto& contract : capture_plan.contracts) {
      rebind_operator(contract.owner);
    }
  }
  for (auto& record : state.process_local_channel_services) {
    const auto found = std::find_if(
        rebound_logical_ids.begin(), rebound_logical_ids.end(), [&](const auto& identity) {
          return identity.first == record.tensor_pool_id;
        });
    if (found != rebound_logical_ids.end()) {
      record.tensor_pool_id = found->second;
    }
    for (auto& field : record.tensor_fields) {
      const auto field_pool = std::find_if(
          rebound_logical_ids.begin(), rebound_logical_ids.end(), [&](const auto& identity) {
            return identity.first == field.pool_id;
          });
      if (field_pool != rebound_logical_ids.end()) {
        field.pool_id = field_pool->second;
      }
    }
  }
  for (auto& record : state.cross_process_channel_services) {
    const auto found = std::find_if(
        rebound_logical_ids.begin(), rebound_logical_ids.end(), [&](const auto& identity) {
          return identity.first == record.tensor_pool_id;
        });
    if (found != rebound_logical_ids.end()) {
      record.tensor_pool_id = found->second;
    }
    for (auto& field : record.tensor_fields) {
      const auto field_pool = std::find_if(
          rebound_logical_ids.begin(), rebound_logical_ids.end(), [&](const auto& identity) {
            return identity.first == field.pool_id;
          });
      if (field_pool != rebound_logical_ids.end()) {
        field.pool_id = field_pool->second;
      }
    }
  }
  for (auto& record : state.tensor_inline_records) {
    rebind_logical_id(record.fixed_record_bucket_id,
                      tensor_inline_record_bucket_id(plan_generation, record.topic));
  }
  for (auto& record : state.imported_generation_capacities) {
    rebind_logical_id(record.runtime_scope_id, runtime_scope_id(plan_generation));
  }
  for (auto& contributor : state.pool_capacity_contributors) {
    for (const auto& [prior, rebound] : rebound_logical_ids) {
      if (contributor.pool_id != prior) {
        continue;
      }
      if (contributor.contributor_id.starts_with(prior)) {
        contributor.contributor_id = rebound + contributor.contributor_id.substr(prior.size());
      }
      contributor.pool_id = rebound;
      break;
    }
  }
  for (auto& dependency : state.pool_capacity_dependencies) {
    for (const auto& [prior, rebound] : rebound_logical_ids) {
      if (dependency.pool_id != prior) {
        continue;
      }
      if (dependency.predecessor_contributor_id.starts_with(prior)) {
        dependency.predecessor_contributor_id =
            rebound + dependency.predecessor_contributor_id.substr(prior.size());
      }
      dependency.pool_id = rebound;
      break;
    }
  }
  for (auto& provider : state.provider_allocation_plans) {
    for (const auto& [prior, rebound] : rebound_logical_ids) {
      if (provider.pool_id == prior) {
        provider.pool_id = rebound;
        break;
      }
    }
  }
  const auto rebind_manifest_object = [&](std::string& object_id) {
    const auto found = std::find_if(
        rebound_logical_ids.begin(), rebound_logical_ids.end(), [&](const auto& identity) {
          return identity.first == object_id;
        });
    if (found != rebound_logical_ids.end()) {
      object_id = found->second;
    }
  };
  for (auto& capacity : state.profile_manifest.derived_capacities) {
    rebind_manifest_object(capacity.object_id);
  }
  for (auto& obligation : state.profile_manifest.materialization_obligations) {
    rebind_manifest_object(obligation.object_id);
  }

  rebind_owner_records(state.operator_completion_rules);
  rebind_owner_records(state.temporal_triggers);
  rebind_owner_records(state.aligned_triggers);
  rebind_owner_records(state.clock_triggers);
  rebind_owner_records(state.deadlines);
  rebind_owner_records(state.missing_aggregates);
  rebind_owner_records(state.window_triggers);
  rebind_owner_records(state.bracketed_windows);
  rebind_owner_records(state.parameter_derived_terms);
  for (auto& record : state.predicate_triggers) {
    rebind_operator(record.owner);
    record.graph_generation = plan_generation;
    record.state_generation = plan_generation;
  }
  for (auto& record : state.predicate_callables) {
    record.evidence_validity_epoch = plan_generation;
  }

  for (auto& record : state.lifecycle_checkpoints) {
    rebind_operator(record.operator_instance);
  }
  for (auto& record : state.lifecycle_dependencies) {
    rebind_operator(record.before);
    rebind_operator(record.after);
  }
  for (auto& record : state.lifecycle_restart_permissions) {
    rebind_operator(record.requester);
    rebind_operator(record.target);
  }
  for (auto& record : state.lifecycle_restart_targets) {
    rebind_operator(record.target);
  }
  for (auto& record : state.lifecycle_restart_closures) {
    rebind_operator(record.target);
    for (auto& member : record.closure) {
      rebind_operator(member);
    }
  }
  rebind_owner_records(state.lifecycle_mutable_fields);
  rebind_owner_records(state.operator_parameters);
  rebind_owner_records(state.parameter_fields);
  rebind_owner_records(state.parameter_generation_zero_operators);
  state.parameter_catalog =
      rebind_parameter_catalog_plan_generation(state.parameter_catalog, plan_generation);
}

}  // namespace holoscan::detail
