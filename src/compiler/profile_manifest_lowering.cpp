/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "profile_manifest_lowering.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "../core/source_id_internal.hpp"
#include "execution_plan_contracts.hpp"

namespace holoscan::detail {
namespace {

void add_evidence(ProfileManifest& manifest, std::string id, std::string kind,
                  std::uint32_t version = 1U) {
  const auto existing = std::find_if(manifest.evidence_references.begin(),
                                     manifest.evidence_references.end(),
                                     [&](const auto& item) { return item.id == id; });
  if (existing != manifest.evidence_references.end() && existing->kind == kind &&
      existing->version == version) {
    return;
  }
  manifest.evidence_references.push_back(ProfileEvidenceReferenceRecord{
      .id = std::move(id),
      .kind = std::move(kind),
      .version = version,
  });
}

void add_capability(ProfileManifest& manifest, std::string id, std::string evidence,
                    std::uint32_t version = 1U, std::string evidence_kind = "compile_requirement") {
  add_evidence(manifest, evidence, std::move(evidence_kind), 1U);
  manifest.required_capabilities.push_back(ProfileCapabilityRequirementRecord{
      .capability_id = std::move(id),
      .version = version,
      .evidence_id = std::move(evidence),
  });
}

void add_feature(ProfileManifest& manifest, std::string id, ProfileFeatureDisposition disposition,
                 std::string rule, std::string evidence = {}) {
  manifest.feature_dispositions.push_back(ProfileFeatureDispositionRecord{
      .feature_id = std::move(id),
      .disposition = disposition,
      .rule_id = std::move(rule),
      .evidence_id = std::move(evidence),
  });
}

void add_capacity(ProfileManifest& manifest, std::string object_id, std::string capacity_id,
                  std::uint64_t limit, std::string unit = "count") {
  manifest.derived_capacities.push_back(ProfileDerivedCapacityRecord{
      .object_id = std::move(object_id),
      .capacity_id = std::move(capacity_id),
      .limit = limit,
      .unit = std::move(unit),
  });
}

void add_materialization(ProfileManifest& manifest, std::string object_id, std::string strategy,
                         LifecycleStage stage, std::string evidence) {
  manifest.materialization_obligations.push_back(ProfileMaterializationObligationRecord{
      .object_id = std::move(object_id),
      .strategy_id = std::move(strategy),
      .complete_before_stage = stage,
      .cleanup_stage_mask = 0xC0U,
      .max_cleanup_actions = 1U,
      .evidence_id = std::move(evidence),
  });
}

[[nodiscard]] bool supplied_catalog_contains(const CompileOptions& options, std::string_view name) {
  if (!options.endpoint_catalog) {
    return false;
  }
  const auto* catalog = ChannelEndpointCatalogAccess::state(*options.endpoint_catalog);
  return catalog != nullptr &&
         std::any_of(catalog->endpoints.begin(),
                     catalog->endpoints.end(),
                     [name](const auto& endpoint) { return endpoint.name == name; });
}

[[nodiscard]] bool deployment_pin_contains(const CompileOptions& options, std::string_view name,
                                           std::string_view algorithm, std::uint32_t version) {
  return std::any_of(options.deployment.generated_endpoint_names().begin(),
                     options.deployment.generated_endpoint_names().end(),
                     [&](const auto& pin) {
                       return pin.final_name == name && pin.algorithm == algorithm &&
                              pin.version == version;
                     });
}

[[nodiscard]] bool profile_present(const ProfileManifest& manifest, Profile profile) {
  return manifest.effective_graph_profile == profile ||
         std::any_of(
             manifest.effective_regions.begin(),
             manifest.effective_regions.end(),
             [profile](const auto& region) { return region.effective_profile == profile; }) ||
         std::any_of(manifest.scope_resolutions.begin(),
                     manifest.scope_resolutions.end(),
                     [profile](const auto& scope) { return scope.effective_profile == profile; });
}

void add_profile_requirements(ProfileManifest& manifest) {
  add_capability(
      manifest, "holoscan.runtime.execution_plan", "holoscan.compile.execution_plan.private.v1");
  if (!manifest.effective_graph_profile) {
    return;
  }
  if (profile_present(manifest, Profile::kResearch)) {
    add_feature(manifest,
                "profile.research",
                ProfileFeatureDisposition::kAdmitted,
                "holoscan.profile.research.named_relaxations/v1");
  }
  if (profile_present(manifest, Profile::kStandard)) {
    add_feature(manifest,
                "profile.standard",
                ProfileFeatureDisposition::kAdmitted,
                "holoscan.profile.standard.baseline/v1");
  }
  if (profile_present(manifest, Profile::kRobotics)) {
    add_capability(manifest,
                   "holoscan.profile.robotics.finite_evidence",
                   "holoscan.profile.robotics.finite_evidence/v1");
    add_feature(manifest,
                "profile.robotics",
                ProfileFeatureDisposition::kRestricted,
                "holoscan.profile.robotics.finite_evidence/v1",
                "holoscan.profile.robotics.finite_evidence/v1");
  }
  const auto has_critical_feature = [&](std::string_view suffix) {
    return std::any_of(
        manifest.feature_dispositions.begin(),
        manifest.feature_dispositions.end(),
        [suffix](const auto& feature) { return feature.feature_id.ends_with(suffix); });
  };
  if (has_critical_feature("holoscan.profile.robotics_critical.latency_budget")) {
    add_capability(manifest,
                   "holoscan.profile.robotics_critical.latency_budget",
                   "holoscan.profile.robotics_critical.latency_budget/v1");
  }
  if (has_critical_feature("holoscan.profile.robotics_critical.bounded_memory")) {
    add_capability(manifest,
                   "holoscan.profile.robotics_critical.bounded_memory",
                   "holoscan.profile.robotics_critical.bounded_memory/v1");
  }
  if (profile_present(manifest, Profile::kRealTime)) {
    add_capability(manifest,
                   "holoscan.profile.real_time.bounded_hot_path",
                   "holoscan.profile.real_time.bounded_hot_path/v1");
    add_feature(manifest,
                "profile.real_time",
                ProfileFeatureDisposition::kAdmitted,
                "holoscan.profile.real_time.evidence_satisfied/v1",
                "holoscan.profile.real_time.bounded_hot_path/v1");
  }
  if (profile_present(manifest, Profile::kRegulated)) {
    add_capability(manifest,
                   "holoscan.profile.regulated.complete_evidence",
                   "holoscan.profile.regulated.complete_evidence/v1");
    add_feature(manifest,
                "profile.regulated",
                ProfileFeatureDisposition::kAdmitted,
                "holoscan.profile.regulated.evidence_satisfied/v1",
                "holoscan.profile.regulated.complete_evidence/v1");
  }
}

void add_partition_and_lane_requirements(ExecutionPlanState& state) {
  ProfileManifest& manifest = state.profile_manifest;
  add_capability(manifest,
                 "holoscan.runtime.ready.deterministic_serial",
                 "holoscan.compile.scheduler.ready_policy/v1");
  add_feature(manifest,
              "partitioning",
              state.partitioning_plan.disposition == PlanDisposition::kAccepted
                  ? ProfileFeatureDisposition::kAdmitted
                  : ProfileFeatureDisposition::kRejected,
              "holoscan.partition.exact_cover/v1",
              "holoscan.compile.partitioning_plan/v1");
  add_capacity(manifest, "scheduler", "execution_lanes", state.scheduler_plan.execution_lane_count);
  add_capacity(manifest,
               "scheduler",
               "max_concurrent_activations",
               state.scheduler_plan.max_concurrent_activations);
  add_capacity(manifest, "partitioning", "partitions", state.execution_partitions.size());
  for (const auto& lane : state.execution_lanes) {
    const std::string object = fmt::format("lane:{}", lane.execution_lane);
    add_materialization(manifest,
                        object,
                        "holoscan.runtime.execution_lane/v1",
                        LifecycleStage::kStart,
                        "holoscan.compile.thread_placement.validation/v1");
  }
  if (!state.partition_placements.empty() || !state.support_thread_affinities.empty()) {
    add_capability(manifest,
                   "holoscan.runtime.thread_placement.apply_readback",
                   "holoscan.runtime.thread_placement.apply_readback/v1",
                   1U,
                   "placement_probe");
  }
  for (const auto& probe : state.placement_probes) {
    if (!probe.evidence_id.empty()) {
      add_evidence(manifest, probe.evidence_id, "placement_probe", 1U);
    }
    if (probe.status == PlacementProbeStatus::kFallbackRecorded) {
      manifest.explicit_fallbacks.push_back(ProfileFallbackRecord{
          .feature_id = fmt::format("execution_partition:{}", probe.partition.value),
          .fallback_id = "holoscan.placement.irq_isolation.unproven/v1",
          .reason_id = "holoscan.placement.probe_warn_fallback/v1",
          .evidence_id = probe.evidence_id,
      });
    }
  }
}

void add_readiness_and_temporal_requirements(ExecutionPlanState& state) {
  ProfileManifest& manifest = state.profile_manifest;
  if (!state.readiness.empty()) {
    add_capability(
        manifest, "holoscan.runtime.readiness_sources", "holoscan.compile.readiness_plan/v1");
  }
  for (const auto& source : state.readiness) {
    const std::string object = fmt::format("source:{}", source_id_value(source.source_id));
    add_capacity(manifest, object, "witness_capacity", source.accepted_capacity);
    add_capacity(
        manifest, object, "sender_reference_capacity", source.accepted_sender_reference_capacity);
    if (source.materialized) {
      add_materialization(manifest,
                          object,
                          "holoscan.runtime.readiness_adapter/v1",
                          LifecycleStage::kArm,
                          "holoscan.compile.readiness_plan/v1");
    }
  }
  if (!state.temporal_triggers.empty()) {
    add_capability(
        manifest, "holoscan.runtime.temporal_contracts", "holoscan.compile.temporal_plan/v1");
    add_feature(manifest,
                "temporal_contracts",
                ProfileFeatureDisposition::kAdmitted,
                "holoscan.temporal.bounded_state/v1",
                "holoscan.compile.temporal_plan/v1");
  }
  for (const auto& selection : state.input_selections) {
    add_capacity(manifest,
                 fmt::format("port:{}", selection.input.value()),
                 "max_selected_samples",
                 selection.max_selected_samples);
  }
}

void add_predicate_replay_requirements(ExecutionPlanState& state) {
  ProfileManifest& manifest = state.profile_manifest;
  for (const auto& trigger : state.predicate_triggers) {
    if (trigger.replay != PredicateReplayCapability::kCrossBuildRecomputation ||
        trigger.identity_scope != PredicateIdentityScope::kIndependentlyKeyed ||
        !trigger.semantic_version) {
      continue;
    }
    const auto callable = std::find_if(
        state.predicate_callables.begin(), state.predicate_callables.end(), [&](const auto& item) {
          return item.predicate_id == trigger.id;
        });
    if (callable == state.predicate_callables.end() ||
        callable->implementation_fingerprint.empty() ||
        trigger.replay_evidence != callable->implementation_fingerprint) {
      continue;
    }
    add_capability(manifest,
                   runtime_predicate_capability_id(trigger.identity, *trigger.semantic_version),
                   callable->implementation_fingerprint);
  }
}

// Physical capability projection is one closed cross-record join.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void add_physical_requirements(ExecutionPlanState& state) {
  ProfileManifest& manifest = state.profile_manifest;
  if (!state.payload_edges.empty()) {
    add_capability(manifest, "holoscan.runtime.payload_edges", "holoscan.compile.physical_plan/v1");
  }
  for (const auto& edge : state.payload_edges) {
    const std::string object = fmt::format("edge:{}", edge.edge.value);
    add_capacity(manifest, object, "queue_depth", edge.queue_depth);
    add_capacity(manifest, object, "max_inflight_samples", edge.max_inflight_samples);
    add_capacity(manifest, object, "cache_capacity", edge.cache_capacity);
    add_capacity(manifest, object, "producer_burst", edge.producer_max_emits_per_compute);
    add_capacity(manifest, object, "fanout", edge.fanout);
  }

  if (!state.channel_services.empty()) {
    add_capability(manifest,
                   "holoscan.runtime.channel_service_attach",
                   "holoscan.compile.channel_service_plan/v1");
  }
  if (!state.cross_process_channel_services.empty()) {
    // Cross-process construction reuses the one package-owned transport
    // provider. Recording its exact offer here keeps locality from becoming a
    // runtime backend selector and makes a missing implementation fail at the
    // package gate before any Operator reconstruction.
    add_capability(manifest,
                   "holoscan.runtime.channel.holomq_transport",
                   "holoscan.channel.holomq_transport/v1");
  }
  struct ChannelServiceCapacities {
    std::uint32_t max_subscriber_queue_depth{};
    std::uint32_t max_subscriber_borrowed_samples{};
    std::uint32_t publisher_history_depth{};
  };
  std::map<std::string, ChannelServiceCapacities> channel_service_capacities;
  for (const auto& service : state.channel_services) {
    auto& capacities = channel_service_capacities[service.service_id];
    capacities.max_subscriber_queue_depth =
        std::max(capacities.max_subscriber_queue_depth, service.max_subscriber_queue_depth);
    capacities.max_subscriber_borrowed_samples = std::max(
        capacities.max_subscriber_borrowed_samples, service.max_subscriber_borrowed_samples);
    capacities.publisher_history_depth =
        std::max(capacities.publisher_history_depth, service.publisher_history_depth);
  }
  for (const auto& [service_id, capacities] : channel_service_capacities) {
    add_capacity(
        manifest, service_id, "max_subscriber_queue_depth", capacities.max_subscriber_queue_depth);
    add_capacity(manifest,
                 service_id,
                 "max_subscriber_borrowed_samples",
                 capacities.max_subscriber_borrowed_samples);
    add_capacity(
        manifest, service_id, "publisher_history_depth", capacities.publisher_history_depth);
    add_materialization(manifest,
                        service_id,
                        "holoscan.runtime.channel_service_attach/v1",
                        LifecycleStage::kStart,
                        "holoscan.compile.channel_service_plan/v1");
  }

  if (!state.generic_output_loans.empty()) {
    add_capability(manifest,
                   "holoscan.runtime.generic_output_loans",
                   "holoscan.compile.generic_output_loan_plan/v1");
  }
  for (const auto& loan : state.generic_output_loans) {
    add_capacity(manifest, loan.pool_id, "slots", loan.slot_count);
    add_capacity(manifest, loan.pool_id, "aggregate_bytes", loan.aggregate_byte_capacity);
    add_capacity(manifest, loan.pool_id, "max_live_loans", loan.max_live_loans);
    add_capacity(manifest, loan.pool_id, "retirement_records", loan.retirement_capacity);
    add_materialization(manifest,
                        loan.pool_id,
                        "holoscan.runtime.generic_output_pool/v1",
                        loan.materialize_before,
                        "holoscan.compile.generic_output_loan_plan/v1");
  }

  const bool has_lifecycle_tensor_provider =
      std::ranges::any_of(state.provider_allocation_plans, [](const auto& provider) {
        return provider.disposition == PlanDisposition::kAccepted &&
               provider.geometry_mode == AllocationGeometryMode::kLifecycleCandidate &&
               (provider.allocation_kind == ProviderAllocationKind::kTensorOutput ||
                provider.allocation_kind == ProviderAllocationKind::kTensorFieldOutput);
      });
  if (!state.tensor_allocations.empty() || !state.tensor_interface_pools.empty() ||
      has_lifecycle_tensor_provider) {
    add_capability(manifest,
                   "holoscan.runtime.tensor_pool_materialization",
                   "holoscan.compile.tensor_pool_plan/v1");
  }
  const auto add_tensor_capacities = [&](std::string_view pool_id,
                                         std::uint32_t slots,
                                         std::uint32_t control,
                                         std::uint32_t readiness,
                                         std::uint32_t release,
                                         std::uint32_t retirement,
                                         std::uint32_t quarantine,
                                         std::uint64_t aggregate_bytes) {
    add_capacity(manifest, std::string(pool_id), "slots", slots);
    add_capacity(manifest, std::string(pool_id), "control_records", control);
    add_capacity(manifest, std::string(pool_id), "readiness_tokens", readiness);
    add_capacity(manifest, std::string(pool_id), "release_records", release);
    add_capacity(manifest, std::string(pool_id), "retirement_records", retirement);
    add_capacity(manifest, std::string(pool_id), "quarantine_records", quarantine);
    add_capacity(manifest, std::string(pool_id), "aggregate_bytes", aggregate_bytes);
  };
  for (const auto& pool : state.tensor_allocations) {
    add_tensor_capacities(pool.pool_id,
                          pool.slot_count,
                          pool.control_capacity,
                          pool.readiness_capacity,
                          pool.release_capacity,
                          pool.retirement_capacity,
                          pool.quarantine_capacity,
                          pool.aggregate_byte_capacity);
    add_materialization(manifest,
                        pool.pool_id,
                        "holoscan.runtime.tensor_output_pool/v1",
                        pool.materialize_before,
                        "holoscan.compile.tensor_pool_plan/v1");
  }
  std::map<std::string, std::uint64_t> cuda_graph_fixed_allocations;
  for (const auto& region : state.cuda_graph_execution_regions) {
    for (const auto& buffer : region.buffers) {
      cuda_graph_fixed_allocations.emplace(buffer.fixed_allocation_id,
                                           buffer.aggregate_byte_capacity);
    }
  }
  for (const auto& [allocation_id, aggregate_bytes] : cuda_graph_fixed_allocations) {
    add_capacity(manifest, allocation_id, "active_slots", 1U);
    add_capacity(manifest, allocation_id, "aggregate_bytes", aggregate_bytes);
  }
  for (const auto& provider : state.provider_allocation_plans) {
    if (provider.disposition != PlanDisposition::kAccepted ||
        provider.geometry_mode != AllocationGeometryMode::kLifecycleCandidate ||
        (provider.allocation_kind != ProviderAllocationKind::kTensorOutput &&
         provider.allocation_kind != ProviderAllocationKind::kTensorFieldOutput)) {
      continue;
    }
    add_materialization(manifest,
                        provider.pool_id,
                        "holoscan.runtime.tensor_output_pool/v1",
                        provider.complete_before,
                        "holoscan.compile.tensor_pool_plan/v1");
  }
  for (const auto& pool : state.tensor_interface_pools) {
    add_tensor_capacities(pool.pool_id,
                          pool.slot_count,
                          pool.control_capacity,
                          pool.readiness_capacity,
                          pool.release_capacity,
                          pool.retirement_capacity,
                          pool.quarantine_capacity,
                          pool.aggregate_byte_capacity);
    add_capacity(manifest, pool.pool_id, "publisher_loan_credit", pool.publisher_loan_capacity);
    add_materialization(manifest,
                        pool.pool_id,
                        "holoscan.runtime.tensor_interface_pool/v1",
                        LifecycleStage::kArm,
                        "holoscan.compile.tensor_pool_plan/v1");
  }
  if (!state.cuda_compute_streams.empty()) {
    add_capability(manifest,
                   "holoscan.runtime.cuda_compute_stream_materialization",
                   "holoscan.compile.cuda_compute_stream_plan/v1");
  }
  if (!state.cuda_green_context_pools.empty()) {
    add_capability(manifest,
                   "holoscan.runtime.cuda_green_context_materialization",
                   "holoscan.compile.cuda_green_context_plan/v1");
    const auto probe = std::find_if(
        state.placement_probes.begin(), state.placement_probes.end(), [](const auto& row) {
          return row.probe_id == "cuda.green_context.split" && !row.evidence_id.empty();
        });
    add_feature(manifest,
                "cuda.green_context",
                ProfileFeatureDisposition::kAdmitted,
                "holoscan.profile.cuda_green_context.hot_path/v1",
                probe == state.placement_probes.end() ? std::string{} : probe->evidence_id);
  }
  for (const auto& pool : state.cuda_green_context_pools) {
    add_materialization(manifest,
                        pool.pool_id,
                        "holoscan.runtime.cuda_green_context_pool/v1",
                        pool.materialize_before,
                        "holoscan.compile.cuda_green_context_plan/v1");
  }
  for (const auto& resource : state.cuda_green_context_resources) {
    add_materialization(manifest,
                        resource.green_context_id,
                        "holoscan.runtime.cuda_green_context/v1",
                        LifecycleStage::kArm,
                        "holoscan.compile.cuda_green_context_plan/v1");
  }
  for (const auto& resource : state.tensor_device_resources) {
    add_capacity(manifest, resource.allocator_root_id, "pools", resource.pool_count);
    add_capacity(manifest, resource.allocator_root_id, "slots", resource.slot_capacity);
    add_capacity(
        manifest, resource.allocator_root_id, "loan_credits", resource.loan_credit_capacity);
    add_capacity(
        manifest, resource.allocator_root_id, "control_records", resource.control_capacity);
    add_capacity(
        manifest, resource.allocator_root_id, "readiness_tokens", resource.readiness_capacity);
    add_capacity(
        manifest, resource.allocator_root_id, "release_records", resource.release_capacity);
    add_capacity(
        manifest, resource.allocator_root_id, "retirement_records", resource.retirement_capacity);
    add_capacity(
        manifest, resource.allocator_root_id, "quarantine_records", resource.quarantine_capacity);
  }
  for (const auto& stream : state.cuda_compute_streams) {
    add_materialization(manifest,
                        stream.stream_id,
                        "holoscan.runtime.cuda_compute_stream/v1",
                        stream.materialize_before,
                        "holoscan.compile.cuda_compute_stream_plan/v1");
  }
  if (!state.tensor_inline_records.empty()) {
    const auto& provider = state.tensor_inline_records.front();
    add_capability(manifest,
                   provider.capability_id,
                   provider.capability_evidence_id,
                   provider.capability_version,
                   "generated_provider_evidence");
  }
  for (const auto& record : state.tensor_inline_records) {
    add_capacity(manifest,
                 record.fixed_record_bucket_id,
                 "fixed_record_bytes",
                 record.fixed_record_bucket_bytes,
                 "bytes");
    add_capacity(manifest,
                 record.fixed_record_bucket_id,
                 "fixed_metadata_overhead_bytes",
                 record.fixed_metadata_overhead_bytes,
                 "bytes");
    add_capacity(manifest,
                 record.fixed_record_bucket_id,
                 "max_inline_logical_bytes",
                 record.accepted_max_inline_logical_bytes,
                 "bytes");
    add_capacity(manifest,
                 record.fixed_record_bucket_id,
                 "provider_profile_max_inline_logical_bytes",
                 record.effective_max_inline_logical_bytes,
                 "bytes");
    add_materialization(manifest,
                        record.fixed_record_bucket_id,
                        "holoscan.runtime.tensor_inline_record_bucket/v1",
                        LifecycleStage::kArm,
                        record.capability_evidence_id);
  }
  std::map<std::string, std::uint32_t> service_generation_capacities;
  std::map<std::string, std::uint32_t> runtime_generation_capacities;
  for (const auto& generation : state.imported_generation_capacities) {
    const std::string endpoint_scope =
        fmt::format("imported-generation-endpoint:{}", generation.topic.value);
    add_capacity(
        manifest, endpoint_scope, "endpoint_generations", generation.endpoint_max_generations);
    add_capacity(manifest,
                 endpoint_scope,
                 "candidate_generation_reservations",
                 generation.candidate_reservation_capacity);
    add_capacity(
        manifest, endpoint_scope, "generation_release_records", generation.release_record_capacity);
    add_capacity(manifest,
                 endpoint_scope,
                 "generation_quarantine_records",
                 generation.quarantine_record_capacity);
    auto& service_capacity = service_generation_capacities[generation.endpoint_or_service_id];
    service_capacity = std::max(service_capacity, generation.service_max_generations);
    auto& runtime_capacity = runtime_generation_capacities[generation.runtime_scope_id];
    runtime_capacity = std::max(runtime_capacity, generation.runtime_max_generations);
  }
  for (const auto& [service_id, capacity] : service_generation_capacities) {
    add_capacity(manifest, service_id, "service_generations", capacity);
  }
  for (const auto& [runtime_scope_id, capacity] : runtime_generation_capacities) {
    add_capacity(manifest, runtime_scope_id, "runtime_generations", capacity);
  }
  if (!state.tensor_transfer_plans.empty()) {
    add_capability(manifest,
                   "holoscan.runtime.tensor_transfer_matrix",
                   "holoscan.compile.tensor_transfer_plan/v1");
  }
  for (const auto& transfer : state.tensor_transfer_plans) {
    if (transfer.disposition != PlanDisposition::kAccepted || !transfer.realization) {
      continue;
    }
    const std::string field_id =
        transfer.field_id ? fmt::format("{}", *transfer.field_id) : std::string{"none"};
    const std::string tensor_index =
        transfer.tensor_index ? fmt::format("{}", *transfer.tensor_index) : std::string{"none"};
    const std::string object =
        fmt::format("tensor-topic:{}:field:{}:tensor:{}:source:{}:bytes:{}-{}",
                    transfer.topic.value,
                    field_id,
                    tensor_index,
                    static_cast<std::uint32_t>(transfer.source_class),
                    transfer.min_logical_bytes,
                    transfer.max_logical_bytes);
    add_capacity(manifest, object, "max_logical_bytes", transfer.max_logical_bytes);
    if (transfer.requires_inline_record) {
      add_materialization(manifest,
                          object,
                          "holoscan.runtime.tensor_inline_record/v1",
                          LifecycleStage::kArm,
                          transfer.evidence_id);
    }
    if (transfer.requires_exportable_copy_pool) {
      add_materialization(manifest,
                          object,
                          "holoscan.runtime.tensor_exportable_copy_pool/v1",
                          LifecycleStage::kArm,
                          transfer.evidence_id);
    }
  }

  if (!state.publication_outcomes.empty()) {
    add_feature(manifest,
                "publication_outcomes",
                ProfileFeatureDisposition::kAdmitted,
                "holoscan.publication.no_runtime_retry/v1",
                "holoscan.compile.physical_plan/v1");
  }
}

void add_endpoint_requirements(const CompileOptions& options, ExecutionPlanState& state) {
  ProfileManifest& manifest = state.profile_manifest;
  const auto* catalog = ChannelEndpointCatalogAccess::state(state.endpoint_catalog);
  const auto endpoints = catalog == nullptr
                             ? std::span<const ChannelEndpointRecord>{}
                             : std::span<const ChannelEndpointRecord>{catalog->endpoints};
  if (!endpoints.empty()) {
    add_capability(manifest,
                   "holoscan.runtime.channel_endpoint_catalog",
                   "holoscan.compile.endpoint_catalog/v1");
    add_capacity(manifest, "endpoint_catalog", "endpoints", endpoints.size());
    add_materialization(manifest,
                        "endpoint_catalog",
                        "holoscan.runtime.endpoint_catalog_attach/v1",
                        LifecycleStage::kStart,
                        "holoscan.compile.endpoint_catalog/v1");
  }
  for (const auto& endpoint : endpoints) {
    if (!endpoint.generated_name) {
      continue;
    }
    const bool catalog = supplied_catalog_contains(options, endpoint.name);
    const bool allowlist = deployment_pin_contains(options,
                                                   endpoint.name,
                                                   endpoint.generated_name->algorithm,
                                                   endpoint.generated_name->version);
    const auto source = catalog     ? GeneratedEndpointNameAdmissionSource::kSuppliedCatalog
                        : allowlist ? GeneratedEndpointNameAdmissionSource::kDeploymentAllowlist
                                    : GeneratedEndpointNameAdmissionSource::kNone;
    const bool admitted = source != GeneratedEndpointNameAdmissionSource::kNone ||
                          manifest.effective_graph_profile == Profile::kStandard ||
                          manifest.effective_graph_profile == Profile::kResearch;
    manifest.generated_endpoint_names.push_back(GeneratedEndpointNameAdmissionRecord{
        .final_name = endpoint.name,
        .algorithm = endpoint.generated_name->algorithm,
        .version = endpoint.generated_name->version,
        .source = source,
        .disposition =
            admitted ? ProfileFeatureDisposition::kAdmitted : ProfileFeatureDisposition::kRejected,
        .evidence_id = catalog     ? "holoscan.deployment.supplied_endpoint_catalog/v1"
                       : allowlist ? "holoscan.deployment.endpoint_allowlist/v1"
                                   : "",
    });
  }
}

[[nodiscard]] const ProfileScopeResolutionRecord* find_scope(const ProfileManifest& manifest,
                                                             std::string_view kind,
                                                             std::string_view id) {
  const auto found = std::find_if(
      manifest.scope_resolutions.begin(), manifest.scope_resolutions.end(), [&](const auto& scope) {
        return scope.object_kind == kind && scope.object_id == id;
      });
  return found == manifest.scope_resolutions.end() ? nullptr : &*found;
}

[[nodiscard]] ProfileScopeResolutionRecord joined_scope(
    const ProfileManifest& manifest, std::string kind, std::string id,
    std::span<const ProfileScopeResolutionRecord* const> contributors) {
  ProfileScopeResolutionRecord result{
      .object_id = std::move(id),
      .object_kind = std::move(kind),
      .effective_profile = manifest.effective_graph_profile.value_or(Profile::kStandard),
  };
  for (const auto* scope : contributors) {
    if (scope == nullptr) {
      continue;
    }
    const auto current = profile_semantic_rank(result.effective_profile);
    const auto candidate = profile_semantic_rank(scope->effective_profile);
    if (current && candidate && *candidate > *current) {
      result.effective_profile = scope->effective_profile;
    }
    result.contributing_regions.insert(result.contributing_regions.end(),
                                       scope->contributing_regions.begin(),
                                       scope->contributing_regions.end());
  }
  std::sort(result.contributing_regions.begin(),
            result.contributing_regions.end(),
            [](RegionId lhs, RegionId rhs) { return lhs.value < rhs.value; });
  result.contributing_regions.erase(
      std::unique(result.contributing_regions.begin(), result.contributing_regions.end()),
      result.contributing_regions.end());
  return result;
}

// Publish all effective scope kinds together so completeness is auditable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void append_complete_profile_scopes(ExecutionPlanState& state) {
  ProfileManifest& manifest = state.profile_manifest;
  const auto port_scope = [&](PortId port) {
    const std::string id = fmt::format("port:{}", port.value());
    return find_scope(manifest, "port", id);
  };
  for (const auto& edge : state.payload_edges) {
    const std::array scopes{port_scope(edge.source), port_scope(edge.target)};
    manifest.scope_resolutions.push_back(
        joined_scope(manifest,
                     "channel",
                     fmt::format("channel:edge:{}", edge.edge.value),
                     std::span<const ProfileScopeResolutionRecord* const>{scopes}));
  }
  // Keep stable scope identities while the destination vector grows. Pointers into
  // scope_resolutions cannot survive the reserve below or a preceding derived-scope append.
  std::map<std::string, std::vector<std::string>> channel_scope_ids;
  for (const auto& service : state.channel_services) {
    const std::string boundary_id = fmt::format("boundary:{}", service.topic.value);
    channel_scope_ids[service.service_id].push_back(boundary_id);
  }
  manifest.scope_resolutions.reserve(manifest.scope_resolutions.size() + channel_scope_ids.size());
  for (const auto& [service_id, scope_ids] : channel_scope_ids) {
    std::vector<const ProfileScopeResolutionRecord*> scopes;
    scopes.reserve(scope_ids.size());
    for (const auto& scope_id : scope_ids) {
      scopes.push_back(find_scope(manifest, "boundary", scope_id));
    }
    manifest.scope_resolutions.push_back(
        joined_scope(manifest, "channel", fmt::format("channel:{}", service_id), scopes));
  }
  for (const auto& binding : state.interface_port_bindings) {
    const std::array scopes{port_scope(binding.public_port), port_scope(binding.internal_port)};
    manifest.scope_resolutions.push_back(joined_scope(
        manifest,
        "bridge",
        fmt::format("bridge:{}:{}", binding.public_port.value(), binding.internal_port.value()),
        std::span<const ProfileScopeResolutionRecord* const>{scopes}));
  }
  for (const auto& conversion : state.clock_conversions) {
    manifest.scope_resolutions.push_back(
        joined_scope(manifest, "converter", fmt::format("converter:{}", conversion.id), {}));
  }
  for (const auto& source : state.readiness) {
    const std::string owner_id = fmt::format("operator:{}", source.owner.value);
    const std::array scopes{find_scope(manifest, "operator", owner_id)};
    manifest.scope_resolutions.push_back(
        joined_scope(manifest,
                     "helper",
                     fmt::format("helper:source:{}", source_id_value(source.source_id)),
                     std::span<const ProfileScopeResolutionRecord* const>{scopes}));
  }
  for (const auto& feature : manifest.feature_dispositions) {
    if (find_scope(manifest, "feature", feature.feature_id) != nullptr) {
      continue;
    }
    ProfileScopeResolutionRecord scope = joined_scope(manifest, "feature", feature.feature_id, {});
    const auto feature_profile = [&]() -> std::optional<Profile> {
      if (feature.feature_id == "profile.research") {
        return Profile::kResearch;
      }
      if (feature.feature_id == "profile.standard") {
        return Profile::kStandard;
      }
      if (feature.feature_id == "profile.robotics") {
        return Profile::kRobotics;
      }
      if (feature.feature_id == "profile.real_time") {
        return Profile::kRealTime;
      }
      if (feature.feature_id == "profile.regulated") {
        return Profile::kRegulated;
      }
      return std::nullopt;
    }();
    if (feature_profile) {
      scope.effective_profile = *feature_profile;
      for (const auto& candidate : manifest.scope_resolutions) {
        if (candidate.effective_profile == *feature_profile) {
          scope.contributing_regions.insert(scope.contributing_regions.end(),
                                            candidate.contributing_regions.begin(),
                                            candidate.contributing_regions.end());
        }
      }
      std::sort(scope.contributing_regions.begin(),
                scope.contributing_regions.end(),
                [](RegionId lhs, RegionId rhs) { return lhs.value < rhs.value; });
      scope.contributing_regions.erase(
          std::unique(scope.contributing_regions.begin(), scope.contributing_regions.end()),
          scope.contributing_regions.end());
    }
    manifest.scope_resolutions.push_back(std::move(scope));
  }
}

[[nodiscard]] Error duplicate_key_error() {
  return Error{ErrorCode::kInvalidArgument, "PROFILE_MANIFEST_DUPLICATE_KEY"};
}

[[nodiscard]] auto scope_resolution_key(const ProfileScopeResolutionRecord& record) noexcept {
  return std::tuple{std::string_view{record.object_kind}, std::string_view{record.object_id}};
}

[[nodiscard]] auto capability_requirement_key(
    const ProfileCapabilityRequirementRecord& record) noexcept {
  return std::tuple{std::string_view{record.capability_id}, record.version};
}

[[nodiscard]] std::string_view evidence_reference_key(
    const ProfileEvidenceReferenceRecord& record) noexcept {
  return record.id;
}

[[nodiscard]] std::string_view feature_disposition_key(
    const ProfileFeatureDispositionRecord& record) noexcept {
  return record.feature_id;
}

[[nodiscard]] auto derived_capacity_key(const ProfileDerivedCapacityRecord& record) noexcept {
  return std::tuple{std::string_view{record.object_id}, std::string_view{record.capacity_id}};
}

[[nodiscard]] auto materialization_obligation_key(
    const ProfileMaterializationObligationRecord& record) noexcept {
  return std::tuple{std::string_view{record.object_id},
                    std::string_view{record.strategy_id},
                    record.complete_before_stage};
}

[[nodiscard]] auto explicit_fallback_key(const ProfileFallbackRecord& record) noexcept {
  return std::tuple{std::string_view{record.feature_id}, std::string_view{record.fallback_id}};
}

[[nodiscard]] auto generated_endpoint_name_key(
    const GeneratedEndpointNameAdmissionRecord& record) noexcept {
  return std::tuple{
      std::string_view{record.final_name}, std::string_view{record.algorithm}, record.version};
}

template <typename Records, typename Key>
[[nodiscard]] bool has_duplicate_key(const Records& records, Key key) {
  return std::adjacent_find(records.begin(), records.end(), [&](const auto& lhs, const auto& rhs) {
           return key(lhs) == key(rhs);
         }) != records.end();
}

template <typename Records, typename Key>
void sort_by_key(Records& records, Key key) {
  std::sort(records.begin(), records.end(), [&](const auto& lhs, const auto& rhs) {
    return key(lhs) < key(rhs);
  });
}

[[nodiscard]] expected<void, Error> canonicalize(ProfileManifest& manifest) {
  sort_by_key(manifest.required_capabilities, capability_requirement_key);
  sort_by_key(manifest.scope_resolutions, scope_resolution_key);
  sort_by_key(manifest.evidence_references, evidence_reference_key);
  sort_by_key(manifest.feature_dispositions, feature_disposition_key);
  sort_by_key(manifest.derived_capacities, derived_capacity_key);
  sort_by_key(manifest.materialization_obligations, materialization_obligation_key);
  sort_by_key(manifest.explicit_fallbacks, explicit_fallback_key);
  sort_by_key(manifest.generated_endpoint_names, generated_endpoint_name_key);
  if (has_duplicate_key(manifest.scope_resolutions, scope_resolution_key) ||
      has_duplicate_key(manifest.required_capabilities, capability_requirement_key) ||
      has_duplicate_key(manifest.evidence_references, evidence_reference_key) ||
      has_duplicate_key(manifest.feature_dispositions, feature_disposition_key) ||
      has_duplicate_key(manifest.derived_capacities, derived_capacity_key) ||
      has_duplicate_key(manifest.materialization_obligations, materialization_obligation_key) ||
      has_duplicate_key(manifest.explicit_fallbacks, explicit_fallback_key) ||
      has_duplicate_key(manifest.generated_endpoint_names, generated_endpoint_name_key)) {
    return make_unexpected(duplicate_key_error());
  }
  if (std::any_of(manifest.materialization_obligations.begin(),
                  manifest.materialization_obligations.end(),
                  [](const auto& obligation) {
                    constexpr std::uint8_t kStopReleaseMask = 0xC0U;
                    return obligation.cleanup_stage_mask != kStopReleaseMask ||
                           obligation.max_cleanup_actions == 0U;
                  })) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "PROFILE_MATERIALIZATION_FAILED"});
  }
  return {};
}

void reject_profile_features(ProfileManifest& manifest) {
  for (auto& feature : manifest.feature_dispositions) {
    if (feature.feature_id == "profile.real_time" || feature.feature_id == "profile.regulated") {
      feature.disposition = ProfileFeatureDisposition::kRejected;
      feature.rule_id = feature.feature_id == "profile.real_time"
                            ? "holoscan.profile.real_time.fail_closed/v1"
                            : "holoscan.profile.regulated.fail_closed/v1";
    }
  }
}

[[nodiscard]] expected<void, Error> match_package_capabilities(
    ProfileManifest& manifest, const RuntimePackageCapabilityView& package) {
  std::string_view token;
  if (!package.supports_plan_schema(manifest.plan_schema_version)) {
    token = "PROFILE_PLAN_SCHEMA_UNSUPPORTED";
  } else if (!package.supports_profile_policy(manifest.profile_policy_id)) {
    token = "PROFILE_POLICY_UNSUPPORTED";
  } else if (!package.supports_runtime_abi(manifest.runtime_abi_version)) {
    token = "PROFILE_RUNTIME_ABI_INCOMPATIBLE";
  } else if (!package.supports_target(manifest.target_triple)) {
    token = "PROFILE_TARGET_INCOMPATIBLE";
  } else {
    for (const auto& requirement : manifest.required_capabilities) {
      const auto* offer = package.find_offer(requirement.capability_id, requirement.version);
      if (offer == nullptr) {
        token = "PROFILE_CAPABILITY_UNSATISFIED";
        break;
      }
      if (offer->evidence_id != requirement.evidence_id) {
        token = "PROFILE_EVIDENCE_UNSATISFIED";
        break;
      }
    }
  }
  if (token.empty()) {
    return {};
  }
  manifest.disposition = ProfileAdmissionDisposition::kRejected;
  reject_profile_features(manifest);
  return make_unexpected(Error{ErrorCode::kNotSupported, std::string{token}});
}

}  // namespace

expected<void, Error> finalize_profile_manifest(const CompileOptions& options,
                                                ExecutionPlanState& state) {
  return finalize_profile_manifest(options, state, runtime_package_capability_view());
}

expected<void, Error> finalize_profile_manifest(
    const CompileOptions& options, ExecutionPlanState& state,
    const RuntimePackageCapabilityView& package_capabilities) {
  ProfileManifest& manifest = state.profile_manifest;
  add_profile_requirements(manifest);
  add_partition_and_lane_requirements(state);
  add_readiness_and_temporal_requirements(state);
  add_predicate_replay_requirements(state);
  add_physical_requirements(state);
  add_endpoint_requirements(options, state);
  append_complete_profile_scopes(state);
  auto canonical = canonicalize(manifest);
  if (!canonical) {
    return canonical;
  }
  return match_package_capabilities(manifest, package_capabilities);
}

}  // namespace holoscan::detail
