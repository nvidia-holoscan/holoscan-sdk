/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/tensor_storage_lowering.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compiler/detail/checked_arithmetic.hpp"
#include "compiler/execution_plan_identity.hpp"
#include "compiler/physical_plan_lowering_internal.hpp"
#include "compiler/pool_capacity_plan_lowering.hpp"
#include "compiler/snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kTensorWidening{"PHYSICAL_TENSOR_ALLOCATION_WIDENING"};
constexpr std::string_view kTensorAllocationBoundsInvalid{
    "PHYSICAL_TENSOR_ALLOCATION_BOUNDS_INVALID"};
constexpr std::string_view kSchemaTensorBufferCountExceeded{
    "CHANNEL_SCHEMA_TENSOR_BUFFER_COUNT_EXCEEDED"};
constexpr std::string_view kLoanEvidenceUnavailable{"GENERIC_LOAN_TRAIT_EVIDENCE_UNAVAILABLE"};

[[nodiscard]] bool allocation_tightens(const TensorPoolBounds& base,
                                       const TensorPoolBounds& candidate) noexcept {
  return candidate.max_rank != 0U && candidate.max_byte_span != 0U &&
         candidate.max_rank <= base.max_rank && candidate.max_byte_span <= base.max_byte_span;
}

[[nodiscard]] std::optional<GenericLoanTraitEvidence> generic_loan_trait_evidence(
    const PortPlanRecord& port) noexcept {
  if (!port.generic_loan_supported || port.generic_loan_slot_size_bytes == 0U) {
    return std::nullopt;
  }
  switch (port.payload_identity.kind) {
    case PayloadIdentityKind::kScalar:
      return GenericLoanTraitEvidence::kStaticallyBoundedScalar;
    case PayloadIdentityKind::kPod:
      return GenericLoanTraitEvidence::kNothrowDefaultConstructiblePOD;
    case PayloadIdentityKind::kSchema:
      return GenericLoanTraitEvidence::kNothrowDefaultConstructibleSchema;
    case PayloadIdentityKind::kTensor:
      break;
  }
  return std::nullopt;
}

[[nodiscard]] std::array<TensorClaimContributor, 5> tensor_claim_contributors(
    MemoryKind memory, std::uint32_t producer, std::uint32_t delivery,
    std::uint32_t transport_retention, std::uint32_t retention, std::uint32_t window,
    std::uint32_t retirement, bool has_cache, bool reuse, bool drain, bool copy, bool reexport) {
  std::vector<TensorSlotClaimReason> delivery_reasons;
  if (delivery != 0U) {
    delivery_reasons = {TensorSlotClaimReason::kAcceptedPublication,
                        TensorSlotClaimReason::kHolomqTransportOrFanout};
    if (transport_retention == 0U) {
      delivery_reasons.push_back(TensorSlotClaimReason::kSubscriberBorrowOrImport);
      if (memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged) {
        delivery_reasons.push_back(TensorSlotClaimReason::kConsumerCudaRelease);
      }
      if (reexport) {
        delivery_reasons.push_back(TensorSlotClaimReason::kRelayOrReexport);
      }
      delivery_reasons.push_back(TensorSlotClaimReason::kDlpackOrFrameworkBorrow);
    }
  }
  std::vector<TensorSlotClaimReason> transport_retention_reasons;
  if (transport_retention != 0U) {
    transport_retention_reasons.push_back(TensorSlotClaimReason::kSubscriberBorrowOrImport);
    if (memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged) {
      transport_retention_reasons.push_back(TensorSlotClaimReason::kConsumerCudaRelease);
    }
    if (reexport) {
      transport_retention_reasons.push_back(TensorSlotClaimReason::kRelayOrReexport);
    }
    transport_retention_reasons.push_back(TensorSlotClaimReason::kDlpackOrFrameworkBorrow);
  }
  std::vector<TensorSlotClaimReason> retention_reasons;
  if (has_cache) {
    retention_reasons.push_back(TensorSlotClaimReason::kTemporalCache);
  }
  if (window != 0U) {
    retention_reasons.push_back(TensorSlotClaimReason::kPendingOrSelectedWindow);
  }
  if (reuse) {
    retention_reasons.push_back(TensorSlotClaimReason::kReuseLast);
  }
  if (drain) {
    retention_reasons.push_back(TensorSlotClaimReason::kDrainAllActivation);
  }
  if (copy) {
    retention_reasons.push_back(TensorSlotClaimReason::kCopyOrRecorderMaterialization);
  }
  return {
      TensorClaimContributor{.name = "producer",
                             .maximum = producer,
                             .reasons = {TensorSlotClaimReason::kProducerLoan,
                                         TensorSlotClaimReason::kProducerWriter,
                                         TensorSlotClaimReason::kProducerReadiness}},
      TensorClaimContributor{
          .name = "delivery", .maximum = delivery, .reasons = std::move(delivery_reasons)},
      TensorClaimContributor{.name = "transport-retention",
                             .maximum = transport_retention,
                             .reasons = std::move(transport_retention_reasons)},
      TensorClaimContributor{
          .name = "retention", .maximum = retention, .reasons = std::move(retention_reasons)},
      TensorClaimContributor{.name = "retirement",
                             .maximum = retirement,
                             .reasons = {TensorSlotClaimReason::kCancellation,
                                         TensorSlotClaimReason::kQuarantine,
                                         TensorSlotClaimReason::kRetirementService}},
  };
}

}  // namespace

// Output storage admission is a compile-cold transaction across all Tensor allocation and
// capacity contributors; keeping it whole preserves fail-closed diagnostic precedence.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void lower_output_storage(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation,
    std::span<const VertexProcessPlacementPlanRecord> process_placements,
    std::span<const ExecutionPartitionRecord> execution_partitions, PhysicalTopologyFacts& topology,
    std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const BracketedWindowPlanRecord> bracketed_windows,
    std::span<const ResolvedQueueSizing> resolved_queue_sizings,
    std::span<const QueueSizingRecipePlanRecord> queue_sizing_recipes,
    bool force_holomq_process_local_inline) {
  const auto& delivery_by_output = topology.delivery_by_output;
  const auto& transport_retention_by_output = topology.transport_retention_by_output;
  const auto& transport_retention_edges = topology.transport_retention_edges;
  const auto& cache_by_output = topology.cache_by_output;
  const auto& window_by_output = topology.window_by_output;
  const auto& reuse_by_output = topology.reuse_by_output;
  const auto& drain_by_output = topology.drain_by_output;
  const auto& copy_by_output = topology.copy_by_output;
  const auto& reexport_by_output = topology.reexport_by_output;
  std::unordered_set<std::uint64_t> cross_process_tensor_sources;
  cross_process_tensor_sources.reserve(result.payload_edges.size());
  for (const auto& edge : result.payload_edges) {
    const bool tensor_backed = is_tensor_backed_payload(edge.payload_identity);
    if (!tensor_backed) {
      continue;
    }
    const auto* source = find_port(snapshot, GraphPortId{edge.source.value()});
    if (edge.locality == ChannelLocalityKind::kExternalExport) {
      if (source != nullptr && process_group_for(process_placements, source->owner)) {
        cross_process_tensor_sources.insert(edge.source.value());
      }
      continue;
    }
    if (!is_graph_internal_locality(edge.locality)) {
      continue;
    }
    const auto* target = find_port(snapshot, GraphPortId{edge.target.value()});
    if (source == nullptr || target == nullptr) {
      continue;
    }
    const auto source_group = process_group_for(process_placements, source->owner);
    const auto target_group = process_group_for(process_placements, target->owner);
    if (source_group && target_group && source_group != target_group) {
      cross_process_tensor_sources.insert(edge.source.value());
    }
  }

  struct PoolPlanProjection {
    std::uint32_t slot_count{};
    std::vector<PoolCapacityContributorRecord> predecessors;
    std::vector<PoolCapacityDependencyPlanRecord> dependencies;
    ProviderAllocationPlanRecord provider;
  };
  const auto project_output_pool =
      // Pool projection is one compile-cold transaction across all immutable contributor families.
      // NOLINTNEXTLINE(readability-function-cognitive-complexity)
      [&](const PortPlanRecord& port,
          std::string_view pool_id,
          MemoryKind memory,
          std::uint32_t history_depth,
          std::span<const TensorClaimContributor>
              aggregate_contributors,
          ProviderAllocationKind allocation_kind,
          std::optional<std::uint16_t>
              field_id,
          std::optional<std::uint16_t>
              tensor_index,
          std::size_t pending_dependency_count,
          std::size_t pending_provider_count) -> std::optional<PoolPlanProjection> {
    PoolPlanProjection projection;
    std::uint32_t next_group = 1U;
    const auto append_dependency = [&](PoolClaimOrigin origin,
                                       PoolCapacityDependencySource source,
                                       QueueSizingRecipeId recipe,
                                       std::string predecessor,
                                       std::vector<TensorSlotClaimReason>
                                           reasons) {
      projection.dependencies.push_back(PoolCapacityDependencyPlanRecord{
          .id = PoolCapacityDependencyId{static_cast<std::uint32_t>(
              result.pool_capacity_dependencies.size() + pending_dependency_count +
              projection.dependencies.size() + 1U)},
          .pool_id = std::string{pool_id},
          .origin = origin,
          .source = source,
          .queue_sizing_recipe = recipe,
          .predecessor_contributor_id = std::move(predecessor),
          .scenario = PoolCapacityScenarioId{1U},
          .alternative_group = PoolCapacityAlternativeGroupId{next_group++},
          .mapping_multiplicity = 1U,
          .reasons = std::move(reasons),
      });
    };
    const auto append_fixed = [&](std::string_view name,
                                  PoolClaimOrigin origin,
                                  std::uint32_t maximum,
                                  std::vector<TensorSlotClaimReason>
                                      reasons) {
      if (maximum == 0U || reasons.empty()) {
        return;
      }
      std::string contributor_id = std::string{pool_id} + "." + std::string{name};
      projection.predecessors.push_back(PoolCapacityContributorRecord{
          .pool_id = std::string{pool_id},
          .contributor_id = contributor_id,
          .maximum_distinct_slots = maximum,
          .reasons = std::move(reasons),
      });
      append_dependency(origin,
                        PoolCapacityDependencySource::kPredecessorContributor,
                        QueueSizingRecipeId{},
                        std::move(contributor_id),
                        {});
    };

    std::vector<const PayloadEdgePlanRecord*> output_edges;
    for (const auto& edge : result.payload_edges) {
      if (edge.source == port.id) {
        output_edges.push_back(&edge);
      }
    }
    const bool dynamic =
        !output_edges.empty() && std::ranges::all_of(output_edges, [&](const auto* edge) {
          const auto* sizing = find_resolved_queue_sizing(resolved_queue_sizings, edge->edge);
          return sizing != nullptr && sizing->recipe;
        });
    if (!dynamic) {
      for (const auto& contributor : aggregate_contributors) {
        append_fixed(contributor.name,
                     contributor.name == "producer"
                         ? PoolClaimOrigin::kProducerLoan
                         : (contributor.name == "retirement" ? PoolClaimOrigin::kRetirement
                                                             : PoolClaimOrigin::kProviderSpecific),
                     contributor.maximum,
                     contributor.reasons);
      }
    } else {
      const auto producer = std::ranges::find(
          aggregate_contributors, std::string_view{"producer"}, &TensorClaimContributor::name);
      const auto retirement = std::ranges::find(
          aggregate_contributors, std::string_view{"retirement"}, &TensorClaimContributor::name);
      if (producer == aggregate_contributors.end() || retirement == aggregate_contributors.end()) {
        return std::nullopt;
      }
      append_fixed(
          producer->name, PoolClaimOrigin::kProducerLoan, producer->maximum, producer->reasons);
      for (const auto* edge : output_edges) {
        const auto* sizing = find_resolved_queue_sizing(resolved_queue_sizings, edge->edge);
        if (sizing == nullptr || !sizing->recipe) {
          return std::nullopt;
        }
        const bool independent_transport = transport_retention_edges.contains(edge->edge.value);
        const bool reexport = edge->locality == ChannelLocalityKind::kExternalExport;
        std::vector<TensorSlotClaimReason> delivery_reasons{
            TensorSlotClaimReason::kAcceptedPublication,
            TensorSlotClaimReason::kHolomqTransportOrFanout,
        };
        if (!independent_transport) {
          delivery_reasons.push_back(TensorSlotClaimReason::kSubscriberBorrowOrImport);
          if (memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged) {
            delivery_reasons.push_back(TensorSlotClaimReason::kConsumerCudaRelease);
          }
          if (reexport) {
            delivery_reasons.push_back(TensorSlotClaimReason::kRelayOrReexport);
          }
          delivery_reasons.push_back(TensorSlotClaimReason::kDlpackOrFrameworkBorrow);
        }
        append_dependency(PoolClaimOrigin::kEdgeDelivery,
                          PoolCapacityDependencySource::kQueueSizingRecipe,
                          sizing->recipe,
                          {},
                          std::move(delivery_reasons));
        if (independent_transport) {
          std::vector<TensorSlotClaimReason> transport_reasons{
              TensorSlotClaimReason::kSubscriberBorrowOrImport};
          if (memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged) {
            transport_reasons.push_back(TensorSlotClaimReason::kConsumerCudaRelease);
          }
          if (reexport) {
            transport_reasons.push_back(TensorSlotClaimReason::kRelayOrReexport);
          }
          transport_reasons.push_back(TensorSlotClaimReason::kDlpackOrFrameworkBorrow);
          append_dependency(PoolClaimOrigin::kTransportRetention,
                            PoolCapacityDependencySource::kQueueSizingRecipe,
                            sizing->recipe,
                            {},
                            std::move(transport_reasons));
        }
        if (edge->cache_capacity != 0U) {
          std::vector<TensorSlotClaimReason> retention_reasons{
              TensorSlotClaimReason::kTemporalCache};
          const auto selection =
              std::ranges::find(input_selections, edge->target, &InputSelectionPlanRecord::input);
          const bool has_window = std::ranges::any_of(
              bracketed_windows, [&](const auto& window) { return window.input == edge->target; });
          if (has_window) {
            retention_reasons.push_back(TensorSlotClaimReason::kPendingOrSelectedWindow);
          }
          if (selection != input_selections.end() &&
              selection->missing == TemporalRecordMissingKind::kReuseLast) {
            retention_reasons.push_back(TensorSlotClaimReason::kReuseLast);
          }
          if (selection != input_selections.end() &&
              selection->selector == InputSelectorKind::kDrainAll) {
            retention_reasons.push_back(TensorSlotClaimReason::kDrainAllActivation);
          }
          const auto authored_edge = std::ranges::find(snapshot.edges, edge->edge, &EdgeRecord::id);
          if (authored_edge != snapshot.edges.end() && authored_edge->effective_tensor_payload &&
              authored_edge->effective_tensor_payload->interface_tensor_emit_policy !=
                  InterfaceTensorEmitPolicy::kRequireExportable) {
            retention_reasons.push_back(TensorSlotClaimReason::kCopyOrRecorderMaterialization);
          }
          append_dependency(PoolClaimOrigin::kTemporalRetention,
                            PoolCapacityDependencySource::kQueueSizingRecipe,
                            sizing->recipe,
                            {},
                            std::move(retention_reasons));
        }
      }
      if (history_depth != 0U) {
        append_fixed("history",
                     PoolClaimOrigin::kHistory,
                     history_depth,
                     {TensorSlotClaimReason::kTemporalCache});
      }
      append_fixed(
          retirement->name, PoolClaimOrigin::kRetirement, retirement->maximum, retirement->reasons);
    }

    auto evaluated = evaluate_pool_capacity_plan(
        pool_id, projection.dependencies, projection.predecessors, resolved_queue_sizings);
    if (!evaluated || evaluated->admitted_entry_ceiling == 0U) {
      return std::nullopt;
    }
    projection.slot_count = evaluated->admitted_entry_ceiling;
    const bool lifecycle_candidate =
        std::ranges::any_of(projection.dependencies, [&](const auto& dependency) {
          if (dependency.source != PoolCapacityDependencySource::kQueueSizingRecipe) {
            return false;
          }
          const auto recipe = std::ranges::find(queue_sizing_recipes,
                                                dependency.queue_sizing_recipe,
                                                &QueueSizingRecipePlanRecord::id);
          return recipe != queue_sizing_recipes.end() &&
                 recipe->geometry_mode == AllocationGeometryMode::kLifecycleCandidate;
        });
    projection.provider = ProviderAllocationPlanRecord{
        .id = ProviderAllocationPlanRecordId{static_cast<std::uint32_t>(
            result.provider_allocation_plans.size() + pending_provider_count + 1U)},
        .pool_id = std::string{pool_id},
        .first_sizing_dependency = projection.dependencies.front().id,
        .sizing_dependency_count = static_cast<std::uint32_t>(projection.dependencies.size()),
        .allocation_kind = allocation_kind,
        .geometry_mode = lifecycle_candidate ? AllocationGeometryMode::kLifecycleCandidate
                                             : AllocationGeometryMode::kCompileConcrete,
        .output = port.id,
        .tensor_field_id = field_id,
        .tensor_index = tensor_index,
        .complete_before = LifecycleStage::kAllocate,
        .disposition = PlanDisposition::kAccepted,
    };
    return projection;
  };

  for (const auto& port : result.ports) {
    if (port.direction != PortDirection::kOutput) {
      continue;
    }
    const auto index = static_cast<std::size_t>(port.id.value() - 1U);
    const bool direct_tensor = port.payload_identity.kind == PayloadIdentityKind::kTensor;
    const bool descriptor_tensor_schema =
        port.payload_identity.kind == PayloadIdentityKind::kSchema &&
        port.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    const auto& authored = snapshot.ports[index];
    const bool multi_descriptor_tensor_schema =
        descriptor_tensor_schema && authored.descriptor_tensor_field_count > 1U;
    const auto& setup = authored.setup.tensor_allocation;
    const auto& override = authored.graph_override.tensor_allocation;
    const auto* derivation =
        find_port_tensor_byte_span_derivation(tensor_byte_span_derivations, authored);
    const bool descriptor_tensor_loan = port.owner_kind == PortOwnerKind::kOperator &&
                                        descriptor_tensor_schema &&
                                        (multi_descriptor_tensor_schema || setup.has_value() ||
                                         override.has_value() || derivation != nullptr);
    const bool tensor_publication_authority = direct_tensor || descriptor_tensor_loan;
    const std::uint32_t retirement = std::max(1U, port.max_emits_per_compute);
    if (!tensor_publication_authority) {
      if (!port.effective_profile) {
        continue;
      }
      if (port.generic_loan_supported) {
        const auto evidence = generic_loan_trait_evidence(port);
        if (!evidence) {
          reject_physical_plan(
              result, kLoanEvidenceUnavailable, source_for(snapshot, port.owner, port.id));
          continue;
        }
        std::uint32_t slots = port.max_emits_per_compute;
        if (!checked_add(slots, delivery_by_output[index], slots) ||
            !checked_add(slots, cache_by_output[index], slots) ||
            !checked_add(slots, port.late_joiner_history_depth, slots) ||
            !checked_add(slots, retirement, slots)) {
          reject_physical_plan(result, kQueueInvalid, source_for(snapshot, port.owner, port.id));
        } else {
          std::uint64_t aggregate_bytes = 0U;
          if (!checked_multiply(slots, port.generic_loan_slot_size_bytes, aggregate_bytes)) {
            reject_physical_plan(result, kQueueInvalid, source_for(snapshot, port.owner, port.id));
            continue;
          }
          result.generic_output_loans.push_back(GenericOutputLoanPlanRecord{
              .output = port.id,
              .plan_generation = plan_generation,
              .run_generation = 0U,
              .pool_id = generic_output_pool_id(plan_generation, port.id),
              .trait_evidence = *evidence,
              .slot_size_bytes = port.generic_loan_slot_size_bytes,
              .aggregate_byte_capacity = aggregate_bytes,
              .slot_count = slots,
              .max_live_loans = port.max_emits_per_compute,
              .max_emits_per_compute = port.max_emits_per_compute,
              .retirement_capacity = retirement,
              .effective_profile = port.effective_profile,
              .disposition = PlanDisposition::kAccepted,
          });
        }
      }
    }
    result.publication_outcomes.push_back(PublicationOutcomePlanRecord{
        .output = port.id,
        .tensor = tensor_publication_authority,
        .pre_submit_authority = tensor_publication_authority
                                    ? PreSubmitAuthorityDisposition::kConsumedToRetirement
                                    : PreSubmitAuthorityDisposition::kRetainedSameActivation,
        .pre_submit_assigns_sequence = false,
        .post_submit_assigns_sequence = true,
        .post_submit_consumes_authority = true,
        .retain_claims_until_retirement = true,
        .runtime_retry_enabled = false,
    });

    if (!tensor_publication_authority) {
      continue;
    }
    if (multi_descriptor_tensor_schema) {
      if (!port.effective_profile) {
        continue;
      }
      const auto source = source_for(snapshot, port.owner, port.id);
      if (authored.setup.tensor_fields.size() != authored.descriptor_tensor_field_count) {
        reject_physical_plan(
            result,
            kSchemaTensorFieldUnbounded,
            source,
            port_pointer(port.id, "/tensor_fields"),
            "one bounded allocation contract for every descriptor Tensor field",
            "configured=" + std::to_string(authored.setup.tensor_fields.size()) +
                ",required=" + std::to_string(authored.descriptor_tensor_field_count));
        continue;
      }
      const auto process_group = process_group_for(process_placements, port.owner);
      if (!process_group) {
        reject_physical_plan(result,
                             kTensorProcessGroupUnresolved,
                             source,
                             port_pointer(port.id, "/owner_process_group"),
                             "one accepted process group",
                             "owner=" + std::to_string(port.owner.value));
        continue;
      }
      if (accepted_cover_count(execution_partitions, port.owner) != 1U) {
        reject_physical_plan(result, kResourceCoverInvalid, source);
        continue;
      }
      std::uint32_t retention = 0U;
      if (!checked_add(cache_by_output[index], port.late_joiner_history_depth, retention)) {
        reject_physical_plan(result, kTensorOverflow, source);
        continue;
      }

      bool fields_valid = true;
      std::uint32_t total_slots = 0U;
      std::vector<TensorFieldAllocationPlanRecord> field_allocations;
      std::vector<PoolCapacityContributorRecord> field_contributors;
      std::vector<PoolCapacityDependencyPlanRecord> field_dependencies;
      std::vector<ProviderAllocationPlanRecord> field_providers;
      field_allocations.reserve(authored.setup.tensor_fields.size());
      for (const auto* const field_ptr : canonical_tensor_fields(authored.setup.tensor_fields)) {
        const auto& field = *field_ptr;
        const auto* field_derivation =
            find_tensor_byte_span_derivation(tensor_byte_span_derivations, port.id, field.field_id);
        const auto field_bounds =
            field_derivation != nullptr
                ? std::optional<TensorPoolBounds>{TensorPoolBounds{
                      .max_rank = field.tensor_contract && field.tensor_contract->max_rank
                                      ? *field.tensor_contract->max_rank
                                      : 0U,
                      .max_byte_span =
                          static_cast<std::size_t>(field_derivation->resolved_max_byte_span),
                  }}
                : field.tensor_allocation;
        const auto field_pointer =
            port_pointer(port.id, "/tensor_fields/" + std::to_string(field.field_id));
        if (field.payload_options &&
            (field.payload_options->max_inline_bytes ||
             field.payload_options->interface_tensor_emit_policy ==
                 InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded)) {
          reject_physical_plan(result,
                               kInlineProviderUnsupported,
                               source,
                               field_pointer + "/payload_options",
                               "descriptor carriage for every multi-Tensor field",
                               "field_id=" + std::to_string(field.field_id) +
                                   ",reason=mixed_inline_descriptor_carriage");
          fields_valid = false;
          continue;
        }
        if (!field_bounds || !field.tensor_layout || !field.tensor_layout->memory_kind ||
            field_bounds->max_rank == 0U || field_bounds->max_byte_span == 0U ||
            (field.tensor_layout->rank && *field.tensor_layout->rank > field_bounds->max_rank)) {
          reject_physical_plan(result,
                               kSchemaTensorFieldUnbounded,
                               source,
                               field_pointer,
                               "layout and positive max_rank/max_byte_span allocation bounds",
                               "field_id=" + std::to_string(field.field_id));
          fields_valid = false;
          continue;
        }
        const MemoryKind memory = *field.tensor_layout->memory_kind;
        if (memory != MemoryKind::kHost && memory != MemoryKind::kPinnedHost &&
            memory != MemoryKind::kCudaDevice && memory != MemoryKind::kCudaManaged) {
          reject_physical_plan(result,
                               kSchemaTensorFieldUnsupportedMemory,
                               source,
                               field_pointer + "/tensor_layout/memory_kind",
                               "kHost|kPinnedHost|kCudaDevice|kCudaManaged",
                               "field_id=" + std::to_string(field.field_id) +
                                   ",memory=" + std::string(memory_kind_name(memory)));
          fields_valid = false;
          continue;
        }
        const bool cross_process_pinned = memory == MemoryKind::kPinnedHost &&
                                          cross_process_tensor_sources.contains(port.id.value());
        std::optional<DeviceId> device{DeviceId{}};
        if (memory == MemoryKind::kCudaDevice || memory == MemoryKind::kCudaManaged ||
            cross_process_pinned) {
          device = resolve_tensor_device(
              result, snapshot, topology, port.id, memory, source, cross_process_pinned);
        }
        if (!device) {
          fields_valid = false;
          continue;
        }
        const std::string pool_id =
            tensor_field_output_pool_id(plan_generation, port.id, field.field_id);
        const auto contributors = tensor_claim_contributors(
            memory,
            port.max_emits_per_compute,
            delivery_by_output[index],
            transport_retention_by_output[index],
            retention,
            window_by_output[index],
            retirement,
            cache_by_output[index] != 0U || port.late_joiner_history_depth != 0U,
            reuse_by_output[index],
            drain_by_output[index],
            copy_by_output[index],
            reexport_by_output[index]);
        auto projection = project_output_pool(port,
                                              pool_id,
                                              memory,
                                              port.late_joiner_history_depth,
                                              contributors,
                                              ProviderAllocationKind::kTensorFieldOutput,
                                              field.field_id,
                                              field.tensor_index,
                                              field_dependencies.size(),
                                              field_providers.size());
        if (projection &&
            projection->provider.geometry_mode == AllocationGeometryMode::kLifecycleCandidate) {
          reject_physical_plan(result,
                               kTensorProviderUnsupported,
                               source,
                               field_pointer + "/rate_geometry",
                               "compile-concrete descriptor Tensor field",
                               "lifecycle_candidate");
          fields_valid = false;
          continue;
        }
        const auto slot_count =
            projection ? std::optional<std::uint32_t>{projection->slot_count} : std::nullopt;
        std::uint32_t next_total = 0U;
        if (!slot_count || *slot_count == 0U ||
            !checked_add(total_slots, slot_count.value_or(0U), next_total)) {
          reject_physical_plan(result, kTensorOverflow, source, field_pointer);
          fields_valid = false;
          continue;
        }
        total_slots = next_total;
        std::uint64_t aggregate_bytes = 0U;
        if (!checked_multiply(*slot_count, field_bounds->max_byte_span, aggregate_bytes)) {
          reject_physical_plan(result, kTensorOverflow, source, field_pointer);
          fields_valid = false;
          continue;
        }
        field_contributors.insert(field_contributors.end(),
                                  std::make_move_iterator(projection->predecessors.begin()),
                                  std::make_move_iterator(projection->predecessors.end()));
        field_dependencies.insert(field_dependencies.end(),
                                  std::make_move_iterator(projection->dependencies.begin()),
                                  std::make_move_iterator(projection->dependencies.end()));
        field_providers.push_back(std::move(projection->provider));
        field_allocations.push_back(TensorFieldAllocationPlanRecord{
            .output = port.id,
            .field_id = field.field_id,
            .tensor_index = field.tensor_index,
            .plan_generation = plan_generation,
            .run_generation = 0U,
            .pool_id = pool_id,
            .allocator_root_id = tensor_allocator_root_id(plan_generation, *process_group, *device),
            .allocator_process_group = *process_group,
            .bounds = *field_bounds,
            .axes = field.tensor_contract ? field.tensor_contract->axes
                                          : std::vector<TensorAxisBounds>{},
            .layout = field.tensor_layout,
            .memory_kind = memory,
            .device = *device,
            .slot_count = *slot_count,
            .aggregate_byte_capacity = aggregate_bytes,
            .max_outstanding_loans = port.max_emits_per_compute,
            .max_emits_per_compute = port.max_emits_per_compute,
            .control_capacity = *slot_count,
            .readiness_capacity = *slot_count,
            .release_capacity = *slot_count,
            .retirement_capacity = *slot_count,
            .quarantine_capacity = *slot_count,
            .effective_profile = port.effective_profile,
            .disposition = PlanDisposition::kAccepted,
        });
      }
      if (!fields_valid) {
        continue;
      }
      if (total_slots > kMaximumDescriptorTensorFieldPoolSlots) {
        reject_physical_plan(
            result,
            kSchemaTensorBufferCountExceeded,
            source,
            port_pointer(port.id, "/tensor_fields"),
            "total_buffer_count<=" + std::to_string(kMaximumDescriptorTensorFieldPoolSlots),
            "total_buffer_count=" + std::to_string(total_slots));
        continue;
      }
      result.pool_capacity_contributors.insert(result.pool_capacity_contributors.end(),
                                               std::make_move_iterator(field_contributors.begin()),
                                               std::make_move_iterator(field_contributors.end()));
      result.pool_capacity_dependencies.insert(result.pool_capacity_dependencies.end(),
                                               std::make_move_iterator(field_dependencies.begin()),
                                               std::make_move_iterator(field_dependencies.end()));
      result.provider_allocation_plans.insert(result.provider_allocation_plans.end(),
                                              std::make_move_iterator(field_providers.begin()),
                                              std::make_move_iterator(field_providers.end()));
      result.tensor_field_allocations.insert(result.tensor_field_allocations.end(),
                                             std::make_move_iterator(field_allocations.begin()),
                                             std::make_move_iterator(field_allocations.end()));
      continue;
    }
    if (!setup && !override && derivation == nullptr) {
      continue;
    }
    if (!port.effective_profile) {
      // Profile admission already records why this owner has no effective profile. Preserve the
      // partial physical plan without duplicating that root-cause diagnostic or granting a pool.
      continue;
    }
    if ((derivation != nullptr && (setup || override)) ||
        (derivation == nullptr &&
         (!setup || (override && !allocation_tightens(*setup, *override))))) {
      reject_physical_plan(result, kTensorWidening, source_for(snapshot, port.owner, port.id));
      continue;
    }
    const TensorPoolBounds bounds = derivation != nullptr
                                        ? TensorPoolBounds{
                                              .max_rank = port.tensor_contract &&
                                                                  port.tensor_contract->max_rank
                                                              ? *port.tensor_contract->max_rank
                                                              : 0U,
                                              .max_byte_span = static_cast<std::size_t>(
                                                  derivation->resolved_max_byte_span),
                                          }
                                        : override.value_or(*setup);
    if (bounds.max_rank == 0U || bounds.max_byte_span == 0U) {
      reject_physical_plan(result,
                           kTensorAllocationBoundsInvalid,
                           source_for(snapshot, port.owner, port.id),
                           port_pointer(port.id, "/tensor_allocation"),
                           "max_rank>0,max_byte_span>0",
                           "max_rank=" + std::to_string(bounds.max_rank) +
                               ",max_byte_span=" + std::to_string(bounds.max_byte_span));
      continue;
    }
    const MemoryKind memory = port.tensor_layout && port.tensor_layout->memory_kind
                                  ? *port.tensor_layout->memory_kind
                                  : MemoryKind::kUnknown;
    if (memory == MemoryKind::kUnknown) {
      reject_physical_plan(result,
                           kTensorMemoryKindUnresolved,
                           source_for(snapshot, port.owner, port.id),
                           port_pointer(port.id, "/tensor_layout/memory_kind"),
                           "kHost|kPinnedHost|kCudaDevice|kCudaManaged",
                           std::string(memory_kind_name(memory)));
      continue;
    }
    const std::string pool_id = tensor_output_pool_id(plan_generation, port.id);
    const auto process_group = process_group_for(process_placements, port.owner);
    if (!process_group) {
      reject_physical_plan(result,
                           kTensorProcessGroupUnresolved,
                           source_for(snapshot, port.owner, port.id),
                           port_pointer(port.id, "/owner_process_group"),
                           "one accepted process group",
                           "owner=" + std::to_string(port.owner.value));
      continue;
    }
    const auto source = source_for(snapshot, port.owner, port.id);
    if (accepted_cover_count(execution_partitions, port.owner) != 1U) {
      reject_physical_plan(result, kResourceCoverInvalid, source);
      continue;
    }
    // HoloMQ's pinned-host generation protocol uses CUDA stream memory operations even though
    // public DLPack metadata has no physical device ordinal. Freeze the private synchronization
    // device only when this output actually crosses a process boundary.
    const bool cross_process_pinned =
        memory == MemoryKind::kPinnedHost && cross_process_tensor_sources.contains(port.id.value());
    const auto device = resolve_tensor_device(
        result, snapshot, topology, port.id, memory, source, cross_process_pinned);
    if (!device) {
      continue;
    }
    std::uint32_t retention = 0U;
    if (!checked_add(cache_by_output[index], port.late_joiner_history_depth, retention)) {
      reject_physical_plan(result, kTensorOverflow, source_for(snapshot, port.owner, port.id));
      continue;
    }
    const auto contributors = tensor_claim_contributors(
        memory,
        port.max_emits_per_compute,
        delivery_by_output[index],
        transport_retention_by_output[index],
        retention,
        window_by_output[index],
        retirement,
        cache_by_output[index] != 0U || port.late_joiner_history_depth != 0U,
        reuse_by_output[index],
        drain_by_output[index],
        copy_by_output[index],
        reexport_by_output[index]);
    auto projection = project_output_pool(port,
                                          pool_id,
                                          memory,
                                          port.late_joiner_history_depth,
                                          contributors,
                                          ProviderAllocationKind::kTensorOutput,
                                          std::nullopt,
                                          std::nullopt,
                                          0U,
                                          0U);
    if (!projection || projection->slot_count == 0U) {
      reject_physical_plan(result, kTensorOverflow, source_for(snapshot, port.owner, port.id));
      continue;
    }
    const bool lifecycle_candidate =
        projection->provider.geometry_mode == AllocationGeometryMode::kLifecycleCandidate;
    if (lifecycle_candidate) {
      const bool supported_direct_host =
          memory == MemoryKind::kHost &&
          std::ranges::all_of(result.payload_edges, [&](const auto& edge) {
            if (edge.source != port.id) {
              return true;
            }
            const auto* target = find_port(snapshot, GraphPortId{edge.target.value()});
            return edge.locality == ChannelLocalityKind::kProcessLocal &&
                   edge.queue_policy == QueuePolicy::kBlocking && target != nullptr &&
                   physical_process_local_delivery_kind(topology.holomq_activation_pairs,
                                                        edge,
                                                        authored,
                                                        *target,
                                                        force_holomq_process_local_inline) ==
                       ProcessLocalChannelDeliveryKind::kDirectRetained;
          });
      if (!supported_direct_host) {
        reject_physical_plan(result,
                             kTensorProviderUnsupported,
                             source,
                             port_pointer(port.id, "/rate_geometry"),
                             "host direct-retained lifecycle candidate",
                             std::string(memory_kind_name(memory)));
        continue;
      }
      projection->provider.tensor_template = TensorProviderTemplatePlanRecord{
          .allocator_root_id = tensor_allocator_root_id(plan_generation, *process_group, *device),
          .allocator_process_group = *process_group,
          .bounds = bounds,
          .axes =
              port.tensor_contract ? port.tensor_contract->axes : std::vector<TensorAxisBounds>{},
          .layout = port.tensor_layout,
          .memory_kind = memory,
          .device = *device,
          .max_outstanding_loans = port.max_emits_per_compute,
          .max_emits_per_compute = port.max_emits_per_compute,
          .effective_profile = port.effective_profile,
          .fallback_rejected = true,
      };
    }
    if (lifecycle_candidate) {
      result.pool_capacity_contributors.insert(
          result.pool_capacity_contributors.end(),
          std::make_move_iterator(projection->predecessors.begin()),
          std::make_move_iterator(projection->predecessors.end()));
      result.pool_capacity_dependencies.insert(
          result.pool_capacity_dependencies.end(),
          std::make_move_iterator(projection->dependencies.begin()),
          std::make_move_iterator(projection->dependencies.end()));
      result.provider_allocation_plans.push_back(std::move(projection->provider));
      continue;
    }
    const std::uint32_t slot_count = projection->slot_count;
    std::uint64_t aggregate_bytes = 0U;
    if (!checked_multiply(slot_count, bounds.max_byte_span, aggregate_bytes)) {
      reject_physical_plan(result, kTensorOverflow, source_for(snapshot, port.owner, port.id));
      continue;
    }
    result.pool_capacity_contributors.insert(
        result.pool_capacity_contributors.end(),
        std::make_move_iterator(projection->predecessors.begin()),
        std::make_move_iterator(projection->predecessors.end()));
    result.pool_capacity_dependencies.insert(
        result.pool_capacity_dependencies.end(),
        std::make_move_iterator(projection->dependencies.begin()),
        std::make_move_iterator(projection->dependencies.end()));
    result.provider_allocation_plans.push_back(std::move(projection->provider));
    result.tensor_allocations.push_back(TensorAllocationPlanRecord{
        .output = port.id,
        .plan_generation = plan_generation,
        .run_generation = 0U,
        .pool_id = pool_id,
        .allocator_root_id = tensor_allocator_root_id(plan_generation, *process_group, *device),
        .allocator_process_group = *process_group,
        .bounds = bounds,
        .axes = port.tensor_contract ? port.tensor_contract->axes : std::vector<TensorAxisBounds>{},
        .layout = port.tensor_layout,
        .memory_kind = memory,
        .device = *device,
        .slot_count = slot_count,
        .aggregate_byte_capacity = aggregate_bytes,
        .max_outstanding_loans = port.max_emits_per_compute,
        .max_emits_per_compute = port.max_emits_per_compute,
        .control_capacity = slot_count,
        .readiness_capacity = slot_count,
        .release_capacity = slot_count,
        .retirement_capacity = slot_count,
        .quarantine_capacity = slot_count,
        .effective_profile = port.effective_profile,
        .disposition = PlanDisposition::kAccepted,
    });
  }

  // A singular descriptor schema shares its compiler-owned Tensor allocation across direct and
  // HoloMQ delivery. Freeze those storage facts once in the service so stream selection and
  // Channel validation consume the same authority. Allocation-free pass-through services remain
  // deliberately empty.
  for (auto& service : result.process_local_channel_services) {
    const bool singular_descriptor =
        service.tensor_payload && service.tensor_fields.empty() &&
        service.payload_identity.kind == PayloadIdentityKind::kSchema &&
        service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
    if (!singular_descriptor) {
      continue;
    }
    const auto allocation = std::find_if(
        result.tensor_allocations.begin(), result.tensor_allocations.end(), [&](const auto& row) {
          return row.output == service.source;
        });
    if (allocation == result.tensor_allocations.end()) {
      continue;
    }
    service.tensor_max_byte_span = allocation->bounds.max_byte_span;
    service.tensor_pool_id = allocation->pool_id;
    service.tensor_memory_kind = allocation->memory_kind;
    service.tensor_device = allocation->device;
  }
}

}  // namespace holoscan::detail
