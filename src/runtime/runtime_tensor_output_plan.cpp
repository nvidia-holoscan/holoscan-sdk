/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_tensor_output_plan.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "../compiler/detail/checked_arithmetic.hpp"
#include "../compiler/execution_plan_contracts.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../core/pool_capacity_reducer.hpp"
#include "runtime_tensor_resource_materialization.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] bool canonical_claim_reasons(
    std::span<const TensorSlotClaimReason> reasons) noexcept {
  if (reasons.empty()) {
    return false;
  }
  for (std::size_t index = 0U; index < reasons.size(); ++index) {
    if (tensor_slot_claim_index(reasons[index]) >= kTensorSlotClaimReasonCount ||
        (index != 0U &&
         tensor_slot_claim_index(reasons[index - 1U]) >= tensor_slot_claim_index(reasons[index]))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool contains_claim_reason(std::span<const TensorSlotClaimReason> reasons,
                                         TensorSlotClaimReason expected) noexcept {
  return std::find(reasons.begin(), reasons.end(), expected) != reasons.end();
}

struct ClaimContributorValidation {
  TensorSlotClaimLedger::Admission admitted_claims{};
  std::uint32_t capacity{};
  bool producer_seen{};
  bool delivery_seen{};
  bool transport_retention_seen{};
  bool delivery_includes_transport_retention{};
  bool retention_seen{};
  bool retirement_seen{};
};

struct ClaimContributorIds {
  std::string producer;
  std::string delivery;
  std::string transport_retention;
  std::string retention;
  std::string retirement;
};

[[nodiscard]] bool tensor_template_matches_allocation(
    const TensorProviderTemplatePlanRecord& tensor_template,
    const TensorAllocationPlanRecord& allocation) noexcept {
  if (allocation.bounds.max_byte_span > std::numeric_limits<std::uint64_t>::max() /
                                            std::max<std::uint32_t>(allocation.slot_count, 1U)) {
    return false;
  }
  const auto aggregate = static_cast<std::uint64_t>(allocation.bounds.max_byte_span) *
                         static_cast<std::uint64_t>(allocation.slot_count);
  return allocation.allocator_root_id == tensor_template.allocator_root_id &&
         allocation.allocator_process_group == tensor_template.allocator_process_group &&
         allocation.bounds.max_rank == tensor_template.bounds.max_rank &&
         allocation.bounds.max_byte_span == tensor_template.bounds.max_byte_span &&
         allocation.axes == tensor_template.axes &&
         tensor_port_layout_exactly_equal(allocation.layout, tensor_template.layout) &&
         allocation.memory_kind == tensor_template.memory_kind &&
         allocation.device == tensor_template.device &&
         allocation.max_outstanding_loans == tensor_template.max_outstanding_loans &&
         allocation.max_emits_per_compute == tensor_template.max_emits_per_compute &&
         allocation.effective_profile == tensor_template.effective_profile &&
         allocation.fallback_rejected == tensor_template.fallback_rejected &&
         allocation.slot_count != 0U && allocation.aggregate_byte_capacity == aggregate &&
         allocation.control_capacity == allocation.slot_count &&
         allocation.readiness_capacity == allocation.slot_count &&
         allocation.release_capacity == allocation.slot_count &&
         allocation.retirement_capacity == allocation.slot_count &&
         allocation.quarantine_capacity == allocation.slot_count &&
         allocation.materialize_before == LifecycleStage::kArm &&
         allocation.disposition == PlanDisposition::kAccepted;
}

[[nodiscard]] bool valid_publication_errors(const PublicationOutcomePlanRecord& outcome) noexcept {
  return outcome.pre_submit_error == ErrorCode::kBackpressured &&
         outcome.post_submit_error == ErrorCode::kPublicationIndeterminate &&
         outcome.post_submit_partial_error == ErrorCode::kPublicationPartial;
}

[[nodiscard]] bool valid_publication_authority(
    const PublicationOutcomePlanRecord& outcome) noexcept {
  return outcome.pre_submit_authority == PreSubmitAuthorityDisposition::kConsumedToRetirement &&
         outcome.post_submit_consumes_authority && outcome.retain_claims_until_retirement;
}

[[nodiscard]] bool valid_publication_sequence(
    const PublicationOutcomePlanRecord& outcome) noexcept {
  return !outcome.pre_submit_assigns_sequence && outcome.post_submit_assigns_sequence &&
         !outcome.runtime_retry_enabled;
}

[[nodiscard]] expected<PublicationOutcomePlanRecord, Error> validate_publication_contract(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation) {
  const auto outcome = std::find_if(
      plan.publication_outcomes.begin(), plan.publication_outcomes.end(), [&](const auto& row) {
        return row.output == allocation.output;
      });
  const auto output_matches = [&](const auto& row) { return row.output == allocation.output; };
  if (outcome == plan.publication_outcomes.end() ||
      std::count_if(plan.publication_outcomes.begin(),
                    plan.publication_outcomes.end(),
                    output_matches) != 1 ||
      !outcome->tensor || !valid_publication_errors(*outcome) ||
      !valid_publication_authority(*outcome) || !valid_publication_sequence(*outcome)) {
    return make_unexpected(tensor_resource_error(
        "publication contract mismatch", "output", allocation.output.value()));
  }
  return *outcome;
}

[[nodiscard]] expected<void, Error> record_claim_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  if (contributor.maximum_distinct_slots == 0U || !canonical_claim_reasons(contributor.reasons) ||
      !checked_add(validation.capacity, contributor.maximum_distinct_slots, validation.capacity)) {
    return make_unexpected(
        tensor_resource_error("claim contributor capacity mismatch", "pool", allocation.pool_id));
  }
  for (const auto reason : contributor.reasons) {
    const std::size_t index = tensor_slot_claim_index(reason);
    if (validation.admitted_claims[index]) {
      return make_unexpected(
          tensor_resource_error("duplicate claim contributor reason", "pool", allocation.pool_id));
    }
    validation.admitted_claims[index] = true;
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_producer_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  constexpr std::array expected_reasons{TensorSlotClaimReason::kProducerLoan,
                                        TensorSlotClaimReason::kProducerWriter,
                                        TensorSlotClaimReason::kProducerReadiness};
  if (validation.producer_seen ||
      contributor.maximum_distinct_slots != allocation.max_outstanding_loans ||
      !std::equal(contributor.reasons.begin(),
                  contributor.reasons.end(),
                  expected_reasons.begin(),
                  expected_reasons.end())) {
    return make_unexpected(
        tensor_resource_error("producer claim contributor mismatch", "pool", allocation.pool_id));
  }
  validation.producer_seen = true;
  return {};
}

[[nodiscard]] expected<void, Error> validate_delivery_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  std::array<TensorSlotClaimReason, 6U> expected_reasons{};
  std::size_t expected_count = 0U;
  expected_reasons[expected_count++] = TensorSlotClaimReason::kAcceptedPublication;
  expected_reasons[expected_count++] = TensorSlotClaimReason::kHolomqTransportOrFanout;
  const bool includes_transport_retention =
      contains_claim_reason(contributor.reasons, TensorSlotClaimReason::kSubscriberBorrowOrImport);
  if (includes_transport_retention) {
    expected_reasons[expected_count++] = TensorSlotClaimReason::kSubscriberBorrowOrImport;
    if (allocation.memory_kind == MemoryKind::kCudaDevice ||
        allocation.memory_kind == MemoryKind::kCudaManaged) {
      expected_reasons[expected_count++] = TensorSlotClaimReason::kConsumerCudaRelease;
    }
    if (std::find(contributor.reasons.begin(),
                  contributor.reasons.end(),
                  TensorSlotClaimReason::kRelayOrReexport) != contributor.reasons.end()) {
      expected_reasons[expected_count++] = TensorSlotClaimReason::kRelayOrReexport;
    }
    expected_reasons[expected_count++] = TensorSlotClaimReason::kDlpackOrFrameworkBorrow;
  }
  if (validation.delivery_seen || contributor.reasons.size() != expected_count ||
      !std::equal(contributor.reasons.begin(),
                  contributor.reasons.end(),
                  expected_reasons.begin(),
                  expected_reasons.begin() + static_cast<std::ptrdiff_t>(expected_count))) {
    return make_unexpected(
        tensor_resource_error("delivery claim contributor mismatch", "pool", allocation.pool_id));
  }
  validation.delivery_seen = true;
  validation.delivery_includes_transport_retention = includes_transport_retention;
  return {};
}

[[nodiscard]] expected<void, Error> validate_transport_retention_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  std::array<TensorSlotClaimReason, 4U> expected_reasons{};
  std::size_t expected_count = 0U;
  expected_reasons[expected_count++] = TensorSlotClaimReason::kSubscriberBorrowOrImport;
  if (allocation.memory_kind == MemoryKind::kCudaDevice ||
      allocation.memory_kind == MemoryKind::kCudaManaged) {
    expected_reasons[expected_count++] = TensorSlotClaimReason::kConsumerCudaRelease;
  }
  if (std::find(contributor.reasons.begin(),
                contributor.reasons.end(),
                TensorSlotClaimReason::kRelayOrReexport) != contributor.reasons.end()) {
    expected_reasons[expected_count++] = TensorSlotClaimReason::kRelayOrReexport;
  }
  expected_reasons[expected_count++] = TensorSlotClaimReason::kDlpackOrFrameworkBorrow;
  if (validation.transport_retention_seen || contributor.reasons.size() != expected_count ||
      !std::equal(contributor.reasons.begin(),
                  contributor.reasons.end(),
                  expected_reasons.begin(),
                  expected_reasons.begin() + static_cast<std::ptrdiff_t>(expected_count))) {
    return make_unexpected(tensor_resource_error(
        "transport-retention claim contributor mismatch", "pool", allocation.pool_id));
  }
  validation.transport_retention_seen = true;
  return {};
}

[[nodiscard]] expected<void, Error> validate_retention_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  const bool has_unknown_reason =
      std::any_of(contributor.reasons.begin(), contributor.reasons.end(), [](const auto reason) {
        return reason != TensorSlotClaimReason::kTemporalCache &&
               reason != TensorSlotClaimReason::kPendingOrSelectedWindow &&
               reason != TensorSlotClaimReason::kReuseLast &&
               reason != TensorSlotClaimReason::kDrainAllActivation &&
               reason != TensorSlotClaimReason::kCopyOrRecorderMaterialization;
      });
  if (validation.retention_seen || has_unknown_reason) {
    return make_unexpected(
        tensor_resource_error("retention claim contributor mismatch", "pool", allocation.pool_id));
  }
  validation.retention_seen = true;
  return {};
}

[[nodiscard]] expected<void, Error> validate_retirement_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const TensorAllocationPlanRecord& allocation) {
  constexpr std::array expected_reasons{TensorSlotClaimReason::kCancellation,
                                        TensorSlotClaimReason::kQuarantine,
                                        TensorSlotClaimReason::kRetirementService};
  if (validation.retirement_seen || !std::equal(contributor.reasons.begin(),
                                                contributor.reasons.end(),
                                                expected_reasons.begin(),
                                                expected_reasons.end())) {
    return make_unexpected(
        tensor_resource_error("retirement claim contributor mismatch", "pool", allocation.pool_id));
  }
  validation.retirement_seen = true;
  return {};
}

[[nodiscard]] expected<void, Error> validate_claim_contributor(
    ClaimContributorValidation& validation, const PoolCapacityContributorRecord& contributor,
    const ClaimContributorIds& ids, const TensorAllocationPlanRecord& allocation) {
  if (contributor.contributor_id == ids.producer) {
    return validate_producer_contributor(validation, contributor, allocation);
  }
  if (contributor.contributor_id == ids.delivery) {
    return validate_delivery_contributor(validation, contributor, allocation);
  }
  if (contributor.contributor_id == ids.transport_retention) {
    return validate_transport_retention_contributor(validation, contributor, allocation);
  }
  if (contributor.contributor_id == ids.retention) {
    return validate_retention_contributor(validation, contributor, allocation);
  }
  if (contributor.contributor_id == ids.retirement) {
    return validate_retirement_contributor(validation, contributor, allocation);
  }
  return make_unexpected(
      tensor_resource_error("unknown claim contributor", "pool", allocation.pool_id));
}

[[nodiscard]] expected<std::uint32_t, Error> release_stream_capacity_impl(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation) noexcept {
  std::uint32_t result = 0U;
  for (const auto& service : plan.process_local_channel_services) {
    if (service.source != allocation.output || !service.tensor_payload) {
      continue;
    }
    if (service.max_subscribers > std::numeric_limits<std::uint32_t>::max() - result) {
      return make_unexpected(tensor_resource_error(
          "release-stream capacity overflow", "output", allocation.output.value()));
    }
    result += service.max_subscribers;
  }
  // Runtime installs one shared device stream before compute. Reserve one additional distinct
  // stream per downstream subscriber for an application-selected working stream.
  if (result == std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(tensor_resource_error(
        "release-stream capacity exhausted", "output", allocation.output.value()));
  }
  return result + 1U;
}

[[nodiscard]] bool requires_transport_retention(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation) noexcept {
  const bool process_local_transport = std::any_of(
      plan.process_local_channel_services.begin(),
      plan.process_local_channel_services.end(),
      [&](const auto& service) {
        return service.source == allocation.output && service.tensor_payload &&
               (service.delivery_kind == ProcessLocalChannelDeliveryKind::kHoloMQ ||
                (service.delivery_kind == ProcessLocalChannelDeliveryKind::kDirectRetained &&
                 service.queue_policy != QueuePolicy::kBlocking));
      });
  const bool cross_process_transport =
      std::any_of(plan.cross_process_channel_services.begin(),
                  plan.cross_process_channel_services.end(),
                  [&](const auto& service) {
                    return service.source == allocation.output && service.tensor_payload;
                  });
  const bool external_export_transport =
      std::any_of(plan.payload_edges.begin(), plan.payload_edges.end(), [&](const auto& edge) {
        return edge.source == allocation.output &&
               edge.locality == ChannelLocalityKind::kExternalExport;
      });
  return process_local_transport || cross_process_transport || external_export_transport;
}

[[nodiscard]] bool record_claim_reason_union(
    TensorSlotClaimLedger::Admission& admitted,
    std::span<const TensorSlotClaimReason> reasons) noexcept {
  if (!canonical_claim_reasons(reasons)) {
    return false;
  }
  for (const auto reason : reasons) {
    admitted[tensor_slot_claim_index(reason)] = true;
  }
  return true;
}

[[nodiscard]] bool valid_dependency_delivery_reasons(
    std::span<const TensorSlotClaimReason> reasons,
    const TensorAllocationPlanRecord& allocation) noexcept {
  std::array<TensorSlotClaimReason, 6U> expected{};
  std::size_t count = 0U;
  expected[count++] = TensorSlotClaimReason::kAcceptedPublication;
  expected[count++] = TensorSlotClaimReason::kHolomqTransportOrFanout;
  if (contains_claim_reason(reasons, TensorSlotClaimReason::kSubscriberBorrowOrImport)) {
    expected[count++] = TensorSlotClaimReason::kSubscriberBorrowOrImport;
    if (allocation.memory_kind == MemoryKind::kCudaDevice ||
        allocation.memory_kind == MemoryKind::kCudaManaged) {
      expected[count++] = TensorSlotClaimReason::kConsumerCudaRelease;
    }
    if (contains_claim_reason(reasons, TensorSlotClaimReason::kRelayOrReexport)) {
      expected[count++] = TensorSlotClaimReason::kRelayOrReexport;
    }
    expected[count++] = TensorSlotClaimReason::kDlpackOrFrameworkBorrow;
  }
  return reasons.size() == count &&
         std::equal(reasons.begin(), reasons.end(), expected.begin(), expected.begin() + count);
}

[[nodiscard]] bool valid_dependency_transport_reasons(
    std::span<const TensorSlotClaimReason> reasons,
    const TensorAllocationPlanRecord& allocation) noexcept {
  std::array<TensorSlotClaimReason, 4U> expected{};
  std::size_t count = 0U;
  expected[count++] = TensorSlotClaimReason::kSubscriberBorrowOrImport;
  if (allocation.memory_kind == MemoryKind::kCudaDevice ||
      allocation.memory_kind == MemoryKind::kCudaManaged) {
    expected[count++] = TensorSlotClaimReason::kConsumerCudaRelease;
  }
  if (contains_claim_reason(reasons, TensorSlotClaimReason::kRelayOrReexport)) {
    expected[count++] = TensorSlotClaimReason::kRelayOrReexport;
  }
  expected[count++] = TensorSlotClaimReason::kDlpackOrFrameworkBorrow;
  return reasons.size() == count &&
         std::equal(reasons.begin(), reasons.end(), expected.begin(), expected.begin() + count);
}

[[nodiscard]] bool valid_dependency_retention_reasons(
    std::span<const TensorSlotClaimReason> reasons) noexcept {
  return canonical_claim_reasons(reasons) && std::ranges::all_of(reasons, [](const auto reason) {
           return reason == TensorSlotClaimReason::kTemporalCache ||
                  reason == TensorSlotClaimReason::kPendingOrSelectedWindow ||
                  reason == TensorSlotClaimReason::kReuseLast ||
                  reason == TensorSlotClaimReason::kDrainAllActivation ||
                  reason == TensorSlotClaimReason::kCopyOrRecorderMaterialization;
         });
}

// This startup validator deliberately keeps the evidence-to-capacity proof in one pass so every
// rejected dependency observes the same accumulated claim state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<OutputPlanContract, Error> validate_dependency_output_plan_contract(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation,
    const PublicationOutcomePlanRecord& publication, const ProviderAllocationPlanRecord& provider,
    std::span<const RuntimeQueueDepthGeneration> queue_depths,
    AllocationGeometryMode expected_geometry,
    const RuntimePoolCapacityGeneration* pool_generation = nullptr) {
  if (provider.geometry_mode != expected_geometry ||
      provider.disposition != PlanDisposition::kAccepted || provider.reason ||
      provider.complete_before != LifecycleStage::kAllocate ||
      provider.pool_id != allocation.pool_id || provider.output != allocation.output ||
      (provider.allocation_kind != ProviderAllocationKind::kTensorOutput &&
       provider.allocation_kind != ProviderAllocationKind::kTensorFieldOutput) ||
      !provider.first_sizing_dependency || provider.sizing_dependency_count == 0U) {
    return make_unexpected(
        tensor_resource_error("pool dependency provider mismatch", "pool", allocation.pool_id));
  }
  const auto dependencies = provider_sizing_dependencies(plan.pool_capacity_dependencies, provider);
  if (!dependencies) {
    return make_unexpected(
        tensor_resource_error("pool dependency range mismatch", "pool", allocation.pool_id));
  }

  TensorSlotClaimLedger::Admission admitted_claims{};
  std::vector<PoolCapacityTerm> capacity_terms;
  capacity_terms.reserve(provider.sizing_dependency_count);
  bool producer_seen = false;
  bool delivery_seen = false;
  bool transport_seen = false;
  bool retirement_seen = false;
  std::size_t predecessor_count = 0U;
  for (const auto& dependency : *dependencies) {
    if (dependency.scenario.value == 0U || dependency.alternative_group.value == 0U ||
        dependency.mapping_multiplicity == 0U) {
      return make_unexpected(
          tensor_resource_error("pool dependency identity mismatch", "pool", allocation.pool_id));
    }

    std::uint32_t capacity = 0U;
    std::optional<std::uint32_t> capacity_ceiling;
    std::span<const TensorSlotClaimReason> reasons;
    if (dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe) {
      if (!dependency.queue_sizing_recipe || !dependency.predecessor_contributor_id.empty() ||
          !canonical_claim_reasons(dependency.reasons)) {
        return make_unexpected(
            tensor_resource_error("queue sizing dependency mismatch", "pool", allocation.pool_id));
      }
      const auto recipe = std::ranges::find(plan.queue_sizing_recipes,
                                            dependency.queue_sizing_recipe,
                                            &QueueSizingRecipePlanRecord::id);
      if (recipe == plan.queue_sizing_recipes.end() ||
          std::ranges::count(plan.queue_sizing_recipes,
                             dependency.queue_sizing_recipe,
                             &QueueSizingRecipePlanRecord::id) != 1 ||
          recipe->disposition != PlanDisposition::kAccepted ||
          recipe->geometry_mode != expected_geometry) {
        return make_unexpected(
            tensor_resource_error("queue sizing recipe mismatch", "pool", allocation.pool_id));
      }
      if (expected_geometry == AllocationGeometryMode::kLifecycleCandidate) {
        const auto generated =
            std::ranges::find(queue_depths, recipe->edge, &RuntimeQueueDepthGeneration::edge);
        if (generated == queue_depths.end() ||
            std::ranges::count(queue_depths, recipe->edge, &RuntimeQueueDepthGeneration::edge) !=
                1 ||
            generated->active_logical_depth == 0U) {
          return make_unexpected(tensor_resource_error(
              "queue sizing generation mismatch", "pool", allocation.pool_id));
        }
        capacity = generated->active_logical_depth;
        capacity_ceiling = generated->admitted_depth_ceiling;
      } else {
        const auto edge =
            std::ranges::find(plan.payload_edges, recipe->edge, &PayloadEdgePlanRecord::edge);
        if (edge == plan.payload_edges.end() ||
            std::ranges::count(plan.payload_edges, recipe->edge, &PayloadEdgePlanRecord::edge) !=
                1 ||
            edge->disposition != PlanDisposition::kAccepted || edge->queue_depth == 0U) {
          return make_unexpected(tensor_resource_error(
              "queue sizing generation mismatch", "pool", allocation.pool_id));
        }
        capacity = edge->queue_depth;
        capacity_ceiling = capacity;
      }
      reasons = dependency.reasons;
      switch (dependency.origin) {
        case PoolClaimOrigin::kEdgeDelivery:
          delivery_seen = true;
          if (!valid_dependency_delivery_reasons(reasons, allocation)) {
            return make_unexpected(tensor_resource_error(
                "delivery pool dependency mismatch", "pool", allocation.pool_id));
          }
          break;
        case PoolClaimOrigin::kTransportRetention:
          transport_seen = true;
          if (!valid_dependency_transport_reasons(reasons, allocation)) {
            return make_unexpected(tensor_resource_error(
                "transport pool dependency mismatch", "pool", allocation.pool_id));
          }
          break;
        case PoolClaimOrigin::kTemporalRetention:
          if (!valid_dependency_retention_reasons(reasons)) {
            return make_unexpected(tensor_resource_error(
                "retention pool dependency mismatch", "pool", allocation.pool_id));
          }
          break;
        default:
          return make_unexpected(
              tensor_resource_error("queue claim origin mismatch", "pool", allocation.pool_id));
      }
    } else if (dependency.source == PoolCapacityDependencySource::kPredecessorContributor) {
      if (dependency.queue_sizing_recipe || dependency.predecessor_contributor_id.empty() ||
          !dependency.reasons.empty()) {
        return make_unexpected(tensor_resource_error(
            "predecessor pool dependency mismatch", "pool", allocation.pool_id));
      }
      const auto contributor = std::ranges::find(plan.pool_capacity_contributors,
                                                 dependency.predecessor_contributor_id,
                                                 &PoolCapacityContributorRecord::contributor_id);
      if (contributor == plan.pool_capacity_contributors.end() ||
          std::ranges::count(plan.pool_capacity_contributors,
                             dependency.predecessor_contributor_id,
                             &PoolCapacityContributorRecord::contributor_id) != 1 ||
          contributor->pool_id != allocation.pool_id || contributor->maximum_distinct_slots == 0U ||
          !canonical_claim_reasons(contributor->reasons)) {
        return make_unexpected(tensor_resource_error(
            "predecessor claim evidence mismatch", "pool", allocation.pool_id));
      }
      ++predecessor_count;
      capacity = contributor->maximum_distinct_slots;
      capacity_ceiling = capacity;
      reasons = contributor->reasons;
      if (dependency.origin == PoolClaimOrigin::kProducerLoan) {
        constexpr std::array expected{TensorSlotClaimReason::kProducerLoan,
                                      TensorSlotClaimReason::kProducerWriter,
                                      TensorSlotClaimReason::kProducerReadiness};
        if (producer_seen || capacity != allocation.max_outstanding_loans ||
            !std::ranges::equal(reasons, expected)) {
          return make_unexpected(tensor_resource_error(
              "producer pool dependency mismatch", "pool", allocation.pool_id));
        }
        producer_seen = true;
      } else if (dependency.origin == PoolClaimOrigin::kRetirement) {
        constexpr std::array expected{TensorSlotClaimReason::kCancellation,
                                      TensorSlotClaimReason::kQuarantine,
                                      TensorSlotClaimReason::kRetirementService};
        if (retirement_seen || !std::ranges::equal(reasons, expected)) {
          return make_unexpected(tensor_resource_error(
              "retirement pool dependency mismatch", "pool", allocation.pool_id));
        }
        retirement_seen = true;
      } else if (dependency.origin != PoolClaimOrigin::kHistory) {
        return make_unexpected(
            tensor_resource_error("fixed claim origin mismatch", "pool", allocation.pool_id));
      }
    } else {
      return make_unexpected(
          tensor_resource_error("pool dependency source mismatch", "pool", allocation.pool_id));
    }

    if (!record_claim_reason_union(admitted_claims, reasons)) {
      return make_unexpected(
          tensor_resource_error("pool dependency capacity mismatch", "pool", allocation.pool_id));
    }
    capacity_terms.push_back(
        PoolCapacityTerm{.scenario = dependency.scenario,
                         .alternative_group = dependency.alternative_group,
                         .active_entries = capacity,
                         .admitted_entry_ceiling = capacity_ceiling,
                         .mapping_multiplicity = dependency.mapping_multiplicity});
  }

  const auto pool_contributor_count = std::ranges::count_if(
      plan.pool_capacity_contributors,
      [&](const auto& contributor) { return contributor.pool_id == allocation.pool_id; });
  const auto pool_dependency_count = std::ranges::count_if(
      plan.pool_capacity_dependencies,
      [&](const auto& dependency) { return dependency.pool_id == allocation.pool_id; });
  if (predecessor_count != pool_contributor_count ||
      provider.sizing_dependency_count != pool_dependency_count || !producer_seen ||
      !retirement_seen || !delivery_seen ||
      transport_seen != requires_transport_retention(plan, allocation)) {
    return make_unexpected(
        tensor_resource_error("incomplete pool dependency coverage", "pool", allocation.pool_id));
  }

  std::vector<MappedPoolCapacityTerm> mapped_terms(capacity_terms.size());
  const auto reduced = reduce_pool_capacity(capacity_terms, mapped_terms);
  if (!reduced) {
    return make_unexpected(
        tensor_resource_error(reduced.error() == PoolCapacityReductionError::kOverflow
                                  ? "pool dependency capacity overflow"
                                  : "pool dependency capacity mismatch",
                              "pool",
                              allocation.pool_id));
  }
  if (reduced->required_entries != allocation.slot_count) {
    return make_unexpected(
        tensor_resource_error("pool dependency allocation mismatch", "pool", allocation.pool_id));
  }
  if (pool_generation != nullptr &&
      (pool_generation->required_entries != reduced->required_entries ||
       pool_generation->admitted_entry_ceiling != reduced->admitted_entry_ceiling)) {
    return make_unexpected(
        tensor_resource_error("pool dependency generation mismatch", "pool", allocation.pool_id));
  }
  return OutputPlanContract{.publication = publication, .admitted_claims = admitted_claims};
}

// Provider geometry, generated capacity, and publication evidence form one startup contract;
// keeping their checks adjacent makes the accepted combination auditable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<OutputPlanContract, Error> validate_output_plan_contract_impl(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation,
    RuntimeRateMaterializationView materialization,
    const RuntimeTensorAllocationGeneration* generated_allocation) {
  auto publication = validate_publication_contract(plan, allocation);
  if (!publication) {
    return make_unexpected(std::move(publication.error()));
  }

  const auto provider = std::ranges::find(
      plan.provider_allocation_plans, allocation.pool_id, &ProviderAllocationPlanRecord::pool_id);
  if (provider != plan.provider_allocation_plans.end()) {
    if (std::ranges::count(plan.provider_allocation_plans,
                           allocation.pool_id,
                           &ProviderAllocationPlanRecord::pool_id) != 1) {
      return make_unexpected(
          tensor_resource_error("duplicate pool dependency provider", "pool", allocation.pool_id));
    }
    const AllocationGeometryMode expected_geometry =
        generated_allocation == nullptr ? AllocationGeometryMode::kCompileConcrete
                                        : AllocationGeometryMode::kLifecycleCandidate;
    const RuntimePoolCapacityGeneration* matched_pool_generation = nullptr;
    if (provider->geometry_mode != expected_geometry ||
        provider->disposition != PlanDisposition::kAccepted || provider->reason ||
        provider->pool_id != allocation.pool_id || provider->output != allocation.output ||
        !provider->first_sizing_dependency || provider->sizing_dependency_count == 0U) {
      return make_unexpected(
          tensor_resource_error("pool dependency provider mismatch", "pool", allocation.pool_id));
    }
    if (expected_geometry == AllocationGeometryMode::kLifecycleCandidate) {
      if (generated_allocation == nullptr) {
        return make_unexpected(tensor_resource_error(
            "lifecycle candidate allocation missing", "pool", allocation.pool_id));
      }
      if (!provider->tensor_template) {
        return make_unexpected(tensor_resource_error(
            "lifecycle candidate provider mismatch", "pool", allocation.pool_id));
      }
      const RuntimeTensorAllocationGeneration& generation = *generated_allocation;
      // The explicit provider check above proves this startup template is present.
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      const TensorProviderTemplatePlanRecord& tensor_template = provider->tensor_template.value();
      const auto pool_generation = std::ranges::find(
          materialization.pool_capacities, provider->id, &RuntimePoolCapacityGeneration::provider);
      if (generation.provider != provider->id || generation.run_generation == 0U ||
          generation.rate_generation == 0U ||
          provider->allocation_kind != ProviderAllocationKind::kTensorOutput || !provider->output ||
          provider->tensor_field_id || provider->tensor_index || provider->interface_topic ||
          provider->complete_before != LifecycleStage::kAllocate ||
          std::ranges::count(plan.tensor_allocations,
                             allocation.output,
                             &TensorAllocationPlanRecord::output) != 0 ||
          pool_generation == materialization.pool_capacities.end() ||
          std::ranges::count(materialization.pool_capacities,
                             provider->id,
                             &RuntimePoolCapacityGeneration::provider) != 1 ||
          pool_generation->required_entries != allocation.slot_count ||
          !tensor_template_matches_allocation(tensor_template, allocation)) {
        return make_unexpected(tensor_resource_error(
            "lifecycle candidate provider mismatch", "pool", allocation.pool_id));
      }
      matched_pool_generation = &*pool_generation;
    } else if (provider->tensor_template || generated_allocation != nullptr) {
      return make_unexpected(
          tensor_resource_error("pool dependency provider mismatch", "pool", allocation.pool_id));
    }
    const auto dependency_range =
        provider_sizing_dependencies(plan.pool_capacity_dependencies, *provider);
    if (!dependency_range) {
      return make_unexpected(
          tensor_resource_error("pool dependency range mismatch", "pool", allocation.pool_id));
    }
    const bool dynamic = std::ranges::any_of(*dependency_range, [](const auto& dependency) {
      return dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe;
    });
    if (dynamic) {
      return validate_dependency_output_plan_contract(plan,
                                                      allocation,
                                                      *publication,
                                                      *provider,
                                                      materialization.queue_depths,
                                                      expected_geometry,
                                                      matched_pool_generation);
    }
    std::size_t predecessor_count = 0U;
    for (const auto& dependency : *dependency_range) {
      const auto predecessor = std::ranges::find(plan.pool_capacity_contributors,
                                                 dependency.predecessor_contributor_id,
                                                 &PoolCapacityContributorRecord::contributor_id);
      if (dependency.source != PoolCapacityDependencySource::kPredecessorContributor ||
          dependency.queue_sizing_recipe || dependency.predecessor_contributor_id.empty() ||
          !dependency.reasons.empty() || predecessor == plan.pool_capacity_contributors.end() ||
          predecessor->pool_id != allocation.pool_id) {
        return make_unexpected(
            tensor_resource_error("static pool dependency mismatch", "pool", allocation.pool_id));
      }
      ++predecessor_count;
    }
    if (predecessor_count !=
        std::ranges::count_if(plan.pool_capacity_contributors, [&](const auto& contributor) {
          return contributor.pool_id == allocation.pool_id;
        })) {
      return make_unexpected(tensor_resource_error(
          "static pool dependency coverage mismatch", "pool", allocation.pool_id));
    }
  } else if (generated_allocation != nullptr) {
    return make_unexpected(
        tensor_resource_error("lifecycle candidate provider missing", "pool", allocation.pool_id));
  }

  ClaimContributorValidation validation;
  const ClaimContributorIds ids{.producer = allocation.pool_id + ".producer",
                                .delivery = allocation.pool_id + ".delivery",
                                .transport_retention = allocation.pool_id + ".transport-retention",
                                .retention = allocation.pool_id + ".retention",
                                .retirement = allocation.pool_id + ".retirement"};
  for (const auto& contributor : plan.pool_capacity_contributors) {
    if (contributor.pool_id != allocation.pool_id) {
      continue;
    }
    auto recorded = record_claim_contributor(validation, contributor, allocation);
    if (!recorded) {
      return make_unexpected(std::move(recorded.error()));
    }
    auto classified = validate_claim_contributor(validation, contributor, ids, allocation);
    if (!classified) {
      return make_unexpected(std::move(classified.error()));
    }
  }

  const bool transport_retention_required = requires_transport_retention(plan, allocation);
  const bool transport_retention_matches =
      transport_retention_required
          ? validation.delivery_seen && validation.transport_retention_seen &&
                !validation.delivery_includes_transport_retention
          : !validation.transport_retention_seen &&
                (!validation.delivery_seen || validation.delivery_includes_transport_retention);
  if (!validation.producer_seen || !validation.retirement_seen || !transport_retention_matches ||
      validation.capacity != allocation.slot_count) {
    return make_unexpected(
        tensor_resource_error("incomplete claim coverage", "pool", allocation.pool_id));
  }
  return OutputPlanContract{.publication = *publication,
                            .admitted_claims = validation.admitted_claims};
}

}  // namespace

expected<std::uint32_t, Error> release_stream_capacity(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation) noexcept {
  return release_stream_capacity_impl(plan, allocation);
}

expected<OutputPlanContract, Error> validate_output_plan_contract(
    const ExecutionPlanState& plan, const TensorAllocationPlanRecord& allocation,
    RuntimeRateMaterializationView materialization,
    const RuntimeTensorAllocationGeneration* generated_allocation) {
  return validate_output_plan_contract_impl(
      plan, allocation, materialization, generated_allocation);
}

}  // namespace holoscan::detail
