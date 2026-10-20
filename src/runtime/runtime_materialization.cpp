/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_materialization.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "../channel/endpoint_catalog_state.hpp"
#include "../compiler/execution_plan_contracts.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../core/source_id_internal.hpp"
#include "execution_lane.hpp"

namespace holoscan::detail {
namespace {

constexpr std::uint8_t kStopMask = 1U << static_cast<std::uint8_t>(LifecycleStage::kStop);
constexpr std::uint8_t kReleaseMask = 1U << static_cast<std::uint8_t>(LifecycleStage::kRelease);
constexpr std::uint8_t kStopReleaseMask = kStopMask | kReleaseMask;

[[nodiscard]] Error materialization_error() {
  return Error{ErrorCode::kFailure, "PROFILE_MATERIALIZATION_FAILED"};
}

[[nodiscard]] bool bounded_token(std::string_view value) noexcept {
  return !value.empty() && value.size() <= kRuntimeMaterializationMaximumTokenBytes;
}

[[nodiscard]] bool parse_unsigned_suffix(std::string_view value, std::string_view prefix,
                                         std::uint64_t& output) noexcept {
  if (!value.starts_with(prefix) || value.size() == prefix.size()) {
    return false;
  }
  const std::string_view digits = value.substr(prefix.size());
  if (digits.size() > 1U && digits.front() == '0') {
    return false;
  }
  const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), output);
  return error == std::errc() && end == digits.data() + digits.size();
}

template <typename Records, typename Predicate>
[[nodiscard]] bool contains_record(const Records& records, Predicate predicate) noexcept {
  return std::any_of(records.begin(), records.end(), std::move(predicate));
}

[[nodiscard]] bool valid_lifecycle_candidate_tensor_output_provider(
    const ExecutionPlanState& plan, const RuntimeMaterializationRecord& record) noexcept {
  if (record.complete_before_stage != LifecycleStage::kAllocate ||
      std::ranges::count(plan.provider_allocation_plans,
                         record.object_id,
                         &ProviderAllocationPlanRecord::pool_id) != 1) {
    return false;
  }
  const auto provider = std::ranges::find(
      plan.provider_allocation_plans, record.object_id, &ProviderAllocationPlanRecord::pool_id);
  if (provider == plan.provider_allocation_plans.end() ||
      provider->allocation_kind != ProviderAllocationKind::kTensorOutput ||
      provider->geometry_mode != AllocationGeometryMode::kLifecycleCandidate || !provider->output ||
      provider->tensor_field_id || provider->tensor_index || provider->interface_topic ||
      !provider->tensor_template || provider->complete_before != record.complete_before_stage ||
      provider->disposition != PlanDisposition::kAccepted || provider->reason ||
      !provider->first_sizing_dependency || provider->sizing_dependency_count == 0U) {
    return false;
  }
  const auto& tensor_template = *provider->tensor_template;
  if (tensor_template.allocator_root_id.empty() || !tensor_template.allocator_process_group ||
      tensor_template.bounds.max_rank == 0U || tensor_template.bounds.max_byte_span == 0U ||
      !tensor_template.layout || !tensor_template.layout->memory_kind ||
      *tensor_template.layout->memory_kind != MemoryKind::kHost ||
      tensor_template.memory_kind != MemoryKind::kHost || tensor_template.device.valid() ||
      tensor_template.max_emits_per_compute == 0U ||
      tensor_template.max_outstanding_loans != tensor_template.max_emits_per_compute ||
      !tensor_template.fallback_rejected) {
    return false;
  }
  bool found_direct_retained_service{};
  for (const auto& service : plan.process_local_channel_services) {
    if (service.source != *provider->output) {
      continue;
    }
    if (service.delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained ||
        service.cross_process_transport || service.queue_policy != QueuePolicy::kBlocking) {
      return false;
    }
    found_direct_retained_service = true;
  }
  if (!found_direct_retained_service ||
      std::ranges::any_of(plan.cross_process_channel_services, [&](const auto& service) {
        return service.source == *provider->output;
      })) {
    return false;
  }
  const auto dependencies =
      provider_sizing_dependencies(plan.pool_capacity_dependencies, *provider);
  if (!dependencies) {
    return false;
  }
  bool lifecycle_queue_dependency = false;
  for (const auto& dependency : *dependencies) {
    if (dependency.source != PoolCapacityDependencySource::kQueueSizingRecipe) {
      continue;
    }
    const auto recipe = std::ranges::find(plan.queue_sizing_recipes,
                                          dependency.queue_sizing_recipe,
                                          &QueueSizingRecipePlanRecord::id);
    if (recipe == plan.queue_sizing_recipes.end() ||
        recipe->geometry_mode != AllocationGeometryMode::kLifecycleCandidate ||
        !recipe->latest_binding_stage ||
        static_cast<std::uint8_t>(*recipe->latest_binding_stage) >
            static_cast<std::uint8_t>(LifecycleStage::kDiscover)) {
      return false;
    }
    lifecycle_queue_dependency = true;
  }
  return lifecycle_queue_dependency &&
         std::ranges::none_of(plan.tensor_allocations, [&](const auto& allocation) {
           return allocation.output == *provider->output && allocation.pool_id == provider->pool_id;
         });
}

[[nodiscard]] std::string tensor_transfer_object_id(const TensorTransferPlanRecord& transfer) {
  const std::string field_id =
      transfer.field_id ? fmt::format("{}", *transfer.field_id) : std::string{"none"};
  const std::string tensor_index =
      transfer.tensor_index ? fmt::format("{}", *transfer.tensor_index) : std::string{"none"};
  return fmt::format("tensor-topic:{}:field:{}:tensor:{}:source:{}:bytes:{}-{}",
                     transfer.topic.value,
                     field_id,
                     tensor_index,
                     static_cast<std::uint32_t>(transfer.source_class),
                     transfer.min_logical_bytes,
                     transfer.max_logical_bytes);
}

[[nodiscard]] bool canonical_obligation_order(const ProfileMaterializationObligationRecord& lhs,
                                              const ProfileMaterializationObligationRecord& rhs) {
  return std::tie(lhs.object_id, lhs.strategy_id, lhs.complete_before_stage) <
         std::tie(rhs.object_id, rhs.strategy_id, rhs.complete_before_stage);
}

[[nodiscard]] bool well_formed_obligations(
    const std::vector<ProfileMaterializationObligationRecord>& obligations) {
  if (obligations.size() > kRuntimeMaterializationMaximumObligations ||
      !std::is_sorted(obligations.begin(), obligations.end(), canonical_obligation_order)) {
    return false;
  }
  for (std::size_t index = 0U; index < obligations.size(); ++index) {
    const auto& obligation = obligations[index];
    if (!bounded_token(obligation.object_id) || !bounded_token(obligation.strategy_id) ||
        !bounded_token(obligation.evidence_id) ||
        !is_lifecycle_forward_stage(obligation.complete_before_stage) ||
        obligation.cleanup_stage_mask != kStopReleaseMask || obligation.max_cleanup_actions == 0U) {
      return false;
    }
    if (index != 0U && !canonical_obligation_order(obligations[index - 1U], obligation)) {
      return false;
    }
  }
  return true;
}

}  // namespace

RuntimeMaterializationState::RuntimeMaterializationState(
    const ExecutionPlanState& plan, const ExecutionLaneSet* lanes,
    std::vector<RuntimeMaterializationRecord> records) noexcept
    : plan_(&plan), lanes_(lanes), records_(std::move(records)) {
  cleanup_order_.reserve(records_.size());
}

expected<RuntimeMaterializationState, Error> RuntimeMaterializationState::create(
    const ExecutionPlanState& plan, const RuntimePackageCapabilityMatchReport& package,
    const ExecutionLaneSet* lanes) {
  const auto& obligations = plan.profile_manifest.materialization_obligations;
  if (!package.compatible() || !bounded_token(package.package_id) ||
      !bounded_token(package.package_manifest_digest) || !well_formed_obligations(obligations)) {
    return make_unexpected(materialization_error());
  }

  std::vector<RuntimeMaterializationRecord> records;
  records.reserve(obligations.size());
  for (const auto& obligation : obligations) {
    records.push_back(RuntimeMaterializationRecord{
        .context = package.context,
        .package_id = package.package_id,
        .package_manifest_digest = package.package_manifest_digest,
        .object_id = obligation.object_id,
        .strategy_id = obligation.strategy_id,
        .complete_before_stage = obligation.complete_before_stage,
        .cleanup_stage_mask = obligation.cleanup_stage_mask,
        .max_cleanup_actions = obligation.max_cleanup_actions,
        .evidence_id = obligation.evidence_id,
    });
  }
  return RuntimeMaterializationState{plan, lanes, std::move(records)};
}

// Keep the closed strategy-to-evidence map adjacent so omissions fail closed and remain reviewable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RuntimeMaterializationState::verify_resource(
    const RuntimeMaterializationRecord& record) const noexcept {
  if (plan_ == nullptr) {
    return false;
  }
  std::uint64_t numeric = 0U;
  if (record.strategy_id == "holoscan.runtime.execution_lane/v1") {
    return record.complete_before_stage == LifecycleStage::kStart &&
           record.evidence_id == "holoscan.compile.thread_placement.validation/v1" &&
           lanes_ != nullptr && lanes_->ready() &&
           parse_unsigned_suffix(record.object_id, "lane:", numeric) &&
           numeric < plan_->scheduler_plan.execution_lane_count &&
           contains_record(
               plan_->execution_lanes,
               [numeric](const auto& lane) { return lane.execution_lane == numeric; });
  }
  if (record.strategy_id == "holoscan.runtime.readiness_adapter/v1") {
    return record.complete_before_stage == LifecycleStage::kArm &&
           record.evidence_id == "holoscan.compile.readiness_plan/v1" &&
           parse_unsigned_suffix(record.object_id, "source:", numeric) && numeric != 0U &&
           contains_record(plan_->readiness, [numeric](const auto& source) {
             return source_id_value(source.source_id) == numeric && source.referenced &&
                    source.materialized;
           });
  }
  if (record.strategy_id == "holoscan.runtime.channel_service_attach/v1") {
    return record.complete_before_stage == LifecycleStage::kStart &&
           record.evidence_id == "holoscan.compile.channel_service_plan/v1" &&
           contains_record(plan_->channel_services, [&](const auto& service) {
             return service.service_id == record.object_id;
           });
  }
  if (record.strategy_id == "holoscan.runtime.generic_output_pool/v1") {
    return record.evidence_id == "holoscan.compile.generic_output_loan_plan/v1" &&
           contains_record(plan_->generic_output_loans, [&](const auto& loan) {
             return loan.pool_id == record.object_id &&
                    loan.materialize_before == record.complete_before_stage &&
                    loan.disposition == PlanDisposition::kAccepted;
           });
  }
  if (record.strategy_id == "holoscan.runtime.tensor_output_pool/v1") {
    if (record.evidence_id != "holoscan.compile.tensor_pool_plan/v1") {
      return false;
    }
    const bool compile_concrete = contains_record(plan_->tensor_allocations, [&](const auto& pool) {
      return pool.pool_id == record.object_id &&
             pool.materialize_before == record.complete_before_stage &&
             pool.disposition == PlanDisposition::kAccepted;
    });
    const bool lifecycle_candidate =
        valid_lifecycle_candidate_tensor_output_provider(*plan_, record);
    return compile_concrete != lifecycle_candidate;
  }
  if (record.strategy_id == "holoscan.runtime.cuda_compute_stream/v1") {
    return record.complete_before_stage == LifecycleStage::kArm &&
           record.evidence_id == "holoscan.compile.cuda_compute_stream_plan/v1" &&
           contains_record(plan_->cuda_compute_streams, [&](const auto& stream) {
             return stream.plan_generation == plan_->plan_generation &&
                    stream.run_generation == 0U && stream.process_group && stream.device.valid() &&
                    !stream.owner_id.empty() && stream.stream_id == record.object_id &&
                    stream.materialize_before == record.complete_before_stage;
           });
  }
  if (record.strategy_id == "holoscan.runtime.cuda_green_context_pool/v1") {
    return record.complete_before_stage == LifecycleStage::kArm &&
           record.evidence_id == "holoscan.compile.cuda_green_context_plan/v1" &&
           contains_record(plan_->cuda_green_context_pools, [&](const auto& pool) {
             return pool.plan_generation == plan_->plan_generation && pool.run_generation == 0U &&
                    pool.process_group && pool.device.valid() && pool.pool_id == record.object_id &&
                    pool.materialize_before == record.complete_before_stage;
           });
  }
  if (record.strategy_id == "holoscan.runtime.cuda_green_context/v1") {
    return record.complete_before_stage == LifecycleStage::kArm &&
           record.evidence_id == "holoscan.compile.cuda_green_context_plan/v1" &&
           contains_record(plan_->cuda_green_context_resources, [&](const auto& resource) {
             return resource.plan_generation == plan_->plan_generation &&
                    resource.run_generation == 0U && resource.process_group &&
                    resource.device.valid() && resource.green_context_id == record.object_id;
           });
  }
  if (record.strategy_id == "holoscan.runtime.tensor_interface_pool/v1") {
    return record.complete_before_stage == LifecycleStage::kArm &&
           record.evidence_id == "holoscan.compile.tensor_pool_plan/v1" &&
           contains_record(plan_->tensor_interface_pools,
                           [&](const auto& pool) { return pool.pool_id == record.object_id; });
  }
  if (record.strategy_id == "holoscan.runtime.tensor_inline_record_bucket/v1") {
    return record.complete_before_stage == LifecycleStage::kArm &&
           contains_record(plan_->tensor_inline_records, [&](const auto& bucket) {
             return bucket.fixed_record_bucket_id == record.object_id &&
                    bucket.capability_evidence_id == record.evidence_id;
           });
  }
  if (record.strategy_id == "holoscan.runtime.tensor_inline_record/v1") {
    return record.complete_before_stage == LifecycleStage::kArm &&
           contains_record(plan_->tensor_transfer_plans, [&](const auto& transfer) {
             return transfer.disposition == PlanDisposition::kAccepted && transfer.realization &&
                    transfer.requires_inline_record && transfer.evidence_id == record.evidence_id &&
                    tensor_transfer_object_id(transfer) == record.object_id;
           });
  }
  if (record.strategy_id == "holoscan.runtime.tensor_exportable_copy_pool/v1") {
    return record.complete_before_stage == LifecycleStage::kArm &&
           contains_record(plan_->tensor_transfer_plans, [&](const auto& transfer) {
             return transfer.disposition == PlanDisposition::kAccepted && transfer.realization &&
                    transfer.requires_exportable_copy_pool &&
                    transfer.evidence_id == record.evidence_id &&
                    tensor_transfer_object_id(transfer) == record.object_id;
           });
  }
  if (record.strategy_id == "holoscan.runtime.endpoint_catalog_attach/v1") {
    const auto* catalog = ChannelEndpointCatalogAccess::state(plan_->endpoint_catalog);
    return record.complete_before_stage == LifecycleStage::kStart &&
           record.evidence_id == "holoscan.compile.endpoint_catalog/v1" &&
           record.object_id == "endpoint_catalog" && catalog != nullptr &&
           !catalog->endpoints.empty();
  }
  return false;
}

LifecycleStatus RuntimeMaterializationState::transition(std::size_t index, LifecycleStage stage,
                                                        LifecycleTransitionPhase phase) noexcept {
  if (index >= records_.size()) {
    failed_ = true;
    return LifecycleStatus::kFatalFailure;
  }
  auto& record = records_[index];
  if (stage == record.complete_before_stage) {
    if (phase != LifecycleTransitionPhase::kStartupForward ||
        record.result != RuntimeMaterializationResult::kPending || !verify_resource(record)) {
      record.result = RuntimeMaterializationResult::kFailed;
      failed_ = true;
      return LifecycleStatus::kFatalFailure;
    }
    record.result = RuntimeMaterializationResult::kCompleted;
    record.cleanup_required = true;
    return LifecycleStatus::kOk;
  }

  if (stage == LifecycleStage::kStop) {
    return LifecycleStatus::kSkipped;
  }
  if (stage == LifecycleStage::kRelease && (phase == LifecycleTransitionPhase::kForwardRollback ||
                                            phase == LifecycleTransitionPhase::kShutdownTeardown)) {
    if (!record.cleanup_required) {
      return LifecycleStatus::kSkipped;
    }
    if (record.result != RuntimeMaterializationResult::kCompleted ||
        record.cleanup_actions >= record.max_cleanup_actions) {
      record.result = RuntimeMaterializationResult::kFailed;
      failed_ = true;
      return LifecycleStatus::kFatalFailure;
    }
    ++record.cleanup_actions;
    record.result = RuntimeMaterializationResult::kCleaned;
    record.cleanup_required = false;
    cleanup_order_.push_back(index);
    return LifecycleStatus::kOk;
  }
  return LifecycleStatus::kSkipped;
}

void RuntimeMaterializationState::cleanup_remaining() noexcept {
  for (std::size_t index = records_.size(); index-- > 0U;) {
    auto& record = records_[index];
    if (record.result != RuntimeMaterializationResult::kCompleted) {
      continue;
    }
    if (record.cleanup_actions >= record.max_cleanup_actions) {
      record.result = RuntimeMaterializationResult::kFailed;
      failed_ = true;
      continue;
    }
    ++record.cleanup_actions;
    record.result = RuntimeMaterializationResult::kCleaned;
    record.cleanup_required = false;
    cleanup_order_.push_back(index);
  }
}

}  // namespace holoscan::detail
