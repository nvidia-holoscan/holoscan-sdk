/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/execution_plan.hpp>

#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "../compiler/process_placement_internal.hpp"
#include "../cuda_execution/cuda_green_context_driver.hpp"

namespace holoscan {
namespace {

const ProfileManifest kEmptyProfileManifest{};
const PartitioningPlanRecord kEmptyPartitioningPlan{};
const ChannelEndpointCatalog kEmptyEndpointCatalog{};

using LifecycleCheckpointRecords = std::span<const LifecycleCheckpointPlanRecord>;
using LifecycleDependencyRecords = std::span<const LifecycleDependencyPlanRecord>;
using LifecyclePermissionRecords = std::span<const LifecycleRestartPermissionPlanRecord>;
using LifecycleTargetRecords = std::span<const LifecycleRestartTargetPlanRecord>;
using LifecycleClosureRecords = std::span<const LifecycleRestartClosurePlanRecord>;
using LifecycleMutableFieldRecords = std::span<const LifecycleMutableFieldPlanRecord>;
using LifecycleCapacity = std::optional<LifecycleTransactionCapacityPlanRecord>;
using OperatorParameterRecords = std::span<const OperatorParameterPlanRecord>;
using ParameterFieldRecords = std::span<const ParameterFieldPlanRecord>;
using ParameterDerivedTermRecords = std::span<const ParameterDerivedTermPlanRecord>;
using ParameterCapacity = std::optional<ParameterRuntimeCapacityPlanRecord>;

template <typename T>
[[nodiscard]] std::span<const T> plan_span(
    const std::shared_ptr<const detail::ExecutionPlanState>& state,
    const std::vector<T> detail::ExecutionPlanState::* field) noexcept {
  return state ? std::span<const T>{state.get()->*field} : std::span<const T>{};
}

}  // namespace

std::string_view tensor_slot_claim_reason_name(TensorSlotClaimReason reason) noexcept {
  switch (reason) {
    case TensorSlotClaimReason::kProducerLoan:
      return "producer_loan";
    case TensorSlotClaimReason::kProducerWriter:
      return "producer_writer";
    case TensorSlotClaimReason::kProducerReadiness:
      return "producer_readiness";
    case TensorSlotClaimReason::kAcceptedPublication:
      return "accepted_publication";
    case TensorSlotClaimReason::kHolomqTransportOrFanout:
      return "holomq_transport_or_fanout";
    case TensorSlotClaimReason::kSubscriberBorrowOrImport:
      return "subscriber_borrow_or_import";
    case TensorSlotClaimReason::kConsumerCudaRelease:
      return "consumer_cuda_release";
    case TensorSlotClaimReason::kRelayOrReexport:
      return "relay_or_reexport";
    case TensorSlotClaimReason::kTemporalCache:
      return "temporal_cache";
    case TensorSlotClaimReason::kPendingOrSelectedWindow:
      return "pending_or_selected_window";
    case TensorSlotClaimReason::kReuseLast:
      return "reuse_last";
    case TensorSlotClaimReason::kDrainAllActivation:
      return "drain_all_activation";
    case TensorSlotClaimReason::kDlpackOrFrameworkBorrow:
      return "dlpack_or_framework_borrow";
    case TensorSlotClaimReason::kCopyOrRecorderMaterialization:
      return "copy_or_recorder_materialization";
    case TensorSlotClaimReason::kCancellation:
      return "cancellation";
    case TensorSlotClaimReason::kQuarantine:
      return "quarantine";
    case TensorSlotClaimReason::kRetirementService:
      return "retirement_service";
  }
  return {};
}

DeploymentBinding& DeploymentBinding::bind_thread_placement(std::string_view partition_name,
                                                            ThreadPlacement placement) {
  thread_placements_.push_back(PartitionThreadPlacementBinding{
      .partition_name = std::string(partition_name),
      .placement = std::move(placement),
  });
  return *this;
}

DeploymentBinding& DeploymentBinding::bind_cuda_green_context(std::string_view partition_name,
                                                              CudaGreenContextPlacement placement) {
  cuda_green_contexts_.push_back(PartitionCudaGreenContextBinding{
      .partition_name = std::string(partition_name),
      .placement = placement,
  });
  return *this;
}

expected<CudaGreenContextCapability, Error> query_cuda_green_context_capability(DeviceId device) {
  if (!device.valid()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "CUDA Green Context capability query requires an explicit nonnegative device"});
  }
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
  auto driver = cuda_execution_detail::cuda_green_context_driver();
  if (!driver) {
    return forward_error(driver);
  }
  auto capability =
      cuda_execution_detail::query_cuda_green_context_device_capability(**driver, device.value);
  if (!capability) {
    return forward_error(capability);
  }
  return CudaGreenContextCapability{
      .device = device,
      .deployment_device_id = capability->deployment_device_id,
      .total_sm_count = capability->device_sm_count,
      .min_sm_partition_size = capability->min_sm_partition_size,
      .sm_coscheduled_alignment = capability->sm_coscheduled_alignment,
      .structured_sm_quantum = capability->structured_sm_quantum,
      .max_structured_context_count = capability->max_structured_context_count,
  };
#else
  return make_unexpected(
      Error{ErrorCode::kFailure,
            "CUDA Green Context capability query requires a CUDA 12.4 or newer toolkit"});
#endif
}

DeploymentBinding& DeploymentBinding::allow_generated_endpoint_name(GeneratedEndpointNamePin pin) {
  generated_endpoint_names_.push_back(GeneratedEndpointNamePinRecord{
      .final_name = std::string(pin.final_name),
      .algorithm = std::string(pin.algorithm),
      .version = pin.version,
  });
  return *this;
}

DeploymentBinding& DeploymentBinding::bind_tensor_output_device(
    TensorOutputDevicePlacement placement) {
  tensor_output_devices_.push_back(TensorOutputDevicePlacementRecord{
      .operator_path = std::string(placement.operator_path),
      .output_port = std::string(placement.output_port),
      .device = placement.device,
  });
  return *this;
}

std::string_view ExecutionPlan::json() const noexcept {
  return state_ ? std::string_view{state_->json} : std::string_view{};
}

GraphGeneration ExecutionGraphView::generation() const noexcept {
  return view_.generation();
}

GraphAuthoringIdentity ExecutionGraphView::identity() const noexcept {
  return view_.identity();
}

std::span<const GraphRecord> ExecutionGraphView::graphs() const noexcept {
  return view_.graphs();
}

std::span<const VertexRecord> ExecutionGraphView::vertices() const noexcept {
  return view_.vertices();
}

std::span<const PortRecord> ExecutionGraphView::ports() const noexcept {
  return view_.ports();
}

std::span<const EdgeRecord> ExecutionGraphView::edges() const noexcept {
  return view_.edges();
}

std::span<const InterfaceBindingRecord> ExecutionGraphView::interface_bindings() const noexcept {
  return view_.interface_bindings();
}

std::span<const ExternalTopicRecord> ExecutionGraphView::external_topics() const noexcept {
  return view_.external_topics();
}

std::span<const ReadinessSourceDeclarationRecord> ExecutionGraphView::readiness_sources()
    const noexcept {
  return view_.readiness_sources();
}

std::span<const RegionRecord> ExecutionGraphView::regions() const noexcept {
  return view_.regions();
}

std::span<const ClockDeclarationRecord> ExecutionGraphView::clocks() const noexcept {
  return view_.clocks();
}

std::span<const ClockImportAuthoringRecord> ExecutionGraphView::clock_imports() const noexcept {
  return view_.clock_imports();
}

std::span<const EmbeddedClockBindingAuthoringRecord> ExecutionGraphView::clock_bindings()
    const noexcept {
  return view_.clock_bindings();
}

std::span<const ClockConversionAuthoringRecord> ExecutionGraphView::clock_conversions()
    const noexcept {
  return view_.clock_conversions();
}

std::span<const LifecycleDependencyAuthoringRecord> ExecutionGraphView::lifecycle_dependencies()
    const noexcept {
  return view_.lifecycle_dependencies();
}

std::span<const LifecycleRestartPermissionAuthoringRecord>
ExecutionGraphView::lifecycle_restart_permissions() const noexcept {
  return view_.lifecycle_restart_permissions();
}

ExecutionGraphView ExecutionPlan::graph() const noexcept {
  return state_ ? ExecutionGraphView{state_->graph} : ExecutionGraphView{};
}

GraphAuthoringIdentity ExecutionPlan::graph_identity() const noexcept {
  return state_ ? state_->graph_identity : GraphAuthoringIdentity{};
}

GraphGeneration ExecutionPlan::generation() const noexcept {
  return graph_identity().version;
}

GraphAuthoringHash ExecutionPlan::graph_hash() const noexcept {
  return graph_identity().hash;
}

PlanDigest ExecutionPlan::digest() const noexcept {
  return state_ ? state_->digest : PlanDigest{};
}

bool ExecutionPlan::ok() const noexcept {
  return state_ && state_->admitted &&
         std::none_of(state_->diagnostics.begin(),
                      state_->diagnostics.end(),
                      [](const Diagnostic& diagnostic) {
                        return diagnostic.severity == DiagnosticSeverity::kError;
                      });
}

std::span<const Diagnostic> ExecutionPlan::diagnostics() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::diagnostics);
}

void ExecutionPlan::require_no_errors() const {
  if (!state_) {
    throw CompileError{"EXECUTION_PLAN_EMPTY"};
  }
  if (!ok()) {
    throw CompileError{"EXECUTION_PLAN_REJECTED: " + format_diagnostics(diagnostics(), graph())};
  }
}

const ProfileManifest& ExecutionPlan::profile_manifest() const noexcept {
  return state_ ? state_->profile_manifest : kEmptyProfileManifest;
}

const PartitioningPlanRecord& ExecutionPlan::partitioning_plan() const noexcept {
  return state_ ? state_->partitioning_plan : kEmptyPartitioningPlan;
}

ExecutionPartitionId ExecutionPartitionView::id() const noexcept {
  return state_->execution_partitions[index_].id;
}

ExecutionPartitionOrigin ExecutionPartitionView::origin() const noexcept {
  return state_->execution_partitions[index_].authored_name ? ExecutionPartitionOrigin::kAuthored
                                                            : ExecutionPartitionOrigin::kAutomatic;
}

std::optional<std::string_view> ExecutionPartitionView::authored_name() const noexcept {
  const auto& name = state_->execution_partitions[index_].authored_name;
  return name ? std::optional<std::string_view>{*name} : std::nullopt;
}

std::span<const VertexId> ExecutionPartitionView::members() const noexcept {
  return state_->execution_partitions[index_].members;
}

std::span<const std::string> ExecutionPartitionView::operator_paths() const noexcept {
  return state_->execution_partitions[index_].operator_paths;
}

std::span<const GraphId> ExecutionPartitionView::graph_members() const noexcept {
  return state_->execution_partitions[index_].graph_members;
}

std::uint32_t ExecutionPartitionView::execution_lane() const noexcept {
  return state_->execution_partitions[index_].execution_lane;
}

PartitioningMode ExecutionPartitionsView::mode() const noexcept {
  return state_ ? state_->partitioning_plan.mode : PartitioningMode::kAutomatic;
}

std::uint32_t ExecutionPartitionsView::authored_count() const noexcept {
  return state_ ? state_->partitioning_plan.authored_partition_count : 0U;
}

std::uint32_t ExecutionPartitionsView::automatic_count() const noexcept {
  return state_ ? state_->partitioning_plan.automatic_partition_count : 0U;
}

std::size_t ExecutionPartitionsView::size() const noexcept {
  return state_ ? state_->execution_partitions.size() : 0U;
}

bool ExecutionPartitionsView::empty() const noexcept {
  return size() == 0U;
}

bool ExecutionPartitionsView::usable() const noexcept {
  return state_ && state_->admitted &&
         state_->partitioning_plan.disposition == PlanDisposition::kAccepted;
}

expected<ExecutionPartitionView, Error> ExecutionPartitionsView::lookup(
    ExecutionPartitionId id) const noexcept {
  if (!usable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!id) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (id.value > state_->execution_partition_index_by_id.size()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::uint32_t table_index =
      state_->execution_partition_index_by_id[static_cast<std::size_t>(id.value - 1U)];
  if (table_index == 0U || table_index > state_->execution_partitions.size()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return ExecutionPartitionView{state_, static_cast<std::size_t>(table_index - 1U)};
}

namespace {

[[nodiscard]] expected<std::size_t, Error> lookup_partition_string_index(
    const std::shared_ptr<const detail::ExecutionPlanState>& state,
    std::span<const detail::ExecutionPartitionStringLookupEntry> entries,
    std::string_view key) noexcept {
  if (!state || !state->admitted ||
      state->partitioning_plan.disposition != PlanDisposition::kAccepted || key.empty()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::lower_bound(
      entries.begin(), entries.end(), key, [](const auto& entry, std::string_view value) {
        return entry.key < value;
      });
  if (found == entries.end() || found->key != key || found->table_index == 0U ||
      found->table_index > state->execution_partitions.size()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return static_cast<std::size_t>(found->table_index - 1U);
}

}  // namespace

expected<ExecutionPartitionView, Error> ExecutionPartitionsView::lookup_named(
    std::string_view authored_name) const noexcept {
  auto index = lookup_partition_string_index(
      state_,
      state_ ? std::span{state_->execution_partition_index_by_authored_name}
             : std::span<const detail::ExecutionPartitionStringLookupEntry>{},
      authored_name);
  if (!index) {
    return make_unexpected(std::move(index).error());
  }
  return ExecutionPartitionView{state_, index.value()};
}

expected<ExecutionPartitionView, Error> ExecutionPartitionsView::lookup_for_path(
    std::string_view canonical_operator_path) const noexcept {
  auto index = lookup_partition_string_index(
      state_,
      state_ ? std::span{state_->execution_partition_index_by_operator_path}
             : std::span<const detail::ExecutionPartitionStringLookupEntry>{},
      canonical_operator_path);
  if (!index) {
    return make_unexpected(std::move(index).error());
  }
  return ExecutionPartitionView{state_, index.value()};
}

ExecutionPartitionView ExecutionPartitionsView::of(OperatorRef plan_operator) const {
  if (!usable() || !plan_operator.plan_bound()) {
    throw RuntimeError{ErrorCode::kInvalidArgument};
  }
  if (plan_operator.plan_generation != state_->plan_generation) {
    throw RuntimeError{ErrorCode::kStaleOperatorRef};
  }
  if (plan_operator.id > state_->execution_partition_index_by_operator_id.size()) {
    throw RuntimeError{ErrorCode::kNotFound};
  }
  const std::uint32_t table_index =
      state_->execution_partition_index_by_operator_id[static_cast<std::size_t>(plan_operator.id -
                                                                                1U)];
  if (table_index == 0U || table_index > state_->execution_partitions.size()) {
    throw RuntimeError{ErrorCode::kNotFound};
  }
  return ExecutionPartitionView{state_, static_cast<std::size_t>(table_index - 1U)};
}

ExecutionPartitionView ExecutionPartitionsView::of_root_locator(
    const detail::GraphOperatorLocator& locator) const {
  if (!usable()) {
    throw RuntimeError{ErrorCode::kInvalidArgument};
  }
  if (locator.graph != state_->graph_identity.graph) {
    throw RuntimeError{ErrorCode::kIncompatible};
  }
  if (locator.generation != state_->graph_identity.version) {
    throw RuntimeError{ErrorCode::kStaleOperatorRef};
  }
  if (locator.root_operator_index >= state_->execution_partition_index_by_root_operator.size()) {
    throw RuntimeError{ErrorCode::kNotFound};
  }
  const std::uint32_t table_index =
      state_->execution_partition_index_by_root_operator[locator.root_operator_index];
  if (table_index == 0U || table_index > state_->execution_partitions.size()) {
    throw RuntimeError{ErrorCode::kNotFound};
  }
  return ExecutionPartitionView{state_, static_cast<std::size_t>(table_index - 1U)};
}

ExecutionPartitionsView ExecutionPlan::partitions() const {
  if (!ok() || state_->partitioning_plan.disposition != PlanDisposition::kAccepted) {
    throw RuntimeError{ErrorCode::kInvalidArgument};
  }
  return ExecutionPartitionsView{state_};
}

std::span<const PartitionDecisionRecord> ExecutionPlan::partition_decisions() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::partition_decisions);
}

std::span<const ExecutionPartitionRecord> ExecutionPlan::execution_partitions() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::execution_partitions);
}

std::span<const ScheduleEdgeRecord> ExecutionPlan::schedule_edges() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::schedule_edges);
}

SchedulerPlanRecord ExecutionPlan::scheduler_plan() const noexcept {
  return state_ ? state_->scheduler_plan : SchedulerPlanRecord{};
}

std::span<const PlacementProbePlanRecord> ExecutionPlan::placement_probes() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::placement_probes);
}

std::span<const PartitionPlacementPlanRecord> ExecutionPlan::partition_placements() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::partition_placements);
}

std::span<const ExecutionLanePlanRecord> ExecutionPlan::execution_lanes() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::execution_lanes);
}

std::span<const detail::ProcessRolePlanRecord> detail::ProcessPlacementPlanAccess::process_roles(
    const ExecutionPlan& plan) noexcept {
  return plan_span(ExecutionPlanAccess::retain(plan), &ExecutionPlanState::process_roles);
}

std::span<const detail::ProcessSelectorPlanRecord>
detail::ProcessPlacementPlanAccess::process_selectors(const ExecutionPlan& plan) noexcept {
  return plan_span(ExecutionPlanAccess::retain(plan), &ExecutionPlanState::process_selectors);
}

std::span<const detail::VertexProcessPlacementPlanRecord>
detail::ProcessPlacementPlanAccess::vertex_process_placements(const ExecutionPlan& plan) noexcept {
  return plan_span(ExecutionPlanAccess::retain(plan),
                   &ExecutionPlanState::vertex_process_placements);
}

std::span<const PortPlanRecord> ExecutionPlan::ports() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::ports);
}

std::span<const PortRatePlanRecord> ExecutionPlan::port_rates() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::port_rates);
}

std::span<const AccumulationHorizonPlanRecord> ExecutionPlan::accumulation_horizons()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::accumulation_horizons);
}

std::span<const HardFloorContributorPlanRecord> ExecutionPlan::hard_floor_contributors()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::hard_floor_contributors);
}

std::span<const QueueSizingRecipePlanRecord> ExecutionPlan::queue_sizing_recipes() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::queue_sizing_recipes);
}

std::span<const InterfacePortBindingPlanRecord> ExecutionPlan::interface_port_bindings()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::interface_port_bindings);
}

std::span<const PayloadEdgePlanRecord> ExecutionPlan::payload_edges() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::payload_edges);
}

std::span<const ChannelLocalityRecord> ExecutionPlan::channel_locality() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::channel_locality);
}

std::span<const ChannelServiceStaticPlanRecord> ExecutionPlan::channel_services() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::channel_services);
}

std::span<const ChannelPublisherEndpointPlanRecord> ExecutionPlan::channel_publishers()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::channel_publishers);
}

std::span<const ChannelSubscriberEndpointPlanRecord> ExecutionPlan::channel_subscribers()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::channel_subscribers);
}

std::span<const GenericOutputLoanPlanRecord> ExecutionPlan::generic_output_loans() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::generic_output_loans);
}

std::span<const TensorAllocationPlanRecord> ExecutionPlan::tensor_allocations() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::tensor_allocations);
}

std::span<const TensorByteSpanDerivationPlanRecord> ExecutionPlan::tensor_byte_span_derivations()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::tensor_byte_span_derivations);
}

const TensorByteSpanDerivationPlanRecord* ExecutionPlan::find_tensor_byte_span_derivation(
    std::string_view operator_path, std::string_view output_name,
    std::optional<std::uint16_t> field_id,
    std::optional<std::uint16_t> tensor_index) const noexcept {
  if (!state_ || operator_path.empty() || output_name.empty() ||
      field_id.has_value() != tensor_index.has_value()) {
    return nullptr;
  }
  const auto owner = std::ranges::find(
      state_->operator_parameters, operator_path, &OperatorParameterPlanRecord::operator_path);
  if (owner == state_->operator_parameters.end()) {
    return nullptr;
  }
  const auto factory = std::ranges::find(state_->executable_factories.operators,
                                         owner->owner.id,
                                         &detail::ExecutableOperatorFactoryRecord::operator_id);
  if (factory == state_->executable_factories.operators.end()) {
    return nullptr;
  }
  const auto output = std::ranges::find_if(state_->ports, [&](const auto& candidate) {
    return candidate.owner == factory->vertex && candidate.direction == PortDirection::kOutput &&
           candidate.local_name == output_name;
  });
  if (output == state_->ports.end()) {
    return nullptr;
  }
  const TensorByteSpanDerivationPlanRecord* match{};
  for (const auto& candidate : state_->tensor_byte_span_derivations) {
    if (candidate.output != output->id ||
        (field_id && (candidate.field_id != field_id || candidate.tensor_index != tensor_index))) {
      continue;
    }
    if (match != nullptr) {
      return nullptr;
    }
    match = &candidate;
  }
  return match;
}

std::span<const TensorMemoryKindDerivationPlanRecord>
ExecutionPlan::tensor_memory_kind_derivations() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::tensor_memory_kind_derivations);
}

const TensorMemoryKindDerivationPlanRecord* ExecutionPlan::find_tensor_memory_kind_derivation(
    std::string_view operator_path, std::string_view output_name,
    std::optional<std::uint16_t> field_id,
    std::optional<std::uint16_t> tensor_index) const noexcept {
  if (!state_ || operator_path.empty() || output_name.empty() ||
      field_id.has_value() != tensor_index.has_value()) {
    return nullptr;
  }
  const auto owner = std::ranges::find(
      state_->operator_parameters, operator_path, &OperatorParameterPlanRecord::operator_path);
  if (owner == state_->operator_parameters.end()) {
    return nullptr;
  }
  const auto factory = std::ranges::find(state_->executable_factories.operators,
                                         owner->owner.id,
                                         &detail::ExecutableOperatorFactoryRecord::operator_id);
  if (factory == state_->executable_factories.operators.end()) {
    return nullptr;
  }
  const auto output = std::ranges::find_if(state_->ports, [&](const auto& candidate) {
    return candidate.owner == factory->vertex && candidate.direction == PortDirection::kOutput &&
           candidate.local_name == output_name;
  });
  if (output == state_->ports.end()) {
    return nullptr;
  }
  const TensorMemoryKindDerivationPlanRecord* match{};
  for (const auto& candidate : state_->tensor_memory_kind_derivations) {
    if (candidate.output != output->id ||
        (field_id && (candidate.field_id != field_id || candidate.tensor_index != tensor_index))) {
      continue;
    }
    if (match != nullptr) {
      return nullptr;
    }
    match = &candidate;
  }
  return match;
}

std::span<const TensorFieldAllocationPlanRecord> ExecutionPlan::tensor_field_allocations()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::tensor_field_allocations);
}

std::span<const TensorInterfacePoolPlanRecord> ExecutionPlan::tensor_interface_pools()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::tensor_interface_pools);
}

std::span<const TensorDeviceResourcePlanRecord> ExecutionPlan::tensor_device_resources()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::tensor_device_resources);
}

std::span<const CudaGreenContextPoolPlanRecord> ExecutionPlan::cuda_green_context_pools()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::cuda_green_context_pools);
}

std::span<const CudaGreenContextResourcePlanRecord> ExecutionPlan::cuda_green_context_resources()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::cuda_green_context_resources);
}

std::span<const CudaGreenContextRequestPlanRecord> ExecutionPlan::cuda_green_context_requests()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::cuda_green_context_requests);
}

std::span<const CudaGreenContextAssignmentPlanRecord>
ExecutionPlan::cuda_green_context_assignments() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::cuda_green_context_assignments);
}

std::span<const CudaComputeStreamPlanRecord> ExecutionPlan::cuda_compute_streams() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::cuda_compute_streams);
}

std::span<const TensorInlineRecordPlanRecord> ExecutionPlan::tensor_inline_records()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::tensor_inline_records);
}

std::span<const ImportedGenerationCapacityPlanRecord>
ExecutionPlan::imported_generation_capacities() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::imported_generation_capacities);
}

std::span<const TensorTransferPlanRecord> ExecutionPlan::tensor_transfer_plans() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::tensor_transfer_plans);
}

std::span<const PoolCapacityContributorRecord> ExecutionPlan::pool_capacity_contributors()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::pool_capacity_contributors);
}

std::span<const PoolCapacityDependencyPlanRecord> ExecutionPlan::pool_capacity_dependencies()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::pool_capacity_dependencies);
}

std::span<const ProviderAllocationPlanRecord> ExecutionPlan::provider_allocation_plans()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::provider_allocation_plans);
}

std::span<const PublicationOutcomePlanRecord> ExecutionPlan::publication_outcomes() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::publication_outcomes);
}

std::span<const ReadinessPlanRecord> ExecutionPlan::readiness() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::readiness);
}

ClockCatalogView ExecutionPlan::clock_catalog() const noexcept {
  return state_ ? ClockCatalogView{state_->plan_generation,
                                   state_->root_default_clock,
                                   std::span<const ClockPlanRecord>{state_->clocks},
                                   state_}
                : ClockCatalogView{};
}

std::span<const ClockConversionRecord> ExecutionPlan::clock_conversions() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::clock_conversions);
}

std::span<const EndpointClockHealthPlanRecord> ExecutionPlan::endpoint_clock_health()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::endpoint_clock_health);
}

std::span<const OperatorCompletionPlanRecord> ExecutionPlan::operator_completion_rules()
    const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::operator_completion_rules);
}

std::span<const TemporalTriggerPlanRecord> ExecutionPlan::temporal_triggers() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::temporal_triggers);
}

std::span<const AlignedTriggerPlanRecord> ExecutionPlan::aligned_triggers() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::aligned_triggers);
}

std::span<const ClockTriggerPlanRecord> ExecutionPlan::clock_triggers() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::clock_triggers);
}

std::span<const DeadlinePlanRecord> ExecutionPlan::deadlines() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::deadlines);
}

std::span<const ReadableTriggerPlanRecord> ExecutionPlan::readable_triggers() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::readable_triggers);
}

std::span<const MissingAggregatePlanRecord> ExecutionPlan::missing_aggregates() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::missing_aggregates);
}

std::span<const WindowTriggerPlanRecord> ExecutionPlan::window_triggers() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::window_triggers);
}

std::span<const BracketedWindowPlanRecord> ExecutionPlan::bracketed_windows() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::bracketed_windows);
}

std::span<const InputSelectionPlanRecord> ExecutionPlan::input_selections() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::input_selections);
}

std::span<const PredicateTriggerPlanRecord> ExecutionPlan::predicate_triggers() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::predicate_triggers);
}

std::span<const PredicateInputPlanRecord> ExecutionPlan::predicate_inputs() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::predicate_inputs);
}

std::span<const PredicateCallablePlanRecord> ExecutionPlan::predicate_callables() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::predicate_callables);
}

LifecycleCheckpointRecords ExecutionPlan::lifecycle_checkpoints() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::lifecycle_checkpoints);
}

LifecycleDependencyRecords ExecutionPlan::lifecycle_dependencies() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::lifecycle_dependencies);
}

LifecyclePermissionRecords ExecutionPlan::lifecycle_restart_permissions() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::lifecycle_restart_permissions);
}

LifecycleTargetRecords ExecutionPlan::lifecycle_restart_targets() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::lifecycle_restart_targets);
}

LifecycleClosureRecords ExecutionPlan::lifecycle_restart_closures() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::lifecycle_restart_closures);
}

LifecycleMutableFieldRecords ExecutionPlan::lifecycle_mutable_fields() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::lifecycle_mutable_fields);
}

LifecycleCapacity ExecutionPlan::lifecycle_transaction_capacity() const noexcept {
  return state_ ? state_->lifecycle_transaction_capacity : std::nullopt;
}

OperatorParameterRecords ExecutionPlan::operator_parameters() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::operator_parameters);
}

ParameterFieldRecords ExecutionPlan::parameter_fields() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::parameter_fields);
}

ParameterCatalogView ExecutionPlan::parameter_catalog(ParamCatalogAccessPolicy policy) const {
  if (policy != ParamCatalogAccessPolicy::kPublic || !state_) {
    return {};
  }
  return state_->parameter_catalog;
}

ParameterDerivedTermRecords ExecutionPlan::parameter_derived_terms() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::parameter_derived_terms);
}

ParameterCapacity ExecutionPlan::parameter_runtime_capacity() const noexcept {
  return state_ ? state_->parameter_runtime_capacity : std::nullopt;
}

const ChannelEndpointCatalog& ExecutionPlan::endpoint_catalog() const noexcept {
  return state_ ? state_->endpoint_catalog : kEmptyEndpointCatalog;
}

}  // namespace holoscan
