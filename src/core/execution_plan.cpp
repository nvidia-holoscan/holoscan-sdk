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

#include "../compiler/execution_plan_state.hpp"
#include "../compiler/process_placement_internal.hpp"

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
    throw CompileError{"EXECUTION_PLAN_REJECTED"};
  }
}

const ProfileManifest& ExecutionPlan::profile_manifest() const noexcept {
  return state_ ? state_->profile_manifest : kEmptyProfileManifest;
}

const PartitioningPlanRecord& ExecutionPlan::partitioning_plan() const noexcept {
  return state_ ? state_->partitioning_plan : kEmptyPartitioningPlan;
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

std::span<const SchemaPackagePlanRecord> ExecutionPlan::schema_packages() const noexcept {
  return plan_span(state_, &detail::ExecutionPlanState::schema_packages);
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
