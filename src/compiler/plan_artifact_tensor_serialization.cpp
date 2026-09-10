/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/plan_artifact_serialization_internal.hpp"

#include <cstdint>
#include <string>
#include <string_view>

#include "core/canonical_text_encoding.hpp"

namespace holoscan::detail::plan_artifact_serialization_internal {

void append_tensor_allocation_bounds(std::string& output, const TensorPoolBounds& bounds) {
  output += "{\"max_byte_span\":";
  append_number(output, bounds.max_byte_span);
  output += ",\"max_rank\":";
  append_number(output, bounds.max_rank);
  output.push_back('}');
}

namespace {

[[nodiscard]] constexpr std::string_view cuda_assignment_source_token(
    CudaGreenContextAssignmentSource value) noexcept {
  switch (value) {
    case CudaGreenContextAssignmentSource::kDeveloperBinding:
      return "developer_binding";
    case CudaGreenContextAssignmentSource::kAutomaticPolicy:
      return "automatic_policy";
    case CudaGreenContextAssignmentSource::kResidualPolicy:
      return "residual_policy";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view cuda_resource_role_token(
    CudaGreenContextResourceRole value) noexcept {
  switch (value) {
    case CudaGreenContextResourceRole::kDedicatedPartition:
      return "dedicated_partition";
    case CudaGreenContextResourceRole::kSharedServiceResidual:
      return "shared_service_residual";
    case CudaGreenContextResourceRole::kReservedResidual:
      return "reserved_residual";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view cuda_co_scheduling_token(
    CudaGreenContextCoScheduling value) noexcept {
  switch (value) {
    case CudaGreenContextCoScheduling::kPreserveDeviceAlignment:
      return "preserve_device_alignment";
    case CudaGreenContextCoScheduling::kRelaxed:
      return "relaxed";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view cuda_workqueue_scope_token(
    CudaGreenContextWorkqueueScope value) noexcept {
  switch (value) {
    case CudaGreenContextWorkqueueScope::kDeviceContext:
      return "device_context";
    case CudaGreenContextWorkqueueScope::kGreenContextBalanced:
      return "green_context_balanced";
  }
  return "invalid";
}

[[nodiscard]] constexpr std::string_view cuda_sm_set_guarantee_token(
    CudaGreenContextSmSetGuarantee value) noexcept {
  switch (value) {
    case CudaGreenContextSmSetGuarantee::kDriverDocumented:
      return "driver_documented";
    case CudaGreenContextSmSetGuarantee::kStrictNoKnownSuperset:
      return "strict_no_known_superset";
  }
  return "invalid";
}

void append_cuda_placement(std::string& output, const CudaGreenContextPlacement& placement) {
  output += "{\"device\":";
  append_number(output, placement.device.value);
  output += ",\"sm_count\":";
  append_number(output, placement.sm_count);
  output.push_back('}');
}

}  // namespace

// Canonical artifact order is clearer as one linear section writer; optional fields and record
// lambdas inflate the metric without introducing control-flow coupling.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void append_tensor_and_cuda_sections(std::string& output, const ExecutionPlanState& state,
                                     const bool include_provenance) {
  if (!state.tensor_field_allocations.empty()) {
    output += ",\"tensor_field_allocations\":";
    append_array(output, state.tensor_field_allocations, [](std::string& json, const auto& record) {
      json += "{\"aggregate_byte_capacity\":";
      append_number(json, record.aggregate_byte_capacity);
      json += ",\"allocator_process_group\":";
      append_number(json, record.allocator_process_group.value);
      json += ",\"allocator_root_id\":";
      append_json_string(json, record.allocator_root_id);
      json += ",\"axes\":";
      append_tensor_axes(json, record.axes);
      json += ",\"bounds\":";
      append_tensor_allocation_bounds(json, record.bounds);
      json += ",\"control_capacity\":";
      append_number(json, record.control_capacity);
      json += ",\"device\":";
      append_number(json, record.device.value);
      json += ",\"disposition\":";
      append_number(json, static_cast<std::uint32_t>(record.disposition));
      json += ",\"effective_profile\":";
      append_optional_profile(json, record.effective_profile);
      json += ",\"fallback_rejected\":";
      append_bool(json, record.fallback_rejected);
      json += ",\"field_id\":";
      append_number(json, record.field_id);
      json += ",\"layout\":";
      append_optional_tensor_layout(json, record.layout);
      json += ",\"materialize_before\":";
      append_number(json, static_cast<std::uint32_t>(record.materialize_before));
      json += ",\"max_emits_per_compute\":";
      append_number(json, record.max_emits_per_compute);
      json += ",\"max_outstanding_loans\":";
      append_number(json, record.max_outstanding_loans);
      json += ",\"memory_kind\":";
      append_number(json, static_cast<std::uint32_t>(record.memory_kind));
      json += ",\"output\":";
      append_number(json, record.output.value());
      json += ",\"plan_generation\":";
      append_number(json, record.plan_generation);
      json += ",\"pool_id\":";
      append_json_string(json, record.pool_id);
      json += ",\"quarantine_capacity\":";
      append_number(json, record.quarantine_capacity);
      json += ",\"readiness_capacity\":";
      append_number(json, record.readiness_capacity);
      json += ",\"release_capacity\":";
      append_number(json, record.release_capacity);
      json += ",\"retirement_capacity\":";
      append_number(json, record.retirement_capacity);
      json += ",\"run_generation\":";
      append_number(json, record.run_generation);
      json += ",\"slot_count\":";
      append_number(json, record.slot_count);
      json += ",\"tensor_index\":";
      append_number(json, record.tensor_index);
      json.push_back('}');
    });
  }
  output += ",\"tensor_allocations\":";
  append_array(output, state.tensor_allocations, [](std::string& json, const auto& record) {
    json += "{\"aggregate_byte_capacity\":";
    append_number(json, record.aggregate_byte_capacity);
    json += ",\"allocator_process_group\":";
    append_number(json, record.allocator_process_group.value);
    json += ",\"allocator_root_id\":";
    append_json_string(json, record.allocator_root_id);
    json += ",\"axes\":";
    append_tensor_axes(json, record.axes);
    json += ",\"bounds\":";
    append_tensor_allocation_bounds(json, record.bounds);
    json += ",\"control_capacity\":";
    append_number(json, record.control_capacity);
    json += ",\"device\":";
    append_number(json, record.device.value);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"fallback_rejected\":";
    append_bool(json, record.fallback_rejected);
    json += ",\"layout\":";
    append_optional_tensor_layout(json, record.layout);
    json += ",\"materialize_before\":";
    append_number(json, static_cast<std::uint32_t>(record.materialize_before));
    json += ",\"max_emits_per_compute\":";
    append_number(json, record.max_emits_per_compute);
    json += ",\"max_outstanding_loans\":";
    append_number(json, record.max_outstanding_loans);
    json += ",\"memory_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.memory_kind));
    json += ",\"output\":";
    append_number(json, record.output.value());
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"quarantine_capacity\":";
    append_number(json, record.quarantine_capacity);
    json += ",\"readiness_capacity\":";
    append_number(json, record.readiness_capacity);
    json += ",\"release_capacity\":";
    append_number(json, record.release_capacity);
    json += ",\"retirement_capacity\":";
    append_number(json, record.retirement_capacity);
    json += ",\"run_generation\":";
    append_number(json, record.run_generation);
    json += ",\"slot_count\":";
    append_number(json, record.slot_count);
    json.push_back('}');
  });
  output += ",\"tensor_byte_span_derivations\":";
  append_array(
      output, state.tensor_byte_span_derivations, [](std::string& json, const auto& record) {
        json += "{\"nodes\":";
        append_array(json, record.nodes, [](std::string& node_json, const auto& node) {
          node_json += "{\"input\":";
          append_number(node_json, node.input.value());
          node_json += ",\"left\":";
          append_number(node_json, node.left);
          node_json += ",\"operand\":";
          append_number(node_json, node.operand);
          node_json += ",\"operation\":";
          append_number(node_json, static_cast<std::uint32_t>(node.operation));
          node_json += ",\"parameter_field\":";
          append_number(node_json, node.parameter_field);
          node_json += ",\"resolved_value\":";
          append_number(node_json, node.resolved_value);
          node_json += ",\"right\":";
          append_number(node_json, node.right);
          node_json.push_back('}');
        });
        json += ",\"field_id\":";
        if (record.field_id) {
          append_number(json, *record.field_id);
        } else {
          json += "null";
        }
        json += ",\"output\":";
        append_number(json, record.output.value());
        json += ",\"plan_generation\":";
        append_number(json, record.plan_generation);
        json += ",\"resolved_max_byte_span\":";
        append_number(json, record.resolved_max_byte_span);
        json += ",\"resolved_max_staging_byte_span\":";
        if (record.resolved_max_staging_byte_span) {
          append_number(json, *record.resolved_max_staging_byte_span);
        } else {
          json += "null";
        }
        json += ",\"run_generation\":";
        append_number(json, record.run_generation);
        json += ",\"tensor_index\":";
        if (record.tensor_index) {
          append_number(json, *record.tensor_index);
        } else {
          json += "null";
        }
        json += ",\"update_disposition\":";
        append_number(json, static_cast<std::uint32_t>(record.update_disposition));
        json.push_back('}');
      });
  output += ",\"cuda_compute_streams\":";
  append_array(output, state.cuda_compute_streams, [](std::string& json, const auto& record) {
    json += "{\"device\":";
    append_number(json, record.device.value);
    json += ",\"fused_linear_edges\":";
    append_array(json, record.fused_linear_edges, [](std::string& values, EdgeId edge) {
      append_number(values, edge.value);
    });
    json += ",\"green_context_id\":";
    append_optional_string(json, record.green_context_id);
    json += ",\"materialize_before\":";
    append_number(json, static_cast<std::uint32_t>(record.materialize_before));
    json += ",\"owner_id\":";
    append_json_string(json, record.owner_id);
    json += ",\"partition\":";
    if (record.partition) {
      append_number(json, record.partition->value);
    } else {
      json += "null";
    }
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"process_group\":";
    append_number(json, record.process_group.value);
    json += ",\"run_generation\":";
    append_number(json, record.run_generation);
    json += ",\"stream_id\":";
    append_json_string(json, record.stream_id);
    json.push_back('}');
  });
  output += ",\"cuda_green_context_assignments\":";
  append_array(
      output, state.cuda_green_context_assignments, [](std::string& json, const auto& record) {
        json += "{\"device\":";
        append_number(json, record.device.value);
        json += ",\"execution_lane\":";
        append_number(json, record.execution_lane);
        json += ",\"green_context_id\":";
        append_json_string(json, record.green_context_id);
        json += ",\"partition\":";
        append_number(json, record.partition.value);
        json += ",\"plan_generation\":";
        append_number(json, record.plan_generation);
        json += ",\"pool_id\":";
        append_json_string(json, record.pool_id);
        json += ",\"process_group\":";
        append_number(json, record.process_group.value);
        json += ",\"run_generation\":";
        append_number(json, record.run_generation);
        json += ",\"source\":";
        append_json_string(json, cuda_assignment_source_token(record.source));
        json.push_back('}');
      });
  output += ",\"cuda_green_context_pools\":";
  append_array(output, state.cuda_green_context_pools, [](std::string& json, const auto& record) {
    json += "{\"co_scheduling\":";
    append_json_string(json, cuda_co_scheduling_token(record.co_scheduling));
    json += ",\"deployment_device_id\":";
    append_json_string(json, record.deployment_device_id);
    json += ",\"device\":";
    append_number(json, record.device.value);
    json += ",\"device_sm_count\":";
    append_number(json, record.device_sm_count);
    json += ",\"fallback_rejected\":";
    append_bool(json, record.fallback_rejected);
    json += ",\"materialize_before\":";
    append_number(json, static_cast<std::uint32_t>(record.materialize_before));
    json += ",\"max_structured_context_count\":";
    append_number(json, record.max_structured_context_count);
    json += ",\"min_sm_partition_size\":";
    append_number(json, record.min_sm_partition_size);
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"process_group\":";
    append_number(json, record.process_group.value);
    json += ",\"run_generation\":";
    append_number(json, record.run_generation);
    json += ",\"sm_coscheduled_alignment\":";
    append_number(json, record.sm_coscheduled_alignment);
    json += ",\"structured_sm_quantum\":";
    append_number(json, record.structured_sm_quantum);
    json += ",\"sm_set_guarantee\":";
    append_json_string(json, cuda_sm_set_guarantee_token(record.sm_set_guarantee));
    json += ",\"unprovisioned_sm_count\":";
    append_number(json, record.unprovisioned_sm_count);
    json += ",\"workqueue_concurrency_limit\":";
    append_number(json, record.workqueue_concurrency_limit);
    json += ",\"workqueue_scope\":";
    append_json_string(json, cuda_workqueue_scope_token(record.workqueue_scope));
    json.push_back('}');
  });
  output += ",\"cuda_green_context_requests\":";
  append_array(
      output, state.cuda_green_context_requests, [](std::string& json, const auto& record) {
        json += "{\"accepted\":";
        if (record.accepted) {
          append_cuda_placement(json, *record.accepted);
        } else {
          json += "null";
        }
        json += ",\"authored_name\":";
        append_json_string(json, record.authored_name);
        json += ",\"disposition\":";
        append_number(json, static_cast<std::uint32_t>(record.disposition));
        json += ",\"execution_lane\":";
        append_number(json, record.execution_lane);
        json += ",\"partition\":";
        append_number(json, record.partition.value);
        json += ",\"plan_generation\":";
        append_number(json, record.plan_generation);
        json += ",\"reason\":";
        append_number(json, record.reason.value);
        json += ",\"requested\":";
        append_cuda_placement(json, record.requested);
        json += ",\"run_generation\":";
        append_number(json, record.run_generation);
        json += ",\"source\":";
        append_json_string(json, cuda_assignment_source_token(record.source));
        json.push_back('}');
      });
  output += ",\"cuda_green_context_resources\":";
  append_array(
      output, state.cuda_green_context_resources, [](std::string& json, const auto& record) {
        json += "{\"accepted_sm_count\":";
        append_number(json, record.accepted_sm_count);
        json += ",\"device\":";
        append_number(json, record.device.value);
        json += ",\"green_context_id\":";
        append_json_string(json, record.green_context_id);
        json += ",\"plan_generation\":";
        append_number(json, record.plan_generation);
        json += ",\"pool_id\":";
        append_json_string(json, record.pool_id);
        json += ",\"process_group\":";
        append_number(json, record.process_group.value);
        json += ",\"role\":";
        append_json_string(json, cuda_resource_role_token(record.role));
        json += ",\"run_generation\":";
        append_number(json, record.run_generation);
        json.push_back('}');
      });
  output += ",\"tensor_device_resources\":";
  append_array(output, state.tensor_device_resources, [](std::string& json, const auto& record) {
    json += "{\"allocator_root_id\":";
    append_json_string(json, record.allocator_root_id);
    json += ",\"control_capacity\":";
    append_number(json, record.control_capacity);
    json += ",\"device\":";
    append_number(json, record.device.value);
    json += ",\"fallback_rejected\":";
    append_bool(json, record.fallback_rejected);
    json += ",\"gpu_resident_backend_selected\":";
    append_bool(json, record.gpu_resident_backend_selected);
    json += ",\"loan_credit_capacity\":";
    append_number(json, record.loan_credit_capacity);
    json += ",\"materialize_before\":";
    append_number(json, static_cast<std::uint32_t>(record.materialize_before));
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"pool_count\":";
    append_number(json, record.pool_count);
    json += ",\"process_group\":";
    append_number(json, record.process_group.value);
    json += ",\"quarantine_capacity\":";
    append_number(json, record.quarantine_capacity);
    json += ",\"readiness_capacity\":";
    append_number(json, record.readiness_capacity);
    json += ",\"release_capacity\":";
    append_number(json, record.release_capacity);
    json += ",\"retirement_capacity\":";
    append_number(json, record.retirement_capacity);
    json += ",\"run_generation\":";
    append_number(json, record.run_generation);
    json += ",\"slot_capacity\":";
    append_number(json, record.slot_capacity);
    json.push_back('}');
  });
  output += ",\"tensor_inline_records\":";
  append_array(output, state.tensor_inline_records, [](std::string& json, const auto& record) {
    json += "{\"accepted_max_inline_logical_bytes\":";
    append_number(json, record.accepted_max_inline_logical_bytes);
    json += ",\"capability_evidence_id\":";
    append_json_string(json, record.capability_evidence_id);
    json += ",\"capability_id\":";
    append_json_string(json, record.capability_id);
    json += ",\"capability_version\":";
    append_number(json, record.capability_version);
    json += ",\"descriptor_arm_materialized\":";
    append_bool(json, record.descriptor_arm_materialized);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"effective_max_inline_logical_bytes\":";
    append_number(json, record.effective_max_inline_logical_bytes);
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"fallback_rejected\":";
    append_bool(json, record.fallback_rejected);
    json += ",\"fixed_metadata_overhead_bytes\":";
    append_number(json, record.fixed_metadata_overhead_bytes);
    json += ",\"fixed_record_bucket_bytes\":";
    append_number(json, record.fixed_record_bucket_bytes);
    json += ",\"fixed_record_bucket_id\":";
    append_json_string(json, record.fixed_record_bucket_id);
    json += ",\"inline_arm_materialized\":";
    append_bool(json, record.inline_arm_materialized);
    json += ",\"inline_arm_schema_id\":";
    append_json_string(json, record.inline_arm_schema_id);
    json += ",\"inline_discriminator\":";
    append_json_string(json, record.inline_discriminator);
    json += ",\"opaque_descriptor_arm_schema_id\":";
    append_json_string(json, record.opaque_descriptor_arm_schema_id);
    json += ",\"opaque_descriptor_discriminator\":";
    append_json_string(json, record.opaque_descriptor_discriminator);
    json += ",\"profile_max_inline_logical_bytes\":";
    append_number(json, record.profile_max_inline_logical_bytes);
    json += ",\"provider_id\":";
    append_json_string(json, record.provider_id);
    json += ",\"provider_max_fixed_record_bytes\":";
    append_number(json, record.provider_max_fixed_record_bytes);
    json += ",\"provider_max_inline_logical_bytes\":";
    append_number(json, record.provider_max_inline_logical_bytes);
    json += ",\"provider_max_tensor_rank\":";
    append_number(json, record.provider_max_tensor_rank);
    json += ",\"record_format_id\":";
    append_json_string(json, record.record_format_id);
    json += ",\"record_format_version\":";
    append_number(json, record.record_format_version);
    json += ",\"service_family_id\":";
    append_json_string(json, record.service_family_id);
    json += ",\"service_id\":";
    append_json_string(json, record.service_id);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    json.push_back('}');
  });
  output += ",\"tensor_interface_pools\":";
  append_array(output, state.tensor_interface_pools, [&](std::string& json, const auto& record) {
    json += "{\"aggregate_byte_capacity\":";
    append_number(json, record.aggregate_byte_capacity);
    json += ",\"allocator_process_group\":";
    append_number(json, record.allocator_process_group.value);
    json += ",\"allocator_root_id\":";
    append_json_string(json, record.allocator_root_id);
    json += ",\"control_capacity\":";
    append_number(json, record.control_capacity);
    json += ",\"device\":";
    append_number(json, record.device.value);
    json += ",\"emit_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.emit_policy));
    json += ",\"effective_profile\":";
    append_optional_profile(json, record.effective_profile);
    json += ",\"exporter_subpool_id\":";
    append_json_string(json, record.exporter_subpool_id);
    if (record.field_id) {
      json += ",\"field_id\":";
      append_number(json, *record.field_id);
    }
    json += ",\"graph_port\":";
    append_number(json, record.graph_port.value());
    json += ",\"imported\":";
    append_bool(json, record.imported);
    json += ",\"logical\":";
    append_logical_tensor_requirements(json, record.logical, include_provenance);
    json += ",\"memory_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.memory_kind));
    json += ",\"plan_generation\":";
    append_number(json, record.plan_generation);
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"publisher_loan_capacity\":";
    append_number(json, record.publisher_loan_capacity);
    json += ",\"quarantine_capacity\":";
    append_number(json, record.quarantine_capacity);
    json += ",\"readiness_capacity\":";
    append_number(json, record.readiness_capacity);
    json += ",\"release_capacity\":";
    append_number(json, record.release_capacity);
    json += ",\"retirement_capacity\":";
    append_number(json, record.retirement_capacity);
    json += ",\"run_generation\":";
    append_number(json, record.run_generation);
    json += ",\"slot_count\":";
    append_number(json, record.slot_count);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    if (record.tensor_index) {
      json += ",\"tensor_index\":";
      append_number(json, *record.tensor_index);
    }
    json.push_back('}');
  });
  output += ",\"tensor_transfer_plans\":";
  append_array(output, state.tensor_transfer_plans, [](std::string& json, const auto& record) {
    json += "{\"delivered_read_only\":";
    append_bool(json, record.delivered_read_only);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"evidence_id\":";
    append_json_string(json, record.evidence_id);
    if (record.field_id) {
      json += ",\"field_id\":";
      append_number(json, *record.field_id);
    }
    if (record.interface_process_group) {
      json += ",\"interface_device\":";
      append_number(json, record.interface_device.value);
      json += ",\"interface_process_group\":";
      append_number(json, record.interface_process_group.value);
    }
    json += ",\"max_logical_bytes\":";
    append_number(json, record.max_logical_bytes);
    json += ",\"min_logical_bytes\":";
    append_number(json, record.min_logical_bytes);
    json += ",\"realization\":";
    if (record.realization) {
      append_number(json, static_cast<std::uint32_t>(*record.realization));
    } else {
      json += "null";
    }
    json += ",\"service_owner_id\":";
    append_json_string(json, record.service_owner_id);
    json += ",\"requires_exportable_copy_pool\":";
    append_bool(json, record.requires_exportable_copy_pool);
    json += ",\"requires_inline_record\":";
    append_bool(json, record.requires_inline_record);
    json += ",\"source_class\":";
    append_number(json, static_cast<std::uint32_t>(record.source_class));
    json += ",\"source_must_be_exportable\":";
    append_bool(json, record.source_must_be_exportable);
    json += ",\"source_must_be_host_ready\":";
    append_bool(json, record.source_must_be_host_ready);
    json += ",\"source_retained_until_materialization\":";
    append_bool(json, record.source_retained_until_materialization);
    json += ",\"topic\":";
    append_number(json, record.topic.value);
    if (record.tensor_index) {
      json += ",\"tensor_index\":";
      append_number(json, *record.tensor_index);
    }
    json.push_back('}');
  });
}

}  // namespace holoscan::detail::plan_artifact_serialization_internal
