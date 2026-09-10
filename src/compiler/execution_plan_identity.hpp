/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include <holoscan/core/execution_plan.hpp>

namespace holoscan::detail {

/// Return the canonical process-local Channel service identity.
[[nodiscard]] inline std::string process_local_channel_service_id(PlanGenerationId generation,
                                                                  PortId source,
                                                                  QueuePolicy policy) {
  return fmt::format("process-local-channel.{}.source.{}.policy.{}",
                     generation,
                     source.value(),
                     static_cast<std::uint32_t>(policy));
}

/// Return the canonical cross-process Channel service identity.
[[nodiscard]] inline std::string cross_process_channel_service_id(PlanGenerationId generation,
                                                                  PortId source, QueuePolicy policy,
                                                                  EdgeId anchor) {
  return fmt::format("cross-process-channel.{}.source.{}.policy.{}.anchor.{}",
                     generation,
                     source.value(),
                     static_cast<std::uint32_t>(policy),
                     anchor.value);
}

/// Return the canonical generic output pool identity.
[[nodiscard]] inline std::string generic_output_pool_id(PlanGenerationId generation,
                                                        PortId output) {
  return fmt::format("generic-output-pool.{}.{}", generation, output.value());
}

/// Return the canonical Tensor output pool identity.
[[nodiscard]] inline std::string tensor_output_pool_id(PlanGenerationId generation, PortId output) {
  return fmt::format("tensor-output-pool.{}.{}", generation, output.value());
}

/// Return the canonical Tensor-field output pool identity.
[[nodiscard]] inline std::string tensor_field_output_pool_id(PlanGenerationId generation,
                                                             PortId output,
                                                             std::uint32_t field_id) {
  return fmt::format(
      "tensor-field-output-pool.{}.{}.field.{}", generation, output.value(), field_id);
}

/// Return the canonical descriptor-schema Tensor staging pool identity.
[[nodiscard]] inline std::string tensor_schema_staging_pool_id(PlanGenerationId generation,
                                                               PortId output) {
  return fmt::format("tensor-schema-staging-pool.{}.{}", generation, output.value());
}

/// Return the canonical descriptor-schema Tensor-field staging pool identity.
[[nodiscard]] inline std::string tensor_schema_staging_field_pool_id(PlanGenerationId generation,
                                                                     PortId output,
                                                                     std::uint32_t field_id) {
  return fmt::format(
      "tensor-schema-staging-pool.{}.{}.field.{}", generation, output.value(), field_id);
}

/// Return the canonical external Tensor interface pool identity.
[[nodiscard]] inline std::string tensor_interface_pool_id(PlanGenerationId generation,
                                                          ExternalTopicId topic) {
  return fmt::format("tensor-interface-pool.{}.{}", generation, topic.value);
}

/// Return the canonical descriptor-schema Tensor-field interface pool identity.
[[nodiscard]] inline std::string tensor_interface_field_pool_id(PlanGenerationId generation,
                                                                ExternalTopicId topic,
                                                                std::uint16_t field_id) {
  return fmt::format(
      "tensor-interface-field-pool.{}.{}.field.{}", generation, topic.value, field_id);
}

/// Return the canonical inline Tensor-record bucket identity.
[[nodiscard]] inline std::string tensor_inline_record_bucket_id(PlanGenerationId generation,
                                                                ExternalTopicId topic) {
  return fmt::format("tensor-inline-record-bucket.{}.{}", generation, topic.value);
}

/// Return the canonical imported-generation Runtime scope identity.
[[nodiscard]] inline std::string runtime_scope_id(PlanGenerationId generation) {
  return fmt::format("runtime-scope.{}.1", generation);
}

/// Return the canonical Tensor allocator-root identity.
[[nodiscard]] inline std::string tensor_allocator_root_id(PlanGenerationId generation,
                                                          ProcessGroupId process_group,
                                                          DeviceId device) {
  return device.valid()
             ? fmt::format("tensor-allocator.{}.process.{}.device.{}",
                           generation,
                           process_group.value,
                           device.value)
             : fmt::format(
                   "tensor-allocator.{}.process.{}.device.host", generation, process_group.value);
}

/// Return the canonical ordinary Tensor compute-stream identity.
[[nodiscard]] inline std::string tensor_compute_stream_id(PlanGenerationId generation,
                                                          ProcessGroupId process_group,
                                                          DeviceId device) {
  return fmt::format(
      "tensor-stream.{}.process.{}.device.{}", generation, process_group.value, device.value);
}

/// Return the canonical CUDA Green Context pool identity.
[[nodiscard]] inline std::string cuda_green_context_pool_id(PlanGenerationId generation,
                                                            ProcessGroupId process_group,
                                                            DeviceId device) {
  return fmt::format(
      "cuda-green-pool.{}.process.{}.device.{}", generation, process_group.value, device.value);
}

/// Return the canonical dedicated Green Context identity under a pool.
[[nodiscard]] inline std::string cuda_green_context_partition_id(std::string_view pool_id,
                                                                 ExecutionPartitionId partition) {
  return fmt::format("{}.partition.{}", pool_id, partition.value);
}

/// Return the canonical residual Green Context identity under a pool.
[[nodiscard]] inline std::string cuda_green_context_residual_id(std::string_view pool_id) {
  return fmt::format("{}.residual", pool_id);
}

/// Return the canonical stream identity owned directly by a Green Context.
[[nodiscard]] inline std::string cuda_green_context_stream_id(std::string_view context_id) {
  return fmt::format("{}.stream", context_id);
}

/// Return the canonical residual partition stream identity.
[[nodiscard]] inline std::string cuda_green_context_partition_stream_id(
    std::string_view residual_context_id, ExecutionPartitionId partition) {
  return fmt::format("{}.partition.{}.stream", residual_context_id, partition.value);
}

/// Return the canonical residual service stream identity.
[[nodiscard]] inline std::string cuda_green_context_service_stream_id(
    std::string_view residual_context_id, std::string_view owner_id) {
  return fmt::format("{}.{}.stream", residual_context_id, owner_id);
}

}  // namespace holoscan::detail
