/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>

#include <gtest/gtest.h>

#include "compiler/execution_plan_identity.hpp"

namespace holoscan::detail {
namespace {

TEST(ExecutionPlanIdentity, RendersChannelAndTensorIdentitiesFromCompleteTypedKeys) {
  // Key setup: render every shared Channel and Tensor identity family with nontrivial keys.
  constexpr PlanGenerationId generation = 17U;
  constexpr PortId port{23U};
  constexpr ExternalTopicId topic{29U};
  constexpr ProcessGroupId process_group{31U};

  // Key assertion: exact stable spellings remain centralized and byte-for-byte compatible.
  EXPECT_EQ(process_local_channel_service_id(generation, port, QueuePolicy::kBlocking),
            "process-local-channel.17.source.23.policy.0");
  EXPECT_EQ(process_local_channel_service_id(generation, port, QueuePolicy::kDropOldest),
            "process-local-channel.17.source.23.policy.1");
  EXPECT_EQ(
      cross_process_channel_service_id(generation, port, QueuePolicy::kDropOldest, EdgeId{37U}),
      "cross-process-channel.17.source.23.policy.1.anchor.37");
  EXPECT_EQ(generic_output_pool_id(generation, port), "generic-output-pool.17.23");
  EXPECT_EQ(tensor_output_pool_id(generation, port), "tensor-output-pool.17.23");
  EXPECT_EQ(tensor_field_output_pool_id(generation, port, 41U),
            "tensor-field-output-pool.17.23.field.41");
  EXPECT_EQ(tensor_schema_staging_pool_id(generation, port), "tensor-schema-staging-pool.17.23");
  EXPECT_EQ(tensor_schema_staging_field_pool_id(generation, port, 43U),
            "tensor-schema-staging-pool.17.23.field.43");
  EXPECT_EQ(tensor_interface_pool_id(generation, topic), "tensor-interface-pool.17.29");
  EXPECT_EQ(tensor_interface_field_pool_id(generation, topic, 7U),
            "tensor-interface-field-pool.17.29.field.7");
  EXPECT_EQ(tensor_inline_record_bucket_id(generation, topic), "tensor-inline-record-bucket.17.29");
  EXPECT_EQ(runtime_scope_id(generation), "runtime-scope.17.1");
  EXPECT_EQ(tensor_allocator_root_id(generation, process_group, DeviceId{}),
            "tensor-allocator.17.process.31.device.host");
  EXPECT_EQ(tensor_allocator_root_id(generation, process_group, DeviceId{5}),
            "tensor-allocator.17.process.31.device.5");
  EXPECT_EQ(tensor_compute_stream_id(generation, process_group, DeviceId{5}),
            "tensor-stream.17.process.31.device.5");
  EXPECT_EQ(tensor_partition_compute_stream_id(
                generation, process_group, DeviceId{5}, ExecutionPartitionId{7U}),
            "tensor-stream.17.process.31.device.5.partition.7");
  EXPECT_NE(tensor_partition_compute_stream_id(
                generation, ProcessGroupId{31U}, DeviceId{5}, ExecutionPartitionId{7U}),
            tensor_partition_compute_stream_id(
                generation, ProcessGroupId{32U}, DeviceId{5}, ExecutionPartitionId{7U}));
}

TEST(ExecutionPlanIdentity, RendersGreenContextDescendantsFromOwningIdentities) {
  // Key setup: derive dedicated, residual, partition, and service stream descendants.
  const std::string pool = cuda_green_context_pool_id(17U, ProcessGroupId{31U}, DeviceId{5});
  const std::string dedicated = cuda_green_context_partition_id(pool, ExecutionPartitionId{7U});
  const std::string residual = cuda_green_context_residual_id(pool);

  // Key assertion: every descendant retains the exact owning prefix and structural suffix.
  EXPECT_EQ(pool, "cuda-green-pool.17.process.31.device.5");
  EXPECT_EQ(dedicated, "cuda-green-pool.17.process.31.device.5.partition.7");
  EXPECT_EQ(residual, "cuda-green-pool.17.process.31.device.5.residual");
  EXPECT_EQ(cuda_green_context_stream_id(dedicated),
            "cuda-green-pool.17.process.31.device.5.partition.7.stream");
  EXPECT_EQ(cuda_green_context_partition_stream_id(residual, ExecutionPartitionId{11U}),
            "cuda-green-pool.17.process.31.device.5.residual.partition.11.stream");
  EXPECT_EQ(cuda_green_context_service_stream_id(residual, "tensor.transfer.47"),
            "cuda-green-pool.17.process.31.device.5.residual.tensor.transfer.47.stream");
}

}  // namespace
}  // namespace holoscan::detail
