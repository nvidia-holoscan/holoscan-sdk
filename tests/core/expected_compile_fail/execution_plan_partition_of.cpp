/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/execution_plan.hpp>

// REQ: COMPILE-068
int main() {
  holoscan::ExecutionPlan plan;
  (void)plan.partition_of(holoscan::ExecutionPartitionId{1U});
}
