/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

#include <holoscan/core/execution_plan.hpp>

namespace holoscan::detail {

/// Retain pointer-free, run-local evidence for one CUDA execution startup or teardown operation.
struct RuntimeCudaExecutionDiagnostic {
  PlanGenerationId plan_generation{};
  RunGenerationId run_generation{};
  LifecycleStage lifecycle_stage{LifecycleStage::kArm};
  std::string operation;
  DeviceId device{};
  std::string deployment_device_id;
  std::string pool_id;
  std::string green_context_id;
  std::string stream_id;
  std::int32_t driver_result{-1};
};

}  // namespace holoscan::detail
