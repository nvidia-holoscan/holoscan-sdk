/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gpu_resident_operator.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

GPUResidentOperator::~GPUResidentOperator() {
  HOLOSCAN_LOG_DEBUG("GPUResidentOperator::~GPUResidentOperator()");
}

std::shared_ptr<ExecutionContext> GPUResidentOperator::initialize_execution_context() {
  auto executor_shared = fragment()->executor_shared();
  auto gpu_resident_executor =
      std::dynamic_pointer_cast<holoscan::GPUResidentExecutor>(executor_shared);
  if (!gpu_resident_executor) {
    throw std::runtime_error(
        "GPUResidentOperator is not configured with a GPU-resident holoscan executor.");
  }
  return gpu_resident_executor->execution_context();
}

std::shared_ptr<cudaStream_t> GPUResidentOperator::cuda_stream() {
  auto gr_executor = gpu_resident_executor();
  return gr_executor->graph_capture_stream();
}

void* GPUResidentOperator::device_memory(const std::string& port_name) {
  auto gr_executor = gpu_resident_executor();
  return gr_executor->device_memory(self_shared(), port_name);
}

std::shared_ptr<GPUResidentExecutor> GPUResidentOperator::gpu_resident_executor() {
  auto gr_executor = std::dynamic_pointer_cast<holoscan::GPUResidentExecutor>(executor());
  if (!gr_executor) {
    throw std::runtime_error(
        "GPUResidentOperator is not configured with a GPU-resident holoscan executor.");
  }
  return gr_executor;
}

}  // namespace holoscan
