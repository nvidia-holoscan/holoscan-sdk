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

#ifndef HOLOSCAN_CORE_GPU_RESIDENT_OPERATOR_HPP
#define HOLOSCAN_CORE_GPU_RESIDENT_OPERATOR_HPP

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>

#include "holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/parameter.hpp"

namespace holoscan {

class GPUResidentOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit GPUResidentOperator(ArgT&& arg, ArgsT&&... args)
      : Operator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {
    set_operator_type();
  }

  GPUResidentOperator() { set_operator_type(); }
  ~GPUResidentOperator();

  void set_operator_type() { operator_type_ = OperatorType::kUnknown; }

  std::shared_ptr<ExecutionContext> initialize_execution_context() override;

  std::shared_ptr<GPUResidentExecutor> gpu_resident_executor();

  /// Custom helper functions for GPU-resident Operators below

  /**
   * @brief Get the CUDA stream for the operator to launch a CUDA workload.
   *
   * @return The CUDA stream for the operator to launch a CUDA workload.
   */
  std::shared_ptr<cudaStream_t> cuda_stream();

  /**
   * @brief Get the CUDA stream for the operator to launch a data ready handler CUDA workload.
   *
   * @return std::shared_ptr<cudaStream_t>
   */
  std::shared_ptr<cudaStream_t> data_ready_handler_cuda_stream();

  /**
   * @brief Get the device memory address of an input or output port corresponding to a given port
   * name.
   *
   * @param port_name The name of the input or output port.
   * @return The device memory address of the input or output port. Returns nullptr if the port does
   * not exist or is the port does not correspond to a device memory address.
   */
  void* device_memory(const std::string& port_name);

  /**
   * @brief Get the CUDA device pointer for the data_ready signal.
   *
   * This address can be used in CUDA kernels to signal that data is ready for processing
   * in the GPU-resident execution pipeline. The data ready handler GPU-resident
   * operators can use this address to atomically signal the data is now ready
   * for processing.
   *
   * @return Pointer to the device memory location for the data_ready signal.
   */
  void* data_ready_device_address();

  /**
   * @brief Get the CUDA device pointer for the result_ready signal.
   *
   * This address can be used in CUDA kernels to signal that results from the
   * GPU-resident execution are ready for consumption.
   *
   * @return Pointer to the device memory location for the result_ready signal.
   */
  void* result_ready_device_address();
};

}  // namespace holoscan

#endif
