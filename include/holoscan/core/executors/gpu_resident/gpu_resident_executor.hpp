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

#ifndef HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GPU_RESIDENT_EXECUTOR_HPP
#define HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GPU_RESIDENT_EXECUTOR_HPP

#include <cuda_runtime.h>

#include <fmt/format.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gpu_resident_deck.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/gpu_resident_operator.hpp"
#include "holoscan/utils/cuda/buffer.hpp"

namespace holoscan {

class GPUResidentExecutor : public Executor {
 public:
  GPUResidentExecutor() = delete;

  /**
   * @brief Construct a new GPUResidentExecutor object.
   *
   * @param fragment The pointer to the fragment of the executor.
   */
  explicit GPUResidentExecutor(Fragment* fragment) : Executor(fragment) {
    gpu_resident_deck_ = std::make_shared<GPUResidentDeck>();
  }

  ~GPUResidentExecutor();

  void run(OperatorGraph& graph) override;

  std::future<void> run_async(OperatorGraph& graph) override;

  void context([[maybe_unused]] void* context) override {
    throw std::runtime_error("GPUResidentExecutor does not support context");
  }

  bool initialize_fragment() override;

  bool initialize_operator([[maybe_unused]] Operator* op) override;

  bool initialize_scheduler([[maybe_unused]] Scheduler* sch) override {
    throw std::runtime_error("GPUResidentExecutor does not support any scheduler");
  }

  bool initialize_network_context([[maybe_unused]] NetworkContext* network_context) override {
    throw std::runtime_error("GPUResidentExecutor does not support any network context");
  }

  bool initialize_fragment_services() override {
    throw std::runtime_error("GPUResidentExecutor does not support any fragment services");
  }

  /**
   * @brief This function prepares the data flow connections between operators
   * It allocates a device memory block for each connection according to the memory block size
   * specified in the operator spec
   *
   * @param graph The operator graph
   */
  void prepare_data_flow(std::shared_ptr<OperatorGraph> graph);

  /**
   * @brief This function initializes CUDA. Currently, it sets the device to 0 by default.
   * Setting a different GPU device for GPU-resident execution is not yet supported.
   */
  void initialize_cuda();

  /**
   * @brief This function returns the device memory address of an input or output port corresponding
   * to a given port name. GPU-resident operators use this function to get the device memory address
   * of the input or output port.
   *
   * @param op The operator
   * @param port_name The name of the input or output port
   * @return The device memory address of the input or output port
   */
  void* device_memory(std::shared_ptr<Operator> op, const std::string& port_name);

  /**
   * @brief This function verifies the graph topology is supported by the GPU-resident execution.
   * Currently, it checks if the graph is a linear chain of operators.
   *
   * @param graph The operator graph
   * @return True if the graph topology is supported by the GPU-resident execution, false otherwise
   */
  virtual bool verify_graph_topology(
      std::shared_ptr<OperatorGraph> graph,
      std::vector<std::shared_ptr<Operator>>& topo_ordered_operators);

  void timeout_ms(unsigned long long timeout_ms);

  /**
   * @brief Sends a tear down signal to the GPU-resident CUDA graph.
   */
  void tear_down();

  /**
   * @brief Indicates whether the result of a single iteration of the
   * GPU-resident CUDA graph is ready or not.
   *
   * @return true if the result is ready, false otherwise.
   */
  bool result_ready();

  /**
   * @brief This function informs GPU-resident CUDA graph that the data is ready
   * for the main workload.
   */
  void data_ready();

  /**
   * @brief Indicates whether the GPU-resident CUDA graph has been launched.
   *
   * @return true if the CUDA graph has been launched, false otherwise.
   */
  bool is_launched();

  /// Get the execution context - currently, this has no meaning for GPU-resident execution
  /// When we need to store something for execution context, we will store a pointer in the
  /// exec_context_ for a ExecutionContext object
  std::shared_ptr<ExecutionContext> execution_context() { return exec_context_; }

  std::shared_ptr<cudaStream_t> graph_capture_stream();

  std::shared_ptr<cudaStream_t> data_ready_handler_capture_stream();

  // Get the CUDA graph of the main workload. This function returns a clone of
  // the main workload graph because the original graph is owned and retained by
  // the executor. All the limitations of graph cloning apply here. Therefore, main workload
  // graphs containing memory allocation, memory free and conditional nodes are
  // not supported.
  // This is a utility helper function.
  cudaGraph_t workload_graph_clone() const;

  /**
   * @brief Get the CUDA device pointer for the data_ready signal.
   *
   * @return Pointer to the device memory location for data_ready signal.
   */
  void* data_ready_device_address();

  /**
   * @brief Get the CUDA device pointer for the result_ready signal.
   *
   * @return Pointer to the device memory location for result_ready signal.
   */
  void* result_ready_device_address();

  /**
   * @brief Get the CUDA device pointer for the tear_down signal.
   *
   * @return Pointer to the device memory location for tear_down signal.
   */
  void* tear_down_device_address();

  /**
   * @brief Register a data ready handler fragment.
   *
   * This function stores a reference to the fragment that will handle data ready events.
   *
   * @param fragment The fragment to register as the data ready handler.
   */
  void data_ready_handler(std::shared_ptr<Fragment> fragment);

  /**
   * @brief Get the registered data ready handler fragment.
   *
   * @return The data ready handler fragment, or nullptr if none is registered.
   */
  std::shared_ptr<Fragment> data_ready_handler_fragment();

 private:
  void allocate_io_device_buffer(std::shared_ptr<Operator> downstream_op,
                                 std::shared_ptr<Operator> upstream_op,
                                 const std::string& source_port, const std::string& target_port,
                                 size_t memory_block_size);
  /**
   * @brief This function creates the full GPU-resident CUDA graph. It also
   * instantiates the CUDA graph to be ready for launch.
   *
   */
  void create_gpu_resident_cuda_graph();

  void create_cuda_graph_from_operators(
      std::vector<std::shared_ptr<Operator>>& topo_ordered_operators, cudaGraph_t& graph,
      cudaStream_t capture_stream);

  /**
   * @brief This function verifies that the operator names are distinct between the main workload
   * fragment and the data ready handler fragment.
   *
   * Assumes topologically ordered operators are already created before calling this function.
   *
   * @return True if the operator names are distinct, false otherwise.
   */
  bool verify_distinct_operator_names();

  bool fragment_initialized_ = false;

  /// @brief Map of input/output port name to the device buffers
  std::unordered_map<std::string, std::shared_ptr<holoscan::utils::cuda::DeviceBuffer>>
      io_device_buffers_;
  /// @brief Vector of topologically ordered operators
  std::vector<std::shared_ptr<Operator>> topo_ordered_main_operators_;

  /// topologically ordered operators of the data ready handler fragment
  std::vector<std::shared_ptr<Operator>> topo_ordered_drh_operators_;

  std::shared_ptr<ExecutionContext> exec_context_;
  unsigned long long timeout_ms_ = 0;

  std::shared_ptr<cudaStream_t> graph_capture_stream_;
  std::shared_ptr<cudaStream_t> drh_capture_stream_;
  cudaGraph_t drh_graph_ = nullptr;       ///< The CUDA graph of the data ready handler.
  cudaGraph_t workload_graph_ = nullptr;  ///< The CUDA graph of the main workload.
  cudaGraph_t gpu_resident_graph_ =
      nullptr;  ///< The full GPU-resident CUDA graph including control flow nodes.

  std::shared_ptr<Fragment> data_ready_handler_fragment_;

  std::shared_ptr<GPUResidentDeck> gpu_resident_deck_;
};
}  // namespace holoscan

#endif  // HOLOSCAN_CORE_EXECUTORS_GPU_RESIDENT_GPU_RESIDENT_EXECUTOR_HPP
