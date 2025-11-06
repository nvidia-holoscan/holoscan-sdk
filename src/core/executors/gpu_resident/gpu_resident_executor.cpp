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

#include <cstring>
#include <memory>
#include <string>

#include "holoscan/core/app_driver.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gpu_resident_operator.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/utils/cuda/buffer.hpp"
#include "holoscan/utils/cuda/cuda_graph_utils.hpp"
#include "holoscan/utils/cuda_macros.hpp"

#include "gr_cuda_controller.cuh"

namespace holoscan {

GPUResidentExecutor::~GPUResidentExecutor() {
  // destroy the workload graph
  if (workload_graph_) {
    HOLOSCAN_CUDA_CALL_ERR_MSG(cudaGraphDestroy(workload_graph_),
                               "Failed to destroy the workload graph");
  }

  // destroy the GPU-resident graph
  if (gpu_resident_graph_) {
    HOLOSCAN_CUDA_CALL_ERR_MSG(cudaGraphDestroy(gpu_resident_graph_),
                               "Failed to destroy the GPU-resident graph");
  }
}

void GPUResidentExecutor::run([[maybe_unused]] OperatorGraph& graph) {
  HOLOSCAN_LOG_DEBUG("GPUResidentExecutor::run()");
  HOLOSCAN_LOG_WARN(
      "GPU-resident execution is asynchronous by design. Even run() is not a blocking operation.");
  run_async(graph);
}

std::future<void> GPUResidentExecutor::run_async([[maybe_unused]] OperatorGraph& graph) {
  if (!initialize_fragment()) {
    throw std::runtime_error("Failed to initialize fragment");
  }

  // create the workload graph
  create_gpu_resident_cuda_graph();
  // instantiate the gpu-resident CUDA graph
  cudaGraphExec_t gpu_resident_graph_exec;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaGraphInstantiate(&gpu_resident_graph_exec, gpu_resident_graph_, 0),
      "Failed to instantiate the GPU-resident CUDA graph");

  // call the GPU-resident deck to launch the CUDA graph asynchronously
  return gpu_resident_deck_->launch_cuda_graph(gpu_resident_graph_exec);
}

bool GPUResidentExecutor::initialize_operator(Operator* op) {
  HOLOSCAN_LOG_DEBUG("GPUResidentExecutor::initialize_operator()");
  // mark the operator as initialized from the executor point-of-view
  op->is_initialized_ = true;
  return true;
}

void GPUResidentExecutor::prepare_data_flow(std::shared_ptr<OperatorGraph> graph) {
  auto operators = graph->get_nodes();

  // For chain of operators, this is the following we will do:
  // Start from the next operator of the root operator
  // For each operator, get its upstream connections
  // allocate a single memory block for each connection according to the specified size

  auto root_node = graph->get_root_nodes()[0];

  auto current_op = root_node;
  current_op->initialize();  // initialize the operators before preparing the data flow
  while (graph->get_next_nodes(current_op).size() > 0) {
    auto next_ops = graph->get_next_nodes(current_op);
    auto next_op = next_ops[0];
    next_op->initialize();

    HOLOSCAN_LOG_INFO("Connection {} -> {}", current_op->name(), next_op->name());
    const auto& port_map = graph->get_port_map(current_op, next_op);
    if (!port_map.has_value()) {
      auto error_msg =
          fmt::format("Could not find port map for {} -> {}", current_op->name(), next_op->name());
      throw std::runtime_error(error_msg);
    }
    const auto& port_map_val = port_map.value();

    // For one-to-one connection, get the first key-value pair
    auto port_connection = port_map_val->begin();
    auto source_port = port_connection->first;  // source port name (key)
    auto destination_port =
        *(port_connection->second.begin());  // destination port name (first element from set)

    // Get memory block size from the source operator's output spec
    auto& outputs = current_op->spec()->outputs();
    size_t memory_block_size = outputs[source_port]->memory_block_size();

    // we know one to one connection
    allocate_io_device_buffer(
        current_op, next_op, source_port, destination_port, memory_block_size);
    current_op = next_op;
  }
}

void GPUResidentExecutor::allocate_io_device_buffer(std::shared_ptr<Operator> downstream_op,
                                                    std::shared_ptr<Operator> upstream_op,
                                                    const std::string& source_port,
                                                    const std::string& target_port,
                                                    size_t memory_block_size) {
  std::shared_ptr<holoscan::utils::cuda::DeviceBuffer> device_buffer =
      std::make_shared<holoscan::utils::cuda::DeviceBuffer>(memory_block_size);

  if (!downstream_op->spec() || !upstream_op->spec()) {
    throw std::runtime_error(
        fmt::format("One of the operator ({} or {}) specifications is not available",
                    downstream_op->name(),
                    upstream_op->name()));
  }
  // check if the port names already exist in the io_device_buffers_
  auto& source_port_unique_id = downstream_op->spec()->outputs()[source_port]->unique_id();
  auto& target_port_unique_id = upstream_op->spec()->inputs()[target_port]->unique_id();

  if (io_device_buffers_.find(source_port_unique_id) != io_device_buffers_.end()) {
    throw std::runtime_error(
        fmt::format("Port name {} already exists in the io_device_buffers_ map", source_port));
  }
  if (io_device_buffers_.find(target_port_unique_id) != io_device_buffers_.end()) {
    throw std::runtime_error(
        fmt::format("Port name {} already exists in the io_device_buffers_ map", target_port));
  }

  io_device_buffers_[source_port_unique_id] = device_buffer;
  io_device_buffers_[target_port_unique_id] = device_buffer;
}

void* GPUResidentExecutor::device_memory(std::shared_ptr<Operator> op,
                                         const std::string& port_name) {
  if (!op->spec()) {
    throw std::runtime_error(fmt::format("Operator ({}) spec is not available", op->name()));
  }

  auto& port_unique_id = op->spec()->input_output_unique_id(port_name);
  auto it = io_device_buffers_.find(port_unique_id);
  if (it != io_device_buffers_.end()) {
    return it->second->data();
  }
  HOLOSCAN_LOG_ERROR(
      "Port name {} of operator {} was not found in io_device_buffers_ map", port_name, op->name());
  return nullptr;
}

bool GPUResidentExecutor::verify_graph_topology(OperatorGraph& graph) {
  auto operators = graph.get_nodes();
  // Check if the graph has a cycle
  auto cycle = graph.has_cycle();
  if (cycle.size() > 0) {
    // throw error
    auto err_msg = fmt::format(
        "Fragment graph ({}) has a cycle: {}. GPU-resident execution only supports a "
        "linear chain of operators",
        fragment_->name(),
        cycle.size());
    HOLOSCAN_LOG_ERROR(err_msg);
    return false;
  }

  // get the root nodes
  auto root_nodes = graph.get_root_nodes();
  if (root_nodes.size() != 1) {
    // throw error
    auto err_msg = fmt::format(
        "Fragment graph ({}) has ({}) root operators. GPU-resident execution only supports a "
        "linear chain of operators.",
        fragment_->name(),
        root_nodes.size());
    HOLOSCAN_LOG_ERROR(err_msg);
    return false;
  }

  // Check all the nodes have exactly one downstream node
  auto current_node = root_nodes[0];
  unsigned int visited_nodes = 1;
  while (current_node) {
    topo_ordered_operators_.push_back(current_node);
    auto next_nodes = graph.get_next_nodes(current_node);
    if (next_nodes.size() > 1) {
      // throw error
      auto err_msg = fmt::format(
          "Operator ({}) has ({}) downstream operators. GPU-resident execution only supports a "
          "linear chain of operators.",
          current_node->name(),
          next_nodes.size());
      HOLOSCAN_LOG_ERROR(err_msg);
      return false;
    } else if (next_nodes.size() == 0) {
      if (visited_nodes < operators.size()) {
        // throw error
        auto err_msg = fmt::format(
            "Fragment graph ({}) has disconnected operators. GPU-resident execution only "
            "supports a linear chain of operators.",
            fragment_->name());
        HOLOSCAN_LOG_ERROR(err_msg);
        return false;
      }
      break;  // reached the leaf node
    }
    HOLOSCAN_LOG_INFO("Connection {} -> {}", current_node->name(), next_nodes[0]->name());
    visited_nodes++;
    current_node = next_nodes[0];
  }
  return true;
}

bool GPUResidentExecutor::initialize_fragment() {
  HOLOSCAN_LOG_DEBUG("GPUResidentExecutor::initialize_fragment()");

  if (fragment_initialized_) {
    HOLOSCAN_LOG_DEBUG("Fragment ({}) has already been initialized.", fragment_->name());
    return true;
  }

  auto& graph = fragment_->graph();
  if (!verify_graph_topology(graph)) {
    throw std::runtime_error("Application graph topology is not valid for GPU-resident execution.");
  }

  // initialize CUDA and set device to 0
  initialize_cuda();

  // prepare the data flow connections between operators
  prepare_data_flow(fragment_->graph_shared());

  // call the start method of the operators
  for (auto& op_node : topo_ordered_operators_) {
    op_node->start();
  }

  // create the workload graph
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGraphCreate(&workload_graph_, 0),
                                 "Failed to create the workload graph");

  // start capturing the graph as we call the compute method of the operators
  cudaStream_t capture_stream = *graph_capture_stream();
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaStreamBeginCaptureToGraph(
          capture_stream, workload_graph_, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal),
      "Failed to capture the workload graph");

  // call the compute method of the operators
  for (auto& op_node : topo_ordered_operators_) {
    HOLOSCAN_LOG_DEBUG("Processing operator: {}", op_node->name());

    // Get the execution context
    exec_context_ = std::make_shared<ExecutionContext>();
    InputContext input_context(exec_context_.get(), op_node.get());
    OutputContext output_context(exec_context_.get(), op_node.get());

    HOLOSCAN_LOG_DEBUG("Calling compute for operator: {}", op_node->name());
    op_node->compute(input_context, output_context, *exec_context_);
    HOLOSCAN_LOG_DEBUG("Successfully called compute for operator: {}", op_node->name());
  }

  // end graph capture
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamEndCapture(capture_stream, &workload_graph_),
                                 "Failed to end graph capture");

  // get the number of nodes in the workload graph
  size_t num_nodes = 0;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGraphGetNodes(workload_graph_, nullptr, &num_nodes),
                                 "Failed to get the number of nodes in the workload graph");
  HOLOSCAN_LOG_DEBUG("Number of nodes in the workload graph: {}", num_nodes);
  if (num_nodes <= 0) {
    HOLOSCAN_LOG_WARN("Workload graph of GPU-resident execution is empty.");
  }

  // call the stop method of the operators
  for (auto& op_node : topo_ordered_operators_) {
    op_node->stop();
  }

  fragment_initialized_ = true;

  return true;
}

void GPUResidentExecutor::create_gpu_resident_cuda_graph() {
  HOLOSCAN_LOG_DEBUG("GPUResidentExecutor::create_gpu_resident_cuda_graph()");

  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGraphCreate(&gpu_resident_graph_, 0),
                                 "Failed to create the GPU-resident graph");

  // Create the root while node of the cuda graph
  cudaGraphNode_t root_while_node;

  cudaGraphConditionalHandle while_node_handle;

  // create the conditional handle
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaGraphConditionalHandleCreate(
          &while_node_handle, gpu_resident_graph_, 1, cudaGraphCondAssignDefault),
      "Failed to create the root while node conditional handle");

  cudaGraphNodeParams while_node_params{};
  while_node_params.type = cudaGraphNodeTypeConditional;
  while_node_params.conditional.handle = while_node_handle;
  while_node_params.conditional.type = cudaGraphCondTypeWhile;
  // while node only has 1 (one) output array corresponding to its body graph
  while_node_params.conditional.size = 1;

  // Add the while node to the main graph
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      holoscan::utils::cuda::cudaGraphAddNodeCompat(
          &root_while_node, gpu_resident_graph_, nullptr, nullptr, 0, &while_node_params),
      "Failed to add the root while node to the GPU-resident graph");

  // get reference to the body of the while node
  auto while_body_graph = while_node_params.conditional.phGraph_out[0];

  cudaGraphConditionalHandle if_node_handle;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaGraphConditionalHandleCreate(
          &if_node_handle, gpu_resident_graph_, 0, cudaGraphCondAssignDefault),
      "Failed to create the if node conditional handle");

  // create the while controller kernel node and add it as the root node in the
  // body graph of the while node
  cudaKernelNodeParams while_controller_kernel_params{};
  // declare both block and grid dim to be all 1
  while_controller_kernel_params.blockDim = dim3(1, 1, 1);
  while_controller_kernel_params.gridDim = dim3(1, 1, 1);
  while_controller_kernel_params.sharedMemBytes = 0;
  while_controller_kernel_params.func = (void*)&while_controller;
  // Store device addresses in variables before taking their addresses
  void* data_ready_addr = gpu_resident_deck_->data_ready_device_address();
  void* result_ready_addr = gpu_resident_deck_->result_ready_device_address();
  void* tear_down_addr = gpu_resident_deck_->tear_down_device_address();

  void* while_controller_args[] = {
      &data_ready_addr, &result_ready_addr, &tear_down_addr, &while_node_handle, &if_node_handle};
  while_controller_kernel_params.kernelParams = while_controller_args;

  // add the while controller kernel node to the body graph
  cudaGraphNode_t while_controller_kernel_node;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaGraphAddKernelNode(&while_controller_kernel_node,
                             while_body_graph,
                             nullptr,
                             0,
                             &while_controller_kernel_params),
      "Failed to add the while controller kernel node to the body graph");

  // add an IF node
  cudaGraphNodeParams if_node_params{};
  if_node_params.type = cudaGraphNodeTypeConditional;
  if_node_params.conditional.handle = if_node_handle;
  if_node_params.conditional.type = cudaGraphCondTypeIf;
  if_node_params.conditional.size = 1;  // we only have if and don't have else

  // add the IF node to the body graph
  cudaGraphNode_t if_node;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      holoscan::utils::cuda::cudaGraphAddNodeCompat(
          &if_node, while_body_graph, &while_controller_kernel_node, nullptr, 1, &if_node_params),
      "Failed to add the IF node to the while body graph");

  // get the IF node's body graph
  auto if_body_graph = if_node_params.conditional.phGraph_out[0];

  // add the main workload graph to the IF node's body graph
  cudaGraphNode_t workload_graph_node;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaGraphAddChildGraphNode(&workload_graph_node, if_body_graph, nullptr, 0, workload_graph_),
      "Failed to add the main workload graph to the IF node's body graph");

  // add the while_end_marker kernel node
  cudaKernelNodeParams while_end_marker_kernel_params{};
  while_end_marker_kernel_params.blockDim = dim3(1, 1, 1);
  while_end_marker_kernel_params.gridDim = dim3(1, 1, 1);
  while_end_marker_kernel_params.sharedMemBytes = 0;
  while_end_marker_kernel_params.func = (void*)&while_end_marker;
  void* while_end_marker_args[] = {&data_ready_addr, &result_ready_addr};
  while_end_marker_kernel_params.kernelParams = while_end_marker_args;

  // add the result ready kernel node to the IF node's body graph
  cudaGraphNode_t while_end_marker_kernel_node;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaGraphAddKernelNode(&while_end_marker_kernel_node,
                             if_body_graph,
                             &workload_graph_node,
                             1,
                             &while_end_marker_kernel_params),
      "Failed to add the while end marker kernel node to the IF node's body graph");
  // the GPU-resident graph is now ready.

  // save the graph as a dot file if environment variable is set
  if (AppDriver::get_bool_env_var("HOLOSCAN_GPU_RESIDENT_SAVE_GRAPH", false)) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaGraphDebugDotPrint(gpu_resident_graph_, "holoscan_gpu_resident_graph.dot", 0),
        "Failed to save the GPU-resident graph");
  }
}

void GPUResidentExecutor::initialize_cuda() {
  HOLOSCAN_LOG_DEBUG("GPUResidentExecutor::initialize_cuda()");

  // Check how many CUDA devices are there
  // If there is more than one, then choose Device 0 with a Holoscan warning
  int gpu_count = 0;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDeviceCount(&gpu_count),
                                 "Could not get the number of CUDA devices");

  if (gpu_count > 1) {
    HOLOSCAN_LOG_WARN(
        "Found more than one CUDA device. Choosing Device 0. Setting a different device for "
        "GPU-resident execution is not yet supported.");
  }

  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaSetDevice(0), "Failed to set device to 0");
}

void GPUResidentExecutor::timeout_ms(unsigned long long timeout_ms) {
  if (!gpu_resident_deck_) {
    throw std::runtime_error(
        "GPUResidentExecutor::timeout_ms(): GPU-resident deck is not initialized/found.");
  } else if (gpu_resident_deck_->is_launched()) {
    HOLOSCAN_LOG_ERROR(
        "GPUResidentExecutor::timeout_ms(): GPU-resident CUDA workload is already launched. "
        "timeout_ms cannot be set.");
    return;
  }
  timeout_ms_ = timeout_ms;
  gpu_resident_deck_->timeout_ms(timeout_ms);
}

void GPUResidentExecutor::tear_down() {
  if (!gpu_resident_deck_) {
    throw std::runtime_error(
        "GPUResidentExecutor::tear_down(): GPU-resident deck is not initialized/found.");
  } else if (!gpu_resident_deck_->is_launched()) {
    HOLOSCAN_LOG_ERROR(
        "GPUResidentExecutor::tear_down(): GPU-resident CUDA workload is not yet launched. "
        "tear_down trigger cannot be performed.");
    return;
  }
  gpu_resident_deck_->tear_down();
}

bool GPUResidentExecutor::result_ready() {
  if (!gpu_resident_deck_) {
    throw std::runtime_error(
        "GPUResidentExecutor::result_ready(): GPU-resident deck is not initialized/found.");
  } else if (!gpu_resident_deck_->is_launched()) {
    HOLOSCAN_LOG_ERROR(
        "GPUResidentExecutor::result_ready(): GPU-resident CUDA workload is not yet launched. "
        "result_ready trigger cannot be performed.");
    return false;
  }
  return gpu_resident_deck_->result_ready();
}

void GPUResidentExecutor::data_ready() {
  if (!gpu_resident_deck_) {
    throw std::runtime_error(
        "GPUResidentExecutor::data_ready(): GPU-resident deck is not initialized/found.");
  } else if (!gpu_resident_deck_->is_launched()) {
    HOLOSCAN_LOG_ERROR(
        "GPUResidentExecutor::data_ready(): GPU-resident CUDA workload is not yet launched. "
        "data_ready trigger cannot be performed.");
    return;
  }
  gpu_resident_deck_->set_data_ready();
}

bool GPUResidentExecutor::is_launched() {
  if (!gpu_resident_deck_) {
    throw std::runtime_error(
        "GPUResidentExecutor::is_launched(): GPU-resident deck is not initialized/found.");
  }
  return gpu_resident_deck_->is_launched();
}

std::shared_ptr<cudaStream_t> GPUResidentExecutor::graph_capture_stream() {
  if (!graph_capture_stream_) {
    // Create a CUDA stream with custom deleter
    cudaStream_t* stream_ptr = new cudaStream_t();
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamCreateWithFlags(stream_ptr, cudaStreamNonBlocking),
                                   "Failed to create a non-blocking CUDA stream");

    // Create shared_ptr with custom deleter
    graph_capture_stream_ = std::shared_ptr<cudaStream_t>(stream_ptr, [](cudaStream_t* stream) {
      if (stream) {
        HOLOSCAN_CUDA_CALL_ERR_MSG(cudaStreamDestroy(*stream), "Failed to destroy CUDA stream");
        delete stream;
      }
    });
  }

  return graph_capture_stream_;
}

}  // namespace holoscan
