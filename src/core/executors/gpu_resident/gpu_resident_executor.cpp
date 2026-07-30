/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/core/app_driver.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp>
#include <holoscan/core/flow_graphs/flow_graph.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gpu_resident_operator.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/utils/cuda/buffer.hpp>
#include <holoscan/utils/cuda/cuda_graph_utils.hpp>
#include <holoscan/utils/cuda_macros.hpp>

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

  // moving the stop method here as stop() might include CUDA deallocations.
  // call the stop method of the operators
  try {
    for (auto& op_node : topo_ordered_drh_operators_) {
      op_node->stop();
    }
    for (auto& op_node : topo_ordered_main_operators_) {
      op_node->stop();
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception during operator cleanup: {}", e.what());
  }
}

void GPUResidentExecutor::run([[maybe_unused]] OperatorFlowGraph& graph) {
  HOLOSCAN_LOG_DEBUG("GPUResidentExecutor::run()");
  HOLOSCAN_LOG_WARN(
      "GPU-resident graph execution is asynchronous by design. Even run() is not a "
      "blocking operation.");
  run_async(graph);
}

std::future<void> GPUResidentExecutor::run_async([[maybe_unused]] OperatorFlowGraph& graph) {
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
  op->set_parameters();
  // mark the operator as initialized from the executor point-of-view
  op->is_initialized_ = true;
  return true;
}

void GPUResidentExecutor::set_unique_ids(const std::shared_ptr<Operator>& op) {
  if (op && op->spec()) {
    for (auto& [port_name, input_spec] : op->spec()->inputs()) {
      // check if the unique_id is already set
      if (input_spec->unique_id().empty()) {
        input_spec->set_unique_id(fmt::format("{}.{}", op->qualified_name(), port_name));
      }
    }
    for (auto& [port_name, output_spec] : op->spec()->outputs()) {
      // check if the unique_id is already set
      if (output_spec->unique_id().empty()) {
        output_spec->set_unique_id(fmt::format("{}.{}", op->qualified_name(), port_name));
      }
    }
  }
}

void GPUResidentExecutor::prepare_data_flow(
    const std::shared_ptr<OperatorFlowGraph>& graph,
    const std::vector<std::shared_ptr<Operator>>& topo_ordered_operators) {
  for (const auto& op : topo_ordered_operators) {
    op->initialize();
    // Ensure framework-level initialization ran even if the operator's initialize()
    // override did not call Operator::initialize(). initialize_base() is idempotent.
    op->initialize_base();
    set_unique_ids(op);
  }

  for (const auto& source_op : topo_ordered_operators) {
    for (const auto& dest_op : graph->get_next_nodes(source_op)) {
      HOLOSCAN_LOG_INFO("Connection {} -> {}", source_op->name(), dest_op->name());

      const auto& port_map = graph->get_port_map(source_op, dest_op);
      if (!port_map.has_value()) {
        auto error_msg =
            fmt::format("Could not find port map for {} -> {}", source_op->name(), dest_op->name());
        throw std::runtime_error(error_msg);
      }
      const auto& port_map_val = port_map.value();

      // A single GPU-resident output port may fan out to multiple downstream input ports.
      for (const auto& [source_port, destination_ports] : *port_map_val) {
        for (const auto& destination_port : destination_ports) {
          connect_ports(source_op, dest_op, source_port, destination_port);
        }
      }
    }
  }
}

void GPUResidentExecutor::connect_ports(const std::shared_ptr<Operator>& source_op,
                                        const std::shared_ptr<Operator>& dest_op,
                                        const std::string& source_port,
                                        const std::string& destination_port) {
  auto output_memory_block_size = source_op->spec()->outputs()[source_port]->memory_block_size();
  auto input_memory_block_size = dest_op->spec()->inputs()[destination_port]->memory_block_size();
  auto output_device_ptr = source_op->spec()->outputs()[source_port]->device_ptr();
  auto input_device_ptr = dest_op->spec()->inputs()[destination_port]->device_ptr();

  // The algorithm to decide between memory block and device pointer is as follows:
  // 1. if both source and destination have valid non-zero memory block size, then use the
  //    memory block size to allocate the memory for their connection
  // 2. if either source or destination has a valid non-null device pointer, then use the
  //    device pointer to connect the operators. LOG warning if the other has a valid non-zero
  //    memory block size.
  // 3. if both source and destination have valid non-null device pointers, then LOG ERROR and
  //    use the source device pointer
  // 4. if either source or destination has a valid non-zero memory block size, then use the
  //    valid memory size to allocate the memory for their connection. Log a warning if the
  //    other object has neither a valid memory block size nor a valid non-null device pointer.

  bool has_output_mem = (output_memory_block_size > 0);
  bool has_input_mem = (input_memory_block_size > 0);
  bool has_output_ptr = (output_device_ptr != nullptr);
  bool has_input_ptr = (input_device_ptr != nullptr);

  if (has_output_mem && has_input_mem) {
    // Case 1: Both have memory block sizes - allocate a shared buffer
    if (output_memory_block_size != input_memory_block_size) {
      throw std::runtime_error(
          fmt::format("Output memory block size ({}) does not match input memory block size "
                      "({}) for connection {}.{} -> {}.{}",
                      output_memory_block_size,
                      input_memory_block_size,
                      source_op->name(),
                      source_port,
                      dest_op->name(),
                      destination_port));
    }
    allocate_io_device_buffer(
        source_op, dest_op, source_port, destination_port, output_memory_block_size);
  } else if (has_output_ptr || has_input_ptr) {
    // Cases 2 & 3: At least one side has a device pointer
    if (has_output_ptr && has_input_ptr) {
      // Case 3: Both have device pointers - LOG ERROR, use the source device pointer
      HOLOSCAN_LOG_ERROR(
          "Both source ({}.{}) and destination ({}.{}) have device pointers. "
          "Using the source device pointer.",
          source_op->name(),
          source_port,
          dest_op->name(),
          destination_port);
      connect_io_device_ptr(source_op, dest_op, source_port, destination_port, output_device_ptr);
    } else {
      // Case 2: Only one side has a device pointer
      void* device_ptr = has_output_ptr ? output_device_ptr : input_device_ptr;
      if (has_output_mem || has_input_mem) {
        HOLOSCAN_LOG_WARN(
            "Using device pointer for connection {}.{} -> {}.{}, "
            "ignoring the memory block size specified on the other end.",
            source_op->name(),
            source_port,
            dest_op->name(),
            destination_port);
      }
      connect_io_device_ptr(source_op, dest_op, source_port, destination_port, device_ptr);
    }
  } else if (has_output_mem || has_input_mem) {
    // Case 4: Only one side has a memory block size, the other has nothing
    size_t mem_size = has_output_mem ? output_memory_block_size : input_memory_block_size;
    const auto& mem_op_name = has_output_mem ? source_op->name() : dest_op->name();
    const auto& no_mem_op_name = has_output_mem ? dest_op->name() : source_op->name();
    HOLOSCAN_LOG_WARN(
        "Only operator '{}' has a valid memory block size for connection {}.{} -> {}.{}. "
        "Operator '{}' has neither a valid memory block size nor a valid device pointer.",
        mem_op_name,
        source_op->name(),
        source_port,
        dest_op->name(),
        destination_port,
        no_mem_op_name);
    allocate_io_device_buffer(
        std::move(source_op), std::move(dest_op), source_port, destination_port, mem_size);
  } else {
    // Neither side has a memory block size or a device pointer
    throw std::runtime_error(
        fmt::format("Neither source ({}.{}) nor destination ({}.{}) has a valid "
                    "memory block size or device pointer for their connection.",
                    source_op->name(),
                    source_port,
                    dest_op->name(),
                    destination_port));
  }
}

void GPUResidentExecutor::allocate_io_device_buffer(const std::shared_ptr<Operator>& source_op,
                                                    const std::shared_ptr<Operator>& dest_op,
                                                    const std::string& source_port,
                                                    const std::string& target_port,
                                                    size_t memory_block_size) {
  if (memory_block_size == 0) {
    throw std::runtime_error(
        fmt::format("The memory block size must be non zero before allocating device memory for "
                    "the port {}.{}/{}.{}",
                    source_op->name(),
                    source_port,
                    dest_op->name(),
                    target_port));
  }

  if (!source_op->spec() || !dest_op->spec()) {
    throw std::runtime_error(
        fmt::format("One of the operator ({} or {}) specifications is not available",
                    source_op->name(),
                    dest_op->name()));
  }
  // check if the port names already exist in the io_device_buffers_
  auto& source_port_unique_id = source_op->spec()->outputs()[source_port]->unique_id();
  auto& target_port_unique_id = dest_op->spec()->inputs()[target_port]->unique_id();

  if (io_device_ptrs_.find(source_port_unique_id) != io_device_ptrs_.end()) {
    throw std::runtime_error(fmt::format(
        "Source port {}.{} is already connected through externally managed device pointers",
        source_op->name(),
        source_port));
  }
  if (io_device_buffers_.find(target_port_unique_id) != io_device_buffers_.end() ||
      io_device_ptrs_.find(target_port_unique_id) != io_device_ptrs_.end()) {
    throw std::runtime_error(
        fmt::format("Internal invariant violated: destination port {}.{} was already connected. "
                    "Fragment::add_flow() should reject multiple upstream "
                    "connections to the same GPU-resident input port.",
                    dest_op->name(),
                    target_port));
  }

  auto source_buffer_it = io_device_buffers_.find(source_port_unique_id);
  if (source_buffer_it != io_device_buffers_.end()) {
    if (source_buffer_it->second->get_bytes() != memory_block_size) {
      throw std::runtime_error(fmt::format(
          "Existing buffer size ({}) for connection source port {}.{} does not match requested "
          "size ({})",
          source_buffer_it->second->get_bytes(),
          source_op->name(),
          source_port,
          memory_block_size));
    }
    io_device_buffers_[target_port_unique_id] = source_buffer_it->second;
    return;
  }

  std::shared_ptr<holoscan::utils::cuda::DeviceBuffer> device_buffer =
      std::make_shared<holoscan::utils::cuda::DeviceBuffer>(memory_block_size);

  io_device_buffers_[source_port_unique_id] = device_buffer;
  io_device_buffers_[target_port_unique_id] = std::move(device_buffer);
}

void GPUResidentExecutor::connect_io_device_ptr(const std::shared_ptr<Operator>& source_op,
                                                const std::shared_ptr<Operator>& dest_op,
                                                const std::string& source_port,
                                                const std::string& target_port, void* device_ptr) {
  if (device_ptr == nullptr) {
    throw std::runtime_error(
        fmt::format("The device pointer must be non-null for connecting ports {}.{} -> {}.{}",
                    source_op->name(),
                    source_port,
                    dest_op->name(),
                    target_port));
  }

  // Check if the device pointer is a valid one using CUDA API
  cudaPointerAttributes ptr_attr;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaPointerGetAttributes(&ptr_attr, device_ptr),
                                 "Failed to get the pointer attributes");
  if (ptr_attr.type != cudaMemoryTypeDevice) {
    throw std::runtime_error(
        fmt::format("Not a valid device memory pointer (i.e., not cudaMemoryTypeDevice) for "
                    "connecting ports {}.{} -> {}.{}",
                    source_op->name(),
                    source_port,
                    dest_op->name(),
                    target_port));
  }

  if (!source_op->spec() || !dest_op->spec()) {
    throw std::runtime_error(
        fmt::format("One of the operator ({} or {}) specifications is not available",
                    source_op->name(),
                    dest_op->name()));
  }

  auto& source_port_unique_id = source_op->spec()->outputs()[source_port]->unique_id();
  auto& target_port_unique_id = dest_op->spec()->inputs()[target_port]->unique_id();

  if (io_device_buffers_.find(source_port_unique_id) != io_device_buffers_.end()) {
    throw std::runtime_error(fmt::format(
        "Source port {}.{} is already connected through executor-allocated device buffers",
        source_op->name(),
        source_port));
  }
  if (io_device_buffers_.find(target_port_unique_id) != io_device_buffers_.end() ||
      io_device_ptrs_.find(target_port_unique_id) != io_device_ptrs_.end()) {
    throw std::runtime_error(
        fmt::format("Internal invariant violated: destination port {}.{} was already connected. "
                    "Fragment::add_flow() should reject multiple upstream "
                    "connections to the same GPU-resident input port.",
                    dest_op->name(),
                    target_port));
  }

  auto source_ptr_it = io_device_ptrs_.find(source_port_unique_id);
  if (source_ptr_it != io_device_ptrs_.end()) {
    if (source_ptr_it->second != device_ptr) {
      throw std::runtime_error(fmt::format(
          "Source port {}.{} is already connected to a different externally managed device pointer",
          source_op->name(),
          source_port));
    }
    io_device_ptrs_[target_port_unique_id] = source_ptr_it->second;
    return;
  }

  io_device_ptrs_[source_port_unique_id] = device_ptr;
  io_device_ptrs_[target_port_unique_id] = device_ptr;
}

void* GPUResidentExecutor::device_memory(const std::shared_ptr<Operator>& op,
                                         const std::string& port_name) {
  if (!op->spec()) {
    throw std::runtime_error(fmt::format("Operator ({}) spec is not available", op->name()));
  }

  auto& port_unique_id = op->spec()->input_output_unique_id(port_name);

  // Check executor-allocated device buffers first
  auto it = io_device_buffers_.find(port_unique_id);
  if (it != io_device_buffers_.end()) {
    return it->second->data();
  }

  // Check externally-owned device pointer connections
  auto ptr_it = io_device_ptrs_.find(port_unique_id);
  if (ptr_it != io_device_ptrs_.end()) {
    return ptr_it->second;
  }

  HOLOSCAN_LOG_ERROR(
      "Port name {} of operator {} was not found in the executor-allocated device buffer map or "
      "the externally managed device pointer map",
      port_name,
      op->name());
  return nullptr;
}

bool GPUResidentExecutor::verify_graph_topology(
    const std::shared_ptr<OperatorFlowGraph>& graph,
    std::vector<std::shared_ptr<Operator>>& topo_ordered_operators) {
  topo_ordered_operators.clear();

  auto operators = graph->get_nodes();
  // Collect the root nodes up front so we can reject unsupported multi-root topologies early.
  auto root_nodes = graph->get_root_nodes();
  if (root_nodes.size() > 1) {
    auto err_msg = fmt::format(
        "Fragment graph has ({}) root operators. GPU-resident graph execution only supports DAGs "
        "with a single source operator.",
        root_nodes.size());
    HOLOSCAN_LOG_ERROR(err_msg);
    return false;
  }

  auto cyclic_roots = graph->has_cycle();
  if (!cyclic_roots.empty()) {
    std::vector<std::string> names;
    names.reserve(cyclic_roots.size());
    for (const auto& node : cyclic_roots) {
      names.push_back(node->name());
    }
    auto err_msg = fmt::format(
        "Fragment graph has a cycle (root nodes of cycle: {}). GPU-resident graph execution only "
        "supports DAGs.",
        fmt::join(names, ", "));
    HOLOSCAN_LOG_ERROR(err_msg);
    return false;
  }

  // at this point, the graph is a single-source DAG
  // topological ordering is straightforward

  std::deque<std::shared_ptr<Operator>> worklist;
  std::unordered_map<std::shared_ptr<Operator>, size_t> indegrees;
  indegrees.reserve(operators.size());

  for (const auto& op : operators) {
    indegrees[op] = graph->get_previous_nodes(op).size();
    if (indegrees[op] == 0) {
      worklist.push_back(op);
    }
  }

  topo_ordered_operators.reserve(operators.size());
  while (!worklist.empty()) {
    auto current_node = worklist.front();
    worklist.pop_front();
    topo_ordered_operators.push_back(current_node);

    for (const auto& next_node : graph->get_next_nodes(current_node)) {
      auto indegree_it = indegrees.find(next_node);
      if (indegree_it == indegrees.end()) {
        auto err_msg =
            fmt::format("Operator ({}) was not found in the indegree map.", next_node->name());
        HOLOSCAN_LOG_ERROR(err_msg);
        return false;
      }
      indegree_it->second--;
      if (indegree_it->second == 0) {
        worklist.push_back(next_node);
      }
    }
  }

  return true;
}

void GPUResidentExecutor::create_cuda_graph_from_operators(
    std::vector<std::shared_ptr<Operator>>& topo_ordered_operators, cudaGraph_t& graph,
    cudaStream_t capture_stream) {
  HOLOSCAN_LOG_DEBUG("GPUResidentExecutor::create_cuda_graph_from_operators()");

  // create the graph
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGraphCreate(&graph, 0), "Failed to create the graph");

  // start capturing the graph as we call the compute method of the operators
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaStreamBeginCaptureToGraph(
          capture_stream, graph, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal),
      "Failed to capture the workload graph");

  // call the compute method of the operators in topological order
  for (auto& op_node : topo_ordered_operators) {
    HOLOSCAN_LOG_DEBUG("Processing operator: {}", op_node->name());

    // Keep the currently captured execution context on the executor so helper paths that fetch
    // GPUResidentExecutor::execution_context() observe the active operator context.
    exec_context_ = std::make_shared<ExecutionContext>();
    InputContext input_context(exec_context_.get(), op_node.get());
    OutputContext output_context(exec_context_.get(), op_node.get());

    HOLOSCAN_LOG_DEBUG("Calling compute for operator: {}", op_node->name());
    op_node->compute(input_context, output_context, *exec_context_);
    HOLOSCAN_LOG_DEBUG("Successfully called compute for operator: {}", op_node->name());
  }

  // end graph capture
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamEndCapture(capture_stream, &graph),
                                 "Failed to end graph capture");

  // get the number of nodes in the graph
  size_t num_nodes = 0;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGraphGetNodes(graph, nullptr, &num_nodes),
                                 "Failed to get the number of nodes in the workload graph");
  HOLOSCAN_LOG_DEBUG("Number of nodes in the graph: {}", num_nodes);
  if (num_nodes <= 0) {
    HOLOSCAN_LOG_WARN("GPU-resident graph is empty.");
  } else {
    // save the main workload graph as a dot file
    if (AppDriver::get_bool_env_var("HOLOSCAN_GPU_RESIDENT_SAVE_GRAPH", false)) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(
          cudaGraphDebugDotPrint(
              graph, "holoscan_gpu_resident_workload_graph.dot", cudaGraphDebugDotFlagsVerbose),
          "Failed to save the main workload graph");
    }
  }
}

bool GPUResidentExecutor::initialize_fragment() {
  HOLOSCAN_LOG_DEBUG("GPUResidentExecutor::initialize_fragment()");

  if (fragment_initialized_) {
    HOLOSCAN_LOG_DEBUG("Main workload fragment ({}) has already been initialized.",
                       fragment_->name());
    if (data_ready_handler_fragment_) {
      HOLOSCAN_LOG_DEBUG("Data ready handler fragment ({}) has already been initialized.",
                         data_ready_handler_fragment_->name());
    }
    return true;
  }

  if (data_ready_handler_fragment_) {
    auto drh_fragment_graph = data_ready_handler_fragment_->graph_shared();
    if (!verify_graph_topology(std::move(drh_fragment_graph), topo_ordered_drh_operators_)) {
      throw std::runtime_error(
          "Data ready handler graph topology is not valid for GPU-resident graph execution.");
    }
  }

  auto main_fragment_graph = fragment_->graph_shared();
  if (!verify_graph_topology(std::move(main_fragment_graph), topo_ordered_main_operators_)) {
    throw std::runtime_error(
        "Application graph topology is not valid for GPU-resident graph execution.");
  }

  if (data_ready_handler_fragment_ && !verify_distinct_operator_names()) {
    auto err_msg = fmt::format(
        "There are duplicate operator names between the main workload fragment ('{}') and the data "
        "ready handler fragment ('{}'). Operator names must be distinct between the two fragments.",
        fragment_->name(),
        data_ready_handler_fragment_->name());
    throw std::runtime_error(err_msg);
  }

  // initialize CUDA and set device to 0
  initialize_cuda();

  // prepare the data flow connections between operators using the flattened DAG ordering
  if (data_ready_handler_fragment_) {
    prepare_data_flow(data_ready_handler_fragment_->graph_shared(), topo_ordered_drh_operators_);
  }
  prepare_data_flow(fragment_->graph_shared(), topo_ordered_main_operators_);

  // call the start method of the operators
  for (auto& op_node : topo_ordered_drh_operators_) {
    op_node->start();
  }
  for (auto& op_node : topo_ordered_main_operators_) {
    op_node->start();
  }

  if (data_ready_handler_fragment_) {
    create_cuda_graph_from_operators(
        topo_ordered_drh_operators_, drh_graph_, *data_ready_handler_capture_stream());
  }
  create_cuda_graph_from_operators(
      topo_ordered_main_operators_, workload_graph_, *graph_capture_stream());

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

  // used to decide the root node of the while body graph
  cudaGraphNode_t whilebody_root_node = nullptr;

  // used to decide the intended parent node of the while controller kernel node
  cudaGraphNode_t whilecontroller_parent_node = nullptr;
  if (perf_enabled_) {
    // create a kernel node for the start_perf_timer kernel
    cudaKernelNodeParams start_perf_timer_kernel_params{};
    start_perf_timer_kernel_params.blockDim = dim3(1, 1, 1);
    start_perf_timer_kernel_params.gridDim = dim3(1, 1, 1);
    start_perf_timer_kernel_params.sharedMemBytes = 0;
    start_perf_timer_kernel_params.func = (void*)&start_perf_timer;
    void* start_time_ns_addr = start_time_ns_dev_->data();
    void* start_perf_timer_args[] = {&start_time_ns_addr};
    start_perf_timer_kernel_params.kernelParams = start_perf_timer_args;

    cudaGraphNode_t start_perf_timer_kernel_node;
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaGraphAddKernelNode(&start_perf_timer_kernel_node,
                               while_body_graph,
                               nullptr,
                               0,
                               &start_perf_timer_kernel_params),
        "Failed to add the start perf timer kernel node to the body graph");
    whilebody_root_node = start_perf_timer_kernel_node;
    whilecontroller_parent_node = start_perf_timer_kernel_node;
  }

  cudaGraphNode_t drh_graph_node = nullptr;
  if (data_ready_handler_fragment_) {
    // if there is a data ready handler fragment, then add the drh_graph_ to the
    // while_body_graph
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaGraphAddChildGraphNode(&drh_graph_node,
                                   while_body_graph,
                                   (whilebody_root_node ? &whilebody_root_node : nullptr),
                                   (whilebody_root_node ? 1 : 0),
                                   drh_graph_),
        "Failed to add the data ready handler graph to the while body graph");
    if (!whilebody_root_node) {  // no start_perf_timer kernel node added
      // set data ready handler graph as the root node of the while body graph
      whilebody_root_node = drh_graph_node;
    }
    // since there is data ready handler graph, the while controller kernel's parent node will be
    // the data ready handler graph.
    whilecontroller_parent_node = drh_graph_node;
  }
  // create the while controller kernel node and add it as the root node in the
  // body graph of the while node
  bool enable_debug_prints = (log_level() == LogLevel::DEBUG);
  cudaKernelNodeParams while_controller_kernel_params{};
  // declare both block and grid dim to be all 1
  while_controller_kernel_params.blockDim = dim3(1, 1, 1);
  while_controller_kernel_params.gridDim = dim3(1, 1, 1);
  while_controller_kernel_params.sharedMemBytes = 0;
  while_controller_kernel_params.func =
      enable_debug_prints ? (void*)&while_controller_debug : (void*)&while_controller;
  // Store device addresses in variables before taking their addresses
  void* data_ready_addr = gpu_resident_deck_->data_ready_device_address();
  void* result_ready_addr = gpu_resident_deck_->result_ready_device_address();
  void* tear_down_addr = gpu_resident_deck_->tear_down_device_address();
  unsigned int sleep_interval_us = data_not_ready_sleep_interval_us_;

  void* while_controller_args[] = {&data_ready_addr,
                                   &result_ready_addr,
                                   &tear_down_addr,
                                   &sleep_interval_us,
                                   &while_node_handle,
                                   &if_node_handle};
  while_controller_kernel_params.kernelParams = while_controller_args;

  // add the while controller kernel node to the WHILE body graph
  // drh node will be a dependency of this kernel, only if drh_graph_node was added, otherwise, the
  // while controller kernel will be the root of the WHILE body graph
  cudaGraphNode_t while_controller_kernel_node;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaGraphAddKernelNode(&while_controller_kernel_node,
                             while_body_graph,
                             (whilecontroller_parent_node ? &whilecontroller_parent_node : nullptr),
                             (whilecontroller_parent_node ? 1 : 0),
                             &while_controller_kernel_params),
      "Failed to add the while controller kernel node to the body graph");

  if (!whilebody_root_node) {  // no while body root node was added, therefore, the while controller
                               // kernel node is the root node of the while body graph
    whilebody_root_node = while_controller_kernel_node;
  }
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
  while_end_marker_kernel_params.func =
      enable_debug_prints ? (void*)&while_end_marker_debug : (void*)&while_end_marker;
  void* start_time_ns_addr = perf_enabled_ ? start_time_ns_dev_->data() : nullptr;
  void* execution_times_us_addr = perf_enabled_ ? execution_times_us_dev_->data() : nullptr;
  void* actual_samples_collected_addr =
      perf_enabled_ ? actual_samples_collected_dev_->data() : nullptr;
  void* while_end_marker_args[] = {&data_ready_addr,
                                   &result_ready_addr,
                                   &execution_times_us_addr,
                                   &num_samples_,
                                   &start_time_ns_addr,
                                   &actual_samples_collected_addr,
                                   &sync_with_host_};
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
        "GPU-resident graph execution is not yet supported.");
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

void GPUResidentExecutor::sync_with_host(bool enable) {
  if (!gpu_resident_deck_) {
    auto err_msg = fmt::format(
        "GPUResidentExecutor::{}(): GPU-resident deck is not "
        "initialized/found.",
        __func__);
    throw std::runtime_error(err_msg);
  } else if (gpu_resident_deck_->is_launched()) {
    HOLOSCAN_LOG_ERROR(
        "GPUResidentExecutor::{}(): GPU-resident CUDA workload is already launched. "
        "{} cannot be set anymore.",
        __func__,
        __func__);
    return;
  }
  sync_with_host_ = enable;
}

void GPUResidentExecutor::data_not_ready_sleep_interval_us(unsigned int sleep_interval_us) {
  if (!gpu_resident_deck_) {
    auto err_msg = fmt::format(
        "GPUResidentExecutor::{}(): GPU-resident deck is not "
        "initialized/found.",
        __func__);
    throw std::runtime_error(err_msg);
  } else if (gpu_resident_deck_->is_launched()) {
    HOLOSCAN_LOG_ERROR(
        "GPUResidentExecutor::{}(): GPU-resident CUDA workload is already launched. "
        "{} cannot be set anymore.",
        __func__,
        __func__);
    return;
  }
  data_not_ready_sleep_interval_us_ = sleep_interval_us;
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

std::shared_ptr<cudaStream_t> GPUResidentExecutor::data_ready_handler_capture_stream() {
  if (!drh_capture_stream_) {
    // Create a CUDA stream with custom deleter
    cudaStream_t* stream_ptr = new cudaStream_t();
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamCreateWithFlags(stream_ptr, cudaStreamNonBlocking),
                                   "Failed to create a non-blocking CUDA stream");

    drh_capture_stream_ = std::shared_ptr<cudaStream_t>(stream_ptr, [](cudaStream_t* stream) {
      if (stream) {
        HOLOSCAN_CUDA_CALL_ERR_MSG(cudaStreamDestroy(*stream), "Failed to destroy CUDA stream");
        delete stream;
      }
    });
  }
  return drh_capture_stream_;
}

cudaGraph_t GPUResidentExecutor::workload_graph_clone() const {
  cudaGraph_t workload_graph_clone;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGraphClone(&workload_graph_clone, workload_graph_),
                                 "Failed to clone the workload graph");
  return workload_graph_clone;
}

void* GPUResidentExecutor::data_ready_device_address() {
  if (!gpu_resident_deck_) {
    throw std::runtime_error(
        "GPUResidentExecutor::data_ready_device_address(): GPU-resident deck is not "
        "initialized/found.");
  }
  return gpu_resident_deck_->data_ready_device_address();
}

void* GPUResidentExecutor::result_ready_device_address() {
  if (!gpu_resident_deck_) {
    throw std::runtime_error(
        "GPUResidentExecutor::result_ready_device_address(): GPU-resident deck is not "
        "initialized/found.");
  }
  return gpu_resident_deck_->result_ready_device_address();
}

void* GPUResidentExecutor::tear_down_device_address() {
  if (!gpu_resident_deck_) {
    throw std::runtime_error(
        "GPUResidentExecutor::tear_down_device_address(): GPU-resident deck is not "
        "initialized/found.");
  }
  return gpu_resident_deck_->tear_down_device_address();
}

void GPUResidentExecutor::data_ready_handler(std::shared_ptr<Fragment> fragment) {
  if (data_ready_handler_fragment_) {
    HOLOSCAN_LOG_WARN(
        "There is already a data ready handler fragment registered. Overwriting it with the new "
        "one.");
  }
  data_ready_handler_fragment_ = std::move(fragment);
  // Set the executor for the data ready handler fragment so that operators in that fragment
  // can access the same GPU resident executor
  if (data_ready_handler_fragment_) {
    data_ready_handler_fragment_->executor(fragment_->executor_shared());
    // call this function so that the data ready handler capture
    // stream is already created when it's required.
    data_ready_handler_capture_stream();
    // also compose the data ready handler fragment as a fail-safe mechanism
    data_ready_handler_fragment_->compose_graph();

    // check if it is a GPU-resident fragment here or throw error
    if (!data_ready_handler_fragment_->is_gpu_resident()) {
      auto err_msg = fmt::format(
          "Data ready handler fragment '{}' is not a GPU-resident fragment. It cannot be "
          "registered as a data ready handler for GPU-resident fragment '{}'.",
          data_ready_handler_fragment_->name(),
          fragment_->name());
      throw std::runtime_error(err_msg);
    }
  }
}

std::shared_ptr<Fragment> GPUResidentExecutor::data_ready_handler_fragment() {
  return data_ready_handler_fragment_;
}

bool GPUResidentExecutor::verify_distinct_operator_names() {
  if (topo_ordered_drh_operators_.size() == 0) {
    HOLOSCAN_LOG_DEBUG(
        "Data ready handler fragment has no operators. No need to verify operator names.");
    return true;
  }

  if (topo_ordered_main_operators_.size() == 0) {
    HOLOSCAN_LOG_WARN("Main workload fragment has no operators. No need to verify operator names.");
    return true;
  }

  const auto& smaller_list =
      (topo_ordered_drh_operators_.size() <= topo_ordered_main_operators_.size())
          ? topo_ordered_drh_operators_
          : topo_ordered_main_operators_;
  const auto& larger_list = (&smaller_list == &topo_ordered_drh_operators_)
                                ? topo_ordered_main_operators_
                                : topo_ordered_drh_operators_;

  std::unordered_set<std::string> name_set;
  // reserve extra capacity to minimize rehashing
  name_set.reserve(smaller_list.size() * 2);

  for (const auto& op_ptr : smaller_list) {
    name_set.insert(op_ptr->name());
  }

  for (const auto& op_ptr : larger_list) {
    if (name_set.find(op_ptr->name()) != name_set.end()) {
      HOLOSCAN_LOG_ERROR(
          "Operator name '{}' appears in both the main workload fragment ('{}') and the data ready "
          "handler fragment ('{}').",
          op_ptr->name(),
          fragment_->name(),
          data_ready_handler_fragment_->name());
      return false;
    }
  }

  return true;
}

void GPUResidentExecutor::enable_perf_measurement(unsigned int num_samples) {
  if (!num_samples) {
    throw std::runtime_error(
        "Number of samples for GPU-resident performance measurement cannot be 0");
  }
  perf_enabled_ = true;
  num_samples_ = num_samples;
  execution_times_us_dev_ =
      std::make_shared<holoscan::utils::cuda::DeviceBuffer>(sizeof(unsigned int) * num_samples);
  start_time_ns_dev_ =
      std::make_shared<holoscan::utils::cuda::DeviceBuffer>(sizeof(unsigned long long));
  // Allocate buffer for actual samples collected counter and initialize to 0
  actual_samples_collected_dev_ =
      std::make_shared<holoscan::utils::cuda::DeviceBuffer>(sizeof(unsigned int));
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemset(actual_samples_collected_dev_->data(), 0, sizeof(unsigned int)),
      "Failed to initialize actual_samples_collected device buffer to 0");
}

std::pair<unsigned int*, unsigned int> GPUResidentExecutor::execution_times_us() {
  if (perf_enabled_) {
    // Get the actual number of samples collected from device memory
    unsigned int actual_samples =
        *static_cast<unsigned int*>(actual_samples_collected_dev_->host_data());
    // synchronize the default stream
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(0),
                                   "Failed to synchronize the default stream");
    auto result = std::make_pair(static_cast<unsigned int*>(execution_times_us_dev_->host_data()),
                                 actual_samples);
    // synchronize the default stream
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(0),
                                   "Failed to synchronize the default stream");
    return result;
  } else {
    HOLOSCAN_LOG_ERROR("Performance measurement is not enabled for GPU-resident graph execution.");
    return std::make_pair(nullptr, 0);
  }
}

}  // namespace holoscan
