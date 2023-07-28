/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_EXECUTORS_EXECUTOR_HPP
#define HOLOSCAN_CORE_EXECUTORS_EXECUTOR_HPP

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "./common.hpp"
#include "./extension_manager.hpp"
#include "./graph.hpp"
#include "./operator.hpp"

namespace holoscan {

/**
 * @brief Base class for all executors.
 *
 * An Executor that manages the execution of a Fragment on a physical node.
 * The framework provides a default Executor that uses a GXF Scheduler to execute an Application.
 */
class Executor {
 public:
  Executor() = delete;
  /**
   * @brief Construct a new Executor object.
   *
   * @param fragment The pointer to the fragment of the executor.
   */
  explicit Executor(Fragment* fragment) : fragment_(fragment) {}
  virtual ~Executor() = default;

  /**
   * @brief Run the graph.
   *
   * @param graph The reference to the graph.
   */
  virtual void run(OperatorGraph& graph) { (void)graph; }

  /**
   * @brief Run the graph asynchronously.
   *
   * @param graph The reference to the graph.
   * @return The future object.
   */
  virtual std::future<void> run_async(OperatorGraph& graph) {
    (void)graph;
    return {};
  }

  /**
   * @brief Interrupt the execution.
   */
  virtual void interrupt() {}

  /**
   * @brief Set the pointer to the fragment of the executor.
   *
   * @param fragment The pointer to the fragment of the executor.
   */
  void fragment(Fragment* fragment) { fragment_ = fragment; }

  /**
   * @brief Get a pointer to Fragment object.
   *
   * @return The Pointer to Fragment object.
   */
  Fragment* fragment() { return fragment_; }

  /**
   * @brief Set the context.
   *
   * @param context The context.
   */
  virtual void context(void* context) { context_ = context; }
  /**
   * @brief Get the context
   *
   * @return The context.
   */
  void* context() { return context_; }

  // add uint64_t context getters/setters for Python API
  void context_uint64(uint64_t context) { context_ = reinterpret_cast<void*>(context); }
  uint64_t context_uint64() { return reinterpret_cast<uint64_t>(context_); }

  /**
   * @brief Get the extension manager.
   * @return The shared pointer of the extension manager.
   */
  virtual std::shared_ptr<ExtensionManager> extension_manager() { return extension_manager_; }

 protected:
  friend class Fragment;        // make Fragment a friend class to access protected members of
                                // Executor (add_receivers()).
  friend class Operator;        // make Operator a friend class to access protected members of
                                // Executor (initialize_operator()).
  friend class Scheduler;       // make Scheduler a friend class to access protected members of
                                // Executor (initialize_scheduler()).
  friend class NetworkContext;  // make NetworkContext a friend class to access protected members
                                // of Executor (initialize_network_context()).

  /**
   * @brief Initialize the fragment_ in this Executor.
   *
   * This method is called by run() to initialize the fragment and the graph of operators in the
   * fragment before execution.
   *
   * @return true if fragment initialization is successful. Otherwise, false.
   */
  virtual bool initialize_fragment() { return false; }
  /**
   * @brief Initialize the given operator.
   *
   * This method is called by Operator::initialize() to initialize the operator.
   *
   * Depending on the type of the operator, this method may be overridden to initialize the
   * operator. For example, the default executor (GXFExecutor) initializes the operator using the
   * GXF API and sets the operator's ID to the ID of the GXF codelet.
   *
   * @param op The pointer to the operator.
   * @return true if the operator is initialized successfully. Otherwise, false.
   */
  virtual bool initialize_operator(Operator* op) {
    (void)op;
    return false;
  }

  /**
   * @brief Initialize the given scheduler.
   *
   * This method is called by Scheduler::initialize() to initialize the operator.
   *
   * Depending on the type of the scheduler, this method may be overridden to initialize the
   * scheduler. For example, the default executor (GXFExecutor) initializes the scheduler using the
   * GXF API and sets the operator's ID to the ID of the GXF scheduler.
   *
   * @param sch The pointer to the scheduler.
   * @return true if the scheduler is initialized successfully. Otherwise, false.
   */
  virtual bool initialize_scheduler(Scheduler* sch) {
    (void)sch;
    return false;
  }

  /**
   * @brief Initialize the given network context.
   *
   * This method is called by NetworkContext::initialize() to initialize the operator.
   *
   * Depending on the type of the network context, this method may be overridden to initialize the
   * network context. For example, the default executor (GXFExecutor) initializes the network
   * context using the GXF API and sets the operator's ID to the ID of the GXF network context.
   *
   * @param network_context The pointer to the network context.
   * @return true if the network context is initialized successfully. Otherwise, false.
   */
  virtual bool initialize_network_context(NetworkContext* network_context) {
    (void)network_context;
    return false;
  }

  /**
   * @brief Add the receivers as input ports of the given operator.
   *
   * This method is to be called by the Fragment::add_flow() method to support for the case where
   * the destination input port label points to the parameter name of the downstream operator, and
   * the parameter type is 'std::vector<holoscan::IOSpec*>'. This finds a parameter with with
   * 'std::vector<holoscan::IOSpec*>' type and create a new input port with a specific label
   * ('<parameter name>:<index>'. e.g, 'receivers:0').
   *
   * @param op The reference to the shared pointer of the operator.
   * @param receivers_name The name of the receivers whose parameter type is
   * 'std::vector<holoscan::IOSpec*>'.
   * @param new_input_labels The reference to the vector of input port labels to which the input
   * port labels are added. In the case of multiple receivers, the input port label is updated to
   * '<parameter name>:<index>' (e.g. 'receivers' => 'receivers:<index>').
   * @param iospec_vector The reference to the vector of IOSpec pointers.
   * @return true if the receivers are added successfully. Otherwise, false.
   */
  virtual bool add_receivers(const std::shared_ptr<Operator>& op, const std::string& receivers_name,
                             std::vector<std::string>& new_input_labels,
                             std::vector<holoscan::IOSpec*>& iospec_vector) {
    (void)op;
    (void)receivers_name;
    (void)new_input_labels;
    (void)iospec_vector;
    return false;
  }

  Fragment* fragment_ = nullptr;                         ///< The fragment of the executor.
  void* context_ = nullptr;                              ///< The context.
  std::shared_ptr<ExtensionManager> extension_manager_;  ///< The extension manager.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_EXECUTORS_EXECUTOR_HPP */
