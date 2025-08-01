/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_SYSTEM_RESOURCES_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_SYSTEM_RESOURCES_HPP

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gxf/std/resources.hpp>
#include "../../component_spec.hpp"
#include "../../gxf/entity_group.hpp"
#include "../../gxf/gxf_resource.hpp"
#include "../../operator.hpp"
#include "cpu_thread.hpp"

namespace holoscan {

/**
 * @brief Thread pool resource.
 *
 * This is a thread pool for use with the EventBasedScheduler or MultiThreadScheduler. This
 * resource should be created via the `Fragment::make_thread_pool` method instead of the usual
 * `Fragment::make_resource` method as it requires additional configuration of an associated
 * GXF EntityGroup.
 *
 * pool1 = make_thread_pool("pool1", Arg("initial_size", static_cast<int64_t>(2)));
 *
 * The operators can be added via the `add` method. For strict thread pinning, the `pin_operator`
 * argument should be true and the initial_size of the thread pool should be at least as large as
 * the number of operators that will be pinned to threads.
 *
 * pool1.add(op1, true);
 * pool1.add(op2, true);
 *
 * This add method takes care of adding any needed holoscan::CPUThread resource to the operator.
 *
 * The MultiThreadScheduler's `strict_job_thread_pinning` argument can be set true to disallow
 * execution of any other entities on the pinned thread. The EventBasedScheduler always uses strict
 * thread pinning.
 *
 */
class ThreadPool : public gxf::GXFSystemResourceBase {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(ThreadPool, gxf::GXFSystemResourceBase)
  ThreadPool() = default;
  ThreadPool(const std::string& name, nvidia::gxf::ThreadPool* component);

  /// @brief The underlying GXF component's name.
  const char* gxf_typename() const override { return "nvidia::gxf::ThreadPool"; }

  void setup(ComponentSpec& spec) override;

  /// @brief The number of threads currently in the thread pool.
  int64_t size() const;

  /**
   * @brief Add an operator to the thread pool
   *
   * @param op The operator to add.
   * @param pin_operator Whether the operator should be pinned to a specific thread in the pool.
   * @param pin_cores CPU core IDs to pin the worker threads to (empty means no core pinning).
   */
  void add(const std::shared_ptr<Operator>& op, bool pin_operator = true,
           std::vector<uint32_t> pin_cores = std::vector<uint32_t>());

  /**
   * @brief Add multiple operators to the thread pool
   *
   * @param ops The operators to add.
   * @param pin_operator Whether the operators should be pinned to a specific thread in the pool.
   * @param pin_cores CPU core IDs to pin the worker thread to (empty means no core pinning).
   */
  void add(std::vector<std::shared_ptr<Operator>> ops, bool pin_operator = true,
           std::vector<uint32_t> pin_cores = std::vector<uint32_t>());

  /**
   * @brief Add an operator to the thread pool with real-time scheduling capabilities
   *
   * @param op The operator to add.
   * @param sched_policy Real-time scheduling policy (kFirstInFirstOut, kRoundRobin, kDeadline).
   * @param pin_operator Whether the operator should be pinned to a specific thread in the pool.
   * @param pin_cores CPU core IDs to pin the worker thread to (empty means no core pinning).
   * @param sched_priority Thread priority for FirstInFirstOut and RoundRobin policies.
   * @param sched_runtime Expected worst case execution time in nanoseconds for Deadline policy.
   * @param sched_deadline Relative deadline in nanoseconds for Deadline policy.
   * @param sched_period Period in nanoseconds for Deadline policy.
   */
  void add_realtime(const std::shared_ptr<Operator>& op, SchedulingPolicy sched_policy,
                    bool pin_operator = true,
                    std::vector<uint32_t> pin_cores = std::vector<uint32_t>(),
                    uint32_t sched_priority = 0, uint64_t sched_runtime = 0,
                    uint64_t sched_deadline = 0, uint64_t sched_period = 0);

  // The entity group that this thread pool is associated with.
  std::shared_ptr<gxf::EntityGroup> entity_group() const { return entity_group_; }

  /// @brief The operators associated with this thread pool.
  std::vector<std::shared_ptr<Operator>> operators() const { return operators_; }

  /**
   * @brief Get a YAML representation of the thread pool.
   *
   * @return YAML node including properties of the base resource and the operators in the pool.
   */
  YAML::Node to_yaml_node() const override;

 protected:
  friend class Fragment;  // allow Fragment::make_thread_pool to set entity_group_

  /// @brief Set the entity group for this thread pool.
  void entity_group(const std::shared_ptr<gxf::EntityGroup>& entity_group) {
    entity_group_ = entity_group;
  }

 private:
  Parameter<int64_t> initial_size_;  ///< Initial size of the thread pool.
  // Note: The GXF priority parameter is not currently functional, so we don't expose it here.
  // Parameter<int64_t> priority_;      ///< priority of the thread pool (0=low, 1=medium, 2=high)

  std::shared_ptr<gxf::EntityGroup> entity_group_;  ///< The entity group associated with the thread
                                                    ///< pool.
  std::vector<std::shared_ptr<Operator>> operators_;  ///< The operators associated with the thread
                                                      ///< pool.
};

/**
 * @brief GPU device resource.
 *
 * This resource can be used to associate a set of components with a particular GPU device ID.
 *
 * The Holoscan SDK components which will use a GPUDevice resource if found include:
 *   - BlockMemoryPool
 *   - CudaStreamPool
 *   - UcxContext
 *   - UcxReceiver
 *   - UcxTransmitter
 *
 * dev0 = make_resource<GPUDevice>("dev0", Arg("dev_id", static_cast<int32_t>(0)));
 *
 * gpu0_group = EntityGroup("gpu0_group");
 * gpu0_group.add(*dev0);
 *
 * Then any other components that need to be associated with this device can be added to that
 * same entity group.
 *
 * ==Parameters==
 *
 * - **dev_id** (int32_t, optional): The CUDA device id specifying which device the memory pool
 * will use (Default: 0).
 */
class GPUDevice : public gxf::GXFSystemResourceBase {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(GPUDevice, gxf::GXFSystemResourceBase)
  GPUDevice() = default;
  GPUDevice(const std::string& name, nvidia::gxf::GPUDevice* component);

  /// @brief The underlying GXF component's name.
  const char* gxf_typename() const override { return "nvidia::gxf::GPUDevice"; }

  void setup(ComponentSpec& spec) override;

  /// @brief The GPU device ID.
  int32_t device_id() const;

 private:
  Parameter<int32_t> dev_id_;  ///< The GPU device ID.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_SYSTEM_RESOURCES_HPP */
