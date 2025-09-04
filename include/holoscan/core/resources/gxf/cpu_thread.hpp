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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_CPU_THREAD_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_CPU_THREAD_HPP

#include <string>
#include <vector>

#include <gxf/std/cpu_thread.hpp>
#include "yaml-cpp/yaml.h"
#include "../../component_spec.hpp"
#include "../../errors.hpp"
#include "../../expected.hpp"
#include "../../gxf/gxf_resource.hpp"
#include "../../parameter.hpp"

// YAML conversion support for SchedulingPolicy
template <>
struct YAML::convert<nvidia::gxf::SchedulingPolicy> {
  static Node encode(const nvidia::gxf::SchedulingPolicy& rhs) {
    Node node;
    switch (rhs) {
      case nvidia::gxf::SchedulingPolicy::kFirstInFirstOut:
        node = "SCHED_FIFO";
        break;
      case nvidia::gxf::SchedulingPolicy::kRoundRobin:
        node = "SCHED_RR";
        break;
      case nvidia::gxf::SchedulingPolicy::kDeadline:
        node = "SCHED_DEADLINE";
        break;
      default:
        node = static_cast<int32_t>(rhs);  // fallback to numeric value
        break;
    }
    return node;
  }

  static bool decode(const Node& node, nvidia::gxf::SchedulingPolicy& rhs) {
    if (!node.IsScalar())
      return false;

    const std::string value = node.Scalar();

    // Support string values
    if (value == "SCHED_FIFO") {
      rhs = nvidia::gxf::SchedulingPolicy::kFirstInFirstOut;
      return true;
    } else if (value == "SCHED_RR") {
      rhs = nvidia::gxf::SchedulingPolicy::kRoundRobin;
      return true;
    } else if (value == "SCHED_DEADLINE") {
      rhs = nvidia::gxf::SchedulingPolicy::kDeadline;
      return true;
    }

    // Support using the numeric enum values as well
    try {
      int32_t numeric_value = std::stoi(value);
      switch (numeric_value) {
        case static_cast<int32_t>(nvidia::gxf::SchedulingPolicy::kFirstInFirstOut):
          rhs = nvidia::gxf::SchedulingPolicy::kFirstInFirstOut;
          return true;
        case static_cast<int32_t>(nvidia::gxf::SchedulingPolicy::kRoundRobin):
          rhs = nvidia::gxf::SchedulingPolicy::kRoundRobin;
          return true;
        case static_cast<int32_t>(nvidia::gxf::SchedulingPolicy::kDeadline):
          rhs = nvidia::gxf::SchedulingPolicy::kDeadline;
          return true;
        default:
          break;  // Invalid numeric value, fall through to return false
      }
    } catch (...) {
      // Not a valid number, continue to return false
    }

    return false;  // Invalid value
  }
};

namespace holoscan {

/**
 * @brief Real-time scheduling policies supported by POSIX and/or the Linux kernel.
 *
 * Available scheduling policy values:
 * - kFirstInFirstOut (1): SCHED_FIFO - First-in-first-out scheduling policy supported by POSIX and
 * Linux kernel
 * - kRoundRobin (2): SCHED_RR - Round-robin scheduling policy supported by POSIX and Linux kernel
 * - kDeadline (6): SCHED_DEADLINE - Deadline scheduling policy supported by Linux kernel
 */
using SchedulingPolicy = nvidia::gxf::SchedulingPolicy;

/**
 * @brief CPU thread class.
 *
 * A CPUThread resource can be added to an operator to control whether it will be pinned to a
 * specific thread in a ThreadPool (as used by MultiThreadScheduler). See the ThreadPool API
 * documentation for a more detailed description of its usage.
 *
 * Application authors should not need to use this class directly. It is used behind the scenes as
 * needed by the `holoscan::ThreadPool` class.
 *
 * ==Parameters==
 *
 * - **pin_entity** (bool, optional): Whether or not an operator should be pinned to a specific
 * thread (Default: false).
 * - **pin_cores** (std::vector<uint32_t>, optional): CPU core IDs to pin the worker thread to.
 * Empty vector means no core pinning (Default: empty).
 * - **sched_policy** (SchedulingPolicy, optional): Linux real-time scheduling policy.
 * kFirstInFirstOut is SCHED_FIFO, kRoundRobin is SCHED_RR, kDeadline is SCHED_DEADLINE
 * (Default: not set).
 * - **sched_priority** (uint32_t, optional): Thread priority for FirstInFirstOut and RoundRobin
 * policies (Default: not set).
 * - **sched_runtime** (uint64_t, optional): Expected worst case execution time in nanoseconds
 * for Deadline policy (Default: not set).
 * - **sched_deadline** (uint64_t, optional): Relative deadline in nanoseconds for Deadline policy
 * (Default: not set).
 * - **sched_period** (uint64_t, optional): Period in nanoseconds for Deadline policy
 * (Default: not set).
 */
class CPUThread : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(CPUThread, gxf::GXFResource)

  explicit CPUThread(bool pin_entity = true) : pin_entity_(pin_entity) { name_ = "cpu_thread"; }

  /// @brief The underlying GXF component's name.
  const char* gxf_typename() const override { return "nvidia::gxf::CPUThread"; }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

  /// @brief Returns whether the component is pinned to a worker thread.
  bool pinned() const;

  /// @brief CPU core IDs to pin the worker thread to (empty means no core pinning).
  std::vector<uint32_t> pin_cores() const;

  /// @brief Real-time scheduling policy.
  holoscan::expected<SchedulingPolicy, holoscan::RuntimeError> sched_policy() const;

  /// @brief Thread priority (only for FirstInFirstOut and RoundRobin policies).
  holoscan::expected<uint32_t, holoscan::RuntimeError> sched_priority() const;

  /// @brief Expected worst case execution time in nanoseconds (only for Deadline policy).
  holoscan::expected<uint64_t, holoscan::RuntimeError> sched_runtime() const;

  /// @brief Relative deadline in nanoseconds (only for Deadline policy).
  holoscan::expected<uint64_t, holoscan::RuntimeError> sched_deadline() const;

  /// @brief Period in nanoseconds (only for Deadline policy).
  holoscan::expected<uint64_t, holoscan::RuntimeError> sched_period() const;

  /// @brief Helper method to check if real-time scheduling is enabled.
  bool is_realtime() const;

 private:
  Parameter<bool> pin_entity_{false};  ///< Whether or not to pin an operator to a specific thread
  Parameter<std::vector<uint32_t>> pin_cores_;  ///< CPU core IDs to pin the worker thread to
  Parameter<YAML::Node> sched_policy_;          ///< Real-time scheduling policy
  Parameter<uint32_t> sched_priority_;          ///< Thread priority
  Parameter<uint64_t> sched_runtime_;   ///< Expected worst case execution time in nanoseconds
  Parameter<uint64_t> sched_deadline_;  ///< Relative deadline in nanoseconds
  Parameter<uint64_t> sched_period_;    ///< Period in nanoseconds
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_CPU_THREAD_HPP */
