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

#include <gxf/std/cpu_thread.hpp>
#include "../../component_spec.hpp"
#include "../../gxf/gxf_resource.hpp"
#include "../../parameter.hpp"

namespace holoscan {

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
 */
class CPUThread : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(CPUThread, gxf::GXFResource)

  explicit CPUThread(bool pin_entity = true) : pin_entity_(pin_entity) { name_ = "cpu_thread"; }

  CPUThread(const std::string& name, nvidia::gxf::CPUThread* component);

  /// @brief The underlying GXF component's name.
  const char* gxf_typename() const override { return "nvidia::gxf::CPUThread"; }

  void setup(ComponentSpec& spec) override;

 private:
  Parameter<bool> pin_entity_{false};  ///< Whether or not to pin an operator to a specific thread
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_CPU_THREAD_HPP */
