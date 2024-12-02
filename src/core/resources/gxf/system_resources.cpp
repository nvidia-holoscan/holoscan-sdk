/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/system_resources.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include <gxf/std/resources.hpp>
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity_group.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/gxf/cpu_thread.hpp"

namespace holoscan {

ThreadPool::ThreadPool(const std::string& name, nvidia::gxf::ThreadPool* component)
    : gxf::GXFSystemResourceBase(name, component) {}

int64_t ThreadPool::size() const {
  if (gxf_cptr_) {
    nvidia::gxf::ThreadPool* pool = static_cast<nvidia::gxf::ThreadPool*>(gxf_cptr_);
    return pool->size();
  }
  return 0;
}

void ThreadPool::setup(ComponentSpec& spec) {
  spec.param(initial_size_,
             "initial_size",
             "Initial ThreadPool Size",
             "Initial number of worker threads in the pool",
             1L);
}

void ThreadPool::add(const std::shared_ptr<Operator>& op, bool pin_operator) {
  // add a CPUThread argument if one did not already exist
  const auto& resource_map = op->resources();
  auto thread_arg_it = std::find_if(resource_map.begin(), resource_map.end(), [](const auto& r) {
    // check type by whether pointer cast to CPUThread succeeds
    return typeid(*r.second) == typeid(CPUThread);
  });
  if (thread_arg_it == resource_map.end()) {
    // Create a CPUThread resource and add it to the Operator's list of arguments
    const std::string thread_name = fmt::format("{}_cpu_thread", op->name());
    auto cpu_thread =
        fragment_->make_resource<CPUThread>(thread_name, Arg{"pin_entity", pin_operator});
    auto cpu_thread_resource = std::dynamic_pointer_cast<holoscan::Resource>(cpu_thread);
    if (!cpu_thread_resource) {
      throw std::runtime_error(
          "Failed to cast std::shared_ptr<CPUThread> to std::shared_ptr<holoscan::Resource>");
    }
    op->add_arg(cpu_thread_resource);
  }

  // store pointer to operators for later assignment to the entity group
  // (need to do the assignment from GXFExecutor only after the Operator has been initialized)
  operators_.push_back(op);
}

void ThreadPool::add(std::vector<std::shared_ptr<Operator>> ops, bool pin_operator) {
  for (const auto& op : ops) { add(op, pin_operator); }
}

YAML::Node ThreadPool::to_yaml_node() const {
  YAML::Node node = GXFSystemResourceBase::to_yaml_node();
  node["operators in pool"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& op : operators_) { node["operators in pool"].push_back(YAML::Node(op->name())); }
  return node;
}

GPUDevice::GPUDevice(const std::string& name, nvidia::gxf::GPUDevice* component)
    : gxf::GXFSystemResourceBase(name, component) {}

int32_t GPUDevice::device_id() const {
  if (gxf_cptr_) {
    nvidia::gxf::GPUDevice* gpu_device = static_cast<nvidia::gxf::GPUDevice*>(gxf_cptr_);
    return gpu_device->device_id();
  }
  return 0;
}

void GPUDevice::setup(ComponentSpec& spec) {
  spec.param(dev_id_,
             "dev_id",
             "Device Id",
             "Create CUDA Stream on which device.",
             static_cast<int32_t>(0));
}

}  // namespace holoscan
