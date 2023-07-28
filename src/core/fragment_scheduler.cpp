/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/fragment_scheduler.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "holoscan/core/schedulers/greedy_fragment_allocation.hpp"

namespace holoscan {

bool AvailableSystemResource::has_enough_resources(
    const SystemResourceRequirement& resource_requirement) const {
  if ((resource_requirement.cpu >= 0 && cpu < resource_requirement.cpu) ||
      (resource_requirement.cpu_limit >= 0 && cpu > resource_requirement.cpu_limit)) {
    return false;
  }

  if ((resource_requirement.gpu >= 0 && gpu < resource_requirement.gpu) ||
      (resource_requirement.gpu_limit >= 0 && gpu > resource_requirement.gpu_limit)) {
    return false;
  }

  if ((resource_requirement.memory > 0 && memory < resource_requirement.memory) ||
      (resource_requirement.memory_limit > 0 && memory > resource_requirement.memory_limit)) {
    return false;
  }

  if ((resource_requirement.shared_memory > 0 &&
       shared_memory < resource_requirement.shared_memory) ||
      (resource_requirement.shared_memory_limit > 0 &&
       shared_memory > resource_requirement.shared_memory_limit)) {
    return false;
  }

  if ((resource_requirement.gpu_memory > 0 && gpu_memory < resource_requirement.gpu_memory) ||
      (resource_requirement.gpu_memory_limit > 0 &&
       gpu_memory > resource_requirement.gpu_memory_limit)) {
    return false;
  }

  return true;
}

void FragmentAllocationStrategy::add_resource_requirement(
    const SystemResourceRequirement& resource_requirement) {
  auto result =
      resource_requirements_.emplace(resource_requirement.fragment_name, resource_requirement);
  if (result.second) { on_add_resource_requirement(result.first->second); }
}

void FragmentAllocationStrategy::add_resource_requirement(
    SystemResourceRequirement&& resource_requirement) {
  auto result = resource_requirements_.emplace(resource_requirement.fragment_name,
                                               std::move(resource_requirement));
  if (result.second) { on_add_resource_requirement(result.first->second); }
}

void FragmentAllocationStrategy::add_available_resource(
    const AvailableSystemResource& available_resource) {
  auto result = available_resources_.emplace(available_resource.app_worker_id, available_resource);
  if (result.second) { on_add_available_resource(result.first->second); }
}

void FragmentAllocationStrategy::add_available_resource(
    AvailableSystemResource&& available_resource) {
  auto result =
      available_resources_.emplace(available_resource.app_worker_id, std::move(available_resource));
  if (result.second) { on_add_available_resource(result.first->second); }
}

FragmentScheduler::FragmentScheduler(
    std::unique_ptr<FragmentAllocationStrategy>&& allocation_strategy)
    : strategy_([&allocation_strategy]() {
        if (allocation_strategy) {
          return std::move(allocation_strategy);
        } else {
          return static_cast<std::unique_ptr<FragmentAllocationStrategy>>(
              std::make_unique<GreedyFragmentAllocationStrategy>());
        }
      }()) {}

FragmentScheduler::~FragmentScheduler() = default;

void FragmentScheduler::add_resource_requirement(
    const SystemResourceRequirement& resource_requirement) {
  strategy_->add_resource_requirement(resource_requirement);
}

void FragmentScheduler::add_resource_requirement(SystemResourceRequirement&& resource_requirement) {
  strategy_->add_resource_requirement(std::move(resource_requirement));
}

void FragmentScheduler::add_available_resource(const AvailableSystemResource& available_resource) {
  strategy_->add_available_resource(available_resource);
}

void FragmentScheduler::add_available_resource(AvailableSystemResource&& available_resource) {
  strategy_->add_available_resource(std::move(available_resource));
}

holoscan::expected<std::unordered_map<std::string, std::string>, std::string>
FragmentScheduler::schedule() {
  if (!strategy_) {
    return holoscan::make_unexpected(std::string("No allocation strategy is set."));
  }
  return strategy_->schedule();
}

}  // namespace holoscan
