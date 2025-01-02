/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/schedulers/greedy_fragment_allocation.hpp"

#include <list>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "holoscan/logger/logger.hpp"

namespace holoscan {

void GreedyFragmentAllocationStrategy::on_add_available_resource(
    const AvailableSystemResource& available_resource) {
  available_resources_pq_.push(available_resource);
}
void GreedyFragmentAllocationStrategy::on_add_resource_requirement(
    const SystemResourceRequirement& resource_requirement) {
  resource_requirements_pq_.push(resource_requirement);
}

holoscan::expected<std::unordered_map<std::string, std::string>, std::string>
GreedyFragmentAllocationStrategy::schedule() {
  HOLOSCAN_LOG_DEBUG("GreedyFragmentAllocationStrategy::schedule()");

  std::unordered_map<std::string, std::string> scheduled_fragments;

  // Copy sorted availableSystemResource (app workers)
  std::vector<AvailableSystemResource> available_resources;
  std::priority_queue<AvailableSystemResource,
                      std::vector<AvailableSystemResource>,
                      AvailableSystemResourceComparator>
      available_resources_pq_copy = available_resources_pq_;
  while (!available_resources_pq_copy.empty()) {
    available_resources.push_back(available_resources_pq_copy.top());
    available_resources_pq_copy.pop();
  }

  // Copy sorted SystemResourceRequirement (fragments)
  std::list<SystemResourceRequirement> resource_requirements;
  std::priority_queue<SystemResourceRequirement,
                      std::vector<SystemResourceRequirement>,
                      SystemResourceRequirementComparator>
      resource_requirements_pq_copy = resource_requirements_pq_;
  while (!resource_requirements_pq_copy.empty()) {
    resource_requirements.push_back(resource_requirements_pq_copy.top());
    resource_requirements_pq_copy.pop();
  }

  // Create map for fast lookup of fragment requirements
  std::unordered_map<std::string, SystemResourceRequirement*> resource_requirements_map;
  resource_requirements_map.reserve(resource_requirements.size());
  for (auto& resource_requirement : resource_requirements) {
    resource_requirements_map[resource_requirement.fragment_name] = &resource_requirement;
  }

  for (auto& available_resource : available_resources) {
    bool is_worker_scheduled = false;
    if (available_resource.target_fragments.empty()) {
      // Check if any fragment can be scheduled for available_resource (app worker).
      for (auto it = resource_requirements.begin(); it != resource_requirements.end(); ++it) {
        const auto& fragment_requirement = *it;
        if (available_resource.has_enough_resources(fragment_requirement)) {
          scheduled_fragments[fragment_requirement.fragment_name] =
              available_resource.app_worker_id;
          resource_requirements_map.erase(fragment_requirement.fragment_name);
          resource_requirements.erase(it);
          is_worker_scheduled = true;
          break;
        }
      }
      if (!is_worker_scheduled) {
        HOLOSCAN_LOG_DEBUG(
            "No fragment can be scheduled for app worker: {}, cpu: {}, gpu: {}, memory: {}, "
            "shared_memory: {}, gpu_memory: {}",
            available_resource.app_worker_id,
            available_resource.cpu,
            available_resource.gpu,
            available_resource.memory,
            available_resource.shared_memory,
            available_resource.gpu_memory);
      }
    } else {
      bool is_target_fragments_schedulable = true;
      for (const auto& fragment_name : available_resource.target_fragments) {
        // Check if all target fragments can be scheduled for available_resource.
        if (resource_requirements_map.find(fragment_name) == resource_requirements_map.end()) {
          is_target_fragments_schedulable = false;
          break;
        }
        const auto& fragment_requirement = *resource_requirements_map[fragment_name];
        if (!available_resource.has_enough_resources(fragment_requirement)) {
          is_target_fragments_schedulable = false;
          break;
        }
      }
      if (is_target_fragments_schedulable) {
        // Remove all target fragments from resource_requirements and resource_requirements_map.
        for (auto it = resource_requirements.begin(); it != resource_requirements.end();) {
          const auto& fragment_requirement = *it;
          if (available_resource.target_fragments.find(fragment_requirement.fragment_name) !=
              available_resource.target_fragments.end()) {
            scheduled_fragments[fragment_requirement.fragment_name] =
                available_resource.app_worker_id;
            resource_requirements_map.erase(fragment_requirement.fragment_name);
            it = resource_requirements.erase(it);
          } else {
            ++it;
          }
        }
      } else {
        HOLOSCAN_LOG_DEBUG(
            "app worker '{}' does not have enough resources to schedule all target "
            "fragments '{}'",
            available_resource.app_worker_id,
            fmt::join(available_resource.target_fragments, ", "));
      }
    }
  }

  // Print scheduled fragments.
  for (const auto& [fragment_name, app_worker_id] : scheduled_fragments) {
    HOLOSCAN_LOG_DEBUG(
        "fragment '{}' is scheduled on app worker '{}'", fragment_name, app_worker_id);
  }

  // Check if all fragments are scheduled.
  if (!resource_requirements_map.empty()) {
    const auto total_requirements = resource_requirements_pq_.size();
    std::string error_message =
        fmt::format("{}/{} fragments scheduled. Awaiting remaining worker connections.\n",
                    total_requirements - resource_requirements_map.size(),
                    total_requirements);
    for (const auto& [fragment_name, fragment_requirement] : resource_requirements_map) {
      error_message += fmt::format(
          "- fragment_name: {}, cpu: {}, cpu_limit: {}, gpu: {}, gpu_limit: {}, memory: {}, "
          "memory_limit: {}, shared_memory: {}, shared_memory_limit: {}, gpu_memory: {}, "
          "gpu_memory_limit: {}\n",
          fragment_name,
          fragment_requirement->cpu,
          fragment_requirement->cpu_limit,
          fragment_requirement->gpu,
          fragment_requirement->gpu_limit,
          fragment_requirement->memory,
          fragment_requirement->memory_limit,
          fragment_requirement->shared_memory,
          fragment_requirement->shared_memory_limit,
          fragment_requirement->gpu_memory,
          fragment_requirement->gpu_memory_limit);
    }
    return holoscan::unexpected<std::string>(error_message);
  }

  return scheduled_fragments;
}

bool GreedyFragmentAllocationStrategy::AvailableSystemResourceComparator::operator()(
    const AvailableSystemResource& a, const AvailableSystemResource& b) const {
  // priority:
  //    1. target_fragments (higher priority if more target fragments are specified)
  //    2. shared_memory (higher priority if less shared memory is available)
  //    3. gpu (higher priority if less gpu is available)
  //    4. gpu_memory (higher priority if less gpu memory is available)
  //    5. cpu (higher priority if less cpu is available)
  //    6. memory (higher priority if less memory is available)
  //    7. fragment names (higher priority if the name is alphabetically first)
  // If the value is zero, then it has lower priority.

  if (a.target_fragments.size() != b.target_fragments.size()) {
    // If 'a' has more target fragments, then 'a' has higher priority.
    return a.target_fragments.size() < b.target_fragments.size();
  }
  if (a.shared_memory != b.shared_memory) { return a.shared_memory > b.shared_memory; }
  if (a.gpu != b.gpu) { return a.gpu > b.gpu; }
  if (a.gpu_memory != b.gpu_memory) { return a.gpu_memory > b.gpu_memory; }
  if (a.cpu != b.cpu) { return a.cpu > b.cpu; }
  if (a.memory != b.memory) { return a.memory > b.memory; }

  auto fragment_size = a.target_fragments.size();
  auto a_fragment_names_sorted_set =
      std::set<std::string>(a.target_fragments.begin(), a.target_fragments.end());
  auto b_fragment_names_sorted_set =
      std::set<std::string>(b.target_fragments.begin(), b.target_fragments.end());
  auto a_fragment_names_sorted = std::vector<std::string>(a_fragment_names_sorted_set.begin(),
                                                          a_fragment_names_sorted_set.end());
  auto b_fragment_names_sorted = std::vector<std::string>(b_fragment_names_sorted_set.begin(),
                                                          b_fragment_names_sorted_set.end());
  for (size_t i = 0; i < fragment_size; ++i) {
    if (a_fragment_names_sorted[i] != b_fragment_names_sorted[i]) {
      return a_fragment_names_sorted[i] > b_fragment_names_sorted[i];
    }
  }
  return true;
}

bool GreedyFragmentAllocationStrategy::SystemResourceRequirementComparator::operator()(
    const SystemResourceRequirement& a, const SystemResourceRequirement& b) const {
  // priority:
  //    1. shared_memory (higher priority if more shared memory is required)
  //    2. shared_memory_limit (higher priority if more shared memory is required)
  //    3. gpu (higher priority if more gpu is required)
  //    4. gpu_limit (higher priority if more gpu is required)
  //    5. gpu_memory (higher priority if more gpu memory is required)
  //    6. gpu_memory_limit (higher priority if more gpu memory is required)
  //    7. cpu (higher priority if more cpu is required)
  //    8. cpu_limit (higher priority if more cpu is required)
  //    9. memory (higher priority if more memory is required)
  //    10. memory_limit (higher priority if more memory is required)
  //    11. fragment name (higher priority if the name is alphabetically first)
  // If the value is -1 (for cpu/cpu_limit/gpu/gpu_limit) or zero (other fields), then it has lower
  // priority.

  if (a.shared_memory != b.shared_memory) { return a.shared_memory < b.shared_memory; }
  if (a.shared_memory_limit != b.shared_memory_limit) {
    return a.shared_memory_limit < b.shared_memory_limit;
  }
  if (a.gpu != b.gpu) { return a.gpu < b.gpu; }
  if (a.gpu_limit != b.gpu_limit) { return a.gpu_limit < b.gpu_limit; }
  if (a.gpu_memory != b.gpu_memory) { return a.gpu_memory < b.gpu_memory; }
  if (a.gpu_memory_limit != b.gpu_memory_limit) { return a.gpu_memory_limit < b.gpu_memory_limit; }
  if (a.cpu != b.cpu) { return a.cpu < b.cpu; }
  if (a.cpu_limit != b.cpu_limit) { return a.cpu_limit < b.cpu_limit; }
  if (a.memory != b.memory) { return a.memory < b.memory; }
  if (a.memory_limit != b.memory_limit) { return a.memory_limit < b.memory_limit; }
  if (a.fragment_name != b.fragment_name) { return a.fragment_name > b.fragment_name; }

  return false;
}

}  // namespace holoscan
