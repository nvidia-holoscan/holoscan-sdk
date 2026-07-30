/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_SCHEDULERS_GREEDY_FRAGMENT_ALLOCATION_HPP
#define HOLOSCAN_CORE_SCHEDULERS_GREEDY_FRAGMENT_ALLOCATION_HPP

#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "../fragment_scheduler.hpp"

namespace holoscan {

class GreedyFragmentAllocationStrategy : public FragmentAllocationStrategy {
 public:
  void on_add_available_resource(const AvailableSystemResource& available_resource) override;
  void on_add_resource_requirement(const SystemResourceRequirement& resource_requirement) override;
  holoscan::expected<std::unordered_map<std::string, std::string>, std::string> schedule() override;

 private:
  struct AvailableSystemResourceComparator {
    bool operator()(const AvailableSystemResource& a, const AvailableSystemResource& b) const;
  };

  struct SystemResourceRequirementComparator {
    bool operator()(const SystemResourceRequirement& a, const SystemResourceRequirement& b) const;
  };

  std::priority_queue<AvailableSystemResource, std::vector<AvailableSystemResource>,
                      AvailableSystemResourceComparator>
      available_resources_pq_;

  std::priority_queue<SystemResourceRequirement, std::vector<SystemResourceRequirement>,
                      SystemResourceRequirementComparator>
      resource_requirements_pq_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SCHEDULERS_GREEDY_FRAGMENT_ALLOCATION_HPP */
