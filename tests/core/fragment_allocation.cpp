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

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <string>

#include "holoscan/core/schedulers/greedy_fragment_allocation.hpp"

namespace holoscan {

// struct AvailableSystemResource {
//   std::string app_worker_id;
//   std::unordered_set<std::string> target_fragments;
//   int32_t cpu = 0;
//   int32_t gpu = 0;
//   uint64_t memory = 0;
//   uint64_t shared_memory = 0;
//   uint64_t gpu_memory = 0;
// };

// struct SystemResourceRequirement {
//   std::string fragment_name;
//   float cpu = -1.0f;
//   float cpu_limit = -1.0f;
//   float gpu = -1.0f;
//   float gpu_limit = -1.0f;
//   uint64_t memory = 0;
//   uint64_t memory_limit = 0;
//   uint64_t shared_memory = 0;
//   uint64_t shared_memory_limit = 0;
//   uint64_t gpu_memory = 0;
//   uint64_t gpu_memory_limit = 0;
// };

TEST(FragmentAllocation, GreedyAllocationTwoNormalFragments) {
  GreedyFragmentAllocationStrategy strategy;

  strategy.add_available_resource(AvailableSystemResource{"app_worker_2", {}, 1, 1, 1, 1, 1});
  strategy.add_available_resource(AvailableSystemResource{"app_worker_1", {}, 1, 1, 1, 1, 1});

  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_2", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_1", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  auto schedule_result = strategy.schedule();
  ASSERT_TRUE(static_cast<bool>(schedule_result));
  auto& schedule = schedule_result.value();
  // If all fragments and workers have the same requirements, the order of scheduling should be
  // determined by the fragment/workers names (alphabetical order)
  ASSERT_EQ(schedule["fragment_1"], "app_worker_1");
  ASSERT_EQ(schedule["fragment_2"], "app_worker_2");
}

TEST(FragmentAllocation, GreedyAllocationCheckResourceRequirement) {
  GreedyFragmentAllocationStrategy strategy;

  // No target fragments
  strategy.add_available_resource(AvailableSystemResource{"app_worker_1", {}, 1, 1, 1, 1, 2});
  strategy.add_available_resource(AvailableSystemResource{"app_worker_2", {}, 1, 1, 1, 1, 1});

  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_1", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_2", 1, 1, 1, 1, 1, 1, 1, 1, 2, 0});

  auto schedule_result = strategy.schedule();
  ASSERT_TRUE(static_cast<bool>(schedule_result));
  auto& schedule = schedule_result.value();
  // Expect fragment_2 to be scheduled on app_worker_1 (gpu_memory requirement == 2)
  ASSERT_EQ(schedule["fragment_2"], "app_worker_1");
  ASSERT_EQ(schedule["fragment_1"], "app_worker_2");
}

TEST(FragmentAllocation, GreedyAllocationCheckAvailableResource) {
  GreedyFragmentAllocationStrategy strategy;

  // No target fragments
  strategy.add_available_resource(AvailableSystemResource{"app_worker_1", {}, 1, 1, 1, 1, 2});
  strategy.add_available_resource(AvailableSystemResource{"app_worker_2", {}, 1, 1, 1, 1, 1});

  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_1", -1, -1, -1, -1, 0, 0, 0, 0, 0, 0});
  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_2", -1, -1, -1, -1, 0, 0, 0, 0, 0, 0});

  auto schedule_result = strategy.schedule();
  ASSERT_TRUE(static_cast<bool>(schedule_result));
  auto& schedule = schedule_result.value();
  // Expect matching more strict available resource first
  // (app_worker_1 has 2 gpu_requirement, so app_worker_2 should be scheduled before fragment_1)
  ASSERT_EQ(schedule["fragment_2"], "app_worker_1");
  ASSERT_EQ(schedule["fragment_1"], "app_worker_2");
}

TEST(FragmentAllocation, GreedyAllocationRedundantWorker) {
  GreedyFragmentAllocationStrategy strategy;

  // No target fragments
  strategy.add_available_resource(
      AvailableSystemResource{"app_worker_1", {"fragment_1", "fragment_2"}, 0, 0, 0, 0, 0});
  strategy.add_available_resource(
      AvailableSystemResource{"app_worker_2", {"fragment_2"}, 0, 0, 0, 0, 0});
  strategy.add_available_resource(
      AvailableSystemResource{"app_worker_3", {"fragment_3"}, 0, 0, 0, 0, 0});

  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_1", -1, -1, -1, -1, 0, 0, 0, 0, 0, 0});
  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_2", -1, -1, -1, -1, 0, 0, 0, 0, 0, 0});
  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_3", -1, -1, -1, -1, 0, 0, 0, 0, 0, 0});

  auto schedule_result = strategy.schedule();
  ASSERT_TRUE(static_cast<bool>(schedule_result));
  auto& schedule = schedule_result.value();
  // Expect matching worker with most target fragments first
  // app_worker_2 is redundant, so app_worker_1 covers both fragment1 and fragment2, and
  // app_worker_3 covers fragment3
  ASSERT_EQ(schedule.size(), 3);
  ASSERT_EQ(schedule["fragment_1"], "app_worker_1");
  ASSERT_EQ(schedule["fragment_2"], "app_worker_1");
  ASSERT_EQ(schedule["fragment_3"], "app_worker_3");
}

TEST(FragmentAllocation, GreedyAllocationRedundantWorker2) {
  GreedyFragmentAllocationStrategy strategy;

  // No target fragments
  strategy.add_available_resource(
      AvailableSystemResource{"app_worker_1", {"fragment_1", "fragment_2"}, 0, 0, 0, 0, 1});
  strategy.add_available_resource(
      AvailableSystemResource{"app_worker_2", {"fragment_2"}, 0, 0, 0, 0, 2});
  strategy.add_available_resource(
      AvailableSystemResource{"app_worker_3", {"fragment_1"}, 0, 0, 0, 0, 0});
  strategy.add_available_resource(
      AvailableSystemResource{"app_worker_4", {"fragment_3"}, 0, 0, 0, 0, 0});

  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_1", -1, -1, -1, -1, 0, 0, 0, 0, 0, 0});
  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_2", -1, -1, -1, -1, 0, 0, 0, 0, 2, 0});
  strategy.add_resource_requirement(
      SystemResourceRequirement{"fragment_3", -1, -1, -1, -1, 0, 0, 0, 0, 0, 0});

  auto schedule_result = strategy.schedule();
  ASSERT_TRUE(static_cast<bool>(schedule_result));
  auto& schedule = schedule_result.value();
  // app_worker_1 doesn't cover the requirement of fragment_2, so app_worker_2 is selected for it.
  // app_worker_3 covers fragment_1 and app_worker_4 covers fragment_3.
  ASSERT_EQ(schedule.size(), 3);
  ASSERT_EQ(schedule["fragment_1"], "app_worker_3");
  ASSERT_EQ(schedule["fragment_2"], "app_worker_2");
  ASSERT_EQ(schedule["fragment_3"], "app_worker_4");
}

}  // namespace holoscan
