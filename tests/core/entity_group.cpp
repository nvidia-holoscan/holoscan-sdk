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

#include <gtest/gtest.h>
#include <gxf/core/gxf.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../config.hpp"
#include "../utils.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity_group.hpp"
#include "holoscan/core/resources/gxf/system_resources.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

using namespace std::string_literals;

namespace holoscan {

using EntityGroupWithGXFContext = TestWithGXFContext;

TEST_F(EntityGroupWithGXFContext, TestEntityGroupAddBasedOnEid) {
  auto g = gxf::EntityGroup(F.executor().context(), "group1");
  auto allocator = F.make_resource<UnboundedAllocator>("allocator");
  auto allocator2 = F.make_resource<UnboundedAllocator>("allocator2");

  // need to initialize the resource before we can add it to an entity group.
  allocator->initialize();
  allocator2->initialize();

  // add to the entity group via the GXF eid
  g.add(allocator->gxf_eid());
  g.add(allocator2->gxf_eid());

  EXPECT_EQ(allocator->gxf_entity_group_name(), "group1"s);
  EXPECT_EQ(allocator2->gxf_entity_group_name(), "group1"s);
}

TEST_F(EntityGroupWithGXFContext, TestEntityGroupAddBasedOnComponent) {
  auto g = gxf::EntityGroup(F.executor().context(), "mygroup");
  auto allocator = F.make_resource<UnboundedAllocator>("allocator");

  // need to initialize the resource before we can add it to an entity group.
  allocator->initialize();

  // GXF will have initially assigned to a default entity group
  // (not sure we should test the exact name here since it could be subject to change)
  EXPECT_EQ(allocator->gxf_entity_group_name(), "default_entity_group"s);

  // add a GXFComponent object to the entity group
  g.add(*allocator);

  // retrieve the name of the entity group to which the component belongs
  EXPECT_EQ(allocator->gxf_entity_group_name(), "mygroup"s);

  // adding the same component a second time will raise an exception
  EXPECT_THROW(
      {
        try {
          g.add(*allocator);
        } catch (const std::runtime_error& e) {
          ASSERT_TRUE(std::string(e.what()).find("failure during GXF call") != std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST_F(EntityGroupWithGXFContext, TestEntityGroupAddThreadPool) {
  auto g = gxf::EntityGroup(F.executor().context(), "mygroup");
  auto allocator = F.make_resource<UnboundedAllocator>("allocator");
  auto thread_pool = F.make_resource<ThreadPool>("thread_pool");

  // need to initialize the resource before we can add it to an entity group.
  allocator->initialize();
  thread_pool->initialize();

  // GXF will have initially assigned to a default entity group
  // (not sure we should test the exact name here since it could be subject to change)
  EXPECT_EQ(allocator->gxf_entity_group_name(), "default_entity_group"s);

  // add a GXFComponent objects to the entity group
  g.add(*allocator);
  g.add(*thread_pool);

  // retrieve the name of the entity group to which the component belongs
  EXPECT_EQ(allocator->gxf_entity_group_name(), "mygroup"s);
  EXPECT_EQ(thread_pool->gxf_entity_group_name(), "mygroup"s);
}

TEST_F(EntityGroupWithGXFContext, TestEntityGroupAddGPUDevice) {
  auto g = gxf::EntityGroup(F.executor().context(), "mygroup");
  auto allocator = F.make_resource<UnboundedAllocator>("allocator");
  auto device = F.make_resource<GPUDevice>("device");

  // need to initialize the resource before we can add it to an entity group.
  allocator->initialize();
  device->initialize();

  // GXF will have initially assigned to a default entity group
  // (not sure we should test the exact name here since it could be subject to change)
  EXPECT_EQ(allocator->gxf_entity_group_name(), "default_entity_group"s);

  // add a GXFComponent object to the entity group
  g.add(*allocator);
  g.add(*device);

  // retrieve the name of the entity group to which the component belongs
  EXPECT_EQ(allocator->gxf_entity_group_name(), "mygroup"s);
  EXPECT_EQ(device->gxf_entity_group_name(), "mygroup"s);
}

}  // namespace holoscan
