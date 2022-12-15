/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <string>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"
#include "../utils.hpp"

namespace holoscan {
TEST(IOSpec, TestIOSpecInitialize) {
  OperatorSpec op_spec = OperatorSpec();

  // kInput
  IOSpec spec = IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput,
                       &typeid(holoscan::gxf::Entity));
  EXPECT_EQ(spec.op_spec(), &op_spec);
  EXPECT_EQ(spec.name(), "a");
  EXPECT_EQ(spec.io_type(), IOSpec::IOType::kInput);
  EXPECT_EQ(spec.typeinfo(), &typeid(holoscan::gxf::Entity));

  // kOutput
  IOSpec out_spec{&op_spec, std::string("b"), IOSpec::IOType::kOutput,
                  &typeid(holoscan::gxf::Entity)};
  EXPECT_EQ(out_spec.name(), "b");
  EXPECT_EQ(out_spec.io_type(), IOSpec::IOType::kOutput);
}

TEST(IOSpec, TestIOSpecConditionMessageAvailable) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec = IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput,
                       &typeid(holoscan::gxf::Entity));
  EXPECT_EQ(spec.conditions().size(), 0);

  // kMessageAvailable with no params
  spec.condition(ConditionType::kMessageAvailable);

  // kMessageAvailable with one params
  spec.condition(ConditionType::kMessageAvailable, Arg("a", static_cast<size_t>(5)));

  // KMessageAvailable with two params
  spec.condition(ConditionType::kMessageAvailable,
                 Arg("a", static_cast<size_t>(5)),
                 Arg("b", static_cast<size_t>(10)));
  auto condition_pairs = spec.conditions();
  EXPECT_EQ(condition_pairs.size(), 3);
  EXPECT_EQ(condition_pairs[0].second->args().size(), 0);
  EXPECT_EQ(condition_pairs[1].second->args().size(), 1);
  EXPECT_EQ(condition_pairs[2].second->args().size(), 2);
}

TEST(IOSpec, TestIOSpecConditionDownstreamMessageAffordable) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec = IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput,
                       &typeid(holoscan::gxf::Entity));

  // kMessageDownstreamAffordable with no params
  spec.condition(ConditionType::kDownstreamMessageAffordable);

  // kMessageDownstreamAffordable with one params
  spec.condition(ConditionType::kDownstreamMessageAffordable, Arg("a", static_cast<size_t>(5)));
  auto condition_pairs = spec.conditions();
  EXPECT_EQ(condition_pairs.size(), 2);
  EXPECT_EQ(condition_pairs[0].second->args().size(), 0);
  EXPECT_EQ(condition_pairs[1].second->args().size(), 1);
}

TEST(IOSpec, TestIOSpecConditionCount) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec = IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput,
                       &typeid(holoscan::gxf::Entity));

  // kCount with no params
  spec.condition(ConditionType::kCount);

  // kCount with one param
  spec.condition(ConditionType::kCount, Arg("a", static_cast<int64_t>(5)));
  auto condition_pairs = spec.conditions();
  EXPECT_EQ(condition_pairs.size(), 2);
  EXPECT_EQ(condition_pairs[0].second->args().size(), 0);
  EXPECT_EQ(condition_pairs[1].second->args().size(), 1);
}

TEST(IOSpec, TestIOSpecConditionBoolean) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec = IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput,
                       &typeid(holoscan::gxf::Entity));

  // kBoolean with no params
  spec.condition(ConditionType::kBoolean);

  // kBoolean with one param
  spec.condition(ConditionType::kBoolean, Arg("a", static_cast<bool>(true)));
  auto condition_pairs = spec.conditions();
  EXPECT_EQ(condition_pairs.size(), 2);
  EXPECT_EQ(condition_pairs[0].second->args().size(), 0);
  EXPECT_EQ(condition_pairs[1].second->args().size(), 1);
}

TEST(IOSpec, TestIOSpecConditionNone) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec = IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput,
                       &typeid(holoscan::gxf::Entity));

  spec.condition(ConditionType::kNone);
  auto condition_pairs = spec.conditions();
  EXPECT_EQ(condition_pairs.size(), 1);
  EXPECT_EQ(condition_pairs[0].second, nullptr);
}

TEST_F(TestWithGXFContext, TESTIOSpecResource) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec = IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput,
                       &typeid(holoscan::gxf::Entity));
  const std::string name{"unbounded"};
  auto resource = F.make_resource<UnboundedAllocator>(name);
  spec.resource(resource);
}
}  // namespace holoscan
