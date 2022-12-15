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
#include <gxf/core/gxf.h>

#include <string>

#include "common/assert.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/conditions/gxf/count.hpp"
#include "holoscan/core/conditions/gxf/downstream_affordable.hpp"
#include "holoscan/core/conditions/gxf/message_available.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/graph.hpp"
#include "../utils.hpp"

using namespace std::string_literals;

namespace holoscan {

using ConditionClassesWithGXFContext = TestWithGXFContext;

TEST(ConditionClasses, TestBooleanCondition) {
  Fragment F;
  const std::string name{"boolean-condition"};
  auto condition = F.make_condition<BooleanCondition>(name, Arg{"enable_tick", true});
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<BooleanCondition>(true)));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::BooleanSchedulingTerm"s);
}

TEST(ConditionClasses, TestBooleanConditionEnabledMethods) {
  Fragment F;
  const std::string name{"boolean-condition"};
  auto condition = F.make_condition<BooleanCondition>(name, Arg{"enable_tick", true});

  // can't call check_tick_enabled before it has been set
  // (make_condition does not call initialize() on the condition)
  try {
    condition->check_tick_enabled();
  } catch (const std::runtime_error& e) {
    // and this tests that it has the correct message
    EXPECT_TRUE(std::string(e.what()).find("'enable_tick' is not set") != std::string::npos);
  }

  // check disable and enable
  condition->disable_tick();
  EXPECT_EQ(condition->check_tick_enabled(), false);
  condition->enable_tick();
  EXPECT_EQ(condition->check_tick_enabled(), true);
}

TEST(ConditionClasses, TestBooleanConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<BooleanCondition>();
}

TEST(ConditionClasses, TestCountCondition) {
  Fragment F;
  const std::string name{"count-condition"};
  auto condition = F.make_condition<CountCondition>(name, Arg{"count", 100});
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<CountCondition>(100)));
}

TEST(ConditionClasses, TestCountConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<CountCondition>();
}

TEST(ConditionClasses, TestCountConditionMethods) {
  Fragment F;
  const std::string name{"count-condition"};
  auto condition = F.make_condition<CountCondition>(name, Arg{"count", 100});

  condition->count(20);
  EXPECT_EQ(condition->count(), 20);

  // can access methods of GXFComponent
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::CountSchedulingTerm"s);
}

TEST(ConditionClasses, TestCountConditionGXFComponentMethods) {
  Fragment F;
  const std::string name{"count-condition"};
  auto condition = F.make_condition<CountCondition>(name, Arg{"count", 100});

  auto gxf_typename = condition->gxf_typename();
  auto context = condition->gxf_context();
  auto cid = condition->gxf_cid();
  auto eid = condition->gxf_eid();
}

TEST(ConditionClassesWithGXFContext, TestCountConditionInitializeWithoutSpec) {
  Fragment F;
  CountCondition count{10};
  count.fragment(&F);
  // TODO: avoid segfault if initialize is called before the fragment is assigned

  // test that an error is logged if initialize is called before a spec as assigned
  testing::internal::CaptureStderr();
  count.initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") != std::string::npos);
  EXPECT_TRUE(log_output.find("No component spec") != std::string::npos);
}

TEST(ConditionClassesWithGXFContext, TestCountConditionInitializeWithUnrecognizedArg) {
  Fragment F;
  auto condition = F.make_condition<CountCondition>(Arg{"count", 100}, Arg("undefined_arg", 5.0));

  // test that an warning is logged if an unknown argument is provided
  testing::internal::CaptureStderr();
  condition->initialize();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("warning") != std::string::npos);
  EXPECT_TRUE(log_output.find("'undefined_arg' not found in spec.params") != std::string::npos);
}

TEST(ConditionClasses, TestDownstreamMessageAffordableCondition) {
  Fragment F;
  const std::string name{"downstream-message-affordable-condition"};
  ArgList arglist{
      Arg{"min_size", 1L},
  };
  auto condition = F.make_condition<DownstreamMessageAffordableCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition),
            typeid(std::make_shared<DownstreamMessageAffordableCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()),
            "nvidia::gxf::DownstreamReceptiveSchedulingTerm"s);
}

TEST(ConditionClasses, TestDownstreamMessageAffordableConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<DownstreamMessageAffordableCondition>();
}

TEST(ConditionClasses, TestDownstreamMessageAffordableConditionSizeMethod) {
  Fragment F;
  const std::string name{"downstream-message-affordable-condition"};
  ArgList arglist{
      Arg{"min_size", 1L},
  };
  auto condition = F.make_condition<DownstreamMessageAffordableCondition>(name, arglist);
  condition->min_size(16);
  EXPECT_EQ(condition->min_size(), 16);
}

TEST(ConditionClasses, TestMessageAvailableCondition) {
  Fragment F;
  const std::string name{"message-available-condition"};
  ArgList arglist{Arg{"min_size", 1L}, Arg{"front_stage_max_size", 2L}};
  auto condition = F.make_condition<MessageAvailableCondition>(name, arglist);
  EXPECT_EQ(condition->name(), name);
  EXPECT_EQ(typeid(condition), typeid(std::make_shared<MessageAvailableCondition>(arglist)));
  EXPECT_EQ(std::string(condition->gxf_typename()), "nvidia::gxf::MessageAvailableSchedulingTerm"s);
}

TEST(ConditionClasses, TestMessageAvailableConditionDefaultConstructor) {
  Fragment F;
  auto condition = F.make_condition<MessageAvailableCondition>();
}

TEST(ConditionClasses, TestMessageAvailableConditionSizeMethods) {
  Fragment F;
  const std::string name{"message-available-condition"};
  ArgList arglist{Arg{"min_size", 1L}, Arg{"front_stage_max_size", 2L}};
  auto condition = F.make_condition<MessageAvailableCondition>(name, arglist);

  condition->min_size(3);
  EXPECT_EQ(condition->min_size(), 3);

  condition->front_stage_max_size(5);
  EXPECT_EQ(condition->front_stage_max_size(), 5);
}

}  // namespace holoscan
