/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <holoscan/core/arg.hpp>
#include <holoscan/core/component.hpp>
#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/condition.hpp>
#include <holoscan/core/fragment.hpp>

namespace holoscan {

TEST(Condition, TestConditionName) {
  // initialization
  Condition C = Condition();
  EXPECT_EQ(C.name(), "");

  C.name("my_name");
  Fragment f;
  C.fragment(&f);

  EXPECT_EQ(C.name(), "my_name");
}

TEST(Condition, TestConditionFragment) {
  // initialization
  Condition C = Condition();
  EXPECT_EQ(C.fragment(), nullptr);

  Fragment f;
  C.fragment(&f);
  EXPECT_EQ(C.fragment(), &f);
}

TEST(Condition, TestConditionSpec) {
  // initialization
  Condition C = Condition();
  Fragment f;

  C.spec(std::make_shared<ComponentSpec>(&f));
}

TEST(Condition, TestConditionChainedAssignments) {
  // initialization
  Condition C;
  Fragment f1, f2, f3;

  C.fragment(&f1).name("name1");
  EXPECT_EQ(C.fragment(), &f1);
  EXPECT_EQ(C.name(), "name1");

  C.name("name2").fragment(&f2);
  EXPECT_EQ(C.fragment(), &f2);
  EXPECT_EQ(C.name(), "name2");

  C.spec(std::make_shared<ComponentSpec>(&f3)).name("name3").fragment(&f3);
  EXPECT_EQ(C.fragment(), &f3);
  EXPECT_EQ(C.name(), "name3");
}

TEST(Condition, TestConditionSpecFragmentNull) {
  // initialization
  Condition C = Condition();

  // component spec can take in nullptr fragment
  C.spec(std::make_shared<ComponentSpec>());
}

TEST(Condition, TestConditionSetup) {
  // initialization
  Condition C = Condition();
  ComponentSpec spec;

  C.setup(spec);
}

}  // namespace holoscan
