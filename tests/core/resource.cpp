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

#include <string>

#include "holoscan/core/resource.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

TEST(Resource, TestResourceName) {
  // initialization
  Resource R = Resource();
  EXPECT_EQ(R.name(), "");

  R.name("my_name");
  Fragment* f;
  R.fragment(f);

  EXPECT_EQ(R.name(), "my_name");
}

TEST(Resource, TestResourceFragment) {
  // initialization
  Resource R = Resource();
  EXPECT_EQ(R.fragment(), nullptr);

  Fragment* f;
  R.fragment(f);
  EXPECT_EQ(R.fragment(), f);
}

TEST(Resource, TestResourceSpec) {
  // initialization
  Resource R = Resource();
  Fragment* f;

  R.spec(std::make_shared<ComponentSpec>(f));
}

TEST(Resource, TestResourceSpecFragmentNull) {
  // initialization
  Resource R = Resource();
  Fragment* f;

  // component spec can take in nullptr fragment
  R.spec(std::make_shared<ComponentSpec>());
}

TEST(Resource, TestResourceChainedAssignments) {
  // initialization
  Resource R;
  Fragment *f1, *f2, *f3;

  R.fragment(f1).name("name1");
  EXPECT_EQ(R.fragment(), f1);
  EXPECT_EQ(R.name(), "name1");

  R.name("name2").fragment(f2);
  EXPECT_EQ(R.fragment(), f2);
  EXPECT_EQ(R.name(), "name2");

  R.spec(std::make_shared<ComponentSpec>(f3)).name("name3").fragment(f3);
  EXPECT_EQ(R.fragment(), f3);
  EXPECT_EQ(R.name(), "name3");
}

TEST(Resource, TestResourceSetup) {
  // initialization
  Resource R = Resource();
  ComponentSpec spec;

  R.setup(spec);
}
}  // namespace holoscan
