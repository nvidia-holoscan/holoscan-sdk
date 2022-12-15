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

#include "holoscan/core/component.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/fragment.hpp"

namespace holoscan {

TEST(Component, TestComponentDefaultConstructor) {
  Component C = Component();
  EXPECT_EQ(C.fragment(), nullptr);
  EXPECT_EQ(C.name(), "");
}

TEST(Component, TestComponentAddArg) {
  Component C = Component();

  // add arg
  Arg a = Arg("alpha1");
  a = 1.5;
  C.add_arg(a);

  // add ConstArg
  Arg a2 = Arg("alpha2");
  a2 = 3.0f;
  const Arg ca2 = a2;
  C.add_arg(ca2);

  // add ArgList
  ArgList arglist{Arg("a"), Arg("b")};
  C.add_arg(arglist);

  // add const ArgList
  const ArgList carglist{Arg("c"), Arg("d")};
  C.add_arg(carglist);
}

TEST(Component, TestComponentInitialize) {
  Component C = Component();

  // initialize method exists
  C.initialize();
}

TEST(Component, TestComponentArgDuplicateName) {
  Component C = Component();

  // add arg
  Arg a = Arg("alpha1");
  a = 1.5;
  C.add_arg(a);

  // no error if a second argument with the same name is added
  a = 2.5;  // change in value here will not modify the value already in C.args_
  C.add_arg(a);

  auto args = C.args();
  ASSERT_EQ(args.size(), 2);
  ASSERT_EQ(std::any_cast<double>(args[0].value()), 1.5);
  ASSERT_EQ(std::any_cast<double>(args[1].value()), 2.5);
}

}  // namespace holoscan
