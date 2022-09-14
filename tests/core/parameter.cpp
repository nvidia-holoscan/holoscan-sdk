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

#include "holoscan/core/parameter.hpp"

#include <string>

#include <gtest/gtest.h>

namespace holoscan {

TEST(Parameter, TestParameterFlagEnum) {
  ParameterFlag p = ParameterFlag::kNone;
  ASSERT_TRUE(p == ParameterFlag::kNone);
  p = ParameterFlag::kOptional;
  ASSERT_TRUE(p == ParameterFlag::kOptional);
  p = ParameterFlag::kDynamic;
  ASSERT_TRUE(p == ParameterFlag::kDynamic);
}

TEST(Parameter, TestMetaParameterInt) {
  // test initialized metaparameter
  MetaParameter p = MetaParameter<int>(5);
  ASSERT_TRUE(p.has_value());
  int val = p.get();
  ASSERT_TRUE(val == 5);

  // value assignment
  int new_val = 7;
  p = new_val;
  ASSERT_TRUE(p.get() == 7);

  // const value assignment
  const int new_val2 = 8;
  p = new_val2;
  ASSERT_TRUE(p.get() == 8);
}

TEST(Parameter, TestMetaParameterUninitialized) {
  // test uninitialized metaparameter
  MetaParameter p2 = MetaParameter<double>();
  ASSERT_FALSE(p2.has_value());

  EXPECT_THROW(
      {
        try {
          p2.get();
        } catch (const std::runtime_error& e) {
          // and this tests that it has the correct message
          EXPECT_STREQ("MetaParameter: value is not set", e.what());
          throw;
        }
      },
      std::runtime_error);
}

}  // namespace holoscan
