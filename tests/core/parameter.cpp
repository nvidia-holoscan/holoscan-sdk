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

#include <gtest/gtest.h>

#include <string>
#include <typeinfo>


namespace holoscan {

TEST(Parameter, TestParameterFlagEnum) {
  ParameterFlag p = ParameterFlag::kNone;
  ASSERT_TRUE(p == ParameterFlag::kNone);
  p = ParameterFlag::kOptional;
  ASSERT_TRUE(p == ParameterFlag::kOptional);
  p = ParameterFlag::kDynamic;
  ASSERT_TRUE(p == ParameterFlag::kDynamic);
}

TEST(Parameter, TestParameterWrapper) {
  // default constructor
  ParameterWrapper p;
  EXPECT_EQ(p.type(), typeid(void));

  // 3-argument constructor
  float beta = 3.5;
  ParameterWrapper p3(beta, &typeid(beta), ArgType::create<float>());
  EXPECT_EQ(p3.type(), typeid(float));
  EXPECT_EQ(p3.arg_type().element_type(), ArgElementType::kFloat32);
  EXPECT_EQ(p3.arg_type().container_type(), ArgContainerType::kNative);
  EXPECT_EQ(std::any_cast<float>(p3.value()), beta);

  // construct with Parameter
  uint32_t expected_val = 5;
  MetaParameter uint32_param = MetaParameter<uint32_t>(expected_val);
  ParameterWrapper p2(uint32_param);
  EXPECT_EQ(p2.type(), typeid(uint32_t));
  EXPECT_EQ(p2.arg_type().element_type(), ArgElementType::kUnsigned32);
  EXPECT_EQ(p2.arg_type().container_type(), ArgContainerType::kNative);
  std::any& val2 = p2.value();
  auto& param2 = *std::any_cast<Parameter<uint32_t>*>(val2);
  EXPECT_EQ(param2.get(), expected_val);
}

TEST(Parameter, TestMetaParameterInt) {
  // test initialized metaparameter
  MetaParameter p = MetaParameter<int>(5);
  ASSERT_TRUE(p.has_value());
  int val = p.get();
  ASSERT_EQ(val, 5);

  // value assignment
  int new_val = 7;
  p = new_val;
  ASSERT_EQ(p.get(), 7);

  // const value assignment
  const int new_val2 = 8;
  p = new_val2;
  ASSERT_EQ(p.get(), 8);
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
          EXPECT_STREQ("MetaParameter: value for '' is not set", e.what());
          throw;
        }
      },
      std::runtime_error);
}

}  // namespace holoscan
