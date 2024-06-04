/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <typeinfo>

#include "dummy_classes.hpp"

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
  ASSERT_EQ(p.get(), new_val);

  // const value assignment
  const int new_val2 = 8;
  p = new_val2;
  ASSERT_EQ(p.get(), new_val2);
}

TEST(Parameter, TestMetaParameterComplex) {
  // test initialized metaparameter
  MetaParameter p = MetaParameter<std::complex<float>>({5.0, -2.0});
  ASSERT_TRUE(p.has_value());
  std::complex<float> val = p.get();
  ASSERT_EQ(val, std::complex<float>(5.0, -2.0));

  // value assignment
  auto new_val = std::complex<float>(7.5, 3.0);
  p = new_val;
  ASSERT_EQ(p.get(), new_val);

  // const value assignment
  const std::complex<float> new_val2 = new_val;
  p = new_val2;
  ASSERT_EQ(p.get(), new_val2);
}

TEST(Parameter, TestMetaParameterTryGet) {
  // check try_get() returning std::nullopt
  MetaParameter p = MetaParameter<int>();
  EXPECT_FALSE(p.try_get().has_value());
  EXPECT_EQ(p.try_get(), std::nullopt);

  // r-value (expecting move)
  Parameter<DummyIntClass> int_param{DummyIntClass{15}};
  auto& int_param_val = int_param.try_get();
  EXPECT_EQ(int_param_val.has_value(), true);
  int_param_val.value().set(20);
  EXPECT_EQ(int_param.get(), DummyIntClass{20});

  // l-value (expecting copy)
  auto dummy_val2 = DummyIntClass{15};
  Parameter<DummyIntClass> int_param2{dummy_val2};
  auto& int_param_val2 = int_param2.try_get();
  EXPECT_EQ(int_param_val2.has_value(), true);
  int_param_val2.value().set(20);

  EXPECT_EQ(int_param2.get(), DummyIntClass{20});
  EXPECT_NE(int_param2.get(), dummy_val2);
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

TEST(Parameter, TestArrowOperator) {
  std::string str = "hello";

  Parameter<std::string*> param_str_ptr;
  param_str_ptr = &str;
  ASSERT_EQ(param_str_ptr->size(), 5);

  Parameter<std::shared_ptr<std::string>> param_str_shared_ptr;
  param_str_shared_ptr = std::make_shared<std::string>(str);
  ASSERT_EQ(param_str_shared_ptr->size(), 5);
}

TEST(Parameter, TestAsteriskOperator) {
  int data = 10;

  Parameter<int*> param_int_ptr;
  param_int_ptr = &data;
  ASSERT_EQ(*param_int_ptr, 10);

  Parameter<std::shared_ptr<int>> param_int_shared_ptr;
  param_int_shared_ptr = std::make_shared<int>(data);
  ASSERT_EQ(*param_int_shared_ptr, 10);
}

// simple test for formatter feature
TEST(Parameter, TestMetaParameterFormatter) {
  MetaParameter p = MetaParameter<int>(5);
  std::string format_message = fmt::format("{}", p);
  EXPECT_EQ(format_message, std::string{"5"});
  // capture output so that we can check that the expected value was logged
  testing::internal::CaptureStderr();
  HOLOSCAN_LOG_INFO("Formatted parameter value: {}", p);
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("5") != std::string::npos);
}

// simple test with format option for formatter feature
TEST(Parameter, TestMetaParameterFormatterSyntax) {
  MetaParameter seconds = MetaParameter<float>(1.32);
  std::string format_message = fmt::format("{:0.3f}", seconds);
  EXPECT_EQ(format_message, std::string{"1.320"});
  // capture output so that we can check that the expected value was logged
  testing::internal::CaptureStderr();
  HOLOSCAN_LOG_INFO("Formatted parameter value: {:0.3f}", seconds);
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("1.320") != std::string::npos);
}

}  // namespace holoscan
