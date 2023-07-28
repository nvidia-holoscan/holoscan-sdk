/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/component_spec.hpp"  // must be before argument_setter import

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "../utils.hpp"
#include "dummy_classes.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/argument_setter.hpp"
#include "holoscan/core/parameter.hpp"

namespace holoscan {

using ComponentSpecWithGXFContext = TestWithGXFContext;

TEST(ComponentSpec, TestComponentSpec) {
  ComponentSpec spec;
  EXPECT_EQ(spec.fragment(), nullptr);
  EXPECT_EQ(spec.params().size(), 0);

  // add one parameter named "beta"
  MetaParameter param = Parameter<double>(5.0);
  spec.param(param, "beta");

  // check the stored values
  EXPECT_EQ(spec.params().size(), 1);
  ParameterWrapper w = spec.params()["beta"];
  std::any& val = w.value();
  auto& p = *std::any_cast<Parameter<double>*>(val);
  EXPECT_EQ(p.get(), 5.0);
  EXPECT_EQ(p.key(), "beta");

  // a change to the underlying Parameter object will be reflected in the spec
  param = 8.0;
  EXPECT_EQ(p.get(), 8.0);

  // repeating a key will not add an additional parameter
  spec.param(param, "beta", "headline1");
  EXPECT_EQ(spec.params().size(), 1);

  // add under a new key with both headline and description
  spec.param(param, "beta2", "headline2", "description2");
  EXPECT_EQ(spec.params().size(), 2);
  w = spec.params()["beta2"];
  val = w.value();
  auto& p3 = *std::any_cast<Parameter<double>*>(val);
  EXPECT_EQ(p3.get(), 8.0);
  EXPECT_EQ(p3.key(), "beta2");
}

TEST(ComponentSpec, TestComponentSpecOptional) {
  ComponentSpec spec;

  // add a parameter without any value
  MetaParameter empty_float = Parameter<float>();
  spec.param(empty_float, "beta1", "headline1", "description1", ParameterFlag::kOptional);
  auto params = spec.params();
  EXPECT_EQ(params.size(), 1);

  // verify that the extracted parameter has no value
  ParameterWrapper w = params["beta1"];
  std::any& val = w.value();
  auto& p = *std::any_cast<Parameter<float>*>(val);
  EXPECT_EQ(p.has_value(), false);

  // now set to the default value
  p.set_default_value();  // set default value if not set.
  EXPECT_EQ(p.try_get(), std::nullopt);
  EXPECT_THROW(p.get(), std::runtime_error);
}

TEST(ComponentSpec, TestComponentSpecDefaultLValue) {
  ComponentSpec spec;

  // add a parameter without any value
  uint32_t default_val = 15u;
  MetaParameter empty_int = Parameter<uint32_t>();
  spec.param(empty_int, "beta3", "headline3", "description3", default_val);
  auto params = spec.params();
  EXPECT_EQ(params.size(), 1);

  // verify that the extracted parameter has no value
  ParameterWrapper w = params["beta3"];
  std::any& val = w.value();
  auto& p = *std::any_cast<Parameter<uint32_t>*>(val);
  EXPECT_EQ(p.has_value(), false);

  // now set to the default value
  p.set_default_value();  // set default value if not set.
  EXPECT_EQ(p.get(), default_val);
  EXPECT_EQ((int)p, default_val);
}

TEST(ComponentSpec, TestComponentSpecDefaultRValue) {
  ComponentSpec spec;

  // add a parameter without any value
  Parameter<DummyIntClass> empty_int;
  spec.param(empty_int, "int", "int param", "Example integer parameter.", DummyIntClass{15});
  auto params = spec.params();
  EXPECT_EQ(params.size(), 1);

  // verify that the extracted parameter has no value
  ParameterWrapper w = params["int"];
  std::any& val = w.value();
  auto& p = *std::any_cast<Parameter<DummyIntClass>*>(val);
  EXPECT_EQ(p.has_value(), false);

  // now set to the default value
  p.set_default_value();  // set default value if not set.
  EXPECT_EQ(p.get(), DummyIntClass{15});
  EXPECT_EQ((DummyIntClass)p,  // NOLINT
            DummyIntClass{15});
}

TEST(ComponentSpec, TestComponentSpecEmptyDefault) {
  ComponentSpec spec;

  // initialize
  Parameter<int64_t> empty_p;

  // add one parameter
  // '{}' needs to be treated as a default value, instead of 'ParameterFlag::kNone'.
  spec.param(empty_p, "int64_t", "headline1", "description1", {});
  EXPECT_EQ(spec.params().size(), 1);

  // verify that the extracted parameter has no value
  ParameterWrapper w = spec.params()["int64_t"];
  std::any& val = w.value();
  auto& p = *std::any_cast<Parameter<int64_t>*>(val);
  EXPECT_EQ(p.has_value(), false);

  // set to the default value
  p.set_default_value();
  EXPECT_EQ(p.get(), 0);
  EXPECT_EQ(static_cast<int64_t>(p), 0);
}

TEST(ComponentSpec, TestComponentSpecDescriptionWithoutFragment) {
  ComponentSpec spec;
  Parameter<bool> b;
  Parameter<std::array<int, 5>> i;
  Parameter<std::vector<std::vector<double>>> d;
  Parameter<std::vector<std::string>> s;
  spec.param(b, "bool_scalar", "Boolean parameter", "true or false");
  spec.param(i, "int_array", "Int array parameter", "5 integers");
  spec.param(
      d, "double_vec_of_vec", "Double 2D vector parameter", "double floats in double vector");
  spec.param(s, "string_vector", "String vector parameter", "");

  constexpr auto description = R"(fragment: ~
params:
  - name: string_vector
    type: std::vector<std::string>
    description: ""
    flag: kNone
  - name: double_vec_of_vec
    type: std::vector<std::vector<double>>
    description: double floats in double vector
    flag: kNone
  - name: int_array
    type: std::array<int32_t,N>
    description: 5 integers
    flag: kNone
  - name: bool_scalar
    type: bool
    description: true or false
    flag: kNone)";
  EXPECT_EQ(spec.description(), description);
}

TEST_F(ComponentSpecWithGXFContext, TestComponentSpecDescription) {
  auto spec = std::make_shared<ComponentSpec>(&F);
  Parameter<bool> b;
  Parameter<std::array<int, 5>> i;
  Parameter<std::vector<std::vector<double>>> d;
  Parameter<std::vector<std::string>> s;
  spec->param(b, "bool_scalar", "Boolean parameter", "true or false");
  spec->param(i, "int_array", "Int array parameter", "5 integers");
  spec->param(
      d, "double_vec_of_vec", "Double 2D vector parameter", "double floats in double vector");
  spec->param(s, "string_vector", "String vector parameter", "");

  constexpr auto description = R"(fragment: ""
params:
  - name: string_vector
    type: std::vector<std::string>
    description: ""
    flag: kNone
  - name: double_vec_of_vec
    type: std::vector<std::vector<double>>
    description: double floats in double vector
    flag: kNone
  - name: int_array
    type: std::array<int32_t,N>
    description: 5 integers
    flag: kNone
  - name: bool_scalar
    type: bool
    description: true or false
    flag: kNone)";

  EXPECT_EQ(spec->description(), description);
}
}  // namespace holoscan
