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

#include <bits/stdc++.h>  // unordered map find
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <unordered_map>
#include <complex>
#include <cstdint>
#include <string>
#include <typeinfo>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"  // must be before argument_setter import
#include "holoscan/core/argument_setter.hpp"

namespace holoscan {

TEST(OperatorSpec, TestOperatorSpecInput) {
  testing::internal::CaptureStderr();

  OperatorSpec spec = OperatorSpec();
  spec.input<gxf::Entity>("a");

  // check if key "a" exist
  EXPECT_TRUE(spec.inputs().find("a") != spec.inputs().end());

  // check size
  EXPECT_EQ(spec.inputs().size(), 1);

  // duplicate name
  spec.input<gxf::Entity>("a");
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("already exists") != std::string::npos);
}

TEST(OperatorSpec, TestOperatorSpectOutput) {
  testing::internal::CaptureStderr();

  OperatorSpec spec = OperatorSpec();
  spec.output<gxf::Entity>("a");

  // check if key "a" exist
  EXPECT_TRUE(spec.outputs().find("a") != spec.outputs().end());

  // check size
  EXPECT_EQ(spec.outputs().size(), 1);

  // duplicate name
  spec.output<gxf::Entity>("a");
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("already exists") != std::string::npos);
}

TEST(OperatorSpec, TestOperatorSpecParam) {
  testing::internal::CaptureStderr();

  OperatorSpec spec = OperatorSpec();
  EXPECT_EQ(spec.fragment(), nullptr);

  // add one parameter named "beta"
  IOSpec in{&spec, "a", IOSpec::IOType::kInput, &typeid(gxf::Entity)};
  IOSpec* in_ptr = &in;
  MetaParameter meta_params = Parameter<holoscan::IOSpec*>(in_ptr);

  spec.param(meta_params, "beta", "headline1", "description1");

  // check the stored values
  EXPECT_EQ(spec.params().size(), 1);
  ParameterWrapper w = spec.params()["beta"];
  std::any& val = w.value();
  auto& p = *std::any_cast<Parameter<holoscan::IOSpec*>*>(val);
  EXPECT_EQ(p.key(), "beta");
  EXPECT_EQ(p.get(), in_ptr);

  // repeating a key will not add an additional parameter
  spec.param(p, "beta", "headline1", "description4");
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("already exists") != std::string::npos);
  EXPECT_EQ(spec.params().size(), 1);
}

TEST(OperatorSpec, TestOperatorSpecParamDefault) {
  OperatorSpec spec = OperatorSpec();

  // initialize
  IOSpec default_input{&spec, "a", IOSpec::IOType::kInput, &typeid(gxf::Entity)};
  IOSpec* default_input_ptr = &default_input;
  MetaParameter empty_p = Parameter<holoscan::IOSpec*>();  // set val

  // add one parameter
  spec.param(empty_p, "beta", "headline1", "description1", default_input_ptr);
  EXPECT_EQ(spec.params().size(), 1);

  // verify that the extracted parameter has no value
  ParameterWrapper w = spec.params()["beta"];
  std::any& val = w.value();
  auto& p = *std::any_cast<Parameter<holoscan::IOSpec*>*>(val);
  EXPECT_EQ(p.has_value(), false);

  // set to the default value
  p.set_default_value();
  EXPECT_EQ(p.get(), default_input_ptr);
  EXPECT_EQ((IOSpec*)p, default_input_ptr);
}

TEST(OperatorSpec, TestOperatorSpecParamVector) {
  testing::internal::CaptureStderr();

  // initialize
  OperatorSpec spec = OperatorSpec();

  IOSpec in1{&spec, "a", IOSpec::IOType::kInput, &typeid(gxf::Entity)};
  IOSpec* in1_ptr = &in1;
  IOSpec in2{&spec, "b", IOSpec::IOType::kInput, &typeid(gxf::Entity)};
  IOSpec* in2_ptr = &in2;
  IOSpec in3{&spec, "c", IOSpec::IOType::kInput, &typeid(gxf::Entity)};
  IOSpec* in3_ptr = &in3;
  std::vector<IOSpec*> v = {in1_ptr, in2_ptr, in3_ptr};

  // add one parameter named "beta"
  MetaParameter meta_params = Parameter<std::vector<holoscan::IOSpec*>>(v);
  spec.param(meta_params, "beta", "headline1", "description1");

  // check the stored values
  EXPECT_EQ(spec.params().size(), 1);
  ParameterWrapper w = spec.params()["beta"];
  std::any& val = w.value();
  auto& p = *std::any_cast<Parameter<std::vector<holoscan::IOSpec*>>*>(val);
  EXPECT_EQ(p.key(), "beta");
  EXPECT_EQ(p.get(), v);

  // repeating a key will not add an additional parameter
  spec.param(p, "beta", "headline1", "description1");
  EXPECT_EQ(spec.params().size(), 1);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("already exists") != std::string::npos);
  EXPECT_EQ(spec.params().size(), 1);
}

TEST(OperatorSpec, TestOperatorSpecParamVectorDefault) {
  OperatorSpec spec = OperatorSpec();

  // initialize
  IOSpec in1{&spec, "a", IOSpec::IOType::kInput, &typeid(gxf::Entity)};
  IOSpec* in1_ptr = &in1;
  IOSpec in2{&spec, "b", IOSpec::IOType::kInput, &typeid(gxf::Entity)};
  IOSpec* in2_ptr = &in2;
  IOSpec in3{&spec, "c", IOSpec::IOType::kInput, &typeid(gxf::Entity)};
  IOSpec* in3_ptr = &in3;
  std::vector<IOSpec*> default_v = {in1_ptr, in2_ptr, in3_ptr};
  MetaParameter empty_p = Parameter<std::vector<holoscan::IOSpec*>>();

  // add one parameter
  spec.param(empty_p, "beta", "headline1", "description1", default_v);
  EXPECT_EQ(spec.params().size(), 1);

  // verify that the extracted parameter has no value
  ParameterWrapper w = spec.params()["beta"];
  std::any& val = w.value();
  auto& p = *std::any_cast<Parameter<std::vector<holoscan::IOSpec*>>*>(val);
  EXPECT_EQ(p.has_value(), false);

  // set to the default value
  p.set_default_value();
  EXPECT_EQ(p.get(), default_v);
  EXPECT_EQ((std::vector<IOSpec*>)p, default_v);
}

TEST(OperatorSpec, TestOperatorSpecDescription) {
  OperatorSpec spec;
  spec.input<gxf::Entity>("gxf_entity_in");
  spec.output<gxf::Entity>("gxf_entity_out");
  spec.input<holoscan::Tensor>("holoscan_tensor_in");
  spec.output<holoscan::Tensor>("holoscan_tensor_out");

  Parameter<bool> b;
  Parameter<std::array<int, 5>> i;
  Parameter<std::vector<std::vector<double>>> d;
  Parameter<std::vector<std::string>> s;
  spec.param(b, "bool_scalar", "Boolean parameter", "true or false");
  spec.param(i, "int_array", "Int array parameter", "5 integers");
  spec.param(
      d, "double_vec_of_vec", "Double 2D vector parameter", "double floats in double vector");
  spec.param(s, "string_vector", "String vector parameter", "");

  std::string description = fmt::format(R"({}
inputs:
  - holoscan_tensor_in
  - gxf_entity_in
outputs:
  - holoscan_tensor_out
  - gxf_entity_out)",
                                        ComponentSpec(spec).description());
  EXPECT_EQ(spec.description(), description);
}
}  // namespace holoscan
