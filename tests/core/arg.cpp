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

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <any>
#include <complex>
#include <cstdint>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "../config.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"

using namespace std::string_literals;

static HoloscanTestConfig test_config;

namespace holoscan {

// testing::AssertionResult
void check_arg_types(Arg A, ArgElementType ExpectedElementType,
                     ArgContainerType ExpectedContainerType) {
  ArgType T = A.arg_type();
  EXPECT_EQ(T.element_type(), ExpectedElementType);
  EXPECT_EQ(T.container_type(), ExpectedContainerType);
}

struct CustomType {
  int a;
  float b;
};

TEST(Arg, TestArgType) {
  // default constructor
  ArgType A = ArgType();
  EXPECT_EQ(A.element_type(), ArgElementType::kCustom);
  EXPECT_EQ(A.container_type(), ArgContainerType::kNative);
  EXPECT_EQ(A.dimension(), 0);
  EXPECT_EQ(A.to_string(), "CustomType");

  // constructor
  A = ArgType(ArgElementType::kInt8, ArgContainerType::kArray, 1);
  EXPECT_EQ(A.element_type(), ArgElementType::kInt8);
  EXPECT_EQ(A.container_type(), ArgContainerType::kArray);
  EXPECT_EQ(A.dimension(), 1);
  EXPECT_EQ(A.to_string(), "std::array<int8_t,N>");

  // initialize a vector of float via ArgType::create
  A = ArgType::create<std::vector<float>>();
  EXPECT_EQ(A.element_type(), ArgElementType::kFloat32);
  EXPECT_EQ(A.container_type(), ArgContainerType::kVector);
  EXPECT_EQ(A.dimension(), 1);
  EXPECT_EQ(A.to_string(), "std::vector<float>");

  // initialize a vector of vector of string via ArgType::create
  A = ArgType::create<std::vector<std::vector<std::string>>>();
  EXPECT_EQ(A.element_type(), ArgElementType::kString);
  EXPECT_EQ(A.container_type(), ArgContainerType::kVector);
  EXPECT_EQ(A.dimension(), 2);
  EXPECT_EQ(A.to_string(), "std::vector<std::vector<std::string>>");

  // initialize an array of int8_t via ArgType::create
  A = ArgType::create<std::array<std::int8_t, 5>>();
  EXPECT_EQ(A.element_type(), ArgElementType::kInt8);
  EXPECT_EQ(A.container_type(), ArgContainerType::kArray);
  EXPECT_EQ(A.dimension(), 1);
  EXPECT_EQ(A.to_string(), "std::array<int8_t,N>");

  // Verify get_element_type returns the expected type
  EXPECT_EQ(ArgType::get_element_type(std::type_index(typeid(5.0))), ArgElementType::kFloat64);

  // Note: more thorough tests of other types are performed via TestArg
}

using ParamTuple = std::tuple<std::any, ArgElementType, ArgContainerType>;

class ArgParameterizedTestFixture : public ::testing::TestWithParam<ParamTuple> {
 protected:
  Arg A = Arg("alpha");
};

void any_to_arg(const std::any& value, Arg& arg) {
  if (value.type() == typeid(double)) arg = std::any_cast<double>(value);
  else if (value.type() == typeid(float))
    arg = std::any_cast<float>(value);
  else if (value.type() == typeid(bool))
    arg = std::any_cast<bool>(value);
  else if (value.type() == typeid(uint8_t))
    arg = std::any_cast<uint8_t>(value);
  else if (value.type() == typeid(uint16_t))
    arg = std::any_cast<uint16_t>(value);
  else if (value.type() == typeid(uint32_t))
    arg = std::any_cast<uint32_t>(value);
  else if (value.type() == typeid(uint64_t))
    arg = std::any_cast<uint64_t>(value);
  else if (value.type() == typeid(int8_t))
    arg = std::any_cast<int8_t>(value);
  else if (value.type() == typeid(int16_t))
    arg = std::any_cast<int16_t>(value);
  else if (value.type() == typeid(int32_t))
    arg = std::any_cast<int32_t>(value);
  else if (value.type() == typeid(int64_t))
    arg = std::any_cast<int64_t>(value);
  else if (value.type() == typeid(std::complex<float>))
    arg = std::any_cast<std::complex<float>>(value);
  else if (value.type() == typeid(std::complex<double>))
    arg = std::any_cast<std::complex<double>>(value);
  else if (value.type() == typeid(YAML::Node))
    arg = std::any_cast<YAML::Node>(value);
  else if (value.type() == typeid(std::string))
    arg = std::any_cast<std::string>(value);
  else if (value.type() == typeid(CustomType))
    arg = std::any_cast<CustomType>(value);
  else if (value.type() == typeid(std::vector<std::string>))
    arg = std::any_cast<std::vector<std::string>>(value);
  else if (value.type() == typeid(std::array<std::uint16_t, 4>))
    arg = std::any_cast<std::array<std::uint16_t, 4>>(value);
}

TEST_P(ArgParameterizedTestFixture, ArgElementAndContainerTypes) {
  auto [value, ExpectedElementType, ExpectedContainerType] = GetParam();
  any_to_arg(value, A);
  check_arg_types(A, ExpectedElementType, ExpectedContainerType);
}

INSTANTIATE_TEST_CASE_P(
    ArgTests, ArgParameterizedTestFixture,
    ::testing::Values(
        ParamTuple{7.0, ArgElementType::kFloat64, ArgContainerType::kNative},
        ParamTuple{7.0F, ArgElementType::kFloat32, ArgContainerType::kNative},
        ParamTuple{false, ArgElementType::kBoolean, ArgContainerType::kNative},
        ParamTuple{std::int8_t{1}, ArgElementType::kInt8, ArgContainerType::kNative},
        ParamTuple{std::int16_t{1}, ArgElementType::kInt16, ArgContainerType::kNative},
        ParamTuple{std::int32_t{1}, ArgElementType::kInt32, ArgContainerType::kNative},
        ParamTuple{std::int64_t{1}, ArgElementType::kInt64, ArgContainerType::kNative},
        ParamTuple{std::uint8_t{1}, ArgElementType::kUnsigned8, ArgContainerType::kNative},
        ParamTuple{std::uint16_t{1}, ArgElementType::kUnsigned16, ArgContainerType::kNative},
        ParamTuple{std::uint32_t{1}, ArgElementType::kUnsigned32, ArgContainerType::kNative},
        ParamTuple{std::uint64_t{1}, ArgElementType::kUnsigned64, ArgContainerType::kNative},
        ParamTuple{
            std::complex<float>{1.0, 2.0}, ArgElementType::kComplex64, ArgContainerType::kNative},
        ParamTuple{std::complex<double>{1.1, -2.5},
                   ArgElementType::kComplex128,
                   ArgContainerType::kNative},
        ParamTuple{YAML::Node(), ArgElementType::kYAMLNode, ArgContainerType::kNative},
        ParamTuple{std::string{"abcd"}, ArgElementType::kString, ArgContainerType::kNative},
        ParamTuple{CustomType(), ArgElementType::kCustom, ArgContainerType::kNative},
        ParamTuple{std::vector<std::string>{"abcd", "ef", "ghijklm"},
                   ArgElementType::kString,
                   ArgContainerType::kVector},
        ParamTuple{std::array<std::uint16_t, 4>{1, 8, 9, 15},
                   ArgElementType::kUnsigned16,
                   ArgContainerType::kArray}));

// name, value, value as string
static std::vector<std::tuple<const char*, std::any, std::optional<std::vector<std::string>>>>
    arg_params = {
        {"A", 7.0, {{"7"}}},
        {"B", 7.0F, {{"7"}}},
        {"C", false, {{"false"}}},
        {"D", int8_t{1}, {{"1"}}},
        {"E", int16_t{1}, {{"1"}}},
        {"F", int32_t{1}, {{"1"}}},
        {"G", int64_t{1}, {{"1"}}},
        {"H", uint8_t{1}, {{"1"}}},
        {"I", uint16_t{1}, {{"1"}}},
        {"J", uint32_t{1}, {{"1"}}},
        {"K", uint64_t{1}, {{"1"}}},
        {"cf", std::complex<float>{1.0, 2.0}, {{"1+2j"}}},
        {"cd", std::complex<double>{1.14, -2.503}, {{"1.14-2.503j"}}},
        {"L", YAML::Node(YAML::NodeType::Null), {{"~"}}},
        {"M", std::string{"abcd"}, {{"abcd"}}},
        {"N", std::vector<std::string>{"abcd", "ef", "ghijklm"}, {{"abcd", "ef", "ghijklm"}}},
        {"O", std::array<uint16_t, 4>{1, 8, 9, 15}, std::nullopt},  // can't guess size
        {"Z", CustomType(), std::nullopt},                          // Unknown type
        {"Z", std::any(), std::nullopt},                            // Unknown type
};

TEST(Arg, TestArgHandleType) {
  // initialize without any value
  Arg A = Arg("alpha");
  ASSERT_FALSE(A.has_value());
  EXPECT_EQ(A.name(), "alpha");

  // test value()
  A = 5.0;
  std::any val = A.value();
  EXPECT_EQ(std::any_cast<double>(val), 5.0);

  // assign a std:any handle
  A = std::any{1};
  check_arg_types(A, ArgElementType::kHandle, ArgContainerType::kNative);

  // test vector of std::any
  A = std::vector<std::any>{"abcd", 5, 3.0};
  check_arg_types(A, ArgElementType::kHandle, ArgContainerType::kVector);
}

TEST(Arg, TestCondition) {
  Fragment F;
  auto bool_cond = F.make_condition<BooleanCondition>("boolean"s, Arg{"enable_tick", true});
  Arg condition_arg{"bool cond", bool_cond};
  check_arg_types(condition_arg, ArgElementType::kCondition, ArgContainerType::kNative);
}

TEST(Arg, TestResource) {
  Fragment F;
  auto allocator = F.make_resource<UnboundedAllocator>("unbounded");
  Arg resource_arg{"allocator", allocator};
  check_arg_types(resource_arg, ArgElementType::kResource, ArgContainerType::kNative);
}

TEST(Arg, TestIOSpec) {
  OperatorSpec op_spec = OperatorSpec();
  IOSpec spec =
      IOSpec(&op_spec, std::string("a"), IOSpec::IOType::kInput, &typeid(holoscan::gxf::Entity));
  Arg spec_arg{"iospec", &spec};
  check_arg_types(spec_arg, ArgElementType::kIOSpec, ArgContainerType::kNative);
}

TEST(Arg, TestArgDescription) {
  for (const auto& [name, value, str_list] : arg_params) {
    Arg arg = Arg(name);
    any_to_arg(value, arg);
    std::string description = fmt::format("name: {}\ntype: {}", name, arg.arg_type().to_string());
    if (str_list) {
      description = fmt::format("{}\nvalue:", description);
      if (str_list->size() == 1) {
        description = fmt::format("{} {}", description, str_list->at(0));
      } else {
        for (const std::string& val_str : str_list.value()) {
          description = fmt::format("{}\n  - {}", description, val_str);
        }
      }
    }
    EXPECT_EQ(arg.description(), description);
  }
}

TEST(Arg, TestArgList) {
  ArgList args;
  EXPECT_EQ(args.size(), 0);
  args.add(Arg("in"));
  args.add(Arg("out"));
  EXPECT_EQ(args.size(), 2);
  Arg param = Arg("param");
  args.add(param);
  EXPECT_EQ(args.size(), 3);
  const Arg c = Arg("c");
  args.add(c);
  EXPECT_EQ(args.size(), 4);

  // initializer_list constructor
  Arg B{"b"};
  B = 5.0;
  ArgList args2{Arg("a"), B};
  EXPECT_EQ(args2.size(), 2);

  // can add another ArgList
  args.add(args2);
  EXPECT_EQ(args.size(), 6);
  args2.clear();
  EXPECT_EQ(args.size(), 6);
  EXPECT_EQ(args2.size(), 0);

  // change value stored in B
  B = 8;

  // we can iterate over ArgList
  std::string arg_name;
  std::vector<std::string> expected_names = {"in", "out", "param", "c", "a", "b"};
  int cnt = 0;
  std::any val;
  for (Arg arg : args) {
    arg_name = arg.name();
    EXPECT_EQ(arg_name, expected_names[cnt]);
    if (arg_name == "b") {
      val = arg.value();
      // same that was in B when it was added to args
      EXPECT_EQ(std::any_cast<double>(val), 5.0);
    }
    cnt++;
  }

  // can also access the vector of args
  std::vector<Arg> argvec = args.args();
  EXPECT_EQ(argvec.size(), 6);

  // clear all arguments from the list
  args.clear();
  EXPECT_EQ(args.size(), 0);

  // check name (no way to set this, though?)
  EXPECT_EQ(args.name(), "arglist");
}

TEST(Arg, TestArgList2) {
  std::string s1 = "abcd";
  ArgList arguments{
      Arg("double_arg", 5.0), Arg("uint8_arg", static_cast<std::uint8_t>(5)), Arg("str_arg", s1)};

  EXPECT_EQ(arguments.size(), 3);
  auto args = arguments.args();
  Arg a0 = args[0];
  EXPECT_EQ(a0.name(), "double_arg");
  EXPECT_EQ(std::any_cast<double>(a0.value()), 5.0);
  Arg a1 = args[1];
  EXPECT_EQ(a1.name(), "uint8_arg");
  EXPECT_EQ(std::any_cast<std::uint8_t>(a1.value()), 5);
  Arg a2 = args[2];
  EXPECT_EQ(a2.name(), "str_arg");
  EXPECT_EQ(std::any_cast<std::string>(a2.value()), s1);
}

TEST(Arg, TestArgListAs) {
  Fragment F;
  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  F.config(config_file);

  // use from_config so that the Arg contained has YAML::Node type as required by as<T>()
  ArgList args = F.from_config("value");
  EXPECT_EQ(args.size(), 1);
  double v = args.as<double>();
  EXPECT_EQ(v, 5.3);
  // could also cast to string
  std::string vstr = args.as<std::string>();
  EXPECT_EQ(vstr, std::string{"5.3"});

  // after adding a second element to the ArgList
  // as<T>() still returns only the first element
  args.add(Arg{"value2", 8.0});
  double v2 = args.as<double>();
  EXPECT_EQ(v2, 5.3);
}

TEST(Arg, TestArgListAsError) {
  Fragment F;
  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  F.config(config_file);

  // use from_config so that the Arg contained has YAML::Node type as required by as<T>()
  ArgList args = F.from_config("value");

  testing::internal::CaptureStderr();

  // if the YAML node cannot be parsed a default constructed value is returned
  int v_int = args.as<int>();
  EXPECT_EQ(v_int, int{});

  // an error will have been logged about the failed parsing
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") != std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Unable to parse YAML node") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Arg, TestArgListDescription) {
  ArgList args;
  std::string description = "name: arglist\nargs:";
  for (const auto& [name, value, _] : arg_params) {
    Arg arg = Arg(name);
    any_to_arg(value, arg);
    std::string indented_arg_description =
        std::regex_replace(arg.description(), std::regex("\n"), "\n    ");
    args.add(arg);
    description = fmt::format("{}\n  - {}", description, indented_arg_description);
  }
  EXPECT_EQ(args.description(), description);
}

TEST(Yaml, TestYamlCplxDecode) {
  // works with spaces around + and with "j" to indicate imaginary component
  YAML::Node node = YAML::Load("2.0 + 1.5j");
  std::complex<float> cf = node.as<std::complex<float>>();
  EXPECT_EQ(cf.real(), 2.0F);
  EXPECT_EQ(cf.imag(), 1.5F);

  // works without white space and with "i" to indicate imaginary component
  node = YAML::Load("-2.102-3i");
  std::complex<double> cd = node.as<std::complex<double>>();
  EXPECT_EQ(cd.real(), -2.102);
  EXPECT_EQ(cd.imag(), -3.0);
}

TEST(Yaml, TestYamlCplxDecodeInvalid) {
  // invalid case (missing "i" or "j" on second number)
  YAML::Node node = YAML::Load("-2 + 3");
  std::complex<double> cd2;
  EXPECT_THROW({ cd2 = node.as<std::complex<double>>(); }, std::runtime_error);

  // invalid case ("i" or "j" on first number)
  node = YAML::Load("-2i + 3");
  EXPECT_THROW({ cd2 = node.as<std::complex<double>>(); }, std::runtime_error);

  // invalid case ("k" is not a valid imaginary component indicator)
  node = YAML::Load("-2 + 3k");
  EXPECT_THROW({ cd2 = node.as<std::complex<double>>(); }, std::runtime_error);
}

}  // namespace holoscan
