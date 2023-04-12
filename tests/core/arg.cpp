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

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <regex>
#include <string>
#include <typeinfo>

#include "holoscan/core/parameter.hpp"

using namespace std::string_literals;

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
        {"D", int8_t{1}, {{"\"\\x01\""}}},  // hex
        {"E", int16_t{1}, {{"1"}}},
        {"F", int32_t{1}, {{"1"}}},
        {"G", int64_t{1}, {{"1"}}},
        {"H", uint8_t{1}, {{"\"\\x01\""}}},  // hex
        {"I", uint16_t{1}, {{"1"}}},
        {"J", uint32_t{1}, {{"1"}}},
        {"K", uint64_t{1}, {{"1"}}},
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

TEST(Arg, TestOtherEnums) {
  // TODO: add parametrized test cases above for these.

  // For now, just check that enum entries exist for:
  //     kIOSpec (holoscan::IOSpec*)
  //     kCondition (std::shared_ptr<Condition>)
  //     kResource (std::shared_ptr<Resource>)
  ArgElementType T = ArgElementType::kIOSpec;
  T = ArgElementType::kCondition;
  T = ArgElementType::kResource;
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

}  // namespace holoscan
