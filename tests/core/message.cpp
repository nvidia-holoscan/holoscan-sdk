/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <any>
#include <cstdint>
#include <memory>
#include <string>

#include "gxf/core/expected.hpp"
#include "holoscan/core/message.hpp"

namespace holoscan {

TEST(Message, TestSetValue) {
  // default construct then call set_value
  Message msg;
  float value = 13.0;
  msg.set_value(value);

  // access via value method
  std::any v = msg.value();
  EXPECT_EQ(std::any_cast<float>(v), value);
}

TEST(Message, TestAs) {
  // value-based constructor
  auto data = std::make_shared<int32_t>(10);
  Message msg{data};

  // access via as method
  auto v = msg.as<int32_t>();
  EXPECT_EQ(typeid(v), typeid(data));
  EXPECT_EQ(*v.get(), *data.get());
}

TEST(Message, TestCopyConstruct) {
  auto data = std::make_shared<std::string>("abcd");
  Message msg{data};

  // test copy constructor
  Message msg2{msg};
  auto v = msg2.as<std::string>();
  EXPECT_EQ(typeid(v), typeid(data));
  EXPECT_EQ(*v.get(), *data.get());
}

TEST(Message, TestMoveConstruct) {
  auto data = std::make_shared<std::string>("abcd");

  // test move constructor
  Message msg{Message{data}};
  auto v = msg.as<std::string>();
  EXPECT_EQ(typeid(v), typeid(data));
  EXPECT_EQ(*v.get(), *data.get());
}

TEST(Message, TestCopyAssign) {
  auto data = 5.0;
  Message msg{data};

  Message msg2 = msg;
  EXPECT_EQ(std::any_cast<double>(msg2.value()), data);
}

TEST(Message, TestMoveAssign) {
  auto data = 5.0;
  Message msg{data};

  Message msg2 = Message{data};
  EXPECT_EQ(std::any_cast<double>(msg2.value()), data);
}

template <typename T>
void check_expected_message(nvidia::Expected<Message, gxf_result_t> maybe_message, T value) {
  // validate that maybe message contains the expected value
  ASSERT_TRUE(maybe_message.has_value());
  auto msg = maybe_message.value();
  EXPECT_EQ(typeid(msg), typeid(Message));
  EXPECT_EQ(std::any_cast<T>(msg.value()), value);
}

TEST(Message, TestExpectedCreation) {
  int value = 5;
  Message msg{value};

  // Expected<Message> can be constructed from Message l-value
  nvidia::gxf::Expected<Message> maybe1{msg};
  check_expected_message<int>(maybe1, value);

  // construction from Message r-value
  nvidia::gxf::Expected<Message> maybe2{Message(value)};
  check_expected_message<int>(maybe2, value);

  // construction via a data r-value
  nvidia::gxf::Expected<Message> maybe3{static_cast<uint8_t>(8)};
  check_expected_message<uint8_t>(maybe3, 8);

  // Expected<Message&> can be constructed
  nvidia::gxf::Expected<Message&> maybe4{msg};
  check_expected_message<int>(maybe4, value);

  // nvidia::Expected requires a second template argument for the error type
  nvidia::Expected<Message, gxf_result_t> maybe5{msg};
  check_expected_message<int>(maybe5, value);
}

TEST(Message, TestExpectedAssignment) {
  int value = 5;
  Message msg{value};

  // copy assignment
  nvidia::gxf::Expected<Message> maybe1 = msg;
  check_expected_message<int>(maybe1, value);

  // move assignment
  uint8_t u8 = 8;
  nvidia::gxf::Expected<Message> maybe2 = Message(u8);
  check_expected_message<uint8_t>(maybe2, u8);

  // can assign Unexpected if a failure occurred
  nvidia::gxf::Expected<Message> maybe3 = nvidia::gxf::Unexpected{GXF_FAILURE};
  ASSERT_FALSE(maybe3.has_value());
  EXPECT_EQ(maybe3.error(), GXF_FAILURE);
}
}  // namespace holoscan
