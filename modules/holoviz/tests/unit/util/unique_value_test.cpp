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

#include <util/unique_value.hpp>

using namespace holoscan::viz;

namespace {

bool g_has_been_destroyed;
using ValueType = uint32_t;
ValueType test_value = 0xDEADBEEF;

void destroy(ValueType value) {
  EXPECT_EQ(test_value, value);
  g_has_been_destroyed = true;
}

}  // anonymous namespace

TEST(UniqueValue, All) {
  g_has_been_destroyed = false;

  {
    UniqueValue<ValueType, decltype(&destroy), &destroy> unique_value;

    // default state
    EXPECT_EQ(ValueType(), unique_value.get());
    EXPECT_FALSE(static_cast<bool>(unique_value));
    EXPECT_EQ(ValueType(), unique_value.release());
  }

  EXPECT_FALSE(g_has_been_destroyed) << "destroy should not be called on default (zero) value";

  // reset and release
  {
    UniqueValue<ValueType, decltype(&destroy), &destroy> unique_value;

    // set to guard to the test value
    EXPECT_NO_THROW(unique_value.reset(test_value));

    EXPECT_EQ(test_value, unique_value.get()) << "did not return the test value";
    EXPECT_TRUE(static_cast<bool>(unique_value))
        << "expect boolean operator to return true if value is set";

    // release the test value
    EXPECT_EQ(test_value, unique_value.release()) << "failed to release";
    EXPECT_FALSE(static_cast<bool>(unique_value))
        << "expect boolean operator to return false if value is not set";
    EXPECT_EQ(ValueType(), unique_value.get())
        << "did not return default value if value is not set";
  }

  EXPECT_FALSE(g_has_been_destroyed) << "destroy should not be called after release";

  // set with constructor
  {
    UniqueValue<ValueType, decltype(&destroy), &destroy> unique_value(test_value);

    EXPECT_EQ(test_value, unique_value.get()) << "did not return the test value";

    // release the test value
    EXPECT_EQ(test_value, unique_value.release()) << "failed to release";
  }

  EXPECT_FALSE(g_has_been_destroyed) << "action should not be called after release";

  // call action of test value
  {
    UniqueValue<ValueType, decltype(&destroy), &destroy> unique_value;

    // set to guard to the test value
    EXPECT_NO_THROW(unique_value.reset(test_value));
  }

  EXPECT_TRUE(g_has_been_destroyed) << "action should be set after UniqueValue had been reset";
  g_has_been_destroyed = false;

  // call action of test value
  {
    // set with constructor
    UniqueValue<ValueType, decltype(&destroy), &destroy> unique_value(test_value);
  }

  EXPECT_TRUE(g_has_been_destroyed) << "action should be set after UniqueValue had been destroyed";
  g_has_been_destroyed = false;

  // move constructor
  {
    // construct
    UniqueValue<ValueType, decltype(&destroy), &destroy> unique_value(test_value);
    // move construct
    UniqueValue<ValueType, decltype(&destroy), &destroy> other_unique_value =
        std::move(unique_value);
    EXPECT_FALSE(g_has_been_destroyed)
        << "action should not be set after UniqueValue had been move constructed";

    EXPECT_EQ(ValueType(), unique_value.get()) << "old unique_value is released";
    EXPECT_EQ(test_value, other_unique_value.get())
        << "new unique_value has the test_value after move construction";
  }

  EXPECT_TRUE(g_has_been_destroyed) << "action should be set after UniqueValue had been reset";
  g_has_been_destroyed = false;

  // move assignment
  {
    // construct
    UniqueValue<ValueType, decltype(&destroy), &destroy> unique_value(test_value);
    // move construct
    UniqueValue<ValueType, decltype(&destroy), &destroy> other_unique_value;
    EXPECT_EQ(ValueType(), other_unique_value.get()) << "other_unique_value is empty";
    other_unique_value = std::move(unique_value);
    EXPECT_FALSE(g_has_been_destroyed)
        << "action should not be set after UniqueValue had been move assigned";

    EXPECT_EQ(ValueType(), unique_value.get()) << "old unique_value is released";
    EXPECT_EQ(test_value, other_unique_value.get())
        << "new unique_value has the test_value after move assignment";
  }

  EXPECT_TRUE(g_has_been_destroyed) << "action should be set after UniqueValue had been reset";
  g_has_been_destroyed = false;
}
