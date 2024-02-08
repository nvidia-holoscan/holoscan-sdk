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
#include <stdlib.h>

#include <holoviz/holoviz.hpp>

namespace viz = holoscan::viz;

TEST(Instance, Create) {
  // create instance
  viz::InstanceHandle instance = nullptr;
  EXPECT_NO_THROW(instance = viz::Create());
  EXPECT_NE(nullptr, instance);
  // shutdown
  EXPECT_NO_THROW(viz::Shutdown(instance));

  // context is implicitly created when calling functions with no current context
  EXPECT_TRUE(!viz::GetCurrent());
  EXPECT_NO_THROW(viz::SetCudaStream(0));
  EXPECT_TRUE(viz::GetCurrent());
  EXPECT_NO_THROW(viz::Shutdown());
}

TEST(Instance, Current) {
  EXPECT_TRUE(!viz::GetCurrent());

  // create an instance
  viz::InstanceHandle instance = nullptr;
  EXPECT_NO_THROW(instance = viz::Create());
  // should be not current
  EXPECT_TRUE(!viz::GetCurrent());

  // make current
  EXPECT_NO_THROW(viz::SetCurrent(instance));
  // should be current now
  EXPECT_EQ(instance, viz::GetCurrent());

  // make not current
  EXPECT_NO_THROW(viz::SetCurrent(nullptr));
  // no longer current
  EXPECT_TRUE(!viz::GetCurrent());

  // make current
  EXPECT_NO_THROW(viz::SetCurrent(instance));
  // shutdown
  EXPECT_NO_THROW(viz::Shutdown(instance));
  // no longer current
  EXPECT_TRUE(!viz::GetCurrent());

  // implicit context creation/make current
  EXPECT_NO_THROW(viz::SetCudaStream(0));
  // context is created and current now
  EXPECT_TRUE(viz::GetCurrent());
  // shutdown
  EXPECT_NO_THROW(viz::Shutdown());
  // no longer current
  EXPECT_TRUE(!viz::GetCurrent());
}

TEST(Instance, Errors) {
  // should throw when calling Shutdown with no current instance
  EXPECT_TRUE(!viz::GetCurrent());
  EXPECT_THROW(viz::Shutdown(), std::runtime_error);
}
