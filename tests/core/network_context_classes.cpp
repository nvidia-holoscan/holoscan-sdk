/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gxf/core/gxf.h>

#include <memory>
#include <string>

#include "common/assert.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/network_contexts/gxf/ucx_context.hpp"
#include "../utils.hpp"

using namespace std::string_literals;

namespace holoscan {

using NetworkContextClassesWithGXFContext = TestWithGXFContext;

TEST(NetworkContextClasses, TestUcxContext) {
  Fragment F;
  const std::string name{"ucx-context"};
  auto network_context = F.make_network_context<UcxContext>(name);
  EXPECT_EQ(network_context->name(), name);
  EXPECT_EQ(typeid(network_context), typeid(std::make_shared<UcxContext>()));
  EXPECT_EQ(std::string(network_context->gxf_typename()), "nvidia::gxf::UcxContext"s);
}

TEST_F(NetworkContextClassesWithGXFContext, TestUcxContextWithArgs) {
  const std::string name{"ucx-context"};
  ArgList arglist{
      Arg{"receiver_address", std::string("0.0.0.0")},
      Arg{"receiver_port", static_cast<uint32_t>(13337)},
  };
  auto network_context = F.make_network_context<UcxContext>(name, arglist);
  EXPECT_TRUE(network_context->description().find("name: " + name) != std::string::npos);
}

TEST(NetworkContextClasses, TestUcxContextDefaultConstructor) {
  Fragment F;
  auto network_context = F.make_network_context<UcxContext>();
  // Verify it was created successfully (no crash)
  EXPECT_NE(network_context, nullptr);
}

TEST(NetworkContextClasses, TestNetworkContextUniqueDefaultNames) {
  Fragment F;

  // Create network context without specifying name
  auto ucx_ctx = F.make_network_context<UcxContext>();

  // Verify the network context has the expected default name
  EXPECT_EQ(ucx_ctx->name(), "ucx_context");
}

}  // namespace holoscan
