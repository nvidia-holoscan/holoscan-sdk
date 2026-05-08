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

#include <holoscan/core/arg.hpp>
#include <holoscan/core/clock.hpp>
#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/extension_manager.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/network_contexts/gxf/pubsub_context.hpp>
#include <holoscan/core/network_contexts/gxf/ucx_context.hpp>
#include <holoscan/core/resource.hpp>
#include <holoscan/core/resources/gxf/manual_clock.hpp>
#include <holoscan/core/resources/gxf/realtime_clock.hpp>
#include "../utils.hpp"
#include "common/assert.hpp"

using namespace std::string_literals;

namespace holoscan {

using NetworkContextClassesWithGXFContext = TestWithGXFContext;

namespace {
bool is_pubsub_context_available(Fragment& fragment) {
  auto extension_manager = fragment.executor().extension_manager();
  if (extension_manager) {
    extension_manager->load_extension("libgxf_pubsub.so", true);
  }
  auto context = fragment.executor().context();
  gxf_tid_t tid;
  return GxfComponentTypeId(context, "nvidia::gxf::PubSubContext", &tid) == GXF_SUCCESS;
}
}  // namespace

//==============================================================================
// PubSubContext Tests
//==============================================================================

TEST(NetworkContextClasses, TestPubSubContext) {
  Fragment F;
  const std::string name{"pubsub-context"};
  auto network_context = F.make_network_context<PubSubContext>(name);
  EXPECT_EQ(network_context->name(), name);
  EXPECT_EQ(typeid(network_context), typeid(std::make_shared<PubSubContext>()));
  EXPECT_EQ(std::string(network_context->gxf_typename()), "nvidia::gxf::PubSubContext"s);
}

TEST(NetworkContextClasses, TestPubSubContextDefaultConstructor) {
  Fragment F;
  auto network_context = F.make_network_context<PubSubContext>();
  // Verify it was created successfully (no crash)
  EXPECT_NE(network_context, nullptr);
}

TEST(NetworkContextClasses, TestPubSubContextWithArgs) {
  Fragment F;
  const std::string name{"pubsub-context"};
  ArgList arglist{
      Arg{"node_name", std::string("test_fragment")},
  };
  auto network_context = F.make_network_context<PubSubContext>(name, arglist);
  EXPECT_TRUE(network_context->description().find("name: " + name) != std::string::npos);
}

TEST(NetworkContextClasses, TestPubSubContextDefaultName) {
  Fragment F;

  // Create network context without specifying name
  auto pubsub_ctx = F.make_network_context<PubSubContext>();

  // Verify the network context has the expected default name
  EXPECT_EQ(pubsub_ctx->name(), "pubsub_context");
}

TEST_F(NetworkContextClassesWithGXFContext, TestPubSubContextAutoCreatesClock) {
  if (!is_pubsub_context_available(F)) {
    GTEST_SKIP() << "GXF PubSubContext type not registered in this test config";
  }

  auto pubsub_ctx = F.make_network_context<PubSubContext>("pubsub_context");
  pubsub_ctx->initialize();

  // Auto-created clocks are registered as resources (not as "clock" args),
  // so we verify via resources() instead of args().
  auto& resources = pubsub_ctx->resources();
  auto resource_it = resources.find("pubsub_context__realtime_clock");
  ASSERT_NE(resource_it, resources.end());

  auto clock_resource = resource_it->second;
  ASSERT_NE(clock_resource, nullptr);
  auto realtime_clock = std::dynamic_pointer_cast<RealtimeClock>(clock_resource);
  EXPECT_NE(realtime_clock, nullptr);
}

TEST_F(NetworkContextClassesWithGXFContext, TestPubSubContextUsesProvidedClock) {
  if (!is_pubsub_context_available(F)) {
    GTEST_SKIP() << "GXF PubSubContext type not registered in this test config";
  }

  auto manual_clock = F.make_resource<ManualClock>("manual_clock");
  auto pubsub_ctx =
      F.make_network_context<PubSubContext>("pubsub_context", Arg("clock", manual_clock));
  pubsub_ctx->initialize();

  auto context_clock = pubsub_ctx->clock();
  ASSERT_NE(context_clock, nullptr);
  auto cast_clock = context_clock->cast_to<ManualClock>();
  ASSERT_NE(cast_clock, nullptr);
  EXPECT_EQ(cast_clock.get(), manual_clock.get());
}

//==============================================================================
// UcxContext Tests
//==============================================================================

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
