/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/arg.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/pubsub/in_memory/network_contexts/gxf/in_memory_pubsub_network_context.hpp>

using namespace std::string_literals;

namespace holoscan {

TEST(PubSubInMemoryNetworkContextClasses, TestInMemoryPubSubNetworkContext) {
  Fragment F;
  const std::string name{"in-memory-pubsub"};
  auto network_context = F.make_network_context<InMemoryPubSubNetworkContext>(name);
  EXPECT_EQ(network_context->name(), name);
  EXPECT_EQ(typeid(network_context), typeid(std::make_shared<InMemoryPubSubNetworkContext>()));
  EXPECT_EQ(std::string(network_context->gxf_typename()), "nvidia::gxf::InMemoryPubSubContext"s);
}

TEST(PubSubInMemoryNetworkContextClasses, TestInMemoryPubSubNetworkContextDefaultConstructor) {
  Fragment F;
  auto network_context = F.make_network_context<InMemoryPubSubNetworkContext>();
  EXPECT_NE(network_context, nullptr);
}

TEST(PubSubInMemoryNetworkContextClasses, TestInMemoryPubSubNetworkContextDefaultName) {
  Fragment F;
  auto ctx = F.make_network_context<InMemoryPubSubNetworkContext>();
  EXPECT_EQ(ctx->name(), "in_memory_pubsub_network_context");
}

TEST(PubSubInMemoryNetworkContextClasses, TestInMemoryPubSubNetworkContextWithArgs) {
  Fragment F;
  const std::string name{"in-memory-pubsub"};
  ArgList arglist{
      Arg{"node_name", std::string("test_fragment")},
      Arg{"serializer_mode", static_cast<int32_t>(0)},
      Arg{"drop_pattern", std::vector<int32_t>{0, 1}},
      Arg{"reorder_pattern", std::vector<int32_t>{}},
  };
  auto network_context = F.make_network_context<InMemoryPubSubNetworkContext>(name, arglist);
  EXPECT_TRUE(network_context->description().find("name: " + name) != std::string::npos);
}

}  // namespace holoscan
