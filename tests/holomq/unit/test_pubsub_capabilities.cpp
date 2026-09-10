/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/context_builder.hpp>
#include <holomq/pubsub_capabilities.hpp>

#include <gtest/gtest.h>

namespace {

void expect_common_typed_payload_capabilities(const holomq::PubSubCapabilities& capabilities) {
  EXPECT_TRUE(capabilities.typed_payloads);
  EXPECT_TRUE(capabilities.fanout);
  EXPECT_TRUE(capabilities.non_consuming_ready);
  EXPECT_TRUE(capabilities.blocking_queue_policy);
  EXPECT_TRUE(capabilities.inline_payload_transfer);
  EXPECT_TRUE(capabilities.local_descriptor_payload_transfer);
  EXPECT_TRUE(capabilities.buffer_handle_payload_transfer);
  EXPECT_FALSE(capabilities.local_descriptor_resolution);
  EXPECT_TRUE(capabilities.buffer_handle_resolution);
}

void expect_same_capabilities(const holomq::PubSubCapabilities& actual,
                              const holomq::PubSubCapabilities& expected) {
  EXPECT_EQ(actual.typed_payloads, expected.typed_payloads);
  EXPECT_EQ(actual.fanout, expected.fanout);
  EXPECT_EQ(actual.non_consuming_ready, expected.non_consuming_ready);
  EXPECT_EQ(actual.waitset, expected.waitset);
  EXPECT_EQ(actual.kernel_wait, expected.kernel_wait);
  EXPECT_EQ(actual.blocking_queue_policy, expected.blocking_queue_policy);
  EXPECT_EQ(actual.drop_oldest_queue_policy, expected.drop_oldest_queue_policy);
  EXPECT_EQ(actual.inline_payload_transfer, expected.inline_payload_transfer);
  EXPECT_EQ(actual.local_descriptor_payload_transfer, expected.local_descriptor_payload_transfer);
  EXPECT_EQ(actual.buffer_handle_payload_transfer, expected.buffer_handle_payload_transfer);
  EXPECT_EQ(actual.local_descriptor_resolution, expected.local_descriptor_resolution);
  EXPECT_EQ(actual.buffer_handle_resolution, expected.buffer_handle_resolution);
  EXPECT_EQ(actual.delivery_attribution, expected.delivery_attribution);
}

}  // namespace

TEST(PubSubCapabilitiesTest, DefaultCapabilitiesExposeTypedControlPlaneSupport) {
  const holomq::PubSubCapabilities capabilities = holomq::default_pubsub_capabilities();

  expect_common_typed_payload_capabilities(capabilities);
  EXPECT_EQ(capabilities.delivery_attribution, holomq::DeliveryAttribution::kPerRecipient);
}

TEST(PubSubCapabilitiesTest, ContextReportsDefaultCapabilities) {
  auto context = holomq::ContextBuilder().create();
  ASSERT_TRUE(context.has_value());

  expect_same_capabilities(context->pubsub_capabilities(), holomq::default_pubsub_capabilities());
}

TEST(PubSubCapabilitiesTest, QueuePolicyPredicatesMatchCapabilityBits) {
  holomq::PubSubCapabilities capabilities{};
  capabilities.blocking_queue_policy = true;
  capabilities.drop_oldest_queue_policy = false;

  EXPECT_TRUE(holomq::supports_queue_policy(capabilities, holomq::QueuePolicy::kBlocking));
  EXPECT_FALSE(holomq::supports_queue_policy(capabilities, holomq::QueuePolicy::kDropOldest));

  capabilities.blocking_queue_policy = false;
  capabilities.drop_oldest_queue_policy = true;

  EXPECT_FALSE(holomq::supports_queue_policy(capabilities, holomq::QueuePolicy::kBlocking));
  EXPECT_TRUE(holomq::supports_queue_policy(capabilities, holomq::QueuePolicy::kDropOldest));
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_FALSE(holomq::supports_queue_policy(capabilities, static_cast<holomq::QueuePolicy>(2U)));
}

TEST(PubSubCapabilitiesTest, BufferTransferPredicatesSeparateTransportFromResolution) {
  const holomq::PubSubCapabilities capabilities = holomq::default_pubsub_capabilities();

  EXPECT_TRUE(holomq::supports_buffer_transfer_mode(capabilities,
                                                    holomq::BufferTransferMode::kInlinePayload));
  EXPECT_TRUE(holomq::supports_buffer_transfer_mode(capabilities,
                                                    holomq::BufferTransferMode::kLocalDescriptor));
  EXPECT_TRUE(holomq::supports_buffer_transfer_mode(capabilities,
                                                    holomq::BufferTransferMode::kBufferHandle));

  EXPECT_TRUE(holomq::resolves_buffer_transfer_mode(capabilities,
                                                    holomq::BufferTransferMode::kInlinePayload));
  EXPECT_FALSE(holomq::resolves_buffer_transfer_mode(capabilities,
                                                     holomq::BufferTransferMode::kLocalDescriptor));
  EXPECT_TRUE(holomq::resolves_buffer_transfer_mode(capabilities,
                                                    holomq::BufferTransferMode::kBufferHandle));
}

#if defined(HOLOMQ_TRANSPORT_IOX2)
TEST(PubSubCapabilitiesTest, Iox2BackendReportsDropOldestAndWaitsetSupport) {
  const holomq::PubSubCapabilities capabilities = holomq::default_pubsub_capabilities();

  EXPECT_TRUE(capabilities.drop_oldest_queue_policy);
  EXPECT_TRUE(capabilities.waitset);
  EXPECT_TRUE(capabilities.kernel_wait);
  EXPECT_EQ(capabilities.delivery_attribution, holomq::DeliveryAttribution::kPerRecipient);
}
#endif
