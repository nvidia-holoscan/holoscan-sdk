/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(HOLOMQ_TRANSPORT_IOX2)

// The test target normally advertises Linux support. Override only this translation unit before
// the capability header is parsed so the real template takes its unsupported-platform branch.
#ifdef HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS
#undef HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS
#endif
#define HOLOMQ_HAS_NATIVE_DESCRIPTOR_READINESS 0

#include <holomq/context_builder.hpp>
#include <holomq/subscriber_builder.hpp>

#include <gtest/gtest.h>

#include <cstdint>

namespace {

struct UnsupportedReadinessPayload {
  std::uint64_t value{0U};
};

// REQ: CHAN-149
TEST(ReadinessPortabilityTest, UnsupportedPlatformDoesNotClaimWaitOwner) {
  auto context = holomq::ContextBuilder().create();
  ASSERT_TRUE(context.has_value());
  auto subscriber = context->subscriber_builder("holomq_readiness_unsupported_platform")
                        .enable_event_wake(true)
                        .payload_type<UnsupportedReadinessPayload>()
                        .create();
  ASSERT_TRUE(subscriber.has_value());

  auto acquired = subscriber->readiness();
  ASSERT_FALSE(acquired);
  EXPECT_EQ(acquired.error().code, holomq::ErrorCode::kNotSupported);
  EXPECT_FALSE(subscriber->has_wait_owner());
  EXPECT_FALSE(subscriber->has_readiness_owner());
}

}  // namespace

#endif  // HOLOMQ_TRANSPORT_IOX2
