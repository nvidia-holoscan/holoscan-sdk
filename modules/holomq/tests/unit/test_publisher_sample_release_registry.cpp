/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <holomq/detail/publisher_sample_release_registry.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace {

struct Payload {
  std::uint64_t field{0};
};

}  // namespace

TEST(PublisherSampleReleaseRegistryTest, DispatchInvokesAndClearsPending) {
  holomq::detail::PublisherSampleReleaseRegistry registry;
  Payload payload{};

  int invoke_count = 0;
  ASSERT_TRUE(registry.register_pending(&payload, [&]() { ++invoke_count; }));
  EXPECT_EQ(registry.pending_count(), 1U);

  registry.dispatch(&payload);
  EXPECT_EQ(invoke_count, 1);
  EXPECT_EQ(registry.pending_count(), 0U);

  registry.dispatch(&payload);
  EXPECT_EQ(invoke_count, 1);
  EXPECT_EQ(registry.pending_count(), 0U);
}

TEST(PublisherSampleReleaseRegistryTest, RetainsCapturedLeaseUntilDispatch) {
  holomq::detail::PublisherSampleReleaseRegistry registry;
  Payload payload{};

  auto lease = std::make_shared<Payload>();
  const std::weak_ptr<Payload> weak_lease = lease;
  ASSERT_TRUE(registry.register_pending(
      &payload, [captured_lease = std::move(lease)]() { static_cast<void>(captured_lease); }));
  EXPECT_EQ(registry.pending_count(), 1U);
  EXPECT_FALSE(weak_lease.expired());

  registry.dispatch(&payload);
  EXPECT_EQ(registry.pending_count(), 0U);
  EXPECT_TRUE(weak_lease.expired());
}

TEST(PublisherSampleReleaseRegistryTest, DuplicateRegisterRejected) {
  holomq::detail::PublisherSampleReleaseRegistry registry;
  Payload payload{};

  ASSERT_TRUE(registry.register_pending(&payload, []() {}));
  EXPECT_FALSE(registry.register_pending(&payload, []() {}));
  EXPECT_EQ(registry.pending_count(), 1U);
}

TEST(PublisherSampleReleaseRegistryTest, RegisterLvalueCallableSucceeds) {
  holomq::detail::PublisherSampleReleaseRegistry registry;
  Payload payload{};

  int invoke_count = 0;
  auto callback = [&]() { ++invoke_count; };
  ASSERT_TRUE(registry.register_pending(&payload, callback));

  registry.dispatch(&payload);
  EXPECT_EQ(invoke_count, 1);
}

TEST(PublisherSampleReleaseRegistryTest, RegisterAfterDispatchSucceeds) {
  holomq::detail::PublisherSampleReleaseRegistry registry;
  Payload payload{};

  ASSERT_TRUE(registry.register_pending(&payload, []() {}));
  EXPECT_EQ(registry.pending_count(), 1U);
  registry.dispatch(&payload);
  EXPECT_EQ(registry.pending_count(), 0U);
  EXPECT_TRUE(registry.register_pending(&payload, []() {}));
  EXPECT_EQ(registry.pending_count(), 1U);
}

TEST(PublisherSampleReleaseRegistryTest, DispatchWithoutPendingCallbackKeepsFastMissEmpty) {
  holomq::detail::PublisherSampleReleaseRegistry registry;
  Payload payload{};

  registry.dispatch(&payload);
  EXPECT_EQ(registry.pending_count(), 0U);

  ASSERT_TRUE(registry.register_pending(&payload, []() {}));
  EXPECT_EQ(registry.pending_count(), 1U);
}
