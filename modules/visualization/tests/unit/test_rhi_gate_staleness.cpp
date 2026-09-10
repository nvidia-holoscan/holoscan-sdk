// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Gate generation-staleness tests: a swapchain recreation (resize) supersedes every
// outstanding AcquireSyncToken and its gates. Stale gates must report kInvalidState from
// every method instead of dereferencing recycled storage; a stale token given to present()
// must report kFrameDropped before touching its gates. Also covers the gate wait()
// contract (consume-on-successful-submit) and destroy() draining an unpresented acquire.

#include <chrono>
#include <utility>

#include <holoviz/rhi.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz::rhi {
namespace {

TEST(RhiGateStaleness, DefaultConstructedGatesKeepTheEmptyHandleError) {
  AcquireSignalGate acquire_gate;
  const auto consumed = acquire_gate.consumed();
  ASSERT_FALSE(consumed.has_value());
  EXPECT_EQ(consumed.error().code, ErrorCode::kInvalidState);

  PresentWaitGate present_gate;
  const auto signaled = present_gate.consumed();
  ASSERT_FALSE(signaled.has_value());
  EXPECT_EQ(signaled.error().code, ErrorCode::kInvalidState);
}

class RhiGateStalenessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto window = create_window({.title = "rhi gate staleness test", .size = {320, 240}});
    if (!window) {
      HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
    }
    window_ = std::move(*window);

    auto surface = window_.create_surface();
    ASSERT_TRUE(surface.has_value()) << surface.error().message;
    surface_ = std::move(*surface);

    auto device = create_device_for_surface({}, surface_);
    if (!device) {
      HOLOVIZ_ENV_SKIP() << "no presentable adapter: " << device.error().message;
    }
    device_ = std::move(*device);

    auto swapchain = device_.create_swapchain(
        {.surface = surface_, .present_mode = PresentMode::kFIFO, .image_count = 3});
    ASSERT_TRUE(swapchain.has_value()) << swapchain.error().message;
    swapchain_ = std::move(*swapchain);
  }

  void TearDown() override {
    if (device_) {
      EXPECT_TRUE(device_.wait_idle().has_value());
    }
    if (swapchain_) {
      EXPECT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
    }
  }

  // Acquire a token and force a recreation underneath it via an explicit resize.
  expected<AcquireSyncToken> acquire_then_recreate() {
    auto token = swapchain_.acquire(std::chrono::seconds(5));
    if (!token) {
      return token;
    }
    const uint64_t generation_before = swapchain_.generation().value();
    EXPECT_TRUE(swapchain_.resize({256, 200}, std::chrono::seconds(5)).has_value());
    EXPECT_NE(swapchain_.generation().value(), generation_before);
    return token;
  }

  Window window_;
  Surface surface_;
  Device device_;
  Swapchain swapchain_;
};

TEST_F(RhiGateStalenessTest, StaleGateMethodsReturnInvalidState) {
  auto token = acquire_then_recreate();
  ASSERT_TRUE(token.has_value()) << token.error().message;

  // Every accessor on a superseded gate must fail BEFORE touching the recycled impl.
  const auto consumed = token->acquire_gate.consumed();
  ASSERT_FALSE(consumed.has_value());
  EXPECT_EQ(consumed.error().code, ErrorCode::kInvalidState);

  const auto stage = token->acquire_gate.wait_stage();
  ASSERT_FALSE(stage.has_value());
  EXPECT_EQ(stage.error().code, ErrorCode::kInvalidState);

  const auto native = token->acquire_gate.native_object(NativeObjectType::kVkSemaphore);
  ASSERT_FALSE(native.has_value());
  EXPECT_EQ(native.error().code, ErrorCode::kInvalidState);

  const auto waited = token->acquire_gate.wait(std::chrono::seconds(1));
  ASSERT_FALSE(waited.has_value());
  EXPECT_EQ(waited.error().code, ErrorCode::kInvalidState);

  const auto present_consumed = token->present_gate.consumed();
  ASSERT_FALSE(present_consumed.has_value());
  EXPECT_EQ(present_consumed.error().code, ErrorCode::kInvalidState);

  const auto signaled = token->present_gate.signal();
  ASSERT_FALSE(signaled.has_value());
  EXPECT_EQ(signaled.error().code, ErrorCode::kInvalidState);
}

TEST_F(RhiGateStalenessTest, PresentOfStaleTokenIsFrameDropped) {
  auto token = acquire_then_recreate();
  ASSERT_TRUE(token.has_value()) << token.error().message;

  const auto presented = swapchain_.present(std::move(*token), SubmitSyncToken{});
  ASSERT_FALSE(presented.has_value());
  EXPECT_EQ(presented.error().code, ErrorCode::kFrameDropped);

  // The swapchain stays fully usable: a fresh acquire/present cycle succeeds.
  auto fresh = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(fresh.has_value()) << fresh.error().message;
  EXPECT_EQ(fresh->generation, swapchain_.generation().value());
  EXPECT_TRUE(swapchain_.present(std::move(*fresh), SubmitSyncToken{}).has_value());
}

TEST_F(RhiGateStalenessTest, StaleGateInSubmitDescIsRejectedWithoutStateChange) {
  auto token = acquire_then_recreate();
  ASSERT_TRUE(token.has_value()) << token.error().message;

  SubmitDesc batch;
  batch.acquire_gates = {&token->acquire_gate, 1};
  batch.present_gates = {&token->present_gate, 1};
  const auto submitted = device_.submit(batch);
  ASSERT_FALSE(submitted.has_value());
  EXPECT_EQ(submitted.error().code, ErrorCode::kInvalidState);

  // Phase-1 rejection mutates nothing: the swapchain still cycles cleanly.
  auto fresh = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(fresh.has_value()) << fresh.error().message;
  EXPECT_TRUE(swapchain_.present(std::move(*fresh), SubmitSyncToken{}).has_value());
}

// FF3: wait() consumes the gate whenever its drain submit was enqueued; a second wait on
// the same (binary) gate must fail instead of enqueueing an invalid second semaphore wait.
TEST_F(RhiGateStalenessTest, GateWaitConsumesAndRejectsASecondWait) {
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  ASSERT_TRUE(token->acquire_gate.wait(std::chrono::seconds(5)).has_value());
  EXPECT_TRUE(token->acquire_gate.consumed().value());

  const auto second_wait = token->acquire_gate.wait(std::chrono::seconds(1));
  ASSERT_FALSE(second_wait.has_value());

  // Balance the gates (present() requires both consumed or both unconsumed) and hand the
  // image back.
  ASSERT_TRUE(token->present_gate.signal().has_value());
  EXPECT_TRUE(swapchain_.present(std::move(*token), SubmitSyncToken{}).has_value());
}

// FF2: destroy() must drain an acquired-but-never-presented image (the pending binary
// signal on its acquire semaphore) instead of destroying the semaphore under it. Run with
// HOLOVIZ_RHI_VALIDATION=1 (the ctest default) this is also a validation-cleanliness test.
TEST_F(RhiGateStalenessTest, AbandonedAcquireThenDestroySucceeds) {
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  // Drop the token without presenting; destroy() must still succeed.
  {
    auto dropped = std::move(*token);
  }
  EXPECT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
}

}  // namespace
}  // namespace holoviz::rhi
