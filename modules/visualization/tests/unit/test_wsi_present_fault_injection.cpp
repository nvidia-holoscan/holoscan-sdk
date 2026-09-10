// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Deterministic failure-path tests for the present/acquire ownership protocol, driven
// through the sanctioned swapchain fault-injection seam (vk_swapchain_fault_injection.hpp):
// pre-queue present failures, post-queue present failures, out-of-date recovery, and
// failed recreations — none of which a real driver produces on cue. Windowed; skipped
// where the environment has no display.

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <holoviz/backends/wsi_present_backend.hpp>
#include <holoviz/present_backend.hpp>
#include <holoviz/rhi.hpp>

// Module-private headers (this fault-injection target builds with src/ on the include
// path, like the imgui fault-injection target).
#include "backends/backend_tokens.hpp"
#include "backends/wsi_present_backend.hpp"
#include "rhi/vulkan/vk_swapchain_fault_injection.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz::backends {
namespace {

class WsiPresentFaultInjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto window = rhi::create_window(
        {.title = "wsi present fault injection test", .size = {320, 240}, .resizable = true});
    if (!window) {
      HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
    }
    window_ = std::move(*window);

    auto surface_handle = make_surface_handle(window_);
    ASSERT_TRUE(surface_handle.has_value()) << surface_handle.error().message;
    surface_handle_ = std::move(*surface_handle);

    auto device = rhi::create_device_for_surface({}, *surface_from_handle(surface_handle_));
    if (!device) {
      HOLOVIZ_ENV_SKIP() << "no presentable adapter: " << device.error().message;
    }
    device_ = std::move(*device);

    auto backend = create_wsi_present_backend(device_);
    ASSERT_TRUE(backend.has_value()) << backend.error().message;
    backend_ = std::move(*backend);

    ASSERT_TRUE(
        backend_
            ->create_swapchain(SwapchainDesc{
                .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
            .has_value());
  }

  void TearDown() override {
    if (backend_ != nullptr) {
      EXPECT_TRUE(backend_->destroy(std::chrono::seconds(5)).has_value());
      backend_.reset();
    }
    surface_handle_.impl.reset();
  }

  rhi::vulkan::SwapchainFaultInjection& fault_injection() {
    return rhi::vulkan::swapchain_fault_injection(
        static_cast<WsiPresentBackend&>(*backend_).swapchain_for_test());
  }

  rhi::Window window_;
  SurfaceHandle surface_handle_;
  rhi::Device device_;
  std::unique_ptr<PresentBackend> backend_;
};

// The frozen-frame-loop regression: a present failure BEFORE the queue submit (here a
// fence-allocation failure) must consume neither the frame nor its id — retrying the very
// same present succeeds and the loop continues.
TEST_F(WsiPresentFaultInjectionTest, PreQueuePresentFailureRetainsFrameAndId) {
  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  fault_injection().fail_next_take_present_fence = 1;
  const auto failed = backend_->present(PresentArgs{
      .image_index = token->image_index, .acquire_issuance = token->issuance, .present_id = 1});
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().code, ErrorCode::kVulkanError);

  // Nothing was consumed: the same present — same image, same id — succeeds on retry.
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = token->image_index,
                                        .acquire_issuance = token->issuance,
                                        .present_id = 1})
                  .has_value());

  // And the loop continues normally with the next id.
  auto next = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(next.has_value()) << next.error().message;
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = next->image_index,
                                        .acquire_issuance = next->issuance,
                                        .present_id = 2})
                  .has_value());
}

// The id-desync regression: a present failure AFTER the queue submit consumed the frame
// and burned its id at the RHI. The backend must mirror both — and report the consumption
// honestly: the public code is kFrameDropped (the contract's consumed signal, with the
// native cause in the message), never a raw retryable-looking error that would invite
// re-presenting a frame that no longer exists.
TEST_F(WsiPresentFaultInjectionTest, PostQueuePresentFailureConsumesFrameAndRecordsDrop) {
  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  fault_injection().override_next_present_result = rhi::vulkan::kVkErrorUnknown;
  const auto failed = backend_->present(PresentArgs{
      .image_index = token->image_index, .acquire_issuance = token->issuance, .present_id = 1});
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().code, ErrorCode::kFrameDropped) << failed.error().message;

  // The frame was consumed: its table entry is gone.
  const auto represent = backend_->present(PresentArgs{
      .image_index = token->image_index, .acquire_issuance = token->issuance, .present_id = 2});
  ASSERT_FALSE(represent.has_value());
  EXPECT_EQ(represent.error().code, ErrorCode::kInvalidState);

  // The id was burned and recorded as a known drop, not left unrecorded (kNotFound).
  const auto waited = backend_->wait_for_present(1, std::chrono::seconds(1));
  ASSERT_TRUE(waited.has_value()) << waited.error().message;
  EXPECT_EQ(*waited, WaitResult::kDropped);

  // Both watermarks stay compatible: the next id is accepted by backend AND RHI.
  auto next = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(next.has_value()) << next.error().message;
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = next->image_index,
                                        .acquire_issuance = next->issuance,
                                        .present_id = 2})
                  .has_value());
}

// An out-of-date present triggers the RHI's automatic recovery; the backend must publish
// the new generation to token consumers on that error exit and record the drop.
TEST_F(WsiPresentFaultInjectionTest, PresentOutOfDateRecoveryPublishesGenerationAndDropsFrame) {
  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  const auto payload = std::static_pointer_cast<const AcquirePayload>(token->image_ready.impl);
  const auto source = payload->source.lock();
  ASSERT_NE(source, nullptr);
  const uint64_t generation_before = payload->generation;

  fault_injection().override_next_present_result = rhi::vulkan::kVkErrorOutOfDate;
  const auto dropped = backend_->present(PresentArgs{
      .image_index = token->image_index, .acquire_issuance = token->issuance, .present_id = 1});
  ASSERT_FALSE(dropped.has_value());
  EXPECT_TRUE(dropped.error().code == ErrorCode::kFrameDropped ||
              dropped.error().code == ErrorCode::kImageCountChanged)
      << dropped.error().message;

  // The recovery recreated the swapchain; the liveness cell must say so before any
  // consumer dereferences a stale payload's image.
  EXPECT_GT(source->current_generation, generation_before);

  const auto waited = backend_->wait_for_present(1, std::chrono::seconds(1));
  ASSERT_TRUE(waited.has_value()) << waited.error().message;
  EXPECT_EQ(*waited, WaitResult::kDropped);

  // The loop continues (a kImageCountChanged outcome may need one reconciling acquire).
  for (int i = 0; i < 3; ++i) {
    auto next = backend_->acquire(std::chrono::seconds(5));
    if (!next.has_value()) {
      continue;
    }
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = next->image_index,
                                          .acquire_issuance = next->issuance,
                                          .present_id = 2})
                    .has_value());
    return;
  }
  FAIL() << "acquire never recovered after the injected out-of-date present";
}

// The image-count-change twin of the drop-recording promise: kImageCountChanged's
// contract says the consumed frame's id still answers kDropped, so the reconciliation's
// tracker reset (sized for the new image count) must never evict the outcome recorded for
// the frame that triggered it. The bump knob makes the count change deterministic.
TEST_F(WsiPresentFaultInjectionTest, ImageCountChangingDropStillAnswersWaitForPresentDropped) {
  const auto info_before = backend_->swapchain_info();
  ASSERT_TRUE(info_before.has_value());

  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  fault_injection().override_next_present_result = rhi::vulkan::kVkErrorOutOfDate;
  fault_injection().bump_next_recreate_image_count = 1;
  const auto dropped = backend_->present(PresentArgs{
      .image_index = token->image_index, .acquire_issuance = token->issuance, .present_id = 1});
  ASSERT_FALSE(dropped.has_value());

  const auto info_after = backend_->swapchain_info();
  ASSERT_TRUE(info_after.has_value());
  if (info_after->image_count == info_before->image_count) {
    // The driver ignored the raised minImageCount (or its max clamped it back): the
    // count-changing path under test is unreachable on this surface.
    EXPECT_EQ(dropped.error().code, ErrorCode::kFrameDropped) << dropped.error().message;
    GTEST_SKIP() << "the recreation kept the image count on this driver";
  }
  EXPECT_EQ(dropped.error().code, ErrorCode::kImageCountChanged) << dropped.error().message;

  // The promise under regression: the consumed frame's outcome survives the tracker's
  // image-count resize — exactly kDropped, never kNotFound/kEvicted.
  const auto waited = backend_->wait_for_present(1, std::chrono::seconds(1));
  ASSERT_TRUE(waited.has_value()) << waited.error().message;
  EXPECT_EQ(*waited, WaitResult::kDropped);

  // The loop continues with the next id after the per-image rebuild signal.
  for (int i = 0; i < 3; ++i) {
    auto next = backend_->acquire(std::chrono::seconds(5));
    if (!next.has_value()) {
      continue;
    }
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = next->image_index,
                                          .acquire_issuance = next->issuance,
                                          .present_id = 2})
                    .has_value());
    return;
  }
  FAIL() << "acquire never recovered after the image-count-changing recreation";
}

// The held-acquire twin: a changed-count recreation resizes the frame table, dropping the
// held frame's slot wholesale. Its delayed present must still answer the consumed-frame
// contract — kFrameDropped exactly once, id burned and recorded as a known drop — never
// the retryable "not acquired" of the resized table, which would invite retrying (or
// empty-presenting) a frame that no longer exists.
TEST_F(WsiPresentFaultInjectionTest, RetiredAcquireAfterImageCountChangePresentsAsDropped) {
  const auto info_before = backend_->swapchain_info();
  ASSERT_TRUE(info_before.has_value());

  auto held = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(held.has_value()) << held.error().message;

  fault_injection().bump_next_recreate_image_count = 1;
  ASSERT_TRUE(backend_->resize(Extent2D{0, 0}).has_value());

  const auto info_after = backend_->swapchain_info();
  ASSERT_TRUE(info_after.has_value());
  if (info_after->image_count == info_before->image_count) {
    GTEST_SKIP() << "the recreation kept the image count on this driver; the "
                    "table-resizing retirement path under test is unreachable here";
  }

  const auto dropped = backend_->present(PresentArgs{
      .image_index = held->image_index, .acquire_issuance = held->issuance, .present_id = 1});
  ASSERT_FALSE(dropped.has_value());
  EXPECT_EQ(dropped.error().code, ErrorCode::kFrameDropped) << dropped.error().message;

  // The id was burned and recorded, exactly as for any other consumed frame.
  const auto waited = backend_->wait_for_present(1, std::chrono::seconds(1));
  ASSERT_TRUE(waited.has_value()) << waited.error().message;
  EXPECT_EQ(*waited, WaitResult::kDropped);

  // The drop is single-shot: re-presenting the dead frame is the ordinary protocol
  // violation again, not a second burned id.
  const auto represent = backend_->present(PresentArgs{
      .image_index = held->image_index, .acquire_issuance = held->issuance, .present_id = 2});
  ASSERT_FALSE(represent.has_value());
  EXPECT_NE(represent.error().code, ErrorCode::kFrameDropped) << represent.error().message;

  // Fresh frames are untouched by the retirement bookkeeping.
  auto next = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(next.has_value()) << next.error().message;
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = next->image_index,
                                        .acquire_issuance = next->issuance,
                                        .present_id = 2})
                  .has_value());
}

// The shrink variant of the retirement promise: grow the swapchain (bump knob), hold an
// acquire on the grown range's top index, then shrink back — the held frame's image index
// is now OUT OF RANGE for the resized table, the harshest misclassification (it would
// read as the caller's kInvalidArgument bug). It must drop like any other retired frame.
TEST_F(WsiPresentFaultInjectionTest, RetiredAcquireBeyondShrunkTableStillDrops) {
  const auto info_before = backend_->swapchain_info();
  ASSERT_TRUE(info_before.has_value());

  fault_injection().bump_next_recreate_image_count = 1;
  ASSERT_TRUE(backend_->resize(Extent2D{0, 0}).has_value());
  const auto info_grown = backend_->swapchain_info();
  ASSERT_TRUE(info_grown.has_value());
  if (info_grown->image_count <= info_before->image_count) {
    GTEST_SKIP() << "the recreation did not grow the image count on this driver; no index "
                    "can fall out of range on the shrink back";
  }

  // Hunt for an acquire whose index the shrink below will remove (presenting the others).
  std::optional<AcquireToken> held;
  PresentId next_id = 1;
  for (uint32_t i = 0; i < 3 * info_grown->image_count && !held.has_value(); ++i) {
    auto token = backend_->acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << "cycle " << i << ": " << token.error().message;
    if (token->image_index >= info_before->image_count) {
      held = std::move(*token);
      break;
    }
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = token->image_index,
                                          .acquire_issuance = token->issuance,
                                          .present_id = next_id++})
                    .has_value())
        << "cycle " << i;
  }
  if (!held.has_value()) {
    GTEST_SKIP() << "the driver never handed out an image index beyond the pre-grow count";
  }

  ASSERT_TRUE(backend_->resize(Extent2D{0, 0}).has_value());  // bump off: shrink back
  const auto info_shrunk = backend_->swapchain_info();
  ASSERT_TRUE(info_shrunk.has_value());
  if (info_shrunk->image_count > held->image_index) {
    GTEST_SKIP() << "the recreation kept the held index in range on this driver; the "
                    "out-of-range retirement path under test is unreachable here";
  }

  const PresentId dropped_id = next_id++;
  const auto dropped = backend_->present(PresentArgs{.image_index = held->image_index,
                                                     .acquire_issuance = held->issuance,
                                                     .present_id = dropped_id});
  ASSERT_FALSE(dropped.has_value());
  EXPECT_EQ(dropped.error().code, ErrorCode::kFrameDropped) << dropped.error().message;

  const auto waited = backend_->wait_for_present(dropped_id, std::chrono::seconds(1));
  ASSERT_TRUE(waited.has_value()) << waited.error().message;
  EXPECT_EQ(*waited, WaitResult::kDropped);

  // The frame loop continues on the shrunk swapchain.
  auto next = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(next.has_value()) << next.error().message;
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = next->image_index,
                                        .acquire_issuance = next->issuance,
                                        .present_id = next_id})
                  .has_value());
}

// A failed recreation (successor creation fails after the RHI retired the old swapchain
// and bumped its generation) must still publish the invalidation: a held payload's source
// advances past it, so a render backend rejects the token instead of dereferencing a
// destroyed image wrapper. The next acquire/resize self-heals.
TEST_F(WsiPresentFaultInjectionTest, FailedResizePublishesGenerationInvalidation) {
  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  const auto payload = std::static_pointer_cast<const AcquirePayload>(token->image_ready.impl);
  const auto source = payload->source.lock();
  ASSERT_NE(source, nullptr);

  fault_injection().fail_next_swapchain_create = 1;
  const auto resized = backend_->resize({0, 0});
  ASSERT_FALSE(resized.has_value());

  // The invalidation was published even though the resize failed.
  EXPECT_GT(source->current_generation, payload->generation);

  // Self-heal: acquire retries the recovery (tolerate the transitional signals).
  for (int i = 0; i < 3; ++i) {
    auto next = backend_->acquire(std::chrono::seconds(5));
    if (!next.has_value()) {
      const ErrorCode code = next.error().code;
      ASSERT_TRUE(code == ErrorCode::kImageCountChanged || code == ErrorCode::kOutOfDate ||
                  code == ErrorCode::kBackpressured)
          << next.error().message;
      continue;
    }
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = next->image_index,
                                          .acquire_issuance = next->issuance})
                    .has_value());
    return;
  }
  FAIL() << "acquire never recovered after the injected failed recreation";
}

// --- RHI-level consumption reporting, against a directly owned rhi::Swapchain. ---

class RhiSwapchainFaultInjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto window = rhi::create_window(
        {.title = "rhi present consumption test", .size = {320, 240}, .resizable = true});
    if (!window) {
      HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
    }
    window_ = std::move(*window);

    auto surface = window_.create_surface();
    ASSERT_TRUE(surface.has_value()) << surface.error().message;
    surface_ = std::move(*surface);

    auto device = rhi::create_device_for_surface({}, surface_);
    if (!device) {
      HOLOVIZ_ENV_SKIP() << "no presentable adapter: " << device.error().message;
    }
    device_ = std::move(*device);

    auto swapchain = device_.create_swapchain(
        {.surface = surface_, .present_mode = PresentMode::kFIFO, .image_count = 2});
    ASSERT_TRUE(swapchain.has_value()) << swapchain.error().message;
    swapchain_ = std::move(*swapchain);
  }

  void TearDown() override {
    if (swapchain_) {
      EXPECT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
    }
  }

  rhi::Window window_;
  rhi::Surface surface_;
  rhi::Device device_;
  rhi::Swapchain swapchain_;
};

TEST_F(RhiSwapchainFaultInjectionTest, PreQueueFailureLeavesAcquireRetryable) {
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  rhi::vulkan::swapchain_fault_injection(swapchain_).fail_next_take_present_fence = 1;
  rhi::PresentConsumed consumed{};
  const auto failed = swapchain_.present(std::move(*token), rhi::SubmitSyncToken{}, 0, &consumed);
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().code, ErrorCode::kVulkanError);
  EXPECT_FALSE(consumed.image);
  EXPECT_FALSE(consumed.id);

  // The token was only bound, not consumed: the same present succeeds on retry and then
  // reports the image consumed.
  const auto retried = swapchain_.present(std::move(*token), rhi::SubmitSyncToken{}, 0, &consumed);
  ASSERT_TRUE(retried.has_value()) << retried.error().message;
  EXPECT_TRUE(consumed.image);
}

// A non-enqueued present failure (spec: OOM results leave every referenced resource and
// synchronization primitive unaffected) must consume nothing: the acquire stays
// outstanding, the id sequence is unadvanced, and the unused fence is recycled — so the
// very same present succeeds on retry and TearDown's bounded destroy proves no
// never-signaling fence was retained.
TEST_F(RhiSwapchainFaultInjectionTest, NonEnqueuedPresentFailureLeavesEverythingRetryable) {
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  rhi::vulkan::swapchain_fault_injection(swapchain_).skip_next_present_with_result =
      rhi::vulkan::kVkErrorOutOfHostMemory;
  rhi::PresentConsumed consumed{};
  const auto failed = swapchain_.present(std::move(*token), rhi::SubmitSyncToken{}, 1, &consumed);
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().code, ErrorCode::kVulkanError);
  EXPECT_FALSE(consumed.image);
  EXPECT_FALSE(consumed.id);

  // Nothing was consumed or advanced: the same present — same token, same id — succeeds.
  const auto retried = swapchain_.present(std::move(*token), rhi::SubmitSyncToken{}, 1, &consumed);
  ASSERT_TRUE(retried.has_value()) << retried.error().message;
  EXPECT_TRUE(consumed.image);
}

// An enqueued presentation rejection outside the out-of-date family (spec: the queue
// operations still execute, the image is consumed, the fence will signal) must commit the
// full frame bookkeeping before returning the error, or the next acquire would reuse
// consumed binary-semaphore state.
TEST_F(RhiSwapchainFaultInjectionTest, EnqueuedPresentRejectionCommitsFrameState) {
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  rhi::vulkan::swapchain_fault_injection(swapchain_).override_next_present_result =
      rhi::vulkan::kVkErrorFullScreenExclusiveModeLost;
  rhi::PresentConsumed consumed{};
  const auto failed = swapchain_.present(std::move(*token), rhi::SubmitSyncToken{}, 0, &consumed);
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().code, ErrorCode::kVulkanError);
  EXPECT_TRUE(consumed.image);

  // The slot was recycled on the error path: the loop continues cleanly.
  auto next = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(next.has_value()) << next.error().message;
  ASSERT_TRUE(swapchain_.present(std::move(*next), rhi::SubmitSyncToken{}).has_value());
}

TEST_F(RhiSwapchainFaultInjectionTest, PresentReportsConsumption) {
  // Success: image consumed; id consumed exactly when the device tags presents.
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  rhi::PresentConsumed consumed{};
  ASSERT_TRUE(
      swapchain_.present(std::move(*token), rhi::SubmitSyncToken{}, 1, &consumed).has_value());
  EXPECT_TRUE(consumed.image);
  const auto caps = device_.capabilities();
  ASSERT_TRUE(caps.has_value());
  EXPECT_EQ(consumed.id, caps->present_completion);

  // A post-queue error still reports the frame consumed.
  auto second = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(second.has_value()) << second.error().message;
  rhi::vulkan::swapchain_fault_injection(swapchain_).override_next_present_result =
      rhi::vulkan::kVkErrorUnknown;
  const auto failed = swapchain_.present(std::move(*second), rhi::SubmitSyncToken{}, 0, &consumed);
  ASSERT_FALSE(failed.has_value());
  EXPECT_TRUE(consumed.image);
}

}  // namespace
}  // namespace holoviz::backends
