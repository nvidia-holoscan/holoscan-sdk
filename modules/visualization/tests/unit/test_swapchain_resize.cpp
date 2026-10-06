// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Swapchain::resize() tests. The empty-handle check is pure logic; the in-place
// recreation protocol (oldSwapchain handover, format stability, index-keyed image
// identity, retired-generation reaping across repeated resizes) runs against a real
// window + swapchain when the environment has a display (skipped otherwise).

#include <atomic>
#include <chrono>
#include <thread>
#include <utility>

#include <cuda.h>

#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz {
namespace {

// Host function parked on a CUDA stream to hold everything queued behind it until the test
// flips the flag — the deterministic "GPU work still in flight" for the drain tests below.
void block_stream_until_released(void* user_data) {
  const auto* released = static_cast<std::atomic<bool>*>(user_data);
  while (!released->load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

TEST(SwapchainResize, EmptyHandleIsInvalidState) {
  const auto resized = Swapchain{}.resize();
  ASSERT_FALSE(resized.has_value());
  EXPECT_EQ(resized.error().code, ErrorCode::kInvalidState);
}

// --- Presentable-surface protocol tests: a window where a window system is available,
// otherwise a direct-display surface on a free display (console session), skipped where
// neither exists. ---

class SwapchainResizeWindowedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto window = create_window({.title = "swapchain resize test", .size = {320, 240}});
    if (window) {
      window_ = std::move(*window);
      auto surface = window_.create_surface();
      ASSERT_TRUE(surface.has_value()) << surface.error().message;
      surface_ = std::move(*surface);
    } else {
      // No window system: latch onto the first display nobody drives (Linux DRM, from a
      // console). The direct-display surface is not resizable, but explicit resize() still
      // runs the full drain-and-recreate protocol these tests exercise.
      auto surface = create_direct_display_surface();
      if (!surface) {
        HOLOVIZ_ENV_SKIP() << "no window system (" << window.error().message
                           << ") and no free display: " << surface.error().message;
      }
      surface_ = std::move(*surface);
    }

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
    // Unblock a still-armed submit rig first: wait_idle() below would otherwise hang on the
    // host-gated stream (idempotent no-op when no rig was armed or it was already released).
    release_blocked_submit();
    if (device_) {
      EXPECT_TRUE(device_.wait_idle().has_value());
    }
    // The rig's semaphore/stream go after wait_idle(): the blocked submit references the
    // imported timeline until it retires.
    if (external_semaphore_) {
      EXPECT_TRUE(external_semaphore_.destroy(std::chrono::seconds(5)).has_value());
    }
    if (cuda_stream_ != nullptr) {
      EXPECT_EQ(cuStreamDestroy(cuda_stream_), CUDA_SUCCESS);
      cuda_stream_ = nullptr;
    }
    if (swapchain_) {
      EXPECT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
    }
  }

  static expected<Surface> create_direct_display_surface() {
    auto adapters = enumerate_adapters();
    if (!adapters) {
      return make_unexpected(adapters.error());
    }
    for (const Adapter& adapter : *adapters) {
      auto displays = enumerate_displays(adapter);
      if (!displays) {
        continue;
      }
      for (const Display& display : *displays) {
        if (display.ownership == DisplayOwnership::kCompositor) {
          continue;
        }
        auto surface = display.create_surface({});
        if (surface) {
          return surface;
        }
      }
    }
    return make_unexpected(Error{ErrorCode::kNotFound, "no display available for direct scanout"});
  }

  /// The surface's current size (window framebuffer, or the display mode's visible region).
  Extent2D surface_extent() const { return surface_.extent().value(); }

  // One full unrendered frame: acquire, then present against the acquire semaphore (both
  // gates unconsumed), the same no-submit cycle the WSI backend uses for empty frames.
  void present_one_frame() {
    auto token = swapchain_.acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << token.error().message;
    ASSERT_TRUE(swapchain_.present(std::move(*token), SubmitSyncToken{}).has_value());
  }

  // Acquire a frame and submit a gate-consuming batch that deterministically stays in
  // flight: the submit waits an external-semaphore signal queued behind a host-gated CUDA
  // stream, and only release_blocked_submit() lets it complete. The acquired token is
  // dropped without present() — the exact shutdown/frame-drop shape the drain tests need.
  void arm_blocked_unpresented_submit() {
    auto cuda_context = device_.cuda_context();
    ASSERT_TRUE(cuda_context.has_value()) << cuda_context.error().message;
    ASSERT_EQ(cuCtxPushCurrent(*cuda_context), CUDA_SUCCESS);
    const CUresult stream_created = cuStreamCreate(&cuda_stream_, CU_STREAM_NON_BLOCKING);
    CUcontext popped = nullptr;
    ASSERT_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
    ASSERT_EQ(stream_created, CUDA_SUCCESS);
    ASSERT_EQ(cuLaunchHostFunc(cuda_stream_, block_stream_until_released, &submit_released_),
              CUDA_SUCCESS);

    auto external_semaphore = device_.create_external_semaphore(cuda_stream_);
    ASSERT_TRUE(external_semaphore.has_value()) << external_semaphore.error().message;
    external_semaphore_ = std::move(*external_semaphore);
    auto external_token = external_semaphore_.signal();
    ASSERT_TRUE(external_token.has_value()) << external_token.error().message;

    auto token = swapchain_.acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << token.error().message;
    const SubmitDesc submit_desc{
        .wait_external_sync_tokens = {&*external_token, 1},
        .acquire_gates = {&token->acquire_gate, 1},
        .present_gates = {&token->present_gate, 1},
    };
    const auto submitted = device_.submit(submit_desc);
    ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
    // *token goes out of scope here without present(): the frame is dropped mid-flight.
  }

  // Shared-present variant of the rig. After the first frame's real handshake, a shared
  // swapchain's acquire() mints pre-consumed gates that must not go to submit(): the render
  // submit carries no gate at all, so nothing records it on a frame slot. The acquired token
  // is dropped without present(), exactly the drop/retire-before-present shape where the
  // swapchain must still find that submit before retiring the shared image.
  void arm_blocked_gateless_shared_submit() {
    auto cuda_context = device_.cuda_context();
    ASSERT_TRUE(cuda_context.has_value()) << cuda_context.error().message;
    ASSERT_EQ(cuCtxPushCurrent(*cuda_context), CUDA_SUCCESS);
    const CUresult stream_created = cuStreamCreate(&cuda_stream_, CU_STREAM_NON_BLOCKING);
    CUcontext popped = nullptr;
    ASSERT_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
    ASSERT_EQ(stream_created, CUDA_SUCCESS);
    ASSERT_EQ(cuLaunchHostFunc(cuda_stream_, block_stream_until_released, &submit_released_),
              CUDA_SUCCESS);

    auto external_semaphore = device_.create_external_semaphore(cuda_stream_);
    ASSERT_TRUE(external_semaphore.has_value()) << external_semaphore.error().message;
    external_semaphore_ = std::move(*external_semaphore);
    auto external_token = external_semaphore_.signal();
    ASSERT_TRUE(external_token.has_value()) << external_token.error().message;

    auto token = swapchain_.acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << token.error().message;
    const auto acquire_consumed = token->acquire_gate.consumed();
    ASSERT_TRUE(acquire_consumed.has_value()) << acquire_consumed.error().message;
    ASSERT_TRUE(*acquire_consumed) << "expected a synthetic (pre-consumed) shared-mode acquire";
    const SubmitDesc submit_desc{
        .wait_external_sync_tokens = {&*external_token, 1},
    };
    const auto submitted = device_.submit(submit_desc);
    ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
    // *token goes out of scope here without present().
  }

  // Replace the fixture's FIFO swapchain with a shared-present one, or skip where the
  // device/surface has no shared presentable images. Presents one frame so the next
  // acquire() is the synthetic shared-mode kind.
  void switch_to_shared_present_or_skip() {
    auto caps = device_.capabilities();
    ASSERT_TRUE(caps.has_value()) << caps.error().message;
    auto surface_caps = device_.surface_capabilities(surface_);
    ASSERT_TRUE(surface_caps.has_value()) << surface_caps.error().message;
    if (!caps->shared_present ||
        !surface_caps->present_modes.contains(PresentMode::kSharedDemandRefresh)) {
      GTEST_SKIP() << "no shared presentable images on this device/surface";
    }
    ASSERT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
    auto swapchain = device_.create_swapchain(
        {.surface = surface_, .present_mode = PresentMode::kSharedDemandRefresh, .image_count = 1});
    ASSERT_TRUE(swapchain.has_value()) << swapchain.error().message;
    swapchain_ = std::move(*swapchain);
    present_one_frame();
  }

  // Let the armed submit complete. Idempotent; TearDown calls it as a safety net so an
  // assertion failure between arm and release cannot wedge wait_idle().
  void release_blocked_submit() {
    submit_released_.store(true);
    if (cuda_stream_ != nullptr) {
      EXPECT_EQ(cuStreamSynchronize(cuda_stream_), CUDA_SUCCESS);
    }
  }

  Window window_;
  Surface surface_;
  Device device_;
  Swapchain swapchain_;

  std::atomic<bool> submit_released_{false};
  CUstream cuda_stream_ = nullptr;
  ExternalSemaphore external_semaphore_;
};

TEST_F(SwapchainResizeWindowedTest, ResizePreservesFormatAndKeepsPresenting) {
  const Format format_before = swapchain_.format().value();
  const ColorSpace color_space_before = swapchain_.color_space().value();

  present_one_frame();

  ASSERT_TRUE(swapchain_.resize(surface_extent()).has_value());

  // The in-place contract: same handle, same format/color space, images re-fetchable by
  // the same indices, and the extent tracks the surface.
  EXPECT_EQ(swapchain_.format().value(), format_before);
  EXPECT_EQ(swapchain_.color_space().value(), color_space_before);
  const uint32_t image_count = swapchain_.image_count().value();
  ASSERT_GT(image_count, 0u);
  for (uint32_t i = 0; i < image_count; ++i) {
    ASSERT_TRUE(swapchain_.image(i).has_value());
  }
  const Extent2D window_extent = surface_extent();
  const Extent2D swapchain_extent = swapchain_.extent().value();
  EXPECT_EQ(swapchain_extent.width, window_extent.width);
  EXPECT_EQ(swapchain_extent.height, window_extent.height);

  present_one_frame();
}

TEST_F(SwapchainResizeWindowedTest, RepeatedResizeCyclesRetiredGenerations) {
  // Back-to-back resizes with presents in between walk a retired generation through the
  // full retire -> reap-on-next-resize cycle; destruction at TearDown reaps the rest.
  for (int i = 0; i < 3; ++i) {
    present_one_frame();
    ASSERT_TRUE(swapchain_.resize(surface_extent()).has_value());
  }
  present_one_frame();
}

TEST_F(SwapchainResizeWindowedTest, ResizeWithOutstandingAcquireReleasesTheImage) {
  // Resizing with an acquired-but-unpresented frame is allowed: the image is given back to
  // the presentation engine internally (vkReleaseSwapchainImages) and the token must simply
  // be dropped. Neither gate is consumed here, exercising the internal semaphore drain.
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  ASSERT_TRUE(swapchain_.resize(surface_extent()).has_value());
  {
    const AcquireSyncToken dropped = std::move(*token);
  }

  present_one_frame();
}

TEST_F(SwapchainResizeWindowedTest, SetPresentModeSwitchesWithoutRecreation) {
  // The mode the swapchain was created with is always in its compatible set.
  EXPECT_EQ(swapchain_.present_mode().value(), PresentMode::kFIFO);
  ASSERT_TRUE(swapchain_.set_present_mode(PresentMode::kFIFO).has_value());
  present_one_frame();

  // Whether other modes are switchable is a surface/driver property: a success must
  // present cleanly at the new mode, a refusal must be kInvalidArgument and change nothing.
  const auto switched = swapchain_.set_present_mode(PresentMode::kImmediate);
  if (switched.has_value()) {
    EXPECT_EQ(swapchain_.present_mode().value(), PresentMode::kImmediate);
    present_one_frame();
    ASSERT_TRUE(swapchain_.set_present_mode(PresentMode::kFIFO).has_value());
    present_one_frame();
  } else {
    EXPECT_EQ(switched.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(swapchain_.present_mode().value(), PresentMode::kFIFO);
  }
}

TEST_F(SwapchainResizeWindowedTest, ScalingModeCreatesAndPresents) {
  // Scaling support is a surface property; an unsupported request must fail creation with
  // kInvalidArgument (and this environment then cannot test further).
  ASSERT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
  auto scaled = device_.create_swapchain({.surface = surface_,
                                          .present_mode = PresentMode::kFIFO,
                                          .image_count = 3,
                                          .scaling = SwapchainScaling::kStretch});
  if (!scaled.has_value()) {
    EXPECT_EQ(scaled.error().code, ErrorCode::kInvalidArgument);
    GTEST_SKIP() << "surface does not support present scaling: " << scaled.error().message;
  }
  swapchain_ = std::move(*scaled);
  present_one_frame();
  ASSERT_TRUE(swapchain_.resize(surface_extent()).has_value());
  present_one_frame();
}

TEST_F(SwapchainResizeWindowedTest, FailedSubmitLeavesGatesUnconsumed) {
  // A submit that fails validation after collecting some gates must consume none of them:
  // the same gates must remain usable in a corrected retry. (Pre-fix, the acquire gate was
  // consumed before vkQueueSubmit; the retry then failed "already consumed" and the
  // acquire semaphore was left with a pending signal — a wedged frame loop.)
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;

  PresentWaitGate null_gate{};  // invalid on purpose: fails phase-1 validation
  SubmitDesc bad{
      .acquire_gates = {&token->acquire_gate, 1},
      .present_gates = {&null_gate, 1},
  };
  const auto failed = device_.submit(bad);
  ASSERT_FALSE(failed.has_value());
  EXPECT_FALSE(token->acquire_gate.consumed().value()) << "a failed submit must not consume gates";
  EXPECT_FALSE(token->present_gate.consumed().value());

  // The corrected retry with the same gates must submit and present cleanly.
  SubmitDesc good{
      .acquire_gates = {&token->acquire_gate, 1},
      .present_gates = {&token->present_gate, 1},
  };
  const auto submitted = device_.submit(good);
  ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
  EXPECT_TRUE(token->acquire_gate.consumed().value());
  EXPECT_TRUE(token->present_gate.consumed().value());
  ASSERT_TRUE(swapchain_.present(std::move(*token), *submitted).has_value());
}

TEST_F(SwapchainResizeWindowedTest, ResizeLoopInvalidatesFramebuffers) {
  // Regression test for the framebuffer cache: render into swapchain images, resize (which
  // destroys the old views), and render again, repeatedly. The render pass must evict the
  // cached framebuffers over the dying views each cycle; with validation enabled
  // (HOLOVIZ_VALIDATION=1) a stale-view framebuffer surfaces as validation errors, and
  // pre-fix the cache leaked one owned key array per resize.
  auto render_pass = device_.create_render_pass({.render_targets = {{
                                                     .format = swapchain_.format().value(),
                                                 }}});
  ASSERT_TRUE(render_pass.has_value()) << render_pass.error().message;
  auto cmd = device_.create_command_list({.type = CommandListType::kTransient});
  ASSERT_TRUE(cmd.has_value()) << cmd.error().message;

  for (int i = 0; i < 8; ++i) {
    auto token = swapchain_.acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << "cycle " << i << ": " << token.error().message;

    const auto image = swapchain_.image(token->image_index);
    ASSERT_TRUE(image.has_value());
    ASSERT_TRUE(
        render_pass->set_render_image(image->get(), 0, Color{0.0F, 0.0F, 0.0F, 1.0F}).has_value());

    const Extent2D extent = swapchain_.extent().value();
    ASSERT_TRUE(cmd->reset(std::chrono::seconds(5)).has_value());
    ASSERT_TRUE(cmd->begin().has_value());
    ASSERT_TRUE(
        cmd->begin_render(*render_pass, Rect2D{0, 0, extent.width, extent.height}).has_value());
    ASSERT_TRUE(cmd->end_render().has_value());
    ASSERT_TRUE(cmd->end().has_value());

    SubmitDesc submit_desc{
        .command_lists = {&*cmd, 1},
        .acquire_gates = {&token->acquire_gate, 1},
        .present_gates = {&token->present_gate, 1},
    };
    const auto submitted = device_.submit(submit_desc);
    ASSERT_TRUE(submitted.has_value()) << "cycle " << i << ": " << submitted.error().message;
    ASSERT_TRUE(swapchain_.present(std::move(*token), *submitted).has_value());

    ASSERT_TRUE(swapchain_.resize(surface_extent()).has_value()) << "cycle " << i;
  }

  ASSERT_TRUE(device_.wait_idle().has_value());
  ASSERT_TRUE(cmd->destroy(std::chrono::seconds(5)).has_value());
  ASSERT_TRUE(render_pass->destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(SwapchainResizeWindowedTest, StalePreResizeTokenIsRejected) {
  // A frame presented after a resize must carry a token minted by the current swapchain;
  // pre-resize tokens are documented as invalidated and must be refused, not presented.
  auto stale = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(stale.has_value()) << stale.error().message;
  ASSERT_TRUE(swapchain_.present(std::move(*stale), SubmitSyncToken{}).has_value());

  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  AcquireSyncToken held = std::move(*token);
  // Consume both gates CPU-side so the resize below is not racing an un-honored acquire
  // wait, then deliberately hold the token across the resize.
  ASSERT_TRUE(held.acquire_gate.wait(std::chrono::seconds(5)).has_value());
  ASSERT_TRUE(held.present_gate.signal().has_value());

  ASSERT_TRUE(device_.wait_idle().has_value());
  ASSERT_TRUE(swapchain_.resize(surface_extent()).has_value());

  const auto presented = swapchain_.present(std::move(held), SubmitSyncToken{});
  EXPECT_FALSE(presented.has_value());

  present_one_frame();
}

TEST_F(SwapchainResizeWindowedTest, TamperedTokenImageIndexIsRejected) {
  // Token fields are caller-mutable and treated as untrusted: an out-of-range image_index
  // must be refused outright (kInvalidArgument) before reaching the presentation engine —
  // a modulo-style check would let image_index + image_count alias a valid index.
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  token->image_index += swapchain_.image_count().value();

  const auto presented = swapchain_.present(std::move(*token), SubmitSyncToken{});
  ASSERT_FALSE(presented.has_value());
  EXPECT_EQ(presented.error().code, ErrorCode::kInvalidArgument);

  // The rejection must leave the swapchain unharmed; the same image can be re-acquired and
  // presented normally. (The tampered token's acquire stays outstanding; the next resize or
  // destroy releases it internally.)
  ASSERT_TRUE(swapchain_.resize(surface_extent()).has_value());
  present_one_frame();
}

TEST_F(SwapchainResizeWindowedTest, DestroyTimeoutIsABoundAndBackpressureIsRetryable) {
  // destroy()'s timeout is a bound, not a floor: with presents potentially still in flight,
  // a zero-timeout probe either succeeds or reports kBackpressured without tearing anything
  // down — the handle must stay fully usable and a later bounded destroy must succeed.
  present_one_frame();

  const auto probed = swapchain_.destroy(std::chrono::nanoseconds::zero());
  if (!probed.has_value()) {
    ASSERT_EQ(probed.error().code, ErrorCode::kBackpressured) << probed.error().message;
    // Nothing was destroyed on backpressure: the swapchain still presents.
    present_one_frame();
    ASSERT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
  }
  EXPECT_FALSE(static_cast<bool>(swapchain_));
}

TEST_F(SwapchainResizeWindowedTest, SecondAcquireWithoutPresentIsRejected) {
  // P1-8 regression: only one acquire may be outstanding. A second acquire used to pass
  // the slot's one-shot binary acquire semaphore to vkAcquireNextImageKHR again and
  // overwrite current_image_, losing the first image; it must be rejected up front with
  // kInvalidState and leave the first token fully usable.
  auto first = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(first.has_value()) << first.error().message;

  const auto second = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_FALSE(second.has_value()) << "second acquire without present must be rejected";
  EXPECT_EQ(second.error().code, ErrorCode::kInvalidState);

  // The rejection must not have touched anything: the first frame presents normally.
  ASSERT_TRUE(swapchain_.present(std::move(*first), SubmitSyncToken{}).has_value());
  present_one_frame();

  // Dropping without presenting goes through resize()/destroy(), which release the frame
  // internally; the loop then resumes cleanly.
  auto abandoned = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(abandoned.has_value()) << abandoned.error().message;
  ASSERT_TRUE(swapchain_.resize(surface_extent()).has_value());
  {
    const AcquireSyncToken dropped = std::move(*abandoned);
  }
  present_one_frame();
}

TEST_F(SwapchainResizeWindowedTest, BackpressuredDestroyLeavesOutstandingAcquirePresentable) {
  // P1-9 regression: destroy() must complete every wait that can backpressure BEFORE the
  // irreversible release of an acquired-but-unpresented image. Pre-fix, with the render
  // submit retired but an older present's fence still pending, a zero-timeout destroy()
  // released image B back to the presentation engine and only then returned
  // kBackpressured — the "fully intact and retryable" contract was broken and the
  // outstanding token could present an image the engine already owned.
  present_one_frame();  // present A queued; its fence signals no earlier than the next vsync

  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  SubmitDesc submit_desc{
      .acquire_gates = {&token->acquire_gate, 1},
      .present_gates = {&token->present_gate, 1},
  };
  const auto submitted = device_.submit(submit_desc);
  ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
  // Retire the render fully so the zero-timeout probe reaches the release fast path
  // directly (no slot-drain backpressure first).
  ASSERT_TRUE(device_.wait_on_submit(*submitted, std::chrono::seconds(5)).has_value());

  const auto probed = swapchain_.destroy(std::chrono::nanoseconds::zero());
  if (!probed.has_value()) {
    ASSERT_EQ(probed.error().code, ErrorCode::kBackpressured) << probed.error().message;
    // Nothing irreversible happened: the outstanding frame is still owned by this
    // swapchain and must present cleanly (pre-fix this presented a released image).
    ASSERT_TRUE(swapchain_.present(std::move(*token), *submitted).has_value());
    ASSERT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
  }
  // When the probe won the race (present A's fence signaled first), destroy() succeeded
  // and released the outstanding frame internally — nothing further to assert.
  EXPECT_FALSE(static_cast<bool>(swapchain_));
}

TEST_F(SwapchainResizeWindowedTest, ResizeWithSubmittedUnpresentedFrameWaitsForTheSubmit) {
  // P1-7 regression: a submit consumed both gates but the frame was never presented. The
  // submit's timeline value must be recorded on the frame slot at submit time, so resize()
  // drains it — pre-fix the slot token was only written by present(), the zero-timeout
  // probe below "succeeded", and the image/semaphores were released while still in flight.
  arm_blocked_unpresented_submit();
  if (HasFatalFailure()) {
    return;  // TearDown unblocks the rig
  }

  const auto probed = swapchain_.resize(surface_extent(), std::chrono::nanoseconds::zero());
  ASSERT_FALSE(probed.has_value()) << "resize() must not proceed under an in-flight submit";
  EXPECT_EQ(probed.error().code, ErrorCode::kBackpressured) << probed.error().message;

  release_blocked_submit();
  ASSERT_TRUE(swapchain_.resize(surface_extent(), std::chrono::seconds(5)).has_value());
  present_one_frame();
}

TEST_F(SwapchainResizeWindowedTest, DestroyWithSubmittedUnpresentedFrameWaitsForTheSubmit) {
  // Same shape as above for destroy(): the zero-timeout contract is kBackpressured with
  // nothing destroyed while the unpresented frame's submit is in flight, then a bounded
  // retry succeeds once the submit can retire.
  arm_blocked_unpresented_submit();
  if (HasFatalFailure()) {
    return;  // TearDown unblocks the rig
  }

  const auto probed = swapchain_.destroy(std::chrono::nanoseconds::zero());
  ASSERT_FALSE(probed.has_value()) << "destroy() must not proceed under an in-flight submit";
  EXPECT_EQ(probed.error().code, ErrorCode::kBackpressured) << probed.error().message;
  ASSERT_TRUE(static_cast<bool>(swapchain_)) << "backpressure must leave the handle usable";

  release_blocked_submit();
  ASSERT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
  EXPECT_FALSE(static_cast<bool>(swapchain_));
}

TEST_F(SwapchainResizeWindowedTest, SharedPresentResizeWithGatelessSubmitWaitsForTheSubmit) {
  // P1-1 regression: in shared present modes the render submit after a synthetic acquire
  // carries no gate, so no frame slot records it and present() has not run yet. resize()
  // must still find it (the drain folds in the device's last submitted timeline value) —
  // pre-fix the zero-timeout probe "succeeded" and recreated the swapchain, destroying the
  // shared image's views while the submit was still in flight.
  switch_to_shared_present_or_skip();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  arm_blocked_gateless_shared_submit();
  if (HasFatalFailure()) {
    return;  // TearDown unblocks the rig
  }

  const auto probed = swapchain_.resize(surface_extent(), std::chrono::nanoseconds::zero());
  ASSERT_FALSE(probed.has_value()) << "resize() must not proceed under an in-flight submit";
  EXPECT_EQ(probed.error().code, ErrorCode::kBackpressured) << probed.error().message;

  release_blocked_submit();
  ASSERT_TRUE(swapchain_.resize(surface_extent(), std::chrono::seconds(5)).has_value());
  present_one_frame();
}

TEST_F(SwapchainResizeWindowedTest, SharedPresentDestroyWithGatelessSubmitWaitsForTheSubmit) {
  // Same shape as above for destroy(): kBackpressured with nothing destroyed while the
  // gate-less shared-image submit is in flight, then a bounded retry succeeds.
  switch_to_shared_present_or_skip();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  arm_blocked_gateless_shared_submit();
  if (HasFatalFailure()) {
    return;  // TearDown unblocks the rig
  }

  const auto probed = swapchain_.destroy(std::chrono::nanoseconds::zero());
  ASSERT_FALSE(probed.has_value()) << "destroy() must not proceed under an in-flight submit";
  EXPECT_EQ(probed.error().code, ErrorCode::kBackpressured) << probed.error().message;
  ASSERT_TRUE(static_cast<bool>(swapchain_)) << "backpressure must leave the handle usable";

  release_blocked_submit();
  ASSERT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
  EXPECT_FALSE(static_cast<bool>(swapchain_));
}

}  // namespace
}  // namespace holoviz
