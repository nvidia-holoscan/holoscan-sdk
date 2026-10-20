// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <memory>
#include <optional>
#include <utility>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/errors.hpp>
#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/backends/present_backend.hpp"
#include "holoviz_op/backends/render_backend.hpp"

#include <gtest/gtest.h>

namespace holoscan::visualization::backends {
namespace {

// Minimal stub backends. Confirm the interfaces are concretely implementable and that a
// RenderBackend's SyncToken flows into PresentBackend::present() without either side
// needing to know the other's concrete type.
//
// The stub also models the quiescence side of the FrameData device-memory lifetime
// contract (frame_data.hpp): recycle_grant_count is bumped only when a quiescing call
// (begin_frame/abort_frame/destroy) returns SUCCESSFULLY. The next_*_error members inject
// a one-shot failure into the corresponding call, which must then grant no recycle
// permission.
class StubRenderBackend : public RenderBackend {
 public:
  expected<void> allocate(const LayerStack& layers, const RenderConfig& config) override {
    allocate_call_count++;
    allocated_layer_count = layers.layers().size();
    last_max_frames_in_flight = config.max_frames_in_flight;
    return {};
  }
  expected<void> build(ImageFormat) override {
    build_call_count++;
    return {};
  }
  expected<void> destroy(std::chrono::nanoseconds) override {
    if (next_destroy_error) {
      auto error = std::move(*next_destroy_error);
      next_destroy_error.reset();
      return make_unexpected(std::move(error));
    }
    recycle_grant_count++;
    return {};
  }
  expected<FrameCommands> begin_frame(const AcquireToken& acquired, uint32_t frame_index) override {
    begin_frame_call_count++;
    last_acquired_image_index = acquired.image_index;
    last_begin_frame_slot = frame_index;
    if (next_begin_frame_error) {
      auto error = std::move(*next_begin_frame_error);
      next_begin_frame_error.reset();
      return make_unexpected(std::move(error));
    }
    recycle_grant_count++;
    return FrameCommands{std::make_shared<int>(0)};
  }
  expected<void> record(FrameCommands&, const RenderList&, const CompositeSpec&) override {
    record_call_count++;
    return {};
  }
  expected<SyncToken> submit(const FrameCommands&, const SubmitContext& ctx) override {
    last_submit_slot = ctx.frame_index;
    return SyncToken{std::make_shared<int>(42)};
  }
  expected<void> abort_frame(FrameCommands&) override {
    abort_frame_call_count++;
    if (next_abort_frame_error) {
      auto error = std::move(*next_abort_frame_error);
      next_abort_frame_error.reset();
      return make_unexpected(std::move(error));
    }
    recycle_grant_count++;
    return {};
  }
  expected<BackendInfo> info() const override { return BackendInfo{"stub"}; }
  expected<AllocationStats> stats() const override { return AllocationStats{}; }

  int allocate_call_count = 0;
  int build_call_count = 0;
  int begin_frame_call_count = 0;
  int record_call_count = 0;
  int abort_frame_call_count = 0;
  std::size_t allocated_layer_count = 0;
  uint32_t last_max_frames_in_flight = 0;
  uint32_t last_acquired_image_index = UINT32_MAX;
  uint32_t last_begin_frame_slot = UINT32_MAX;
  uint32_t last_submit_slot = UINT32_MAX;
  // One-shot injected failures for the three quiescing calls.
  std::optional<Error> next_begin_frame_error;
  std::optional<Error> next_abort_frame_error;
  std::optional<Error> next_destroy_error;
  // Bumped only on a SUCCESSFUL begin_frame/abort_frame/destroy return — the only events
  // that end the caller's device-buffer retention duty per frame_data.hpp.
  int recycle_grant_count = 0;
};

class StubPresentBackend : public PresentBackend {
 public:
  // Returns the actual clamped values. Uses 3 as the windowed default (not 2) per the
  // SwapchainDesc doc: "typically 2 for exclusive display, 3 for windowed."
  expected<CreatedSwapchainInfo> create_swapchain(const SwapchainDesc& desc) override {
    last_created_image_count = desc.image_count.value_or(3);
    return CreatedSwapchainInfo{
        desc.preferred_extent, last_created_image_count, desc.preferred_mode};
  }
  expected<void> resize(Extent2D) override { return {}; }
  expected<void> destroy(std::chrono::nanoseconds) override { return {}; }
  expected<AcquireToken> acquire(std::chrono::nanoseconds timeout) override {
    acquire_call_count++;
    return AcquireToken{next_image_index++, SyncToken{}, ++next_issuance};
  }
  expected<void> present(const PresentArgs& args) override {
    last_present_id = args.present_id;
    last_render_done = args.render_done;
    last_presented_image_index = args.image_index;
    last_presented_issuance = args.acquire_issuance;
    return {};
  }
  expected<WaitResult> wait_for_present(PresentId id, std::chrono::nanoseconds timeout) override {
    last_wait_id = id;
    last_wait_timeout = timeout;
    return WaitResult::kDisplayed;
  }
  PresentInfo info() const override { return PresentInfo{"stub", false, false}; }
  expected<SwapchainInfo> swapchain_info() const override {
    return SwapchainInfo{
        {1920, 1080}, last_created_image_count, holoviz::PresentMode::kFIFO, ImageFormat::kUnknown};
  }

  int acquire_call_count = 0;
  uint32_t next_image_index = 7;  // non-zero start to catch hardcoded-zero bugs
  uint64_t next_issuance = 100;   // non-zero start, distinct from image indices
  uint32_t last_created_image_count = 0;
  PresentId last_present_id = 0;
  SyncToken last_render_done;
  uint32_t last_presented_image_index = UINT32_MAX;
  uint64_t last_presented_issuance = 0;
  PresentId last_wait_id = 0;
  std::chrono::nanoseconds last_wait_timeout{};
};

TEST(RenderBackendTest, AllocateSeesFullLayerSetAndMaxFrames) {
  LayerStack stack;
  stack.add_image_layer("video_in", ImageLayerDesc{ImageFormat::kNV12, 1920, 1080});
  stack.add_geometry_layer("detections",
                           GeometryLayerDesc{{GeometryTopology::kRectangleList}, 256});

  StubRenderBackend backend;
  RenderConfig config;
  config.max_frames_in_flight = 3;
  ASSERT_TRUE(backend.allocate(stack, config).has_value());

  EXPECT_EQ(backend.allocate_call_count, 1);
  EXPECT_EQ(backend.allocated_layer_count, 2u);
  EXPECT_EQ(backend.last_max_frames_in_flight, 3u);
}

// Verifies the complete acquire→begin_frame→record→submit→present sequence:
// (1) begin_frame is called and receives the exact AcquireToken image_index;
// (2) SubmitContext::frame_index matches begin_frame's frame_index (required invariant);
// (3) acquired.image_index reaches present() unchanged;
// (4) SyncToken from submit() reaches present() (neither seam knows the other's type).
TEST(RenderPresentSeamTest, AcquireTokenFlowsThroughRenderIntoPresent) {
  StubRenderBackend render_backend;
  StubPresentBackend present_backend;

  // present_backend.acquire() starts at next_image_index=7 (non-zero to catch hardcoded-0 bugs).
  auto acquired_try = present_backend.acquire(std::chrono::seconds(1));
  ASSERT_TRUE(acquired_try.has_value());
  auto acquired = *acquired_try;
  EXPECT_EQ(acquired.image_index, 7u);

  constexpr uint32_t kFrameSlot = 0;
  auto cmds_try = render_backend.begin_frame(acquired, kFrameSlot);
  ASSERT_TRUE(cmds_try.has_value());
  auto cmds = *cmds_try;
  EXPECT_EQ(render_backend.begin_frame_call_count, 1);
  // begin_frame must receive the exact image_index from acquire().
  EXPECT_EQ(render_backend.last_acquired_image_index, acquired.image_index);
  EXPECT_EQ(render_backend.last_begin_frame_slot, kFrameSlot);

  ASSERT_TRUE(render_backend.record(cmds, RenderList{}, CompositeSpec{}).has_value());
  EXPECT_EQ(render_backend.record_call_count, 1);

  // SubmitContext::frame_index must equal begin_frame's frame_index.
  auto render_done_try = render_backend.submit(cmds, SubmitContext{kFrameSlot});
  ASSERT_TRUE(render_done_try.has_value());
  auto render_done = *render_done_try;
  EXPECT_EQ(render_backend.last_submit_slot, kFrameSlot);

  const PresentId kFrameId = 42;
  ASSERT_TRUE(
      present_backend
          .present(PresentArgs{acquired.image_index, acquired.issuance, render_done, kFrameId, 0})
          .has_value());

  // acquired.image_index and acquired.issuance must reach present() unchanged — the
  // issuance is what lets present() redeem the exact acquire, not a recycled index.
  EXPECT_EQ(present_backend.last_presented_image_index, acquired.image_index);
  EXPECT_EQ(present_backend.last_presented_issuance, acquired.issuance);
  EXPECT_NE(acquired.issuance, 0u);
  EXPECT_EQ(present_backend.last_present_id, kFrameId);
  // SyncToken is opaque — verify the same shared_ptr impl flows through.
  EXPECT_EQ(present_backend.last_render_done.impl, render_done.impl);
}

// Verifies the abort path is expressible through the seam: a frame opened by begin_frame
// can be abandoned via abort_frame() and the image released via an empty present (null
// render_done), without either seam knowing the other's type.
TEST(RenderPresentSeamTest, AbortFrameAndEmptyPresentFlowThroughTheSeam) {
  StubRenderBackend render_backend;
  StubPresentBackend present_backend;

  auto acquired_try = present_backend.acquire(std::chrono::seconds(1));
  ASSERT_TRUE(acquired_try.has_value());
  auto cmds_try = render_backend.begin_frame(*acquired_try, 0);
  ASSERT_TRUE(cmds_try.has_value());

  ASSERT_TRUE(render_backend.abort_frame(*cmds_try).has_value());
  EXPECT_EQ(render_backend.abort_frame_call_count, 1);

  ASSERT_TRUE(present_backend
                  .present(PresentArgs{
                      acquired_try->image_index, acquired_try->issuance, SyncToken{nullptr}, 0, 0})
                  .has_value());
  EXPECT_EQ(present_backend.last_presented_image_index, acquired_try->image_index);
  EXPECT_EQ(present_backend.last_render_done.impl, nullptr);
}

// The next three tests encode the failure side of the FrameData device-memory lifetime
// contract (frame_data.hpp): an ERROR return from a quiescing call (begin_frame's
// kBackpressured timeout, a failed abort_frame, a failed destroy) proves nothing about GPU
// quiescence and must not release producer buffers. The stub grants recycle permission
// (recycle_grant_count) only on success, so a caller that keys buffer release off "the call
// returned" instead of "the call returned successfully" trips these assertions.

// A kBackpressured begin_frame() grants no recycle permission; only the later successful
// retry on the same frame_index does.
TEST(FrameDataLifetimeSeamTest, BeginFrameBackpressureDoesNotGrantRecycle) {
  StubRenderBackend render_backend;
  StubPresentBackend present_backend;

  auto acquired_try = present_backend.acquire(std::chrono::seconds(1));
  ASSERT_TRUE(acquired_try.has_value());

  constexpr uint32_t kFrameSlot = 0;
  render_backend.next_begin_frame_error = Error{ErrorCode::kBackpressured, "slot busy"};
  auto backpressured = render_backend.begin_frame(*acquired_try, kFrameSlot);
  ASSERT_FALSE(backpressured.has_value());
  EXPECT_EQ(backpressured.error().code, ErrorCode::kBackpressured);
  // The slot did not recycle: the previous frame's producer buffers must stay retained.
  EXPECT_EQ(render_backend.recycle_grant_count, 0);

  // Recovery: retry begin_frame on the same slot; success is the quiescence point.
  auto retried = render_backend.begin_frame(*acquired_try, kFrameSlot);
  ASSERT_TRUE(retried.has_value());
  EXPECT_EQ(render_backend.last_begin_frame_slot, kFrameSlot);
  EXPECT_EQ(render_backend.recycle_grant_count, 1);
}

// A failed abort_frame() leaves the frame's staged uploads possibly in flight; retention
// continues until a later successful abort_frame() (a retry) succeeds.
TEST(FrameDataLifetimeSeamTest, AbortFrameFailureKeepsRetention) {
  StubRenderBackend render_backend;
  StubPresentBackend present_backend;

  auto acquired_try = present_backend.acquire(std::chrono::seconds(1));
  ASSERT_TRUE(acquired_try.has_value());
  auto cmds_try = render_backend.begin_frame(*acquired_try, 0);
  ASSERT_TRUE(cmds_try.has_value());
  const int grants_after_open = render_backend.recycle_grant_count;

  render_backend.next_abort_frame_error = Error{ErrorCode::kFailure, "cancel failed"};
  auto abort_try = render_backend.abort_frame(*cmds_try);
  ASSERT_FALSE(abort_try.has_value());
  // The failed abort released nothing: this frame's FrameData memory may still be read.
  EXPECT_EQ(render_backend.recycle_grant_count, grants_after_open);

  // Recovery: retry abort_frame; only its successful return ends the retention duty.
  ASSERT_TRUE(render_backend.abort_frame(*cmds_try).has_value());
  EXPECT_EQ(render_backend.recycle_grant_count, grants_after_open + 1);
}

// A failed destroy() means the device may still reference FrameData memory; only a
// successful retry (e.g. with a longer timeout) is a quiescence point.
TEST(FrameDataLifetimeSeamTest, DestroyFailureKeepsRetention) {
  StubRenderBackend render_backend;

  render_backend.next_destroy_error = Error{ErrorCode::kFailure, "device wait timed out"};
  auto destroy_try = render_backend.destroy(std::chrono::milliseconds(1));
  ASSERT_FALSE(destroy_try.has_value());
  EXPECT_EQ(render_backend.recycle_grant_count, 0);

  ASSERT_TRUE(render_backend.destroy(std::chrono::seconds(1)).has_value());
  EXPECT_EQ(render_backend.recycle_grant_count, 1);
}

// Verifies the create_swapchain → image_count lifecycle:
// The returned CreatedSwapchainInfo::image_count is the actual ring size the renderer
// must size to. Windowed default is 3, not 2.
TEST(SwapchainLifecycleTest, CreateSwapchainReturnsActualImageCount) {
  StubPresentBackend present_backend;

  // nullopt → backend chooses; windowed default is 3.
  SwapchainDesc desc;
  desc.preferred_extent = {1920, 1080};
  desc.preferred_mode = holoviz::PresentMode::kFIFOLatestReady;
  auto info_result = present_backend.create_swapchain(desc);
  ASSERT_TRUE(info_result.has_value());
  auto info = *info_result;

  EXPECT_EQ(info.image_count, 3u);  // windowed default, not 2
  EXPECT_EQ(present_backend.last_created_image_count, 3u);
  EXPECT_EQ(info.extent.width, 1920u);
  EXPECT_EQ(info.extent.height, 1080u);
  EXPECT_EQ(info.present_mode, holoviz::PresentMode::kFIFOLatestReady);

  // The caller must feed this count into RenderConfig::max_frames_in_flight.
  RenderConfig render_config;
  render_config.max_frames_in_flight = info.image_count;
  StubRenderBackend render_backend;
  ASSERT_TRUE(render_backend.allocate(LayerStack{}, render_config).has_value());
  EXPECT_EQ(render_backend.last_max_frames_in_flight, info.image_count);
}

// kWaitForever is part of the public API contract: it must reach the backend's
// wait_for_present() unchanged as the duration-typed "no deadline" sentinel.
TEST(PresentBackendApiTest, KWaitForeverReachesWaitForPresentUnchanged) {
  StubPresentBackend present_backend;

  auto result = present_backend.wait_for_present(1, kWaitForever);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, WaitResult::kDisplayed);
  EXPECT_EQ(present_backend.last_wait_id, 1u);
  EXPECT_EQ(present_backend.last_wait_timeout, std::chrono::nanoseconds::max());
}

}  // namespace
}  // namespace holoscan::visualization::backends
