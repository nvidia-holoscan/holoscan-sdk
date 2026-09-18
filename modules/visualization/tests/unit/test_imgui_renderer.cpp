// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Regression tests for the raw Renderer (public API only — the impl and its
// detail headers stay out of tests):
//  - the full ImDrawData display transform (DisplayPos, DisplaySize, FramebufferScale)
//    driven by hand-built draw data through upload_draw_data()/record_draws();
//  - draw lists past the 16-bit vertex boundary (ImGuiBackendFlags_RendererHasVtxOffset);
//  - buffer growth from tiny initial capacities, staging-slot ring reuse past
//    frames_in_flight with real per-slot submit tokens (frame-keyed markers make every
//    frame's pixels individually identifiable), and slot reuse under a host-gated
//    (backlogged) CUDA stream — including the 2-slot/3-image mismatch the per-image
//    token contract used to corrupt;
//  - the factory's null-stream rejection.
// The error-injection cases for staging failures live in
// test_imgui_renderer_fault_injection.cpp, which reaches through the detail seam.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoviz/holoviz.hpp>
#include <holoviz/imgui/ui.hpp>  // Renderer + the bundled Dear ImGui API
#include <holoviz/text/text_renderer.hpp>

#include "image_compare_test_util.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz::imgui {
namespace {

using test_util::kInkThreshold;
using test_util::try_create_device;

bool is_ink(const std::vector<uint8_t>& pixels, Extent2D extent, uint32_t x, uint32_t y) {
  const size_t i = (static_cast<size_t>(y) * extent.width + x) * 4;
  return pixels[i] > kInkThreshold || pixels[i + 1] > kInkThreshold ||
         pixels[i + 2] > kInkThreshold;
}

/// Bounding box of all inked pixels; {0,0,0,0} when the image is blank.
struct InkBox {
  uint32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // half-open [x0,x1) x [y0,y1)
  bool any = false;
};

InkBox ink_bbox(const std::vector<uint8_t>& pixels, Extent2D extent) {
  InkBox box{extent.width, extent.height, 0, 0, false};
  for (uint32_t y = 0; y < extent.height; ++y) {
    for (uint32_t x = 0; x < extent.width; ++x) {
      if (!is_ink(pixels, extent, x, y)) {
        continue;
      }
      box.any = true;
      box.x0 = std::min(box.x0, x);
      box.y0 = std::min(box.y0, y);
      box.x1 = std::max(box.x1, x + 1);
      box.y1 = std::max(box.y1, y + 1);
    }
  }
  if (!box.any) {
    box = InkBox{};
  }
  return box;
}

/// Offscreen render scaffolding reused across frames so consecutive frames target the same
/// image (unlike render_with(), which builds and tears down per call).
class ImGuiRendererTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = try_create_device();
    if (!device_) {
      HOLOVIZ_ENV_SKIP() << "no usable GPU/adapter on this machine";
    }
    const auto context = device_.cuda_context();
    ASSERT_TRUE(context.has_value()) << context.error().message;
    cuda_context_ = *context;
    if (cuCtxSetCurrent(cuda_context_) != CUDA_SUCCESS ||
        cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
      HOLOVIZ_ENV_SKIP() << "cannot create a CUDA stream on the device context";
    }
  }

  void TearDown() override {
    release_stream_gate();  // idempotent; wait_idle would hang on a still-gated stream
    if (releaser_.joinable()) {
      releaser_.join();
    }
    if (device_) {
      EXPECT_TRUE(device_.wait_idle().has_value());
      drain_command_lists();
    }
    if (stream_ != nullptr) {
      cuStreamSynchronize(stream_);
      cuStreamDestroy(stream_);
    }
    if (target_) {
      EXPECT_TRUE(render_pass_.destroy().has_value());
      EXPECT_TRUE(target_.destroy().has_value());
    }
    if (device_) {
      EXPECT_TRUE(device_.destroy().has_value());
    }
  }

  void create_target(Extent2D extent) {
    extent_ = extent;
    auto target = device_.create_image({
        .extent = extent,
        .format = Format::kR8G8B8A8Unorm,
        .usage = ImageUsage::kRenderTarget,
        .debug_name = "imgui backend test target",
    });
    ASSERT_TRUE(target.has_value()) << target.error().message;
    target_ = std::move(*target);
    auto render_pass =
        device_.create_render_pass({.render_targets = {{.format = Format::kR8G8B8A8Unorm}}});
    ASSERT_TRUE(render_pass.has_value()) << render_pass.error().message;
    render_pass_ = std::move(*render_pass);
    ASSERT_TRUE(render_pass_.set_render_image(target_, 0, {0.0F, 0.0F, 0.0F, 1.0F}).has_value());
  }

  /// Record the backend's draws into the shared target and submit, waiting on the upload
  /// token and (when non-default) the previous frame's submit so target writes stay ordered.
  SubmitSyncToken record_and_submit(Renderer& backend, const FrameSync& sync,
                                    SubmitSyncToken wait_previous = {}) {
    return record_and_submit(backend, sync, render_pass_, wait_previous);
  }

  /// As above, into an explicit render pass (multi-target tests).
  SubmitSyncToken record_and_submit(Renderer& backend, const FrameSync& sync,
                                    RenderPass& render_pass, SubmitSyncToken wait_previous = {}) {
    auto command_list = device_.create_command_list({.type = CommandListType::kTransient});
    EXPECT_TRUE(command_list.has_value());
    EXPECT_TRUE(command_list->begin().has_value());
    EXPECT_TRUE(
        command_list->begin_render(render_pass, {0, 0, extent_.width, extent_.height}).has_value());
    const auto recorded = backend.record_draws(*command_list, extent_);
    EXPECT_TRUE(recorded.has_value()) << (recorded ? "" : recorded.error().message);
    EXPECT_TRUE(command_list->end_render().has_value());
    EXPECT_TRUE(command_list->end().has_value());
    SubmitDesc submit_desc{
        .command_lists = {&command_list.value(), 1},
    };
    ExternalSyncToken upload_token = sync.upload_done;
    submit_desc.wait_external_sync_tokens = {&upload_token, 1};
    if (wait_previous.timeline_value != 0) {
      submit_desc.wait_submit_sync_tokens = {&wait_previous, 1};
    }
    const auto token = device_.submit(submit_desc);
    EXPECT_TRUE(token.has_value()) << (token ? "" : token.error().message);
    // Keep the list alive while its submit is in flight; drained after the test's waits.
    pending_command_lists_.push_back(std::move(*command_list));
    return token.value_or(SubmitSyncToken{});
  }

  /// Destroy the recorded command lists. Call only after the submits retired
  /// (wait_on_submit/wait_idle).
  void drain_command_lists() {
    for (auto& command_list : pending_command_lists_) {
      EXPECT_TRUE(command_list.destroy(std::chrono::seconds(5)).has_value());
    }
    pending_command_lists_.clear();
  }

  void read_target(std::vector<uint8_t>& out_pixels) { read_image(target_, out_pixels); }

  void read_image(Image& image, std::vector<uint8_t>& out_pixels) {
    const size_t row_pitch = static_cast<size_t>(extent_.width) * 4;
    out_pixels.resize(row_pitch * extent_.height);
    ASSERT_TRUE(
        device_
            .download_from_image(image, 0, std::as_writable_bytes(std::span(out_pixels)), row_pitch)
            .has_value());
  }

  /// Host-gate the CUDA stream: everything enqueued after this call stays pending until
  /// release_stream_gate(). Deterministic "backlogged stream" (same rig as the swapchain
  /// resize tests: a host function spinning on an atomic).
  void gate_stream() {
    stream_gate_released_.store(false);
    ASSERT_EQ(cuLaunchHostFunc(
                  stream_,
                  [](void* user) {
                    auto* released = static_cast<std::atomic<bool>*>(user);
                    while (!released->load()) {
                      std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                  },
                  &stream_gate_released_),
              CUDA_SUCCESS);
  }

  void release_stream_gate() { stream_gate_released_.store(true); }

  /// Release the gate from another thread after \p delay, so a host-blocking call made
  /// while the stream is gated (the staging fence, the growth wait) can complete.
  void release_stream_gate_after(std::chrono::milliseconds delay) {
    releaser_ = std::thread([this, delay] {
      std::this_thread::sleep_for(delay);
      release_stream_gate();
    });
  }

  Device device_;
  CUcontext cuda_context_ = nullptr;
  CUstream stream_ = nullptr;
  Image target_;
  RenderPass render_pass_;
  Extent2D extent_{};
  std::vector<CommandList> pending_command_lists_;
  std::atomic<bool> stream_gate_released_{true};
  std::thread releaser_;
};

// ---------------------------------------------------------------------------
// Display transform (hand-built ImDrawData through the public upload path)
// ---------------------------------------------------------------------------

/// One solid quad as a self-contained ImDrawData. Vertices are in display coordinates;
/// the caller controls DisplayPos/DisplaySize/FramebufferScale and the command clip rect.
struct QuadDrawData {
  QuadDrawData(ImGuiContext* context, float x0, float y0, float x1, float y1, ImVec2 display_pos,
               ImVec2 display_size, ImVec2 framebuffer_scale, ImVec4 clip_rect)
      : draw_list(nullptr) {
    ImGui::SetCurrentContext(context);
    const ImVec2 uv = ImGui::GetIO().Fonts->TexUvWhitePixel;
    const ImU32 white = IM_COL32(255, 255, 255, 255);
    draw_list.VtxBuffer.push_back(ImDrawVert{{x0, y0}, uv, white});
    draw_list.VtxBuffer.push_back(ImDrawVert{{x1, y0}, uv, white});
    draw_list.VtxBuffer.push_back(ImDrawVert{{x1, y1}, uv, white});
    draw_list.VtxBuffer.push_back(ImDrawVert{{x0, y1}, uv, white});
    for (const ImDrawIdx index :
         {ImDrawIdx{0}, ImDrawIdx{1}, ImDrawIdx{2}, ImDrawIdx{0}, ImDrawIdx{2}, ImDrawIdx{3}}) {
      draw_list.IdxBuffer.push_back(index);
    }
    ImDrawCmd command;
    command.ClipRect = clip_rect;
    command.TextureId = ImGui::GetIO().Fonts->TexID;
    command.VtxOffset = 0;
    command.IdxOffset = 0;
    command.ElemCount = 6;
    draw_list.CmdBuffer.push_back(command);

    list_ptr = &draw_list;
    draw_data.Valid = true;
    draw_data.CmdLists = &list_ptr;
    draw_data.CmdListsCount = 1;
    draw_data.TotalVtxCount = 4;
    draw_data.TotalIdxCount = 6;
    draw_data.DisplayPos = display_pos;
    draw_data.DisplaySize = display_size;
    draw_data.FramebufferScale = framebuffer_scale;
  }

  ImDrawList draw_list;
  ImDrawList* list_ptr = nullptr;
  ImDrawData draw_data;
};

TEST_F(ImGuiRendererTest, DisplayTransformHonorsPosAndScale) {
  auto backend_result = create_renderer(device_, {.stream = stream_, .frames_in_flight = 2});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  Renderer backend = std::move(*backend_result);
  const auto context = backend.context();
  ASSERT_TRUE(context.has_value());

  const Extent2D extent{128, 128};
  ASSERT_NO_FATAL_FAILURE(create_target(extent));

  uint32_t frame_counter = 0;
  const auto render_quad = [&](const QuadDrawData& quad, std::vector<uint8_t>& pixels) {
    ASSERT_TRUE(backend.begin_frame(frame_counter++, {}).has_value());
    const auto sync = backend.upload_draw_data(&quad.draw_data);
    ASSERT_TRUE(sync.has_value()) << sync.error().message;
    const auto token = record_and_submit(backend, *sync);
    ASSERT_TRUE(device_.wait_on_submit(token, std::chrono::seconds(5)).has_value());
    ASSERT_NO_FATAL_FAILURE(read_target(pixels));
  };

  // Baseline: identity transform, quad at (40,40)-(56,56).
  const QuadDrawData baseline(
      *context, 40, 40, 56, 56, {0, 0}, {128, 128}, {1, 1}, {0, 0, 128, 128});
  std::vector<uint8_t> baseline_pixels;
  ASSERT_NO_FATAL_FAILURE(render_quad(baseline, baseline_pixels));
  const InkBox baseline_box = ink_bbox(baseline_pixels, extent);
  ASSERT_TRUE(baseline_box.any) << "baseline quad did not render";
  EXPECT_NEAR(baseline_box.x0, 40u, 1);
  EXPECT_NEAR(baseline_box.y0, 40u, 1);
  EXPECT_NEAR(baseline_box.x1, 56u, 1);
  EXPECT_NEAR(baseline_box.y1, 56u, 1);

  // Nonzero DisplayPos: the same quad expressed at (140,90) in a display rect whose origin
  // is (100,50) must land exactly where the baseline did.
  const QuadDrawData offset(
      *context, 140, 90, 156, 106, {100, 50}, {128, 128}, {1, 1}, {100, 50, 228, 178});
  std::vector<uint8_t> offset_pixels;
  ASSERT_NO_FATAL_FAILURE(render_quad(offset, offset_pixels));
  const InkBox offset_box = ink_bbox(offset_pixels, extent);
  ASSERT_TRUE(offset_box.any) << "DisplayPos quad did not render";
  EXPECT_EQ(offset_box.x0, baseline_box.x0);
  EXPECT_EQ(offset_box.y0, baseline_box.y0);
  EXPECT_EQ(offset_box.x1, baseline_box.x1);
  EXPECT_EQ(offset_box.y1, baseline_box.y1);

  // 2x FramebufferScale: a 64x64 logical display onto the 128x128 framebuffer. The quad at
  // logical (20,20)-(28,28) must land at (40,40)-(56,56), and a logical clip rect ending at
  // x=24 must scissor the quad's right half at framebuffer x=48.
  const QuadDrawData scaled(*context, 20, 20, 28, 28, {0, 0}, {64, 64}, {2, 2}, {0, 0, 64, 64});
  std::vector<uint8_t> scaled_pixels;
  ASSERT_NO_FATAL_FAILURE(render_quad(scaled, scaled_pixels));
  const InkBox scaled_box = ink_bbox(scaled_pixels, extent);
  ASSERT_TRUE(scaled_box.any) << "FramebufferScale quad did not render";
  EXPECT_NEAR(scaled_box.x0, 40u, 1);
  EXPECT_NEAR(scaled_box.y0, 40u, 1);
  EXPECT_NEAR(scaled_box.x1, 56u, 1);
  EXPECT_NEAR(scaled_box.y1, 56u, 1);

  const QuadDrawData clipped(*context, 20, 20, 28, 28, {0, 0}, {64, 64}, {2, 2}, {0, 0, 24, 64});
  std::vector<uint8_t> clipped_pixels;
  ASSERT_NO_FATAL_FAILURE(render_quad(clipped, clipped_pixels));
  const InkBox clipped_box = ink_bbox(clipped_pixels, extent);
  ASSERT_TRUE(clipped_box.any) << "clipped quad did not render";
  EXPECT_NEAR(clipped_box.x0, 40u, 1);
  EXPECT_NEAR(clipped_box.x1, 48u, 1) << "clip rect was not scaled into framebuffer pixels";

  // A non-empty frame with a degenerate display rect is rejected, not mis-rendered.
  QuadDrawData degenerate(*context, 0, 0, 8, 8, {0, 0}, {0, 0}, {1, 1}, {0, 0, 8, 8});
  ASSERT_TRUE(backend.begin_frame(frame_counter++, {}).has_value());
  const auto rejected = backend.upload_draw_data(&degenerate.draw_data);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(backend.destroy().has_value());
}

// ---------------------------------------------------------------------------
// 16-bit vertex boundary (ImGuiBackendFlags_RendererHasVtxOffset)
// ---------------------------------------------------------------------------

TEST_F(ImGuiRendererTest, DrawListPastSixteenBitVertexBoundaryRenders) {
  auto backend_result = create_renderer(device_,
                                        {.stream = stream_,
                                         .frames_in_flight = 2,
                                         .initial_vertex_capacity = 4096,
                                         .initial_index_capacity = 8192});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  Renderer backend = std::move(*backend_result);
  const auto context = backend.context();
  ASSERT_TRUE(context.has_value());
  {
    ImGui::SetCurrentContext(*context);
    ASSERT_NE(ImGui::GetIO().BackendFlags & ImGuiBackendFlags_RendererHasVtxOffset, 0)
        << "backend must advertise VtxOffset support for large draw lists";
  }

  const Extent2D extent{256, 256};
  ASSERT_NO_FATAL_FAILURE(create_target(extent));

  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  // 17000 quads = 68000 vertices, past the 65536-vertex ImDrawIdx range, then one marker
  // quad that only renders correctly if the VtxOffset-bearing tail commands are honored.
  const auto sync = backend.render_ui(extent, [] {
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    for (int i = 0; i < 17000; ++i) {
      draw_list->AddRectFilled(ImVec2(8, 8), ImVec2(16, 16), IM_COL32(64, 64, 64, 255));
    }
    draw_list->AddRectFilled(ImVec2(200, 200), ImVec2(240, 240), IM_COL32(255, 255, 255, 255));
  });
  ASSERT_TRUE(sync.has_value()) << sync.error().message;

  {
    // The flag must have actually engaged ImGui's offset path (public draw-data fields).
    ImGui::SetCurrentContext(*context);
    const ImDrawData* draw_data = ImGui::GetDrawData();
    ASSERT_NE(draw_data, nullptr);
    EXPECT_GT(draw_data->TotalVtxCount, 65536);
    bool any_vtx_offset = false;
    for (int list_index = 0; list_index < draw_data->CmdListsCount; ++list_index) {
      for (const ImDrawCmd& command : draw_data->CmdLists[list_index]->CmdBuffer) {
        any_vtx_offset |= command.VtxOffset > 0;
      }
    }
    EXPECT_TRUE(any_vtx_offset) << "ImGui did not emit VtxOffset commands past 64k vertices";
  }

  const auto token = record_and_submit(backend, *sync);
  ASSERT_TRUE(device_.wait_on_submit(token, std::chrono::seconds(5)).has_value());
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(read_target(pixels));
  EXPECT_TRUE(is_ink(pixels, extent, 12, 12)) << "pre-boundary quads missing";
  EXPECT_TRUE(is_ink(pixels, extent, 220, 220)) << "post-boundary marker quad missing";
  EXPECT_FALSE(is_ink(pixels, extent, 120, 120)) << "stray geometry (corrupt indices?)";

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(backend.destroy().has_value());
}

// ---------------------------------------------------------------------------
// Buffer growth + staging-slot ring reuse (+ backlogged stream)
// ---------------------------------------------------------------------------

/// X of the frame-keyed marker quad build_growing_ui() draws: a distinct column per frame
/// (mod 10), so stale slot data, a wrong slot, or a truncated upload renders a DIFFERENT
/// marker than the frame's own — unlike a fixed marker, which every frame would satisfy.
uint32_t frame_marker_x(uint64_t frame) {
  return 8 + 12 * static_cast<uint32_t>(frame % 10);
}

/// Draw 2^frame quads plus a frame-keyed marker quad, so the vertex/index counts outgrow a
/// 1-element initial capacity repeatedly while every frame stays individually identifiable.
/// The marker band (y 8..16) never overlaps the bulk band (y 32..48).
void build_growing_ui(uint64_t frame) {
  ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
  const int quads = 1 << std::min<uint64_t>(frame, 10);
  for (int i = 0; i < quads; ++i) {
    draw_list->AddRectFilled(ImVec2(32, 32), ImVec2(48, 48), IM_COL32(128, 128, 128, 255));
  }
  const auto x0 = static_cast<float>(frame_marker_x(frame));
  draw_list->AddRectFilled(ImVec2(x0, 8), ImVec2(x0 + 8, 16), IM_COL32(255, 255, 255, 255));
}

/// Whether \p frame's marker quad is inked (probes the marker center).
bool has_frame_marker(const std::vector<uint8_t>& pixels, Extent2D extent, uint64_t frame) {
  return is_ink(pixels, extent, frame_marker_x(frame) + 4, 12);
}

/// Assert exactly \p frame's marker is present: every other frame in [0, max_frame) whose
/// marker column differs must be absent, so stale or mis-slotted data cannot pass.
void expect_only_frame_marker(const std::vector<uint8_t>& pixels, Extent2D extent, uint64_t frame,
                              uint64_t max_frame) {
  EXPECT_TRUE(has_frame_marker(pixels, extent, frame)) << "frame " << frame << " marker missing";
  for (uint64_t other = 0; other < max_frame; ++other) {
    if (frame_marker_x(other) == frame_marker_x(frame)) {
      continue;
    }
    EXPECT_FALSE(has_frame_marker(pixels, extent, other))
        << "frame " << other << " marker present while frame " << frame << " should own the image";
  }
}

TEST_F(ImGuiRendererTest, TinyCapacitiesGrowAcrossRingReuse) {
  auto backend_result = create_renderer(device_,
                                        {.stream = stream_,
                                         .frames_in_flight = 2,
                                         .initial_vertex_capacity = 1,
                                         .initial_index_capacity = 1});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  Renderer backend = std::move(*backend_result);

  const Extent2D extent{128, 128};
  ASSERT_NO_FATAL_FAILURE(create_target(extent));

  // 8 frames with frames_in_flight = 2: each slot is reused 4 times with the real token of
  // the submit that last consumed it, and the doubling content forces the growth path
  // (destroy/recreate exported buffers, host-wait, pinned staging realloc) on most frames.
  // Every reuse-regime frame is read back and must carry exactly its own marker, so a stale
  // slot rendering the previous owner's bytes fails per frame, not just at the end.
  constexpr uint32_t kFramesInFlight = 2;
  constexpr uint64_t kFrames = 8;
  SubmitSyncToken slot_tokens[kFramesInFlight] = {};
  SubmitSyncToken previous_submit{};
  for (uint64_t frame = 0; frame < kFrames; ++frame) {
    ASSERT_TRUE(backend.begin_frame(frame, slot_tokens[frame % kFramesInFlight]).has_value());
    const auto sync = backend.render_ui(extent, [frame] { build_growing_ui(frame); });
    ASSERT_TRUE(sync.has_value()) << "frame " << frame << ": " << sync.error().message;
    const auto token = record_and_submit(backend, *sync, previous_submit);
    slot_tokens[frame % kFramesInFlight] = token;
    previous_submit = token;
    if (frame >= kFramesInFlight) {
      ASSERT_TRUE(device_.wait_on_submit(token, std::chrono::seconds(5)).has_value());
      std::vector<uint8_t> frame_pixels;
      ASSERT_NO_FATAL_FAILURE(read_target(frame_pixels));
      expect_only_frame_marker(frame_pixels, extent, frame, kFrames);
      EXPECT_TRUE(is_ink(frame_pixels, extent, 40, 40)) << "bulk quads missing on frame " << frame;
    }
  }
  ASSERT_TRUE(device_.wait_on_submit(previous_submit, std::chrono::seconds(5)).has_value());

  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(read_target(pixels));
  expect_only_frame_marker(pixels, extent, kFrames - 1, kFrames);
  EXPECT_TRUE(is_ink(pixels, extent, 40, 40)) << "bulk quads missing after growth";
  EXPECT_FALSE(is_ink(pixels, extent, 100, 100)) << "stray geometry after ring reuse";

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(backend.destroy().has_value());
}

TEST_F(ImGuiRendererTest, SlotReuseAndGrowthUnderBackloggedStream) {
  auto backend_result = create_renderer(device_,
                                        {.stream = stream_,
                                         .frames_in_flight = 2,
                                         .initial_vertex_capacity = 16,
                                         .initial_index_capacity = 32});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  Renderer backend = std::move(*backend_result);

  const Extent2D extent{128, 128};
  ASSERT_NO_FATAL_FAILURE(create_target(extent));

  // Gate the stream so the uploads of frames 0 and 1 genuinely stay in flight, then release
  // the gate from another thread: frame 2 reuses slot 0 while its staging copy is still
  // pending (the staging fence must block, not corrupt) AND outgrows the capacity while the
  // slot's consuming submit is still gated (the growth wait must block until release).
  gate_stream();
  release_stream_gate_after(std::chrono::milliseconds(200));

  constexpr uint32_t kFramesInFlight = 2;
  constexpr uint64_t kFrames = 6;
  SubmitSyncToken slot_tokens[kFramesInFlight] = {};
  SubmitSyncToken previous_submit{};
  for (uint64_t frame = 0; frame < kFrames; ++frame) {
    ASSERT_TRUE(backend.begin_frame(frame, slot_tokens[frame % kFramesInFlight]).has_value());
    const auto sync = backend.render_ui(extent, [frame] { build_growing_ui(frame + 2); });
    ASSERT_TRUE(sync.has_value()) << "frame " << frame << ": " << sync.error().message;
    const auto token = record_and_submit(backend, *sync, previous_submit);
    slot_tokens[frame % kFramesInFlight] = token;
    previous_submit = token;
  }
  // Readback only after the gate released and the last submit retired: a mid-loop readback
  // would host-wait on the still-gated stream.
  ASSERT_TRUE(device_.wait_on_submit(previous_submit, std::chrono::seconds(5)).has_value());

  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(read_target(pixels));
  // The last frame drew build_growing_ui(kFrames - 1 + 2); exactly that marker must remain.
  expect_only_frame_marker(pixels, extent, kFrames - 1 + 2, 10);
  EXPECT_TRUE(is_ink(pixels, extent, 40, 40)) << "bulk quads missing after backlog";
  EXPECT_FALSE(is_ink(pixels, extent, 100, 100)) << "stray geometry after backlog";

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(backend.destroy().has_value());
}

TEST_F(ImGuiRendererTest, TextRendererCancelFrameClosesTheImGuiFrame) {
  auto renderer_result =
      text::create_text_renderer(device_, {.stream = stream_, .frames_in_flight = 2});
  ASSERT_TRUE(renderer_result.has_value()) << renderer_result.error().message;
  text::TextRenderer renderer = std::move(*renderer_result);
  const Extent2D extent{128, 64};

  // Open a frame (ImGui::NewFrame ran), queue text, then abandon it.
  ASSERT_TRUE(renderer.begin_frame(extent, 0, {}).has_value());
  ASSERT_TRUE(renderer.add_text({.position = {4.0F, 4.0F}, .size_px = 13.0F}, "x").has_value());
  EXPECT_TRUE(renderer.cancel_frame().has_value());
  EXPECT_TRUE(renderer.cancel_frame().has_value()) << "cancel must be idempotent";

  // The context accepts a fresh frame after the cancel (EndFrame closed the abandoned one).
  ASSERT_TRUE(renderer.begin_frame(extent, 1, {}).has_value());
  ASSERT_TRUE(renderer.add_text({.position = {4.0F, 4.0F}, .size_px = 13.0F}, "ok").has_value());
  const auto token = renderer.end_frame();
  ASSERT_TRUE(token.has_value()) << token.error().message;

  EXPECT_TRUE(device_.wait_idle().has_value());
  // The end_frame() upload above was never consumed by a submit, so its copies may still
  // be in flight on the CUDA stream: give destroy() a bound to retire them (a zero timeout
  // would honestly report kBackpressured).
  EXPECT_TRUE(renderer.destroy(std::chrono::seconds(1)).has_value());
}

// The pragmatic negative harness for the frame-keyed oracle: one rendered frame satisfies
// exactly its own marker probe, so a test that renders frame N but reads frame M's bytes
// (stale slot, wrong slot, truncated upload) cannot pass expect_only_frame_marker.
TEST_F(ImGuiRendererTest, FrameMarkerOracleDistinguishesFrames) {
  auto backend_result = create_renderer(device_, {.stream = stream_, .frames_in_flight = 2});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  Renderer backend = std::move(*backend_result);

  const Extent2D extent{128, 128};
  ASSERT_NO_FATAL_FAILURE(create_target(extent));

  ASSERT_TRUE(backend.begin_frame(3, {}).has_value());
  const auto sync = backend.render_ui(extent, [] { build_growing_ui(3); });
  ASSERT_TRUE(sync.has_value()) << sync.error().message;
  const auto token = record_and_submit(backend, *sync);
  ASSERT_TRUE(device_.wait_on_submit(token, std::chrono::seconds(5)).has_value());

  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(read_target(pixels));
  EXPECT_TRUE(has_frame_marker(pixels, extent, 3));
  EXPECT_FALSE(has_frame_marker(pixels, extent, 2)) << "oracle cannot tell frame 3 from 2";
  EXPECT_FALSE(has_frame_marker(pixels, extent, 4)) << "oracle cannot tell frame 3 from 4";

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(backend.destroy().has_value());
}

// Regression for the per-swapchain-image token contract this API replaced: 2 renderer slots
// against 3 "swapchain images" (three targets), a gated stream keeping frame 0's consuming
// submit in flight, and per-slot tokens. Frame 2 reuses and outgrows slot 0, so its growth
// must host-wait frame 0's submit (the slot's real consumer) — observable as a lower-bound
// delay against the gate — and no target may end up with another frame's bytes.
TEST_F(ImGuiRendererTest, SlotTokensGateReuseAcrossMismatchedImageCount) {
  auto backend_result = create_renderer(device_,
                                        {.stream = stream_,
                                         .frames_in_flight = 2,
                                         .initial_vertex_capacity = 16,
                                         .initial_index_capacity = 32});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  Renderer backend = std::move(*backend_result);

  const Extent2D extent{128, 128};
  extent_ = extent;
  constexpr uint32_t kFramesInFlight = 2;
  constexpr uint32_t kImages = 3;
  constexpr uint64_t kFrames = 6;
  Image targets[kImages];
  RenderPass passes[kImages];
  for (uint32_t i = 0; i < kImages; ++i) {
    auto target = device_.create_image({
        .extent = extent,
        .format = Format::kR8G8B8A8Unorm,
        .usage = ImageUsage::kRenderTarget,
        .debug_name = "imgui slot-token test target",
    });
    ASSERT_TRUE(target.has_value()) << target.error().message;
    targets[i] = std::move(*target);
    auto render_pass =
        device_.create_render_pass({.render_targets = {{.format = Format::kR8G8B8A8Unorm}}});
    ASSERT_TRUE(render_pass.has_value()) << render_pass.error().message;
    passes[i] = std::move(*render_pass);
    ASSERT_TRUE(passes[i].set_render_image(targets[i], 0, {0.0F, 0.0F, 0.0F, 1.0F}).has_value());
  }

  gate_stream();
  const auto gate_start = std::chrono::steady_clock::now();
  release_stream_gate_after(std::chrono::milliseconds(300));

  SubmitSyncToken slot_tokens[kFramesInFlight] = {};
  SubmitSyncToken image_tokens[kImages] = {};
  for (uint64_t frame = 0; frame < kFrames; ++frame) {
    const auto slot = static_cast<uint32_t>(frame % kFramesInFlight);
    const auto image = static_cast<uint32_t>(frame % kImages);
    ASSERT_TRUE(backend.begin_frame(frame, slot_tokens[slot]).has_value());
    const auto sync = backend.render_ui(extent, [frame] { build_growing_ui(frame); });
    ASSERT_TRUE(sync.has_value()) << "frame " << frame << ": " << sync.error().message;
    if (frame == 2) {
      // Frame 2 (20 vertices) outgrew slot 0's 16-vertex capacity; the growth host-waits the
      // slot's consuming submit (frame 0), which the 300 ms gate is still holding in flight.
      // Lower bound only, so a slow machine cannot flake this — only a missing wait can.
      EXPECT_GE(std::chrono::steady_clock::now() - gate_start, std::chrono::milliseconds(250))
          << "growth did not wait for the renderer slot's consuming submit";
    }
    const auto token = record_and_submit(backend, *sync, passes[image], image_tokens[image]);
    slot_tokens[slot] = token;
    image_tokens[image] = token;
  }

  std::vector<uint8_t> pixels;
  for (uint32_t i = 0; i < kImages; ++i) {
    ASSERT_TRUE(device_.wait_on_submit(image_tokens[i], std::chrono::seconds(5)).has_value());
    ASSERT_NO_FATAL_FAILURE(read_image(targets[i], pixels));
    // Image i was last written by frame kImages + i; exactly that frame's marker must remain.
    expect_only_frame_marker(pixels, extent, kImages + i, kFrames);
  }

  EXPECT_TRUE(device_.wait_idle().has_value());
  drain_command_lists();
  for (uint32_t i = 0; i < kImages; ++i) {
    EXPECT_TRUE(passes[i].destroy().has_value());
    EXPECT_TRUE(targets[i].destroy().has_value());
  }
  EXPECT_TRUE(backend.destroy().has_value());
}

// Public factory contract: a null stream is rejected up front with kInvalidArgument (the
// descriptor documents the non-null requirement), not deep inside resource creation.
TEST_F(ImGuiRendererTest, FactoryRejectsNullStream) {
  const auto backend = create_renderer(device_, {.stream = nullptr});
  ASSERT_FALSE(backend.has_value());
  EXPECT_EQ(backend.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(backend.error().message.find("stream"), std::string::npos);
  // TearDown proves the device still tears down cleanly after the rejected create.
}

}  // namespace
}  // namespace holoviz::imgui
