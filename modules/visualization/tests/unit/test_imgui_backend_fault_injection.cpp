// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Error-injection regressions for the ImGui backend's pinned-staging growth transaction and
// staging-fence refresh — the cases test_imgui_backend.cpp deliberately keeps out of the
// public-API-only file. This file includes the module-private impl header
// (text/imgui_backend_impl.hpp) to reach the CUDA fault-injection seams:
//  - a failed cuMemHostAlloc during growth must leave the slot coherent and the identical
//    frame retryable (the pre-fix code published the grown capacities first, so the retry
//    flattened draw data through a null staging pointer);
//  - a failed cuMemFreeHost during growth must surface the error but still commit the grown
//    slot, so the retry succeeds through the fast path;
//  - a failed cuEventRecord after enqueued stream work (copies or an empty frame's semaphore
//    signal) must drain the stream instead, leaving the frame retryable;
//  - when the drain fails too, the BACKEND must go terminal (retirement is stream-wide, not
//    per-slot): every entry point rejects new frames, destroy() leaks the CUDA-referenced
//    set with a bounded return even while the stream is still wedged, and the owning device
//    is poisoned so it never destroys the leaked children's native parents;
//  - a healthy destroy() must honor its timeout: pending copies that do not retire in time
//    return kBackpressured with the backend intact and retryable.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoviz/rhi.hpp>
#include <holoviz/text/imgui_ui.hpp>  // ImGuiBackend + the bundled Dear ImGui API

#include "image_compare_test_util.hpp"
// Sanctioned test seam: the quarantine tests observe the terminal device poison through
// device_resources_leaked(), which has no public spelling by design.
#include "rhi/vulkan/vk_swapchain_fault_injection.hpp"
#include "text/imgui_backend_impl.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz::text {
namespace {

namespace rhi = holoviz::rhi;

using test_util::kInkThreshold;
using test_util::try_create_device;

// The failing fakes the tests swap into the impl's seam. The failing "free" frees for real
// before reporting failure, so the test leaks nothing while exercising the error path.
CUresult failing_host_alloc(void** /*pp*/, size_t /*bytes*/, unsigned int /*flags*/) {
  return CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult failing_host_free(void* p) {
  cuMemFreeHost(p);
  return CUDA_ERROR_UNKNOWN;
}

CUresult failing_event_record(CUevent /*event*/, CUstream /*stream*/) {
  return CUDA_ERROR_UNKNOWN;
}

CUresult failing_stream_sync(CUstream /*stream*/) {
  return CUDA_ERROR_UNKNOWN;
}

CUresult failing_event_query(CUevent /*event*/) {
  return CUDA_ERROR_UNKNOWN;
}

// Lets the first HtoD copy through for real (so stream work is genuinely enqueued) and
// fails the second: upload_draw_data() then has an earlier upload error to preserve while
// the fence refresh quarantines.
std::atomic<int> memcpy_call_count{0};
CUresult failing_second_memcpy(CUdeviceptr dst, const void* src, size_t bytes, CUstream stream) {
  if (memcpy_call_count.fetch_add(1) == 0) {
    return cuMemcpyHtoDAsync(dst, src, bytes, stream);
  }
  return CUDA_ERROR_UNKNOWN;
}

bool is_ink(const std::vector<uint8_t>& pixels, rhi::Extent2D extent, uint32_t x, uint32_t y) {
  const size_t i = (static_cast<size_t>(y) * extent.width + x) * 4;
  return pixels[i] > kInkThreshold || pixels[i + 1] > kInkThreshold ||
         pixels[i + 2] > kInkThreshold;
}

/// Draw enough quads to outgrow a 16-vertex/32-index initial capacity, plus a frame-keyed
/// marker column (mirrors test_imgui_backend.cpp's oracle so readbacks identify the frame).
void build_growth_ui(uint64_t frame) {
  ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
  for (int i = 0; i < 8; ++i) {  // 8 quads = 32 vertices / 48 indices
    draw_list->AddRectFilled(ImVec2(32, 32), ImVec2(48, 48), IM_COL32(128, 128, 128, 255));
  }
  const auto x0 = static_cast<float>(8 + 12 * (frame % 10));
  draw_list->AddRectFilled(ImVec2(x0, 8), ImVec2(x0 + 8, 16), IM_COL32(255, 255, 255, 255));
}

/// Small UI that fits the initial capacities (establishes real slot state without growth).
void build_small_ui() {
  ImGui::GetBackgroundDrawList()->AddRectFilled(
      ImVec2(60, 60), ImVec2(70, 70), IM_COL32(255, 255, 255, 255));
}

class ImGuiBackendFaultInjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = try_create_device();
    if (!device_) {
      HOLOVIZ_ENV_SKIP() << "no usable GPU/adapter on this machine";
    }
    const auto context = device_.cuda_context();
    ASSERT_TRUE(context.has_value()) << context.error().message;
    if (cuCtxSetCurrent(*context) != CUDA_SUCCESS ||
        cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
      HOLOVIZ_ENV_SKIP() << "cannot create a CUDA stream on the device context";
    }
  }

  void TearDown() override {
    release_stream_gate();  // never leave the stream wedged, whatever the test's outcome
    if (device_) {
      EXPECT_TRUE(device_.wait_idle().has_value());
      for (auto& command_list : pending_command_lists_) {
        EXPECT_TRUE(command_list.destroy(std::chrono::seconds(5)).has_value());
      }
      pending_command_lists_.clear();
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
      const auto destroyed = device_.destroy();
      // Quarantine tests deliberately leak backend children (buffers, the semaphore), so
      // the device may report them; that is the honest outcome, not a test failure.
      if (!expect_leaked_children_) {
        EXPECT_TRUE(destroyed.has_value());
      }
    }
  }

  /// Host-gate the CUDA stream: everything enqueued after this call stays pending until
  /// release_stream_gate() (same rig as test_imgui_backend.cpp — a host function spinning
  /// on an atomic). Makes "queued CUDA work has genuinely not retired" deterministic.
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

  void create_target(rhi::Extent2D extent) {
    extent_ = extent;
    auto target = device_.create_image({
        .extent = extent,
        .format = rhi::Format::kR8G8B8A8Unorm,
        .usage = rhi::ImageUsage::kRenderTarget,
        .debug_name = "imgui fault-injection test target",
    });
    ASSERT_TRUE(target.has_value()) << target.error().message;
    target_ = std::move(*target);
    auto render_pass =
        device_.create_render_pass({.render_targets = {{.format = rhi::Format::kR8G8B8A8Unorm}}});
    ASSERT_TRUE(render_pass.has_value()) << render_pass.error().message;
    render_pass_ = std::move(*render_pass);
    ASSERT_TRUE(render_pass_.set_render_image(target_, 0, {0.0F, 0.0F, 0.0F, 1.0F}).has_value());
  }

  rhi::SubmitSyncToken record_and_submit(ImGuiBackend& backend, const ImGuiFrameSync& sync,
                                         rhi::SubmitSyncToken wait_previous = {}) {
    auto command_list = device_.create_command_list({.type = rhi::CommandListType::kTransient});
    EXPECT_TRUE(command_list.has_value());
    EXPECT_TRUE(command_list->begin().has_value());
    EXPECT_TRUE(command_list->begin_render(render_pass_, {0, 0, extent_.width, extent_.height})
                    .has_value());
    const auto recorded = backend.record_draws(*command_list, extent_);
    EXPECT_TRUE(recorded.has_value()) << (recorded ? "" : recorded.error().message);
    EXPECT_TRUE(command_list->end_render().has_value());
    EXPECT_TRUE(command_list->end().has_value());
    rhi::SubmitDesc submit_desc{
        .command_lists = {&command_list.value(), 1},
    };
    rhi::ExternalSyncToken upload_token = sync.upload_done;
    submit_desc.wait_external_sync_tokens = {&upload_token, 1};
    if (wait_previous.timeline_value != 0) {
      submit_desc.wait_submit_sync_tokens = {&wait_previous, 1};
    }
    const auto token = device_.submit(submit_desc);
    EXPECT_TRUE(token.has_value()) << (token ? "" : token.error().message);
    pending_command_lists_.push_back(std::move(*command_list));
    return token.value_or(rhi::SubmitSyncToken{});
  }

  void read_target(std::vector<uint8_t>& out_pixels) {
    const size_t row_pitch = static_cast<size_t>(extent_.width) * 4;
    out_pixels.resize(row_pitch * extent_.height);
    ASSERT_TRUE(device_
                    .download_from_image(
                        target_, 0, std::as_writable_bytes(std::span(out_pixels)), row_pitch)
                    .has_value());
  }

  rhi::Device device_;
  CUstream stream_ = nullptr;
  rhi::Image target_;
  rhi::RenderPass render_pass_;
  rhi::Extent2D extent_{};
  std::vector<rhi::CommandList> pending_command_lists_;
  std::atomic<bool> stream_gate_released_{true};
  bool expect_leaked_children_ = false;
};

TEST_F(ImGuiBackendFaultInjectionTest, StagingAllocFailureDuringGrowthLeavesSlotUsable) {
  auto backend_result = create_imgui_backend(device_,
                                             {.stream = stream_,
                                              .frames_in_flight = 2,
                                              .initial_vertex_capacity = 16,
                                              .initial_index_capacity = 32});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  ImGuiBackend backend = std::move(*backend_result);

  const rhi::Extent2D extent{128, 128};
  ASSERT_NO_FATAL_FAILURE(create_target(extent));

  // Frame 0 establishes real slot state (buffers, staging, staging fence) without growth.
  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  const auto small_sync = backend.render_ui(extent, [] { build_small_ui(); });
  ASSERT_TRUE(small_sync.has_value()) << small_sync.error().message;
  const auto small_token = record_and_submit(backend, *small_sync);
  ASSERT_TRUE(device_.wait_on_submit(small_token, std::chrono::seconds(5)).has_value());

  // Frame 2 reuses slot 0 and outgrows it while the pinned staging allocation fails: the
  // growth must report the error and leave the old coherent slot in place.
  backend.impl().host_alloc_fn_ = &failing_host_alloc;
  ASSERT_TRUE(backend.begin_frame(2, small_token).has_value());
  const auto failed = backend.render_ui(extent, [] { build_growth_ui(2); });
  ASSERT_FALSE(failed.has_value()) << "growth must fail when the staging allocation fails";
  EXPECT_EQ(failed.error().code, ErrorCode::kCudaError);
  backend.impl().host_alloc_fn_ = &cuMemHostAlloc;

  // Retrying the identical frame must succeed — with the pre-fix ordering this retry took
  // the capacity-only fast path and memcpy'd through the null staging pointer.
  ASSERT_TRUE(backend.begin_frame(2, small_token).has_value());
  const auto retried = backend.render_ui(extent, [] { build_growth_ui(2); });
  ASSERT_TRUE(retried.has_value()) << retried.error().message;
  const auto token = record_and_submit(backend, *retried, small_token);
  ASSERT_TRUE(device_.wait_on_submit(token, std::chrono::seconds(5)).has_value());

  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(read_target(pixels));
  EXPECT_TRUE(is_ink(pixels, extent, 8 + 12 * 2 + 4, 12)) << "frame 2 marker missing after retry";
  EXPECT_TRUE(is_ink(pixels, extent, 40, 40)) << "grown bulk quads missing after retry";

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(backend.destroy().has_value());
}

TEST_F(ImGuiBackendFaultInjectionTest, StagingFreeFailureDuringGrowthKeepsSlotConsistent) {
  auto backend_result = create_imgui_backend(device_,
                                             {.stream = stream_,
                                              .frames_in_flight = 2,
                                              .initial_vertex_capacity = 16,
                                              .initial_index_capacity = 32});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  ImGuiBackend backend = std::move(*backend_result);

  const rhi::Extent2D extent{128, 128};
  ASSERT_NO_FATAL_FAILURE(create_target(extent));

  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  const auto small_sync = backend.render_ui(extent, [] { build_small_ui(); });
  ASSERT_TRUE(small_sync.has_value()) << small_sync.error().message;
  const auto small_token = record_and_submit(backend, *small_sync);
  ASSERT_TRUE(device_.wait_on_submit(small_token, std::chrono::seconds(5)).has_value());

  // Growth whose retirement of the OLD staging block fails: the grown slot must still commit
  // (the failure is reported, the old block is treated as leaked), so the identical retry
  // succeeds through the fast path.
  backend.impl().host_free_fn_ = &failing_host_free;
  ASSERT_TRUE(backend.begin_frame(2, small_token).has_value());
  const auto failed = backend.render_ui(extent, [] { build_growth_ui(2); });
  ASSERT_FALSE(failed.has_value()) << "growth must surface the staging free failure";
  EXPECT_EQ(failed.error().code, ErrorCode::kCudaError);
  backend.impl().host_free_fn_ = &cuMemFreeHost;

  ASSERT_TRUE(backend.begin_frame(2, small_token).has_value());
  const auto retried = backend.render_ui(extent, [] { build_growth_ui(2); });
  ASSERT_TRUE(retried.has_value()) << retried.error().message;
  const auto token = record_and_submit(backend, *retried, small_token);
  ASSERT_TRUE(device_.wait_on_submit(token, std::chrono::seconds(5)).has_value());

  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(read_target(pixels));
  EXPECT_TRUE(is_ink(pixels, extent, 8 + 12 * 2 + 4, 12)) << "frame 2 marker missing after retry";
  EXPECT_TRUE(is_ink(pixels, extent, 40, 40)) << "grown bulk quads missing after retry";

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(backend.destroy().has_value());
}

// A failed staging-fence record after enqueued stream work must drain the stream instead
// (the fence is the only retirement proof slot reuse, growth, and destroy() trust), report
// the failure, and leave both the empty-frame and the copy path retryable.
TEST_F(ImGuiBackendFaultInjectionTest, FenceRecordFailureDrainsAndStaysRetryable) {
  auto backend_result = create_imgui_backend(device_, {.stream = stream_, .frames_in_flight = 2});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  ImGuiBackend backend = std::move(*backend_result);

  const rhi::Extent2D extent{128, 128};
  ASSERT_NO_FATAL_FAILURE(create_target(extent));

  backend.impl().event_record_fn_ = &failing_event_record;
  // Empty frame: the semaphore signal is the enqueued stream work.
  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  const auto empty_failed = backend.render_ui(extent, [] {});
  ASSERT_FALSE(empty_failed.has_value())
      << "an empty frame must surface the failed fence record too";
  EXPECT_EQ(empty_failed.error().code, ErrorCode::kCudaError);

  // Nonempty frame under the same fault: the HtoD copies are the enqueued stream work.
  ASSERT_TRUE(backend.begin_frame(1, {}).has_value());
  const auto failed = backend.render_ui(extent, [] { build_small_ui(); });
  ASSERT_FALSE(failed.has_value()) << "the copy path must surface the failed fence record";
  EXPECT_EQ(failed.error().code, ErrorCode::kCudaError);
  backend.impl().event_record_fn_ = &cuEventRecord;

  // The drain kept both slots coherent: the identical frame retries and renders.
  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  const auto retried = backend.render_ui(extent, [] { build_small_ui(); });
  ASSERT_TRUE(retried.has_value()) << retried.error().message;
  const auto token = record_and_submit(backend, *retried);
  ASSERT_TRUE(device_.wait_on_submit(token, std::chrono::seconds(5)).has_value());

  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(read_target(pixels));
  EXPECT_TRUE(is_ink(pixels, extent, 65, 65)) << "retried frame missing after fence drain";

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(backend.destroy().has_value());
}

// Finding-1 trigger: an empty frame's semaphore signal is enqueued on a genuinely blocked
// stream, then both the fence record and the drain fail. destroy() must not synchronize
// stale evidence and destroy the imported semaphore under the queued signal — it must
// report the quarantine and leak the dependent set instead, and it must do so with a
// bounded return WHILE the stream is still wedged (the terminal path never syncs or frees,
// so the still-held gate cannot deadlock it).
TEST_F(ImGuiBackendFaultInjectionTest, EmptyFrameDoubleFaultQuarantinesTeardown) {
  expect_leaked_children_ = true;
  auto backend_result = create_imgui_backend(device_, {.stream = stream_, .frames_in_flight = 2});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  ImGuiBackend backend = std::move(*backend_result);

  gate_stream();  // the signal below stays queued, not retired
  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  backend.impl().event_record_fn_ = &failing_event_record;
  backend.impl().stream_sync_fn_ = &failing_stream_sync;
  const auto failed = backend.render_ui({128, 128}, [] {});
  ASSERT_FALSE(failed.has_value()) << "the double fault must surface";
  EXPECT_EQ(failed.error().code, ErrorCode::kCudaError);

  // The unprovable retirement is stream-wide, so the whole backend is terminal — a new
  // frame in the OTHER slot must be rejected up front, not hung on the wedged stream.
  const auto other_slot = backend.begin_frame(1, {});
  ASSERT_FALSE(other_slot.has_value()) << "a terminal backend must reject every slot";
  EXPECT_EQ(other_slot.error().code, ErrorCode::kCudaError);

  // The leaked children still reference the device; its teardown must know to leak its
  // native state instead of destroying the parents under the queued signal.
  EXPECT_TRUE(rhi::vulkan::device_resources_leaked(device_));

  // Teardown of the terminal backend under the STILL-HELD gate: an error report and a
  // deliberate leak of the slots' resources and the semaphore import — no sync, no free,
  // no use-after-destroy. (The pre-fix code returned from the record failure with the
  // fence at an older stream position, so destroy() "proved" retirement against stale
  // evidence and destroyed the semaphore import under the queued signal.)
  const auto destroyed = backend.destroy();
  ASSERT_FALSE(destroyed.has_value()) << "quarantined teardown must report the leak";
  EXPECT_EQ(destroyed.error().code, ErrorCode::kCudaError);
}

// Finding-2 trigger: HtoD copies are enqueued on a blocked stream, then fence record and
// drain both fail. The backend must go terminal — retries are rejected in begin_frame()
// before they can flatten into the slot or swap in replacements and destroy the in-flight
// buffers — and destroy() leaks the complete set with a bounded return while still gated.
TEST_F(ImGuiBackendFaultInjectionTest, UploadDoubleFaultQuarantinesSlotAndRejectsRetry) {
  expect_leaked_children_ = true;
  auto backend_result = create_imgui_backend(device_,
                                             {.stream = stream_,
                                              .frames_in_flight = 2,
                                              .initial_vertex_capacity = 16,
                                              .initial_index_capacity = 32});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  ImGuiBackend backend = std::move(*backend_result);

  const rhi::Extent2D extent{128, 128};
  ASSERT_NO_FATAL_FAILURE(create_target(extent));

  // Frame 0 establishes real slot state (buffers, staging, a recorded fence) and retires.
  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  const auto small_sync = backend.render_ui(extent, [] { build_small_ui(); });
  ASSERT_TRUE(small_sync.has_value()) << small_sync.error().message;
  const auto small_token = record_and_submit(backend, *small_sync);
  ASSERT_TRUE(device_.wait_on_submit(small_token, std::chrono::seconds(5)).has_value());

  gate_stream();  // this frame's copies stay queued, not retired
  ASSERT_TRUE(backend.begin_frame(2, small_token).has_value());
  backend.impl().event_record_fn_ = &failing_event_record;
  backend.impl().stream_sync_fn_ = &failing_stream_sync;
  const auto failed = backend.render_ui(extent, [] { build_small_ui(); });
  ASSERT_FALSE(failed.has_value()) << "the double fault must surface";
  EXPECT_EQ(failed.error().code, ErrorCode::kCudaError);
  backend.impl().event_record_fn_ = &cuEventRecord;
  backend.impl().stream_sync_fn_ = &cuStreamSynchronize;

  // Retries are rejected in begin_frame(), before anything could flatten into the slot,
  // re-upload, or reach the growth rebuild that would destroy the old buffers while the
  // queued DMA still targets them (the pre-fix poison zeroed the capacities, sending
  // exactly the growing retry into that rebuild). The rejection must also cover the OTHER
  // slot: retirement is stream-wide, and that slot's fence sync would hang on the gate.
  const auto retried_same = backend.begin_frame(2, small_token);
  ASSERT_FALSE(retried_same.has_value()) << "a terminal backend must reject reuse";
  EXPECT_EQ(retried_same.error().code, ErrorCode::kCudaError);
  const auto retried_other = backend.begin_frame(3, small_token);
  ASSERT_FALSE(retried_other.has_value()) << "a terminal backend must reject every slot";
  EXPECT_EQ(retried_other.error().code, ErrorCode::kCudaError);

  EXPECT_TRUE(rhi::vulkan::device_resources_leaked(device_));

  // Teardown of the terminal backend under the STILL-HELD gate (the terminal path never
  // syncs or frees, so the gate cannot deadlock it): error + deliberate leak of the set.
  const auto destroyed = backend.destroy();
  ASSERT_FALSE(destroyed.has_value()) << "quarantined teardown must report the leak";
  EXPECT_EQ(destroyed.error().code, ErrorCode::kCudaError);
}

// A HEALTHY backend's destroy() must honor its timeout instead of synchronizing unboundedly
// behind a wedged stream: pending copies that do not retire in time return kBackpressured
// with the backend fully intact, and the identical call succeeds once the stream drains.
TEST_F(ImGuiBackendFaultInjectionTest, GatedDestroyBackpressuresAndStaysRetryable) {
  auto backend_result = create_imgui_backend(device_, {.stream = stream_, .frames_in_flight = 2});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  ImGuiBackend backend = std::move(*backend_result);

  const rhi::Extent2D extent{128, 128};
  gate_stream();  // this frame's copies stay queued, not retired
  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  const auto sync = backend.render_ui(extent, [] { build_small_ui(); });
  ASSERT_TRUE(sync.has_value()) << sync.error().message;

  // Bounded: the pre-fix destroy() blocked in cuEventSynchronize until the gate released.
  const auto backpressured = backend.destroy(std::chrono::milliseconds(50));
  ASSERT_FALSE(backpressured.has_value()) << "gated destroy must not claim success";
  EXPECT_EQ(backpressured.error().code, ErrorCode::kBackpressured);
  EXPECT_TRUE(static_cast<bool>(backend)) << "a backpressured destroy must keep the handle";

  // Nothing went terminal: the device is unpoisoned, and after the stream drains the same
  // backend destroys cleanly.
  EXPECT_FALSE(rhi::vulkan::device_resources_leaked(device_));
  release_stream_gate();
  ASSERT_EQ(cuStreamSynchronize(stream_), CUDA_SUCCESS);
  EXPECT_TRUE(backend.destroy().has_value());
}

// Teardown seam the double-fault tests cannot reach: the staging fence QUERY itself fails
// during a healthy destroy(). Retirement is then unprovable, so destroy() must quarantine —
// leak the CUDA-referenced set, poison the device — instead of freeing under possibly-live
// copies (the pre-fix code logged the failed sync as a warning and freed anyway).
TEST_F(ImGuiBackendFaultInjectionTest, TeardownFenceQueryFailureQuarantinesAndLeaks) {
  expect_leaked_children_ = true;
  auto backend_result = create_imgui_backend(device_, {.stream = stream_, .frames_in_flight = 2});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  ImGuiBackend backend = std::move(*backend_result);

  const rhi::Extent2D extent{128, 128};
  ASSERT_NO_FATAL_FAILURE(create_target(extent));
  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  const auto sync = backend.render_ui(extent, [] { build_small_ui(); });
  ASSERT_TRUE(sync.has_value()) << sync.error().message;
  const auto token = record_and_submit(backend, *sync);
  ASSERT_TRUE(device_.wait_on_submit(token, std::chrono::seconds(5)).has_value());

  backend.impl().event_query_fn_ = &failing_event_query;
  const auto destroyed = backend.destroy(std::chrono::seconds(1));
  ASSERT_FALSE(destroyed.has_value()) << "an unprovable teardown retirement must be reported";
  EXPECT_EQ(destroyed.error().code, ErrorCode::kCudaError);
  EXPECT_FALSE(static_cast<bool>(backend)) << "a terminal destroy still empties the handle";
  EXPECT_TRUE(rhi::vulkan::device_resources_leaked(device_));
}

// When the copy/signal already failed and the fence refresh then quarantines, the terminal
// transition must dominate the returned error while preserving the earlier upload failure
// as the root cause (the pre-fix code returned the earlier error alone, so the caller never
// learned the backend went terminal).
TEST_F(ImGuiBackendFaultInjectionTest, QuarantineReportPreservesEarlierUploadCause) {
  expect_leaked_children_ = true;
  auto backend_result = create_imgui_backend(device_, {.stream = stream_, .frames_in_flight = 2});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  ImGuiBackend backend = std::move(*backend_result);

  gate_stream();  // the first (real) copy stays queued, not retired
  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  memcpy_call_count.store(0);
  backend.impl().host_memcpy_fn_ = &failing_second_memcpy;
  backend.impl().event_record_fn_ = &failing_event_record;
  backend.impl().stream_sync_fn_ = &failing_stream_sync;
  const auto failed = backend.render_ui({128, 128}, [] { build_small_ui(); });
  ASSERT_FALSE(failed.has_value()) << "the triple fault must surface";
  EXPECT_EQ(failed.error().code, ErrorCode::kCudaError);
  EXPECT_NE(failed.error().message.find("terminal"), std::string::npos)
      << "the terminal transition must dominate the report: " << failed.error().message;
  EXPECT_NE(failed.error().message.find("earlier upload failure"), std::string::npos)
      << "the earlier copy failure must be preserved as the cause: " << failed.error().message;

  EXPECT_TRUE(rhi::vulkan::device_resources_leaked(device_));
  const auto destroyed = backend.destroy();
  ASSERT_FALSE(destroyed.has_value()) << "quarantined teardown must report the leak";
  EXPECT_EQ(destroyed.error().code, ErrorCode::kCudaError);
}

}  // namespace
}  // namespace holoviz::text
