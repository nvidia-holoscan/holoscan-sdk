// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CudaPresentBackend tests: argument validation without a ring, then the full seam protocol
// against HolovizRenderBackend with pixel readback — a clear-only frame rendered into the
// backend's exported ring must come out of caller device memory as the clear color, including
// when the ring wraps and a render must wait for the previous resolve of the same image, when
// the destination's writer stream still has work queued ahead of the resolve, and when a later
// frame into the same image is still pending while an earlier id is queried; a BGRA output format
// comes out channel-swapped from its RGBA render target; and teardown proves the whole backend
// stream idle, including a render-completion wait a failed present left queued. GPU tests skip on
// machines with no adapter.

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>
#include "holoviz_op/backends/cuda_present_backend.hpp"
#include "holoviz_op/backends/holoviz_render_backend.hpp"
#include "holoviz_op/backends/present_backend.hpp"
#include "holoviz_op/backends/render_backend.hpp"

#include "cuda/cu_scoped_context.hpp"
#include "holoviz_op/backends/backend_tokens.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoscan::visualization::backends {
namespace {

constexpr Extent2D kExtent{64, 48};
constexpr uint32_t kBytesPerPixel = 4;
// RenderConfig's default clear color is opaque black; use something recognizable instead.
constexpr std::array<float, 4> kClearColor{1.0F, 0.5F, 0.0F, 1.0F};
constexpr std::array<uint8_t, 4> kClearBytesRgba{255, 128, 0, 255};

holoviz::Device try_create_device() {
  auto adapters = holoviz::enumerate_adapters();
  if (!adapters || adapters->empty()) {
    return {};
  }
  auto device = adapters->front().create_device({});
  if (!device) {
    return {};
  }
  return std::move(*device);
}

/// Device memory sized for one frame; freed on destruction.
struct DeviceFrame {
  explicit DeviceFrame(CUcontext context) : context_(context) {
    const holoviz::ScopedCudaContext scoped(context_);
    if (cuMemAlloc(&ptr, size_bytes()) != CUDA_SUCCESS) {
      ptr = 0;
    }
  }
  ~DeviceFrame() {
    if (ptr != 0) {
      const holoviz::ScopedCudaContext scoped(context_);
      cuMemFree(ptr);
    }
  }
  DeviceFrame(const DeviceFrame&) = delete;
  DeviceFrame& operator=(const DeviceFrame&) = delete;

  static constexpr size_t pitch() { return static_cast<size_t>(kExtent.width) * kBytesPerPixel; }
  static constexpr size_t size_bytes() { return pitch() * kExtent.height; }

  CUcontext context_;
  CUdeviceptr ptr = 0;
};

/// A host-side gate enqueued into a CUDA stream: everything queued behind block() on that stream
/// waits until release() — a deterministic stand-in for "GPU work still pending on the stream".
///
/// The destructor releases the gate and then joins the enqueued host callback, so an early test
/// exit cannot destroy the mutex and condition variable while the driver's callback thread is
/// still waiting on them. A gate that never armed block() destructs without waiting.
class HostGate {
 public:
  ~HostGate() {
    release();
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock, [this] { return !armed_ || callback_finished_; });
  }

  [[nodiscard]] CUresult block(CUstream stream) {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      armed_ = true;
    }
    const CUresult result = cuLaunchHostFunc(stream, &HostGate::wait_open, this);
    if (result != CUDA_SUCCESS) {
      // Nothing was enqueued, so there is no callback for the destructor to join.
      const std::lock_guard<std::mutex> lock(mutex_);
      armed_ = false;
    }
    return result;
  }

  void release() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      open_ = true;
    }
    changed_.notify_all();
  }

 private:
  static void CUDA_CB wait_open(void* user) {
    auto* gate = static_cast<HostGate*>(user);
    std::unique_lock<std::mutex> lock(gate->mutex_);
    gate->changed_.wait(lock, [gate] { return gate->open_; });
    gate->callback_finished_ = true;
    // Notify while still holding the lock: the destructor cannot pass its wait until this unlock
    // completes, after which the callback no longer touches any member.
    gate->changed_.notify_all();
  }

  std::mutex mutex_;
  std::condition_variable changed_;
  bool open_ = false;
  bool armed_ = false;
  bool callback_finished_ = false;
};

class CudaPresentBackendTest : public ::testing::Test {
 protected:
  struct Frame {
    AcquireToken acquired;
    SyncToken render_done;
  };

  void SetUp() override {
    device_ = try_create_device();
    if (!device_) {
      HOLOVIZ_ENV_SKIP() << "no usable rendering device/adapter.";
    }
    auto backend = create_cuda_present_backend(device_, {.format = ImageFormat::kR8G8B8A8Unorm});
    ASSERT_TRUE(backend.has_value()) << backend.error().message;
    present_ = std::move(*backend);
    auto render = create_holoviz_render_backend(device_, {});
    ASSERT_TRUE(render.has_value()) << render.error().message;
    render_ = std::move(*render);
    auto context = device_.cuda_context();
    ASSERT_TRUE(context.has_value());
    cuda_context_ = *context;
    // Stands in for the operator's activation stream: the stream a destination's write is
    // opened on, which set_resolve_target() orders the resolve after.
    const holoviz::ScopedCudaContext scoped(cuda_context_);
    ASSERT_EQ(cuStreamCreate(&writer_stream_, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  }

  // Keep the fixture retirement and cleanup sequence together so failed assertions retain their
  // context.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void TearDown() override {
    if (render_ != nullptr) {
      EXPECT_TRUE(render_->destroy(std::chrono::seconds(5)).has_value());
      render_.reset();
    }
    if (present_ != nullptr) {
      EXPECT_TRUE(present_->destroy(std::chrono::seconds(5)).has_value());
      present_.reset();
    }
    if (writer_stream_ != nullptr) {
      const holoviz::ScopedCudaContext scoped(cuda_context_);
      EXPECT_EQ(cuStreamSynchronize(writer_stream_), CUDA_SUCCESS);
      EXPECT_EQ(cuStreamDestroy(writer_stream_), CUDA_SUCCESS);
      writer_stream_ = nullptr;
    }
  }

  /// Register \p destination for \p frame's image with the fixture's writer stream and present
  /// the frame under \p id.
  void present_into(const Frame& frame, const DeviceFrame& destination, PresentId id) {
    ASSERT_TRUE(present_
                    ->set_resolve_target(
                        frame.acquired.image_index,
                        {.device_ptr = destination.ptr, .row_pitch_bytes = DeviceFrame::pitch()},
                        writer_stream_)
                    .has_value());
    ASSERT_TRUE(present_
                    ->present(PresentArgs{.image_index = frame.acquired.image_index,
                                          .acquire_issuance = frame.acquired.issuance,
                                          .render_done = frame.render_done,
                                          .present_id = id})
                    .has_value());
  }

  /// create_swapchain + allocate/build/register_targets for a clear-only scene.
  // Keep this fixture setup lifecycle together so GPU prerequisites and skips remain visible.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void build_frame_loop(uint32_t image_count = 2) {
    auto created = present_->create_swapchain(SwapchainDesc{
        .surface = {},
        .preferred_extent = kExtent,
        .preferred_mode = holoviz::PresentMode::kAuto,
        .image_count = image_count,
    });
    ASSERT_TRUE(created.has_value()) << created.error().message;
    EXPECT_EQ(created->extent.width, kExtent.width);
    EXPECT_EQ(created->extent.height, kExtent.height);
    EXPECT_EQ(created->image_count, image_count);

    RenderConfig config{.max_frames_in_flight = image_count};
    for (size_t i = 0; i < kClearColor.size(); ++i) {
      config.clear_color[i] = kClearColor[i];
    }
    ASSERT_TRUE(render_->allocate(LayerStack{}, config).has_value());
    ASSERT_TRUE(render_->build(present_->swapchain_info()->format).has_value());
    auto targets = present_->targets();
    ASSERT_TRUE(targets.has_value()) << targets.error().message;
    ASSERT_TRUE(render_->register_targets(*targets).has_value());
  }

  /// acquire → begin_frame → record(empty) → submit into \p frame (ASSERTs need a void body).
  void render_clear_frame(Frame& frame) {
    auto acquired = present_->acquire(std::chrono::seconds{1});
    ASSERT_TRUE(acquired.has_value()) << acquired.error().message;
    frame.acquired = std::move(*acquired);
    auto cmds = render_->begin_frame(frame.acquired, frame.acquired.image_index);
    ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
    ASSERT_TRUE(render_->record(*cmds, RenderList{}, CompositeSpec{}).has_value());
    auto done = render_->submit(*cmds, SubmitContext{frame.acquired.image_index});
    ASSERT_TRUE(done.has_value()) << done.error().message;
    frame.render_done = std::move(*done);
  }

  std::vector<uint8_t> read_back_device(CUdeviceptr ptr) {
    std::vector<uint8_t> bytes(DeviceFrame::size_bytes());
    const holoviz::ScopedCudaContext scoped(cuda_context_);
    EXPECT_EQ(cuMemcpyDtoH(bytes.data(), ptr, bytes.size()), CUDA_SUCCESS);
    return bytes;
  }

  static void expect_clear_color(const std::vector<uint8_t>& bytes) {
    ASSERT_EQ(bytes.size(), DeviceFrame::size_bytes());
    // Check the four corners and the center rather than every pixel: enough to catch a
    // missing, partial, or misordered resolve while keeping the failure output readable.
    const std::array<size_t, 5> pixels{
        0,
        static_cast<size_t>(kExtent.width) - 1,
        static_cast<size_t>(kExtent.width) * (kExtent.height - 1),
        static_cast<size_t>(kExtent.width) * kExtent.height - 1,
        static_cast<size_t>(kExtent.width) * (kExtent.height / 2) + kExtent.width / 2,
    };
    for (const size_t pixel : pixels) {
      for (size_t c = 0; c < 4; ++c) {
        EXPECT_NEAR(bytes[pixel * kBytesPerPixel + c], kClearBytesRgba[c], 1) << "pixel " << pixel;
      }
    }
  }

  /// Zero the destination so a skipped resolve cannot pass on stale bytes.
  void clear_destination(const DeviceFrame& destination) {
    const holoviz::ScopedCudaContext scoped(cuda_context_);
    ASSERT_EQ(cuMemsetD8(destination.ptr, 0, DeviceFrame::size_bytes()), CUDA_SUCCESS);
  }

  /// Replace the fixture's (still ringless) present backend with one of another output format.
  void recreate_present(ImageFormat format) {
    ASSERT_TRUE(present_->destroy(std::chrono::seconds{0}).has_value());
    auto backend = create_cuda_present_backend(device_, {.format = format});
    ASSERT_TRUE(backend.has_value()) << backend.error().message;
    present_ = std::move(*backend);
  }

  holoviz::Device device_;
  CUcontext cuda_context_ = nullptr;
  CUstream writer_stream_ = nullptr;
  std::unique_ptr<CudaPresentBackend> present_;
  std::unique_ptr<RenderBackend> render_;
};

TEST_F(CudaPresentBackendTest, RejectsBadCreationArguments) {
  holoviz::Device empty;
  EXPECT_EQ(create_cuda_present_backend(empty, {}).error().code, ErrorCode::kInvalidArgument);

  // Everything before create_swapchain() is kInvalidState.
  EXPECT_EQ(present_->acquire(std::chrono::seconds{0}).error().code, ErrorCode::kInvalidState);
  EXPECT_EQ(present_->swapchain_info().error().code, ErrorCode::kInvalidState);
  EXPECT_EQ(present_->targets().error().code, ErrorCode::kInvalidState);
  EXPECT_EQ(present_->stream().error().code, ErrorCode::kInvalidState);

  // A capture sink has no surface to size from.
  EXPECT_EQ(present_->create_swapchain(SwapchainDesc{.preferred_extent = {0, 0}}).error().code,
            ErrorCode::kInvalidArgument);

  // A multi-plane format has no single row pitch; RGB8 has no render-target lowering.
  for (const ImageFormat format : {ImageFormat::kNV12, ImageFormat::kR8G8B8Unorm}) {
    auto rejected = create_cuda_present_backend(device_, {.format = format});
    ASSERT_TRUE(rejected.has_value());
    EXPECT_EQ(
        (*rejected)->create_swapchain(SwapchainDesc{.preferred_extent = kExtent}).error().code,
        ErrorCode::kInvalidArgument);
  }
}

TEST_F(CudaPresentBackendTest, ReportsCaptureSinkIdentity) {
  const PresentInfo info = present_->info();
  EXPECT_EQ(info.name, "cuda_sink");
  EXPECT_TRUE(info.has_present_id);
  EXPECT_TRUE(info.has_present_wait);

  build_frame_loop();
  const auto info_after = present_->swapchain_info();
  ASSERT_TRUE(info_after.has_value());
  EXPECT_EQ(info_after->format, ImageFormat::kR8G8B8A8Unorm);
  EXPECT_NE(info_after->active_mode, holoviz::PresentMode::kAuto);
  EXPECT_TRUE(present_->stream().has_value());
  EXPECT_EQ(present_->create_swapchain(SwapchainDesc{.preferred_extent = kExtent}).error().code,
            ErrorCode::kInvalidState);
}

TEST_F(CudaPresentBackendTest, ClearFrameReachesTheDestination) {
  build_frame_loop();
  const DeviceFrame destination(cuda_context_);
  ASSERT_NE(destination.ptr, 0U);
  clear_destination(destination);

  Frame frame;
  render_clear_frame(frame);
  if (HasFatalFailure()) {
    return;
  }
  // Presenting without a destination leaves the frame pending and retryable.
  EXPECT_EQ(present_
                ->present(PresentArgs{.image_index = frame.acquired.image_index,
                                      .acquire_issuance = frame.acquired.issuance,
                                      .render_done = frame.render_done,
                                      .present_id = 1})
                .error()
                .code,
            ErrorCode::kInvalidArgument);
  // A null or too-narrow destination is rejected, a null writer stream and an out-of-range
  // image as well.
  EXPECT_EQ(
      present_->set_resolve_target(frame.acquired.image_index, {}, writer_stream_).error().code,
      ErrorCode::kInvalidArgument);
  EXPECT_EQ(present_
                ->set_resolve_target(frame.acquired.image_index,
                                     {.device_ptr = destination.ptr, .row_pitch_bytes = 1},
                                     writer_stream_)
                .error()
                .code,
            ErrorCode::kInvalidArgument);
  EXPECT_EQ(present_
                ->set_resolve_target(
                    frame.acquired.image_index,
                    {.device_ptr = destination.ptr, .row_pitch_bytes = DeviceFrame::pitch()},
                    nullptr)
                .error()
                .code,
            ErrorCode::kInvalidArgument);
  EXPECT_EQ(present_->set_resolve_target(99, {}, writer_stream_).error().code,
            ErrorCode::kInvalidArgument);
  ASSERT_TRUE(present_
                  ->set_resolve_target(
                      frame.acquired.image_index,
                      {.device_ptr = destination.ptr, .row_pitch_bytes = DeviceFrame::pitch()},
                      writer_stream_)
                  .has_value());
  // Nothing presented yet on this image.
  EXPECT_EQ(present_->ready_event(frame.acquired.image_index).error().code, ErrorCode::kNotFound);

  ASSERT_TRUE(present_
                  ->present(PresentArgs{.image_index = frame.acquired.image_index,
                                        .acquire_issuance = frame.acquired.issuance,
                                        .render_done = frame.render_done,
                                        .present_id = 1})
                  .has_value());
  auto waited = present_->wait_for_present(1, std::chrono::seconds{5});
  ASSERT_TRUE(waited.has_value()) << waited.error().message;
  EXPECT_EQ(*waited, WaitResult::kDisplayed);

  auto ready = present_->ready_event(frame.acquired.image_index);
  ASSERT_TRUE(ready.has_value()) << ready.error().message;
  {
    const holoviz::ScopedCudaContext scoped(cuda_context_);
    EXPECT_EQ(cuEventSynchronize(*ready), CUDA_SUCCESS);
  }
  expect_clear_color(read_back_device(destination.ptr));

  // The destination was consumed by that resolve: the image must not be presented into it
  // again without a new registration. Its ring slot is free, so it can be re-acquired.
  EXPECT_EQ(
      present_->set_resolve_target(frame.acquired.image_index, {}, writer_stream_).error().code,
      ErrorCode::kInvalidState);
}

TEST_F(CudaPresentBackendTest, RingWrapsAndBackpressuresWhenExhausted) {
  build_frame_loop(/*image_count=*/2);
  const DeviceFrame destination(cuda_context_);
  ASSERT_NE(destination.ptr, 0U);

  // Two acquires exhaust the ring; a third is backpressured without consuming anything.
  auto first = present_->acquire(std::chrono::seconds{1});
  ASSERT_TRUE(first.has_value());
  auto second = present_->acquire(std::chrono::seconds{1});
  ASSERT_TRUE(second.has_value());
  EXPECT_NE(first->image_index, second->image_index);
  EXPECT_EQ(present_->acquire(std::chrono::seconds{1}).error().code, ErrorCode::kBackpressured);

  // Empty presents release both; a tagged empty present is a known drop.
  ASSERT_TRUE(present_
                  ->present(PresentArgs{.image_index = first->image_index,
                                        .acquire_issuance = first->issuance,
                                        .present_id = 1})
                  .has_value());
  EXPECT_EQ(present_->wait_for_present(1, std::chrono::seconds{0}).value(), WaitResult::kDropped);
  ASSERT_TRUE(present_
                  ->present(PresentArgs{.image_index = second->image_index,
                                        .acquire_issuance = second->issuance})
                  .has_value());
  // Presenting a released acquire again is a protocol violation, and unknown ids are kNotFound.
  EXPECT_EQ(present_
                ->present(PresentArgs{.image_index = second->image_index,
                                      .acquire_issuance = second->issuance})
                .error()
                .code,
            ErrorCode::kInvalidState);
  EXPECT_EQ(present_->wait_for_present(7, std::chrono::seconds{0}).error().code,
            ErrorCode::kNotFound);

  // Three real frames wrap the two-image ring; each one resolves to the clear color, and the
  // render into a reused image is ordered after its previous resolve (render_wait).
  for (PresentId id = 2; id <= 4; ++id) {
    clear_destination(destination);
    Frame frame;
    render_clear_frame(frame);
    if (HasFatalFailure()) {
      return;
    }
    present_into(frame, destination, id);
    if (HasFatalFailure()) {
      return;
    }
    EXPECT_EQ(present_->wait_for_present(id, std::chrono::seconds{5}).value(),
              WaitResult::kDisplayed);
    expect_clear_color(read_back_device(destination.ptr));
  }
}

// acquire() takes the first free image from its cursor, so an image held across presents of the
// others does not exhaust the ring: hold A, acquire and release B, and the next acquire is B again.
TEST_F(CudaPresentBackendTest, AcquireSkipsHeldImages) {
  build_frame_loop(/*image_count=*/2);
  auto held = present_->acquire(std::chrono::seconds{1});
  ASSERT_TRUE(held.has_value());
  auto released = present_->acquire(std::chrono::seconds{1});
  ASSERT_TRUE(released.has_value());
  ASSERT_TRUE(present_
                  ->present(PresentArgs{.image_index = released->image_index,
                                        .acquire_issuance = released->issuance})
                  .has_value());
  auto again = present_->acquire(std::chrono::seconds{1});
  ASSERT_TRUE(again.has_value()) << again.error().message;
  EXPECT_EQ(again->image_index, released->image_index);
  EXPECT_EQ(present_->acquire(std::chrono::seconds{1}).error().code, ErrorCode::kBackpressured);
  for (const auto* token : {&*held, &*again}) {
    ASSERT_TRUE(present_
                    ->present(PresentArgs{.image_index = token->image_index,
                                          .acquire_issuance = token->issuance})
                    .has_value());
  }
}

// The resolve is ordered after everything already queued on the destination's writer stream:
// a write to the destination queued there behind a host gate must land BEFORE the resolve, so the
// destination ends up holding the frame, not the stale write.
TEST_F(CudaPresentBackendTest, ResolveWaitsForTheWriterStream) {
  build_frame_loop();
  const DeviceFrame destination(cuda_context_);
  ASSERT_NE(destination.ptr, 0U);
  clear_destination(destination);

  Frame frame;
  render_clear_frame(frame);
  if (HasFatalFailure()) {
    return;
  }
  HostGate gate;
  {
    const holoviz::ScopedCudaContext scoped(cuda_context_);
    ASSERT_EQ(gate.block(writer_stream_), CUDA_SUCCESS);
    // Stands in for a consumer read of a recycled allocation that is still queued on the
    // writer stream when the pool hands the allocation out again.
    ASSERT_EQ(cuMemsetD8Async(destination.ptr, 0x11, DeviceFrame::size_bytes(), writer_stream_),
              CUDA_SUCCESS);
  }
  present_into(frame, destination, 1);
  if (HasFatalFailure()) {
    gate.release();
    return;
  }
  // The resolve is stuck behind the gate: not complete, and not timing out the backend.
  EXPECT_EQ(present_->wait_for_present(1, std::chrono::milliseconds{20}).value(),
            WaitResult::kTimeout);
  gate.release();
  EXPECT_EQ(present_->wait_for_present(1, std::chrono::seconds{5}).value(), WaitResult::kDisplayed);
  expect_clear_color(read_back_device(destination.ptr));
}

// A completed present id keeps answering kDisplayed while a later frame into the same image is
// still resolving: completion is tracked per id, not per reusable image slot.
TEST_F(CudaPresentBackendTest, CompletedPresentStaysDisplayedWhileALaterFrameIsPending) {
  build_frame_loop(/*image_count=*/2);
  const DeviceFrame destination(cuda_context_);
  ASSERT_NE(destination.ptr, 0U);

  // Ids 1 and 2 fill both images; id 1 completes.
  Frame first;
  render_clear_frame(first);
  if (HasFatalFailure()) {
    return;
  }
  present_into(first, destination, 1);
  Frame second;
  render_clear_frame(second);
  if (HasFatalFailure()) {
    return;
  }
  present_into(second, destination, 2);
  if (HasFatalFailure()) {
    return;
  }
  EXPECT_EQ(present_->wait_for_present(1, std::chrono::seconds{5}).value(), WaitResult::kDisplayed);
  EXPECT_EQ(present_->wait_for_present(2, std::chrono::seconds{5}).value(), WaitResult::kDisplayed);

  // Id 3 reuses id 1's image; its resolve is held behind a gate on the writer stream.
  Frame third;
  render_clear_frame(third);
  if (HasFatalFailure()) {
    return;
  }
  EXPECT_EQ(third.acquired.image_index, first.acquired.image_index);
  HostGate gate;
  {
    const holoviz::ScopedCudaContext scoped(cuda_context_);
    ASSERT_EQ(gate.block(writer_stream_), CUDA_SUCCESS);
  }
  present_into(third, destination, 3);
  if (HasFatalFailure()) {
    gate.release();
    return;
  }
  EXPECT_EQ(present_->wait_for_present(3, std::chrono::seconds{0}).value(), WaitResult::kTimeout);
  EXPECT_EQ(present_->wait_for_present(1, std::chrono::seconds{0}).value(), WaitResult::kDisplayed);
  EXPECT_EQ(present_->wait_for_present(2, std::chrono::seconds{0}).value(), WaitResult::kDisplayed);
  gate.release();
  EXPECT_EQ(present_->wait_for_present(3, std::chrono::seconds{5}).value(), WaitResult::kDisplayed);
  EXPECT_EQ(present_->wait_for_present(1, std::chrono::seconds{0}).value(), WaitResult::kDisplayed);
}

// destroy() must prove the render side too: a submitted, unpresented render still references its
// ring image. Gating the backend stream holds the acquire signal — and so the render — on the GPU;
// a zero-timeout destroy() must refuse with everything intact, and succeed once the gate opens.
TEST_F(CudaPresentBackendTest, DestroyRefusesWhileARenderIsPending) {
  build_frame_loop();
  auto stream = present_->stream();
  ASSERT_TRUE(stream.has_value());
  HostGate gate;
  {
    const holoviz::ScopedCudaContext scoped(cuda_context_);
    ASSERT_EQ(gate.block(*stream), CUDA_SUCCESS);
  }
  Frame frame;
  render_clear_frame(frame);
  if (HasFatalFailure()) {
    gate.release();
    return;
  }
  const auto refused = present_->destroy(std::chrono::seconds{0});
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().code, ErrorCode::kBackpressured);
  // Nothing was released: the ring, its targets, and the pending acquire are all still live.
  EXPECT_TRUE(present_->swapchain_info().has_value());
  EXPECT_TRUE(present_->targets().has_value());
  EXPECT_TRUE(present_->stream().has_value());
  gate.release();
  ASSERT_TRUE(present_
                  ->present(PresentArgs{.image_index = frame.acquired.image_index,
                                        .acquire_issuance = frame.acquired.issuance})
                  .has_value());
  // The render backend still holds the frame slot's submit; quiesce it first, as HolovizOp does.
  ASSERT_TRUE(render_->destroy(std::chrono::seconds{5}).has_value());
  render_.reset();
  EXPECT_TRUE(present_->destroy(std::chrono::seconds{5}).has_value());
}

// A present() that fails after its copy was enqueued must tell the caller the destination is
// still in use, refuse to prove completion until an event covers the copy, and recover once a
// later present of the same image records one behind it.
TEST_F(CudaPresentBackendTest, PostEnqueueRecordFailureKeepsTheResolveOwned) {
  build_frame_loop(/*image_count=*/1);
  const DeviceFrame destination(cuda_context_);
  ASSERT_NE(destination.ptr, 0U);

  // Case 1: the slot ready-event record fails. Completion is unprovable.
  Frame first;
  render_clear_frame(first);
  if (HasFatalFailure()) {
    return;
  }
  ASSERT_TRUE(present_
                  ->set_resolve_target(
                      first.acquired.image_index,
                      {.device_ptr = destination.ptr, .row_pitch_bytes = DeviceFrame::pitch()},
                      writer_stream_)
                  .has_value());
  present_->set_fault_injection_for_tests({.fail_ready_record = true});
  EXPECT_FALSE(present_
                   ->present(PresentArgs{.image_index = first.acquired.image_index,
                                         .acquire_issuance = first.acquired.issuance,
                                         .render_done = first.render_done,
                                         .present_id = 1})
                   .has_value());
  EXPECT_TRUE(present_->resolve_may_be_in_flight(first.acquired.image_index));
  EXPECT_EQ(present_->ready_event(first.acquired.image_index).error().code, ErrorCode::kFailure);
  EXPECT_EQ(present_->destroy(std::chrono::seconds{0}).error().code, ErrorCode::kFailure);
  EXPECT_TRUE(present_->swapchain_info().has_value());  // nothing released
  // The frame is still pending (its target was consumed): release the image. Drop the token
  // too — a one-image ring has three acquire payloads and this test cycles four frames.
  ASSERT_TRUE(present_
                  ->present(PresentArgs{.image_index = first.acquired.image_index,
                                        .acquire_issuance = first.acquired.issuance})
                  .has_value());
  const uint32_t image_index = first.acquired.image_index;
  first = Frame{};

  // A later present of the same image records its ready event behind the orphaned copy.
  clear_destination(destination);
  Frame second;
  render_clear_frame(second);
  if (HasFatalFailure()) {
    return;
  }
  ASSERT_EQ(second.acquired.image_index, image_index);
  present_into(second, destination, 2);
  if (HasFatalFailure()) {
    return;
  }
  EXPECT_TRUE(present_->ready_event(image_index).has_value());
  EXPECT_EQ(present_->wait_for_present(2, std::chrono::seconds{5}).value(), WaitResult::kDisplayed);
  expect_clear_color(read_back_device(destination.ptr));
  second = Frame{};

  // Case 2: only the per-id record fails; the ready event already covers the copy.
  Frame third;
  render_clear_frame(third);
  if (HasFatalFailure()) {
    return;
  }
  ASSERT_TRUE(present_
                  ->set_resolve_target(
                      third.acquired.image_index,
                      {.device_ptr = destination.ptr, .row_pitch_bytes = DeviceFrame::pitch()},
                      writer_stream_)
                  .has_value());
  present_->set_fault_injection_for_tests({.fail_present_record = true});
  EXPECT_FALSE(present_
                   ->present(PresentArgs{.image_index = third.acquired.image_index,
                                         .acquire_issuance = third.acquired.issuance,
                                         .render_done = third.render_done,
                                         .present_id = 3})
                   .has_value());
  EXPECT_TRUE(present_->resolve_may_be_in_flight(third.acquired.image_index));
  auto ready = present_->ready_event(third.acquired.image_index);
  ASSERT_TRUE(ready.has_value()) << ready.error().message;
  {
    const holoviz::ScopedCudaContext scoped(cuda_context_);
    EXPECT_EQ(cuEventSynchronize(*ready), CUDA_SUCCESS);
  }
  ASSERT_TRUE(present_
                  ->present(PresentArgs{.image_index = third.acquired.image_index,
                                        .acquire_issuance = third.acquired.issuance})
                  .has_value());
  third = Frame{};
  // A plain successful frame afterwards reports its own (proven) in-flight resolve.
  Frame fourth;
  render_clear_frame(fourth);
  if (HasFatalFailure()) {
    return;
  }
  present_into(fourth, destination, 4);
  EXPECT_TRUE(present_->resolve_may_be_in_flight(fourth.acquired.image_index));
  EXPECT_EQ(present_->wait_for_present(4, std::chrono::seconds{5}).value(), WaitResult::kDisplayed);
}

// A recreation whose allocation fails leaves the live ring untouched: same extent and targets,
// a pre-resize acquire still presentable, and the frame loop still producing frames.
TEST_F(CudaPresentBackendTest, FailedResizeKeepsTheOldRing) {
  build_frame_loop();
  const DeviceFrame destination(cuda_context_);
  ASSERT_NE(destination.ptr, 0U);
  auto held = present_->acquire(std::chrono::seconds{1});
  ASSERT_TRUE(held.has_value());

  // Far beyond any adapter's image limits and memory, so the replacement ring cannot be built.
  const auto resized = present_->resize({1U << 20, 1U << 20});
  if (resized.has_value()) {
    GTEST_SKIP() << "the adapter accepted a 2^20 x 2^20 render target; nothing to fail";
  }
  const auto info = present_->swapchain_info();
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->extent.width, kExtent.width);
  EXPECT_EQ(info->extent.height, kExtent.height);
  EXPECT_TRUE(present_->targets().has_value());

  // The pre-resize acquire survived (no ring was replaced): releasing it is an ordinary present.
  ASSERT_TRUE(present_
                  ->present(PresentArgs{.image_index = held->image_index,
                                        .acquire_issuance = held->issuance})
                  .has_value());

  clear_destination(destination);
  Frame frame;
  render_clear_frame(frame);
  if (HasFatalFailure()) {
    return;
  }
  present_into(frame, destination, 1);
  if (HasFatalFailure()) {
    return;
  }
  EXPECT_EQ(present_->wait_for_present(1, std::chrono::seconds{5}).value(), WaitResult::kDisplayed);
  expect_clear_color(read_back_device(destination.ptr));
}

TEST_F(CudaPresentBackendTest, ResizeRecreatesTheRingAndDropsPendingAcquires) {
  build_frame_loop();
  auto stale = present_->acquire(std::chrono::seconds{1});
  ASSERT_TRUE(stale.has_value());

  ASSERT_TRUE(present_->resize({32, 32}).has_value());
  const auto info = present_->swapchain_info();
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->extent.width, 32U);
  EXPECT_EQ(info->extent.height, 32U);

  // The pre-resize acquire died with the old ring: presenting it is a consumed frame.
  EXPECT_EQ(present_
                ->present(PresentArgs{.image_index = stale->image_index,
                                      .acquire_issuance = stale->issuance,
                                      .present_id = 1})
                .error()
                .code,
            ErrorCode::kFrameDropped);
  EXPECT_EQ(present_->wait_for_present(1, std::chrono::seconds{0}).value(), WaitResult::kDropped);

  // The render backend re-warms the new targets on its next begin_frame(); release the frame
  // with an empty present (no destination was registered for the smaller ring).
  Frame frame;
  render_clear_frame(frame);
  if (HasFatalFailure()) {
    return;
  }
  ASSERT_TRUE(present_
                  ->present(PresentArgs{.image_index = frame.acquired.image_index,
                                        .acquire_issuance = frame.acquired.issuance})
                  .has_value());
}

// Every format the CudaSinkDesc table publishes must build a CUDA-exportable ring; the BGRA kinds
// render into their RGBA sibling, which swapchain_info() reports so the renderer builds for it.
TEST_F(CudaPresentBackendTest, AdmitsEveryPublishedFormat) {
  const std::array<std::pair<ImageFormat, ImageFormat>, 7> cases{{
      {ImageFormat::kR8Unorm, ImageFormat::kR8Unorm},
      {ImageFormat::kR8G8B8A8Unorm, ImageFormat::kR8G8B8A8Unorm},
      {ImageFormat::kB8G8R8A8Unorm, ImageFormat::kR8G8B8A8Unorm},
      {ImageFormat::kR8G8B8A8Srgb, ImageFormat::kR8G8B8A8Srgb},
      {ImageFormat::kB8G8R8A8Srgb, ImageFormat::kR8G8B8A8Srgb},
      {ImageFormat::kR16Unorm, ImageFormat::kR16Unorm},
      {ImageFormat::kR16G16B16A16Unorm, ImageFormat::kR16G16B16A16Unorm},
  }};
  for (const auto& [output_format, render_target_format] : cases) {
    auto backend = create_cuda_present_backend(device_, {.format = output_format});
    ASSERT_TRUE(backend.has_value());
    auto created = (*backend)->create_swapchain(SwapchainDesc{.preferred_extent = kExtent});
    ASSERT_TRUE(created.has_value())
        << static_cast<int>(output_format) << ": " << created.error().message;
    EXPECT_EQ((*backend)->swapchain_info()->format, render_target_format)
        << static_cast<int>(output_format);
    EXPECT_TRUE((*backend)->destroy(std::chrono::seconds{5}).has_value());
  }
}

// A BGRA sink renders into an RGBA target and swaps the channels in the resolve: the clear color
// (255, 128, 0, 255) must come out of the destination as B, G, R, A = (0, 128, 255, 255).
TEST_F(CudaPresentBackendTest, BgraFrameIsChannelSwapped) {
  recreate_present(ImageFormat::kB8G8R8A8Unorm);
  build_frame_loop();
  EXPECT_EQ(present_->swapchain_info()->format, ImageFormat::kR8G8B8A8Unorm);
  const DeviceFrame destination(cuda_context_);
  ASSERT_NE(destination.ptr, 0U);
  clear_destination(destination);

  Frame frame;
  render_clear_frame(frame);
  if (HasFatalFailure()) {
    return;
  }
  present_into(frame, destination, 1);
  if (HasFatalFailure()) {
    return;
  }
  EXPECT_EQ(present_->wait_for_present(1, std::chrono::seconds{5}).value(), WaitResult::kDisplayed);
  const std::vector<uint8_t> bytes = read_back_device(destination.ptr);
  ASSERT_EQ(bytes.size(), DeviceFrame::size_bytes());
  const std::array<uint8_t, 4> expected_bgra{
      kClearBytesRgba[2], kClearBytesRgba[1], kClearBytesRgba[0], kClearBytesRgba[3]};
  for (const size_t pixel :
       {size_t{0},
        static_cast<size_t>(kExtent.width) * (kExtent.height / 2) + kExtent.width / 2,
        static_cast<size_t>(kExtent.width) * kExtent.height - 1}) {
    for (size_t c = 0; c < 4; ++c) {
      EXPECT_NEAR(bytes[pixel * kBytesPerPixel + c], expected_bgra[c], 1) << "pixel " << pixel;
    }
  }
}

// The sRGB BGRA kind: same swap on an sRGB-encoded RGBA target. Red (1.0) and blue (0.0) encode
// to 255 and 0 regardless of the transfer function, so they pin the channel order; alpha is linear.
TEST_F(CudaPresentBackendTest, BgraSrgbFrameIsChannelSwapped) {
  recreate_present(ImageFormat::kB8G8R8A8Srgb);
  build_frame_loop();
  EXPECT_EQ(present_->swapchain_info()->format, ImageFormat::kR8G8B8A8Srgb);
  const DeviceFrame destination(cuda_context_);
  ASSERT_NE(destination.ptr, 0U);
  clear_destination(destination);

  Frame frame;
  render_clear_frame(frame);
  if (HasFatalFailure()) {
    return;
  }
  present_into(frame, destination, 1);
  if (HasFatalFailure()) {
    return;
  }
  EXPECT_EQ(present_->wait_for_present(1, std::chrono::seconds{5}).value(), WaitResult::kDisplayed);
  const std::vector<uint8_t> bytes = read_back_device(destination.ptr);
  ASSERT_EQ(bytes.size(), DeviceFrame::size_bytes());
  for (const size_t pixel : {size_t{0}, static_cast<size_t>(kExtent.width) * kExtent.height - 1}) {
    EXPECT_EQ(bytes[pixel * kBytesPerPixel + 0], 0) << "blue, pixel " << pixel;
    EXPECT_EQ(bytes[pixel * kBytesPerPixel + 2], 255) << "red, pixel " << pixel;
    EXPECT_EQ(bytes[pixel * kBytesPerPixel + 3], 255) << "alpha, pixel " << pixel;
  }
}

// A present() whose copy fails to enqueue after its render-completion wait was queued leaves that
// wait on the backend stream with no event behind it. destroy() must still prove the stream idle
// before releasing the ring, the semaphore import, and the stream: gated, a zero-timeout destroy()
// refuses with everything intact, and succeeds once the gate opens.
TEST_F(CudaPresentBackendTest, DestroyRefusesWhileAPreCopyWaitIsPending) {
  build_frame_loop();
  const DeviceFrame destination(cuda_context_);
  ASSERT_NE(destination.ptr, 0U);
  Frame frame;
  render_clear_frame(frame);  // the acquire signal is queued ahead of the gate
  if (HasFatalFailure()) {
    return;
  }
  auto stream = present_->stream();
  ASSERT_TRUE(stream.has_value());
  HostGate gate;
  {
    const holoviz::ScopedCudaContext scoped(cuda_context_);
    ASSERT_EQ(gate.block(*stream), CUDA_SUCCESS);
  }
  ASSERT_TRUE(present_
                  ->set_resolve_target(
                      frame.acquired.image_index,
                      {.device_ptr = destination.ptr, .row_pitch_bytes = DeviceFrame::pitch()},
                      writer_stream_)
                  .has_value());
  present_->set_fault_injection_for_tests({.fail_resolve_enqueue = true});
  EXPECT_FALSE(present_
                   ->present(PresentArgs{.image_index = frame.acquired.image_index,
                                         .acquire_issuance = frame.acquired.issuance,
                                         .render_done = frame.render_done,
                                         .present_id = 1})
                   .has_value());
  // No copy entered the stream, so the destination is not in use; the wait ahead of it is.
  EXPECT_FALSE(present_->resolve_may_be_in_flight(frame.acquired.image_index));
  // The render side is provably done (its submit waited only on the pre-gate acquire signal).
  ASSERT_TRUE(render_->destroy(std::chrono::seconds{5}).has_value());
  render_.reset();

  const auto refused = present_->destroy(std::chrono::seconds{0});
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().code, ErrorCode::kBackpressured);
  EXPECT_TRUE(present_->swapchain_info().has_value());
  EXPECT_TRUE(present_->targets().has_value());
  EXPECT_TRUE(present_->stream().has_value());
  gate.release();
  EXPECT_TRUE(present_->destroy(std::chrono::seconds{5}).has_value());
}

// A present() rejected for a missing destination queues nothing, but the stream it would have
// used may still carry other work (a consumer's ordering, here a gate): destroy() proves the whole
// stream idle, not only the ready events it recorded itself.
TEST_F(CudaPresentBackendTest, DestroyRefusesWhileTheStreamIsBusyAfterAMissingTarget) {
  build_frame_loop();
  Frame frame;
  render_clear_frame(frame);
  if (HasFatalFailure()) {
    return;
  }
  auto stream = present_->stream();
  ASSERT_TRUE(stream.has_value());
  HostGate gate;
  {
    const holoviz::ScopedCudaContext scoped(cuda_context_);
    ASSERT_EQ(gate.block(*stream), CUDA_SUCCESS);
  }
  EXPECT_EQ(present_
                ->present(PresentArgs{.image_index = frame.acquired.image_index,
                                      .acquire_issuance = frame.acquired.issuance,
                                      .render_done = frame.render_done,
                                      .present_id = 1})
                .error()
                .code,
            ErrorCode::kInvalidArgument);
  ASSERT_TRUE(render_->destroy(std::chrono::seconds{5}).has_value());
  render_.reset();

  const auto refused = present_->destroy(std::chrono::seconds{0});
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().code, ErrorCode::kBackpressured);
  EXPECT_TRUE(present_->swapchain_info().has_value());
  gate.release();
  EXPECT_TRUE(present_->destroy(std::chrono::seconds{5}).has_value());
}

}  // namespace
}  // namespace holoscan::visualization::backends
