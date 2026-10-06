// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HolovizRenderBackend skeleton tests: the lifecycle state machine (CPU-checkable protocol
// violations) and the headless clear-only frame path with real readback. GPU tests skip on
// machines with no adapter, following the pattern of test_text_renderer_compare.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>
#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/backends/holoviz_render_backend.hpp"
#include "holoviz_op/backends/render_backend.hpp"

// Module-private headers (src/ is on this target's include path).
#include "holoviz_op/backends/backend_tokens.hpp"
#include "holoviz_op/backends/handles.hpp"
// Sanctioned test seam: the destructor-quarantine test observes the terminal device
// poison through device_resources_leaked(), which has no public spelling by design.
#include "vulkan/vk_swapchain_fault_injection.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoscan::visualization::backends {
namespace {

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

#define SKIP_WITHOUT_DEVICE(device)                              \
  if (!(device)) {                                               \
    HOLOVIZ_ENV_SKIP() << "no usable rendering device/adapter."; \
  }

class HolovizRenderBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = try_create_device();
    SKIP_WITHOUT_DEVICE(device_);
    auto backend = create_holoviz_render_backend(device_, {});
    ASSERT_TRUE(backend.has_value());
    backend_ = std::move(*backend);
  }

  void TearDown() override {
    backend_.reset();  // children (command lists, images) must go before the device
    if (device_) {
      ASSERT_TRUE(device_.destroy(std::chrono::seconds(5)).has_value());
    }
  }

  static RenderConfig headless_config(uint32_t width = 64, uint32_t height = 48) {
    RenderConfig config;
    config.headless = true;
    config.headless_width = width;
    config.headless_height = height;
    config.max_frames_in_flight = 2;
    return config;
  }

  holoviz::Device device_;
  std::unique_ptr<RenderBackend> backend_;
};

TEST_F(HolovizRenderBackendTest, LifecycleOrderIsEnforced) {
  // Everything before allocate() is kInvalidState.
  EXPECT_EQ(backend_->build(ImageFormat::kB8G8R8A8Unorm).error().code, ErrorCode::kInvalidState);
  EXPECT_EQ(backend_->begin_frame(AcquireToken{}, 0).error().code, ErrorCode::kInvalidState);

  ASSERT_TRUE(backend_->allocate(LayerStack{}, headless_config()).has_value());

  // begin_frame before build, allocate twice.
  EXPECT_EQ(backend_->begin_frame(AcquireToken{}, 0).error().code, ErrorCode::kInvalidState);
  EXPECT_EQ(backend_->allocate(LayerStack{}, headless_config()).error().code,
            ErrorCode::kInvalidState);

  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());
  EXPECT_EQ(backend_->build(ImageFormat::kB8G8R8A8Unorm).error().code, ErrorCode::kInvalidState);

  // Frame protocol: record/submit need FrameCommands from begin_frame.
  FrameCommands bogus = FrameCommands{std::make_shared<int>(0)};
  RenderList empty_list;
  EXPECT_EQ(backend_->record(bogus, empty_list, CompositeSpec{}).error().code,
            ErrorCode::kInvalidArgument);
  EXPECT_EQ(backend_->submit(bogus, SubmitContext{0}).error().code, ErrorCode::kInvalidArgument);

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  EXPECT_EQ(backend_->begin_frame(AcquireToken{}, 0).error().code, ErrorCode::kInvalidState);
  EXPECT_EQ(backend_->submit(*cmds, SubmitContext{1}).error().code, ErrorCode::kInvalidArgument);
  ASSERT_TRUE(backend_->submit(*cmds, SubmitContext{0}).has_value());
  EXPECT_EQ(backend_->submit(*cmds, SubmitContext{0}).error().code, ErrorCode::kInvalidState);
}

TEST_F(HolovizRenderBackendTest, UndeclaredLayerIndexIsRejected) {
  ASSERT_TRUE(backend_->allocate(LayerStack{}, headless_config()).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());

  RenderList list;
  list.layers.push_back(LayerFrame{0, nullptr});  // no layer 0 was declared
  EXPECT_EQ(backend_->record(*cmds, list, CompositeSpec{}).error().code,
            ErrorCode::kInvalidArgument);

  ASSERT_TRUE(backend_->submit(*cmds, SubmitContext{0}).has_value());
}

TEST_F(HolovizRenderBackendTest, AllFourLayerKindsAllocateAndBuild) {
  LayerStack bundle;
  bundle.add_image_layer("video", ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, 64, 48});
  bundle.add_lut_layer("mask", LUTLayerDesc{.lut_size = 16, .max_width = 64, .max_height = 48});
  bundle.add_geometry_layer("overlay", GeometryLayerDesc{{GeometryTopology::kRectangleList}, 16});
  bundle.add_text_layer("labels", TextLayerDesc{256});

  ASSERT_TRUE(backend_->allocate(bundle, headless_config()).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  auto stats_try = backend_->stats();
  ASSERT_TRUE(stats_try.has_value());
  const AllocationStats& stats = *stats_try;
  EXPECT_GT(stats.texture_pool_bytes, 0u);
  EXPECT_GT(stats.geometry_pool_bytes, 0u);
}

TEST_F(HolovizRenderBackendTest, StaleFrameCommandsFromPreviousIssuanceAreRejected) {
  ASSERT_TRUE(backend_->allocate(LayerStack{}, headless_config()).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  auto stale = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(stale.has_value());
  ASSERT_TRUE(backend_->submit(*stale, SubmitContext{0}).has_value());

  // Re-issue the same slot: the retained handle must not observe (or drive) the new frame.
  auto fresh = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(fresh.has_value());

  RenderList empty_list;
  EXPECT_EQ(backend_->record(*stale, empty_list, CompositeSpec{}).error().code,
            ErrorCode::kInvalidState);
  EXPECT_EQ(backend_->submit(*stale, SubmitContext{0}).error().code, ErrorCode::kInvalidState);
  // Aborting through the stale handle is the contract's no-op success: its frame was
  // already submitted, and the abort must not touch the LIVE issuance.
  EXPECT_TRUE(backend_->abort_frame(*stale).has_value());

  // The current issuance is unaffected and completes normally, even after the stale abort.
  ASSERT_TRUE(backend_->record(*fresh, empty_list, CompositeSpec{}).has_value());
  ASSERT_TRUE(backend_->submit(*fresh, SubmitContext{0}).has_value());

  // And the slot stays healthy for the next issuance (which reuses the stale handle's
  // ticket, so release the retained handle first — otherwise the retention bound applies).
  stale->impl.reset();
  auto again = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(again.has_value()) << again.error().message;
  ASSERT_TRUE(backend_->submit(*again, SubmitContext{0}).has_value());
}

TEST_F(HolovizRenderBackendTest, ThirdIssuanceWithTwoRetainedHandlesIsBackpressured) {
  ASSERT_TRUE(backend_->allocate(LayerStack{}, headless_config()).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  auto first = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(backend_->submit(*first, SubmitContext{0}).has_value());
  const void* first_ticket = first->impl.get();

  auto second = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(second.has_value());
  ASSERT_TRUE(backend_->submit(*second, SubmitContext{0}).has_value());

  // Both preallocated tickets are pinned: the third issuance must refuse with
  // kBackpressured instead of allocating a replacement ticket on the frame path (the old
  // behavior), leaving the slot untouched and retryable.
  const auto exhausted = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_FALSE(exhausted.has_value());
  EXPECT_EQ(exhausted.error().code, ErrorCode::kBackpressured);

  // The retained handles keep their strict stale semantics for record/submit and the
  // no-op contract for abort.
  RenderList empty_list;
  EXPECT_EQ(backend_->record(*first, empty_list, CompositeSpec{}).error().code,
            ErrorCode::kInvalidState);
  EXPECT_TRUE(backend_->abort_frame(*first).has_value());

  // Releasing the older handle unblocks the slot, and — the honest allocation-free proxy,
  // absent a counting allocator in this suite — the new issuance REUSES the released
  // ticket by pointer identity instead of minting a fresh one.
  first->impl.reset();
  auto third = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(third.has_value()) << third.error().message;
  EXPECT_EQ(third->impl.get(), first_ticket);
  ASSERT_TRUE(backend_->submit(*third, SubmitContext{0}).has_value());
}

TEST_F(HolovizRenderBackendTest, ForeignFrameCommandsAreNeverDereferenced) {
  ASSERT_TRUE(backend_->allocate(LayerStack{}, headless_config()).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  // A one-byte foreign payload: validation must reject by registry miss (pointer compare),
  // never by reading through the pointer — run under ASan/UBSan to prove it.
  FrameCommands tiny = FrameCommands{std::make_shared<char>('x')};
  RenderList empty_list;
  EXPECT_EQ(backend_->record(tiny, empty_list, CompositeSpec{}).error().code,
            ErrorCode::kInvalidArgument);
  EXPECT_EQ(backend_->abort_frame(tiny).error().code, ErrorCode::kInvalidArgument);

  // A same-kind handle minted by ANOTHER backend instance: same rejection (previously this
  // passed a magic check and indexed the wrong backend's slot vector).
  auto other_try = create_holoviz_render_backend(device_, {});
  ASSERT_TRUE(other_try.has_value());
  std::unique_ptr<RenderBackend> other = std::move(*other_try);
  ASSERT_TRUE(other->allocate(LayerStack{}, headless_config()).has_value());
  ASSERT_TRUE(other->build(ImageFormat::kB8G8R8A8Unorm).has_value());
  auto other_cmds = other->begin_frame(AcquireToken{}, 1);
  ASSERT_TRUE(other_cmds.has_value());

  EXPECT_EQ(backend_->record(*other_cmds, empty_list, CompositeSpec{}).error().code,
            ErrorCode::kInvalidArgument);
  EXPECT_EQ(backend_->submit(*other_cmds, SubmitContext{1}).error().code,
            ErrorCode::kInvalidArgument);
  EXPECT_EQ(backend_->abort_frame(*other_cmds).error().code, ErrorCode::kInvalidArgument);

  ASSERT_TRUE(other->abort_frame(*other_cmds).has_value());
  ASSERT_TRUE(other->destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(HolovizRenderBackendTest, StatsAccumulateExactlyAcrossTwoGeometryLayers) {
  const GeometryLayerDesc layer_desc{{GeometryTopology::kRectangleList}, 16};

  LayerStack one_layer;
  one_layer.add_geometry_layer("a", layer_desc);
  ASSERT_TRUE(backend_->allocate(one_layer, headless_config()).has_value());
  const auto one_stats = backend_->stats();
  ASSERT_TRUE(one_stats.has_value());
  ASSERT_GT(one_stats->geometry_pool_bytes, 0u);

  auto two_try = create_holoviz_render_backend(device_, {});
  ASSERT_TRUE(two_try.has_value());
  std::unique_ptr<RenderBackend> two = std::move(*two_try);
  LayerStack two_layers;
  two_layers.add_geometry_layer("a", layer_desc);
  two_layers.add_geometry_layer("b", layer_desc);
  ASSERT_TRUE(two->allocate(two_layers, headless_config()).has_value());
  const auto two_stats = two->stats();
  ASSERT_TRUE(two_stats.has_value());

  // Each layer owns a distinct pool; the diagnostic must sum them, not report the max.
  EXPECT_EQ(two_stats->geometry_pool_bytes, 2 * one_stats->geometry_pool_bytes);
  ASSERT_TRUE(two->destroy(std::chrono::seconds(5)).has_value());
}

// A mid-loop allocate() failure (an earlier layer's pool already committed, a later layer
// rejected) is terminal: the partial state is released and every later call — including a
// corrected retry — refuses with kInvalidState. The old behavior appended the retry's
// state to the leftovers, misaligning layer_states_ with the declared bundle so record()
// cast erased frame data through the wrong alternative (type-confusion UB under ASan).
TEST_F(HolovizRenderBackendTest, FailedAllocateIsTerminalAndReleasesPartialState) {
  LayerStack bundle;
  bundle.add_geometry_layer("overlay", GeometryLayerDesc{{GeometryTopology::kLineList}, 16});
  bundle.add_image_layer("video",
                         ImageLayerDesc{ImageFormat::kNV12, 64, 48});  // missing conversion

  const auto failed = backend_->allocate(bundle, headless_config());
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().code, ErrorCode::kInvalidArgument);

  // Terminal: the corrected retry must not run over the leftovers.
  LayerStack corrected;
  corrected.add_image_layer("video", ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, 64, 48});
  const auto retry = backend_->allocate(corrected, headless_config());
  ASSERT_FALSE(retry.has_value());
  EXPECT_EQ(retry.error().code, ErrorCode::kInvalidState);
  EXPECT_EQ(backend_->build(ImageFormat::kB8G8R8A8Unorm).error().code, ErrorCode::kInvalidState);

  // Destroy stays an idempotent no-op on the terminal backend...
  ASSERT_TRUE(backend_->destroy(std::chrono::seconds(5)).has_value());

  // ...and a fresh backend on the same device allocates the corrected set cleanly (the
  // failed allocate released its partial state instead of leaking or poisoning anything).
  auto fresh_try = create_holoviz_render_backend(device_, {});
  ASSERT_TRUE(fresh_try.has_value());
  std::unique_ptr<RenderBackend> fresh = std::move(*fresh_try);
  ASSERT_TRUE(fresh->allocate(corrected, headless_config()).has_value());
  ASSERT_TRUE(fresh->build(ImageFormat::kB8G8R8A8Unorm).has_value());
  ASSERT_TRUE(fresh->destroy(std::chrono::seconds(5)).has_value());
}

// Argument-validation rejections precede any mutation, so they stay retryable — only
// failures past that point are terminal.
TEST_F(HolovizRenderBackendTest, AllocateArgumentRejectionStaysRetryable) {
  RenderConfig bad = headless_config();
  bad.max_frames_in_flight = 0;
  const auto rejected = backend_->allocate(LayerStack{}, bad);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, ErrorCode::kInvalidArgument);

  ASSERT_TRUE(backend_->allocate(LayerStack{}, headless_config()).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());
}

TEST_F(HolovizRenderBackendTest, BeginFrameRejectsStaleOrExpiredAcquireSources) {
  RenderConfig config;
  config.headless = false;
  config.max_frames_in_flight = 2;
  ASSERT_TRUE(backend_->allocate(LayerStack{}, config).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  auto target = device_.create_image(holoviz::ImageDesc{
      .extent = {64, 48},
      .format = holoviz::Format::kB8G8R8A8Unorm,
      .usage = holoviz::ImageUsage::kRenderTarget,
      .debug_name = "begin_frame stale-token test target",
  });
  ASSERT_TRUE(target.has_value()) << target.error().message;

  auto source = std::make_shared<AcquireSource>();
  source->current_generation = 2;
  // Genuine provenance (this backend's device) so the token passes the registry check and
  // exercises the generation/liveness rejections behind it.
  auto payload = std::make_shared<AcquirePayload>(device_identity(device_));
  payload->image = &*target;
  payload->extent = {64, 48};
  payload->generation = 1;  // predates the producer's current generation
  payload->source = source;

  // Stale generation: rejected as kFrameDropped before the image is ever dereferenced.
  const auto stale = backend_->begin_frame(AcquireToken{0, SyncToken{payload}}, 0);
  ASSERT_FALSE(stale.has_value());
  EXPECT_EQ(stale.error().code, ErrorCode::kFrameDropped);

  // Expired producer: rejected as kInvalidState.
  payload->generation = 2;
  source.reset();
  const auto expired = backend_->begin_frame(AcquireToken{0, SyncToken{payload}}, 0);
  ASSERT_FALSE(expired.has_value());
  EXPECT_EQ(expired.error().code, ErrorCode::kInvalidState);

  // A live, generation-matched token opens the slot normally (the rejections above left it
  // fully untouched).
  auto live_source = std::make_shared<AcquireSource>();
  live_source->current_generation = 2;
  payload->source = live_source;
  auto cmds = backend_->begin_frame(AcquireToken{0, SyncToken{payload}}, 0);
  ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
  ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());

  ASSERT_TRUE(target->destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(HolovizRenderBackendTest, BeginFrameRejectsForeignAndCrossDeviceAcquireTokens) {
  RenderConfig config;
  config.headless = false;
  config.max_frames_in_flight = 2;
  ASSERT_TRUE(backend_->allocate(LayerStack{}, config).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  // A one-byte foreign payload: rejected by registry miss (pointer compare), never
  // dereferenced.
  const auto foreign =
      backend_->begin_frame(AcquireToken{0, SyncToken{std::make_shared<char>('x')}}, 0);
  ASSERT_FALSE(foreign.has_value());
  EXPECT_EQ(foreign.error().code, ErrorCode::kInvalidArgument);

  // A genuine acquire payload minted for a different device: rejected before its image or
  // gates can reach this backend's device.
  constexpr uintptr_t kOtherDevice = 0xB;
  auto source = std::make_shared<AcquireSource>();
  source->current_generation = 1;
  auto payload = std::make_shared<AcquirePayload>(kOtherDevice);
  payload->generation = 1;
  payload->source = source;
  const auto cross_device = backend_->begin_frame(AcquireToken{0, SyncToken{payload}}, 0);
  ASSERT_FALSE(cross_device.has_value());
  EXPECT_EQ(cross_device.error().code, ErrorCode::kInvalidArgument);
}

namespace {
// One fabricated windowed target set: the render backend consumes AcquireSource::targets
// exactly as WindowSurface publishes them, so tests can exercise the windowed
// framebuffer-prewarm path without a window (same pattern as the stale-source tests).
struct FakeTargets {
  std::shared_ptr<backends::AcquireSource> source;
  std::vector<holoviz::Image> images;
};

FakeTargets make_fake_targets(holoviz::Device& device, uint64_t generation, size_t count) {
  FakeTargets fake;
  fake.source = std::make_shared<backends::AcquireSource>();
  fake.source->current_generation = generation;
  fake.images.reserve(count);  // targets borrow &images[i]; growth must never relocate
  for (size_t i = 0; i < count; ++i) {
    auto image = device.create_image(holoviz::ImageDesc{
        .extent = {64, 48},
        .format = holoviz::Format::kB8G8R8A8Unorm,
        .usage = holoviz::ImageUsage::kRenderTarget,
        .debug_name = "register_targets test target",
    });
    if (!image.has_value()) {
      fake.source.reset();
      return fake;
    }
    fake.images.push_back(std::move(*image));
    fake.source->targets.push_back(backends::AcquireSource::Target{&fake.images.back(), {64, 48}});
  }
  return fake;
}
}  // namespace

TEST_F(HolovizRenderBackendTest, RegisterTargetsPrecreatesFramebuffersAndFramesCreateNone) {
  RenderConfig config;
  config.headless = false;
  config.max_frames_in_flight = 2;
  ASSERT_TRUE(backend_->allocate(LayerStack{}, config).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());
  auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());
  ASSERT_EQ(concrete->render_targets_created_for_test(), 0u)
      << "a windowed build with no layers must not create framebuffers yet";

  FakeTargets fake = make_fake_targets(device_, 1, 2);
  ASSERT_NE(fake.source, nullptr);
  // Vector growth must not move the images the targets point at.
  ASSERT_EQ(fake.source->targets[0].image, &fake.images[0]);
  ASSERT_EQ(fake.source->targets[1].image, &fake.images[1]);

  auto targets_payload = std::make_shared<TargetSetPayload>(device_identity(device_));
  targets_payload->source = fake.source;
  ASSERT_TRUE(backend_->register_targets(SyncToken{targets_payload}).has_value());
  const uint64_t after_register = concrete->render_targets_created_for_test();
  EXPECT_EQ(after_register, 2u) << "one framebuffer per registered target, on the setup path";

  // Frames over both targets: begin_render must be a pure cache hit — the counter freezes.
  for (int cycle = 0; cycle < 2; ++cycle) {
    for (uint32_t index = 0; index < 2; ++index) {
      auto payload = std::make_shared<AcquirePayload>(device_identity(device_));
      payload->image = fake.source->targets[index].image;
      payload->extent = {64, 48};
      payload->generation = 1;
      payload->source = fake.source;
      auto cmds = backend_->begin_frame(AcquireToken{index, SyncToken{payload}}, 0);
      ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
      ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());
    }
  }
  EXPECT_EQ(concrete->render_targets_created_for_test(), after_register)
      << "the frame path created a framebuffer after register_targets()";

  for (holoviz::Image& image : fake.images) {
    ASSERT_TRUE(image.destroy(std::chrono::seconds(5)).has_value());
  }
}

TEST_F(HolovizRenderBackendTest, SwapchainRecreationRewarmsExactlyOnceOnNextFrame) {
  RenderConfig config;
  config.headless = false;
  config.max_frames_in_flight = 2;
  ASSERT_TRUE(backend_->allocate(LayerStack{}, config).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());
  auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());

  FakeTargets fake = make_fake_targets(device_, 1, 2);
  ASSERT_NE(fake.source, nullptr);
  auto targets_payload = std::make_shared<TargetSetPayload>(device_identity(device_));
  targets_payload->source = fake.source;
  ASSERT_TRUE(backend_->register_targets(SyncToken{targets_payload}).has_value());
  const uint64_t warm_generation_1 = concrete->render_targets_created_for_test();

  // "Recreation": the old targets die (their cached framebuffers are evicted with their
  // views), a new generation of targets appears in the same source cell.
  FakeTargets fresh = make_fake_targets(device_, 2, 2);
  ASSERT_NE(fresh.source, nullptr);
  for (holoviz::Image& image : fake.images) {
    ASSERT_TRUE(image.destroy(std::chrono::seconds(5)).has_value());
  }
  fake.source->current_generation = 2;
  fake.source->targets = fresh.source->targets;

  // A stale (pre-recreation) payload is rejected BEFORE any warm work.
  auto stale = std::make_shared<AcquirePayload>(device_identity(device_));
  stale->image = fake.source->targets[0].image;
  stale->extent = {64, 48};
  stale->generation = 1;
  stale->source = fake.source;
  const auto dropped = backend_->begin_frame(AcquireToken{0, SyncToken{stale}}, 0);
  ASSERT_FALSE(dropped.has_value());
  EXPECT_EQ(dropped.error().code, ErrorCode::kFrameDropped);
  EXPECT_EQ(concrete->render_targets_created_for_test(), warm_generation_1);

  // The first frame of the new generation re-warms every target exactly once...
  for (int cycle = 0; cycle < 2; ++cycle) {
    auto payload = std::make_shared<AcquirePayload>(device_identity(device_));
    payload->image = fake.source->targets[0].image;
    payload->extent = {64, 48};
    payload->generation = 2;
    payload->source = fake.source;
    auto cmds = backend_->begin_frame(AcquireToken{0, SyncToken{payload}}, 0);
    ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
    ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());
    // ...and the second frame leaves the counter untouched.
    EXPECT_EQ(concrete->render_targets_created_for_test(), warm_generation_1 + 2)
        << "cycle " << cycle;
  }

  for (holoviz::Image& image : fresh.images) {
    ASSERT_TRUE(image.destroy(std::chrono::seconds(5)).has_value());
  }
}

TEST_F(HolovizRenderBackendTest, RegisterTargetsRejectsHeadlessForeignAndExpiredTokens) {
  ASSERT_TRUE(backend_->allocate(LayerStack{}, headless_config()).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  // Headless backends own their targets; registering is a caller error.
  const auto headless = backend_->register_targets(SyncToken{});
  ASSERT_FALSE(headless.has_value());
  EXPECT_EQ(headless.error().code, ErrorCode::kInvalidArgument);

  auto windowed_try = create_holoviz_render_backend(device_, {});
  ASSERT_TRUE(windowed_try.has_value());
  std::unique_ptr<RenderBackend> windowed = std::move(*windowed_try);
  RenderConfig config;
  config.headless = false;
  config.max_frames_in_flight = 2;
  ASSERT_TRUE(windowed->allocate(LayerStack{}, config).has_value());
  ASSERT_TRUE(windowed->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  // One-byte foreign token: rejected by registry identity, never dereferenced.
  const auto foreign = windowed->register_targets(SyncToken{std::make_shared<char>('x')});
  ASSERT_FALSE(foreign.has_value());
  EXPECT_EQ(foreign.error().code, ErrorCode::kInvalidArgument);

  // Genuine payload, expired producer: kInvalidState.
  auto expired_payload = std::make_shared<TargetSetPayload>(device_identity(device_));
  const auto expired = windowed->register_targets(SyncToken{expired_payload});
  ASSERT_FALSE(expired.has_value());
  EXPECT_EQ(expired.error().code, ErrorCode::kInvalidState);

  ASSERT_TRUE(windowed->destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(HolovizRenderBackendTest, AbortAfterRecordWithUploadsRetiresCudaWorkBounded) {
  LayerStack bundle;
  bundle.add_geometry_layer("overlay", GeometryLayerDesc{{GeometryTopology::kLineList}, 16});
  ASSERT_TRUE(backend_->allocate(bundle, headless_config()).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const float coords[] = {0.25F, 0.5F, 0.75F, 0.5F};
  const GeometryBatch batch{GeometryTopology::kLineList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}};
  const GeometryFrameData geometry{{&batch, 1}};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &geometry});
  ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());

  // The abort must retire the recorded upload (bounded) and reopen the slot cleanly.
  ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());
  auto again = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(again.has_value()) << again.error().message;
  ASSERT_TRUE(backend_->record(*again, list, CompositeSpec{}).has_value());
  ASSERT_TRUE(backend_->submit(*again, SubmitContext{0}).has_value());
}

TEST_F(HolovizRenderBackendTest, DestroyAfterRecordedButUnsubmittedFrameSucceeds) {
  LayerStack bundle;
  bundle.add_geometry_layer("overlay", GeometryLayerDesc{{GeometryTopology::kLineList}, 16});
  ASSERT_TRUE(backend_->allocate(bundle, headless_config()).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const float coords[] = {0.25F, 0.5F, 0.75F, 0.5F};
  const GeometryBatch batch{GeometryTopology::kLineList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}};
  const GeometryFrameData geometry{{&batch, 1}};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &geometry});
  ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());

  // Destroy with the recorded (never submitted, never aborted) CUDA upload still pending:
  // the copies must be retired before the staging/pool they touch is freed.
  ASSERT_TRUE(backend_->destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(HolovizRenderBackendTest, DestroyWithBorrowedStreamRetiresUploadsAndKeepsTheStream) {
  auto context_try = device_.cuda_context();
  ASSERT_TRUE(context_try.has_value()) << context_try.error().message;
  CUcontext context = *context_try;
  ASSERT_EQ(cuCtxPushCurrent(context), CUDA_SUCCESS);
  CUstream user_stream = nullptr;
  ASSERT_EQ(cuStreamCreate(&user_stream, CU_STREAM_DEFAULT), CUDA_SUCCESS);
  CUcontext popped = nullptr;
  cuCtxPopCurrent(&popped);

  auto borrowed_try = create_holoviz_render_backend(device_, {.stream = user_stream});
  ASSERT_TRUE(borrowed_try.has_value());
  std::unique_ptr<RenderBackend> borrowed = std::move(*borrowed_try);
  LayerStack bundle;
  bundle.add_geometry_layer("overlay", GeometryLayerDesc{{GeometryTopology::kLineList}, 16});
  ASSERT_TRUE(borrowed->allocate(bundle, headless_config()).has_value());
  ASSERT_TRUE(borrowed->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  auto cmds = borrowed->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const float coords[] = {0.25F, 0.5F, 0.75F, 0.5F};
  const GeometryBatch batch{GeometryTopology::kLineList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}};
  const GeometryFrameData geometry{{&batch, 1}};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &geometry});
  ASSERT_TRUE(borrowed->record(*cmds, list, CompositeSpec{}).has_value());

  // Destroy must retire the caller-stream uploads too — and must NOT destroy the stream.
  ASSERT_TRUE(borrowed->destroy(std::chrono::seconds(5)).has_value());
  borrowed.reset();

  ASSERT_EQ(cuCtxPushCurrent(context), CUDA_SUCCESS);
  EXPECT_EQ(cuStreamSynchronize(user_stream), CUDA_SUCCESS)
      << "the borrowed stream must survive the backend";
  EXPECT_EQ(cuStreamDestroy(user_stream), CUDA_SUCCESS);
  cuCtxPopCurrent(&popped);
}

namespace {
// Host-gated stream rig (the blocked-producer pattern from test_holoviz_swapchain_resize):
// a host function parks the stream on an atomic until the test releases it, making "the
// destructor's bounded drain times out with work still pending" deterministic.
std::atomic<bool> destructor_gate_released{false};

void block_stream_until_gate_released(void* /*unused*/) {
  while (!destructor_gate_released.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
}  // namespace

TEST_F(HolovizRenderBackendTest, DestructorTimeoutQuarantinesInsteadOfFreeing) {
  auto context_try = device_.cuda_context();
  ASSERT_TRUE(context_try.has_value()) << context_try.error().message;
  CUcontext context = *context_try;
  destructor_gate_released.store(false, std::memory_order_release);

  ASSERT_EQ(cuCtxPushCurrent(context), CUDA_SUCCESS);
  CUstream user_stream = nullptr;
  ASSERT_EQ(cuStreamCreate(&user_stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuLaunchHostFunc(user_stream, &block_stream_until_gate_released, nullptr),
            CUDA_SUCCESS);
  CUcontext popped = nullptr;
  cuCtxPopCurrent(&popped);

  // Uploads recorded behind the gate, never submitted: the destructor's 1 s Phase A drain
  // must time out with those uploads still referencing the backend's staging/pools.
  auto gated_try = create_holoviz_render_backend(device_, {.stream = user_stream});
  ASSERT_TRUE(gated_try.has_value());
  std::unique_ptr<RenderBackend> gated = std::move(*gated_try);
  LayerStack bundle;
  bundle.add_geometry_layer("overlay", GeometryLayerDesc{{GeometryTopology::kLineList}, 16});
  ASSERT_TRUE(gated->allocate(bundle, headless_config()).has_value());
  ASSERT_TRUE(gated->build(ImageFormat::kB8G8R8A8Unorm).has_value());
  auto cmds = gated->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const float coords[] = {0.25F, 0.5F, 0.75F, 0.5F};
  const GeometryBatch batch{GeometryTopology::kLineList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}};
  const GeometryFrameData geometry{{&batch, 1}};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &geometry});
  ASSERT_TRUE(gated->record(*cmds, list, CompositeSpec{}).has_value());

  // The destructor times out, quarantines the owned state (freeing it would be a
  // use-after-free — the gated uploads still reference it), and poisons the device.
  gated.reset();
  EXPECT_TRUE(holoviz::vulkan::device_resources_leaked(device_))
      << "a timed-out destructor must poison the device, not free live resources";

  // Release the gate: the quarantined uploads retire against still-alive memory — no
  // crash and no validation error (the suite runs with HOLOVIZ_VALIDATION=1).
  destructor_gate_released.store(true, std::memory_order_release);
  ASSERT_EQ(cuCtxPushCurrent(context), CUDA_SUCCESS);
  EXPECT_EQ(cuStreamSynchronize(user_stream), CUDA_SUCCESS);
  EXPECT_EQ(cuStreamDestroy(user_stream), CUDA_SUCCESS);
  cuCtxPopCurrent(&popped);

  // TearDown destroys the poisoned device: by design it leaks the native device instead
  // of destroying the parent of quarantined children (vk_device.cpp's terminal poison).
}

// Two independent backend/device pairs timing out into the destructor quarantine on two
// threads at once: the quarantine list is process-wide, so the concurrent push_backs must
// be synchronized (run under TSan to prove the mutex) and both bundles must survive —
// releasing the gate afterwards retires both rigs' quarantined uploads against
// still-alive memory, with the suite's HOLOVIZ_VALIDATION=1 watching.
TEST_F(HolovizRenderBackendTest, ConcurrentDestructorQuarantinesDoNotRace) {
  holoviz::Device second_device = try_create_device();
  SKIP_WITHOUT_DEVICE(second_device);
  destructor_gate_released.store(false, std::memory_order_release);

  struct Rig {
    holoviz::Device* device = nullptr;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    std::unique_ptr<RenderBackend> backend;
  };
  Rig rigs[2];
  rigs[0].device = &device_;
  rigs[1].device = &second_device;

  const float coords[] = {0.25F, 0.5F, 0.75F, 0.5F};
  const GeometryBatch batch{GeometryTopology::kLineList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}};
  const GeometryFrameData geometry{{&batch, 1}};
  for (Rig& rig : rigs) {
    auto context_try = rig.device->cuda_context();
    ASSERT_TRUE(context_try.has_value()) << context_try.error().message;
    rig.context = *context_try;
    ASSERT_EQ(cuCtxPushCurrent(rig.context), CUDA_SUCCESS);
    ASSERT_EQ(cuStreamCreate(&rig.stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
    ASSERT_EQ(cuLaunchHostFunc(rig.stream, &block_stream_until_gate_released, nullptr),
              CUDA_SUCCESS);
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);

    auto backend_try = create_holoviz_render_backend(*rig.device, {.stream = rig.stream});
    ASSERT_TRUE(backend_try.has_value());
    rig.backend = std::move(*backend_try);
    LayerStack bundle;
    bundle.add_geometry_layer("overlay", GeometryLayerDesc{{GeometryTopology::kLineList}, 16});
    ASSERT_TRUE(rig.backend->allocate(bundle, headless_config()).has_value());
    ASSERT_TRUE(rig.backend->build(ImageFormat::kB8G8R8A8Unorm).has_value());
    auto cmds = rig.backend->begin_frame(AcquireToken{}, 0);
    ASSERT_TRUE(cmds.has_value());
    RenderList list;
    list.layers.push_back(LayerFrame{0, &geometry});
    ASSERT_TRUE(rig.backend->record(*cmds, list, CompositeSpec{}).has_value());
  }

  // Each backend is single-threaded and pair-independent; running the two destructors
  // concurrently is legal and races them into quarantine_owned_state() together.
  std::thread first_teardown([&rigs] { rigs[0].backend.reset(); });
  std::thread second_teardown([&rigs] { rigs[1].backend.reset(); });
  first_teardown.join();
  second_teardown.join();
  EXPECT_TRUE(holoviz::vulkan::device_resources_leaked(device_));
  EXPECT_TRUE(holoviz::vulkan::device_resources_leaked(second_device));

  destructor_gate_released.store(true, std::memory_order_release);
  for (Rig& rig : rigs) {
    ASSERT_EQ(cuCtxPushCurrent(rig.context), CUDA_SUCCESS);
    EXPECT_EQ(cuStreamSynchronize(rig.stream), CUDA_SUCCESS);
    EXPECT_EQ(cuStreamDestroy(rig.stream), CUDA_SUCCESS);
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
  }
  // Both devices carry the terminal poison; the fixture's TearDown handles device_ and
  // this destroy leaks the second device's native handle the same deliberate way.
  ASSERT_TRUE(second_device.destroy(std::chrono::seconds(5)).has_value());
}

TEST_F(HolovizRenderBackendTest, ExternalSemaphoreCreationWithNoCurrentContext) {
  // allocate() imports a CUDA external semaphore for upload-bearing layers; that import
  // must be self-scoped to the device's context, not the caller's ambient one (which is
  // null here — gtest never made any context current on this thread).
  CUcontext ambient = nullptr;
  for (int i = 0; i < 8; ++i) {  // ensure nothing is current (pop any leftover stack)
    if (cuCtxGetCurrent(&ambient) != CUDA_SUCCESS || ambient == nullptr) {
      break;
    }
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
  }
  ASSERT_EQ(cuCtxGetCurrent(&ambient), CUDA_SUCCESS);
  ASSERT_EQ(ambient, nullptr) << "test precondition: no ambient CUDA context";

  LayerStack bundle;
  bundle.add_geometry_layer("overlay", GeometryLayerDesc{{GeometryTopology::kLineList}, 16});
  ASSERT_TRUE(backend_->allocate(bundle, headless_config()).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  // A full upload-bearing frame proves the semaphore signal/wait round-trip works.
  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const float coords[] = {0.25F, 0.5F, 0.75F, 0.5F};
  const GeometryBatch batch{GeometryTopology::kLineList, coords, 1, {1.0F, 1.0F, 1.0F, 1.0F}};
  const GeometryFrameData geometry{{&batch, 1}};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &geometry});
  ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());
  ASSERT_TRUE(backend_->submit(*cmds, SubmitContext{0}).has_value());

  // The push/pop discipline restored the caller's (null) ambient context.
  ASSERT_EQ(cuCtxGetCurrent(&ambient), CUDA_SUCCESS);
  EXPECT_EQ(ambient, nullptr) << "backend leaked a CUDA context onto the calling thread";
}

TEST_F(HolovizRenderBackendTest, HeadlessClearFrameReadsBackTheClearColor) {
  constexpr uint32_t kWidth = 64;
  constexpr uint32_t kHeight = 48;
  ASSERT_TRUE(backend_->allocate(LayerStack{}, headless_config(kWidth, kHeight)).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

  // Two frames through different slots, to exercise slot cycling.
  SyncToken last_token;
  for (uint32_t frame = 0; frame < 2; ++frame) {
    auto cmds = backend_->begin_frame(AcquireToken{}, frame % 2);
    ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
    ASSERT_TRUE(backend_->record(*cmds, RenderList{}, CompositeSpec{}).has_value());
    auto token = backend_->submit(*cmds, SubmitContext{frame % 2});
    ASSERT_TRUE(token.has_value()) << token.error().message;
    last_token = *token;
  }

  const SubmitPayload* payload = submit_payload(last_token, device_identity(device_)).payload;
  ASSERT_NE(payload, nullptr);
  ASSERT_TRUE(device_.wait_on_submit(payload->token, std::chrono::seconds(5)).has_value());

  auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());
  const auto target = concrete->headless_target(1);
  ASSERT_TRUE(target.has_value());

  std::vector<std::byte> pixels(static_cast<size_t>(kWidth) * kHeight * 4);
  ASSERT_TRUE(
      device_
          .download_from_image(
              target->get(), 0, std::span<std::byte>(pixels), static_cast<size_t>(kWidth) * 4)
          .has_value());

  // The skeleton clears to opaque black; every pixel must read back exactly that.
  for (size_t i = 0; i < pixels.size(); i += 4) {
    ASSERT_EQ(std::to_integer<int>(pixels[i + 0]), 0) << "pixel " << i / 4;
    ASSERT_EQ(std::to_integer<int>(pixels[i + 1]), 0) << "pixel " << i / 4;
    ASSERT_EQ(std::to_integer<int>(pixels[i + 2]), 0) << "pixel " << i / 4;
    ASSERT_EQ(std::to_integer<int>(pixels[i + 3]), 255) << "pixel " << i / 4;
  }
}

}  // namespace
}  // namespace holoscan::visualization::backends
