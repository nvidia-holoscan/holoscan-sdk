// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// WsiPresentBackend unit tests. The mode resolver, image-count clamp, frame table, and
// token payloads are pure logic tested without a device; the acquire/present protocol runs
// against a real window + swapchain when the environment has a display (skipped otherwise).

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoviz/backends/wsi_present_backend.hpp>
#include <holoviz/present_backend.hpp>
#include <holoviz/rhi.hpp>

// Module-private headers (tests build with src/ on the include path, like test_rhi_destroy).
#include "backends/backend_tokens.hpp"
#include "backends/wsi_frame_table.hpp"
#include "backends/wsi_present_backend.hpp"
#include "backends/wsi_present_tracker.hpp"
#include "holoviz/handles.hpp"
#include "rhi/vulkan/vk_swapchain_util.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz::backends {
namespace {

// Fake device identities for pure-logic registry/payload tests: the registry compares
// uintptr_t keys, so it cannot tell a real DeviceImpl address from a fake one.
constexpr uintptr_t kFakeDeviceA = 0xA;
constexpr uintptr_t kFakeDeviceB = 0xB;

rhi::PresentModeSet make_set(std::initializer_list<PresentMode> modes) {
  rhi::PresentModeSet set;
  for (const PresentMode mode : modes) {
    set.add(mode);
  }
  return set;
}

TEST(ResolvePresentMode, AutoPrefersFifoLatestReadyThenFifoThenImmediate) {
  const auto all = make_set({PresentMode::kFIFOLatestReady,
                             PresentMode::kFIFO,
                             PresentMode::kMailbox,
                             PresentMode::kImmediate});
  EXPECT_EQ(resolve_present_mode(PresentMode::kAuto, all).value(), PresentMode::kFIFOLatestReady);

  const auto no_latest = make_set({PresentMode::kFIFO, PresentMode::kImmediate});
  EXPECT_EQ(resolve_present_mode(PresentMode::kAuto, no_latest).value(), PresentMode::kFIFO);

  const auto immediate_only = make_set({PresentMode::kImmediate});
  EXPECT_EQ(resolve_present_mode(PresentMode::kAuto, immediate_only).value(),
            PresentMode::kImmediate);

  const auto empty = make_set({});
  EXPECT_FALSE(resolve_present_mode(PresentMode::kAuto, empty).has_value());
}

TEST(ResolvePresentMode, SupportedExplicitRequestPassesThrough) {
  const auto all = make_set({PresentMode::kFIFOLatestReady,
                             PresentMode::kFIFO,
                             PresentMode::kMailbox,
                             PresentMode::kImmediate});
  for (const PresentMode mode : {PresentMode::kFIFO,
                                 PresentMode::kFIFOLatestReady,
                                 PresentMode::kMailbox,
                                 PresentMode::kImmediate}) {
    EXPECT_EQ(resolve_present_mode(mode, all).value(), mode);
  }
}

TEST(ResolvePresentMode, UnsupportedFifoFamilyDegradesTearFree) {
  const auto fifo_only = make_set({PresentMode::kFIFO});
  EXPECT_EQ(resolve_present_mode(PresentMode::kFIFOLatestReady, fifo_only).value(),
            PresentMode::kFIFO);
}

TEST(ResolvePresentMode, UnsupportedOptInFailsLoudly) {
  const auto fifo_only = make_set({PresentMode::kFIFO});
  const auto mailbox = resolve_present_mode(PresentMode::kMailbox, fifo_only);
  ASSERT_FALSE(mailbox.has_value());
  EXPECT_EQ(mailbox.error().code, ErrorCode::kInvalidArgument);

  const auto immediate = resolve_present_mode(PresentMode::kImmediate, fifo_only);
  ASSERT_FALSE(immediate.has_value());
  EXPECT_EQ(immediate.error().code, ErrorCode::kInvalidArgument);
}

TEST(ClampImageCount, RespectsSurfaceBounds) {
  EXPECT_EQ(rhi::vulkan::clamp_image_count(2, 3, 8), 3u);    // raised to min
  EXPECT_EQ(rhi::vulkan::clamp_image_count(16, 2, 8), 8u);   // capped to max
  EXPECT_EQ(rhi::vulkan::clamp_image_count(4, 2, 8), 4u);    // in range
  EXPECT_EQ(rhi::vulkan::clamp_image_count(64, 2, 0), 64u);  // max 0 = unbounded
}

TEST(WsiFrameTable, StorePeekEraseRoundTrip) {
  WsiFrameTable table;
  table.reset(3);

  auto payload = std::make_shared<AcquirePayload>(kFakeDeviceA);
  payload->generation = 1;
  payload->issuance = 7;
  ASSERT_TRUE(table.store(1, payload).has_value());

  // peek() is non-consuming: the entry survives until an explicit erase(), so a lower
  // present failure that consumed nothing leaves the frame retryable.
  auto peeked = table.peek(1, 1, 7);
  ASSERT_TRUE(peeked.has_value());
  EXPECT_EQ(peeked.value(), payload);
  auto peeked_again = table.peek(1, 1, 7);
  ASSERT_TRUE(peeked_again.has_value());
  EXPECT_EQ(peeked_again.value(), payload);

  table.erase(1);
  const auto after_erase = table.peek(1, 1, 7);
  ASSERT_FALSE(after_erase.has_value());
  EXPECT_EQ(after_erase.error().code, ErrorCode::kInvalidState);

  // The slot is free again after erase().
  ASSERT_TRUE(table.store(1, payload).has_value());
}

TEST(WsiFrameTable, DoubleAcquireIsInvalidState) {
  WsiFrameTable table;
  table.reset(2);
  ASSERT_TRUE(table.store(0, std::make_shared<AcquirePayload>(kFakeDeviceA)).has_value());
  const auto second = table.store(0, std::make_shared<AcquirePayload>(kFakeDeviceA));
  ASSERT_FALSE(second.has_value());
  EXPECT_EQ(second.error().code, ErrorCode::kInvalidState);
}

TEST(WsiFrameTable, StoreReplacesStaleGenerationEntry) {
  WsiFrameTable table;
  table.reset(2);
  auto stale = std::make_shared<AcquirePayload>(kFakeDeviceA);
  stale->generation = 1;
  stale->issuance = 1;
  ASSERT_TRUE(table.store(0, stale).has_value());

  // A same-image-count recreation keeps stale entries (presenting one reports
  // kFrameDropped), but a fresh acquire landing on the same index must replace the
  // occupant instead of wedging the frame loop with "acquired twice".
  auto fresh = std::make_shared<AcquirePayload>(kFakeDeviceA);
  fresh->generation = 2;
  fresh->issuance = 2;
  ASSERT_TRUE(table.store(0, fresh).has_value());
  auto peeked = table.peek(0, 2, 2);
  ASSERT_TRUE(peeked.has_value());
  EXPECT_EQ(peeked.value(), fresh);
}

// The stale/fresh ABA guard: after a fresh acquire replaced a superseded occupant on the
// same (recyclable) image index, the superseded acquire's issuance must never redeem the
// fresh occupant — it reports kFrameDropped, and holds() denies it erase permission so the
// fresh frame stays pending.
TEST(WsiFrameTable, SupersededIssuanceNeverRedeemsTheFreshOccupant) {
  WsiFrameTable table;
  table.reset(2);
  auto stale = std::make_shared<AcquirePayload>(kFakeDeviceA);
  stale->generation = 1;
  stale->issuance = 1;
  ASSERT_TRUE(table.store(0, stale).has_value());
  auto fresh = std::make_shared<AcquirePayload>(kFakeDeviceA);
  fresh->generation = 2;  // same-count recreation: fresh generation, same image index
  fresh->issuance = 2;
  ASSERT_TRUE(table.store(0, fresh).has_value());

  // The stale acquire's delayed present: dropped, without exposing (or erasing) B.
  const auto superseded = table.peek(0, 2, 1);
  ASSERT_FALSE(superseded.has_value());
  EXPECT_EQ(superseded.error().code, ErrorCode::kFrameDropped);
  EXPECT_FALSE(table.holds(0, 1)) << "a superseded acquire must not get erase permission";
  EXPECT_TRUE(table.holds(0, 2));

  // The fresh acquire is untouched and still redeemable.
  auto peeked = table.peek(0, 2, 2);
  ASSERT_TRUE(peeked.has_value());
  EXPECT_EQ(peeked.value(), fresh);
}

TEST(WsiFrameTable, PresentWithoutAcquireIsInvalidState) {
  WsiFrameTable table;
  table.reset(2);
  const auto peeked = table.peek(0, 1, 1);
  ASSERT_FALSE(peeked.has_value());
  EXPECT_EQ(peeked.error().code, ErrorCode::kInvalidState);
  EXPECT_FALSE(table.holds(0, 1));
}

TEST(WsiFrameTable, PeekStaleGenerationReportsFrameDroppedAndKeepsSlot) {
  WsiFrameTable table;
  table.reset(2);
  auto payload = std::make_shared<AcquirePayload>(kFakeDeviceA);
  payload->generation = 1;
  payload->issuance = 1;
  ASSERT_TRUE(table.store(0, payload).has_value());

  // The stale report is repeatable (non-consuming) until the caller erases explicitly.
  for (int i = 0; i < 2; ++i) {
    const auto peeked = table.peek(0, 2, 1);  // swapchain was recreated since the acquire
    ASSERT_FALSE(peeked.has_value());
    EXPECT_EQ(peeked.error().code, ErrorCode::kFrameDropped);
  }
  // A stale-generation occupant is still the caller's own frame: erase is permitted.
  EXPECT_TRUE(table.holds(0, 1));

  table.erase(0);
  ASSERT_TRUE(table.store(0, std::make_shared<AcquirePayload>(kFakeDeviceA)).has_value());
}

TEST(WsiFrameTable, OutOfRangeIndexIsInvalidArgument) {
  WsiFrameTable table;
  table.reset(2);
  EXPECT_EQ(table.store(2, std::make_shared<AcquirePayload>(kFakeDeviceA)).error().code,
            ErrorCode::kInvalidArgument);
  EXPECT_EQ(table.peek(2, 1, 1).error().code, ErrorCode::kInvalidArgument);
  EXPECT_FALSE(table.holds(2, 1));  // out-of-range holds is a defined "no"
  table.erase(2);                   // out-of-range erase is a defined no-op
}

TEST(WsiFrameTable, PendingIssuancesReportOnlyOccupiedSlots) {
  WsiFrameTable table;
  table.reset(3);
  EXPECT_TRUE(table.pending_issuances().empty());

  for (const auto [index, issuance] : {std::pair{2u, uint64_t{5}}, std::pair{0u, uint64_t{3}}}) {
    auto payload = std::make_shared<AcquirePayload>(kFakeDeviceA);
    payload->issuance = issuance;
    ASSERT_TRUE(table.store(index, payload).has_value());
  }

  // Sorted, and consumed frames (erased slots) are not reported — they must never grow
  // a drop tombstone.
  EXPECT_EQ(table.pending_issuances(), (std::vector<uint64_t>{3, 5}));
  table.erase(2);
  EXPECT_EQ(table.pending_issuances(), (std::vector<uint64_t>{3}));
}

TEST(WsiRetiredAcquireLog, ConsumeIsSingleShotAndUnknownIssuancesPassThrough) {
  WsiRetiredAcquireLog log;
  EXPECT_FALSE(log.consume(1));  // empty log: nothing is retired

  log.remember({1, 3});
  EXPECT_FALSE(log.consume(2));  // never retired: left for the frame table to classify
  EXPECT_TRUE(log.consume(3));
  EXPECT_FALSE(log.consume(3)) << "a retired acquire drops (and burns its id) exactly once";
  EXPECT_TRUE(log.consume(1));

  log.remember({4});
  log.clear();
  EXPECT_FALSE(log.consume(4));
}

TEST(WsiRetiredAcquireLog, CapacityEvictsOldestFirst) {
  WsiRetiredAcquireLog log;
  // Batches arrive in issuance order (the counter never resets), so overflowing the
  // bound must forget the OLDEST retirees while every recent one stays answerable.
  for (uint64_t issuance = 1; issuance <= WsiRetiredAcquireLog::kCapacity + 2; ++issuance) {
    log.remember({issuance});
  }
  EXPECT_FALSE(log.consume(1));
  EXPECT_FALSE(log.consume(2));
  EXPECT_TRUE(log.consume(3));
  EXPECT_TRUE(log.consume(WsiRetiredAcquireLog::kCapacity + 2));
}

TEST(WsiPresentTracker, SparseIdsReportGapsAsNeverIssued) {
  WsiPresentTracker tracker;
  tracker.reset(2);
  tracker.record(1, WsiPresentTracker::Outcome::kQueued, {});
  tracker.record(100, WsiPresentTracker::Outcome::kQueued, {});

  EXPECT_EQ(tracker.last_id(), 100u);
  EXPECT_EQ(tracker.lookup(50).state, WsiPresentTracker::Lookup::kNeverIssued);
  EXPECT_EQ(tracker.lookup(1).state, WsiPresentTracker::Lookup::kQueuedLive);
  EXPECT_EQ(tracker.lookup(100).state, WsiPresentTracker::Lookup::kQueuedLive);
}

TEST(WsiPresentTracker, KnownDropsStayDropsEvenAfterRetirement) {
  WsiPresentTracker tracker;
  tracker.reset(2);
  tracker.record(1, WsiPresentTracker::Outcome::kQueued, {});
  tracker.record(2, WsiPresentTracker::Outcome::kDropped, {});
  tracker.retire_all();

  EXPECT_EQ(tracker.lookup(1).state, WsiPresentTracker::Lookup::kQueuedRetired);
  EXPECT_EQ(tracker.lookup(2).state, WsiPresentTracker::Lookup::kDropped);
}

TEST(WsiPresentTracker, RingEvictionForgetsOutcomesExactly) {
  WsiPresentTracker tracker;
  tracker.reset(2);  // ring holds 4 records
  for (PresentId id = 1; id <= 6; ++id) {
    tracker.record(id, WsiPresentTracker::Outcome::kQueued, {});
  }
  // Ids 1 and 2 were overwritten: evicted, never a defaulted "displayed".
  EXPECT_EQ(tracker.lookup(1).state, WsiPresentTracker::Lookup::kEvicted);
  EXPECT_EQ(tracker.lookup(2).state, WsiPresentTracker::Lookup::kEvicted);
  EXPECT_EQ(tracker.lookup(3).state, WsiPresentTracker::Lookup::kQueuedLive);
  EXPECT_EQ(tracker.lookup(6).state, WsiPresentTracker::Lookup::kQueuedLive);
}

TEST(WsiPresentTracker, ResetForgetsAllOutcomesAndResetIdsRestartsTheSequence) {
  WsiPresentTracker tracker;
  tracker.reset(2);
  tracker.record(1, WsiPresentTracker::Outcome::kQueued, {});
  tracker.record(2, WsiPresentTracker::Outcome::kDropped, {});

  tracker.reset(3);                  // image-count-changing recreation
  EXPECT_EQ(tracker.last_id(), 2u);  // the id sequence survives a recreation...
  EXPECT_EQ(tracker.lookup(1).state, WsiPresentTracker::Lookup::kEvicted);
  EXPECT_EQ(tracker.lookup(2).state, WsiPresentTracker::Lookup::kEvicted);
  EXPECT_FALSE(tracker.is_monotonic(2));
  EXPECT_TRUE(tracker.is_monotonic(3));

  tracker.reset_ids();  // ...but a brand-new swapchain restarts it
  EXPECT_EQ(tracker.last_id(), 0u);
  EXPECT_TRUE(tracker.is_monotonic(1));
  EXPECT_TRUE(tracker.is_monotonic(0));  // untagged always passes
}

TEST(WsiPresentTracker, FallbackRenderDoneRidesQueuedLiveLookups) {
  WsiPresentTracker tracker;
  tracker.reset(2);
  tracker.record(1, WsiPresentTracker::Outcome::kQueued, rhi::SubmitSyncToken{42});
  const WsiPresentTracker::LookupResult looked_up = tracker.lookup(1);
  EXPECT_EQ(looked_up.state, WsiPresentTracker::Lookup::kQueuedLive);
  EXPECT_EQ(looked_up.render_done.timeline_value, 42u);
}

TEST(ErrorCodes, ResizeSignalsHaveStableValues) {
  // Numeric values are append-only once landed (errors.hpp).
  EXPECT_EQ(static_cast<uint32_t>(ErrorCode::kOutOfDate), 12u);
  EXPECT_EQ(static_cast<uint32_t>(ErrorCode::kImageCountChanged), 13u);
  EXPECT_EQ(static_cast<uint32_t>(ErrorCode::kFrameDropped), 14u);
}

TEST(BackendTokens, PayloadsRoundTripThroughOpaqueTokens) {
  auto submit = std::make_shared<SubmitPayload>(kFakeDeviceA);
  submit->token = rhi::SubmitSyncToken{42};
  const SyncToken opaque = SyncToken{submit};

  const SubmitLookup unwrapped = submit_payload(opaque, kFakeDeviceA);
  ASSERT_NE(unwrapped.payload, nullptr);
  EXPECT_EQ(unwrapped.check, TokenRegistry::Check::kOk);
  EXPECT_EQ(unwrapped.payload->token.timeline_value, 42u);
}

TEST(BackendTokens, MismatchedPayloadKindsAreRejected) {
  const SyncToken acquire_token = SyncToken{std::make_shared<AcquirePayload>(kFakeDeviceA)};
  const SyncToken submit_token = SyncToken{std::make_shared<SubmitPayload>(kFakeDeviceA)};

  EXPECT_EQ(submit_payload(acquire_token, kFakeDeviceA).payload, nullptr);
  EXPECT_EQ(submit_payload(acquire_token, kFakeDeviceA).check, TokenRegistry::Check::kWrongKind);
  EXPECT_EQ(acquire_payload(submit_token, kFakeDeviceA).payload, nullptr);
  EXPECT_EQ(acquire_payload(submit_token, kFakeDeviceA).check, TokenRegistry::Check::kWrongKind);
  EXPECT_EQ(submit_payload(SyncToken{}, kFakeDeviceA).payload, nullptr);
}

TEST(BackendTokens, ForeignTokensAreRejectedWithoutDereference) {
  // A one-byte foreign object: any scheme that probed a payload field would read out of
  // bounds; the registry rejects by address without touching the pointee.
  const SyncToken one_byte = SyncToken{std::make_shared<char>('x')};
  EXPECT_EQ(acquire_payload(one_byte, kFakeDeviceA).payload, nullptr);
  EXPECT_EQ(acquire_payload(one_byte, kFakeDeviceA).check, TokenRegistry::Check::kUnknown);
  EXPECT_EQ(submit_payload(one_byte, kFakeDeviceA).check, TokenRegistry::Check::kUnknown);
  EXPECT_EQ(acquire_payload(SyncToken{std::make_shared<int>(7)}, kFakeDeviceA).payload, nullptr);
}

TEST(BackendTokens, CrossDeviceTokensAreRejected) {
  // A genuine payload minted for one device must not pass validation against another —
  // the registry-level twin of the two-adapter integration case.
  const SyncToken token = SyncToken{std::make_shared<SubmitPayload>(kFakeDeviceA)};
  const SubmitLookup wrong_device = submit_payload(token, kFakeDeviceB);
  EXPECT_EQ(wrong_device.payload, nullptr);
  EXPECT_EQ(wrong_device.check, TokenRegistry::Check::kWrongDevice);
  EXPECT_NE(submit_payload(token, kFakeDeviceA).payload, nullptr);
}

TEST(BackendTokens, RegistryEntriesDieWithTheirPayload) {
  const void* address = nullptr;
  {
    const auto payload = std::make_shared<SubmitPayload>(kFakeDeviceA);
    address = payload.get();
    EXPECT_EQ(token_registry().check(address, TokenRegistry::Kind::kSubmit, kFakeDeviceA),
              TokenRegistry::Check::kOk);
  }
  EXPECT_EQ(token_registry().check(address, TokenRegistry::Kind::kSubmit, kFakeDeviceA),
            TokenRegistry::Check::kUnknown);
}

TEST(BackendTokens, SurfaceHandlesRequireRegistration) {
  EXPECT_EQ(surface_from_handle(SurfaceHandle{std::make_shared<char>('x')}), nullptr);
  EXPECT_EQ(surface_from_handle(SurfaceHandle{}), nullptr);
}

// --- Windowed protocol tests: skipped where the environment has no display. ---

class WsiPresentBackendWindowedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Resizable, so the swapchain's automatic-resize machinery is active in these tests
    // (a non-resizable surface skips it entirely).
    auto window = rhi::create_window(
        {.title = "wsi present backend test", .size = {320, 240}, .resizable = true});
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
  }

  void TearDown() override {
    if (backend_ != nullptr) {
      const auto destroyed = backend_->destroy(std::chrono::seconds(5));
      EXPECT_TRUE(destroyed.has_value());
      backend_.reset();
    }
    surface_handle_.impl.reset();
  }

  rhi::Window window_;
  SurfaceHandle surface_handle_;
  rhi::Device device_;
  std::unique_ptr<PresentBackend> backend_;
};

TEST_F(WsiPresentBackendWindowedTest, CreateSwapchainRejectsForeignSurfaceHandles) {
  // A one-byte foreign handle: rejected by registry identity, never dereferenced.
  const auto created = backend_->create_swapchain(
      SwapchainDesc{.surface = SurfaceHandle{std::make_shared<char>('x')},
                    .preferred_mode = PresentMode::kAuto,
                    .image_count = 2});
  ASSERT_FALSE(created.has_value());
  EXPECT_EQ(created.error().code, ErrorCode::kInvalidArgument);
}

TEST_F(WsiPresentBackendWindowedTest, CreateReportsResolvedState) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 3})
          .has_value());

  const auto info = backend_->swapchain_info();
  ASSERT_TRUE(info.has_value());
  EXPECT_GT(info->image_count, 0u);
  EXPECT_GT(info->extent.width, 0u);
  EXPECT_GT(info->extent.height, 0u);
  EXPECT_NE(info->active_mode, PresentMode::kAuto);

  // Feature availability must be reported honestly: exactly what the device negotiated.
  const auto device_caps = device_.capabilities();
  ASSERT_TRUE(device_caps.has_value());
  const PresentInfo static_info = backend_->info();
  EXPECT_EQ(static_info.name, "wsi");
  EXPECT_EQ(static_info.has_present_id, device_caps->present_completion);
  EXPECT_EQ(static_info.has_present_wait, device_caps->present_completion);
}

TEST_F(WsiPresentBackendWindowedTest, EmptyFramePresentAndWaitMatchesFeedbackCapability) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());

  // A few full cycles: acquire -> present with no rendering (null render_done). The RHI
  // presents an unrendered frame against the acquire semaphore, so no submit is needed.
  for (PresentId id = 1; id <= 3; ++id) {
    auto token = backend_->acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << token.error().message;
    ASSERT_NE(token->image_ready.impl, nullptr);

    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = token->image_index,
                                          .acquire_issuance = token->issuance,
                                          .render_done = {nullptr},
                                          .present_id = id})
                    .has_value());
  }

  // kDisplayed exclusively means display-confirmed: with native present-wait the queued id
  // resolves; without it the wait reports kNotImplemented (no render-done approximation).
  const auto waited = backend_->wait_for_present(2, std::chrono::seconds(5));
  if (backend_->info().has_present_wait) {
    ASSERT_TRUE(waited.has_value()) << waited.error().message;
    EXPECT_EQ(*waited, WaitResult::kDisplayed);
  } else {
    ASSERT_FALSE(waited.has_value());
    EXPECT_EQ(waited.error().code, ErrorCode::kNotImplemented);
  }

  // A never-presented id is kNotFound regardless of the capability.
  const auto unknown = backend_->wait_for_present(99, std::chrono::seconds(1));
  ASSERT_FALSE(unknown.has_value());
  EXPECT_EQ(unknown.error().code, ErrorCode::kNotFound);
}

TEST_F(WsiPresentBackendWindowedTest, WaitAcrossRecreationIsNotFound) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());

  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = token->image_index,
                                        .acquire_issuance = token->issuance,
                                        .present_id = 1})
                  .has_value());

  // The recreation drains rendering, which proves nothing about display: the queued
  // outcome is unresolvable, never a defaulted kDisplayed.
  ASSERT_TRUE(backend_->resize({0, 0}).has_value());
  const auto crossed = backend_->wait_for_present(1, std::chrono::seconds(1));
  ASSERT_FALSE(crossed.has_value());
  EXPECT_EQ(crossed.error().code, ErrorCode::kNotFound);
}

TEST_F(WsiPresentBackendWindowedTest, AbandonedAcquireReleasesViaEmptyPresent) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());

  // The documented abort/release path (present_backend.hpp): an acquired image that will
  // not be rendered is handed back with a null render_done and present_id 0 (untagged).
  // Cycling more times than there are images proves the frame-table slot is freed each
  // time — a leaked slot would fail the next acquire of that image index with
  // "acquired twice without an intervening present".
  for (int i = 0; i < 5; ++i) {
    auto token = backend_->acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << "cycle " << i << ": " << token.error().message;
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = token->image_index,
                                          .acquire_issuance = token->issuance,
                                          .render_done = {nullptr},
                                          .present_id = 0})
                    .has_value())
        << "cycle " << i;
  }
}

TEST_F(WsiPresentBackendWindowedTest, ProtocolViolationsAreRejected) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());

  // A missing acquire issuance (a caller that never copied AcquireToken::issuance) is
  // rejected loudly, before any consumption.
  const auto no_issuance = backend_->present(PresentArgs{.image_index = 0});
  ASSERT_FALSE(no_issuance.has_value());
  EXPECT_EQ(no_issuance.error().code, ErrorCode::kInvalidArgument);

  // Present without acquire.
  const auto bogus = backend_->present(PresentArgs{.image_index = 0, .acquire_issuance = 1});
  ASSERT_FALSE(bogus.has_value());
  EXPECT_EQ(bogus.error().code, ErrorCode::kInvalidState);

  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value());

  // A foreign render_done token is rejected before anything is consumed — a one-byte
  // pointee, so any scheme that probed a payload field would read out of bounds.
  const auto foreign = backend_->present(PresentArgs{.image_index = token->image_index,
                                                     .acquire_issuance = token->issuance,
                                                     .render_done = {std::make_shared<char>('x')}});
  ASSERT_FALSE(foreign.has_value());
  EXPECT_EQ(foreign.error().code, ErrorCode::kInvalidArgument);

  // A genuine payload minted for a different device is rejected the same way.
  const auto wrong_device = backend_->present(
      PresentArgs{.image_index = token->image_index,
                  .acquire_issuance = token->issuance,
                  .render_done = {std::make_shared<SubmitPayload>(kFakeDeviceB)}});
  ASSERT_FALSE(wrong_device.has_value());
  EXPECT_EQ(wrong_device.error().code, ErrorCode::kInvalidArgument);

  // Provenance is device identity, not producer identity: a payload minted by another
  // backend instance on the SAME device passes (its default token waits on nothing, so
  // this behaves as an empty present).
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{
                      .image_index = token->image_index,
                      .acquire_issuance = token->issuance,
                      .render_done = {std::make_shared<SubmitPayload>(device_identity(device_))},
                      .present_id = 1})
                  .has_value());

  // Non-monotonic present id.
  auto token2 = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token2.has_value());
  const auto stale_id = backend_->present(PresentArgs{
      .image_index = token2->image_index, .acquire_issuance = token2->issuance, .present_id = 1});
  ASSERT_FALSE(stale_id.has_value());
  EXPECT_EQ(stale_id.error().code, ErrorCode::kInvalidArgument);

  // The frame is still pending after the rejected present; presenting it correctly works.
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = token2->image_index,
                                        .acquire_issuance = token2->issuance,
                                        .present_id = 2})
                  .has_value());
}

TEST_F(WsiPresentBackendWindowedTest, ResizeRecreatesAndInvalidatesPendingAcquires) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());
  const auto info_before = backend_->swapchain_info();
  ASSERT_TRUE(info_before.has_value());

  // Resize before create_swapchain is the only invalid ordering; with a live swapchain a
  // resize must succeed even while a frame is acquired and unpresented.
  auto pending = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(pending.has_value());

  ASSERT_TRUE(backend_->resize(Extent2D{0, 0}).has_value());

  const auto info_after = backend_->swapchain_info();
  ASSERT_TRUE(info_after.has_value());
  EXPECT_EQ(info_after->format, info_before->format);
  EXPECT_NE(info_after->active_mode, PresentMode::kAuto);

  // The pre-resize token predates the recreation: its frame died with the old swapchain
  // generation (the payload's dead gates are never touched), reported as the same
  // continue-with-the-next-frame signal a mid-present resize produces.
  const auto stale = backend_->present(
      PresentArgs{.image_index = pending->image_index, .acquire_issuance = pending->issuance});
  ASSERT_FALSE(stale.has_value());
  EXPECT_EQ(stale.error().code, ErrorCode::kFrameDropped);

  // ...and the recreated swapchain runs full frames again.
  auto fresh = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(fresh.has_value()) << fresh.error().message;
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = fresh->image_index,
                                        .acquire_issuance = fresh->issuance,
                                        .present_id = 1})
                  .has_value());
}

TEST_F(WsiPresentBackendWindowedTest, SparseNeverIssuedIdReportsNotFound) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());

  // Ids are only required to increase, not to be contiguous: 1 then 100.
  for (const PresentId id : {PresentId{1}, PresentId{100}}) {
    auto token = backend_->acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << token.error().message;
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = token->image_index,
                                          .acquire_issuance = token->issuance,
                                          .render_done = {nullptr},
                                          .present_id = id})
                    .has_value());
  }

  // The gap id was never presented — kNotFound, not a defaulted kDisplayed (which the
  // native present-wait would also falsely satisfy because id 100 already passed).
  const auto gap = backend_->wait_for_present(50, std::chrono::seconds(1));
  ASSERT_FALSE(gap.has_value());
  EXPECT_EQ(gap.error().code, ErrorCode::kNotFound);

  const auto real = backend_->wait_for_present(100, std::chrono::seconds(5));
  if (backend_->info().has_present_wait) {
    ASSERT_TRUE(real.has_value()) << real.error().message;
    EXPECT_EQ(*real, WaitResult::kDisplayed);
  } else {
    // Queued but no display feedback: kNotImplemented, never a fake kDisplayed.
    ASSERT_FALSE(real.has_value());
    EXPECT_EQ(real.error().code, ErrorCode::kNotImplemented);
  }
}

TEST_F(WsiPresentBackendWindowedTest, AcquirePayloadsAreLeasedFromAFixedPoolWithoutAliasing) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());
  const auto info = backend_->swapchain_info();
  ASSERT_TRUE(info.has_value());
  const size_t pool_size = static_cast<size_t>(info->image_count) + 2;

  // Cycle more frames than the pool holds while retaining the previous frame's token
  // across each acquire: a retained (still-referenced) payload must never be re-leased.
  std::vector<const void*> seen;
  SyncToken held{};
  for (size_t i = 0; i < 2 * pool_size; ++i) {
    auto token = backend_->acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << "cycle " << i << ": " << token.error().message;
    const void* payload_ptr = token->image_ready.impl.get();
    ASSERT_NE(payload_ptr, nullptr);
    EXPECT_NE(payload_ptr, held.impl.get()) << "cycle " << i << ": aliased a retained payload";
    seen.push_back(payload_ptr);
    held = token->image_ready;  // keep this frame's payload alive across the next acquire
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = token->image_index,
                                          .acquire_issuance = token->issuance,
                                          .render_done = {nullptr},
                                          .present_id = 0})
                    .has_value())
        << "cycle " << i;
  }

  // Steady state recycles: with 2 * pool_size leases from a pool of pool_size (and no
  // heap fallback — exhaustion is kBackpressured), some payload pointer must repeat.
  std::sort(seen.begin(), seen.end());
  EXPECT_NE(std::adjacent_find(seen.begin(), seen.end()), seen.end())
      << "no payload was ever recycled — the pool is not being reused";
}

TEST_F(WsiPresentBackendWindowedTest, AcquirePoolExhaustionIsBackpressuredAndRecoverable) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());
  const auto info = backend_->swapchain_info();
  ASSERT_TRUE(info.has_value());
  const size_t pool_size = static_cast<size_t>(info->image_count) + 2;

  // Pin every pool payload: acquire, retain the token's payload reference, and release the
  // image back with an empty present so the swapchain itself never runs out of images.
  std::vector<SyncToken> pinned;
  for (size_t i = 0; i < pool_size; ++i) {
    auto token = backend_->acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << "cycle " << i << ": " << token.error().message;
    pinned.push_back(token->image_ready);
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = token->image_index,
                                          .acquire_issuance = token->issuance})
                    .has_value())
        << "cycle " << i;
  }

  // Exhausted: the lease is refused BEFORE the RHI acquire, so nothing is consumed and no
  // allocation happens (the old behavior heap-allocated a fallback payload here).
  const auto exhausted = backend_->acquire(std::chrono::seconds(5));
  ASSERT_FALSE(exhausted.has_value());
  EXPECT_EQ(exhausted.error().code, ErrorCode::kBackpressured);

  // Releasing one retained payload makes the very next acquire succeed — proof the
  // exhausted call left the RHI's single outstanding acquire available.
  pinned.pop_back();
  auto recovered = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message;
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = recovered->image_index,
                                        .acquire_issuance = recovered->issuance})
                  .has_value());
}

TEST_F(WsiPresentBackendWindowedTest, AcquireAfterSameCountRecreationReplacesStaleTableEntry) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());
  const auto info = backend_->swapchain_info();
  ASSERT_TRUE(info.has_value());

  // Hold an acquired-but-unpresented frame across an extent-only (same image count)
  // recreation. Its table entry deliberately survives so presenting it reports
  // kFrameDropped — but fresh acquires that land on the same image index must replace it
  // instead of failing "acquired twice" (which would abandon the fresh token and wedge
  // the frame loop).
  auto stale = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(stale.has_value()) << stale.error().message;
  ASSERT_TRUE(backend_->resize({0, 0}).has_value());

  for (uint32_t i = 0; i < info->image_count + 1; ++i) {
    auto token = backend_->acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << "cycle " << i << ": " << token.error().message;
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = token->image_index,
                                          .acquire_issuance = token->issuance})
                    .has_value())
        << "cycle " << i;
  }
}

// The stale/fresh ABA regression: retain acquire A across a same-image-count recreation,
// acquire fresh frames until B lands on A's (recycled) image index, then issue A's delayed
// present. A must drop by its own issuance — recorded as a known drop, without redeeming
// or disturbing B's pending frame — and B must still present successfully afterwards.
TEST_F(WsiPresentBackendWindowedTest, SupersededAcquireDropsWithoutStealingTheFreshFrame) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());
  const auto info_before = backend_->swapchain_info();
  ASSERT_TRUE(info_before.has_value());

  auto stale = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(stale.has_value()) << stale.error().message;
  EXPECT_NE(stale->issuance, 0u);

  ASSERT_TRUE(backend_->resize({0, 0}).has_value());
  const auto info_after = backend_->swapchain_info();
  ASSERT_TRUE(info_after.has_value());
  if (info_after->image_count != info_before->image_count) {
    GTEST_SKIP() << "the resize changed the image count on this driver; the same-count "
                    "replacement path under test is unreachable here";
  }

  // Hunt for the fresh acquire that reuses A's image index, releasing the others.
  std::optional<AcquireToken> fresh;
  for (uint32_t i = 0; i < 3 * info_after->image_count && !fresh.has_value(); ++i) {
    auto token = backend_->acquire(std::chrono::seconds(5));
    ASSERT_TRUE(token.has_value()) << "cycle " << i << ": " << token.error().message;
    if (token->image_index == stale->image_index) {
      fresh = std::move(*token);
      break;
    }
    ASSERT_TRUE(backend_
                    ->present(PresentArgs{.image_index = token->image_index,
                                          .acquire_issuance = token->issuance})
                    .has_value())
        << "cycle " << i;
  }
  ASSERT_TRUE(fresh.has_value()) << "no fresh acquire ever reused the stale image index";
  EXPECT_NE(fresh->issuance, stale->issuance);

  // A's delayed present: dropped by issuance identity (B replaced it in the table), the
  // tagged id burned and answerable, and B's pending frame untouched.
  const auto superseded = backend_->present(PresentArgs{
      .image_index = stale->image_index, .acquire_issuance = stale->issuance, .present_id = 1});
  ASSERT_FALSE(superseded.has_value());
  EXPECT_EQ(superseded.error().code, ErrorCode::kFrameDropped);
  const auto waited = backend_->wait_for_present(1, std::chrono::seconds(1));
  ASSERT_TRUE(waited.has_value()) << waited.error().message;
  EXPECT_EQ(*waited, WaitResult::kDropped);

  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = fresh->image_index,
                                        .acquire_issuance = fresh->issuance,
                                        .present_id = 2})
                  .has_value())
      << "the fresh frame must survive the superseded present untouched";
}

TEST_F(WsiPresentBackendWindowedTest, AcquirePayloadCarriesLiveSource) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());

  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  const auto payload = std::static_pointer_cast<const AcquirePayload>(token->image_ready.impl);
  ASSERT_NE(payload, nullptr);

  const std::shared_ptr<const AcquireSource> source = payload->source.lock();
  ASSERT_NE(source, nullptr);
  EXPECT_EQ(source->current_generation, payload->generation);

  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = token->image_index,
                                        .acquire_issuance = token->issuance,
                                        .render_done = {nullptr},
                                        .present_id = 0})
                  .has_value());
}

TEST_F(WsiPresentBackendWindowedTest, ResizeAdvancesAcquireSourceGeneration) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());

  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  const auto payload = std::static_pointer_cast<const AcquirePayload>(token->image_ready.impl);

  ASSERT_TRUE(backend_->resize(Extent2D{0, 0}).has_value());

  const std::shared_ptr<const AcquireSource> source = payload->source.lock();
  ASSERT_NE(source, nullptr);
  EXPECT_NE(source->current_generation, payload->generation)
      << "a recreation must advance the published generation so retained tokens are stale";
}

TEST_F(WsiPresentBackendWindowedTest, DestroyExpiresAcquireSource) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());

  auto token = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  const auto payload = std::static_pointer_cast<const AcquirePayload>(token->image_ready.impl);
  ASSERT_NE(payload->source.lock(), nullptr);

  ASSERT_TRUE(backend_->destroy(std::chrono::seconds(5)).has_value());
  EXPECT_EQ(payload->source.lock(), nullptr)
      << "destroy must expire the liveness cell so retained tokens are rejectable";
}

// A TAGGED frame dropped in the pre-present path (the frame table's generation check,
// before the RHI present is even attempted) consumes its id from the caller's strictly
// increasing sequence. The id must be recorded as a known drop, exactly like the
// post-present drop path records it: wait_for_present(id) answers kDropped instead of
// kNotFound now / a false answer for every later id.
TEST_F(WsiPresentBackendWindowedTest, DroppedTaggedFrameKeepsWaitForPresentConsistent) {
  ASSERT_TRUE(
      backend_
          ->create_swapchain(SwapchainDesc{
              .surface = surface_handle_, .preferred_mode = PresentMode::kAuto, .image_count = 2})
          .has_value());

  auto first = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(first.has_value()) << first.error().message;
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = first->image_index,
                                        .acquire_issuance = first->issuance,
                                        .present_id = 1})
                  .has_value());

  auto pending = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(pending.has_value()) << pending.error().message;
  ASSERT_TRUE(backend_->resize(Extent2D{0, 0}).has_value());

  const auto dropped = backend_->present(PresentArgs{
      .image_index = pending->image_index, .acquire_issuance = pending->issuance, .present_id = 2});
  ASSERT_FALSE(dropped.has_value());
  EXPECT_EQ(dropped.error().code, ErrorCode::kFrameDropped);

  // The dropped id answers exactly: kDropped (the frame never reached the display), never
  // kNotFound and never a blessed kDisplayed. Later ids keep working.
  const auto wait_dropped = backend_->wait_for_present(2, std::chrono::seconds(1));
  ASSERT_TRUE(wait_dropped.has_value()) << wait_dropped.error().message;
  EXPECT_EQ(*wait_dropped, WaitResult::kDropped);

  auto next = backend_->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(next.has_value()) << next.error().message;
  ASSERT_TRUE(backend_
                  ->present(PresentArgs{.image_index = next->image_index,
                                        .acquire_issuance = next->issuance,
                                        .present_id = 3})
                  .has_value());
  const auto wait_next = backend_->wait_for_present(3, std::chrono::seconds(5));
  if (backend_->info().has_present_wait) {
    ASSERT_TRUE(wait_next.has_value()) << wait_next.error().message;
  } else {
    ASSERT_FALSE(wait_next.has_value());
    EXPECT_EQ(wait_next.error().code, ErrorCode::kNotImplemented);
  }
}

TEST_F(WsiPresentBackendWindowedTest, RhiSwapchainGenerationTracksRecreations) {
  // Straight at the RHI: generation() is the staleness signal layered code keys off, and
  // an explicit in-place resize() must advance it while keeping the frame loop working.
  auto& surface = *std::static_pointer_cast<rhi::Surface>(surface_handle_.impl);
  ASSERT_TRUE(surface.resizable().value());  // the fixture's window is created resizable

  auto swapchain = device_.create_swapchain({.surface = std::cref(surface)});
  ASSERT_TRUE(swapchain.has_value()) << swapchain.error().message;

  const uint64_t initial = swapchain->generation().value();
  EXPECT_GT(initial, 0u);  // creation itself is the first generation

  ASSERT_TRUE(swapchain->resize().has_value());
  EXPECT_EQ(swapchain->generation().value(), initial + 1);

  auto acquired = swapchain->acquire(std::chrono::seconds(5));
  ASSERT_TRUE(acquired.has_value()) << acquired.error().message;
  ASSERT_TRUE(swapchain->present(std::move(*acquired), rhi::SubmitSyncToken{}).has_value());
  EXPECT_TRUE(swapchain->destroy(std::chrono::seconds(5)).has_value());
}

}  // namespace
}  // namespace holoviz::backends
