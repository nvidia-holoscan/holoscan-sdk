// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Present-timing tests (SwapchainDesc::enable_present_timing, Swapchain::
// past_present_timings() / refresh_duration_ns() / present_timing_domain(),
// Device::calibrated_timestamps()). The contract checks on empty handles are pure logic;
// everything else runs against a real window + swapchain and skips where the environment
// has no display or the driver lacks the extensions.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"
#include "vulkan/vk_swapchain_fault_injection.hpp"

namespace holoviz {
namespace {

TEST(RhiPresentTiming, EmptyHandlesAreInvalidState) {
  std::array<PresentTimingSample, 1> samples{};
  const auto drained = Swapchain{}.past_present_timings(samples);
  ASSERT_FALSE(drained.has_value());
  EXPECT_EQ(drained.error().code, ErrorCode::kInvalidState);

  const auto refresh = Swapchain{}.refresh_duration_ns();
  ASSERT_FALSE(refresh.has_value());
  EXPECT_EQ(refresh.error().code, ErrorCode::kInvalidState);

  const auto domain = Swapchain{}.present_timing_domain();
  ASSERT_FALSE(domain.has_value());
  EXPECT_EQ(domain.error().code, ErrorCode::kInvalidState);

  const auto calibrated = Device{}.calibrated_timestamps();
  ASSERT_FALSE(calibrated.has_value());
  EXPECT_EQ(calibrated.error().code, ErrorCode::kInvalidState);
}

// --- Windowed tests: skipped where the environment has no display. ---

class RhiPresentTimingWindowedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // A window where a window system is available, otherwise a direct-display surface on a
    // free display (console session) — the same fallback as the swapchain-resize suite, so the
    // present-timing protocol tests still run on a rig without a compositor.
    auto window = create_window({.title = "present timing test", .size = {320, 240}});
    if (window) {
      window_ = std::move(*window);
      auto surface = window_.create_surface();
      ASSERT_TRUE(surface.has_value()) << surface.error().message;
      surface_ = std::move(*surface);
    } else {
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

    auto caps = device_.capabilities();
    ASSERT_TRUE(caps.has_value()) << caps.error().message;
    caps_ = *caps;

    auto surface_caps = device_.surface_capabilities(surface_);
    ASSERT_TRUE(surface_caps.has_value()) << surface_caps.error().message;
    surface_caps_ = *surface_caps;
  }

  void TearDown() override {
    if (device_) {
      EXPECT_TRUE(device_.wait_idle().has_value());
    }
    for (auto& command_list : command_lists_) {
      EXPECT_TRUE(command_list.destroy(std::chrono::seconds(5)).has_value());
    }
    if (swapchain_) {
      EXPECT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
    }
  }

  /// Create the fixture swapchain; returns the creation result for the tests that expect
  /// failure.
  expected<void> create_swapchain(const SwapchainDesc& desc) {
    SwapchainDesc full_desc = desc;
    full_desc.surface = surface_;
    auto swapchain = device_.create_swapchain(full_desc);
    if (!swapchain) {
      return make_unexpected(swapchain.error());
    }
    swapchain_ = std::move(*swapchain);
    return {};
  }

  bool timing_supported() const { return caps_.present_timing && surface_caps_.present_timing; }

  /// The first display nobody drives (Linux DRM, from a console), as a direct-scanout surface.
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
        if (auto surface = display.create_surface({})) {
          return surface;
        }
      }
    }
    return make_unexpected(Error{ErrorCode::kNotFound, "no display available for direct scanout"});
  }

  /// One frame tagged with \p present_id: acquire, submit an empty batch that consumes
  /// both gates (so the present waits on real queue work, as every application frame
  /// does; with present timing enabled this driver stalls ~1 s on presents whose wait
  /// semaphore is the bare acquire semaphore), then present.
  expected<void> present_frame(uint64_t present_id, PresentConsumed* consumed = nullptr) {
    auto token = swapchain_.acquire(std::chrono::seconds(5));
    if (!token) {
      return make_unexpected(token.error());
    }
    auto command_list = device_.create_command_list({});
    if (!command_list) {
      return make_unexpected(command_list.error());
    }
    if (auto begun = command_list->begin(); !begun) {
      return make_unexpected(begun.error());
    }
    if (auto ended = command_list->end(); !ended) {
      return make_unexpected(ended.error());
    }
    auto submitted = device_.submit({
        .command_lists = {&*command_list, 1},
        .acquire_gates = {&token->acquire_gate, 1},
        .present_gates = {&token->present_gate, 1},
    });
    if (!submitted) {
      return make_unexpected(submitted.error());
    }
    command_lists_.push_back(std::move(*command_list));
    return swapchain_.present(std::move(*token), *submitted, present_id, consumed);
  }

  /// Acquire an image and submit an empty command list gated on it, like present_frame()
  /// does, but hand the tokens back so the caller can retry present() with them: acquire()
  /// refuses while a previous acquire is outstanding, so a backpressured present must be
  /// retried with the same token, never re-acquired.
  expected<void> acquire_and_submit(AcquireSyncToken& token, SubmitSyncToken& submitted) {
    auto acquired = swapchain_.acquire(std::chrono::seconds(5));
    if (!acquired) {
      return make_unexpected(acquired.error());
    }
    auto command_list = device_.create_command_list({});
    if (!command_list) {
      return make_unexpected(command_list.error());
    }
    if (auto begun = command_list->begin(); !begun) {
      return make_unexpected(begun.error());
    }
    if (auto ended = command_list->end(); !ended) {
      return make_unexpected(ended.error());
    }
    auto submit = device_.submit({
        .command_lists = {&*command_list, 1},
        .acquire_gates = {&acquired->acquire_gate, 1},
        .present_gates = {&acquired->present_gate, 1},
    });
    if (!submit) {
      return make_unexpected(submit.error());
    }
    command_lists_.push_back(std::move(*command_list));
    token = std::move(*acquired);
    submitted = *submit;
    return {};
  }

  /// Present \p token with \p present_id, draining completed timing reports and retrying
  /// while the local timing queue is full, until it is admitted or two seconds pass. A
  /// completed report frees its slot whether or not it yields a representable sample, so
  /// admission (not a sample count) is the observable. A kBackpressured refusal that consumed
  /// nothing leaves the token untouched, which is what makes retrying it valid; one that
  /// consumed the frame (a timed-out recovery drain) is returned instead.
  expected<void> retry_present_after_drain(AcquireSyncToken&& token, SubmitSyncToken submitted,
                                           uint64_t present_id, PresentConsumed* consumed) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    std::array<PresentTimingSample, 16> buffer{};
    for (;;) {
      auto drained = swapchain_.past_present_timings(buffer);
      if (!drained) {
        return make_unexpected(drained.error());
      }
      *consumed = {};
      // Only a refusal that consumed nothing reaches the next iteration, so the token is
      // still valid when it is moved again.
      // NOLINTNEXTLINE(bugprone-use-after-move)
      auto result = swapchain_.present(std::move(token), submitted, present_id, consumed);
      const bool nothing_consumed = !consumed->image && !consumed->id;
      if (result || result.error().code != ErrorCode::kBackpressured || !nothing_consumed ||
          std::chrono::steady_clock::now() >= deadline) {
        return result;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  /// Present a fresh frame, recovering from either way the engine can refuse one, until it
  /// is admitted or \p timeout passes. A kBackpressured refusal that consumed nothing leaves
  /// the token valid, so it is retried as-is; kFrameDropped means the engine consumed the frame
  /// and its id, so the next attempt needs a fresh acquire and the next present id (ids must
  /// strictly increase). Reports are drained before every attempt,
  /// because it is a completed report freeing its slot that eventually admits the present.
  /// \p attempts and \p waited report what it took, so a caller can tell a slow engine from
  /// one that never frees the slot at all.
  expected<void> present_fresh_after_drain(uint64_t present_id, PresentConsumed* consumed,
                                           std::chrono::milliseconds timeout, unsigned* attempts,
                                           std::chrono::milliseconds* waited) {
    const auto started = std::chrono::steady_clock::now();
    const auto deadline = started + timeout;
    std::array<PresentTimingSample, 16> buffer{};
    AcquireSyncToken token{};
    SubmitSyncToken submitted{};
    *attempts = 0;
    if (auto ready = acquire_and_submit(token, submitted); !ready) {
      return make_unexpected(ready.error());
    }
    for (;;) {
      auto drained = swapchain_.past_present_timings(buffer);
      if (!drained) {
        return make_unexpected(drained.error());
      }
      *consumed = {};
      ++*attempts;
      // A retried token consumed nothing; a dropped frame is replaced by acquire_and_submit.
      // NOLINTNEXTLINE(bugprone-use-after-move)
      auto result = swapchain_.present(std::move(token), submitted, present_id, consumed);
      *waited = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - started);
      const bool nothing_consumed = !consumed->image && !consumed->id;
      const bool retryable =
          !result && ((result.error().code == ErrorCode::kBackpressured && nothing_consumed) ||
                      result.error().code == ErrorCode::kFrameDropped);
      if (result || !retryable || std::chrono::steady_clock::now() >= deadline) {
        return result;
      }
      if (result.error().code == ErrorCode::kFrameDropped) {
        ++present_id;
        if (auto ready = acquire_and_submit(token, submitted); !ready) {
          return make_unexpected(ready.error());
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  /// Drain samples into \p out until \p wanted have arrived or \p timeout elapses.
  void drain_into(std::vector<PresentTimingSample>& out, size_t wanted,
                  std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::array<PresentTimingSample, 16> buffer{};
    while (out.size() < wanted && std::chrono::steady_clock::now() < deadline) {
      auto drained = swapchain_.past_present_timings(buffer);
      ASSERT_TRUE(drained.has_value()) << drained.error().message;
      if (*drained == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      out.insert(out.end(), buffer.begin(), buffer.begin() + *drained);
    }
  }

  Window window_;
  Surface surface_;
  Device device_;
  Swapchain swapchain_;
  std::vector<CommandList> command_lists_;  ///< One per presented frame; destroyed after wait_idle.
  CapabilityReport caps_{};
  SurfaceCapabilities surface_caps_{};
};

TEST_F(RhiPresentTimingWindowedTest, NotOptedInIsInvalidState) {
  ASSERT_TRUE(create_swapchain({.present_mode = PresentMode::kFIFO, .image_count = 3}).has_value());
  std::array<PresentTimingSample, 1> samples{};
  const auto drained = swapchain_.past_present_timings(samples);
  ASSERT_FALSE(drained.has_value());
  EXPECT_EQ(drained.error().code, ErrorCode::kInvalidState);
  const auto refresh = swapchain_.refresh_duration_ns();
  ASSERT_FALSE(refresh.has_value());
  EXPECT_EQ(refresh.error().code, ErrorCode::kInvalidState);
  const auto domain = swapchain_.present_timing_domain();
  ASSERT_FALSE(domain.has_value());
  EXPECT_EQ(domain.error().code, ErrorCode::kInvalidState);
  const auto mode = swapchain_.refresh_mode();
  ASSERT_FALSE(mode.has_value());
  EXPECT_EQ(mode.error().code, ErrorCode::kInvalidState);
}

TEST_F(RhiPresentTimingWindowedTest, OptInWithoutSupportIsInvalidArgument) {
  if (timing_supported()) {
    GTEST_SKIP() << "present timing is supported here; the rejection path is not reachable";
  }
  const auto created = create_swapchain(
      {.present_mode = PresentMode::kFIFO, .image_count = 3, .enable_present_timing = true});
  ASSERT_FALSE(created.has_value());
  EXPECT_EQ(created.error().code, ErrorCode::kInvalidArgument);
}

TEST_F(RhiPresentTimingWindowedTest, ZeroQueueSizeIsInvalidArgument) {
  if (!caps_.present_timing) {
    GTEST_SKIP() << "no VK_EXT_present_timing on this device";
  }
  const auto created = create_swapchain({.present_mode = PresentMode::kFIFO,
                                         .image_count = 3,
                                         .enable_present_timing = true,
                                         .present_timing_queue_size = 0});
  ASSERT_FALSE(created.has_value());
  EXPECT_EQ(created.error().code, ErrorCode::kInvalidArgument);
}

TEST_F(RhiPresentTimingWindowedTest, SurfaceCapabilitiesReportStages) {
  if (!caps_.present_timing) {
    GTEST_SKIP() << "no VK_EXT_present_timing on this device";
  }
  if (!surface_caps_.present_timing) {
    EXPECT_TRUE(surface_caps_.present_stages.empty());
    GTEST_SKIP() << "this surface does not support present timing";
  }
  EXPECT_FALSE(surface_caps_.present_stages.empty());
}

TEST_F(RhiPresentTimingWindowedTest, ExplicitDomainIsHonoredOrRejected) {
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  bool any_succeeded = false;
  for (const TimeDomain requested : {TimeDomain::kDevice,
                                     TimeDomain::kHostMonotonic,
                                     TimeDomain::kSwapchainLocal,
                                     TimeDomain::kPresentStageLocal}) {
    const auto created = create_swapchain({.present_mode = PresentMode::kFIFO,
                                           .image_count = 3,
                                           .enable_present_timing = true,
                                           .present_timing_domain = requested});
    if (created) {
      any_succeeded = true;
      const auto domain = swapchain_.present_timing_domain();
      ASSERT_TRUE(domain.has_value()) << domain.error().message;
      EXPECT_EQ(*domain, requested);
      ASSERT_TRUE(swapchain_.destroy(std::chrono::seconds(5)).has_value());
    } else {
      EXPECT_EQ(created.error().code, ErrorCode::kInvalidArgument) << created.error().message;
    }
  }
  EXPECT_TRUE(any_succeeded);
}

TEST_F(RhiPresentTimingWindowedTest, CalibrationContract) {
  const auto calibrated = device_.calibrated_timestamps();
  if (!caps_.calibrated_timestamps) {
    ASSERT_FALSE(calibrated.has_value());
    EXPECT_EQ(calibrated.error().code, ErrorCode::kNotImplemented);
    return;
  }
  ASSERT_TRUE(calibrated.has_value()) << calibrated.error().message;
  const auto host_now = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch() /
                                              std::chrono::nanoseconds(1));
  EXPECT_LT(calibrated->max_deviation_ns, 1'000'000ULL);  // < 1 ms
#ifndef _WIN32
  // CLOCK_MONOTONIC is steady_clock's clock on glibc: the sample must be "now" within the
  // deviation plus generous slack for the two calls.
  const uint64_t slack_ns = calibrated->max_deviation_ns + 5'000'000ULL;
  EXPECT_LE(calibrated->host_ns, host_now + slack_ns);
  EXPECT_GE(calibrated->host_ns + slack_ns, host_now);
#endif
  const auto later = device_.calibrated_timestamps();
  ASSERT_TRUE(later.has_value()) << later.error().message;
  EXPECT_GE(later->device_ns, calibrated->device_ns);
  EXPECT_GE(later->host_ns, calibrated->host_ns);
  EXPECT_EQ(later->to_host_ns(later->device_ns), later->host_ns);
}

TEST_F(RhiPresentTimingWindowedTest, TaggedPresentsYieldOrderedSamples) {
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(create_swapchain({.present_mode = PresentMode::kFIFO,
                                .image_count = 3,
                                .enable_present_timing = true,
                                .present_timing_queue_size = 32})
                  .has_value());
  const auto domain = swapchain_.present_timing_domain();
  ASSERT_TRUE(domain.has_value()) << domain.error().message;

  // Warm up: the engine only knows the refresh period once frames are on the display (see
  // RefreshMode::kFixed), and a surface with no live scanout never reports one. The warm-up
  // samples are drained and discarded; only the tagged frames below are checked.
  constexpr uint64_t kWarmupFrames = 8;
  for (uint64_t id = 1; id <= kWarmupFrames; ++id) {
    ASSERT_TRUE(present_frame(id).has_value());
    std::array<PresentTimingSample, 16> buffer{};
    ASSERT_TRUE(swapchain_.past_present_timings(buffer, true).has_value());
  }
  const auto refresh = swapchain_.refresh_duration_ns();
  ASSERT_TRUE(refresh.has_value()) << refresh.error().message;
  const bool refresh_known = *refresh != 0;
  if (refresh_known) {
    EXPECT_GE(*refresh, 1'000'000ULL);    // >= 1 ms (<= 1000 Hz)
    EXPECT_LE(*refresh, 100'000'000ULL);  // <= 100 ms (>= 10 Hz)
  }
  if (caps_.present_completion) {
    EXPECT_TRUE(swapchain_.wait_for_present(kWarmupFrames, std::chrono::seconds(2)).has_value());
  }
  for (;;) {
    std::array<PresentTimingSample, 16> warmup{};
    const auto drained = swapchain_.past_present_timings(warmup, true);
    ASSERT_TRUE(drained.has_value()) << drained.error().message;
    if (*drained == 0) {
      break;
    }
  }

  constexpr uint64_t kFrames = 60;
  std::vector<PresentTimingSample> samples;
  std::array<PresentTimingSample, 16> buffer{};
  for (uint64_t id = kWarmupFrames + 1; id <= kWarmupFrames + kFrames; ++id) {
    ASSERT_TRUE(present_frame(id).has_value());
    auto drained = swapchain_.past_present_timings(buffer);
    ASSERT_TRUE(drained.has_value()) << drained.error().message;
    samples.insert(samples.end(), buffer.begin(), buffer.begin() + *drained);
  }
  if (caps_.present_completion) {
    EXPECT_TRUE(
        swapchain_.wait_for_present(kWarmupFrames + kFrames, std::chrono::seconds(2)).has_value());
  }
  drain_into(samples, kFrames, std::chrono::seconds(2));

  // A warm-up sample the engine reported late is not this test's concern.
  std::erase_if(samples, [](const PresentTimingSample& sample) {
    return sample.present_id <= kWarmupFrames;
  });
  // past_present_timings() drops reports the engine stamps on a clock this API cannot
  // represent (see PresentTimingSample::time_domain), so the number of samples that arrive is
  // not a contract: zero is legal. Every sample that does arrive is checked below.
  const uint32_t dropped =
      vulkan::swapchain_fault_injection(swapchain_).dropped_unrepresentable_reports;
  ASSERT_GE(samples.size() + dropped, 50U)
      << "tagged presents lost their reports: " << samples.size() << " delivered, " << dropped
      << " dropped for an unrepresentable clock";
  if (samples.empty()) {
    GTEST_SKIP() << "every report was on a clock this API cannot represent (" << dropped
                 << " dropped); nothing to compare";
  }
  // Per-present clock: the engine may fall back from the configured domain for an individual
  // present, and a complete report may still leave a supported stage untimed (optional empty).
  // Both are legal, so timestamps are pooled only across samples on one comparable clock
  // (same domain and, for opaque domains, the same id) and only where the stage was timed.
  std::vector<std::pair<uint64_t, uint64_t>> first_pixel_out;  // (present_id, time)
  std::optional<std::pair<TimeDomain, uint64_t>> pacing_clock;
  size_t fallback_domain = 0;
  size_t untimed_first_pixel_out = 0;
  for (size_t i = 0; i < samples.size(); ++i) {
    const auto& sample = samples[i];
    EXPECT_TRUE(sample.complete);
    if (sample.time_domain != *domain) {
      ++fallback_domain;
    }
    if (i > 0) {
      EXPECT_GT(sample.present_id, samples[i - 1].present_id);
    }
    if (surface_caps_.present_stages.contains(PresentStage::kFirstPixelOut)) {
      if (!sample.first_pixel_out_ns.has_value()) {
        ++untimed_first_pixel_out;
        continue;
      }
      const std::pair<TimeDomain, uint64_t> clock{sample.time_domain, sample.time_domain_id};
      if (!pacing_clock) {
        pacing_clock = clock;
      }
      if (pacing_clock == clock) {
        first_pixel_out.emplace_back(sample.present_id, *sample.first_pixel_out_ns);
      }
    }
  }
  if (fallback_domain > 0 || untimed_first_pixel_out > 0) {
    std::cout << "[  NOTE    ] " << fallback_domain << " sample(s) on a fallback clock, "
              << untimed_first_pixel_out << " without a first-pixel-out time (both legal)\n";
  }
  if (!refresh_known) {
    // The ordering checks above still count (a failed EXPECT keeps the test failed); only
    // the refresh-dependent pacing check is unreachable here.
    GTEST_SKIP() << "presentation engine did not report a refresh period (0 ns), so the "
                    "first-pixel-out pacing check is unreachable; ordering checks passed";
  }
  // FIFO with a full queue presents once per refresh, so the first-pixel-out spacing between
  // *consecutive* presents is the refresh period. A sample dropped above (fallback clock or
  // untimed stage) leaves a gap of two or more refreshes between its neighbours, so only pairs
  // whose present ids are adjacent are compared.
  std::vector<uint64_t> deltas;
  for (size_t i = 1; i < first_pixel_out.size(); ++i) {
    if (first_pixel_out[i].first == first_pixel_out[i - 1].first + 1) {
      deltas.push_back(first_pixel_out[i].second - first_pixel_out[i - 1].second);
    }
  }
  if (window_ && deltas.size() < 2 &&
      surface_caps_.present_stages.contains(PresentStage::kFirstPixelOut)) {
    GTEST_SKIP() << "fewer than two consecutive-present first-pixel-out intervals on one "
                    "comparable clock; the pacing check is unreachable; ordering checks passed";
  }
  if (window_ && deltas.size() >= 2) {
    // The median tolerates the occasional compositor hiccup. Windowed only: a direct-display
    // surface's pacing right after its modeset is not characterized.
    std::sort(deltas.begin(), deltas.end());
    const double median = static_cast<double>(deltas[deltas.size() / 2]);
    EXPECT_NEAR(median, static_cast<double>(*refresh), 0.25 * static_cast<double>(*refresh));
  }
}

// refresh_mode() must agree with refresh_duration_ns(): a fixed or variable verdict comes
// with a plausible period; only kUnknown may accompany a zero period. A composited window is
// expected to report fixed refresh or nothing; variable is tolerated for exotic setups.
TEST_F(RhiPresentTimingWindowedTest, RefreshModeIsConsistentWithRefreshDuration) {
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(
      create_swapchain(
          {.present_mode = PresentMode::kFIFO, .image_count = 3, .enable_present_timing = true})
          .has_value());
  for (uint64_t id = 1; id <= 8; ++id) {
    ASSERT_TRUE(present_frame(id).has_value());
    std::array<PresentTimingSample, 8> buffer{};
    ASSERT_TRUE(swapchain_.past_present_timings(buffer, true).has_value());
  }
  const auto mode = swapchain_.refresh_mode();
  ASSERT_TRUE(mode.has_value()) << mode.error().message;
  const auto refresh = swapchain_.refresh_duration_ns();
  ASSERT_TRUE(refresh.has_value()) << refresh.error().message;
  if (*mode == RefreshMode::kUnknown) {
    GTEST_SKIP() << "presentation engine did not report a refresh mode (refresh " << *refresh
                 << " ns)";
  }
  EXPECT_GE(*refresh, 1'000'000ULL);    // >= 1 ms (<= 1000 Hz)
  EXPECT_LE(*refresh, 100'000'000ULL);  // <= 100 ms (>= 10 Hz)
  EXPECT_TRUE(*mode == RefreshMode::kFixed || *mode == RefreshMode::kVariable);
}

TEST_F(RhiPresentTimingWindowedTest, UntaggedPresentsYieldNoSamples) {
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(
      create_swapchain(
          {.present_mode = PresentMode::kFIFO, .image_count = 3, .enable_present_timing = true})
          .has_value());
  for (int i = 0; i < 20; ++i) {
    ASSERT_TRUE(present_frame(0).has_value());
  }
  ASSERT_TRUE(device_.wait_idle().has_value());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  std::array<PresentTimingSample, 16> buffer{};
  const auto drained = swapchain_.past_present_timings(buffer);
  ASSERT_TRUE(drained.has_value()) << drained.error().message;
  EXPECT_EQ(*drained, 0U);
}

TEST_F(RhiPresentTimingWindowedTest, QueueFullBackpressuresWithoutConsuming) {
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(create_swapchain({.present_mode = PresentMode::kFIFO,
                                .image_count = 3,
                                .enable_present_timing = true,
                                .present_timing_queue_size = 2})
                  .has_value());
  ASSERT_TRUE(present_frame(1).has_value());
  ASSERT_TRUE(present_frame(2).has_value());

  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  PresentConsumed consumed{};
  const auto rejected = swapchain_.present(std::move(*token), SubmitSyncToken{}, 3, &consumed);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, ErrorCode::kBackpressured);
  EXPECT_FALSE(consumed.image);
  EXPECT_FALSE(consumed.id);

  // Draining a completed report frees its queue slot (whether or not the report was
  // representable and returned as a sample); the same token can then be presented.
  const auto retried =
      retry_present_after_drain(std::move(*token), SubmitSyncToken{}, 3, &consumed);
  ASSERT_TRUE(retried.has_value()) << retried.error().message;
  EXPECT_TRUE(consumed.image);
  EXPECT_TRUE(consumed.id);
}

TEST_F(RhiPresentTimingWindowedTest, PartialPollKeepsSlotOccupied) {
  // Regression: draining under allow_partial used to free a local queue slot for every
  // returned result, including incomplete ones the engine still holds. The local guard then
  // admitted a present the engine rejects with VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT.
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(create_swapchain({.present_mode = PresentMode::kFIFO,
                                .image_count = 3,
                                .enable_present_timing = true,
                                .present_timing_queue_size = 2})
                  .has_value());
  ASSERT_TRUE(present_frame(1).has_value());
  ASSERT_TRUE(present_frame(2).has_value());

  // Poll partial results right away; keep going until an incomplete report shows up or the
  // engine has completed everything. Three outcomes are legitimate on a conforming engine:
  // both reports partial (no slot freed), one complete + one partial (exactly one slot freed),
  // or both complete (this driver never reports partials here: nothing to test, skip).
  std::array<PresentTimingSample, 4> buffer{};
  uint32_t incomplete = 0;
  uint32_t complete = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (incomplete == 0 && complete < 2 && std::chrono::steady_clock::now() < deadline) {
    const auto drained = swapchain_.past_present_timings(buffer, /*allow_partial=*/true);
    ASSERT_TRUE(drained.has_value()) << drained.error().message;
    for (uint32_t i = 0; i < *drained; ++i) {
      (buffer[i].complete ? complete : incomplete)++;
    }
    if (*drained == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  if (incomplete == 0) {
    GTEST_SKIP() << "the presentation engine only ever returned complete reports here";
  }
  ASSERT_LE(complete, 1U) << "a partial report cannot coexist with two complete ones";

  // Every *complete* report freed exactly one slot: refill those (each present is admitted),
  // so that afterwards the only occupied slots are the incomplete reports' own.
  uint64_t next_id = 3;
  for (uint32_t i = 0; i < complete; ++i) {
    const auto refilled = present_frame(next_id++);
    ASSERT_TRUE(refilled.has_value())
        << "a complete report frees its slot: " << refilled.error().message;
  }

  // The incomplete report still occupies its slot (only complete reports free one), so the
  // next tagged present must be refused pre-queue, with nothing consumed.
  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  PresentConsumed consumed{};
  const auto rejected =
      swapchain_.present(std::move(*token), SubmitSyncToken{}, next_id, &consumed);
  ASSERT_FALSE(rejected.has_value()) << "the incomplete report must not have freed a slot";
  EXPECT_EQ(rejected.error().code, ErrorCode::kBackpressured);
  EXPECT_FALSE(consumed.image);
  EXPECT_FALSE(consumed.id);

  // Once the engine completes a report a slot frees up and the same token presents.
  const auto retried =
      retry_present_after_drain(std::move(*token), SubmitSyncToken{}, next_id, &consumed);
  ASSERT_TRUE(retried.has_value())
      << "no report completed within the deadline: " << retried.error().message;
  EXPECT_TRUE(consumed.image);
  EXPECT_TRUE(consumed.id);
}

TEST_F(RhiPresentTimingWindowedTest, RejectedPresentFenceIsRetiredByResize) {
  // A present rejected with VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT is enqueued (the spec
  // lists it with OUT_OF_DATE), so its present fence must stay in the retirement proof: in
  // flight on the current generation, then carried by the generation resize() retires, then
  // waited on by destroy(). The rejection is the *last* present before the resize — no later
  // present may be needed to make teardown safe.
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(create_swapchain({.present_mode = PresentMode::kFIFO,
                                .image_count = 3,
                                .enable_present_timing = true,
                                .present_timing_queue_size = 2})
                  .has_value());
  ASSERT_TRUE(present_frame(1).has_value());
  EXPECT_EQ(vulkan::swapchain_fence_counts(swapchain_).pending, 1U);

  vulkan::swapchain_fault_injection(swapchain_).override_next_present_result =
      vulkan::kVkErrorPresentTimingQueueFull;
  const auto rejected = present_frame(2);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, ErrorCode::kFrameDropped) << rejected.error().message;
  // Frame 1's fence may already have signaled and been recycled by this present, so the
  // count is 1 or 2 — but the rejected present's own fence must be among the pending ones.
  const auto after_reject = vulkan::swapchain_fence_counts(swapchain_);
  EXPECT_GE(after_reject.pending, 1U)
      << "the rejected present's fence must be tracked like every other present's";
  EXPECT_EQ(after_reject.pending_timing_rejected, 1U);
  EXPECT_LE(after_reject.pending_timing_rejected, after_reject.pending);

  // Resize immediately: the retired generation owns the fences until they signal (or has
  // been reaped already because they all did — never dropped).
  ASSERT_TRUE(swapchain_.resize({400, 300}).has_value());
  const auto after_resize = vulkan::swapchain_fence_counts(swapchain_);
  EXPECT_EQ(after_resize.pending, 0U);
  EXPECT_EQ(after_resize.pending_timing_rejected, 0U);
  if (after_resize.retired_generations != 0) {
    EXPECT_EQ(after_resize.retired_generations, 1U);
    EXPECT_EQ(after_resize.retired_fences, after_reject.pending)
        << "a fence left the proof on retirement";
    EXPECT_EQ(after_resize.retired_timing_rejected, 1U)
        << "the rejected present's fence left the proof on retirement";
  }

  const auto destroyed = swapchain_.destroy(std::chrono::seconds(5));
  EXPECT_TRUE(destroyed.has_value()) << destroyed.error().message;
  EXPECT_FALSE(vulkan::device_resources_leaked(device_));
}

TEST_F(RhiPresentTimingWindowedTest, RejectedPresentThenZeroTimeoutDestroy) {
  // Same rejection as the final operation, then an immediate zero-timeout destroy(): the
  // only admissible outcomes are success (every fence, including the rejected present's,
  // already signaled) or kBackpressured with the swapchain intact — never a teardown that
  // skipped the rejected present's fence. A retry with a real timeout then succeeds.
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(create_swapchain({.present_mode = PresentMode::kFIFO,
                                .image_count = 3,
                                .enable_present_timing = true,
                                .present_timing_queue_size = 2})
                  .has_value());
  ASSERT_TRUE(present_frame(1).has_value());
  vulkan::swapchain_fault_injection(swapchain_).override_next_present_result =
      vulkan::kVkErrorPresentTimingQueueFull;
  const auto rejected = present_frame(2);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, ErrorCode::kFrameDropped) << rejected.error().message;
  const auto after_reject = vulkan::swapchain_fence_counts(swapchain_);
  EXPECT_GE(after_reject.pending, 1U);
  EXPECT_EQ(after_reject.pending_timing_rejected, 1U);

  auto destroyed = swapchain_.destroy(std::chrono::nanoseconds{0});
  if (!destroyed.has_value()) {
    EXPECT_EQ(destroyed.error().code, ErrorCode::kBackpressured) << destroyed.error().message;
    EXPECT_TRUE(swapchain_) << "a backpressured destroy() leaves the swapchain intact";
    EXPECT_EQ(vulkan::swapchain_fence_counts(swapchain_).pending, after_reject.pending);
    destroyed = swapchain_.destroy(std::chrono::seconds(5));
  }
  EXPECT_TRUE(destroyed.has_value()) << destroyed.error().message;
  EXPECT_FALSE(vulkan::device_resources_leaked(device_));
}

TEST_F(RhiPresentTimingWindowedTest, TimingPropertiesBeforeFirstPresentAreNotAnError) {
  // Vulkan lets vkGetSwapchainTimingPropertiesEXT return VK_NOT_READY (a success code) until
  // the engine has determined the refresh cycle — on some platforms not before the first
  // present. That is the documented kUnknown / 0 ns state, never kVulkanError.
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(
      create_swapchain(
          {.present_mode = PresentMode::kFIFO, .image_count = 3, .enable_present_timing = true})
          .has_value());
  const auto mode = swapchain_.refresh_mode();
  ASSERT_TRUE(mode.has_value()) << mode.error().message;
  const auto refresh = swapchain_.refresh_duration_ns();
  ASSERT_TRUE(refresh.has_value()) << refresh.error().message;
  // kUnknown says nothing about the period: the engine may know the refresh duration before it
  // can tell fixed from variable refresh (the two are independent in VK_EXT_present_timing).
  if (*mode != RefreshMode::kUnknown) {
    EXPECT_GT(*refresh, 0U) << "a fixed/variable verdict comes with a period";
  }
}

TEST_F(RhiPresentTimingWindowedTest, KnownPeriodWithUnknownModeIsValid) {
  // Injected refreshInterval = 0 with a real (or, failing that, injected) nonzero
  // refreshDuration: a legal engine report — it knows the period but not yet whether the
  // display runs fixed or variable refresh. Both accessors succeed, the mode is kUnknown and the
  // period is reported, not zeroed.
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(
      create_swapchain(
          {.present_mode = PresentMode::kFIFO, .image_count = 3, .enable_present_timing = true})
          .has_value());
  for (uint64_t id = 1; id <= 3; ++id) {
    ASSERT_TRUE(present_frame(id).has_value());
  }

  vulkan::swapchain_fault_injection(swapchain_).override_next_timing_unknown_refresh_mode = true;
  const auto mode = swapchain_.refresh_mode();
  ASSERT_TRUE(mode.has_value()) << mode.error().message;
  EXPECT_EQ(*mode, RefreshMode::kUnknown);

  vulkan::swapchain_fault_injection(swapchain_).override_next_timing_unknown_refresh_mode = true;
  const auto refresh = swapchain_.refresh_duration_ns();
  ASSERT_TRUE(refresh.has_value()) << refresh.error().message;
  EXPECT_GT(*refresh, 0U) << "an unknown mode must not zero a known period";

  // The knob is one-shot: the next query reports whatever the engine really says, which may
  // legitimately still be "not ready" (mode unknown, period 0) this early.
  EXPECT_FALSE(
      vulkan::swapchain_fault_injection(swapchain_).override_next_timing_unknown_refresh_mode);
  const auto real = swapchain_.refresh_mode();
  ASSERT_TRUE(real.has_value()) << real.error().message;
}

TEST_F(RhiPresentTimingWindowedTest, NotReadyTimingPropertiesReportUnknown) {
  // Injected VK_NOT_READY (the real query ran; only its result is rewritten): both accessors
  // report the unknown state as a success, and the very next query is unaffected. Real
  // Vulkan failures keep their kVulkanError mapping (unchanged, not exercised here).
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(
      create_swapchain(
          {.present_mode = PresentMode::kFIFO, .image_count = 3, .enable_present_timing = true})
          .has_value());
  for (uint64_t id = 1; id <= 3; ++id) {
    ASSERT_TRUE(present_frame(id).has_value());
  }

  vulkan::swapchain_fault_injection(swapchain_).override_next_timing_properties_result =
      vulkan::kVkNotReady;
  const auto mode = swapchain_.refresh_mode();
  ASSERT_TRUE(mode.has_value()) << mode.error().message;
  EXPECT_EQ(*mode, RefreshMode::kUnknown);

  vulkan::swapchain_fault_injection(swapchain_).override_next_timing_properties_result =
      vulkan::kVkNotReady;
  const auto refresh = swapchain_.refresh_duration_ns();
  ASSERT_TRUE(refresh.has_value()) << refresh.error().message;
  EXPECT_EQ(*refresh, 0U);

  // The knob is one-shot: the following query reports whatever the engine really says.
  const auto real = swapchain_.refresh_duration_ns();
  ASSERT_TRUE(real.has_value()) << real.error().message;
}

TEST_F(RhiPresentTimingWindowedTest, UnrepresentableClockReportsAreDroppedButFreeTheirSlot) {
  // The documented drop contract, deterministically: a completed report the engine stamps on
  // a clock this API cannot represent yields no sample, yet frees its timing-queue slot (the
  // credit follows the native report's completeness, not what was written). Then a control
  // present with no injection is polled normally.
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(create_swapchain({.present_mode = PresentMode::kFIFO,
                                .image_count = 3,
                                .enable_present_timing = true,
                                .present_timing_queue_size = 1})
                  .has_value());
  ASSERT_TRUE(present_frame(1).has_value());
  if (caps_.present_completion) {
    EXPECT_TRUE(swapchain_.wait_for_present(1, std::chrono::seconds(2)).has_value());
  }

  // Poll until native results come back (the override is consumed exactly then); with
  // allow_partial=false those are complete reports, each of which must have freed its slot.
  auto& injection = vulkan::swapchain_fault_injection(swapchain_);
  const uint32_t dropped_before = injection.dropped_unrepresentable_reports;
  std::array<PresentTimingSample, 4> buffer{};
  uint32_t written_total = 0;
  injection.override_next_timing_report_domain = vulkan::kVkTimeDomainClockMonotonicRaw;
  for (const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
       injection.override_next_timing_report_domain >= 0 &&
       std::chrono::steady_clock::now() < deadline;) {
    const auto drained = swapchain_.past_present_timings(buffer);
    ASSERT_TRUE(drained.has_value()) << drained.error().message;
    written_total += *drained;
    if (*drained == 0 && injection.override_next_timing_report_domain >= 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  if (injection.override_next_timing_report_domain >= 0) {
    injection.override_next_timing_report_domain = -1;
    GTEST_SKIP() << "the presentation engine completed no timing report within the deadline";
  }
  EXPECT_EQ(written_total, 0U) << "a report on an unrepresentable clock must not be a sample";
  EXPECT_GE(injection.dropped_unrepresentable_reports, dropped_before + 1)
      << "the dropped report must be counted";

  // The completed (dropped) report freed its slot: at queue size one the next tagged present
  // is admitted immediately. A refusal here is the accounting regression this test guards.
  AcquireSyncToken token{};
  SubmitSyncToken submitted{};
  ASSERT_TRUE(acquire_and_submit(token, submitted).has_value());
  PresentConsumed consumed{};
  const auto next = swapchain_.present(std::move(token), submitted, 2, &consumed);
  ASSERT_TRUE(next.has_value()) << "a completed report must free its timing-queue slot: "
                                << next.error().message;
  EXPECT_TRUE(consumed.image);
  EXPECT_TRUE(consumed.id);

  // Control: with no injection the same present's report, once complete, is delivered — or
  // legitimately dropped if this engine really stamps on an unrepresentable clock.
  std::vector<PresentTimingSample> samples;
  drain_into(samples, 1, std::chrono::seconds(2));
  for (const PresentTimingSample& sample : samples) {
    EXPECT_EQ(sample.present_id, 2U);
    EXPECT_TRUE(sample.complete);
  }
}

TEST_F(RhiPresentTimingWindowedTest, EnqueuedQueueFullIsConsumedAndDropped) {
  // The presentation engine's own queue-full rejection is an enqueued present (the spec lists
  // it with OUT_OF_DATE: the semaphore wait executes, the image is released). present() must
  // commit that state and report kFrameDropped (fresh acquire), never a same-token retry.
  //
  // Two rejections are exercised. The first is injected (override: the real present executed
  // and did reserve an engine slot, only the result is rewritten), which deliberately leaves
  // the local count one below the engine's. The second is therefore the engine's *native*
  // rejection, reached through the local guard exactly as a partial-poll desync would reach it.
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(create_swapchain({.present_mode = PresentMode::kFIFO,
                                .image_count = 3,
                                .enable_present_timing = true,
                                .present_timing_queue_size = 2})
                  .has_value());
  ASSERT_TRUE(present_frame(1).has_value());

  vulkan::swapchain_fault_injection(swapchain_).override_next_present_result =
      vulkan::kVkErrorPresentTimingQueueFull;
  PresentConsumed consumed{};
  const auto injected = present_frame(2, &consumed);
  ASSERT_FALSE(injected.has_value());
  EXPECT_EQ(injected.error().code, ErrorCode::kFrameDropped) << injected.error().message;
  EXPECT_TRUE(consumed.image) << "an enqueued rejection consumes the image";
  EXPECT_TRUE(consumed.id) << "the tagged id sequence advanced";

  // Local count 1, engine 2 (its queue is full): the guard admits this present and the engine
  // rejects it for real. A driver with slack beyond the requested queue size may accept it
  // instead; the native leg is then unobservable here.
  consumed = {};
  const auto native = present_frame(3, &consumed);
  bool native_rejected = false;
  if (!native.has_value()) {
    native_rejected = true;
    EXPECT_EQ(native.error().code, ErrorCode::kFrameDropped) << native.error().message;
    EXPECT_TRUE(consumed.image) << "the engine's rejection consumed the image";
    EXPECT_TRUE(consumed.id);
  }

  // Recovery: drain what the engine holds, then a fresh acquire/present continues past the
  // dropped tags (id sequence advanced, gates reset).
  // Frames 1 and 2 really presented, so their reports complete and free their slots; the
  // samples themselves may be dropped for an unrepresentable clock, so the freed slots are
  // proven by the next present being admitted (retried while the engine completes its
  // reports, with a fresh acquire whenever the engine drops an attempt), not by a sample count.
  PresentConsumed after{};
  unsigned attempts4 = 0;
  std::chrono::milliseconds waited4{};
  const auto recovered =
      present_fresh_after_drain(4, &after, std::chrono::seconds(2), &attempts4, &waited4);
  ASSERT_TRUE(recovered.has_value()) << recovered.error().message << " (after " << attempts4
                                     << " attempts over " << waited4.count() << " ms of draining)";
  EXPECT_TRUE(after.image);
  EXPECT_TRUE(after.id);

  // Fence retirement: teardown must complete (no wedged present fence, nothing leaked).
  const auto destroyed = swapchain_.destroy(std::chrono::seconds(5));
  EXPECT_TRUE(destroyed.has_value())
      << (native_rejected ? "after the engine's own rejection: " : "") << destroyed.error().message;
  EXPECT_FALSE(vulkan::device_resources_leaked(device_));
  if (!native_rejected) {
    GTEST_SKIP() << "the presentation engine accepted a present beyond its advertised timing "
                    "queue; only the injected rejection was verified";
  }
}

TEST_F(RhiPresentTimingWindowedTest, ResizeReappliesTimingSetup) {
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  ASSERT_TRUE(
      create_swapchain(
          {.present_mode = PresentMode::kFIFO, .image_count = 3, .enable_present_timing = true})
          .has_value());
  const auto domain_before = swapchain_.present_timing_domain();
  ASSERT_TRUE(domain_before.has_value());
  const auto generation_before = swapchain_.generation();
  ASSERT_TRUE(generation_before.has_value());

  ASSERT_TRUE(swapchain_.resize({400, 300}).has_value());
  const auto generation_after = swapchain_.generation();
  ASSERT_TRUE(generation_after.has_value());
  EXPECT_GT(*generation_after, *generation_before);
  const auto domain_after = swapchain_.present_timing_domain();
  ASSERT_TRUE(domain_after.has_value());
  EXPECT_EQ(*domain_after, *domain_before);
  EXPECT_TRUE(swapchain_.refresh_duration_ns().has_value());

  for (uint64_t id = 1; id <= 5; ++id) {
    ASSERT_TRUE(present_frame(id).has_value());
  }
  // Timing is still armed on the recreated swapchain: a report for one of the tagged presents
  // either arrives as a sample or is provably dropped for an unrepresentable clock. Neither
  // happening within the window means the recreated swapchain is not producing reports.
  std::vector<PresentTimingSample> samples;
  drain_into(samples, 1, std::chrono::seconds(2));
  EXPECT_TRUE(!samples.empty() ||
              vulkan::swapchain_fault_injection(swapchain_).dropped_unrepresentable_reports > 0)
      << "no timing report (delivered or dropped) from the recreated swapchain";
}

// The headline contract: a GPU timestamp written at the end of the frame's submit and the
// frame's first-pixel-out time are on the same clock, and scanout follows render completion
// within a few refresh periods.
TEST_F(RhiPresentTimingWindowedTest, FirstPixelOutFollowsGpuEndOnDeviceClock) {
  if (!timing_supported()) {
    GTEST_SKIP() << "present timing unsupported on this device/surface";
  }
  if (!surface_caps_.present_stages.contains(PresentStage::kFirstPixelOut)) {
    GTEST_SKIP() << "this surface does not report first-pixel-out";
  }
  ASSERT_TRUE(
      create_swapchain(
          {.present_mode = PresentMode::kFIFO, .image_count = 3, .enable_present_timing = true})
          .has_value());
  const auto domain = swapchain_.present_timing_domain();
  ASSERT_TRUE(domain.has_value());
  if (*domain != TimeDomain::kDevice) {
    GTEST_SKIP() << "present timing is not reported on the device clock here";
  }
  auto query_pool = device_.create_query_pool({.timestamp_count = 2});
  ASSERT_TRUE(query_pool.has_value()) << query_pool.error().message;
  auto command_list = device_.create_command_list({});
  ASSERT_TRUE(command_list.has_value()) << command_list.error().message;

  auto token = swapchain_.acquire(std::chrono::seconds(5));
  ASSERT_TRUE(token.has_value()) << token.error().message;
  ASSERT_TRUE(command_list->begin().has_value());
  ASSERT_TRUE(command_list->reset_queries(*query_pool, 0, 2).has_value());
  ASSERT_TRUE(command_list->write_timestamp(*query_pool, 1, TimestampLocation::kEnd).has_value());
  ASSERT_TRUE(command_list->resolve_queries(*query_pool, 0, 2).has_value());
  ASSERT_TRUE(command_list->end().has_value());
  auto submitted = device_.submit({
      .command_lists = {&*command_list, 1},
      .acquire_gates = {&token->acquire_gate, 1},
      .present_gates = {&token->present_gate, 1},
  });
  ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
  ASSERT_TRUE(swapchain_.present(std::move(*token), *submitted, 1).has_value());
  ASSERT_TRUE(device_.wait_on_submit(*submitted, std::chrono::seconds(5)).has_value());
  std::array<uint64_t, 2> stamps{};
  ASSERT_TRUE(query_pool->results_ns(1, 1, std::span<uint64_t>(stamps).subspan(1)).has_value());

  std::vector<PresentTimingSample> samples;
  drain_into(samples, 1, std::chrono::seconds(2));
  if (samples.empty()) {
    ASSERT_GT(vulkan::swapchain_fault_injection(swapchain_).dropped_unrepresentable_reports, 0U)
        << "no timing report for the present arrived within the polling window";
    GTEST_SKIP() << "the report was stamped on a clock this API cannot represent; nothing to "
                    "compare";
  }
  ASSERT_EQ(samples[0].present_id, 1U);
  EXPECT_TRUE(samples[0].complete);
  // Comparing with the GPU timestamp needs this present's first-pixel-out on the device clock.
  // The engine may legally fall back to another clock for an individual present, or leave a
  // supported stage untimed; neither is a defect, so the latency check is skipped, not failed.
  if (samples[0].time_domain != TimeDomain::kDevice) {
    GTEST_SKIP() << "the engine stamped this present on a fallback clock ("
                 << static_cast<int>(samples[0].time_domain)
                 << "); no device-clock latency to compare";
  }
  const auto first_pixel_out_ns = samples[0].first_pixel_out_ns;
  if (!first_pixel_out_ns.has_value()) {
    GTEST_SKIP() << "the engine reported no first-pixel-out time for this present";
  }
  EXPECT_GE(*first_pixel_out_ns, stamps[1]);
  // The period-based upper bound needs a known period; before the first present the engine
  // may legitimately report none (VK_NOT_READY), so query it after presenting and skip the
  // bound (not the test) while it is still unknown.
  const auto refresh = swapchain_.refresh_duration_ns();
  ASSERT_TRUE(refresh.has_value()) << refresh.error().message;
  if (window_ && *refresh > 0) {
    // Windowed only: a direct-display surface's first scan-out after its modeset can trail
    // the GPU by far more than a few refreshes.
    EXPECT_LT(*first_pixel_out_ns - stamps[1], 4 * *refresh);
  }

  ASSERT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(command_list->destroy(std::chrono::seconds(5)).has_value());
  EXPECT_TRUE(query_pool->destroy().has_value());
}

}  // namespace
}  // namespace holoviz
