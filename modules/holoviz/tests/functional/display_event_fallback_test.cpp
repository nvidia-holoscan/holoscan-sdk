/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file display_event_fallback_test.cpp
 *
 * Regression test for NVBug 6194895:
 *   On AGX Thor Jedha, `VK_EXT_display_control` is advertised as available but
 *   `vkRegisterDisplayEventEXT` returns `VK_ERROR_UNKNOWN` at runtime. The
 *   resulting `vk::UnknownError` previously escaped the worker thread inside
 *   `FirstPixelOutCondition` and triggered `std::terminate()` ("Aborted (core
 *   dumped)").
 *
 * The fix translates any `vk::Error` thrown from
 * `Vulkan::Impl::wait_for_display_event` into a `std::runtime_error` whose
 * message identifies `VK_EXT_display_control`, so callers (in particular the
 * condition's worker thread) can catch and degrade gracefully.
 *
 * Test strategy
 * -------------
 * This test exercises the public `viz::WaitForDisplayEvent` contract. The
 * full Thor-specific `ErrorUnknown` path requires hardware and cannot be
 * reproduced on headless CI — headless builds will hit the earlier
 * "no display" branch of `Vulkan::Impl::wait_for_display_event`. Either way,
 * the contract under test is the same: when the call fails, it must throw a
 * `std::runtime_error` (i.e. no raw `vk::Error` leaks out), with a
 * human-readable message.
 *
 * On extension-capable hardware with a real display the call may instead
 * return cleanly (timeout) — the test is permissive on the happy path.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include <holoviz/holoviz.hpp>

#include "test_fixture.hpp"

namespace viz = holoscan::viz;

class DisplayEventFallback : public TestHeadless {};

/**
 * Asserts the per-platform contract of `viz::WaitForDisplayEvent`:
 *
 *   - On the happy path (extension available, display present) the call may
 *     return cleanly (timeout). No exception is required.
 *   - On any failure path (extension missing, extension advertised but broken
 *     at runtime, headless without a display) the call must throw a
 *     `std::runtime_error`. It must NOT leak a raw `vk::Error` to the caller.
 *
 * The strictest assertion this test can make on headless CI is the latter:
 * any thrown exception must be a `std::runtime_error` (the base type used by
 * `Vulkan::Impl::wait_for_display_event` after translating vk-hpp errors).
 */
TEST_F(DisplayEventFallback, NoVulkanExceptionLeaksToCallers) {
  try {
    // Default timeout matches the FirstPixelOutCondition worker (~10Hz pace).
    (void)viz::WaitForDisplayEvent(viz::DisplayEventType::FIRST_PIXEL_OUT,
                                   100'000'000 /* 100 ms */);
    // Happy path: returned cleanly. Nothing else to assert here; the test
    // exists to guard the failure contract.
  } catch (const std::runtime_error& e) {
    // Failure path: must surface an actionable message. Two possible substrings
    // depending on the CI platform — either is acceptable, what matters is that
    // we did NOT see a raw vk::Error escape:
    //   - Headless CI: "There is no display..." (display_ == nullptr branch).
    //   - Thor / extension-broken hardware: "VK_EXT_display_control reported as
    //     available but failed at runtime ...".
    //   - Hardware without the extension: "VK_EXT_display_control is not available".
    const std::string msg(e.what());
    const bool mentions_extension = msg.find("VK_EXT_display_control") != std::string::npos;
    const bool mentions_no_display = msg.find("no display") != std::string::npos;
    EXPECT_TRUE(mentions_extension || mentions_no_display)
        << "Expected message to mention VK_EXT_display_control or the absent display, got: " << msg;
  }
}
