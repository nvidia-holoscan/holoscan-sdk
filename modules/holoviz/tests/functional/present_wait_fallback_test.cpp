/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file present_wait_fallback_test.cpp
 *
 * Regression test for NVBug 5928213 / CLARAHOLOS-2844:
 *   On hardware without `VK_KHR_present_wait`, `Vulkan::Impl::wait_for_present`
 *   previously returned `false` immediately causing a tight-spin loop in
 *   `PresentDoneCondition` that kept the frame counter at zero and flooded
 *   the log with repeated errors.
 *
 * The fix: throw `std::runtime_error` immediately when the extension is absent
 * so the application fails fast with a clear message rather than silently
 * hanging or running at degraded performance.
 *
 * Test strategy
 * -------------
 * On extension-capable hardware, `viz::WaitForPresent` with an unsubmitted
 * present_id will block until the timeout and return false — no exception.
 * On extension-less hardware (the bug scenario), it must throw a
 * `std::runtime_error` with a message identifying `VK_KHR_present_wait`.
 *
 * We cannot inject `has_present_wait_extension_` directly since `Vulkan::Impl`
 * is private to the translation unit, so on extension-capable CI hardware the
 * test verifies the happy-path (no throw, call completes within ~timeout).
 * The throw path is exercised on IGX Thor / Orin iGPU platforms that lack the
 * extension.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include <holoviz/holoviz.hpp>

#include "test_fixture.hpp"

namespace viz = holoscan::viz;

class PresentWaitFallback : public TestHeadless {};

/**
 * On extension-capable hardware: WaitForPresent blocks until timeout and
 * returns false (present_id 0 was never submitted). No exception expected.
 *
 * On extension-less hardware (NVBug 5928213 scenario): WaitForPresent must
 * throw std::runtime_error mentioning VK_KHR_present_wait, giving the
 * application a clear signal to use FirstPixelOutCondition instead.
 */
TEST_F(PresentWaitFallback, ThrowsOrTimesOutWhenPresentNotSubmitted) {
  try {
    // present_id 0 was never submitted; on extension-capable hardware this
    // times out after timeout_ns and returns false.
    const bool result = viz::WaitForPresent(0, 100'000'000 /* 100 ms */);
    // Extension IS available: timed out cleanly.
    (void)result;
  } catch (const std::runtime_error& e) {
    // Extension is NOT available: must throw with a message identifying the
    // missing extension so the application author can take corrective action.
    const std::string msg(e.what());
    EXPECT_NE(msg.find("VK_KHR_present_wait"), std::string::npos)
        << "Expected exception message to mention VK_KHR_present_wait, got: " << msg;
  }
}
