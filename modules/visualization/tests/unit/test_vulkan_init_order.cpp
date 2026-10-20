// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Lock-order regression for one-time Vulkan instance initialization. The initializer takes a
// GLFW reference under the window-system lock, and window paths (create_surface(), callbacks
// delivered with that lock held) may be the first to need the instance. If those two orders
// disagree, a first-time initializer on one thread and a window call on another deadlock.
//
// Each case holds the first initializer open through the instance-init test seam
// (vulkan/vk_instance_init_hook.hpp), overlaps a window call that also needs the instance, and
// requires both to finish. Initialization runs once per process, so each case needs a fresh
// process: CTest runs them as separate entries (tests/unit/CMakeLists.txt), and a case that
// finds the instance already initialized skips. A deadlock exits the process instead of
// hanging the run. Windowed: skips without a window system.

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"
#include "vulkan/vk_instance_init_hook.hpp"

namespace holoviz {
namespace {

constexpr auto kTimeout = std::chrono::seconds(10);
/// How long the first initializer stays open after the overlapping call starts, so that call
/// reaches its blocking point (or, with the old lock order, the cycle closes).
constexpr auto kInitHold = std::chrono::milliseconds(200);

struct InitGate {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false;
  bool overlap_started = false;
};
InitGate g_gate;

/// Instance-init hook: report that the first initializer is running, then hold it open until
/// the overlapping call has started.
void hold_first_init() {
  {
    const std::lock_guard<std::mutex> lock(g_gate.mutex);
    g_gate.entered = true;
  }
  g_gate.cv.notify_all();
  {
    std::unique_lock<std::mutex> lock(g_gate.mutex);
    g_gate.cv.wait_for(lock, kTimeout, [] { return g_gate.overlap_started; });
  }
  std::this_thread::sleep_for(kInitHold);
}

template <typename T>
void wait_or_exit(std::future<T>& future, const char* what) {
  if (future.wait_for(kTimeout) != std::future_status::ready) {
    // Deadlocked threads cannot be joined, and a future's destructor would block on them.
    std::fprintf(stderr,
                 "[  FAILED  ] deadlock: %s did not finish within %llds\n",
                 what,
                 static_cast<long long>(kTimeout.count()));
    std::fflush(stderr);
    std::_Exit(1);
  }
}

enum class Overlap { kRan, kSkipFreshProcess, kSkipNoVulkan };

/// Start first-time initialization on one thread (through enumerate_adapters()), and once its
/// initializer is held open run \p overlap_call on another. Both must finish.
Overlap overlap_first_init(const std::function<void()>& overlap_call) {
  vulkan::set_instance_init_hook(&hold_first_init);
  auto init = std::async(std::launch::async, [] { return enumerate_adapters(); });

  bool entered = false;
  {
    std::unique_lock<std::mutex> lock(g_gate.mutex);
    entered = g_gate.cv.wait_for(lock, kTimeout, [] { return g_gate.entered; });
  }
  if (!entered) {
    wait_or_exit(init, "enumerate_adapters()");
    vulkan::set_instance_init_hook(nullptr);
    return Overlap::kSkipFreshProcess;  // the instance existed before this case ran
  }

  auto overlap = std::async(std::launch::async, [&overlap_call] {
    {
      const std::lock_guard<std::mutex> lock(g_gate.mutex);
      g_gate.overlap_started = true;
    }
    g_gate.cv.notify_all();
    overlap_call();
  });

  wait_or_exit(init, "the first initializer (enumerate_adapters())");
  wait_or_exit(overlap, "the overlapping window call");
  vulkan::set_instance_init_hook(nullptr);
  overlap.get();
  return init.get().has_value() ? Overlap::kRan : Overlap::kSkipNoVulkan;
}

TEST(VulkanInitOrder, SurfaceCreationOverlapsFirstInit) {
  auto window = create_window({.title = "vulkan init order surface", .size = {64, 48}});
  if (!window) {
    HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
  }

  expected<Surface> surface = make_unexpected(Error{ErrorCode::kFailure, "not run"});
  switch (overlap_first_init([&] { surface = window->create_surface(); })) {
    case Overlap::kRan:
      break;
    case Overlap::kSkipFreshProcess:
      GTEST_SKIP() << "Vulkan was already initialized; run this case in a fresh process";
    case Overlap::kSkipNoVulkan:
      HOLOVIZ_ENV_SKIP() << "no Vulkan instance on this machine";
  }
  EXPECT_TRUE(surface.has_value()) << surface.error().message;
}

TEST(VulkanInitOrder, CallbackReentryOverlapsFirstInit) {
  auto window = create_window({.title = "vulkan init order callback", .size = {64, 48}});
  if (!window) {
    HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
  }

  // set_event_callbacks() delivers the current framebuffer size synchronously with the
  // window-system lock held, like a poll does; the callback then needs the instance.
  struct CallbackResult {
    bool ran = false;
    bool adapters_ok = false;
  } result;
  expected<void> registered;
  switch (overlap_first_init([&] {
    registered = window->set_event_callbacks({
        .user_data = &result,
        .framebuffer_size =
            [](void* user_data, uint32_t /*width*/, uint32_t /*height*/) {
              auto* r = static_cast<CallbackResult*>(user_data);
              r->ran = true;
              r->adapters_ok = enumerate_adapters().has_value();
            },
    });
  })) {
    case Overlap::kRan:
      break;
    case Overlap::kSkipFreshProcess:
      GTEST_SKIP() << "Vulkan was already initialized; run this case in a fresh process";
    case Overlap::kSkipNoVulkan:
      HOLOVIZ_ENV_SKIP() << "no Vulkan instance on this machine";
  }
  ASSERT_TRUE(registered.has_value()) << registered.error().message;
  EXPECT_TRUE(result.ran);
  EXPECT_TRUE(result.adapters_ok);
}

}  // namespace
}  // namespace holoviz
