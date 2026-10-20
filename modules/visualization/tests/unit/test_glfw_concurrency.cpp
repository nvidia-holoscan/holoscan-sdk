// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Concurrency regression for the GLFW window system: several HolovizOps on a multi-threaded
// scheduler each create a window, poll events every frame, and destroy the window, all from
// different threads. GLFW is not thread-safe, so libholoviz serializes those calls; without
// that, concurrent glfwPollEvents()/glfwCreateWindow()/glfwDestroyWindow() crash or trip the
// sanitizers. Windowed: skips without a window system.
//
// The dispatch-overlap cases pin the read side: glfwPollEvents() dispatches every window's
// events, so one thread's poll runs another window's callbacks. A native close request for
// window B is delivered while thread A polls; B's close callback holds the dispatch open, and
// B's owner queries its window meanwhile. The query must wait for the dispatch to finish.
// That includes the ImGui polling bridge (Renderer::forward_input), which reads B's native
// input state. X11 only (the request goes through the test's own Xlib connection);
// ForwardInput additionally needs a GPU with CUDA.
//
// The install case pins the publication side: Renderer::install_input_callbacks() must
// register its callbacks and publish the state they read as one operation under the
// window-system lock, so another thread's poll cannot dispatch into a half-installed renderer.
// It asserts the exclusion itself rather than a scaled cursor position: on X11 the framebuffer
// and window sizes match, so an unpublished window would not change the delivered point.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

#if defined(HOLOVIZ_TEST_HAVE_X11)
#include <unistd.h>

#include <cuda.h>

#include <holoviz/imgui/renderer.hpp>

#include "image_compare_test_util.hpp"
#include "imgui/renderer_install_hook.hpp"
#include "x11_close_request.hpp"
#endif

namespace holoviz {
namespace {

constexpr int kThreadCount = 2;
constexpr int kCycles = 3;
constexpr int kPollsPerCycle = 200;

TEST(GlfwConcurrency, ConcurrentCreatePollDestroy) {
  {
    auto probe = create_window({.title = "glfw concurrency probe", .size = {64, 48}});
    if (!probe) {
      HOLOVIZ_ENV_SKIP() << "no window system available: " << probe.error().message;
    }
  }

  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (int t = 0; t < kThreadCount; ++t) {
    threads.emplace_back([t, &failures] {
      for (int cycle = 0; cycle < kCycles; ++cycle) {
        auto window =
            create_window({.title = "glfw concurrency " + std::to_string(t), .size = {64, 48}});
        if (!window) {
          failures.fetch_add(1);
          return;
        }
        for (int i = 0; i < kPollsPerCycle; ++i) {
          poll_window_events();
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(failures.load(), 0);
}

#if defined(HOLOVIZ_TEST_HAVE_X11)

/// The window query B's owner runs while A's poll dispatches B's close event.
enum class OwnerQuery {
  kCloseRequested,
  kExtent,
  kWindowSize,
  kMinimized,
  kContentScale,
  kSetInputCallbacks,
  kForwardInput,
};

std::string owner_query_name(const ::testing::TestParamInfo<OwnerQuery>& info) {
  switch (info.param) {
    case OwnerQuery::kCloseRequested:
      return "CloseRequested";
    case OwnerQuery::kExtent:
      return "Extent";
    case OwnerQuery::kWindowSize:
      return "WindowSize";
    case OwnerQuery::kMinimized:
      return "Minimized";
    case OwnerQuery::kContentScale:
      return "ContentScale";
    case OwnerQuery::kSetInputCallbacks:
      return "SetInputCallbacks";
    case OwnerQuery::kForwardInput:
      return "ForwardInput";
  }
  return "Unknown";
}

/// Shared between B's close callback (running on the poller thread) and B's owner.
struct DispatchProbe {
  std::mutex mutex;
  std::condition_variable cv;
  bool in_dispatch = false;
  std::atomic<bool> dispatch_done{false};
};

/// How long B's close callback keeps the dispatch open. An unserialized query returns well
/// within this window, so the outcome does not depend on scheduling luck.
constexpr auto kDispatchHold = std::chrono::milliseconds(200);
constexpr auto kEventTimeout = std::chrono::seconds(5);

/// Device, stream, and ImGui renderer for the ForwardInput case. Declared after the windows
/// so it tears down first; device teardown follows the renderer, as the renderer requires.
struct ImGuiBridge {
  Device device;
  CUstream stream = nullptr;
  imgui::Renderer renderer;

  /// Empty string on success, otherwise why this machine cannot run the case.
  std::string create() {
    device = test_util::try_create_device();
    if (!device) {
      return "no usable GPU/adapter on this machine";
    }
    const auto context = device.cuda_context();
    if (!context || cuCtxSetCurrent(*context) != CUDA_SUCCESS ||
        cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
      return "cannot create a CUDA stream on the device context";
    }
    auto created = imgui::create_renderer(device, {.stream = stream, .frames_in_flight = 2});
    if (!created) {
      return "cannot create the ImGui renderer: " + created.error().message;
    }
    renderer = std::move(*created);
    return {};
  }

  ~ImGuiBridge() {
    if (renderer) {
      EXPECT_TRUE(renderer.destroy().has_value());
    }
    if (device) {
      EXPECT_TRUE(device.wait_idle().has_value());
    }
    if (stream != nullptr) {
      cuStreamSynchronize(stream);
      cuStreamDestroy(stream);
    }
    if (device) {
      EXPECT_TRUE(device.destroy().has_value());
    }
  }
};

class GlfwDispatchOverlap : public ::testing::TestWithParam<OwnerQuery> {};

TEST_P(GlfwDispatchOverlap, OwnerQueryWaitsForOtherThreadsDispatch) {
  const std::string suffix = owner_query_name({GetParam(), 0}) + " " + std::to_string(getpid());
  const std::string title_b = "glfw overlap target " + suffix;

  // Declared before the windows so B's callback target outlives B.
  DispatchProbe probe;

  auto window_a = create_window({.title = "glfw overlap poller " + suffix, .size = {64, 48}});
  if (!window_a) {
    HOLOVIZ_ENV_SKIP() << "no window system available: " << window_a.error().message;
  }
  auto window_b = create_window({.title = title_b, .size = {64, 48}});
  ASSERT_TRUE(window_b.has_value()) << window_b.error().message;

  ImGuiBridge bridge;
  if (GetParam() == OwnerQuery::kForwardInput) {
    const std::string unavailable = bridge.create();
    if (!unavailable.empty()) {
      HOLOVIZ_ENV_SKIP() << unavailable;
    }
  }

  ASSERT_TRUE(window_b
                  ->set_close_callback(&probe,
                                       [](void* user_data) {
                                         auto* p = static_cast<DispatchProbe*>(user_data);
                                         {
                                           const std::lock_guard<std::mutex> lock(p->mutex);
                                           p->in_dispatch = true;
                                         }
                                         p->cv.notify_all();
                                         std::this_thread::sleep_for(kDispatchHold);
                                         p->dispatch_done = true;
                                       })
                  .has_value());

  switch (test_util::send_x11_close_request(title_b, std::chrono::seconds(2))) {
    case test_util::X11CloseRequest::kSent:
      break;
    case test_util::X11CloseRequest::kNoDisplay:
      HOLOVIZ_ENV_SKIP() << "no X11 display reachable";
    case test_util::X11CloseRequest::kWindowNotFound:
      HOLOVIZ_ENV_SKIP() << "window not found on X11 (GLFW may be using native Wayland)";
  }

  // Thread A: the other operator, pumping events for every window. B's owner never polls.
  std::thread poller([&probe] {
    const auto deadline = std::chrono::steady_clock::now() + kEventTimeout;
    while (!probe.dispatch_done && std::chrono::steady_clock::now() < deadline) {
      poll_window_events();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  bool dispatched = false;
  {
    std::unique_lock<std::mutex> lock(probe.mutex);
    dispatched = probe.cv.wait_for(lock, kEventTimeout, [&probe] { return probe.in_dispatch; });
  }
  bool done_when_query_returned = false;
  if (dispatched) {
    switch (GetParam()) {
      case OwnerQuery::kCloseRequested: {
        auto closed = window_b->close_requested();
        done_when_query_returned = probe.dispatch_done;
        // EXPECT, not ASSERT: returning here would skip poller.join().
        EXPECT_TRUE(closed.has_value() && *closed);
        break;
      }
      case OwnerQuery::kExtent:
        (void)window_b->extent();
        done_when_query_returned = probe.dispatch_done;
        break;
      case OwnerQuery::kWindowSize:
        (void)window_b->window_size();
        done_when_query_returned = probe.dispatch_done;
        break;
      case OwnerQuery::kMinimized:
        (void)window_b->minimized();
        done_when_query_returned = probe.dispatch_done;
        break;
      case OwnerQuery::kContentScale:
        (void)window_b->content_scale();
        done_when_query_returned = probe.dispatch_done;
        break;
      case OwnerQuery::kSetInputCallbacks:
        (void)window_b->set_input_callbacks({});
        done_when_query_returned = probe.dispatch_done;
        break;
      case OwnerQuery::kForwardInput: {
        auto forwarded = bridge.renderer.forward_input(*window_b);
        done_when_query_returned = probe.dispatch_done;
        EXPECT_TRUE(forwarded.has_value()) << forwarded.error().message;
        break;
      }
    }
  }
  poller.join();

  ASSERT_TRUE(dispatched) << "the poller thread never dispatched B's close event";
  EXPECT_TRUE(done_when_query_returned)
      << "the owner's query returned while another thread was still dispatching its window's "
         "events";
}

INSTANTIATE_TEST_SUITE_P(Queries, GlfwDispatchOverlap,
                         ::testing::Values(OwnerQuery::kCloseRequested, OwnerQuery::kExtent,
                                           OwnerQuery::kWindowSize, OwnerQuery::kMinimized,
                                           OwnerQuery::kContentScale,
                                           OwnerQuery::kSetInputCallbacks,
                                           OwnerQuery::kForwardInput),
                         owner_query_name);

/// Shared between the install hook (installer thread, which takes no user data) and the poller.
struct InstallProbe {
  std::mutex mutex;
  std::condition_variable cv;
  bool in_install = false;
  std::atomic<bool> install_published{false};
};

InstallProbe* g_install_probe = nullptr;

void hold_install_open() {
  InstallProbe& probe = *g_install_probe;
  {
    const std::lock_guard<std::mutex> lock(probe.mutex);
    probe.in_install = true;
  }
  probe.cv.notify_all();
  std::this_thread::sleep_for(kDispatchHold);
  // The renderer publishes its cached window state right after the hook returns, under the same
  // lock; a poll that observes this flag set could only have entered after that publication.
  probe.install_published = true;
}

TEST(GlfwConcurrency, InputCallbackInstallExcludesOtherThreadsDispatch) {
  const std::string suffix = std::to_string(getpid());
  auto window_a = create_window({.title = "glfw install poller " + suffix, .size = {64, 48}});
  if (!window_a) {
    HOLOVIZ_ENV_SKIP() << "no window system available: " << window_a.error().message;
  }
  auto window_b = create_window({.title = "glfw install target " + suffix, .size = {64, 48}});
  ASSERT_TRUE(window_b.has_value()) << window_b.error().message;

  ImGuiBridge bridge;
  const std::string unavailable = bridge.create();
  if (!unavailable.empty()) {
    HOLOVIZ_ENV_SKIP() << unavailable;
  }

  InstallProbe probe;
  g_install_probe = &probe;
  imgui::detail::set_input_install_hook(&hold_install_open);

  // Thread A: the other operator. Its poll may dispatch B's events once B's installation has
  // registered the callbacks; it must not enter until the installation has completed.
  bool reached = false;
  bool published_when_poll_returned = false;
  std::thread poller([&] {
    {
      std::unique_lock<std::mutex> lock(probe.mutex);
      reached = probe.cv.wait_for(lock, kEventTimeout, [&probe] { return probe.in_install; });
    }
    if (reached) {
      poll_window_events();
      published_when_poll_returned = probe.install_published;
    }
  });

  auto installed = bridge.renderer.install_input_callbacks(*window_b);
  poller.join();
  imgui::detail::set_input_install_hook(nullptr);
  g_install_probe = nullptr;

  ASSERT_TRUE(installed.has_value()) << installed.error().message;
  EXPECT_TRUE(bridge.renderer.uninstall_input_callbacks().has_value());
  ASSERT_TRUE(reached) << "the installation never reached the publication gap";
  EXPECT_TRUE(published_when_poll_returned)
      << "another thread's poll ran while the renderer had registered its callbacks but not yet "
         "published the state they read";
}

#endif  // HOLOVIZ_TEST_HAVE_X11

}  // namespace
}  // namespace holoviz
