// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Regression for the X11 close-request test helper: its window-tree search walks a snapshot
// of other clients' windows, which may vanish mid-search. An unrelated parent is created
// before the target (so the depth-first search visits it first) while a second connection
// keeps creating and destroying its children. Without BadWindow handling the default Xlib
// handler exits the process. Both windows stay unmapped so a window manager never reparents,
// frames, or closes them; the search walks unmapped windows too. Pure Xlib, no GPU; skips
// without an X server.

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include <unistd.h>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"
#include "x11_close_request.hpp"

// After gtest: Xlib's macros (None, Status, Bool, ...) collide with its names.
#include <X11/Xlib.h>

namespace holoviz::test_util {
namespace {

constexpr int kRequests = 20;
/// Xlib never reuses a connection's window IDs; stay well below its ~2M ID range.
constexpr int kMaxChurnWindows = 200000;

TEST(X11CloseRequestTest, SurvivesUnrelatedWindowChurn) {
  if (XInitThreads() == 0) {  // two threads use Xlib; implicit only since libX11 1.8
    HOLOVIZ_ENV_SKIP() << "Xlib thread support is unavailable";
  }
  Display* display = XOpenDisplay(nullptr);
  if (display == nullptr) {
    HOLOVIZ_ENV_SKIP() << "no X11 display reachable";
  }
  Display* churn_display = XOpenDisplay(nullptr);
  ASSERT_NE(churn_display, nullptr);

  const ::Window root = DefaultRootWindow(display);
  const ::Window unrelated = XCreateSimpleWindow(display, root, 0, 0, 32, 32, 0, 0, 0);
  const ::Window target = XCreateSimpleWindow(display, root, 0, 0, 32, 32, 0, 0, 0);
  const std::string title = "x11 close request target " + std::to_string(getpid());
  XStoreName(display, target, title.c_str());
  XSync(display, False);

  std::atomic<bool> stop{false};
  std::thread churn([churn_display, unrelated, &stop] {
    for (int i = 0; i < kMaxChurnWindows && !stop; ++i) {
      const ::Window child = XCreateSimpleWindow(churn_display, unrelated, 0, 0, 8, 8, 0, 0, 0);
      XFlush(churn_display);
      XDestroyWindow(churn_display, child);
      XFlush(churn_display);
    }
    XSync(churn_display, False);
  });

  int sent = 0;
  for (int i = 0; i < kRequests; ++i) {
    if (send_x11_close_request(title, std::chrono::seconds(2)) == X11CloseRequest::kSent) {
      ++sent;
    }
  }
  stop = true;
  churn.join();
  EXPECT_EQ(sent, kRequests);

  // Each request reaches the target's owner as a ClientMessage.
  XSync(display, False);
  int received = 0;
  XEvent event{};
  while (XCheckTypedWindowEvent(display, target, ClientMessage, &event) != False) {
    ++received;
  }
  EXPECT_EQ(received, sent);

  XDestroyWindow(display, target);
  XDestroyWindow(display, unrelated);
  XCloseDisplay(churn_display);
  XCloseDisplay(display);
}

}  // namespace
}  // namespace holoviz::test_util
