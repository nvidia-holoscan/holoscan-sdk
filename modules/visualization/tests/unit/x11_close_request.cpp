// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "x11_close_request.hpp"

#include <string>
#include <thread>
#include <utility>

#include <X11/Xlib.h>

namespace holoviz::test_util {
namespace {

/// BadWindow trap for the request's own connection. The child list XQueryTree() returns is a
/// snapshot: an unrelated client's window can vanish before the next request on it, and the
/// default Xlib handler exits the process on that BadWindow. Only BadWindow on the trapped
/// display is absorbed; every other error goes to the previously installed handler.
/// XSetErrorHandler() is process-global, hence the file-static state.
struct BadWindowTrapState {
  Display* display = nullptr;
  bool bad_window = false;
  XErrorHandler previous = nullptr;
};
BadWindowTrapState trap_state;

int trap_bad_window(Display* display, XErrorEvent* error) {
  if (display == trap_state.display && error->error_code == BadWindow) {
    trap_state.bad_window = true;
    return 0;
  }
  return trap_state.previous != nullptr ? trap_state.previous(display, error) : 0;
}

class ScopedBadWindowTrap {
 public:
  explicit ScopedBadWindowTrap(Display* display) {
    trap_state = {.display = display, .bad_window = false, .previous = nullptr};
    trap_state.previous = XSetErrorHandler(trap_bad_window);
  }
  ~ScopedBadWindowTrap() {
    // Deliver this connection's outstanding errors while the trap is still installed.
    XSync(trap_state.display, False);
    XSetErrorHandler(trap_state.previous);
    trap_state = {};
  }
  ScopedBadWindowTrap(const ScopedBadWindowTrap&) = delete;
  ScopedBadWindowTrap& operator=(const ScopedBadWindowTrap&) = delete;

  /// Whether a BadWindow arrived since the last call; clears the flag.
  static bool take_bad_window() { return std::exchange(trap_state.bad_window, false); }
};

/// Depth-first search below \p parent for a window whose WM_NAME equals \p title. Window
/// managers reparent clients into frame windows, so the client is not always a root child.
/// Windows that vanish mid-search fail their requests (trapped BadWindow) and are skipped.
Window find_window_by_name(Display* display, Window parent, const std::string& title) {
  char* name = nullptr;
  if (XFetchName(display, parent, &name) != 0 && name != nullptr) {
    const bool match = title == name;
    XFree(name);
    if (match) {
      return parent;
    }
  }

  Window root_return = 0;
  Window parent_return = 0;
  Window* children = nullptr;
  unsigned int child_count = 0;
  if (XQueryTree(display, parent, &root_return, &parent_return, &children, &child_count) == 0) {
    return 0;
  }
  Window found = 0;
  for (unsigned int i = 0; i < child_count && found == 0; ++i) {
    found = find_window_by_name(display, children[i], title);
  }
  if (children != nullptr) {
    XFree(children);
  }
  return found;
}

}  // namespace

X11CloseRequest send_x11_close_request(const std::string& title,
                                       std::chrono::milliseconds timeout) {
  Display* display = XOpenDisplay(nullptr);
  if (display == nullptr) {
    return X11CloseRequest::kNoDisplay;
  }

  X11CloseRequest result = X11CloseRequest::kWindowNotFound;
  {
    const ScopedBadWindowTrap trap(display);
    const Atom wm_protocols = XInternAtom(display, "WM_PROTOCOLS", False);
    const Atom wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
      const Window target = find_window_by_name(display, DefaultRootWindow(display), title);
      if (target != 0) {
        XEvent event{};
        event.xclient.type = ClientMessage;
        event.xclient.window = target;
        event.xclient.message_type = wm_protocols;
        event.xclient.format = 32;
        event.xclient.data.l[0] = static_cast<long>(wm_delete_window);
        event.xclient.data.l[1] = CurrentTime;
        (void)ScopedBadWindowTrap::take_bad_window();  // discard errors from the search
        XSendEvent(display, target, False, NoEventMask, &event);
        // Round-trip so a BadWindow (target destroyed after it was found) is seen here.
        XSync(display, False);
        if (!ScopedBadWindowTrap::take_bad_window()) {
          result = X11CloseRequest::kSent;
          break;
        }
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  XCloseDisplay(display);
  return result;
}

}  // namespace holoviz::test_util
