/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <stdlib.h>

#include <algorithm>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoviz/holoviz.hpp>
#include "test_fixture.hpp"

namespace viz = holoscan::viz;

class Callback : public TestWindow {
 public:
  void SetUp() override {
    TestWindow::SetUp();

    if (glfwGetPlatform() != GLFW_PLATFORM_X11) { GTEST_SKIP() << "Test runs on X11 only."; }

    dpy_ = XOpenDisplay(NULL);
    ASSERT_NE(dpy_, nullptr) << "XOpenDisplay() failed";

    root_window_ = XDefaultRootWindow(dpy_);
    ASSERT_NE(root_window_, 0) << "XDefaultRootWindow() failed";

    int revert_to;
    ASSERT_NE(XGetInputFocus(dpy_, &window_, &revert_to), 0);
  }

  void TearDown() override {
    if (dpy_) { XCloseDisplay(dpy_); }
    TestWindow::TearDown();
  }

  /**
   * Sends a key event using XSendEvent() and adds the key event to the expected key event array
   */
  void send_key_event(viz::Key key, viz::KeyAndButtonAction action, viz::KeyModifiers modifiers) {
    XKeyEvent event{};

    event.display = dpy_;
    event.window = window_;
    event.root = root_window_;
    event.subwindow = None;
    event.time = CurrentTime;
    event.x = 1;
    event.y = 1;
    event.x_root = 1;
    event.y_root = 1;
    event.same_screen = True;

    if (modifiers.shift) { event.state |= ShiftMask; }
    if (modifiers.control) { event.state |= ControlMask; }
    if (modifiers.alt) { event.state |= Mod1Mask; }
    // for caps lock and num lock we need to send actual events since thet are not part of the
    // XSendEvent() modifiers
    if (modifiers.caps_lock) {
      event.keycode = XKeysymToKeycode(dpy_, XK_Caps_Lock);
      event.type = KeyPress;
      EXPECT_NE(XSendEvent(dpy_, window_, True, KeyPressMask, (XEvent*)&event), 0);
      // caps lock is unset before the caps lock key is pressed
      modifiers.caps_lock = 0;
      expected_key_events_.emplace_back(
          viz::Key::CAPS_LOCK, viz::KeyAndButtonAction::PRESS, modifiers);
      modifiers.caps_lock = 1;
    }
    if (modifiers.num_lock) {
      event.keycode = XKeysymToKeycode(dpy_, XK_Num_Lock);
      event.type = KeyPress;
      EXPECT_NE(XSendEvent(dpy_, window_, True, KeyPressMask, (XEvent*)&event), 0);
      // num lock is unset before the num lock key is pressed
      modifiers.num_lock = 0;
      expected_key_events_.emplace_back(
          viz::Key::NUM_LOCK, viz::KeyAndButtonAction::PRESS, modifiers);
      modifiers.num_lock = 1;
    }

    ASSERT_TRUE((key >= viz::Key::A) && (key <= viz::Key::Z));
    event.keycode = XKeysymToKeycode(dpy_, XK_A + int(key) - int(viz::Key::A));

    int event_mask = 0;
    if (action == viz::KeyAndButtonAction::PRESS) {
      event.type = KeyPress;
      event_mask = KeyPressMask;
    } else if (action == viz::KeyAndButtonAction::RELEASE) {
      event.type = KeyRelease;
      event_mask = KeyReleaseMask;
    }

    EXPECT_NE(XSendEvent(dpy_, window_, True, event_mask, (XEvent*)&event), 0);
    expected_key_events_.emplace_back(key, action, modifiers);

    if (modifiers.caps_lock) {
      event.keycode = XKeysymToKeycode(dpy_, XK_Caps_Lock);
      event.type = KeyRelease;
      EXPECT_NE(XSendEvent(dpy_, window_, True, KeyReleaseMask, (XEvent*)&event), 0);
      expected_key_events_.emplace_back(
          viz::Key::CAPS_LOCK, viz::KeyAndButtonAction::RELEASE, modifiers);
    }
    if (modifiers.num_lock) {
      event.keycode = XKeysymToKeycode(dpy_, XK_Num_Lock);
      event.type = KeyRelease;
      EXPECT_NE(XSendEvent(dpy_, window_, True, KeyReleaseMask, (XEvent*)&event), 0);
      expected_key_events_.emplace_back(
          viz::Key::NUM_LOCK, viz::KeyAndButtonAction::RELEASE, modifiers);
    }
    ASSERT_NE(XFlush(dpy_), 0);
  }

  /**
   * Sends a mouse button event using XSendEvent() and adds the mouse button event to the expected
   * mouse button event array
   */
  void send_button_event(viz::MouseButton button, viz::KeyAndButtonAction action) {
    XButtonEvent event{};

    event.display = dpy_;
    event.window = window_;
    event.root = root_window_;
    event.subwindow = None;
    event.time = CurrentTime;
    event.x = 1;
    event.y = 1;
    event.x_root = 1;
    event.y_root = 1;
    event.same_screen = True;

    event.state = Button1Mask << int(button);
    event.button = Button1 + int(button);

    int event_mask = 0;
    if (action == viz::KeyAndButtonAction::PRESS) {
      event.type = ButtonPress;
      event_mask = ButtonPressMask;
    } else if (action == viz::KeyAndButtonAction::RELEASE) {
      event.type = ButtonRelease;
      event_mask = ButtonReleaseMask;
    }

    EXPECT_NE(XSendEvent(dpy_, window_, True, event_mask, (XEvent*)&event), 0);
    ASSERT_NE(XFlush(dpy_), 0);

    expected_mouse_button_events_.emplace_back(button, action, viz::KeyModifiers{});
  }

  /**
   * Sends mouse button events using XSendEvent() and adds the scroll event to the expected
   * scroll event array
   */
  void send_scroll_event(int x_offset, int y_offset) {
    // X11 uses button 4 and 5 for y scroll, button 6 and 7 for x scroll (x11 buttons start at 1 and
    // viz::MouseButton at 0, therefore the buttons below are one lower)
    while (y_offset < 0) {
      send_button_event(viz::MouseButton(3), viz::KeyAndButtonAction::PRESS);
      ++y_offset;
    }
    while (y_offset > 0) {
      send_button_event(viz::MouseButton(4), viz::KeyAndButtonAction::PRESS);
      --y_offset;
    }
    while (x_offset < 0) {
      send_button_event(viz::MouseButton(5), viz::KeyAndButtonAction::PRESS);
      ++x_offset;
    }
    while (x_offset > 0) {
      send_button_event(viz::MouseButton(6), viz::KeyAndButtonAction::PRESS);
      --x_offset;
    }
  }

  /**
   * Sends a pointer moved event using XSendEvent() and adds the cursor pos event to the expected
   * cursor pos event array
   */
  void send_cursor_pos_event(int x, int y) {
    XPointerMovedEvent event{};

    event.type = MotionNotify;
    event.display = dpy_;
    event.window = window_;
    event.root = root_window_;
    event.subwindow = None;
    event.time = CurrentTime;
    event.x = x;
    event.y = y;
    event.same_screen = True;

    EXPECT_NE(XSendEvent(dpy_, window_, True, MotionNotify, (XEvent*)&event), 0);
    ASSERT_NE(XFlush(dpy_), 0);

    expected_cursor_pos_events_.emplace_back(x, y);
  }

  /**
   * Sends a resize event using XResizeWindow() and adds the size event to the expected
   * framebuffer and window size event array
   */
  void send_resize_event(int width, int height) {
    EXPECT_NE(XResizeWindow(dpy_, window_, width, height), 0);
    ASSERT_NE(XFlush(dpy_), 0);

    expected_framebuffer_size_events_.emplace_back(width, height);
    expected_window_size_events_.emplace_back(width, height);
  }

  Display* dpy_ = nullptr;
  Window root_window_ = 0;
  Window window_ = 0;

  class KeyEvent {
   public:
    KeyEvent(viz::Key key, viz::KeyAndButtonAction action, viz::KeyModifiers modifiers)
        : key_(key), action_(action), modifiers_(modifiers) {}
    viz::Key key_{};
    viz::KeyAndButtonAction action_{};
    viz::KeyModifiers modifiers_{};
  };
  std::vector<KeyEvent> key_events_;
  std::vector<KeyEvent> expected_key_events_;

  std::vector<uint32_t> unicode_char_events_;

  class MouseButtonEvent {
   public:
    MouseButtonEvent(viz::MouseButton mouse_button, viz::KeyAndButtonAction action,
                     viz::KeyModifiers modifiers)
        : mouse_button_(mouse_button), action_(action), modifiers_(modifiers) {}
    viz::MouseButton mouse_button_{};
    viz::KeyAndButtonAction action_{};
    viz::KeyModifiers modifiers_{};
  };
  std::vector<MouseButtonEvent> mouse_button_events_;
  std::vector<MouseButtonEvent> expected_mouse_button_events_;

  class ScrollEvent {
   public:
    ScrollEvent(double x_offset, double y_offset) : x_offset_(x_offset), y_offset_(y_offset) {}
    double x_offset_{};
    double y_offset_{};
  };
  std::vector<ScrollEvent> scroll_events_;
  std::vector<ScrollEvent> expected_scroll_events_;

  class CursorPosEvent {
   public:
    CursorPosEvent(double x_pos, double y_pos) : x_pos_(x_pos), y_pos_(x_pos) {}
    double x_pos_{};
    double y_pos_{};
  };
  std::vector<CursorPosEvent> cursor_pos_events_;
  std::vector<CursorPosEvent> expected_cursor_pos_events_;

  class SizeEvent {
   public:
    SizeEvent(int width, int height) : width_(width), height_(height) {}
    double width_{};
    double height_{};
  };
  std::vector<SizeEvent> framebuffer_size_events_;
  std::vector<SizeEvent> expected_framebuffer_size_events_;
  std::vector<SizeEvent> window_size_events_;
  std::vector<SizeEvent> expected_window_size_events_;
};

namespace holoscan::viz {
bool operator==(const KeyModifiers& lhs, const KeyModifiers& rhs) {
  return ((lhs.shift == rhs.shift) && (lhs.control == rhs.control) && (lhs.alt == rhs.alt) &&
          (lhs.caps_lock == rhs.caps_lock) && (lhs.num_lock == rhs.num_lock));
}
};  // namespace holoscan::viz

bool operator==(const Callback::KeyEvent& lhs, const Callback::KeyEvent& rhs) {
  return ((lhs.key_ == rhs.key_) && (lhs.action_ == rhs.action_) &&
          (lhs.modifiers_ == rhs.modifiers_));
}

bool operator==(const Callback::MouseButtonEvent& lhs, const Callback::MouseButtonEvent& rhs) {
  return ((lhs.mouse_button_ == rhs.mouse_button_) && (lhs.action_ == rhs.action_) &&
          (lhs.modifiers_ == rhs.modifiers_));
}

bool operator==(const Callback::ScrollEvent& lhs, const Callback::ScrollEvent& rhs) {
  return ((lhs.x_offset_ == rhs.x_offset_) && (lhs.y_offset_ == rhs.y_offset_));
}

bool operator==(const Callback::CursorPosEvent& lhs, const Callback::CursorPosEvent& rhs) {
  return ((lhs.x_pos_ == rhs.x_pos_) && (lhs.y_pos_ == rhs.y_pos_));
}

bool operator==(const Callback::SizeEvent& lhs, const Callback::SizeEvent& rhs) {
  return ((lhs.width_ == rhs.width_) && (lhs.height_ == rhs.height_));
}

static void key_callback(void* user_pointer, viz::Key key, viz::KeyAndButtonAction action,
                         viz::KeyModifiers modifiers) {
  Callback* instance = reinterpret_cast<Callback*>(user_pointer);
  instance->key_events_.emplace_back(key, action, modifiers);
}

TEST_F(Callback, Key) {
  // set the key callback
  EXPECT_NO_THROW(viz::SetKeyCallback(this, &key_callback));

  // test all modifiers
  std::vector<viz::KeyModifiers> modifiers_to_test{{0, 0, 0, 0, 0},
                                                   {1, 0, 0, 0, 0},
                                                   {0, 1, 0, 0, 0},
                                                   {0, 0, 1, 0, 0},
                                                   {0, 0, 0, 1, 0},
                                                   {0, 0, 0, 0, 1}};
  for (auto&& modifiers : modifiers_to_test) {
    key_events_.clear();
    expected_key_events_.clear();

    // press and release a key
    send_key_event(viz::Key::A, viz::KeyAndButtonAction::PRESS, modifiers);
    send_key_event(viz::Key::A, viz::KeyAndButtonAction::RELEASE, modifiers);

    // need to call Begin()/End() to process events
    uint32_t count = 0;
    while ((key_events_.size() < expected_key_events_.size()) && (count < 10)) {
      EXPECT_NO_THROW(viz::Begin());
      EXPECT_NO_THROW(viz::End());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      ++count;
    }
    EXPECT_EQ(key_events_.size(), expected_key_events_.size())
        << "Unexpected event count after 100 ms";

    // check that the received events match the expected events
    EXPECT_EQ(key_events_, expected_key_events_);
  }

  // unset
  ASSERT_NO_THROW(viz::SetKeyCallback(nullptr, nullptr));

  // make sure no events are generated when the callback is unset
  key_events_.clear();
  send_key_event(viz::Key::A, viz::KeyAndButtonAction::PRESS, viz::KeyModifiers{});
  uint32_t count = 0;
  while (key_events_.empty() && (count < 10)) {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::End());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ++count;
  }
  EXPECT_TRUE(key_events_.empty());
}

static void unicode_char_callback(void* user_pointer, uint32_t code_point) {
  Callback* instance = reinterpret_cast<Callback*>(user_pointer);
  instance->unicode_char_events_.emplace_back(code_point);
}

TEST_F(Callback, UnicodeChar) {
  // set the key callback
  EXPECT_NO_THROW(viz::SetUnicodeCharCallback(this, &unicode_char_callback));

  const std::string test_string("testcallback");
  for (auto&& character : test_string) {
    send_key_event(viz::Key(int(viz::Key::A) + character - 'a'),
                   viz::KeyAndButtonAction::PRESS,
                   viz::KeyModifiers{});
    send_key_event(viz::Key(int(viz::Key::A) + character - 'a'),
                   viz::KeyAndButtonAction::RELEASE,
                   viz::KeyModifiers{});
  }

  // need to call Begin()/End() to process events
  uint32_t count = 0;
  while ((unicode_char_events_.size() < (expected_key_events_.size() / 2)) && (count < 10)) {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::End());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ++count;
  }
  EXPECT_EQ(unicode_char_events_.size(), (expected_key_events_.size() / 2))
      << "Unexpected event count after 100 ms";

  // check that the received events match the expected events
  for (size_t index = 0;
       index < std::min(unicode_char_events_.size(), expected_key_events_.size() / 2);
       ++index) {
    EXPECT_EQ(unicode_char_events_[index],
              uint32_t(expected_key_events_[index * 2].key_) - uint32_t(viz::Key::A) + 'a');
  }

  // unset
  ASSERT_NO_THROW(viz::SetUnicodeCharCallback(nullptr, nullptr));

  // make sure no events are generated when the callback is unset
  unicode_char_events_.clear();
  send_key_event(viz::Key::A, viz::KeyAndButtonAction::PRESS, viz::KeyModifiers{});
  send_key_event(viz::Key::A, viz::KeyAndButtonAction::RELEASE, viz::KeyModifiers{});
  count = 0;
  while (unicode_char_events_.empty() && (count < 10)) {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::End());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ++count;
  }
  EXPECT_TRUE(unicode_char_events_.empty());
}

static void mouse_button_callback(void* user_pointer, viz::MouseButton button,
                                  viz::KeyAndButtonAction action, viz::KeyModifiers modifiers) {
  Callback* instance = reinterpret_cast<Callback*>(user_pointer);
  instance->mouse_button_events_.emplace_back(button, action, modifiers);
}

TEST_F(Callback, MouseButton) {
  // set the mouse button callback
  EXPECT_NO_THROW(viz::SetMouseButtonCallback(this, &mouse_button_callback));

  for (auto&& button :
       {viz::MouseButton::LEFT, viz::MouseButton::MIDDLE, viz::MouseButton::RIGHT}) {
    mouse_button_events_.clear();
    expected_mouse_button_events_.clear();

    send_button_event(button, viz::KeyAndButtonAction::PRESS);
    send_button_event(button, viz::KeyAndButtonAction::RELEASE);

    // need to call Begin()/End() to process events
    uint32_t count = 0;
    while ((mouse_button_events_.size() < expected_mouse_button_events_.size()) && (count < 10)) {
      EXPECT_NO_THROW(viz::Begin());
      EXPECT_NO_THROW(viz::End());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      ++count;
    }
    EXPECT_EQ(mouse_button_events_.size(), expected_mouse_button_events_.size())
        << "Unexpected event count after 100 ms";

    EXPECT_EQ(mouse_button_events_, expected_mouse_button_events_);
  }

  // unset
  ASSERT_NO_THROW(viz::SetMouseButtonCallback(nullptr, nullptr));

  // make sure no events are generated when the callback is unset
  mouse_button_events_.clear();
  send_button_event(viz::MouseButton::LEFT, viz::KeyAndButtonAction::PRESS);
  send_button_event(viz::MouseButton::LEFT, viz::KeyAndButtonAction::RELEASE);
  uint32_t count = 0;
  while (mouse_button_events_.empty() && (count < 10)) {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::End());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ++count;
  }
  EXPECT_TRUE(mouse_button_events_.empty());
}

static void scroll_callback(void* user_pointer, double x_offset, double y_offset) {
  Callback* instance = reinterpret_cast<Callback*>(user_pointer);
  instance->scroll_events_.emplace_back(x_offset, y_offset);
}

TEST_F(Callback, Scroll) {
  // set the mouse button callback
  EXPECT_NO_THROW(viz::SetScrollCallback(this, &scroll_callback));

  std::vector<std::pair<int, int>> scroll_offsets_to_test{
      {1, -1}, {-1, 1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}};

  for (auto&& scroll_offset : scroll_offsets_to_test) {
    scroll_events_.clear();
    expected_scroll_events_.clear();

    send_scroll_event(scroll_offset.first, scroll_offset.second);

    // need to call Begin()/End() to process events
    uint32_t count = 0;
    while ((scroll_events_.size() < expected_scroll_events_.size()) && (count < 10)) {
      EXPECT_NO_THROW(viz::Begin());
      EXPECT_NO_THROW(viz::End());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      ++count;
    }
    EXPECT_EQ(scroll_events_.size(), expected_scroll_events_.size())
        << "Unexpected event count after 100 ms";

    EXPECT_EQ(scroll_events_, expected_scroll_events_);
  }

  // unset
  ASSERT_NO_THROW(viz::SetScrollCallback(nullptr, nullptr));

  // make sure no events are generated when the callback is unset
  scroll_events_.clear();
  send_scroll_event(1, 1);
  uint32_t count = 0;
  while (scroll_events_.empty() && (count < 10)) {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::End());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ++count;
  }
  EXPECT_TRUE(scroll_events_.empty());
}

static void cursor_pos_callback(void* user_pointer, double x_pos, double y_pos) {
  Callback* instance = reinterpret_cast<Callback*>(user_pointer);
  instance->cursor_pos_events_.emplace_back(x_pos, y_pos);
}

TEST_F(Callback, CursorPos) {
  // set the mouse button callback
  EXPECT_NO_THROW(viz::SetCursorPosCallback(this, &cursor_pos_callback));

  std::vector<std::pair<int, int>> cursor_pos_to_test{{10, 4}, {0, 12}, {0, -1}, {-1, 0}};

  for (auto&& cursor_pos : cursor_pos_to_test) {
    cursor_pos_events_.clear();
    expected_cursor_pos_events_.clear();

    send_cursor_pos_event(cursor_pos.first, cursor_pos.second);

    // need to call Begin()/End() to process events
    uint32_t count = 0;
    while ((cursor_pos_events_.size() < expected_cursor_pos_events_.size()) && (count < 10)) {
      EXPECT_NO_THROW(viz::Begin());
      EXPECT_NO_THROW(viz::End());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      ++count;
    }
    EXPECT_EQ(cursor_pos_events_.size(), expected_cursor_pos_events_.size())
        << "Unexpected event count after 100 ms";

    EXPECT_EQ(cursor_pos_events_, expected_cursor_pos_events_);
  }

  // unset
  ASSERT_NO_THROW(viz::SetCursorPosCallback(nullptr, nullptr));

  // make sure no events are generated when the callback is unset
  cursor_pos_events_.clear();
  send_cursor_pos_event(1, 1);
  uint32_t count = 0;
  while (cursor_pos_events_.empty() && (count < 10)) {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::End());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ++count;
  }
  EXPECT_TRUE(cursor_pos_events_.empty());
}

static void framebuffer_size_callback(void* user_pointer, int width, int height) {
  Callback* instance = reinterpret_cast<Callback*>(user_pointer);
  instance->framebuffer_size_events_.emplace_back(width, height);
}

static void window_size_callback(void* user_pointer, int width, int height) {
  Callback* instance = reinterpret_cast<Callback*>(user_pointer);
  instance->window_size_events_.emplace_back(width, height);
}

TEST_F(Callback, Size) {
  // set the size callbacks
  EXPECT_NO_THROW(viz::SetFramebufferSizeCallback(this, &framebuffer_size_callback));
  EXPECT_NO_THROW(viz::SetWindowSizeCallback(this, &window_size_callback));

  // when setting the framebuffer and window callbacks the callback is called once with the initial
  // size
  EXPECT_EQ(framebuffer_size_events_.size(), 1);
  EXPECT_EQ(framebuffer_size_events_[0].width_, width_);
  EXPECT_EQ(framebuffer_size_events_[0].height_, height_);

  EXPECT_EQ(window_size_events_.size(), 1);
  EXPECT_EQ(window_size_events_[0].width_, width_);
  EXPECT_EQ(window_size_events_[0].height_, height_);

  std::vector<std::pair<int, int>> sizes_to_test{{18, 44}, {112, 63}, {1, 1}, {1, 20}};

  for (auto&& size : sizes_to_test) {
    framebuffer_size_events_.clear();
    expected_framebuffer_size_events_.clear();
    window_size_events_.clear();
    expected_window_size_events_.clear();

    send_resize_event(size.first, size.second);

    // need to call Begin()/End() to process events
    uint32_t count = 0;
    while (((framebuffer_size_events_.size() < expected_framebuffer_size_events_.size()) ||
            (window_size_events_.size() < expected_window_size_events_.size())) &&
           (count < 100)) {
      EXPECT_NO_THROW(viz::Begin());
      EXPECT_NO_THROW(viz::End());
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      ++count;
    }
    EXPECT_EQ(framebuffer_size_events_.size(), expected_framebuffer_size_events_.size())
        << "Unexpected event count after 1000 ms";
    EXPECT_EQ(window_size_events_.size(), expected_window_size_events_.size())
        << "Unexpected event count after 1000 ms";

    EXPECT_EQ(framebuffer_size_events_, expected_framebuffer_size_events_);
    EXPECT_EQ(window_size_events_, expected_window_size_events_);
  }

  // unset
  ASSERT_NO_THROW(viz::SetFramebufferSizeCallback(nullptr, nullptr));
  ASSERT_NO_THROW(viz::SetWindowSizeCallback(nullptr, nullptr));

  // make sure no events are generated when the callback is unset
  framebuffer_size_events_.clear();
  window_size_events_.clear();
  send_resize_event(32, 64);
  uint32_t count = 0;
  while ((framebuffer_size_events_.empty() || window_size_events_.empty()) && (count < 100)) {
    EXPECT_NO_THROW(viz::Begin());
    EXPECT_NO_THROW(viz::End());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ++count;
  }
  EXPECT_TRUE(framebuffer_size_events_.empty());
  EXPECT_TRUE(window_size_events_.empty());
}

class CallbackHeadless : public TestHeadless {
 public:
  std::vector<Callback::SizeEvent> framebuffer_size_events_;
  std::vector<Callback::SizeEvent> window_size_events_;
};

static void headless_framebuffer_size_callback(void* user_pointer, int width, int height) {
  CallbackHeadless* instance = reinterpret_cast<CallbackHeadless*>(user_pointer);
  instance->framebuffer_size_events_.emplace_back(width, height);
}

static void headless_window_size_callback(void* user_pointer, int width, int height) {
  CallbackHeadless* instance = reinterpret_cast<CallbackHeadless*>(user_pointer);
  instance->window_size_events_.emplace_back(width, height);
}

TEST_F(CallbackHeadless, Size) {
  EXPECT_NO_THROW(viz::SetFramebufferSizeCallback(this, &headless_framebuffer_size_callback));
  EXPECT_NO_THROW(viz::SetWindowSizeCallback(this, &headless_window_size_callback));

  // when setting the framebuffer and window callbacks the callback is called once with the initial
  // size
  EXPECT_EQ(framebuffer_size_events_.size(), 1);
  EXPECT_EQ(framebuffer_size_events_[0].width_, width_);
  EXPECT_EQ(framebuffer_size_events_[0].height_, height_);

  EXPECT_EQ(window_size_events_.size(), 1);
  EXPECT_EQ(window_size_events_[0].width_, width_);
  EXPECT_EQ(window_size_events_[0].height_, height_);
}
