// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Input-callback regression for the ImGui backend, public surface only: install the GLFW
// callbacks through install_input_callbacks(), capture the key trampoline GLFW now holds
// (glfwSetKeyCallback returns the previous callback), and invoke it directly with mapped,
// unmapped, and out-of-range key codes. Unmapped keys must be dropped before
// ImGuiIO::AddKeyEvent — ImGui's named-key API must never see ImGuiKey_None — while mapped
// keys still register. Windowed: skips without a window system, and skips when the test
// executable's GLFW is not the instance libholoviz initialized (shared-library builds hide
// their statically absorbed GLFW, so the capture cannot work there).

#include <array>
#include <utility>

#include <cuda.h>

#include <GLFW/glfw3.h>

#include <holoviz/holoviz.hpp>
#include <holoviz/imgui/ui.hpp>

#include "image_compare_test_util.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz::imgui {
namespace {

using test_util::try_create_device;

class ImGuiInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto window = create_window({.title = "imgui input test", .size = {160, 120}});
    if (!window) {
      HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
    }
    window_ = std::move(*window);
    device_ = try_create_device();
    if (!device_) {
      HOLOVIZ_ENV_SKIP() << "no usable GPU/adapter on this machine";
    }
    const auto context = device_.cuda_context();
    ASSERT_TRUE(context.has_value()) << context.error().message;
    if (cuCtxSetCurrent(*context) != CUDA_SUCCESS ||
        cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
      HOLOVIZ_ENV_SKIP() << "cannot create a CUDA stream on the device context";
    }
  }

  void TearDown() override {
    if (device_) {
      EXPECT_TRUE(device_.wait_idle().has_value());
    }
    if (stream_ != nullptr) {
      cuStreamSynchronize(stream_);
      cuStreamDestroy(stream_);
    }
    if (device_) {
      EXPECT_TRUE(device_.destroy().has_value());
    }
    if (window_) {
      EXPECT_TRUE(window_.destroy().has_value());
    }
  }

  Window window_;
  Device device_;
  CUstream stream_ = nullptr;
};

TEST_F(ImGuiInputTest, UnmappedKeysAreDroppedAndMappedKeysRegister) {
  auto backend_result = create_renderer(device_, {.stream = stream_, .frames_in_flight = 2});
  ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
  Renderer backend = std::move(*backend_result);
  ASSERT_TRUE(backend.install_input_callbacks(window_).has_value());

  const auto native = window_.native_object(NativeObjectType::kGLFWwindow);
  ASSERT_TRUE(native.has_value()) << native.error().message;
  auto* glfw_window = reinterpret_cast<GLFWwindow*>(native->value);
  ASSERT_NE(glfw_window, nullptr);

  // Borrow the key trampoline libholoviz registered on the window. A null return means this
  // executable's GLFW is not the initialized instance (statically absorbed into a shared
  // libholoviz) — the invoke-the-trampoline technique cannot work there.
  GLFWkeyfun trampoline = glfwSetKeyCallback(glfw_window, nullptr);
  if (trampoline == nullptr) {
    EXPECT_TRUE(backend.uninstall_input_callbacks().has_value());
    EXPECT_TRUE(backend.destroy().has_value());
    GTEST_SKIP() << "cannot capture the GLFW key trampoline in this build configuration";
  }
  glfwSetKeyCallback(glfw_window, trampoline);

  // Keys with no named ImGuiKey in the pinned Dear ImGui (1.89.3 stops at F12). Each of
  // these must be silently dropped; forwarding ImGuiKey_None to AddKeyEvent would fire
  // ImGui's named-key assertion in assertion-enabled builds.
  const std::array<int, 6> unmapped_keys = {{GLFW_KEY_UNKNOWN,
                                             GLFW_KEY_WORLD_1,
                                             GLFW_KEY_WORLD_2,
                                             GLFW_KEY_F13,
                                             GLFW_KEY_F19,
                                             GLFW_KEY_F25}};
  for (const int key : unmapped_keys) {
    trampoline(glfw_window, key, 0, GLFW_PRESS, 0);
    trampoline(glfw_window, key, 0, GLFW_RELEASE, 0);
  }

  // Representative mapped keys (letter, keypad, high function key) still register: press
  // them, then observe the state through a normal ImGui frame. Trickling would spread the
  // queue across frames once a modifier changes twice — turn it off so one frame suffices;
  // the Shift-carrying press comes last so the modifier's final state is "down".
  {
    ImGuiContext* imgui_context = backend.context().value();
    ImGui::SetCurrentContext(imgui_context);
    ImGui::GetIO().ConfigInputTrickleEventQueue = false;
  }
  trampoline(glfw_window, GLFW_KEY_A, 0, GLFW_PRESS, 0);
  trampoline(glfw_window, GLFW_KEY_KP_5, 0, GLFW_PRESS, 0);
  trampoline(glfw_window, GLFW_KEY_F12, 0, GLFW_PRESS, GLFW_MOD_SHIFT);
  ASSERT_TRUE(backend.begin_frame(0, {}).has_value());
  bool a_down = false;
  bool f12_down = false;
  bool kp5_down = false;
  bool shift_down = false;
  const auto sync = backend.render_ui({160, 120}, [&] {
    a_down = ImGui::IsKeyDown(ImGuiKey_A);
    f12_down = ImGui::IsKeyDown(ImGuiKey_F12);
    kp5_down = ImGui::IsKeyDown(ImGuiKey_Keypad5);
    shift_down = ImGui::GetIO().KeyShift;
  });
  ASSERT_TRUE(sync.has_value()) << sync.error().message;
  EXPECT_TRUE(a_down);
  EXPECT_TRUE(f12_down);
  EXPECT_TRUE(kp5_down);
  EXPECT_TRUE(shift_down);

  trampoline(glfw_window, GLFW_KEY_A, 0, GLFW_RELEASE, 0);
  trampoline(glfw_window, GLFW_KEY_F12, 0, GLFW_RELEASE, 0);
  trampoline(glfw_window, GLFW_KEY_KP_5, 0, GLFW_RELEASE, 0);

  EXPECT_TRUE(backend.uninstall_input_callbacks().has_value());
  EXPECT_TRUE(device_.wait_idle().has_value());
  // The render_ui() uploads above were never consumed by a submit, so their copies may
  // still be in flight on the CUDA stream: give destroy() a bound to retire them (a zero
  // timeout would honestly report kBackpressured).
  EXPECT_TRUE(backend.destroy(std::chrono::seconds(1)).has_value());
}

}  // namespace
}  // namespace holoviz::imgui
