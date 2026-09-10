// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <variant>

#include <gtest/gtest.h>

#include <holoviz/backend_preference.hpp>
#include <holoviz/frame.hpp>
#include <holoviz/sink_desc.hpp>

namespace holoviz {

// The caller-facing present declaration surface: a SinkDesc holds exactly one sink kind, and
// the EA2-in-scope kinds (Window, ExclusiveDisplay) declare a present mode.
TEST(SinkDescTest, HoldsExactlyOneSinkKind) {
  SinkDesc window = WindowSinkDesc{"viz", 1920, 1080, PresentMode::kFIFOLatestReady};
  ASSERT_TRUE(std::holds_alternative<WindowSinkDesc>(window));
  EXPECT_EQ(std::get<WindowSinkDesc>(window).present_mode, PresentMode::kFIFOLatestReady);

  SinkDesc exclusive = ExclusiveDisplaySinkDesc{/*display_index=*/0};
  ASSERT_TRUE(std::holds_alternative<ExclusiveDisplaySinkDesc>(exclusive));
  EXPECT_FALSE(std::holds_alternative<WindowSinkDesc>(exclusive));
}

// The CUDA sinks are part of the stable declaration surface even though EA2 does not wire
// their present path — callers can construct code against them today.
TEST(SinkDescTest, CudaSinksAreDeclarable) {
  SinkDesc pitch = CudaPitchLinearSinkDesc{1920, 1080, ImageFormat::kR8G8B8A8Unorm};
  SinkDesc array = CudaArraySinkDesc{1920, 1080, ImageFormat::kR8G8B8A8Unorm};
  EXPECT_TRUE(std::holds_alternative<CudaPitchLinearSinkDesc>(pitch));
  EXPECT_TRUE(std::holds_alternative<CudaArraySinkDesc>(array));
}

TEST(BackendPreferenceTest, DefaultsAreSelectable) {
  BackendPreference pref = BackendPreference::kBuiltinVulkan;
  EXPECT_NE(pref, BackendPreference::kAuto);
}

// {0,0,0,0} is the full-extent sentinel; a nonzero width/height is a sub-rect binding.
TEST(FrameTest, SubregionFullExtentSentinel) {
  Subregion full{};
  EXPECT_EQ(full.x, 0u);
  EXPECT_EQ(full.y, 0u);
  EXPECT_EQ(full.width, 0u);
  EXPECT_EQ(full.height, 0u);
}

TEST(FrameTest, SubregionBindsWithinDeclaredMax) {
  ImageFrame frame{};
  frame.subregion = Subregion{0, 0, 1280, 720};
  EXPECT_EQ(frame.subregion.width, 1280u);
  EXPECT_EQ(frame.subregion.height, 720u);
}

// ImageFrame carries per-plane device-memory bindings. The declared ImageLayerDesc format
// (not repeated per frame) determines the active plane count: 1 for packed formats, 2 for NV12.
TEST(FrameTest, MultiPlaneBindingForNv12) {
  // Simulate a contiguous NV12 allocation: Y plane at base, UV plane at base + y_size.
  const void* base = reinterpret_cast<const void*>(static_cast<uintptr_t>(0x1000));
  const std::size_t y_pitch = 1920;
  const std::size_t y_capacity = y_pitch * 1088;  // aligned height
  const void* uv_base = reinterpret_cast<const void*>(static_cast<uintptr_t>(0x1000) + y_capacity);

  ImageFrame frame{};
  frame.planes[0] = ImagePlaneBinding{base, y_pitch, y_capacity};
  frame.planes[1] = ImagePlaneBinding{uv_base, y_pitch, y_capacity / 2};
  frame.subregion = Subregion{0, 0, 1920, 1080};

  EXPECT_EQ(frame.planes[0].device_ptr, base);
  EXPECT_EQ(frame.planes[0].row_pitch_bytes, y_pitch);
  EXPECT_EQ(frame.planes[1].device_ptr, uv_base);
  EXPECT_EQ(frame.planes[2].device_ptr, nullptr);  // unused plane is default-zero
}

}  // namespace holoviz
