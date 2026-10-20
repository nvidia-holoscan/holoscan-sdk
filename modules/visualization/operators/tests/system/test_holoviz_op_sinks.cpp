// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HolovizOp::start() sink dispatch, without a Graph (an empty LayerStack has no layer to drive
// rendering, hence ClockTrigger): the sinks that cannot be served throw a
// typed holoscan::RuntimeError before any GPU resource is created, so these cases run on any
// machine (no display, window system, or free adapter required).

#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

namespace holoscan::visualization {
namespace {

template <typename Fn>
holoscan::RuntimeError expect_runtime_error(Fn&& fn) {
  try {
    std::forward<Fn>(fn)();
  } catch (const holoscan::RuntimeError& e) {
    return e;
  }
  ADD_FAILURE() << "expected a holoscan::RuntimeError";
  return holoscan::RuntimeError{holoscan::ErrorCode::kSuccess, "no error thrown"};
}

TEST(HolovizOpSinks, UnknownExclusiveDisplayNameIsNotFound) {
  HolovizOp op{
      LayerStack{}, ExclusiveDisplaySinkDesc{.display_name = "no-such-display"}, ClockTrigger{}};
  const auto error = expect_runtime_error([&] { op.start(); });
  EXPECT_EQ(error.code(), holoscan::ErrorCode::kNotFound);
  EXPECT_NE(std::string(error.what()).find("no display matches \"no-such-display\""),
            std::string::npos)
      << error.what();
}

// The CUDA sink validates its declaration before any GPU resource exists, so these run headless.
TEST(HolovizOpSinks, CudaSinkRejectsZeroExtent) {
  HolovizOp op{LayerStack{}, CudaSinkDesc{.width = 64, .height = 0}, ClockTrigger{}};
  const auto error = expect_runtime_error([&] { op.start(); });
  EXPECT_EQ(error.code(), holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(std::string(error.what()).find("nonzero width and height"), std::string::npos)
      << error.what();
}

TEST(HolovizOpSinks, CudaSinkRejectsPlanarFormat) {
  HolovizOp op{LayerStack{},
               CudaSinkDesc{.width = 64, .height = 64, .format = ImageFormat::kNV12},
               ClockTrigger{}};
  const auto error = expect_runtime_error([&] { op.start(); });
  EXPECT_EQ(error.code(), holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(std::string(error.what()).find("single-plane ImageFormat"), std::string::npos)
      << error.what();
}

// RGB8 has a byte width but no render-target lowering: setup() leaves the output port unbound
// (same predicate) and start() rejects the sink.
TEST(HolovizOpSinks, CudaSinkRejectsRgb8Format) {
  HolovizOp op{LayerStack{},
               CudaSinkDesc{.width = 64, .height = 64, .format = ImageFormat::kR8G8B8Unorm},
               ClockTrigger{}};
  const auto error = expect_runtime_error([&] { op.start(); });
  EXPECT_EQ(error.code(), holoscan::ErrorCode::kInvalidArgument);
}

TEST(HolovizOpSinks, CudaSinkRejectsUnknownFormat) {
  HolovizOp op{LayerStack{},
               CudaSinkDesc{.width = 64, .height = 64, .format = ImageFormat::kUnknown},
               ClockTrigger{}};
  const auto error = expect_runtime_error([&] { op.start(); });
  EXPECT_EQ(error.code(), holoscan::ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace holoscan::visualization

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
