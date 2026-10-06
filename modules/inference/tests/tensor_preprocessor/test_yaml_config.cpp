/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure CPU tests for TensorPreprocessorOpParams::from_yaml_node()
// (see include/holoscan/operators/tensor_preprocessor/tensor_preprocessor.hpp /
// modules/inference/operators/tensor_preprocessor/tensor_preprocessor.cpp).
// None of these tests construct a TensorPreprocessorOp or run a graph -- they only exercise
// YAML parsing, so they do not require GPU access and are registered as ordinary gtests.

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include <yaml-cpp/yaml.h>

#include <holoscan/operators/tensor_preprocessor/tensor_preprocessor.hpp>

namespace holoscan::ops {

namespace {

TensorPreprocessorOpParams load(const std::string& yaml) {
  const YAML::Node root = YAML::Load(yaml);
  return TensorPreprocessorOpParams::from_yaml_node(root, std::filesystem::path{});
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorYaml, Defaults_MinimalYaml_PopulatesRequiredAndDefaults) {
  const auto params = load(R"(
output_width: 256
output_height: 256
)");

  EXPECT_EQ(params.output_width, 256);
  EXPECT_EQ(params.output_height, 256);
  EXPECT_EQ(params.input_port_name, "in");
  EXPECT_EQ(params.output_port_name, "out");
  EXPECT_EQ(params.input_pix_fmt, "rgb24");
  EXPECT_EQ(params.output_pix_fmt, "rgb_fp32");
  EXPECT_EQ(params.output_layout, "nchw");
  EXPECT_TRUE(params.prepend_batch_dim);
  EXPECT_EQ(params.normalize_mean, (std::array<float, 3>{0.0F, 0.0F, 0.0F}));
  EXPECT_EQ(params.normalize_std, (std::array<float, 3>{1.0F, 1.0F, 1.0F}));
  EXPECT_EQ(params.memory_kind, holoscan::MemoryKind::kCudaDevice);
}

// ---------------------------------------------------------------------------------------------
// Field-parsing round trips
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorYaml, PortNames_RoundTrip) {
  const auto params = load(R"(
input_port_name: raw_frame
output_port_name: de_id_pre
output_width: 320
output_height: 240
)");
  EXPECT_EQ(params.input_port_name, "raw_frame");
  EXPECT_EQ(params.output_port_name, "de_id_pre");
}

TEST(TensorPreprocessorYaml, InputPixFmt_Yuyv_Accepted) {
  const auto params = load(R"(
input_pix_fmt: yuyv
output_width: 128
output_height: 128
)");
  EXPECT_EQ(params.input_pix_fmt, "yuyv");
}

TEST(TensorPreprocessorYaml, OutputPixFmt_RgbU8_Accepted) {
  const auto params = load(R"(
output_pix_fmt: rgb_u8
output_width: 128
output_height: 128
)");
  EXPECT_EQ(params.output_pix_fmt, "rgb_u8");
}

TEST(TensorPreprocessorYaml, OutputLayout_Nhwc_Accepted) {
  const auto params = load(R"(
output_layout: nhwc
output_width: 128
output_height: 128
)");
  EXPECT_EQ(params.output_layout, "nhwc");
}

TEST(TensorPreprocessorYaml, PrependBatchDim_False_RoundTrips) {
  const auto params = load(R"(
prepend_batch_dim: false
output_width: 128
output_height: 128
)");
  EXPECT_FALSE(params.prepend_batch_dim);
}

TEST(TensorPreprocessorYaml, NormalizeMean_ThreeEntries_RoundTrips) {
  const auto params = load(R"(
output_width: 224
output_height: 224
normalize_mean: [0.485, 0.456, 0.406]
)");
  EXPECT_FLOAT_EQ(params.normalize_mean[0], 0.485F);
  EXPECT_FLOAT_EQ(params.normalize_mean[1], 0.456F);
  EXPECT_FLOAT_EQ(params.normalize_mean[2], 0.406F);
}

TEST(TensorPreprocessorYaml, NormalizeStd_ThreeEntries_RoundTrips) {
  const auto params = load(R"(
output_width: 224
output_height: 224
normalize_std: [0.229, 0.224, 0.225]
)");
  EXPECT_FLOAT_EQ(params.normalize_std[0], 0.229F);
  EXPECT_FLOAT_EQ(params.normalize_std[1], 0.224F);
  EXPECT_FLOAT_EQ(params.normalize_std[2], 0.225F);
}

TEST(TensorPreprocessorYaml, MemoryKind_CudaManaged_RoundTrips) {
  const auto params = load(R"(
output_width: 128
output_height: 128
memory_kind: cuda_managed
)");
  EXPECT_EQ(params.memory_kind, holoscan::MemoryKind::kCudaManaged);
}

// ---------------------------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorYaml, MissingOutputWidth_Throws) {
  EXPECT_THROW(load(R"(output_height: 128)"), std::runtime_error);
}

TEST(TensorPreprocessorYaml, MissingOutputHeight_Throws) {
  EXPECT_THROW(load(R"(output_width: 128)"), std::runtime_error);
}

TEST(TensorPreprocessorYaml, InvalidInputPixFmt_Throws) {
  EXPECT_THROW(load(R"(
input_pix_fmt: bgr24
output_width: 128
output_height: 128
)"),
               std::runtime_error);
}

TEST(TensorPreprocessorYaml, InvalidOutputPixFmt_Throws) {
  EXPECT_THROW(load(R"(
output_pix_fmt: rgb_int16
output_width: 128
output_height: 128
)"),
               std::runtime_error);
}

TEST(TensorPreprocessorYaml, InvalidOutputLayout_Throws) {
  EXPECT_THROW(load(R"(
output_layout: hwcn
output_width: 128
output_height: 128
)"),
               std::runtime_error);
}

TEST(TensorPreprocessorYaml, NormalizeMean_TooFewEntries_Throws) {
  EXPECT_THROW(load(R"(
output_width: 128
output_height: 128
normalize_mean: [0.5, 0.5]
)"),
               std::runtime_error);
}

TEST(TensorPreprocessorYaml, NormalizeMean_TooManyEntries_Throws) {
  EXPECT_THROW(load(R"(
output_width: 128
output_height: 128
normalize_mean: [0.1, 0.2, 0.3, 0.4]
)"),
               std::runtime_error);
}

TEST(TensorPreprocessorYaml, NormalizeStd_TooFewEntries_Throws) {
  EXPECT_THROW(load(R"(
output_width: 128
output_height: 128
normalize_std: [1.0, 1.0]
)"),
               std::runtime_error);
}

TEST(TensorPreprocessorYaml, InvalidMemoryKind_Throws) {
  EXPECT_THROW(load(R"(
output_width: 128
output_height: 128
memory_kind: host
)"),
               std::runtime_error);
}

// ---------------------------------------------------------------------------------------------
// Camera-integration params: input_kind, yuv_matrix, yuv_range, nv12/i420 pix_fmt
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorYaml, InputKind_Image_Accepted) {
  const auto params = load(R"(
input_kind: image
output_width: 128
output_height: 128
)");
  EXPECT_EQ(params.input_kind, "image");
}

TEST(TensorPreprocessorYaml, InputKind_Invalid_Throws) {
  EXPECT_THROW(load(R"(
input_kind: bytes
output_width: 128
output_height: 128
)"),
               std::runtime_error);
}

TEST(TensorPreprocessorYaml, InputPixFmt_Nv12_Accepted) {
  const auto params = load(R"(
input_pix_fmt: nv12
output_width: 128
output_height: 128
)");
  EXPECT_EQ(params.input_pix_fmt, "nv12");
}

TEST(TensorPreprocessorYaml, InputPixFmt_I420_Accepted) {
  const auto params = load(R"(
input_pix_fmt: i420
output_width: 128
output_height: 128
)");
  EXPECT_EQ(params.input_pix_fmt, "i420");
}

TEST(TensorPreprocessorYaml, YuvMatrix_Bt709_RoundTrips) {
  const auto params = load(R"(
yuv_matrix: bt709
output_width: 128
output_height: 128
)");
  EXPECT_EQ(params.yuv_matrix, "bt709");
}

TEST(TensorPreprocessorYaml, YuvMatrix_Invalid_Throws) {
  EXPECT_THROW(load(R"(
yuv_matrix: bt2020
output_width: 128
output_height: 128
)"),
               std::runtime_error);
}

TEST(TensorPreprocessorYaml, YuvRange_Limited_RoundTrips) {
  const auto params = load(R"(
yuv_range: limited
output_width: 128
output_height: 128
)");
  EXPECT_EQ(params.yuv_range, "limited");
}

TEST(TensorPreprocessorYaml, YuvRange_Invalid_Throws) {
  EXPECT_THROW(load(R"(
yuv_range: studio
output_width: 128
output_height: 128
)"),
               std::runtime_error);
}

// ---------------------------------------------------------------------------------------------
// Flat bare-Tensor NV12/I420 geometry: input_width, input_height, input_planes
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorYaml, FlatNv12_InputPlanes_RoundTrip) {
  const auto params = load(R"(
input_kind: tensor
input_pix_fmt: nv12
input_width: 320
input_height: 240
input_planes:
  - {offset_bytes: 0,     row_stride_bytes: 320}
  - {offset_bytes: 76800, row_stride_bytes: 320}
output_width: 128
output_height: 128
)");
  EXPECT_EQ(params.input_width, 320);
  EXPECT_EQ(params.input_height, 240);
  ASSERT_EQ(params.input_planes.size(), 2U);
  EXPECT_EQ(params.input_planes[0].offset_bytes, 0U);
  EXPECT_EQ(params.input_planes[0].row_stride_bytes, 320U);
  EXPECT_EQ(params.input_planes[1].offset_bytes, 76800U);
  EXPECT_EQ(params.input_planes[1].row_stride_bytes, 320U);
}

TEST(TensorPreprocessorYaml, FlatI420_ThreePlanes_RoundTrip) {
  const auto params = load(R"(
input_pix_fmt: i420
input_width: 320
input_height: 240
input_planes:
  - {offset_bytes: 0,     row_stride_bytes: 320}
  - {offset_bytes: 76800, row_stride_bytes: 160}
  - {offset_bytes: 96000, row_stride_bytes: 160}
output_width: 128
output_height: 128
)");
  ASSERT_EQ(params.input_planes.size(), 3U);
  EXPECT_EQ(params.input_planes[2].offset_bytes, 96000U);
  EXPECT_EQ(params.input_planes[2].row_stride_bytes, 160U);
}

TEST(TensorPreprocessorYaml, InputPlanes_MissingRowStride_Throws) {
  EXPECT_THROW(load(R"(
input_planes:
  - {offset_bytes: 0}
output_width: 128
output_height: 128
)"),
               std::runtime_error);
}

TEST(TensorPreprocessorYaml, InputPlanes_NotASequence_Throws) {
  EXPECT_THROW(load(R"(
input_planes: 42
output_width: 128
output_height: 128
)"),
               std::runtime_error);
}

}  // namespace holoscan::ops
