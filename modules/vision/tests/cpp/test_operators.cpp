// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <nvcv/DataType.h>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>

#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/vision/bayer_demosaic_op.hpp>
#include <holoscan/vision/convert_to_op.hpp>
#include <holoscan/vision/copy_make_border_op.hpp>
#include <holoscan/vision/custom_crop_op.hpp>
#include <holoscan/vision/non_maximum_suppression_op.hpp>
#include <holoscan/vision/normalize_op.hpp>
#include <holoscan/vision/reformat_op.hpp>
#include <holoscan/vision/remap_op.hpp>
#include <holoscan/vision/resize_op.hpp>
#include <holoscan/vision/vision_types.hpp>

#include "detail/cvcuda_tensor.hpp"
#include "operator_test_fixture.hpp"

namespace holoscan::vision::detail {
namespace {

constexpr DLDataType kU8{kDLUInt, 8U, 1U};
constexpr DLDataType kU32{kDLUInt, 32U, 1U};
constexpr DLDataType kS8{kDLInt, 8U, 1U};
constexpr DLDataType kS16{kDLInt, 16U, 1U};
constexpr DLDataType kF32{kDLFloat, 32U, 1U};
constexpr DLDataType kF64{kDLFloat, 64U, 1U};
constexpr DLDevice kCudaDevice{kDLCUDA, 0};

static_assert(static_cast<std::uint8_t>(BorderMode::kReflect101) == 4U);
static_assert(static_cast<std::uint8_t>(RemapMapValueMode::kRelativeNormalized) == 2U);
static_assert(static_cast<std::uint8_t>(RemapMapSynchronizationMode::kExact) == 1U);
static_assert(static_cast<std::uint8_t>(VisionElementType::kF64) == 6U);
static_assert(static_cast<std::uint8_t>(RoundingMode::kTruncate) == 1U);
static_assert(static_cast<std::uint8_t>(ReformatMode::kPlanarToInterleaved) == 1U);

TEST(CvCudaTensor, WrapsHwcTensorWithoutChangingPointerOrMetadata) {
  std::array<std::byte, 24U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{2, 3, 4};
  constexpr std::array<std::int64_t, 3U> kStrides{12, 4, 1};

  auto view = describe_image_tensor(
      storage.data(), kShape, kStrides, kU8, kCudaDevice, TensorOperation::kResize, 1U);
  ASSERT_TRUE(view);
  EXPECT_EQ(view->batch, 1);
  EXPECT_EQ(view->height, 2);
  EXPECT_EQ(view->width, 3);
  EXPECT_EQ(view->channels, 4);
  EXPECT_STREQ(view->layout(), "HWC");

  auto wrapped = wrap_nvcv_tensor(*view);
  ASSERT_TRUE(wrapped);
  auto exported = wrapped->exportData<nvcv::TensorDataStridedCuda>();
  ASSERT_TRUE(exported);
  EXPECT_EQ(exported->basePtr(), reinterpret_cast<nvcv::Byte*>(storage.data()));
  EXPECT_EQ(exported->shape()[0], kShape[0]);
  EXPECT_EQ(exported->shape()[1], kShape[1]);
  EXPECT_EQ(exported->shape()[2], kShape[2]);
  EXPECT_EQ(exported->stride(0), kStrides[0]);
  EXPECT_EQ(exported->stride(1), kStrides[1]);
  EXPECT_EQ(exported->stride(2), kStrides[2]);
  EXPECT_EQ(static_cast<NVCVDataType>(wrapped->dtype()), NVCV_DATA_TYPE_U8);
  EXPECT_EQ(wrapped->layout(), nvcv::TENSOR_HWC);
}

TEST(CvCudaTensor, DescribesPitchedNhwcTensor) {
  std::array<std::byte, 160U> storage{};
  constexpr std::array<std::int64_t, 4U> kShape{2, 4, 5, 3};
  constexpr std::array<std::int64_t, 4U> kStrides{80, 20, 3, 1};

  auto view = describe_image_tensor(
      storage.data(), kShape, kStrides, kS8, kCudaDevice, TensorOperation::kCustomCrop, 2U);
  ASSERT_TRUE(view);
  EXPECT_EQ(view->batch, 2);
  EXPECT_EQ(view->height, 4);
  EXPECT_EQ(view->width, 5);
  EXPECT_EQ(view->channels, 3);
  EXPECT_STREQ(view->layout(), "NHWC");
}

TEST(CvCudaTensor, DescribesPlanarNchwTensor) {
  std::array<std::byte, 120U> storage{};
  constexpr std::array<std::int64_t, 4U> kShape{2, 3, 4, 5};
  constexpr std::array<std::int64_t, 4U> kStrides{60, 20, 5, 1};

  auto view = describe_image_tensor(storage.data(),
                                    kShape,
                                    kStrides,
                                    kU8,
                                    kCudaDevice,
                                    TensorOperation::kReformat,
                                    2U,
                                    ImageTensorLayout::kPlanar);
  ASSERT_TRUE(view);
  EXPECT_EQ(view->batch, 2);
  EXPECT_EQ(view->height, 4);
  EXPECT_EQ(view->width, 5);
  EXPECT_EQ(view->channels, 3);
  EXPECT_STREQ(view->layout(), "NCHW");
}

TEST(CvCudaTensor, RejectsPlanarLayoutsOutsideReformat) {
  std::array<std::byte, 12U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{3, 2, 2};
  constexpr std::array<std::int64_t, 3U> kStrides{4, 2, 1};
  constexpr std::array<TensorOperation, 4U> kInterleavedOnlyOperations{
      TensorOperation::kCopyMakeBorder,
      TensorOperation::kRemap,
      TensorOperation::kNormalize,
      TensorOperation::kConvertTo,
  };

  for (const TensorOperation operation : kInterleavedOnlyOperations) {
    auto view = describe_image_tensor(storage.data(),
                                      kShape,
                                      kStrides,
                                      kU8,
                                      kCudaDevice,
                                      operation,
                                      1U,
                                      ImageTensorLayout::kPlanar);
    ASSERT_FALSE(view) << "operation " << static_cast<int>(operation);
    EXPECT_EQ(view.error().code, ErrorCode::kInvalidArgument);
  }
}

TEST(CvCudaTensor, AcceptsTwoChannelF32RemapMap) {
  alignas(8) std::array<std::byte, 48U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{2, 3, 2};
  constexpr std::array<std::int64_t, 3U> kStrides{24, 8, 4};

  auto view = describe_image_tensor(
      storage.data(), kShape, kStrides, kF32, kCudaDevice, TensorOperation::kRemapMap, 1U);
  ASSERT_TRUE(view);
  EXPECT_EQ(view->batch, 1);
  EXPECT_EQ(view->channels, 2);
  EXPECT_STREQ(view->layout(), "HWC");
}

TEST(CvCudaTensor, AcceptsTwoChannelCustomCropInput) {
  std::array<std::byte, 8U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{2, 2, 2};
  constexpr std::array<std::int64_t, 3U> kStrides{4, 2, 1};

  auto view = describe_image_tensor(
      storage.data(), kShape, kStrides, kU8, kCudaDevice, TensorOperation::kCustomCrop, 1U);
  ASSERT_TRUE(view);
  EXPECT_EQ(view->channels, 2);
  EXPECT_STREQ(view->layout(), "HWC");
}

TEST(CvCudaTensor, RejectsMisalignedF32x4BaseAddress) {
  alignas(16) std::array<std::byte, 160U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{2, 2, 4};
  constexpr std::array<std::int64_t, 3U> kStrides{32, 16, 4};

  auto view = describe_image_tensor(
      storage.data() + 4U, kShape, kStrides, kF32, kCudaDevice, TensorOperation::kResize, 1U);
  ASSERT_FALSE(view);
  EXPECT_EQ(view.error().code, ErrorCode::kInvalidArgument);
}

TEST(CvCudaTensor, RejectsMisalignedF32x4RowStride) {
  alignas(16) std::array<std::byte, 160U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{2, 2, 4};
  constexpr std::array<std::int64_t, 3U> kStrides{36, 16, 4};

  auto view = describe_image_tensor(
      storage.data(), kShape, kStrides, kF32, kCudaDevice, TensorOperation::kResize, 1U);
  ASSERT_FALSE(view);
  EXPECT_EQ(view.error().code, ErrorCode::kInvalidArgument);
}

TEST(CvCudaTensor, RejectsMisalignedF32x4SampleStride) {
  alignas(16) std::array<std::byte, 160U> storage{};
  constexpr std::array<std::int64_t, 4U> kShape{2, 2, 2, 4};
  constexpr std::array<std::int64_t, 4U> kStrides{68, 32, 16, 4};

  auto view = describe_image_tensor(
      storage.data(), kShape, kStrides, kF32, kCudaDevice, TensorOperation::kResize, 2U);
  ASSERT_FALSE(view);
  EXPECT_EQ(view.error().code, ErrorCode::kInvalidArgument);
}

TEST(CvCudaTensor, AcceptsF32x3WithFourByteAccessAlignment) {
  alignas(16) std::array<std::byte, 32U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{2, 1, 3};
  constexpr std::array<std::int64_t, 3U> kStrides{12, 12, 4};

  auto view = describe_image_tensor(
      storage.data() + 4U, kShape, kStrides, kF32, kCudaDevice, TensorOperation::kResize, 1U);
  ASSERT_TRUE(view);
  EXPECT_EQ(view->channels, 3);
}

TEST(CvCudaTensor, RejectsMisalignedPlanarF32AdvancingStrides) {
  alignas(16) std::array<std::byte, 256U> storage{};
  constexpr std::array<std::int64_t, 4U> kShape{2, 4, 2, 2};
  constexpr std::array<std::int64_t, 4U> kMisalignedRowStrides{96, 24, 10, 4};
  constexpr std::array<std::int64_t, 4U> kMisalignedPlaneStrides{96, 22, 8, 4};
  constexpr std::array<std::int64_t, 4U> kMisalignedSampleStrides{90, 20, 8, 4};

  EXPECT_FALSE(describe_image_tensor(storage.data(),
                                     kShape,
                                     kMisalignedRowStrides,
                                     kF32,
                                     kCudaDevice,
                                     TensorOperation::kReformat,
                                     2U,
                                     ImageTensorLayout::kPlanar));
  EXPECT_FALSE(describe_image_tensor(storage.data(),
                                     kShape,
                                     kMisalignedPlaneStrides,
                                     kF32,
                                     kCudaDevice,
                                     TensorOperation::kReformat,
                                     2U,
                                     ImageTensorLayout::kPlanar));
  EXPECT_FALSE(describe_image_tensor(storage.data(),
                                     kShape,
                                     kMisalignedSampleStrides,
                                     kF32,
                                     kCudaDevice,
                                     TensorOperation::kReformat,
                                     2U,
                                     ImageTensorLayout::kPlanar));
}

TEST(CvCudaTensor, RejectsTwoChannelPlanarCustomCropInput) {
  std::array<std::byte, 8U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{2, 2, 2};
  constexpr std::array<std::int64_t, 3U> kStrides{4, 2, 1};

  auto view = describe_image_tensor(storage.data(),
                                    kShape,
                                    kStrides,
                                    kU8,
                                    kCudaDevice,
                                    TensorOperation::kCustomCrop,
                                    1U,
                                    ImageTensorLayout::kPlanar);
  ASSERT_FALSE(view);
  EXPECT_EQ(view.error().code, ErrorCode::kInvalidArgument);
}

TEST(CvCudaTensor, RejectsU32CustomCropInput) {
  std::array<std::byte, 16U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{2, 2, 1};
  constexpr std::array<std::int64_t, 3U> kStrides{8, 4, 4};

  auto view = describe_image_tensor(
      storage.data(), kShape, kStrides, kU32, kCudaDevice, TensorOperation::kCustomCrop, 1U);
  ASSERT_FALSE(view);
  EXPECT_EQ(view.error().code, ErrorCode::kInvalidArgument);
}

TEST(CvCudaTensor, RejectsInvalidOperationDtypeAndChannelPairs) {
  std::array<std::byte, 96U> storage{};
  constexpr std::array<std::int64_t, 3U> kInterleavedShape{2, 2, 3};
  constexpr std::array<std::int64_t, 3U> kF32Strides{24, 12, 4};
  constexpr std::array<std::int64_t, 3U> kF64Strides{48, 24, 8};
  constexpr std::array<std::int64_t, 3U> kPlanarShape{2, 2, 2};
  constexpr std::array<std::int64_t, 3U> kPlanarStrides{4, 2, 1};

  EXPECT_FALSE(describe_image_tensor(storage.data(),
                                     kPlanarShape,
                                     kPlanarStrides,
                                     kU8,
                                     kCudaDevice,
                                     TensorOperation::kCopyMakeBorder,
                                     1U,
                                     ImageTensorLayout::kPlanar));
  EXPECT_FALSE(describe_image_tensor(storage.data(),
                                     kInterleavedShape,
                                     kF64Strides,
                                     kF64,
                                     kCudaDevice,
                                     TensorOperation::kNormalize,
                                     1U));
  EXPECT_FALSE(describe_image_tensor(storage.data(),
                                     kInterleavedShape,
                                     kF32Strides,
                                     kF32,
                                     kCudaDevice,
                                     TensorOperation::kRemap,
                                     1U));
}

TEST(CvCudaTensor, WrapsExplicitNwTensorWithoutChangingPointer) {
  std::array<std::byte, 16U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{1, 2, 4};
  constexpr std::array<std::int64_t, 3U> kStrides{16, 8, 2};

  auto wrapped = wrap_nvcv_tensor(storage.data(), kShape, kStrides, kS16, "NWC");
  ASSERT_TRUE(wrapped);
  auto exported = wrapped->exportData<nvcv::TensorDataStridedCuda>();
  ASSERT_TRUE(exported);
  EXPECT_EQ(exported->basePtr(), reinterpret_cast<nvcv::Byte*>(storage.data()));
  EXPECT_EQ(exported->layout(), nvcv::TENSOR_NWC);
  EXPECT_EQ(exported->shape()[0], 1);
  EXPECT_EQ(exported->shape()[1], 2);
  EXPECT_EQ(exported->shape()[2], 4);
  EXPECT_EQ(exported->stride(0), 16);
  EXPECT_EQ(exported->stride(1), 8);
  EXPECT_EQ(exported->stride(2), 2);
  EXPECT_EQ(static_cast<NVCVDataType>(wrapped->dtype()), NVCV_DATA_TYPE_S16);
}

TEST(CvCudaTensor, EnforcesOperationDtypes) {
  alignas(4) std::array<std::byte, 4U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{2, 2, 1};
  constexpr std::array<std::int64_t, 3U> kStrides{2, 1, 1};

  auto resize = describe_image_tensor(
      storage.data(), kShape, kStrides, kS8, kCudaDevice, TensorOperation::kResize, 1U);
  ASSERT_FALSE(resize);
  EXPECT_EQ(resize.error().code, ErrorCode::kInvalidArgument);

  auto custom_crop = describe_image_tensor(
      storage.data(), kShape, kStrides, kS8, kCudaDevice, TensorOperation::kCustomCrop, 1U);
  EXPECT_TRUE(custom_crop);

  constexpr std::array<std::int64_t, 3U> kFloatStrides{8, 4, 4};
  auto float_resize = describe_image_tensor(
      storage.data(), kShape, kFloatStrides, kF32, kCudaDevice, TensorOperation::kResize, 1U);
  EXPECT_TRUE(float_resize);
}

TEST(CvCudaTensor, RejectsInvalidShapeStorageAndStrides) {
  std::array<std::byte, 24U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{2, 3, 4};
  constexpr std::array<std::int64_t, 3U> kStrides{12, 4, 1};
  constexpr std::array<std::int64_t, 3U> kBadChannels{2, 3, 2};
  constexpr std::array<std::int64_t, 3U> kBadStrides{12, 5, 1};

  EXPECT_FALSE(describe_image_tensor(
      nullptr, kShape, kStrides, kU8, kCudaDevice, TensorOperation::kResize, 1U));
  EXPECT_FALSE(describe_image_tensor(
      storage.data(), kShape, kStrides, kU8, DLDevice{kDLCPU, 0}, TensorOperation::kResize, 1U));
  EXPECT_FALSE(describe_image_tensor(
      storage.data(), kBadChannels, kStrides, kU8, kCudaDevice, TensorOperation::kResize, 1U));
  EXPECT_FALSE(describe_image_tensor(
      storage.data(), kShape, kBadStrides, kU8, kCudaDevice, TensorOperation::kResize, 1U));
  EXPECT_FALSE(describe_image_tensor(storage.data(),
                                     kShape,
                                     kStrides,
                                     kU8,
                                     kCudaDevice,
                                     TensorOperation::kResize,
                                     kMaximumBatchSize + 1U));

  constexpr std::array<std::int64_t, 4U> kBatchedShape{3, 2, 3, 4};
  constexpr std::array<std::int64_t, 4U> kBatchedStrides{24, 12, 4, 1};
  EXPECT_FALSE(describe_image_tensor(storage.data(),
                                     kBatchedShape,
                                     kBatchedStrides,
                                     kU8,
                                     kCudaDevice,
                                     TensorOperation::kResize,
                                     2U));
}

TEST(CvCudaTensor, RejectsStridesAndTensorSpansOutsideCvCuda32BitAddressing) {
  std::array<std::byte, 1U> storage{};
  constexpr std::array<std::int64_t, 3U> kShape{1, 1, 1};
  constexpr std::array<std::int64_t, 3U> kMaximumStrides{INT32_MAX, 1, 1};
  constexpr std::array<std::int64_t, 3U> kOversizedStrides{
      static_cast<std::int64_t>(INT32_MAX) + 1, 1, 1};

  EXPECT_TRUE(describe_image_tensor(
      storage.data(), kShape, kMaximumStrides, kU8, kCudaDevice, TensorOperation::kResize, 1U));
  EXPECT_FALSE(describe_image_tensor(
      storage.data(), kShape, kOversizedStrides, kU8, kCudaDevice, TensorOperation::kResize, 1U));

  constexpr std::array<std::int64_t, 4U> kBatchedShape{2, 1, 1, 1};
  constexpr std::array<std::int64_t, 4U> kBatchedStrides{INT32_MAX, 1, 1, 1};
  EXPECT_FALSE(describe_image_tensor(storage.data(),
                                     kBatchedShape,
                                     kBatchedStrides,
                                     kU8,
                                     kCudaDevice,
                                     TensorOperation::kCustomCrop,
                                     2U));
}

// REQ: VISION-011
TEST(CvCudaTensor, DerivesContiguousStridesAndCheckedAllocationBound) {
  constexpr std::array<std::int64_t, 4U> kShape{2, 4, 5, 3};
  auto strides = contiguous_byte_strides(kShape, kF32);
  ASSERT_TRUE(strides);
  EXPECT_EQ((*strides)[0], 240);
  EXPECT_EQ((*strides)[1], 60);
  EXPECT_EQ((*strides)[2], 12);
  EXPECT_EQ((*strides)[3], 4);

  auto byte_span = projected_image_output_byte_span(640, 480, 8U, sizeof(float));
  ASSERT_TRUE(byte_span);
  EXPECT_EQ(*byte_span, 640U * 480U * 8U * 4U * sizeof(float));

  const auto overflow = projected_image_output_byte_span(std::numeric_limits<std::int32_t>::max(),
                                                         std::numeric_limits<std::int32_t>::max(),
                                                         kMaximumBatchSize,
                                                         sizeof(double));
  ASSERT_FALSE(overflow);
  EXPECT_EQ(overflow.error().code, ErrorCode::kArithmeticOverflow);
  const auto exceeds_cvcuda = projected_image_output_byte_span(32768, 32768, 1U, sizeof(float));
  ASSERT_FALSE(exceeds_cvcuda);
  EXPECT_EQ(exceeds_cvcuda.error().code, ErrorCode::kNotSupported);
  EXPECT_TRUE(projected_image_output_byte_span(1, 1, kMaximumBatchSize, sizeof(float)));
  const auto invalid_batch =
      projected_image_output_byte_span(1, 1, kMaximumBatchSize + 1U, sizeof(float));
  ASSERT_FALSE(invalid_batch);
  EXPECT_EQ(invalid_batch.error().code, ErrorCode::kInvalidArgument);
}

Tensor make_tensor_with_dlpack_strides(void* data, std::vector<std::int64_t> shape,
                                       std::vector<std::int64_t> element_strides,
                                       DLDataType dtype) {
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = std::move(shape);
  context->dl_strides = std::move(element_strides);
  context->tensor.dl_tensor.data = data;
  context->tensor.dl_tensor.device = kCudaDevice;
  context->tensor.dl_tensor.ndim = static_cast<std::int32_t>(context->dl_shape.size());
  context->tensor.dl_tensor.dtype = dtype;
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides =
      context->dl_strides.empty() ? nullptr : context->dl_strides.data();
  context->tensor.dl_tensor.byte_offset = 0U;
  context->tensor.manager_ctx = nullptr;
  context->tensor.deleter = nullptr;
  return Tensor{std::move(context)};
}

TEST(CvCudaTensor, ConvertsExplicitDlpackElementStridesWithoutDynamicStrideStorage) {
  alignas(4) std::array<std::byte, 40U> storage{};
  Tensor tensor = make_tensor_with_dlpack_strides(storage.data(), {2, 3, 1}, {5, 1, 1}, kF32);

  auto byte_strides = fixed_tensor_byte_strides(tensor);
  ASSERT_TRUE(byte_strides);
  constexpr std::array<std::int64_t, 3U> kExpectedStrides{20, 4, 4};
  EXPECT_TRUE(std::ranges::equal(byte_strides->span(), kExpectedStrides));

  auto view = describe_image_tensor(tensor, TensorOperation::kResize, 1U);
  ASSERT_TRUE(view);
  EXPECT_EQ(view->byte_strides[0], 20);
  EXPECT_EQ(view->byte_strides[1], 4);
  EXPECT_EQ(view->byte_strides[2], 4);
}

TEST(CvCudaTensor, DerivesImplicitContiguousDlpackStridesIntoFixedStorage) {
  alignas(2) std::array<std::byte, 12U> storage{};
  Tensor tensor =
      make_tensor_with_dlpack_strides(storage.data(), {2, 3, 1}, {}, DLDataType{kDLUInt, 16U, 1U});

  auto byte_strides = fixed_tensor_byte_strides(tensor);
  ASSERT_TRUE(byte_strides);
  constexpr std::array<std::int64_t, 3U> kExpectedStrides{6, 2, 2};
  EXPECT_TRUE(std::ranges::equal(byte_strides->span(), kExpectedStrides));

  auto view = describe_image_tensor(tensor, TensorOperation::kResize, 1U);
  ASSERT_TRUE(view);
  EXPECT_EQ(view->byte_strides[0], 6);
  EXPECT_EQ(view->byte_strides[1], 2);
  EXPECT_EQ(view->byte_strides[2], 2);
}

void expect_nms_alignment_failure(Tensor boxes, Tensor scores) {
  NonMaximumSuppressionOp op;
  holoscan::test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_initial_params(
      NonMaximumSuppressionOpParams{.max_boxes = 2, .max_batch_size = 1U}));
  ASSERT_TRUE(fixture.bind_value(op.boxes, std::move(boxes)));
  ASSERT_TRUE(fixture.bind_value(op.scores, std::move(scores)));
  ASSERT_NO_THROW(op.start());

  auto result = fixture.invoke_once(holoscan::detail::SyntheticActivationOptions{
      .cuda_stream = reinterpret_cast<cudaStream_t>(std::uintptr_t{1U}),
  });
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(result.error().message.find("alignment"), std::string::npos);
}

TEST(NonMaximumSuppressionOpValidation, RejectsMisalignedBoxesBaseAddress) {
  alignas(8) std::array<std::byte, 40U> boxes_storage{};
  alignas(4) std::array<std::byte, 8U> scores_storage{};
  Tensor boxes =
      make_tensor_with_dlpack_strides(boxes_storage.data() + 2U, {1, 2, 4}, {8, 4, 1}, kS16);
  Tensor scores = make_tensor_with_dlpack_strides(scores_storage.data(), {1, 2}, {2, 1}, kF32);

  expect_nms_alignment_failure(std::move(boxes), std::move(scores));
}

TEST(NonMaximumSuppressionOpValidation, RejectsMisalignedBoxesBatchStride) {
  alignas(8) std::array<std::byte, 40U> boxes_storage{};
  alignas(4) std::array<std::byte, 8U> scores_storage{};
  Tensor boxes = make_tensor_with_dlpack_strides(boxes_storage.data(), {1, 2, 4}, {9, 4, 1}, kS16);
  Tensor scores = make_tensor_with_dlpack_strides(scores_storage.data(), {1, 2}, {2, 1}, kF32);

  expect_nms_alignment_failure(std::move(boxes), std::move(scores));
}

TEST(NonMaximumSuppressionOpValidation, RejectsMisalignedScoresBaseAddress) {
  alignas(8) std::array<std::byte, 32U> boxes_storage{};
  alignas(4) std::array<std::byte, 12U> scores_storage{};
  Tensor boxes = make_tensor_with_dlpack_strides(boxes_storage.data(), {1, 2, 4}, {8, 4, 1}, kS16);
  Tensor scores = make_tensor_with_dlpack_strides(scores_storage.data() + 2U, {1, 2}, {2, 1}, kF32);

  expect_nms_alignment_failure(std::move(boxes), std::move(scores));
}

TEST(CvCudaTensor, ValidatesCropContainmentWithoutSignedOverflow) {
  EXPECT_TRUE(validate_crop_rectangle(8, 6, 2, 1, 6, 5));
  EXPECT_FALSE(validate_crop_rectangle(8, 6, 3, 1, 6, 5));
  EXPECT_FALSE(validate_crop_rectangle(8, 6, 2, 2, 6, 5));
  EXPECT_FALSE(validate_crop_rectangle(8, 6, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX));
}

}  // namespace
}  // namespace holoscan::vision::detail

namespace holoscan::vision {
namespace {

template <typename Op>
void expect_uninitialized_resources(std::string_view name) {
  SCOPED_TRACE(name);
  Op op;
  holoscan::test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_initial_params(typename Op::Params{}));
  const auto result = fixture.invoke_once();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(result.error().message, std::string(name) + " resources are not initialized");
}

// REQ: VISION-008
TEST(VisionOperators, UninitializedImageResourcesHaveAccurateDiagnostics) {
  expect_uninitialized_resources<ResizeOp>("ResizeOp");
  expect_uninitialized_resources<ConvertToOp>("ConvertToOp");
  expect_uninitialized_resources<CopyMakeBorderOp>("CopyMakeBorderOp");
  expect_uninitialized_resources<CustomCropOp>("CustomCropOp");
  expect_uninitialized_resources<NormalizeOp>("NormalizeOp");
  expect_uninitialized_resources<ReformatOp>("ReformatOp");
  expect_uninitialized_resources<RemapOp>("RemapOp");
  expect_uninitialized_resources<BayerDemosaicOp>("BayerDemosaicOp");
}

template <typename Params>
void expect_fields(std::initializer_list<std::string_view> field_names, ParamMutability mutability,
                   bool required_on_load = false) {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<Params>();
  ASSERT_NE(schema, nullptr);
  for (const std::string_view field_name : field_names) {
    const auto field =
        std::ranges::find(schema->fields, field_name, &ParamFieldDescriptor::canonical_name);
    ASSERT_NE(field, schema->fields.end()) << field_name;
    EXPECT_EQ(field->mutability, mutability) << field_name;
    EXPECT_EQ(field->required_on_load, required_on_load) << field_name;
  }
}

template <typename Params, typename Enum>
void expect_named_enum_domain(std::string_view field_name,
                              std::initializer_list<NamedParamEnumValue<Enum>> expected_choices) {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<Params>();
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->version, 2U);
  const auto field =
      std::ranges::find(schema->fields, field_name, &ParamFieldDescriptor::canonical_name);
  ASSERT_NE(field, schema->fields.end()) << field_name;
  ASSERT_EQ(field->enum_choices.count, expected_choices.size()) << field_name;
  ASSERT_EQ(field->constraints.allowed_value_count, expected_choices.size()) << field_name;

  std::size_t index = 0U;
  for (const NamedParamEnumValue<Enum>& expected : expected_choices) {
    const ParamEnumChoice& actual = field->enum_choices.values[index];
    const ParamScalarValue expected_value = ParamScalarValue::from(expected.value);
    EXPECT_EQ(actual.name, expected.name) << field_name << '[' << index << ']';
    EXPECT_EQ(actual.value.kind, expected_value.kind) << field_name << '[' << index << ']';
    EXPECT_EQ(actual.value.bit_width, expected_value.bit_width)
        << field_name << '[' << index << ']';
    EXPECT_EQ(actual.value.raw_bits, expected_value.raw_bits) << field_name << '[' << index << ']';
    ++index;
  }
}

template <typename Params, auto Member>
void expect_named_and_numeric_enum_assignment(
    std::string_view field_name, std::string_view label,
    std::remove_cvref_t<decltype(std::declval<Params>().*Member)> expected_value) {
  using Enum = std::remove_cvref_t<decltype(std::declval<Params>().*Member)>;
  using Underlying = std::underlying_type_t<Enum>;

  const std::array named_assignments{
      authoring::ParameterAssignment{field_name, authoring::ParameterValue::from(label)}};
  auto named = authoring::materialize_parameters(
      param_type_descriptor<Params>(), param_schema_descriptor<Params>(), named_assignments);
  ASSERT_TRUE(named) << named.error().message;
  EXPECT_EQ(static_cast<const Params*>(named->data())->*Member, expected_value);

  const std::array numeric_assignments{authoring::ParameterAssignment{
      field_name,
      authoring::ParameterValue::from(
          static_cast<std::uint64_t>(static_cast<Underlying>(expected_value)))}};
  auto numeric = authoring::materialize_parameters(
      param_type_descriptor<Params>(), param_schema_descriptor<Params>(), numeric_assignments);
  ASSERT_TRUE(numeric) << numeric.error().message;
  EXPECT_EQ(static_cast<const Params*>(numeric->data())->*Member, expected_value);
}

// REQ: AUTHOR-018
TEST(VisionParameterSchema, PublishesStableNamedEnumDomains) {
  expect_named_enum_domain<BayerDemosaicOpParams>(
      "bayer_grid_pos",
      {param_enum_value("bggr", BayerGridPosition::kBGGR),
       param_enum_value("rggb", BayerGridPosition::kRGGB),
       param_enum_value("gbrg", BayerGridPosition::kGBRG),
       param_enum_value("grbg", BayerGridPosition::kGRBG)});
  expect_named_enum_domain<ConvertToOpParams>(
      "output_type",
      {param_enum_value("uint8", VisionElementType::kU8),
       param_enum_value("int8", VisionElementType::kS8),
       param_enum_value("uint16", VisionElementType::kU16),
       param_enum_value("int16", VisionElementType::kS16),
       param_enum_value("int32", VisionElementType::kS32),
       param_enum_value("float32", VisionElementType::kF32),
       param_enum_value("float64", VisionElementType::kF64)});
  expect_named_enum_domain<ConvertToOpParams>(
      "rounding",
      {param_enum_value("nearest", RoundingMode::kNearest),
       param_enum_value("truncate", RoundingMode::kTruncate)});
  expect_named_enum_domain<CopyMakeBorderOpParams>(
      "border_mode",
      {param_enum_value("constant", BorderMode::kConstant),
       param_enum_value("replicate", BorderMode::kReplicate),
       param_enum_value("reflect", BorderMode::kReflect),
       param_enum_value("wrap", BorderMode::kWrap),
       param_enum_value("reflect_101", BorderMode::kReflect101)});
  expect_named_enum_domain<ResizeOpParams>(
      "interpolation",
      {param_enum_value("nearest", InterpolationMode::kNearest),
       param_enum_value("linear", InterpolationMode::kLinear),
       param_enum_value("cubic", InterpolationMode::kCubic),
       param_enum_value("area", InterpolationMode::kArea)});
  expect_named_enum_domain<ReformatOpParams>(
      "mode",
      {param_enum_value("interleaved_to_planar", ReformatMode::kInterleavedToPlanar),
       param_enum_value("planar_to_interleaved", ReformatMode::kPlanarToInterleaved)});
  expect_named_enum_domain<RemapOpParams>("input_interpolation",
                                          {param_enum_value("nearest", InterpolationMode::kNearest),
                                           param_enum_value("linear", InterpolationMode::kLinear),
                                           param_enum_value("cubic", InterpolationMode::kCubic)});
  expect_named_enum_domain<RemapOpParams>("map_interpolation",
                                          {param_enum_value("nearest", InterpolationMode::kNearest),
                                           param_enum_value("linear", InterpolationMode::kLinear),
                                           param_enum_value("cubic", InterpolationMode::kCubic)});
  expect_named_enum_domain<RemapOpParams>(
      "map_value_mode",
      {param_enum_value("absolute", RemapMapValueMode::kAbsolute),
       param_enum_value("absolute_normalized", RemapMapValueMode::kAbsoluteNormalized),
       param_enum_value("relative_normalized", RemapMapValueMode::kRelativeNormalized)});
  expect_named_enum_domain<RemapOpParams>(
      "border_mode",
      {param_enum_value("constant", BorderMode::kConstant),
       param_enum_value("replicate", BorderMode::kReplicate),
       param_enum_value("reflect", BorderMode::kReflect),
       param_enum_value("wrap", BorderMode::kWrap),
       param_enum_value("reflect_101", BorderMode::kReflect101)});
  expect_named_enum_domain<RemapOpParams>(
      "map_synchronization",
      {param_enum_value("latest", RemapMapSynchronizationMode::kLatest),
       param_enum_value("exact", RemapMapSynchronizationMode::kExact)});
}

// REQ: AUTHOR-018
TEST(VisionParameterSchema, MaterializesNamedAndNumericEnumAssignments) {
  expect_named_and_numeric_enum_assignment<BayerDemosaicOpParams,
                                           &BayerDemosaicOpParams::bayer_grid_pos>(
      "bayer_grid_pos", "rggb", BayerGridPosition::kRGGB);
  expect_named_and_numeric_enum_assignment<ConvertToOpParams, &ConvertToOpParams::output_type>(
      "output_type", "float64", VisionElementType::kF64);
  expect_named_and_numeric_enum_assignment<ConvertToOpParams, &ConvertToOpParams::rounding>(
      "rounding", "truncate", RoundingMode::kTruncate);
  expect_named_and_numeric_enum_assignment<CopyMakeBorderOpParams,
                                           &CopyMakeBorderOpParams::border_mode>(
      "border_mode", "reflect_101", BorderMode::kReflect101);
  expect_named_and_numeric_enum_assignment<ResizeOpParams, &ResizeOpParams::interpolation>(
      "interpolation", "area", InterpolationMode::kArea);
  expect_named_and_numeric_enum_assignment<ReformatOpParams, &ReformatOpParams::mode>(
      "mode", "planar_to_interleaved", ReformatMode::kPlanarToInterleaved);
  expect_named_and_numeric_enum_assignment<RemapOpParams, &RemapOpParams::input_interpolation>(
      "input_interpolation", "cubic", InterpolationMode::kCubic);
  expect_named_and_numeric_enum_assignment<RemapOpParams, &RemapOpParams::map_interpolation>(
      "map_interpolation", "nearest", InterpolationMode::kNearest);
  expect_named_and_numeric_enum_assignment<RemapOpParams, &RemapOpParams::map_value_mode>(
      "map_value_mode", "relative_normalized", RemapMapValueMode::kRelativeNormalized);
  expect_named_and_numeric_enum_assignment<RemapOpParams, &RemapOpParams::border_mode>(
      "border_mode", "replicate", BorderMode::kReplicate);
  expect_named_and_numeric_enum_assignment<RemapOpParams, &RemapOpParams::map_synchronization>(
      "map_synchronization", "exact", RemapMapSynchronizationMode::kExact);
}

static_assert(std::derived_from<ResizeOp, Operator<ResizeOpParams>>);
static_assert(std::same_as<decltype(ResizeOp::input), Input<schema::ImageT>>);
static_assert(std::same_as<decltype(ResizeOp::output), Output<schema::ImageT>>);
static_assert(!std::copy_constructible<ResizeOp>);
static_assert(!std::move_constructible<ResizeOp>);

// REQ: VISION-011
TEST(ResizeOp, ParameterSchemaAcceptsValidDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(ResizeOpParams{}).ok());

  ResizeOpParams params;
  params.width = 0;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.height = -1;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.max_batch_size = 0U;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.interpolation = InterpolationMode::kCubic;
  EXPECT_TRUE(validate_params(params).ok());

  params = {};
  params.interpolation = static_cast<InterpolationMode>(UINT8_MAX);
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);
}

// REQ: VISION-003
TEST(ResizeOp, OwnsStableNamedImagePorts) {
  ResizeOp op;
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<CustomCropOp, Operator<CustomCropOpParams>>);
static_assert(std::same_as<decltype(CustomCropOp::input), Input<schema::ImageT>>);
static_assert(std::same_as<decltype(CustomCropOp::output), Output<schema::ImageT>>);
static_assert(!std::copy_constructible<CustomCropOp>);
static_assert(!std::move_constructible<CustomCropOp>);

// REQ: VISION-011
TEST(CustomCropOp, ParameterSchemaAcceptsValidDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(CustomCropOpParams{}).ok());

  CustomCropOpParams params;
  params.x = -1;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.y = -1;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.width = 0;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.height = 0;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.max_batch_size = 0U;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);
}

// REQ: VISION-003
TEST(CustomCropOp, OwnsStableNamedImagePorts) {
  CustomCropOp op;
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<CopyMakeBorderOp, Operator<CopyMakeBorderOpParams>>);
static_assert(std::same_as<decltype(CopyMakeBorderOp::input), Input<schema::ImageT>>);
static_assert(std::same_as<decltype(CopyMakeBorderOp::output), Output<schema::ImageT>>);
static_assert(!std::copy_constructible<CopyMakeBorderOp>);
static_assert(!std::move_constructible<CopyMakeBorderOp>);

// REQ: VISION-011
TEST(CopyMakeBorderOp, ParameterSchemaAcceptsDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(CopyMakeBorderOpParams{}).ok());

  CopyMakeBorderOpParams params;
  params.max_width = 0;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.max_height = 0;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.max_batch_size = 0U;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.top = -1;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.border_mode = static_cast<BorderMode>(UINT8_MAX);
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);

  params = {};
  params.border_value = {0.F, 1.F, 2.F, 3.F, 4.F};
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kCardinalityExceeded);

  params = {};
  params.border_value = {std::numeric_limits<float>::infinity()};
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kValidatorRejected);
}

// REQ: VISION-003
TEST(CopyMakeBorderOp, OwnsStableNamedImagePorts) {
  CopyMakeBorderOp op;
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<RemapOp, Operator<RemapOpParams>>);
static_assert(std::same_as<decltype(RemapOp::input), Input<schema::ImageT>>);
static_assert(std::same_as<decltype(RemapOp::map), Input<Tensor>>);
static_assert(std::same_as<decltype(RemapOp::output), Output<schema::ImageT>>);
static_assert(!std::copy_constructible<RemapOp>);
static_assert(!std::move_constructible<RemapOp>);

// REQ: VISION-011
TEST(RemapOp, ParameterSchemaAcceptsDefaultsAndRejectsInvalidValues) {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<RemapOpParams>();
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->version, 2U);
  EXPECT_TRUE(validate_params(RemapOpParams{}).ok());

  RemapOpParams params;
  params.width = 0;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.height = 0;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.max_batch_size = 0U;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.input_interpolation = InterpolationMode::kArea;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);

  params = {};
  params.map_interpolation = InterpolationMode::kArea;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);

  params = {};
  params.map_value_mode = static_cast<RemapMapValueMode>(UINT8_MAX);
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);

  params = {};
  params.border_value = {0.F, 1.F, 2.F, 3.F, 4.F};
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kCardinalityExceeded);

  params = {};
  params.border_value = {std::numeric_limits<float>::quiet_NaN()};
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kValidatorRejected);

  params = {};
  params.map_max_age = std::chrono::nanoseconds::zero();
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.map_synchronization = static_cast<RemapMapSynchronizationMode>(UINT8_MAX);
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);
}

// REQ: VISION-003
TEST(RemapOp, OwnsStableNamedImagePorts) {
  RemapOp op;
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.map.name(), "map");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<NormalizeOp, Operator<NormalizeOpParams>>);
static_assert(std::same_as<decltype(NormalizeOp::input), Input<schema::ImageT>>);
static_assert(std::same_as<decltype(NormalizeOp::output), Output<schema::ImageT>>);
static_assert(!std::copy_constructible<NormalizeOp>);
static_assert(!std::move_constructible<NormalizeOp>);

TEST(NormalizeOp, ParameterSchemaAcceptsDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(NormalizeOpParams{}).ok());

  NormalizeOpParams params;
  params.epsilon = -1.F;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.scale_is_stddev = true;
  params.epsilon = 0.F;
  EXPECT_TRUE(validate_params(params).ok());

  params = {};
  params.base = {0.F, 1.F, 2.F, 3.F, 4.F};
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kCardinalityExceeded);

  params = {};
  params.scale = {0.F, 1.F, 2.F, 3.F, 4.F};
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kCardinalityExceeded);

  params = {};
  params.base = {std::numeric_limits<float>::quiet_NaN()};
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kValidatorRejected);

  params = {};
  params.scale = {std::numeric_limits<float>::infinity()};
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kValidatorRejected);

  params = {};
  params.global_scale = -std::numeric_limits<float>::infinity();
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kValidatorRejected);

  params = {};
  params.shift = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kValidatorRejected);

  params = {};
  params.epsilon = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);
}

TEST(NormalizeOp, RejectsZeroStandardDeviationOnlyWithoutEpsilon) {
  Graph graph{"vision-normalize-standard-deviation-test"};
  auto normalize = graph.op<NormalizeOp>("normalize");
  NormalizeOpParams params;
  params.scale = {0.F};
  params.scale_is_stddev = true;

  bool rejected = false;
  try {
    normalize.set_params(params);
  } catch (const RuntimeError& error) {
    rejected = true;
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
  }
  EXPECT_TRUE(rejected);

  params.epsilon = 1e-6F;
  EXPECT_NO_THROW(normalize.set_params(params));

  params.epsilon = 0.F;
  params.scale_is_stddev = false;
  EXPECT_NO_THROW(normalize.set_params(params));
}

// REQ: VISION-003
TEST(NormalizeOp, OwnsStableNamedImagePorts) {
  NormalizeOp op;
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<ConvertToOp, Operator<ConvertToOpParams>>);
static_assert(std::same_as<decltype(ConvertToOp::input), Input<schema::ImageT>>);
static_assert(std::same_as<decltype(ConvertToOp::output), Output<schema::ImageT>>);
static_assert(!std::copy_constructible<ConvertToOp>);
static_assert(!std::move_constructible<ConvertToOp>);

TEST(ConvertToOp, ParameterSchemaAcceptsDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(ConvertToOpParams{}).ok());

  ConvertToOpParams params;
  params.output_type = VisionElementType::kF64;
  params.rounding = RoundingMode::kTruncate;
  EXPECT_TRUE(validate_params(params).ok());

  params = {};
  params.output_type = static_cast<VisionElementType>(UINT8_MAX);
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);

  params = {};
  params.rounding = static_cast<RoundingMode>(UINT8_MAX);
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);

  params = {};
  params.alpha = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kValidatorRejected);

  params = {};
  params.beta = std::numeric_limits<double>::infinity();
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kValidatorRejected);
}

// REQ: VISION-003
TEST(ConvertToOp, OwnsStableNamedImagePorts) {
  ConvertToOp op;
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<ReformatOp, Operator<ReformatOpParams>>);
static_assert(std::same_as<decltype(ReformatOp::input), Input<schema::ImageT>>);
static_assert(std::same_as<decltype(ReformatOp::output), Output<schema::ImageT>>);
static_assert(!std::copy_constructible<ReformatOp>);
static_assert(!std::move_constructible<ReformatOp>);

TEST(ReformatOp, ParameterSchemaAcceptsDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(ReformatOpParams{}).ok());

  ReformatOpParams params;
  params.mode = ReformatMode::kPlanarToInterleaved;
  EXPECT_TRUE(validate_params(params).ok());

  params = {};
  params.mode = static_cast<ReformatMode>(UINT8_MAX);
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);
}

// REQ: VISION-003
TEST(ReformatOp, OwnsStableNamedImagePorts) {
  ReformatOp op;
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<NonMaximumSuppressionOp, Operator<NonMaximumSuppressionOpParams>>);
static_assert(std::same_as<decltype(NonMaximumSuppressionOp::boxes), Input<Tensor>>);
static_assert(std::same_as<decltype(NonMaximumSuppressionOp::scores), Input<Tensor>>);
static_assert(std::same_as<decltype(NonMaximumSuppressionOp::mask), Output<Tensor>>);
static_assert(!std::copy_constructible<NonMaximumSuppressionOp>);
static_assert(!std::move_constructible<NonMaximumSuppressionOp>);

// REQ: VISION-011
TEST(NonMaximumSuppressionOp, ParameterSchemaAcceptsDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(NonMaximumSuppressionOpParams{}).ok());

  NonMaximumSuppressionOpParams params;
  params.max_boxes = 0;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.max_batch_size = 0U;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.score_threshold = 1.F;
  params.iou_threshold = 1.F;
  EXPECT_TRUE(validate_params(params).ok());

  params = {};
  params.score_threshold = -0.1F;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.iou_threshold = 0.F;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.score_threshold = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.iou_threshold = std::numeric_limits<float>::infinity();
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);
}

TEST(NonMaximumSuppressionOp, OwnsStableNamedTensorPorts) {
  NonMaximumSuppressionOp op;
  EXPECT_EQ(op.boxes.name(), "boxes");
  EXPECT_EQ(op.scores.name(), "scores");
  EXPECT_EQ(op.mask.name(), "mask");
}

static_assert(std::derived_from<BayerDemosaicOp, Operator<BayerDemosaicOpParams>>);
static_assert(std::same_as<decltype(BayerDemosaicOp::input), Input<schema::ImageT>>);
static_assert(std::same_as<decltype(BayerDemosaicOp::output), Output<schema::ImageT>>);
static_assert(!std::copy_constructible<BayerDemosaicOp>);
static_assert(!std::move_constructible<BayerDemosaicOp>);

// REQ: VISION-011
TEST(BayerDemosaicOp, ParameterSchemaAcceptsDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(BayerDemosaicOpParams{}).ok());

  BayerDemosaicOpParams params;
  params.bayer_grid_pos = BayerGridPosition::kRGGB;
  params.generate_alpha = true;
  params.alpha_value = 0U;
  EXPECT_TRUE(validate_params(params).ok());

  params = {};
  params.bayer_grid_pos = static_cast<BayerGridPosition>(UINT8_MAX);
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);
}

TEST(BayerDemosaicOp, OwnsStableNamedImagePorts) {
  BayerDemosaicOp op;
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

[[nodiscard]] schema::ImageT bayer_mosaic_image(std::shared_ptr<Tensor> data, std::int32_t width,
                                                std::int32_t height) {
  schema::ImageT image;
  image.width = width;
  image.height = height;
  image.encoding = schema::ImageEncoding_BAYER_GBRG;
  image.data = std::move(data);
  return image;
}

// The output is deliberately unbound: accepted input reaches allocation and returns
// kNotReady, while invalid input must return kInvalidArgument before allocation or NPP.
[[nodiscard]] expected<void, Error> validate_bayer_input(schema::ImageEncoding encoding,
                                                         BayerDemosaicOpParams params,
                                                         std::uint8_t bits = 8U) {
  alignas(2) std::array<std::byte, 32U> storage{};
  auto tensor = std::make_shared<Tensor>(detail::make_tensor_with_dlpack_strides(
      storage.data(), {4, 4, 1}, {}, DLDataType{kDLUInt, bits, 1U}));
  auto image = bayer_mosaic_image(tensor, 4, 4);
  image.encoding = encoding;
  BayerDemosaicOp op;
  holoscan::test::SyntheticOperatorFixture fixture{op};
  if (auto result = fixture.declare_ports(); !result) {
    return result;
  }
  if (auto result = fixture.bind_initial_params(params); !result) {
    return result;
  }
  if (auto result = fixture.bind_value(op.input, std::move(image)); !result) {
    return result;
  }
  op.start();
  return fixture.invoke_once(holoscan::detail::SyntheticActivationOptions{
      .cuda_stream = reinterpret_cast<cudaStream_t>(std::uintptr_t{1U}),
  });
}

// REQ: VISION-010
TEST(BayerDemosaicOpValidation, NamedEncodingMustMatchConfiguredPhase) {
  constexpr std::array kPhases{
      std::pair{schema::ImageEncoding_BAYER_RGGB, BayerGridPosition::kRGGB},
      std::pair{schema::ImageEncoding_BAYER_BGGR, BayerGridPosition::kBGGR},
      std::pair{schema::ImageEncoding_BAYER_GRBG, BayerGridPosition::kGRBG},
      std::pair{schema::ImageEncoding_BAYER_GBRG, BayerGridPosition::kGBRG},
  };
  for (const auto& [encoding, expected_phase] : kPhases) {
    for (const auto& [unused_encoding, configured_phase] : kPhases) {
      for (const auto bits : {8U, 16U}) {
        SCOPED_TRACE(testing::Message() << "encoding=" << encoding << " phase="
                                        << static_cast<int>(configured_phase) << " bits=" << bits);
        const auto result = validate_bayer_input(
            encoding, BayerDemosaicOpParams{.bayer_grid_pos = configured_phase}, bits);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code,
                  expected_phase == configured_phase ? ErrorCode::kNotReady
                                                     : ErrorCode::kInvalidArgument);
        if (expected_phase != configured_phase) {
          EXPECT_NE(result.error().message.find("bayer_grid_pos"), std::string::npos);
        } else {
          // An empty message distinguishes "validation passed, reached the deliberately
          // unbound output" (Output<T>::allocate_tensor()'s plain Error{kNotReady}) from
          // BayerDemosaicOp's own "resources are not initialized" kNotReady, which would
          // otherwise mask a regression that silently dropped validate_bayer_input's start().
          EXPECT_TRUE(result.error().message.empty());
        }
      }
    }
  }
}

// REQ: VISION-010
TEST(BayerDemosaicOpValidation, UnnamedPhaseUsesConfiguredPhase) {
  for (const auto phase : {BayerGridPosition::kRGGB,
                           BayerGridPosition::kBGGR,
                           BayerGridPosition::kGRBG,
                           BayerGridPosition::kGBRG}) {
    for (const auto bits : {8U, 16U}) {
      for (const auto encoding :
           {schema::ImageEncoding_CUSTOM,
            bits == 8U ? schema::ImageEncoding_MONO8 : schema::ImageEncoding_MONO16}) {
        SCOPED_TRACE(testing::Message()
                     << "encoding=" << encoding << " phase=" << static_cast<int>(phase));
        const auto result =
            validate_bayer_input(encoding, BayerDemosaicOpParams{.bayer_grid_pos = phase}, bits);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code, ErrorCode::kNotReady);
        // See NamedEncodingMustMatchConfiguredPhase above: an empty message confirms this
        // reached the deliberately unbound output, not BayerDemosaicOp's own uninitialized-
        // resources guard.
        EXPECT_TRUE(result.error().message.empty());
      }
    }
  }
}

// REQ: VISION-010
TEST(BayerDemosaicOpValidation, AlphaMustFitTheOutputElementWhenEnabled) {
  struct Case {
    std::uint8_t bits;
    std::uint16_t alpha;
    bool generate_alpha;
    ErrorCode expected_code;
  };
  for (const auto& item : {Case{8U, 255U, true, ErrorCode::kNotReady},
                           Case{8U, 256U, true, ErrorCode::kInvalidArgument},
                           Case{8U, 65535U, true, ErrorCode::kInvalidArgument},
                           Case{16U, 65535U, true, ErrorCode::kNotReady},
                           Case{8U, 65535U, false, ErrorCode::kNotReady}}) {
    SCOPED_TRACE(testing::Message() << "bits=" << static_cast<int>(item.bits) << " alpha="
                                    << item.alpha << " generate_alpha=" << item.generate_alpha);
    const auto result = validate_bayer_input(
        schema::ImageEncoding_BAYER_GBRG,
        BayerDemosaicOpParams{.generate_alpha = item.generate_alpha, .alpha_value = item.alpha},
        item.bits);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, item.expected_code);
    if (item.expected_code == ErrorCode::kInvalidArgument) {
      EXPECT_NE(result.error().message.find("alpha_value"), std::string::npos);
    } else {
      // See NamedEncodingMustMatchConfiguredPhase above: an empty message confirms this
      // reached the deliberately unbound output, not BayerDemosaicOp's own uninitialized-
      // resources guard.
      EXPECT_TRUE(result.error().message.empty());
    }
  }
}

TEST(BayerDemosaicOpValidation, RejectsOddWidthOrHeight) {
  alignas(2) std::array<std::byte, 12U> storage{};
  auto tensor = std::make_shared<Tensor>(detail::make_tensor_with_dlpack_strides(
      storage.data(), {4, 3, 1}, {}, DLDataType{kDLUInt, 8U, 1U}));

  BayerDemosaicOp op;
  holoscan::test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_initial_params(BayerDemosaicOpParams{}));
  ASSERT_TRUE(fixture.bind_value(op.input, bayer_mosaic_image(tensor, 3, 4)));
  ASSERT_NO_THROW(op.start());

  auto result = fixture.invoke_once(holoscan::detail::SyntheticActivationOptions{
      .cuda_stream = reinterpret_cast<cudaStream_t>(std::uintptr_t{1U}),
  });
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(result.error().message.find("even"), std::string::npos);
}

TEST(BayerDemosaicOpValidation, RejectsUnsupportedDtype) {
  // Unsigned 8-bit and unsigned 16-bit are the two dtypes schema::ImageT can name an output
  // encoding for (RGB8/RGBA8, RGB16/RGBA16); everything else -- including a same-width signed
  // int -- must still be rejected. dtype_supported() in detail/cvcuda_tensor.cpp enforces this
  // ahead of BayerDemosaicOp's own compute(), inside describe_image_payload().
  for (const DLDataType dtype :
       {DLDataType{kDLInt, 16U, 1U}, DLDataType{kDLUInt, 32U, 1U}, DLDataType{kDLFloat, 32U, 1U}}) {
    SCOPED_TRACE(testing::Message() << "code=" << static_cast<int>(dtype.code)
                                    << " bits=" << static_cast<int>(dtype.bits));
    alignas(4) std::array<std::byte, 64U> storage{};
    auto tensor = std::make_shared<Tensor>(
        detail::make_tensor_with_dlpack_strides(storage.data(), {4, 4, 1}, {}, dtype));

    BayerDemosaicOp op;
    holoscan::test::SyntheticOperatorFixture fixture{op};
    ASSERT_TRUE(fixture.declare_ports());
    ASSERT_TRUE(fixture.bind_initial_params(BayerDemosaicOpParams{}));
    ASSERT_TRUE(fixture.bind_value(op.input, bayer_mosaic_image(tensor, 4, 4)));
    ASSERT_NO_THROW(op.start());

    auto result = fixture.invoke_once(holoscan::detail::SyntheticActivationOptions{
        .cuda_stream = reinterpret_cast<cudaStream_t>(std::uintptr_t{1U}),
    });
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("dtype"), std::string::npos);
  }
}

constexpr DLDataType kNmsBoxesDtype{kDLInt, 16U, 1U};
constexpr DLDataType kNmsScoresDtype{kDLFloat, 32U, 1U};
constexpr DLDataType kNmsMaskDtype{kDLUInt, 8U, 1U};

const PortRecord* declared_port(const GraphView& view, std::string_view name) {
  const auto found = std::ranges::find(view.ports(), name, &PortRecord::name);
  return found == view.ports().end() ? nullptr : &*found;
}

void expect_image_tensor_contracts(const GraphView& view, std::uint32_t max_rank = 4U) {
  // The graph canonicalizes the singular Image.data contract onto the port.
  const PortRecord* input = declared_port(view, "input");
  ASSERT_NE(input, nullptr);
  ASSERT_TRUE(input->setup.tensor_contract);
  EXPECT_EQ(input->setup.tensor_contract->max_rank, max_rank);
  EXPECT_FALSE(input->setup.tensor_contract->representation.strides);

  const PortRecord* output = declared_port(view, "output");
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(output->setup.tensor_contract);
  EXPECT_EQ(output->setup.tensor_contract->max_rank, max_rank);
  EXPECT_EQ(output->setup.tensor_contract->representation.strides,
            TensorStridePolicy::kContiguousOnly);
  EXPECT_FALSE(output->setup.tensor_contract->representation.dtype);
  EXPECT_FALSE(output->setup.tensor_contract->representation.rank);
}

// REQ: VISION-003
TEST(VisionTensorContracts, ImageOperatorsBoundInputRankAndPromiseContiguousOutput) {
  {
    Graph graph{"vision-bayer-contract"};
    (void)graph.op<BayerDemosaicOp>("bayer");
    expect_image_tensor_contracts(graph.view(), 3U);
  }
  {
    Graph graph{"vision-resize-contract"};
    (void)graph.op<ResizeOp>("resize");
    expect_image_tensor_contracts(graph.view());
  }
  {
    Graph graph{"vision-crop-contract"};
    (void)graph.op<CustomCropOp>("crop");
    expect_image_tensor_contracts(graph.view());
  }
  {
    Graph graph{"vision-border-contract"};
    (void)graph.op<CopyMakeBorderOp>("border");
    expect_image_tensor_contracts(graph.view());
  }
  {
    Graph graph{"vision-remap-contract"};
    (void)graph.op<RemapOp>("remap");
    const GraphView view = graph.view();
    expect_image_tensor_contracts(view);
    const PortRecord* map = declared_port(view, "map");
    ASSERT_NE(map, nullptr);
    ASSERT_TRUE(map->setup.tensor_contract);
    EXPECT_EQ(map->setup.tensor_contract->max_rank, 4U);
    EXPECT_FALSE(map->setup.tensor_contract->representation.strides);
  }
  {
    Graph graph{"vision-normalize-contract"};
    (void)graph.op<NormalizeOp>("normalize");
    expect_image_tensor_contracts(graph.view());
  }
  {
    Graph graph{"vision-convert-contract"};
    (void)graph.op<ConvertToOp>("convert");
    expect_image_tensor_contracts(graph.view());
  }
  {
    Graph graph{"vision-reformat-contract"};
    (void)graph.op<ReformatOp>("reformat");
    expect_image_tensor_contracts(graph.view());
  }
}

TEST(VisionTensorContracts, NmsDeclaresExactInputsAndContiguousMask) {
  Graph graph{"vision-nms-contract"};
  (void)graph.op<NonMaximumSuppressionOp>("nms");
  const GraphView view = graph.view();

  const PortRecord* boxes = declared_port(view, "boxes");
  ASSERT_NE(boxes, nullptr);
  ASSERT_TRUE(boxes->setup.tensor_contract);
  ASSERT_TRUE(boxes->setup.tensor_contract->representation.dtype);
  EXPECT_EQ(boxes->setup.tensor_contract->representation.dtype->code, kNmsBoxesDtype.code);
  EXPECT_EQ(boxes->setup.tensor_contract->representation.dtype->bits, kNmsBoxesDtype.bits);
  EXPECT_EQ(boxes->setup.tensor_contract->representation.dtype->lanes, kNmsBoxesDtype.lanes);
  EXPECT_EQ(boxes->setup.tensor_contract->representation.rank, 3U);

  const PortRecord* scores = declared_port(view, "scores");
  ASSERT_NE(scores, nullptr);
  ASSERT_TRUE(scores->setup.tensor_contract);
  ASSERT_TRUE(scores->setup.tensor_contract->representation.dtype);
  EXPECT_EQ(scores->setup.tensor_contract->representation.dtype->code, kNmsScoresDtype.code);
  EXPECT_EQ(scores->setup.tensor_contract->representation.dtype->bits, kNmsScoresDtype.bits);
  EXPECT_EQ(scores->setup.tensor_contract->representation.dtype->lanes, kNmsScoresDtype.lanes);
  EXPECT_EQ(scores->setup.tensor_contract->representation.rank, 2U);

  const PortRecord* mask = declared_port(view, "mask");
  ASSERT_NE(mask, nullptr);
  ASSERT_TRUE(mask->setup.tensor_contract);
  ASSERT_TRUE(mask->setup.tensor_contract->representation.dtype);
  EXPECT_EQ(mask->setup.tensor_contract->representation.dtype->code, kNmsMaskDtype.code);
  EXPECT_EQ(mask->setup.tensor_contract->representation.dtype->bits, kNmsMaskDtype.bits);
  EXPECT_EQ(mask->setup.tensor_contract->representation.dtype->lanes, kNmsMaskDtype.lanes);
  EXPECT_EQ(mask->setup.tensor_contract->representation.rank, 2U);
  EXPECT_EQ(mask->setup.tensor_contract->representation.strides,
            TensorStridePolicy::kContiguousOnly);
}

// REQ: PARAM-003
// REQ: VISION-011
TEST(VisionOperators, CapacityDependenciesHaveStructuralMutability) {
  // Key assertion: projector dependencies are compile-time or recompile-required, and true
  // capacity limits cannot be omitted by a future complete parameter loader.
  expect_fields<ResizeOpParams>({"width", "height"}, ParamMutability::kRecompile);
  expect_fields<ResizeOpParams>({"max_batch_size"}, ParamMutability::kRecompile, true);
  expect_fields<CustomCropOpParams>({"width", "height"}, ParamMutability::kRecompile);
  expect_fields<CustomCropOpParams>({"max_batch_size"}, ParamMutability::kRecompile, true);
  expect_fields<RemapOpParams>({"width", "height"}, ParamMutability::kRecompile);
  expect_fields<RemapOpParams>({"max_batch_size"}, ParamMutability::kRecompile, true);
  expect_fields<CopyMakeBorderOpParams>(
      {"max_width", "max_height", "max_batch_size"}, ParamMutability::kRecompile, true);
  expect_fields<ConvertToOpParams>({"output_type"}, ParamMutability::kCompileTime);
  expect_fields<NonMaximumSuppressionOpParams>(
      {"max_boxes", "max_batch_size"}, ParamMutability::kRecompile, true);

  // Unrelated behavior fields retain their prior compile-time mutability.
  expect_fields<CustomCropOpParams>({"x", "y"}, ParamMutability::kCompileTime);
  expect_fields<CopyMakeBorderOpParams>({"top", "bottom", "left", "right"},
                                        ParamMutability::kCompileTime);
}

// REQ: VISION-011
TEST(VisionOperators, ParameterSchemaVersionsMatchStructuralChanges) {
  // Key assertion: capacity and named-enum schema changes publish version 2.
  ASSERT_NE(param_schema_descriptor<BayerDemosaicOpParams>(), nullptr);
  EXPECT_EQ(param_schema_descriptor<BayerDemosaicOpParams>()->version, 2U);
  ASSERT_NE(param_schema_descriptor<ResizeOpParams>(), nullptr);
  EXPECT_EQ(param_schema_descriptor<ResizeOpParams>()->version, 2U);
  ASSERT_NE(param_schema_descriptor<CustomCropOpParams>(), nullptr);
  EXPECT_EQ(param_schema_descriptor<CustomCropOpParams>()->version, 2U);
  ASSERT_NE(param_schema_descriptor<RemapOpParams>(), nullptr);
  EXPECT_EQ(param_schema_descriptor<RemapOpParams>()->version, 2U);
  ASSERT_NE(param_schema_descriptor<CopyMakeBorderOpParams>(), nullptr);
  EXPECT_EQ(param_schema_descriptor<CopyMakeBorderOpParams>()->version, 2U);
  ASSERT_NE(param_schema_descriptor<NonMaximumSuppressionOpParams>(), nullptr);
  EXPECT_EQ(param_schema_descriptor<NonMaximumSuppressionOpParams>()->version, 2U);

  ASSERT_NE(param_schema_descriptor<NormalizeOpParams>(), nullptr);
  EXPECT_EQ(param_schema_descriptor<NormalizeOpParams>()->version, 1U);
  ASSERT_NE(param_schema_descriptor<ConvertToOpParams>(), nullptr);
  EXPECT_EQ(param_schema_descriptor<ConvertToOpParams>()->version, 2U);
  ASSERT_NE(param_schema_descriptor<ReformatOpParams>(), nullptr);
  EXPECT_EQ(param_schema_descriptor<ReformatOpParams>()->version, 2U);
}

}  // namespace
}  // namespace holoscan::vision
