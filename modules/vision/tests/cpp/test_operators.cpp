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

#include <holoscan/core/graph.hpp>
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

TEST(CvCudaTensor, DerivesContiguousStridesAndCheckedAllocationBound) {
  constexpr std::array<std::int64_t, 4U> kShape{2, 4, 5, 3};
  auto strides = contiguous_byte_strides(kShape, kF32);
  ASSERT_TRUE(strides);
  EXPECT_EQ((*strides)[0], 240);
  EXPECT_EQ((*strides)[1], 60);
  EXPECT_EQ((*strides)[2], 12);
  EXPECT_EQ((*strides)[3], 4);

  auto byte_span = maximum_output_byte_span(640, 480, 8U, sizeof(float));
  ASSERT_TRUE(byte_span);
  EXPECT_EQ(*byte_span, 640U * 480U * 8U * 4U * sizeof(float));

  EXPECT_FALSE(maximum_output_byte_span(std::numeric_limits<std::int32_t>::max(),
                                        std::numeric_limits<std::int32_t>::max(),
                                        std::numeric_limits<std::uint32_t>::max(),
                                        sizeof(double)));
  EXPECT_FALSE(maximum_output_byte_span(32768, 32768, 1U, sizeof(float)));
  EXPECT_TRUE(maximum_output_byte_span(1, 1, kMaximumBatchSize, sizeof(float)));
  EXPECT_FALSE(maximum_output_byte_span(1, 1, kMaximumBatchSize + 1U, sizeof(float)));
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
  NonMaximumSuppressionOp op{NMSCapacity{.max_boxes = 2, .max_batch_size = 1U}};
  holoscan::test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_initial_params(NonMaximumSuppressionOpParams{}));
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
void expect_structural_capacity_error(Op& op, typename Op::Params params,
                                      std::string_view field_name) {
  test::SyntheticOperatorFixture fixture{op};
  auto declared = fixture.declare_ports();
  ASSERT_TRUE(declared) << declared.error().message;
  auto bound = fixture.bind_initial_params(std::move(params));
  ASSERT_TRUE(bound) << bound.error().message;

  auto result = fixture.invoke_once();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(result.error().message.find(field_name), std::string::npos);
}

template <typename Params>
void expect_compile_time_capacity_fields(std::initializer_list<std::string_view> field_names) {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<Params>();
  ASSERT_NE(schema, nullptr);
  for (const std::string_view field_name : field_names) {
    const auto field =
        std::ranges::find(schema->fields, field_name, &ParamFieldDescriptor::canonical_name);
    ASSERT_NE(field, schema->fields.end()) << field_name;
    EXPECT_EQ(field->mutability, ParamMutability::kCompileTime) << field_name;
  }
}

static_assert(std::derived_from<ResizeOp, Operator<ResizeOpParams>>);
static_assert(std::same_as<decltype(ResizeOp::input), Input<Tensor>>);
static_assert(std::same_as<decltype(ResizeOp::output), Output<Tensor>>);
static_assert(!std::copy_constructible<ResizeOp>);
static_assert(!std::move_constructible<ResizeOp>);

TEST(ResizeOp, ParameterSchemaAcceptsValidDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(ResizeOpParams{}).ok());

  ResizeOpParams params;
  params.width = 0;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.height = -1;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kOutOfRange);

  params = {};
  params.interpolation = InterpolationMode::kCubic;
  EXPECT_TRUE(validate_params(params).ok());

  params = {};
  params.interpolation = static_cast<InterpolationMode>(UINT8_MAX);
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);
}

TEST(ResizeOp, OwnsStableNamedTensorPorts) {
  ResizeOp op{VisionImageCapacity{.max_width = 1, .max_height = 1, .max_batch_size = 1U}};
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<CustomCropOp, Operator<CustomCropOpParams>>);
static_assert(std::same_as<decltype(CustomCropOp::input), Input<Tensor>>);
static_assert(std::same_as<decltype(CustomCropOp::output), Output<Tensor>>);
static_assert(!std::copy_constructible<CustomCropOp>);
static_assert(!std::move_constructible<CustomCropOp>);

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
}

TEST(CustomCropOp, OwnsStableNamedTensorPorts) {
  CustomCropOp op{VisionImageCapacity{.max_width = 1, .max_height = 1, .max_batch_size = 1U}};
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<CopyMakeBorderOp, Operator<CopyMakeBorderOpParams>>);
static_assert(std::same_as<decltype(CopyMakeBorderOp::input), Input<Tensor>>);
static_assert(std::same_as<decltype(CopyMakeBorderOp::output), Output<Tensor>>);
static_assert(!std::copy_constructible<CopyMakeBorderOp>);
static_assert(!std::move_constructible<CopyMakeBorderOp>);

TEST(CopyMakeBorderOp, ParameterSchemaAcceptsDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(CopyMakeBorderOpParams{}).ok());

  CopyMakeBorderOpParams params;
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

TEST(CopyMakeBorderOp, OwnsStableNamedTensorPorts) {
  CopyMakeBorderOp op{VisionImageCapacity{.max_width = 1, .max_height = 1, .max_batch_size = 1U}};
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<RemapOp, Operator<RemapOpParams>>);
static_assert(std::same_as<decltype(RemapOp::input), Input<Tensor>>);
static_assert(std::same_as<decltype(RemapOp::map), Input<Tensor>>);
static_assert(std::same_as<decltype(RemapOp::output), Output<Tensor>>);
static_assert(!std::copy_constructible<RemapOp>);
static_assert(!std::move_constructible<RemapOp>);

TEST(RemapOp, ParameterSchemaAcceptsDefaultsAndRejectsInvalidValues) {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<RemapOpParams>();
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->version, 1U);
  EXPECT_TRUE(validate_params(RemapOpParams{}).ok());

  RemapOpParams params;
  params.width = 0;
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

TEST(RemapOp, OwnsStableNamedTensorPorts) {
  RemapOp op{VisionImageCapacity{.max_width = 1, .max_height = 1, .max_batch_size = 1U}};
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.map.name(), "map");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<NormalizeOp, Operator<NormalizeOpParams>>);
static_assert(std::same_as<decltype(NormalizeOp::input), Input<Tensor>>);
static_assert(std::same_as<decltype(NormalizeOp::output), Output<Tensor>>);
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
  auto normalize = graph.op<NormalizeOp>(
      "normalize", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
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

TEST(NormalizeOp, OwnsStableNamedTensorPorts) {
  NormalizeOp op{VisionImageCapacity{.max_width = 1, .max_height = 1, .max_batch_size = 1U}};
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<ConvertToOp, Operator<ConvertToOpParams>>);
static_assert(std::same_as<decltype(ConvertToOp::input), Input<Tensor>>);
static_assert(std::same_as<decltype(ConvertToOp::output), Output<Tensor>>);
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

TEST(ConvertToOp, OwnsStableNamedTensorPorts) {
  ConvertToOp op{VisionImageCapacity{.max_width = 1, .max_height = 1, .max_batch_size = 1U}};
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<ReformatOp, Operator<ReformatOpParams>>);
static_assert(std::same_as<decltype(ReformatOp::input), Input<Tensor>>);
static_assert(std::same_as<decltype(ReformatOp::output), Output<Tensor>>);
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

TEST(ReformatOp, OwnsStableNamedTensorPorts) {
  ReformatOp op{VisionImageCapacity{.max_width = 1, .max_height = 1, .max_batch_size = 1U}};
  EXPECT_EQ(op.input.name(), "input");
  EXPECT_EQ(op.output.name(), "output");
}

static_assert(std::derived_from<NonMaximumSuppressionOp, Operator<NonMaximumSuppressionOpParams>>);
static_assert(std::same_as<decltype(NonMaximumSuppressionOp::boxes), Input<Tensor>>);
static_assert(std::same_as<decltype(NonMaximumSuppressionOp::scores), Input<Tensor>>);
static_assert(std::same_as<decltype(NonMaximumSuppressionOp::mask), Output<Tensor>>);
static_assert(!std::copy_constructible<NonMaximumSuppressionOp>);
static_assert(!std::move_constructible<NonMaximumSuppressionOp>);

TEST(NonMaximumSuppressionOp, ParameterSchemaAcceptsDefaultsAndRejectsInvalidValues) {
  EXPECT_TRUE(validate_params(NonMaximumSuppressionOpParams{}).ok());

  NonMaximumSuppressionOpParams params;
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
  NonMaximumSuppressionOp op{NMSCapacity{.max_boxes = 1, .max_batch_size = 1U}};
  EXPECT_EQ(op.boxes.name(), "boxes");
  EXPECT_EQ(op.scores.name(), "scores");
  EXPECT_EQ(op.mask.name(), "mask");
}

constexpr DLDataType kNmsBoxesDtype{kDLInt, 16U, 1U};
constexpr DLDataType kNmsScoresDtype{kDLFloat, 32U, 1U};
constexpr DLDataType kNmsMaskDtype{kDLUInt, 8U, 1U};

const PortRecord* declared_port(const GraphView& view, std::string_view name) {
  const auto found = std::ranges::find(view.ports(), name, &PortRecord::name);
  return found == view.ports().end() ? nullptr : &*found;
}

void expect_image_tensor_contracts(const GraphView& view) {
  const PortRecord* input = declared_port(view, "input");
  ASSERT_NE(input, nullptr);
  ASSERT_TRUE(input->setup.tensor_contract);
  EXPECT_EQ(input->setup.tensor_contract->max_rank, 4U);
  EXPECT_FALSE(input->setup.tensor_contract->representation.strides);

  const PortRecord* output = declared_port(view, "output");
  ASSERT_NE(output, nullptr);
  ASSERT_TRUE(output->setup.tensor_contract);
  EXPECT_EQ(output->setup.tensor_contract->max_rank, 4U);
  EXPECT_EQ(output->setup.tensor_contract->representation.strides,
            TensorStridePolicy::kContiguousOnly);
  EXPECT_FALSE(output->setup.tensor_contract->representation.dtype);
  EXPECT_FALSE(output->setup.tensor_contract->representation.rank);
}

TEST(VisionTensorContracts, ImageOperatorsBoundInputRankAndPromiseContiguousOutput) {
  {
    Graph graph{"vision-resize-contract"};
    (void)graph.op<ResizeOp>(
        "resize", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
    expect_image_tensor_contracts(graph.view());
  }
  {
    Graph graph{"vision-crop-contract"};
    (void)graph.op<CustomCropOp>(
        "crop", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
    expect_image_tensor_contracts(graph.view());
  }
  {
    Graph graph{"vision-border-contract"};
    (void)graph.op<CopyMakeBorderOp>(
        "border", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
    expect_image_tensor_contracts(graph.view());
  }
  {
    Graph graph{"vision-remap-contract"};
    (void)graph.op<RemapOp>(
        "remap", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
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
    (void)graph.op<NormalizeOp>(
        "normalize", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
    expect_image_tensor_contracts(graph.view());
  }
  {
    Graph graph{"vision-convert-contract"};
    (void)graph.op<ConvertToOp>(
        "convert", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
    expect_image_tensor_contracts(graph.view());
  }
  {
    Graph graph{"vision-reformat-contract"};
    (void)graph.op<ReformatOp>(
        "reformat", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
    expect_image_tensor_contracts(graph.view());
  }
}

TEST(VisionTensorContracts, NmsDeclaresExactInputsAndContiguousMask) {
  Graph graph{"vision-nms-contract"};
  (void)graph.op<NonMaximumSuppressionOp>("nms", NMSCapacity{.max_boxes = 2, .max_batch_size = 1U});
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

TEST(VisionOperators, ShapeFieldsRemainCompileTime) {
  // Compile-time fields are classified as requiring recompilation before an update candidate is
  // created, so active mutation remains unavailable even when runtime updates are opted in.
  expect_compile_time_capacity_fields<ResizeOpParams>({"width", "height"});
  expect_compile_time_capacity_fields<CustomCropOpParams>({"x", "y", "width", "height"});
  expect_compile_time_capacity_fields<RemapOpParams>({"width", "height"});
  expect_compile_time_capacity_fields<CopyMakeBorderOpParams>({"top", "bottom", "left", "right"});
}

TEST(VisionOperators, ComputeRejectsSpatialParametersBeyondAuthoredCapacity) {
  // A 3x1 U8C1 output occupies three bytes, below even Remap's 2x2x1 authored 16-byte span.
  // The individual width axis must still remain bounded by the authored structural capacity.
  static_assert(3U < 4U * 2U * 2U * 1U);

  {
    ResizeOp op{VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U}};
    ResizeOpParams params{.width = 3, .height = 1};
    expect_structural_capacity_error(op, std::move(params), "width");
  }
  {
    ResizeOp op{VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U}};
    ResizeOpParams params{.width = 1, .height = 3};
    expect_structural_capacity_error(op, std::move(params), "height");
  }
  {
    CustomCropOp op{VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U}};
    CustomCropOpParams params{.width = 3, .height = 1};
    expect_structural_capacity_error(op, std::move(params), "width");
  }
  {
    CustomCropOp op{VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U}};
    CustomCropOpParams params{.width = 1, .height = 3};
    expect_structural_capacity_error(op, std::move(params), "height");
  }
  {
    RemapOp op{VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U}};
    RemapOpParams params{.width = 3, .height = 1};
    expect_structural_capacity_error(op, std::move(params), "width");
  }
  {
    RemapOp op{VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U}};
    RemapOpParams params{.width = 1, .height = 3};
    expect_structural_capacity_error(op, std::move(params), "height");
  }
}

TEST(VisionOperators, ShapeParameterOverridesCannotExceedAuthoredPoolCapacity) {
  Graph graph{"vision-capacity-test"};
  constexpr VisionImageCapacity kCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U};
  auto resize = graph.op<ResizeOp>("resize", kCapacity);
  EXPECT_THROW(resize.set_params(ResizeOpParams{.width = 3, .height = 2}), RuntimeError);

  auto crop = graph.op<CustomCropOp>("crop", kCapacity);
  EXPECT_THROW(crop.set_params(CustomCropOpParams{.width = 2, .height = 3}), RuntimeError);

  auto border = graph.op<CopyMakeBorderOp>("border", kCapacity);
  EXPECT_THROW(border.set_params(CopyMakeBorderOpParams{.top = 1, .bottom = 1}), RuntimeError);

  auto remap = graph.op<RemapOp>("remap", kCapacity);
  EXPECT_THROW(remap.set_params(RemapOpParams{.width = 3, .height = 2}), RuntimeError);
}

TEST(VisionOperators, RejectAuthoredBatchCapacityAboveSharedLimit) {
  Graph resize_graph{"vision-image-batch-capacity-test"};
  EXPECT_THROW((void)resize_graph.op<ResizeOp>(
                   "resize",
                   VisionImageCapacity{
                       .max_width = 1, .max_height = 1, .max_batch_size = kMaximumBatchSize + 1U}),
               RuntimeError);

  Graph nms_graph{"vision-nms-batch-capacity-test"};
  EXPECT_THROW((void)nms_graph.op<NonMaximumSuppressionOp>(
                   "nms", NMSCapacity{.max_boxes = 1, .max_batch_size = kMaximumBatchSize + 1U}),
               RuntimeError);
}

}  // namespace
}  // namespace holoscan::vision
