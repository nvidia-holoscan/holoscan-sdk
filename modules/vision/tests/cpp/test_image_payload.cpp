// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <nvcv/TensorData.hpp>

#include "common/allocation_counter.hpp"
#include "detail/image_payload.hpp"

namespace holoscan::vision::detail {
namespace {

ImageTensorView rgb_view(ImageTensorLayout layout = ImageTensorLayout::kInterleaved) {
  return ImageTensorView{.shape = layout == ImageTensorLayout::kInterleaved
                                      ? std::array<std::int64_t, 4>{2, 3, 3, 0}
                                      : std::array<std::int64_t, 4>{3, 2, 3, 0},
                         .byte_strides = layout == ImageTensorLayout::kInterleaved
                                             ? std::array<std::int64_t, 4>{9, 3, 1, 0}
                                             : std::array<std::int64_t, 4>{6, 3, 1, 0},
                         .dtype = {kDLUInt, 8, 1},
                         .device = {kDLCUDA, 0},
                         .rank = 3,
                         .batch = 1,
                         .height = 2,
                         .width = 3,
                         .channels = 3,
                         .image_layout = layout};
}

schema::ImageT rgb_image() {
  schema::ImageT image;
  image.width = 3;
  image.height = 2;
  image.encoding = schema::ImageEncoding_RGB8;
  image.color_space = schema::ColorSpace_SRGB;
  image.roi_offset_x = 10;
  image.roi_offset_y = 20;
  image.significant_bits = 7;
  image.exposure_time_ns = 123;
  image.gain = 2.F;
  image.header = std::make_shared<schema::HeaderT>();
  image.header->frame_id = "camera";
  image.header->capture_timestamp_ns = 321;
  return image;
}

ImageTensorView scalar_image_view(std::int64_t channels, DLDataType dtype,
                                  ImageTensorLayout layout = ImageTensorLayout::kInterleaved) {
  auto view = rgb_view(layout);
  view.channels = channels;
  view.dtype = dtype;
  const std::int64_t element_bytes = dtype.bits / 8;
  if (layout == ImageTensorLayout::kPlanar) {
    view.shape = {channels, 2, 3, 0};
    view.byte_strides = {6 * element_bytes, 3 * element_bytes, element_bytes, 0};
  } else {
    view.shape = {2, 3, channels, 0};
    view.byte_strides = {3 * channels * element_bytes, channels * element_bytes, element_bytes, 0};
  }
  return view;
}

// REQ: VISION-008
TEST(VisionImageMetadata, PreparedScratchEditsDoNotAllocateOnFirstOrRepeatedUse) {
  // Exercise the production metadata helpers, not CUDA submission or the whole compute callback.
  for (const auto channels : {3, 4}) {
    for (const auto planar : {false, true}) {
      SCOPED_TRACE(channels);
      SCOPED_TRACE(planar);
      auto source = rgb_image();
      source.encoding = channels == 3 ? schema::ImageEncoding_RGB8 : schema::ImageEncoding_RGBA8;
      source.header->frame_id.assign(512, 'c');
      source.header->stamp = std::make_shared<schema::TimeT>();
      source.header->stamp->time_ns = 456;
      const auto input = scalar_image_view(channels, {kDLUInt, 8, 1});
      auto output = scalar_image_view(channels, {kDLUInt, 8, 1}, ImageTensorLayout::kPlanar);
      if (!planar) {
        output = input;
        output.width = 2;
        output.height = 1;
        output.shape = {1, 2, channels, 0};
        output.byte_strides = {2 * channels, channels, 1, 0};
      }
      const ImageMetadataTransform transform{.geometry = planar ? ImageGeometryChange::kPreserve
                                                                : ImageGeometryChange::kResize};
      auto scratch = sensor_io::ImageDescriptorScratch::create();
      ASSERT_TRUE(scratch);
      for (std::size_t iteration = 0; iteration != 64; ++iteration) {
        SCOPED_TRACE(iteration);
        expected<void, Error> edited;
        std::size_t allocations{};
        {
          // No warmup: iteration zero is the first edit after cold preparation.
          const test::ScopedAllocationCounter counter;
          edited = scratch->copy_metadata_from(source);
          if (edited)
            edited = update_image_metadata(scratch->image(), input, output, transform);
          allocations = counter.count();
        }
        ASSERT_TRUE(edited);
        EXPECT_EQ(allocations, 0U);
        const auto& image = scratch->image();
        EXPECT_FALSE(image.data);
        EXPECT_EQ(image.width, output.width);
        EXPECT_EQ(image.height, output.height);
        EXPECT_EQ(image.roi_offset_x, planar ? 10 : 0);
        EXPECT_EQ(image.plane_layouts.size(), planar ? static_cast<std::size_t>(channels) : 0U);
        ASSERT_TRUE(image.header);
        ASSERT_TRUE(image.header->stamp);
        EXPECT_NE(image.header, source.header);
        EXPECT_NE(image.header->stamp, source.header->stamp);
        EXPECT_EQ(image.header->frame_id, source.header->frame_id);
        EXPECT_EQ(image.header->stamp->time_ns, 456);
        EXPECT_EQ(source.width, 3);
        EXPECT_EQ(source.height, 2);
        EXPECT_EQ(source.roi_offset_x, 10);
        EXPECT_TRUE(source.plane_layouts.empty());
      }
    }
  }
}

// REQ: VISION-004
TEST(VisionImageMetadata, ImageAdapterBorrowsTheOriginalPitchedBatchedTensor) {
  alignas(16) std::array<std::byte, 64> storage{};
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = {2, 2, 3, 3};
  context->dl_strides = {32, 16, 3, 1};
  context->tensor.dl_tensor = DLTensor{.data = storage.data(),
                                       .device = {kDLCUDA, 0},
                                       .ndim = 4,
                                       .dtype = {kDLUInt, 8, 1},
                                       .shape = context->dl_shape.data(),
                                       .strides = context->dl_strides.data(),
                                       .byte_offset = 0};
  auto image = rgb_image();
  const auto missing_data = describe_image_payload(image, TensorOperation::kResize, 2);
  ASSERT_FALSE(missing_data);
  EXPECT_NE(missing_data.error().message.find("Image data"), std::string::npos);
  image.data = std::make_shared<Tensor>(context);
  const auto tensor_view = describe_image_tensor(*image.data, TensorOperation::kResize, 2);
  const auto image_view = describe_image_payload(image, TensorOperation::kResize, 2);
  ASSERT_TRUE(tensor_view);
  ASSERT_TRUE(image_view);
  EXPECT_EQ(image_view->data, storage.data());
  EXPECT_EQ(image_view->data, tensor_view->data);
  EXPECT_EQ(image_view->shape, tensor_view->shape);
  EXPECT_EQ(image_view->byte_strides, tensor_view->byte_strides);
  auto wrapped = wrap_nvcv_tensor(*image_view);
  ASSERT_TRUE(wrapped);
  const auto exported = wrapped->exportData<nvcv::TensorDataStridedCuda>();
  ASSERT_TRUE(exported);
  EXPECT_EQ(exported->basePtr(), reinterpret_cast<nvcv::Byte*>(storage.data()));
  EXPECT_EQ(exported->stride(0), 32);
  EXPECT_EQ(exported->stride(1), 16);
}

// REQ: VISION-004
TEST(VisionImageMetadata, RGB16AndRGBA16RequireInterleavedUnsigned16BitComponents) {
  for (const auto encoding : {schema::ImageEncoding_RGB16, schema::ImageEncoding_RGBA16}) {
    SCOPED_TRACE(schema::EnumNameImageEncoding(encoding));
    auto image = rgb_image();
    image.encoding = encoding;
    image.significant_bits = 12;
    const std::int64_t channels = encoding == schema::ImageEncoding_RGB16 ? 3 : 4;
    const auto view = scalar_image_view(channels, {kDLUInt, 16, 1});
    ASSERT_TRUE(validate_image_payload(image, view));
    ASSERT_TRUE(update_image_metadata(image, view, view));
    EXPECT_EQ(image.encoding, encoding);
    EXPECT_EQ(image.significant_bits, 12);
    EXPECT_EQ(image.color_space, schema::ColorSpace_SRGB);
    EXPECT_FALSE(validate_image_payload(image, scalar_image_view(1, {kDLUInt, 16, 1})));
    EXPECT_FALSE(validate_image_payload(
        image, scalar_image_view(channels, {kDLUInt, 16, 1}, ImageTensorLayout::kPlanar)));
    for (const DLDataType dtype : {DLDataType{kDLUInt, 8, 1},
                                   DLDataType{kDLUInt, 32, 1},
                                   DLDataType{kDLInt, 16, 1},
                                   DLDataType{kDLFloat, 16, 1},
                                   DLDataType{kDLFloat, 32, 1},
                                   DLDataType{kDLUInt, 16, 2}}) {
      const auto invalid = validate_image_payload(image, scalar_image_view(channels, dtype));
      ASSERT_FALSE(invalid);
      EXPECT_EQ(invalid.error().code, ErrorCode::kInvalidArgument);
      EXPECT_NE(invalid.error().message.find("dtype"), std::string::npos);
    }
  }
}

// REQ: VISION-006
TEST(VisionImageMetadata, NoOpGeometryPreservesPrecisionWithUnusedInterpolationOrBorder) {
  for (const auto operation : {ImageGeometryChange::kResize, ImageGeometryChange::kPad}) {
    auto image = rgb_image();
    ASSERT_TRUE(update_image_metadata(
        image, rgb_view(), rgb_view(), {.geometry = operation, .preserves_precision = false}));
    EXPECT_EQ(image.significant_bits, 7);
  }
}

// REQ: VISION-006
TEST(VisionImageMetadata, ValuePreservingTransformsRetainSourcePrecision) {
  auto output = rgb_view();
  output.shape = {2, 6, 3, 0};
  output.byte_strides = {18, 3, 1, 0};
  output.width = 6;
  for (const auto geometry : {ImageGeometryChange::kResize, ImageGeometryChange::kPad}) {
    auto image = rgb_image();
    ASSERT_TRUE(update_image_metadata(image, rgb_view(), output, {.geometry = geometry}));
    EXPECT_EQ(image.significant_bits, 7);
    ASSERT_TRUE(update_image_metadata(
        image,
        output,
        output,
        {.geometry = ImageGeometryChange::kPreserve, .values_changed = true}));
    EXPECT_EQ(image.significant_bits, 255);
  }
}

// REQ: VISION-006
TEST(VisionImageMetadata, LosslessCastsRetainPrecisionAndLossyCastsReportUnknown) {
  struct CastCase {
    DLDataType input_type;
    std::uint8_t significant_bits;
    DLDataType output_type;
    std::uint8_t expected_bits;
  };
  const CastCase cases[]{
      {{kDLUInt, 8, 1}, 0, {kDLUInt, 16, 1}, 8},
      {{kDLUInt, 8, 1}, 0, {kDLFloat, 32, 1}, 8},
      {{kDLUInt, 16, 1}, 10, {kDLUInt, 8, 1}, 255},
      {{kDLUInt, 16, 1}, 12, {kDLFloat, 32, 1}, 12},
      {{kDLUInt, 16, 1}, 255, {kDLFloat, 32, 1}, 255},
      {{kDLFloat, 32, 1}, 0, {kDLUInt, 16, 1}, 255},
      {{kDLUInt, 8, 1}, 7, {kDLInt, 8, 1}, 255},
      {{kDLUInt, 8, 1}, 0, {kDLInt, 16, 1}, 8},
      {{kDLInt, 16, 1}, 0, {kDLInt, 32, 1}, 16},
      {{kDLInt, 16, 1}, 12, {kDLUInt, 32, 1}, 255},
      {{kDLInt, 32, 1}, 0, {kDLFloat, 32, 1}, 255},
      {{kDLInt, 32, 1}, 0, {kDLFloat, 64, 1}, 32},
      {{kDLFloat, 32, 1}, 0, {kDLFloat, 64, 1}, 32},
      {{kDLFloat, 64, 1}, 0, {kDLFloat, 32, 1}, 255},
  };
  for (const auto& test : cases) {
    auto image = rgb_image();
    image.encoding = schema::ImageEncoding_CUSTOM;
    image.significant_bits = test.significant_bits;
    image.color_space = schema::ColorSpace_REC2020_PQ;
    const auto input = scalar_image_view(3, test.input_type);
    const auto output = scalar_image_view(3, test.output_type);
    ASSERT_TRUE(
        update_image_metadata(image, input, output, {.geometry = ImageGeometryChange::kPreserve}));
    EXPECT_EQ(image.significant_bits, test.expected_bits);
    EXPECT_EQ(image.color_space, schema::ColorSpace_UNSPECIFIED);
  }
}

// REQ: VISION-006
TEST(VisionImageMetadata, TenBitMonoCastToFloatPreservesPrecisionWithoutScaling) {
  auto image = rgb_image();
  image.encoding = schema::ImageEncoding_MONO16;
  image.significant_bits = 10;
  const auto input = scalar_image_view(1, {kDLUInt, 16, 1});
  const auto output = scalar_image_view(1, {kDLFloat, 32, 1});
  ASSERT_TRUE(update_image_metadata(image, input, output));
  EXPECT_EQ(image.significant_bits, 10);
  EXPECT_EQ(image.encoding, schema::ImageEncoding_MONO32F);
}

// REQ: VISION-006
TEST(VisionImageMetadata, ValueChangingTransformsReportUnknownPrecision) {
  for (const auto geometry :
       {ImageGeometryChange::kResize, ImageGeometryChange::kWarp, ImageGeometryChange::kPad}) {
    auto image = rgb_image();
    auto output = rgb_view();
    output.width = 6;
    output.shape = {2, 6, 3, 0};
    output.byte_strides = {18, 3, 1, 0};
    ASSERT_TRUE(update_image_metadata(
        image, rgb_view(), output, {.geometry = geometry, .preserves_precision = false}));
    EXPECT_EQ(image.significant_bits, 255);
    EXPECT_EQ(image.color_space, schema::ColorSpace_SRGB);
    ASSERT_TRUE(update_image_metadata(image, output, output));
    EXPECT_EQ(image.significant_bits, 255);
  }
  auto image = rgb_image();
  ASSERT_TRUE(update_image_metadata(
      image, rgb_view(), scalar_image_view(3, {kDLFloat, 32, 1}), {.values_changed = true}));
  EXPECT_EQ(image.significant_bits, 255);
  EXPECT_EQ(image.color_space, schema::ColorSpace_UNSPECIFIED);
}

// REQ: VISION-006
TEST(VisionImageMetadata, IdentityConversionPreservesDeclaredHdrRepresentation) {
  auto image = rgb_image();
  image.encoding = schema::ImageEncoding_MONO16;
  image.significant_bits = 10;
  image.color_space = schema::ColorSpace_REC2020_PQ;
  const auto view = scalar_image_view(1, {kDLUInt, 16, 1});
  ASSERT_TRUE(
      update_image_metadata(image, view, view, {.geometry = ImageGeometryChange::kPreserve}));
  EXPECT_EQ(image.significant_bits, 10);
  EXPECT_EQ(image.color_space, schema::ColorSpace_REC2020_PQ);
  EXPECT_EQ(image.encoding, schema::ImageEncoding_MONO16);
}

// REQ: VISION-004
TEST(VisionImageMetadata, CustomInterleavedPlaneMustMatchTheTensor) {
  auto image = rgb_image();
  image.encoding = schema::ImageEncoding_CUSTOM;
  image.plane_layouts.emplace_back(0, 9, 18);
  EXPECT_TRUE(validate_image_payload(image, rgb_view()));
  image.plane_layouts[0] = schema::ImagePlaneLayout{1, 9, 18};
  EXPECT_FALSE(validate_image_payload(image, rgb_view()));
  image.plane_layouts[0] = schema::ImagePlaneLayout{0, 10, 18};
  EXPECT_FALSE(validate_image_payload(image, rgb_view()));
}

// REQ: VISION-004
TEST(VisionImageMetadata, RejectsContradictoryGeometryChannelsAndPackedEncodings) {
  auto image = rgb_image();
  const auto view = rgb_view();
  EXPECT_TRUE(validate_image_payload(image, view));
  image.width = 4;
  EXPECT_FALSE(validate_image_payload(image, view));
  image.width = 3;
  image.encoding = schema::ImageEncoding_RGBA8;
  EXPECT_FALSE(validate_image_payload(image, view));
  for (const auto encoding : {schema::ImageEncoding_NV12,
                              schema::ImageEncoding_YUYV,
                              schema::ImageEncoding_RGBA1010102}) {
    image.encoding = encoding;
    EXPECT_FALSE(validate_image_payload(image, view));
  }
}

// REQ: VISION-004
TEST(VisionImageMetadata, DescriptorErrorsIdentifyTheInconsistentFields) {
  auto image = rgb_image();
  const auto view = rgb_view();
  const auto rejects = [&image](const ImageTensorView& tensor,
                                const char* field,
                                ErrorCode code = ErrorCode::kInvalidArgument) {
    const auto result = validate_image_payload(image, tensor);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, code);
    EXPECT_NE(result.error().message.find(field), std::string::npos) << result.error().message;
    EXPECT_NE(result.error().message.find("Tensor"), std::string::npos) << result.error().message;
  };
  image.width = 4;
  rejects(view, "width");
  image = rgb_image();
  image.height = 4;
  rejects(view, "height");
  image = rgb_image();
  image.encoding = schema::ImageEncoding_RGBA8;
  rejects(view, "channels");
  image = rgb_image();
  rejects(scalar_image_view(3, {kDLFloat, 32, 1}), "dtype");
  rejects(rgb_view(ImageTensorLayout::kPlanar), "layout");
  image.encoding = schema::ImageEncoding_NV12;
  rejects(view, "encoding", ErrorCode::kNotSupported);
  image = rgb_image();
  image.plane_layouts = {{0, 9, 18}, {18, 9, 18}};
  rejects(view, "plane_layouts count");
  image.plane_layouts = {{1, 9, 18}};
  rejects(view, "plane_layouts[0].offset_bytes");
  image.plane_layouts = {{0, 10, 18}};
  rejects(view, "plane_layouts[0].row_stride_bytes");
  image.plane_layouts = {{0, 9, 17}};
  rejects(view, "plane_layouts[0].size_bytes");
  image.plane_layouts = {{0, 9, 19}};
  rejects(view, "plane_layouts[0].size_bytes");
}

// REQ: VISION-005
TEST(VisionImageMetadata, CropUpdatesSensorOriginAndPreservesAcquisitionMetadata) {
  auto image = rgb_image();
  auto output = rgb_view();
  output.shape = {1, 2, 3, 0};
  output.byte_strides = {6, 3, 1, 0};
  output.width = 2;
  output.height = 1;
  ASSERT_TRUE(update_image_metadata(
      image, rgb_view(), output, {.geometry = ImageGeometryChange::kCrop, .x = 1, .y = 1}));
  EXPECT_EQ(image.width, 2);
  EXPECT_EQ(image.height, 1);
  EXPECT_EQ(image.roi_offset_x, 11);
  EXPECT_EQ(image.roi_offset_y, 21);
  EXPECT_EQ(image.encoding, schema::ImageEncoding_RGB8);
  EXPECT_EQ(image.color_space, schema::ColorSpace_SRGB);
  EXPECT_EQ(image.significant_bits, 7);
  EXPECT_EQ(image.exposure_time_ns, 123);
  EXPECT_EQ(image.gain, 2.F);
  EXPECT_EQ(image.header->frame_id, "camera");
  EXPECT_EQ(image.header->capture_timestamp_ns, 321);
  image = rgb_image();
  image.roi_offset_x = std::numeric_limits<std::int32_t>::max();
  EXPECT_FALSE(update_image_metadata(
      image, rgb_view(), output, {.geometry = ImageGeometryChange::kCrop, .x = 1}));
}

// REQ: VISION-005
TEST(VisionImageMetadata, ReformatRoundTripRebuildsPlanesAndEncoding) {
  auto image = rgb_image();
  const auto interleaved = rgb_view();
  const auto planar = rgb_view(ImageTensorLayout::kPlanar);
  ASSERT_TRUE(update_image_metadata(
      image, interleaved, planar, {.geometry = ImageGeometryChange::kPreserve}));
  EXPECT_EQ(image.encoding, schema::ImageEncoding_R8_G8_B8);
  ASSERT_EQ(image.plane_layouts.size(), 3);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(image.plane_layouts[i].offset_bytes(), i * 6);
    EXPECT_EQ(image.plane_layouts[i].row_stride_bytes(), 3);
    EXPECT_EQ(image.plane_layouts[i].size_bytes(), 6);
  }
  EXPECT_TRUE(validate_image_payload(image, planar));
  image.plane_layouts[1] = schema::ImagePlaneLayout{7, 3, 6};
  EXPECT_FALSE(validate_image_payload(image, planar));
  image.plane_layouts[1] = schema::ImagePlaneLayout{6, 3, 6};
  ASSERT_TRUE(update_image_metadata(
      image, planar, interleaved, {.geometry = ImageGeometryChange::kPreserve}));
  EXPECT_EQ(image.encoding, schema::ImageEncoding_RGB8);
  EXPECT_TRUE(image.plane_layouts.empty());
  EXPECT_EQ(image.roi_offset_x, 10);
}

// REQ: VISION-006
TEST(VisionImageMetadata, ConversionUpdatesEncodingAndPrecisionWithoutInventingFormats) {
  auto image = rgb_image();
  const auto float_output = scalar_image_view(3, {kDLFloat, 32, 1});
  ASSERT_TRUE(update_image_metadata(
      image, rgb_view(), float_output, {.geometry = ImageGeometryChange::kPreserve}));
  EXPECT_EQ(image.encoding, schema::ImageEncoding_RGB32F);
  EXPECT_EQ(image.significant_bits, 7);
  EXPECT_EQ(image.color_space, schema::ColorSpace_UNSPECIFIED);
  const auto signed_output = scalar_image_view(3, {kDLInt, 16, 1});
  ASSERT_TRUE(update_image_metadata(
      image, float_output, signed_output, {.geometry = ImageGeometryChange::kPreserve}));
  EXPECT_EQ(image.encoding, schema::ImageEncoding_CUSTOM);
  EXPECT_EQ(image.significant_bits, 255);
}

// REQ: VISION-005
TEST(VisionImageMetadata, UpdateRejectsEncodingThatContradictsOutputChannels) {
  auto image = rgb_image();
  const auto input = rgb_view();
  const auto output = scalar_image_view(4, {kDLUInt, 8, 1});
  ASSERT_TRUE(validate_image_payload(image, input));

  // A channel-changing operation must choose a matching encoding before publishing its output.
  const auto result = update_image_metadata(image, input, output);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// REQ: VISION-005
TEST(VisionImageMetadata, SpatialResamplingRebasesOriginAndBayerCropChangesPhase) {
  auto image = rgb_image();
  auto output = rgb_view();
  output.shape = {2, 6, 3, 0};
  output.byte_strides = {18, 3, 1, 0};
  output.width = 6;
  ASSERT_TRUE(
      update_image_metadata(image, rgb_view(), output, {.geometry = ImageGeometryChange::kResize}));
  EXPECT_EQ(image.roi_offset_x, 0);
  EXPECT_EQ(image.roi_offset_y, 0);
  image = rgb_image();
  image.encoding = schema::ImageEncoding_BAYER_RGGB;
  const auto mono = scalar_image_view(1, {kDLUInt, 8, 1});
  auto cropped = mono;
  cropped.shape = {2, 2, 1, 0};
  cropped.byte_strides = {2, 1, 1, 0};
  cropped.width = 2;
  ASSERT_TRUE(update_image_metadata(
      image, mono, cropped, {.geometry = ImageGeometryChange::kCrop, .x = 1}));
  EXPECT_EQ(image.encoding, schema::ImageEncoding_BAYER_GRBG);
  ASSERT_TRUE(
      update_image_metadata(image, cropped, cropped, {.geometry = ImageGeometryChange::kWarp}));
  EXPECT_EQ(image.encoding, schema::ImageEncoding_CUSTOM);
  EXPECT_EQ(image.roi_offset_x, 0);
  EXPECT_EQ(image.roi_offset_y, 0);
}

// REQ: VISION-004
TEST(VisionImageMetadata, PitchedPlanarOutputFitsTheTensorAddressableSpan) {
  auto image = rgb_image();
  auto output = rgb_view(ImageTensorLayout::kPlanar);
  output.byte_strides = {8, 4, 1, 0};
  ASSERT_TRUE(update_image_metadata(image, rgb_view(), output));
  ASSERT_EQ(image.plane_layouts.size(), 3U);
  EXPECT_EQ(image.plane_layouts.back().offset_bytes(), 16U);
  EXPECT_EQ(image.plane_layouts.back().size_bytes(), 7U);

  alignas(16) std::array<std::byte, 23> storage{};
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = {3, 2, 3};
  context->dl_strides = {8, 4, 1};
  context->tensor.dl_tensor = DLTensor{.data = storage.data(),
                                       .device = {kDLCUDA, 0},
                                       .ndim = 3,
                                       .dtype = {kDLUInt, 8, 1},
                                       .shape = context->dl_shape.data(),
                                       .strides = context->dl_strides.data(),
                                       .byte_offset = 0};
  Tensor tensor{context};
  auto metadata = schema::build_image_metadata(
      tensor,
      schema::ImageMetadataOptions{.height = image.height,
                                   .width = image.width,
                                   .encoding = image.encoding,
                                   .plane_layouts = image.plane_layouts});
  ASSERT_TRUE(metadata);
}

// REQ: VISION-005
TEST(VisionImageMetadata, BayerCropPhaseIsRelativeToTheInputMosaic) {
  using namespace schema;
  struct MosaicCase {
    ImageEncoding input;
    std::array<ImageEncoding, 4> cropped;
  };
  // Expected mosaics at origins (0,0), (1,0), (0,1), and (1,1).
  const MosaicCase cases[]{
      {ImageEncoding_BAYER_RGGB,
       {ImageEncoding_BAYER_RGGB,
        ImageEncoding_BAYER_GRBG,
        ImageEncoding_BAYER_GBRG,
        ImageEncoding_BAYER_BGGR}},
      {ImageEncoding_BAYER_BGGR,
       {ImageEncoding_BAYER_BGGR,
        ImageEncoding_BAYER_GBRG,
        ImageEncoding_BAYER_GRBG,
        ImageEncoding_BAYER_RGGB}},
      {ImageEncoding_BAYER_GRBG,
       {ImageEncoding_BAYER_GRBG,
        ImageEncoding_BAYER_RGGB,
        ImageEncoding_BAYER_BGGR,
        ImageEncoding_BAYER_GBRG}},
      {ImageEncoding_BAYER_GBRG,
       {ImageEncoding_BAYER_GBRG,
        ImageEncoding_BAYER_BGGR,
        ImageEncoding_BAYER_RGGB,
        ImageEncoding_BAYER_GRBG}},
  };
  const auto input = scalar_image_view(1, {kDLUInt, 16, 1});
  auto output = input;
  output.shape = {1, 2, 1, 0};
  output.byte_strides = {4, 2, 2, 0};
  output.width = 2;
  output.height = 1;
  for (const auto& test : cases) {
    for (std::int32_t y = 0; y < 2; ++y) {
      for (std::int32_t x = 0; x < 2; ++x) {
        SCOPED_TRACE(::testing::Message() << "encoding=" << test.input << " x=" << x << " y=" << y);
        auto image = rgb_image();
        image.encoding = test.input;
        image.significant_bits = 12;
        image.roi_offset_x = 11;
        image.roi_offset_y = 21;
        ASSERT_TRUE(validate_image_payload(image, input));
        ASSERT_TRUE(update_image_metadata(
            image, input, output, {.geometry = ImageGeometryChange::kCrop, .x = x, .y = y}));
        EXPECT_EQ(image.encoding, test.cropped[y * 2 + x]);
        EXPECT_EQ(image.roi_offset_x, 11 + x);
        EXPECT_EQ(image.roi_offset_y, 21 + y);
        EXPECT_EQ(image.significant_bits, 12);
        EXPECT_TRUE(validate_image_payload(image, output));
      }
    }
  }
}

// REQ: VISION-005
TEST(VisionImageMetadata, PaddingAdjustsRepresentableSensorOriginsAndResetsOthers) {
  const auto input = rgb_view();
  auto output = input;
  output.shape = {5, 5, 3, 0};
  output.byte_strides = {15, 3, 1, 0};
  output.width = 5;
  output.height = 5;
  const std::array<std::int32_t, 4> origins[]{{10, 20, 8, 17}, {1, 20, 0, 0}, {10, 2, 0, 0}};
  for (const auto& origin : origins) {
    auto image = rgb_image();
    image.roi_offset_x = origin[0];
    image.roi_offset_y = origin[1];
    ASSERT_TRUE(update_image_metadata(
        image, input, output, {.geometry = ImageGeometryChange::kPad, .x = 2, .y = 3}));
    EXPECT_EQ(image.roi_offset_x, origin[2]);
    EXPECT_EQ(image.roi_offset_y, origin[3]);
    EXPECT_EQ(image.encoding, schema::ImageEncoding_RGB8);
    EXPECT_EQ(image.color_space, schema::ColorSpace_SRGB);
    EXPECT_EQ(image.significant_bits, 7);
  }

  auto mosaic = rgb_image();
  mosaic.encoding = schema::ImageEncoding_BAYER_RGGB;
  const auto mono = scalar_image_view(1, {kDLUInt, 8, 1});
  ASSERT_TRUE(update_image_metadata(
      mosaic, mono, mono, {.geometry = ImageGeometryChange::kPad, .preserves_precision = false}));
  EXPECT_EQ(mosaic.roi_offset_x, 10);
  EXPECT_EQ(mosaic.roi_offset_y, 20);
  EXPECT_EQ(mosaic.encoding, schema::ImageEncoding_BAYER_RGGB);
  EXPECT_EQ(mosaic.significant_bits, 7);
}

// REQ: VISION-006
TEST(VisionImageMetadata, MonoReformatRetainsSinglePlaneEncodingAndPrecision) {
  auto image = rgb_image();
  image.encoding = schema::ImageEncoding_MONO16;
  image.significant_bits = 12;
  const auto interleaved = scalar_image_view(1, {kDLUInt, 16, 1});
  const auto planar = scalar_image_view(1, {kDLUInt, 16, 1}, ImageTensorLayout::kPlanar);
  ASSERT_TRUE(validate_image_payload(image, interleaved));
  ASSERT_TRUE(update_image_metadata(image, interleaved, planar));
  EXPECT_EQ(image.encoding, schema::ImageEncoding_MONO16);
  EXPECT_TRUE(image.plane_layouts.empty());
  EXPECT_EQ(image.significant_bits, 12);
  ASSERT_TRUE(validate_image_payload(image, planar));
  ASSERT_TRUE(update_image_metadata(image, planar, interleaved));
  EXPECT_EQ(image.encoding, schema::ImageEncoding_MONO16);
  EXPECT_TRUE(image.plane_layouts.empty());
  EXPECT_EQ(image.color_space, schema::ColorSpace_SRGB);
  EXPECT_TRUE(validate_image_payload(image, interleaved));
}

// REQ: VISION-006
TEST(VisionImageMetadata, ReformatUsesCustomForPlanarFormatsWithoutSchemaEncodings) {
  struct FormatCase {
    schema::ImageEncoding encoding;
    std::int64_t channels;
    DLDataType dtype;
  };
  const FormatCase cases[]{
      {schema::ImageEncoding_CUSTOM, 2, {kDLUInt, 8, 1}},
      {schema::ImageEncoding_RGBA8, 4, {kDLUInt, 8, 1}},
      {schema::ImageEncoding_RGB16, 3, {kDLUInt, 16, 1}},
      {schema::ImageEncoding_RGBA16, 4, {kDLUInt, 16, 1}},
      {schema::ImageEncoding_RGB32F, 3, {kDLFloat, 32, 1}},
      {schema::ImageEncoding_RGBA32F, 4, {kDLFloat, 32, 1}},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.encoding);
    auto image = rgb_image();
    image.encoding = test.encoding;
    image.significant_bits = 0;
    const auto input = scalar_image_view(test.channels, test.dtype);
    const auto output = scalar_image_view(test.channels, test.dtype, ImageTensorLayout::kPlanar);
    ASSERT_TRUE(validate_image_payload(image, input));
    ASSERT_TRUE(update_image_metadata(image, input, output));
    EXPECT_EQ(image.encoding, schema::ImageEncoding_CUSTOM);
    ASSERT_EQ(image.plane_layouts.size(), static_cast<std::size_t>(test.channels));
    EXPECT_EQ(image.color_space, schema::ColorSpace_SRGB);
    EXPECT_EQ(image.significant_bits, 0);
    ASSERT_TRUE(validate_image_payload(image, output));
    ASSERT_TRUE(update_image_metadata(image, output, input));
    EXPECT_EQ(image.encoding, schema::ImageEncoding_CUSTOM);
    EXPECT_TRUE(image.plane_layouts.empty());
    EXPECT_TRUE(validate_image_payload(image, input));
  }
}

// REQ: VISION-006
TEST(VisionImageMetadata, NumericConversionMapsNamedMonoAndRgbStorageEncodings) {
  struct CastCase {
    schema::ImageEncoding input_encoding;
    DLDataType input_type;
    std::int64_t channels;
    DLDataType output_type;
    schema::ImageEncoding output_encoding;
    std::uint8_t expected_bits;
  };
  const CastCase cases[]{
      {schema::ImageEncoding_MONO8,
       {kDLUInt, 8, 1},
       1,
       {kDLUInt, 16, 1},
       schema::ImageEncoding_MONO16,
       7},
      {schema::ImageEncoding_MONO16,
       {kDLUInt, 16, 1},
       1,
       {kDLFloat, 32, 1},
       schema::ImageEncoding_MONO32F,
       7},
      {schema::ImageEncoding_MONO32F,
       {kDLFloat, 32, 1},
       1,
       {kDLUInt, 8, 1},
       schema::ImageEncoding_MONO8,
       255},
      {schema::ImageEncoding_MONO16,
       {kDLUInt, 16, 1},
       1,
       {kDLInt, 16, 1},
       schema::ImageEncoding_CUSTOM,
       255},
      {schema::ImageEncoding_RGB8,
       {kDLUInt, 8, 1},
       3,
       {kDLFloat, 32, 1},
       schema::ImageEncoding_RGB32F,
       7},
      {schema::ImageEncoding_RGB32F,
       {kDLFloat, 32, 1},
       3,
       {kDLUInt, 8, 1},
       schema::ImageEncoding_RGB8,
       255},
      {schema::ImageEncoding_RGB8,
       {kDLUInt, 8, 1},
       3,
       {kDLUInt, 16, 1},
       schema::ImageEncoding_RGB16,
       7},
      {schema::ImageEncoding_RGB16,
       {kDLUInt, 16, 1},
       3,
       {kDLUInt, 8, 1},
       schema::ImageEncoding_RGB8,
       255},
      {schema::ImageEncoding_RGB16,
       {kDLUInt, 16, 1},
       3,
       {kDLFloat, 32, 1},
       schema::ImageEncoding_RGB32F,
       7},
      {schema::ImageEncoding_RGB32F,
       {kDLFloat, 32, 1},
       3,
       {kDLUInt, 16, 1},
       schema::ImageEncoding_RGB16,
       255},
      {schema::ImageEncoding_RGBA8,
       {kDLUInt, 8, 1},
       4,
       {kDLUInt, 16, 1},
       schema::ImageEncoding_RGBA16,
       7},
      {schema::ImageEncoding_RGBA16,
       {kDLUInt, 16, 1},
       4,
       {kDLUInt, 8, 1},
       schema::ImageEncoding_RGBA8,
       255},
      {schema::ImageEncoding_RGBA16,
       {kDLUInt, 16, 1},
       4,
       {kDLFloat, 32, 1},
       schema::ImageEncoding_RGBA32F,
       7},
      {schema::ImageEncoding_RGBA32F,
       {kDLFloat, 32, 1},
       4,
       {kDLUInt, 16, 1},
       schema::ImageEncoding_RGBA16,
       255},
      {schema::ImageEncoding_RGB8,
       {kDLUInt, 8, 1},
       3,
       {kDLUInt, 32, 1},
       schema::ImageEncoding_CUSTOM,
       7},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "input=" << test.input_encoding << " output=" << test.output_encoding);
    auto image = rgb_image();
    image.encoding = test.input_encoding;
    const auto input = scalar_image_view(test.channels, test.input_type);
    const auto output = scalar_image_view(test.channels, test.output_type);
    ASSERT_TRUE(validate_image_payload(image, input));
    ASSERT_TRUE(update_image_metadata(image, input, output));
    EXPECT_EQ(image.encoding, test.output_encoding);
    EXPECT_EQ(image.significant_bits, test.expected_bits);
    EXPECT_EQ(image.color_space, schema::ColorSpace_UNSPECIFIED);
    EXPECT_TRUE(validate_image_payload(image, output));
  }
}

// REQ: VISION-004
TEST(VisionImageMetadata, PitchedNCHWBatchDescribesTheFirstFramePlanes) {
  alignas(16) std::array<std::byte, 55> storage{};
  auto context = std::make_shared<DLManagedTensorContext>();
  context->dl_shape = {2, 3, 2, 3};
  context->dl_strides = {32, 8, 4, 1};
  context->tensor.dl_tensor = DLTensor{.data = storage.data(),
                                       .device = {kDLCUDA, 0},
                                       .ndim = 4,
                                       .dtype = {kDLUInt, 8, 1},
                                       .shape = context->dl_shape.data(),
                                       .strides = context->dl_strides.data(),
                                       .byte_offset = 0};
  auto image = rgb_image();
  image.data = std::make_shared<Tensor>(context);
  const auto output =
      describe_image_tensor(*image.data, TensorOperation::kReformat, 2, ImageTensorLayout::kPlanar);
  ASSERT_TRUE(output);
  auto input = rgb_view();
  input.shape = {2, 2, 3, 3};
  input.byte_strides = {32, 12, 3, 1};
  input.rank = 4;
  input.batch = 2;
  ASSERT_TRUE(update_image_metadata(image, input, *output));
  EXPECT_EQ(image.encoding, schema::ImageEncoding_R8_G8_B8);
  ASSERT_EQ(image.plane_layouts.size(), 3U);
  const std::array<std::uint64_t, 3> offsets{0, 8, 16};
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_EQ(image.plane_layouts[i].offset_bytes(), offsets[i]);
    EXPECT_EQ(image.plane_layouts[i].row_stride_bytes(), 4U);
    EXPECT_EQ(image.plane_layouts[i].size_bytes(), 7U);
  }
  EXPECT_TRUE(
      describe_image_payload(image, TensorOperation::kReformat, 2, ImageTensorLayout::kPlanar));
  const auto metadata = schema::build_image_metadata(
      *image.data,
      schema::ImageMetadataOptions{.height = image.height,
                                   .width = image.width,
                                   .encoding = image.encoding,
                                   .plane_layouts = image.plane_layouts});
  ASSERT_TRUE(metadata);
}

}  // namespace
}  // namespace holoscan::vision::detail
