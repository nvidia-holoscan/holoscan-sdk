// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/vision/holoscan_vision_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/vision/vision_types.hpp>

namespace holoscan::vision {

/// @brief Configure Bayer demosaic reconstruction for \ref BayerDemosaicOp.
struct BayerDemosaicOpParams {
  /// CFA phase at the input ROI origin. Named BAYER_* encodings must match this value.
  /// For CUSTOM, MONO8, and MONO16, this parameter declares the otherwise unnamed phase.
  BayerGridPosition bayer_grid_pos{BayerGridPosition::kGBRG};
  bool generate_alpha{false};  ///< Emit a fourth (alpha) output channel.
  /// Constant alpha value when generate_alpha is set. Passed to NPP unscaled: the default 255
  /// is a typical opaque 8-bit value, not a fraction of the 16-bit range, so a caller wanting
  /// full-scale 16-bit alpha (e.g. 65535) must say so explicitly for the RGBA16 path. The 8-bit
  /// path requires 0-255 when generate_alpha is set; larger values are rejected before allocation.
  std::uint16_t alpha_value{255U};
};

HOLOSCAN_PARAM_SCHEMA(
    BayerDemosaicOpParams,
    holoscan::param_schema("com.nvidia.holoscan.vision.BayerDemosaicOpParams", 2),
    holoscan::field<&BayerDemosaicOpParams::bayer_grid_pos>("bayer_grid_pos")
        .enum_values({holoscan::param_enum_value("bggr", BayerGridPosition::kBGGR),
                      holoscan::param_enum_value("rggb", BayerGridPosition::kRGGB),
                      holoscan::param_enum_value("gbrg", BayerGridPosition::kGBRG),
                      holoscan::param_enum_value("grbg", BayerGridPosition::kGRBG)})
        .description("Color-filter-array phase of the input mosaic's top-left 2x2 block."),
    holoscan::field<&BayerDemosaicOpParams::generate_alpha>("generate_alpha")
        .description("Emit a fourth, constant-valued alpha output channel."),
    holoscan::field<&BayerDemosaicOpParams::alpha_value>("alpha_value")
        .description("Constant alpha channel value used when generate_alpha is set; passed to "
                     "NPP unscaled, so 8-bit output must stay within 0-255."));

/// @brief Reconstruct full-color images from a single-channel Bayer mosaic with NPP.
///
/// The operator wraps NVIDIA Performance Primitives' (NPP) `nppiCFAToRGB`/`nppiCFAToRGBA`
/// kernels, which have no CV-CUDA equivalent. NPP's Bayer kernels are stateless free functions,
/// so the operator retains no CUDA resource; the retained state is bounded, host-side metadata
/// scratch (see below), not a CV-CUDA algorithm handle.
///
/// Input images carry a single-channel (HWC, C=1) CFA mosaic; width and height must both be
/// even. Output images have three (RGB) or four (RGBA, when generate_alpha is set) channels
/// at the same dimensions and element type as the input. Unsigned 8-bit and unsigned 16-bit
/// input are both supported, dispatching to NPP's 8u or 16u Bayer kernels accordingly.
///
/// The operator borrows its selected input via `Input<T>::selected_sample()` and edits bounded
/// metadata-only scratch prepared during `start()`, avoiding a generated descriptor copy on
/// every callback; see \ref sensor_io::ImageDescriptorScratch.
///
/// Ports carry schema::ImageT with a scalar CUDA tensor in its data field. Image dimensions,
/// encoding, and supplied planes must match tensor geometry, dtype, channels, and strides.
/// Output encoding becomes RGB8/RGBA8 for 8-bit input or RGB16/RGBA16 for 16-bit input; image
/// geometry (dimensions, ROI offsets) is preserved since demosaic reconstructs pixels without
/// moving or resampling the sensor field of view. significant_bits and color_space are reset,
/// since interpolated RGB values no longer share the raw sensor's code range or color
/// interpretation. Publication attaches the committed output loan through emit_tensor()
/// without copying pixel storage.
class HOLOSCAN_VISION_EXPORT BayerDemosaicOp final : public Operator<BayerDemosaicOpParams> {
 public:
  /// @brief Construct an operator whose metadata scratch is prepared in `start()`.
  BayerDemosaicOp();

  /// @brief Destroy the operator after runtime quiescence.
  ~BayerDemosaicOp() override;

  /// @brief Declare image ports and bounded output allocation.
  /// @param spec Fresh operator declaration surface.
  void setup(OperatorSpec& spec) override;

  /// @brief Trigger once for each selected input image.
  /// @return The operator's cold temporal contract.
  [[nodiscard]] Contract contract() const override;

  /// @brief Prepare bounded metadata-only scratch for a running session.
  void start() override;

  /// @brief End a session while retaining prepared scratch through runtime quiescence.
  void stop() override;

  /// @brief Demosaic one Bayer mosaic image and publish a plan-owned output image.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<schema::ImageT> input{"input"};     ///< Image carrying a single-channel Bayer mosaic.
  Output<schema::ImageT> output{"output"};  ///< Image carrying the reconstructed RGB(A) tensor.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

static_assert(param_schema_availability<BayerDemosaicOpParams>() ==
                  ParamSchemaAvailability::kDeclared,
              "BayerDemosaicOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
