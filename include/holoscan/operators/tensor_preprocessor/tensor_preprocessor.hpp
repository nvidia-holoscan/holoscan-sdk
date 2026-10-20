/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include <holoscan/holoscan_core_export.h>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace YAML {
class Node;
}  // namespace YAML

namespace holoscan::ops {

/// @brief Describe one plane inside a flat byte buffer input Tensor.
///
/// Used only when `input_kind == "tensor"` and `input_pix_fmt` names a planar YUV format
/// (`nv12` / `i420`) whose plane geometry is not carried by the Tensor's shape (e.g. an
/// Argus / SIPL capture operator publishing a flat `NvSciBuf`-backed buffer). Every plane
/// referenced by this vector must lie entirely inside `tensor.byte_size()`.
struct TensorInputPlane {
  /// Byte offset of this plane's first sample from the tensor's data pointer.
  std::uint64_t offset_bytes{};
  /// Distance in bytes between consecutive rows of this plane. Must be at least the plane's
  /// row-width-in-bytes (e.g. `input_width` for a Y plane; `input_width / 2` for an I420
  /// chroma plane; `input_width` for the NV12 interleaved CbCr plane).
  std::uint64_t row_stride_bytes{};
};

/**
 * @brief Compile-time-authored configuration for TensorPreprocessorOp.
 *
 * Resizes (via NPP `nppiResize_8u_C3R`), optionally converts to float32 with per-channel
 * mean/std normalization, and optionally transposes HWC->CHW to produce the fixed
 * shape/dtype/layout an inference model expects. Runs entirely on the GPU against the
 * Holoscan CUDA stream using NVIDIA Performance Primitives (NPP).
 */
struct TensorPreprocessorOpParams {
  /// Name of the single declared input port.
  std::string input_port_name{"in"};
  /// Name of the single declared `Output<Tensor>` port.
  std::string output_port_name{"out"};
  /// Kind of input port to declare.
  /// - `tensor` (default): `Input<Tensor>` with `.accepts_projected_tensor()`. The pixel format
  ///   comes from `input_pix_fmt`; the shape/dtype are read directly from the incoming Tensor.
  /// - `image`: `Input<schema::ImageT>` with `.tensor_field(schema::ImageTensorFields::data)`.
  ///   The pixel format and plane layout come from the `ImageT` metadata (encoding,
  ///   plane_layouts, color_space) per frame; `input_pix_fmt` and `yuv_matrix`/`yuv_range` are
  ///   overrides used only when the metadata field is UNSPECIFIED.
  std::string input_kind{"tensor"};
  /// Pixel format of the incoming Tensor (only consulted when `input_kind == "tensor"`, or as
  /// a fallback when the incoming ImageT carries `ImageEncoding_CUSTOM`).
  /// - `rgb24` (default): interleaved [H, W, 3] uint8 RGB. No color conversion.
  /// - `yuyv`: packed 4:2:2 YCbYCr (Y0-Cb-Y1-Cr per two-pixel macropixel), shape [H, W, 2] or
  ///   [1, H, W, 2] uint8; width must be even.
  /// - `nv12`: semi-planar 4:2:0 (Y plane + interleaved CbCr plane). Requires either
  ///   `input_kind=image` (plane geometry from `ImageT.plane_layouts`) or `input_kind=tensor`
  ///   with `input_width`/`input_height`/`input_planes` set.
  /// - `i420`: 3-plane 4:2:0 (Y, Cb, Cr planes). Same geometry requirement as `nv12`.
  std::string input_pix_fmt{"rgb24"};
  /// Pixel width of the incoming frame; consulted only for the flat bare-Tensor NV12/I420
  /// path (where the Tensor is a rank-1 byte buffer and shape carries no width). Ignored for
  /// rank-3/4 RGB24 and rank-2/3/4 YUYV inputs (width is derived from `tensor.shape()`) and
  /// for the ImageT path (width is derived from `ImageT.width`).
  int32_t input_width{};
  /// Pixel height of the incoming frame; consulted only for the flat bare-Tensor NV12/I420
  /// path. Ignored otherwise for the same reason as `input_width`.
  int32_t input_height{};
  /// Plane descriptors (offset + row stride) for the flat bare-Tensor NV12/I420 path. Must
  /// contain exactly 2 entries for NV12 (Y, interleaved CbCr) or 3 for I420 (Y, Cb, Cr).
  /// Ignored for every other `input_pix_fmt` and for the ImageT path (which uses
  /// `ImageT.plane_layouts`). Every plane must fit inside `tensor.byte_size()` and every
  /// row-stride must be at least the plane's width-in-bytes (checked at `compute()`-time).
  /// Typical Argus NV12 layout at `W × H`: `{ {0, W}, {W * H, W} }`. Typical Argus I420:
  /// `{ {0, W}, {W * H, W / 2}, {W * H + (W * H) / 4, W / 2} }`. SIPL NV12 with independent
  /// Y and CbCr pitches: `{ {0, y_pitch}, {uv_offset, uv_pitch} }` — but NPP's NV12 CSC
  /// requires equal Y and CbCr strides, so mismatched pitches are rejected here.
  std::vector<TensorInputPlane> input_planes{};
  /// Required output width, in pixels.
  int32_t output_width{};
  /// Required output height, in pixels.
  int32_t output_height{};
  /// Pixel format of the emitted Tensor: `rgb_u8` (interleaved uint8, unnormalized) or `rgb_fp32`
  /// (per-channel-normalized float32).
  std::string output_pix_fmt{"rgb_fp32"};
  /// Channel layout of the emitted Tensor: `nchw` (channel-first) or `nhwc` (channel-last).
  std::string output_layout{"nchw"};
  /// When `true`, prepend a leading extent of `1` to the emitted Tensor's shape (a batch
  /// dimension), matching InferenceOp's typical `[1, C, H, W]` / `[1, H, W, C]` expectation.
  bool prepend_batch_dim{true};
  /// Per-channel normalization applied only when `output_pix_fmt == "rgb_fp32"`:
  /// `x_fp32 = (x_uint8 / 255 - mean[c]) / std[c]`. Fixed 3 entries (R, G, B).
  std::array<float, 3> normalize_mean{0.0F, 0.0F, 0.0F};
  std::array<float, 3> normalize_std{1.0F, 1.0F, 1.0F};
  /// YCbCr → RGB colorimetry matrix. Consulted per-format for YUV inputs.
  /// - `bt601` (default): ITU-R BT.601 matrix (SDTV, JFIF).
  /// - `bt709`: ITU-R BT.709 matrix (HDTV, most modern camera pipelines).
  ///
  /// Overridden per-frame when `input_kind == "image"` and the incoming ImageT declares a
  /// non-`UNSPECIFIED` `color_space` (REC601 → bt601, REC709 → bt709).
  std::string yuv_matrix{"bt601"};
  /// YCbCr sample range. Note: NPP's default YUYV/YCbCr422 and YCbCr420 dispatchers assume
  /// full-range luma; the `nv12` path additionally supports `bt709` via
  /// `nppiNV12ToRGB_709CSC_...`. The `limited` value is accepted and recorded but only
  /// meaningfully differentiated for the NV12 dispatch today.
  /// - `full` (default when unspecified): 0..255 luma / chroma.
  /// - `limited`: 16..235 luma, 16..240 chroma (video-typical).
  std::string yuv_range{"full"};
  /// Memory kind of the emitted Tensor.
  holoscan::MemoryKind memory_kind{holoscan::MemoryKind::kCudaDevice};

  /**
   * @brief Populate one `TensorPreprocessorOpParams` from a `preprocessor:` YAML map.
   *
   * ```yaml
   * preprocessor:
   *   input_port_name: raw_frame          # default: "in"
   *   output_port_name: de_id_preprocessed # default: "out"
   *   input_pix_fmt: rgb24                 # default: "rgb24"; also "yuyv"
   *   output_width: 256                    # required
   *   output_height: 256                   # required
   *   output_pix_fmt: rgb_fp32             # default: "rgb_fp32"; also "rgb_u8"
   *   output_layout: nchw                  # default: "nchw"; also "nhwc"
   *   prepend_batch_dim: true              # default: true
   *   normalize_mean: [0.0, 0.0, 0.0]       # default: [0, 0, 0]
   *   normalize_std: [1.0, 1.0, 1.0]        # default: [1, 1, 1]
   *   memory_kind: cuda_device              # default: cuda_device
   * ```
   *
   * @param node The `preprocessor:` map.
   * @param config_dir Unused (present for signature parity with the other `*_from_yaml_node()`
   *   helpers in this port; TensorPreprocessorOpParams has no path-valued field).
   * @return The populated `TensorPreprocessorOpParams`.
   * @throws std::runtime_error if `output_width`/`output_height` is missing, a
   *   `normalize_mean`/`normalize_std` list does not have exactly 3 entries, or
   *   `output_pix_fmt`/`output_layout` names an unsupported value.
   */
  [[nodiscard]] static TensorPreprocessorOpParams from_yaml_node(
      const YAML::Node& node, const std::filesystem::path& config_dir);
};

/**
 * @brief GPU-native resize + optional normalize + optional HWC->CHW transpose.
 *
 * All work runs on the CUDA device against the Holoscan stream, using NVIDIA Performance
 * Primitives (NPP): `nppiResize_8u_C3R_Ctx` for bilinear resize, `nppiConvert_8u32f_C3R_Ctx`
 * for the u8->f32 dtype cast, `nppiMulC_32f_C3IR_Ctx` / `nppiSubC_32f_C3IR_Ctx` /
 * `nppiDivC_32f_C3IR_Ctx` for the `x = (x/255 - mean) / std` normalization, and
 * `nppiCopy_8u_C3P3R_Ctx` / `nppiCopy_32f_C3P3R_Ctx` for the HWC->CHW plane split. Only the
 * device memory kind (`kCudaDevice` / `kCudaManaged`) is supported for both input and output;
 * requesting a host-side output raises an error.
 */
class HOLOSCAN_CORE_EXPORT TensorPreprocessorOp
    : public holoscan::Operator<TensorPreprocessorOpParams> {
 public:
  TensorPreprocessorOp() = default;

  /// @brief Construct with `params` already bound (mirrors `InferenceOp(InferenceOpParams)`).
  /// @throws std::runtime_error if `params` fails aggregate validation.
  explicit TensorPreprocessorOp(TensorPreprocessorOpParams params);

  ~TensorPreprocessorOp() override;

  TensorPreprocessorOp(const TensorPreprocessorOp&) = delete;
  TensorPreprocessorOp& operator=(const TensorPreprocessorOp&) = delete;

  void setup(OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  [[nodiscard]] Contract contract() const override;

  /// @brief The single dynamically-named `Input<Tensor>` port (declared when
  /// `params().input_kind == "tensor"`). Non-null even when the ImageT port is active, so
  /// downstream `graph.add_flow()` calls that reach through `input_port()` produce a clear
  /// declared-nowhere diagnostic instead of a null-dereference.
  [[nodiscard]] Input<Tensor>* input_port() const { return &input_; }

  /// @brief The single dynamically-named `Input<schema::ImageT>` port (declared when
  /// `params().input_kind == "image"`).
  [[nodiscard]] Input<schema::ImageT>* image_input_port() const { return &image_input_; }

  /// @brief The single dynamically-named output port (declared in setup() as
  /// `params().output_port_name`).
  [[nodiscard]] Output<Tensor>* output_port() const { return &output_; }

 private:
  // Mutable so the const input_port()/output_port() accessors (needed because graph.op<>()
  // returns a const-qualified OperatorHandle -- see VideoFileReaderOp::output_port() for the
  // identical pattern) can still hand out mutable Input<Tensor>*/Output<Tensor>* for
  // graph.add_flow() to bind at compile time.
  mutable Input<Tensor> input_;
  mutable Input<schema::ImageT> image_input_;
  mutable Output<Tensor> output_;

  int32_t channels_{3};
  std::vector<int64_t> output_shape_;
  std::size_t output_frame_bytes_{};

  // Device staging buffers, sized once at start(). `d_resized_u8_` holds the interleaved-RGB
  // uint8 image at output resolution; `d_resized_f32_` holds the same shape as float32 after
  // dtype conversion + normalization. `d_resized_f32_` is only allocated for the fp32 path.
  void* d_resized_u8_{};
  void* d_resized_f32_{};
  std::size_t d_resized_u8_bytes_{};
  std::size_t d_resized_f32_bytes_{};

  void* d_yuv_rgb_u8_{};
  std::size_t d_yuv_rgb_u8_bytes_{};

  // Heap-allocated NppStreamContext (kept as void* here so the header does not pull in
  // <nppdefs.h>). Initialized once in start() and re-bound to the frame's CUDA stream in
  // compute().
  void* npp_ctx_{};
};

}  // namespace holoscan::ops
