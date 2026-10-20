/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/tensor_preprocessor/tensor_preprocessor.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <npp.h>
#include <yaml-cpp/yaml.h>

#include <holoscan/core/graph.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>

namespace holoscan::ops {

namespace {

[[nodiscard]] holoscan::MemoryKind memory_kind_from_string(const std::string& name) {
  if (name == "cuda_device")
    return holoscan::MemoryKind::kCudaDevice;
  if (name == "cuda_managed")
    return holoscan::MemoryKind::kCudaManaged;
  throw std::runtime_error(
      "unsupported memory_kind '" + name +
      "' (TensorPreprocessorOp runs on-device via NPP; only cuda_device/cuda_managed are "
      "supported)");
}

[[nodiscard]] bool is_float_out(const std::string& pix_fmt) {
  if (pix_fmt == "rgb_u8")
    return false;
  if (pix_fmt == "rgb_fp32")
    return true;
  throw std::runtime_error("unsupported output_pix_fmt '" + pix_fmt +
                           "' (expected one of rgb_u8/rgb_fp32)");
}

[[nodiscard]] bool is_nchw(const std::string& layout) {
  if (layout == "nchw")
    return true;
  if (layout == "nhwc")
    return false;
  throw std::runtime_error("unsupported output_layout '" + layout + "' (expected nchw or nhwc)");
}

// Input pixel format taxonomy. Every YUV variant lands in interleaved 8-bit RGB before
// resize; the specific NPP CSC call is chosen by pix_fmt + yuv_matrix at compute time.
enum class InputPixFmt : std::uint8_t {
  kRgb24,  // interleaved [H, W, 3] uint8 RGB. No color conversion.
  kYuyv,   // packed 4:2:2 YCbYCr ([H, W, 2] uint8, even W).
  kNv12,   // semi-planar 4:2:0 (Y plane + interleaved CbCr plane).
  kI420,   // 3-plane 4:2:0 (Y, Cb, Cr planes).
};

enum class InputKind : std::uint8_t { kTensor, kImage };

enum class YuvMatrix : std::uint8_t { kBt601, kBt709 };

enum class YuvRange : std::uint8_t { kFull, kLimited };

[[nodiscard]] InputPixFmt parse_input_pix_fmt(const std::string& name) {
  if (name == "rgb24")
    return InputPixFmt::kRgb24;
  if (name == "yuyv")
    return InputPixFmt::kYuyv;
  if (name == "nv12")
    return InputPixFmt::kNv12;
  if (name == "i420")
    return InputPixFmt::kI420;
  throw std::runtime_error("unsupported input_pix_fmt '" + name +
                           "' (expected one of rgb24/yuyv/nv12/i420)");
}

[[nodiscard]] InputKind parse_input_kind(const std::string& name) {
  if (name == "tensor")
    return InputKind::kTensor;
  if (name == "image")
    return InputKind::kImage;
  throw std::runtime_error("unsupported input_kind '" + name + "' (expected tensor/image)");
}

[[nodiscard]] YuvMatrix parse_yuv_matrix(const std::string& name) {
  if (name == "bt601")
    return YuvMatrix::kBt601;
  if (name == "bt709")
    return YuvMatrix::kBt709;
  throw std::runtime_error("unsupported yuv_matrix '" + name + "' (expected bt601/bt709)");
}

[[nodiscard]] YuvRange parse_yuv_range(const std::string& name) {
  if (name == "full")
    return YuvRange::kFull;
  if (name == "limited")
    return YuvRange::kLimited;
  throw std::runtime_error("unsupported yuv_range '" + name + "' (expected full/limited)");
}

[[nodiscard]] bool is_yuv(InputPixFmt fmt) {
  return fmt == InputPixFmt::kYuyv || fmt == InputPixFmt::kNv12 || fmt == InputPixFmt::kI420;
}

// Translate an ImageT ColorSpace to the internal YuvMatrix. Returns std::nullopt when the
// ImageT metadata does not identify a matrix (UNSPECIFIED, sRGB) -- the caller falls back
// to the configured yuv_matrix override in that case.
[[nodiscard]] std::optional<YuvMatrix> matrix_from_color_space(schema::ColorSpace cs) noexcept {
  switch (cs) {
    case schema::ColorSpace_REC601:
      return YuvMatrix::kBt601;
    case schema::ColorSpace_REC709:
      return YuvMatrix::kBt709;
    default:
      return std::nullopt;
  }
}

// Translate an ImageT ImageEncoding to the internal InputPixFmt. Returns std::nullopt for
// ImageEncoding_CUSTOM (caller falls back to the configured input_pix_fmt override) and
// throws for encodings this operator does not implement yet.
[[nodiscard]] std::optional<InputPixFmt> pix_fmt_from_encoding(schema::ImageEncoding enc) {
  switch (enc) {
    case schema::ImageEncoding_CUSTOM:
      return std::nullopt;
    case schema::ImageEncoding_RGB8:
      return InputPixFmt::kRgb24;
    case schema::ImageEncoding_YUYV:
      return InputPixFmt::kYuyv;
    case schema::ImageEncoding_NV12:
      return InputPixFmt::kNv12;
    case schema::ImageEncoding_I420:
      return InputPixFmt::kI420;
    default:
      throw std::runtime_error(
          "TensorPreprocessorOp: unsupported ImageT encoding value " +
          std::to_string(static_cast<int>(enc)) +
          " (supported: RGB8, YUYV, NV12, I420; use ImageEncoding_CUSTOM to defer to "
          "input_pix_fmt)");
  }
}

[[nodiscard]] Error npp_error(const char* what, NppStatus status) {
  return Error{ErrorCode::kFailure,
               std::string("TensorPreprocessorOp: ") + what +
                   " (NppStatus=" + std::to_string(static_cast<int>(status)) + ")"};
}

[[nodiscard]] Error cuda_error(const char* what, cudaError_t status) {
  return Error{ErrorCode::kFailure,
               std::string("TensorPreprocessorOp: ") + what +
                   " (cudaError=" + std::string(cudaGetErrorString(status)) + ")"};
}

}  // namespace

// Each independent optional field is parsed and validated in schema order; splitting this
// declarative loader would scatter the parameter contract.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TensorPreprocessorOpParams TensorPreprocessorOpParams::from_yaml_node(
    const YAML::Node& node, const std::filesystem::path& config_dir) {
  static_cast<void>(config_dir);

  if (!node["output_width"] || !node["output_height"]) {
    throw std::runtime_error(
        "TensorPreprocessorOpParams::from_yaml_node: missing required key 'output_width' or "
        "'output_height'");
  }

  TensorPreprocessorOpParams params;

  if (const auto in_node = node["input_port_name"]) {
    params.input_port_name = in_node.as<std::string>();
  }
  if (const auto out_node = node["output_port_name"]) {
    params.output_port_name = out_node.as<std::string>();
  }
  if (const auto kind_node = node["input_kind"]) {
    params.input_kind = kind_node.as<std::string>();
  }
  if (const auto ipf_node = node["input_pix_fmt"]) {
    params.input_pix_fmt = ipf_node.as<std::string>();
  }
  if (const auto matrix_node = node["yuv_matrix"]) {
    params.yuv_matrix = matrix_node.as<std::string>();
  }
  if (const auto range_node = node["yuv_range"]) {
    params.yuv_range = range_node.as<std::string>();
  }
  if (const auto iw_node = node["input_width"]) {
    params.input_width = iw_node.as<int32_t>();
  }
  if (const auto ih_node = node["input_height"]) {
    params.input_height = ih_node.as<int32_t>();
  }
  if (const auto planes_node = node["input_planes"]) {
    if (!planes_node.IsSequence()) {
      throw std::runtime_error(
          "TensorPreprocessorOpParams::from_yaml_node: 'input_planes' must be a sequence of "
          "{offset_bytes, row_stride_bytes} maps");
    }
    for (const auto& plane : planes_node) {
      TensorInputPlane p{};
      if (const auto off = plane["offset_bytes"]) {
        p.offset_bytes = off.as<std::uint64_t>();
      }
      if (const auto stride = plane["row_stride_bytes"]) {
        p.row_stride_bytes = stride.as<std::uint64_t>();
      } else {
        throw std::runtime_error(
            "TensorPreprocessorOpParams::from_yaml_node: each 'input_planes' entry must "
            "include 'row_stride_bytes'");
      }
      params.input_planes.push_back(p);
    }
  }
  // Validate now so from_yaml_node() fails loudly on a bad value instead of deferring to
  // setup()/start().
  static_cast<void>(parse_input_kind(params.input_kind));
  static_cast<void>(parse_input_pix_fmt(params.input_pix_fmt));
  static_cast<void>(parse_yuv_matrix(params.yuv_matrix));
  static_cast<void>(parse_yuv_range(params.yuv_range));
  params.output_width = node["output_width"].as<int32_t>();
  params.output_height = node["output_height"].as<int32_t>();
  if (const auto pix_fmt_node = node["output_pix_fmt"]) {
    params.output_pix_fmt = pix_fmt_node.as<std::string>();
  }
  if (const auto layout_node = node["output_layout"]) {
    params.output_layout = layout_node.as<std::string>();
  }
  static_cast<void>(is_float_out(params.output_pix_fmt));
  static_cast<void>(is_nchw(params.output_layout));

  if (const auto batch_node = node["prepend_batch_dim"]) {
    params.prepend_batch_dim = batch_node.as<bool>();
  }
  if (const auto mean_node = node["normalize_mean"]) {
    std::vector<float> mean;
    for (const auto& value : mean_node)
      mean.push_back(value.as<float>());
    if (mean.size() != 3U) {
      throw std::runtime_error(
          "TensorPreprocessorOpParams::from_yaml_node: 'normalize_mean' must have exactly 3 "
          "entries");
    }
    params.normalize_mean = {mean[0], mean[1], mean[2]};
  }
  if (const auto std_node = node["normalize_std"]) {
    std::vector<float> std_dev;
    for (const auto& value : std_node)
      std_dev.push_back(value.as<float>());
    if (std_dev.size() != 3U) {
      throw std::runtime_error(
          "TensorPreprocessorOpParams::from_yaml_node: 'normalize_std' must have exactly 3 "
          "entries");
    }
    params.normalize_std = {std_dev[0], std_dev[1], std_dev[2]};
  }
  if (const auto memory_kind_node = node["memory_kind"]) {
    params.memory_kind = memory_kind_from_string(memory_kind_node.as<std::string>());
  }

  return params;
}

TensorPreprocessorOp::TensorPreprocessorOp(TensorPreprocessorOpParams params) {
  auto bound = holoscan::detail::GraphOperatorAccess::bind(*this, std::move(params));
  if (!bound) {
    throw std::runtime_error("TensorPreprocessorOp: invalid TensorPreprocessorOpParams: " +
                             bound.error().message);
  }
}

TensorPreprocessorOp::~TensorPreprocessorOp() {
  if (d_resized_u8_ != nullptr) {
    cudaFree(d_resized_u8_);
    d_resized_u8_ = nullptr;
  }
  if (d_resized_f32_ != nullptr) {
    cudaFree(d_resized_f32_);
    d_resized_f32_ = nullptr;
  }
  if (d_yuv_rgb_u8_ != nullptr) {
    cudaFree(d_yuv_rgb_u8_);
    d_yuv_rgb_u8_ = nullptr;
    d_yuv_rgb_u8_bytes_ = 0;
  }
  delete static_cast<NppStreamContext*>(npp_ctx_);
  npp_ctx_ = nullptr;
}

void TensorPreprocessorOp::setup(OperatorSpec& spec) {
  const TensorPreprocessorOpParams& config = params();
  if (config.output_width <= 0 || config.output_height <= 0) {
    throw std::runtime_error(
        "TensorPreprocessorOp: output_width/output_height must both be positive");
  }
  if (config.memory_kind != holoscan::MemoryKind::kCudaDevice &&
      config.memory_kind != holoscan::MemoryKind::kCudaManaged) {
    throw std::runtime_error(
        "TensorPreprocessorOp: NPP path only supports cuda_device / cuda_managed memory_kind");
  }
  // Validate now so setup()'s tensor-contract build fails early on a bad enum value.
  const InputKind input_kind = parse_input_kind(config.input_kind);
  static_cast<void>(parse_input_pix_fmt(config.input_pix_fmt));
  static_cast<void>(parse_yuv_matrix(config.yuv_matrix));
  static_cast<void>(parse_yuv_range(config.yuv_range));
  const bool out_is_float = is_float_out(config.output_pix_fmt);
  const bool nchw = is_nchw(config.output_layout);
  channels_ = 3;

  output_shape_.clear();
  if (config.prepend_batch_dim) {
    output_shape_.push_back(1);
  }
  if (nchw) {
    output_shape_.push_back(channels_);
    output_shape_.push_back(config.output_height);
    output_shape_.push_back(config.output_width);
  } else {
    output_shape_.push_back(config.output_height);
    output_shape_.push_back(config.output_width);
    output_shape_.push_back(channels_);
  }
  const auto rank = static_cast<std::uint32_t>(output_shape_.size());

  const std::size_t pixels = static_cast<std::size_t>(config.output_width) *
                             static_cast<std::size_t>(config.output_height);
  const std::size_t element_bytes = out_is_float ? sizeof(float) : sizeof(std::uint8_t);
  output_frame_bytes_ = pixels * static_cast<std::size_t>(channels_) * element_bytes;

  const DLDataType out_dtype =
      out_is_float ? DLDataType{kDLFloat, 32U, 1U} : DLDataType{kDLUInt, 8U, 1U};

  if (input_kind == InputKind::kTensor) {
    // Bare-Tensor input path. `.accepts_projected_tensor()` opts into projected connections so
    // upstream schema-producers (e.g. a capture operator emitting `schema::ImageT`) can narrow
    // their Tensor field onto this port via graph.add_flow() without an explicit unwrap.
    spec.input(input_, config.input_port_name)
        .accepts_projected_tensor()
        .expects_tensor(TensorInputSpec{
            .representation = TensorRepresentation{.memory_kind = config.memory_kind}});
  } else {
    // ImageT input path. The consumer receives the full schema payload so encoding,
    // color_space, and plane_layouts are available per-frame; the Tensor field of the
    // ImageT is the actual pixel buffer.
    spec.input(image_input_, config.input_port_name)
        .tensor_field(schema::ImageTensorFields::data)
        .expects_tensor(TensorInputSpec{
            .representation = TensorRepresentation{.memory_kind = config.memory_kind}});
  }
  spec.output(output_, config.output_port_name)
      .produces_tensor(TensorOutputSpec{
          .representation =
              TensorRepresentation{
                  .memory_kind = config.memory_kind, .dtype = out_dtype, .rank = rank},
          .bounds = TensorBounds{.max_rank = rank, .max_byte_span = output_frame_bytes_},
          .storage = TensorOutputStorage::kRuntimePool,
      });
}

Contract TensorPreprocessorOp::contract() const {
  Contract result;
  if (parse_input_kind(params().input_kind) == InputKind::kImage) {
    result.trigger(OnEach{image_input_});
  } else {
    result.trigger(OnEach{input_});
  }
  return result;
}

void TensorPreprocessorOp::start() {
  const TensorPreprocessorOpParams& config = params();
  const std::size_t pixels = static_cast<std::size_t>(config.output_width) *
                             static_cast<std::size_t>(config.output_height);
  d_resized_u8_bytes_ = pixels * 3U * sizeof(std::uint8_t);
  if (cudaMalloc(&d_resized_u8_, d_resized_u8_bytes_) != cudaSuccess) {
    throw std::runtime_error("TensorPreprocessorOp: cudaMalloc(d_resized_u8_) failed");
  }

  if (is_float_out(config.output_pix_fmt)) {
    d_resized_f32_bytes_ = pixels * 3U * sizeof(float);
    if (cudaMalloc(&d_resized_f32_, d_resized_f32_bytes_) != cudaSuccess) {
      throw std::runtime_error("TensorPreprocessorOp: cudaMalloc(d_resized_f32_) failed");
    }
  }

  // NPP's nppGetStreamContext() is not exposed by this CUDA toolkit's header; populate the
  // context from CUDA device queries directly (each field is documented in nppdefs.h as a
  // one-to-one restatement of the corresponding cudaGetDevice*/cudaStreamGetFlags call).
  auto* ctx = new NppStreamContext{};
  int device_id = 0;
  if (cudaGetDevice(&device_id) != cudaSuccess) {
    delete ctx;
    throw std::runtime_error("TensorPreprocessorOp: cudaGetDevice failed");
  }
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess) {
    delete ctx;
    throw std::runtime_error("TensorPreprocessorOp: cudaGetDeviceProperties failed");
  }
  ctx->hStream = nullptr;
  ctx->nCudaDeviceId = device_id;
  ctx->nMultiProcessorCount = props.multiProcessorCount;
  ctx->nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
  ctx->nMaxThreadsPerBlock = props.maxThreadsPerBlock;
  ctx->nSharedMemPerBlock = props.sharedMemPerBlock;
  ctx->nCudaDevAttrComputeCapabilityMajor = props.major;
  ctx->nCudaDevAttrComputeCapabilityMinor = props.minor;
  ctx->nStreamFlags = 0U;  // rebound per-frame in compute() via cudaStreamGetFlags
  npp_ctx_ = ctx;
}

void TensorPreprocessorOp::stop() {
  if (d_resized_u8_ != nullptr) {
    cudaFree(d_resized_u8_);
    d_resized_u8_ = nullptr;
    d_resized_u8_bytes_ = 0;
  }
  if (d_resized_f32_ != nullptr) {
    cudaFree(d_resized_f32_);
    d_resized_f32_ = nullptr;
    d_resized_f32_bytes_ = 0;
  }
  if (d_yuv_rgb_u8_ != nullptr) {
    cudaFree(d_yuv_rgb_u8_);
    d_yuv_rgb_u8_ = nullptr;
    d_yuv_rgb_u8_bytes_ = 0;
  }
  delete static_cast<NppStreamContext*>(npp_ctx_);
  npp_ctx_ = nullptr;
}

// Per-frame view of the input, unified across the Tensor and ImageT paths. NV12 has one Y
// plane pointer + one CbCr plane pointer; I420 additionally uses cbcr_plane_v for the Cr plane.
// For RGB24 / YUYV, `y_ptr` is the interleaved base and the chroma pointers are null.
struct InputView {
  const void* y_ptr{};
  const void* cbcr_plane{};    // NV12 UV (interleaved); I420 Cb plane.
  const void* cbcr_plane_v{};  // I420 Cr plane; unused otherwise.
  int y_step_bytes{};          // Row stride of Y plane / interleaved buffer.
  int cbcr_step_bytes{};       // NV12 CbCr row stride / I420 Cb row stride.
  int cbcr_step_bytes_v{};     // I420 Cr row stride (may differ from Cb under Argus/SIPL).
  int64_t src_width{};
  int64_t src_height{};
  InputPixFmt pix_fmt{InputPixFmt::kRgb24};
  YuvMatrix yuv_matrix{YuvMatrix::kBt601};
  YuvRange yuv_range{YuvRange::kFull};
};

// Overflow-checked helpers used for TensorInputPlane bounds validation.
[[nodiscard]] std::optional<std::uint64_t> checked_add(std::uint64_t lhs,
                                                       std::uint64_t rhs) noexcept {
  std::uint64_t out{};
  if (__builtin_add_overflow(lhs, rhs, &out)) {
    return std::nullopt;
  }
  return out;
}

[[nodiscard]] std::optional<std::uint64_t> checked_mul(std::uint64_t lhs,
                                                       std::uint64_t rhs) noexcept {
  std::uint64_t out{};
  if (__builtin_mul_overflow(lhs, rhs, &out)) {
    return std::nullopt;
  }
  return out;
}

// Confirm one plane fits inside `total_bytes` and has row stride >= `min_row_bytes`.
[[nodiscard]] std::optional<std::string> validate_plane_fits(const TensorInputPlane& plane,
                                                             std::uint64_t plane_rows,
                                                             std::uint64_t min_row_bytes,
                                                             std::uint64_t total_bytes,
                                                             const char* plane_name) {
  if (plane.row_stride_bytes < min_row_bytes) {
    return std::string("TensorPreprocessorOp: ") + plane_name + " row_stride_bytes (" +
           std::to_string(plane.row_stride_bytes) + ") smaller than plane's row-width-in-bytes (" +
           std::to_string(min_row_bytes) + ")";
  }
  const auto rows_bytes = checked_mul(plane.row_stride_bytes, plane_rows);
  if (!rows_bytes) {
    return std::string("TensorPreprocessorOp: ") + plane_name + " coverage overflows uint64_t";
  }
  const auto end = checked_add(plane.offset_bytes, *rows_bytes);
  if (!end) {
    return std::string("TensorPreprocessorOp: ") + plane_name +
           " offset + coverage overflows uint64_t";
  }
  if (*end > total_bytes) {
    return std::string("TensorPreprocessorOp: ") + plane_name +
           " extends past input Tensor byte span (end=" + std::to_string(*end) +
           ", tensor bytes=" + std::to_string(total_bytes) + ")";
  }
  return std::nullopt;
}

// NPP argument types are `int`. Reject values that would overflow when we cast into them.
[[nodiscard]] bool fits_in_npp_int(std::int64_t value) noexcept {
  return value > 0 && value <= std::numeric_limits<int>::max();
}
[[nodiscard]] bool fits_in_npp_int(std::uint64_t value) noexcept {
  return value <= static_cast<std::uint64_t>(std::numeric_limits<int>::max());
}

// Parse a bare Tensor per `pix_fmt`. RGB24/YUYV expect rank-3 [H, W, C] or rank-4 [1, H, W, C];
// NV12/I420 expect rank-1/2 byte buffers of length 1.5 * H * W with H/W taken from
// output_width/output_height x aspect... which we don't have. Instead we require the tensor
// to be either rank-2 [1.5*H, W] or rank-1 [1.5*H*W] with H and W recoverable from the ImageT
// path; for the Tensor path we reject NV12/I420 without an ImageT to describe geometry (this
// aligns with the reviewer's design: Tensor-path is for pre-decoded formats).
// Parse the flat bare-Tensor NV12/I420 path used by Argus/SIPL producers that emit a rank-1
// byte buffer whose plane layout is carried by the operator config (input_planes) rather
// than the tensor shape.
// Keep the packed-plane geometry and stride rejection paths together so offsets are
// validated against the same source tensor before a view is returned.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<InputView, Error> parse_flat_yuv_tensor(
    const Tensor& tensor, const TensorPreprocessorOpParams& config, InputPixFmt pix_fmt,
    YuvMatrix yuv_matrix, YuvRange yuv_range) {
  if (config.input_width <= 0 || config.input_height <= 0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "TensorPreprocessorOp: flat bare-Tensor NV12/I420 requires input_width and "
              "input_height to be set"});
  }
  if ((config.input_width % 2) != 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 "TensorPreprocessorOp: NV12/I420 requires an even input_width"});
  }
  if ((config.input_height % 2) != 0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "TensorPreprocessorOp: NV12/I420 requires an even input_height (4:2:0 chroma)"});
  }
  const std::size_t required_planes = pix_fmt == InputPixFmt::kNv12 ? 2U : 3U;
  if (config.input_planes.size() != required_planes) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string("TensorPreprocessorOp: ") +
            (pix_fmt == InputPixFmt::kNv12 ? "NV12 expects exactly 2 input_planes (Y, CbCr)"
                                           : "I420 expects exactly 3 input_planes (Y, Cb, Cr)")});
  }
  const std::vector<int64_t>& in_shape = tensor.shape();
  if (in_shape.size() != 1U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "TensorPreprocessorOp: flat NV12/I420 input Tensor must be rank-1 (a byte buffer); "
              "configure geometry via input_width/input_height/input_planes"});
  }
  const auto& in_strides = tensor.strides();
  if (in_strides.size() == 1U && in_strides[0] != 1) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "TensorPreprocessorOp: flat NV12/I420 input Tensor must have unit byte stride"});
  }
  const std::uint64_t width_u = static_cast<std::uint64_t>(config.input_width);
  const std::uint64_t height_u = static_cast<std::uint64_t>(config.input_height);
  const std::uint64_t tensor_bytes = static_cast<std::uint64_t>(in_shape[0]);

  const auto& planes = config.input_planes;
  if (auto err = validate_plane_fits(planes[0], height_u, width_u, tensor_bytes, "Y plane")) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, std::move(*err)});
  }
  if (pix_fmt == InputPixFmt::kNv12) {
    // CbCr interleaved plane: H/2 rows, W bytes/row (W/2 chroma samples * 2 bytes each).
    if (auto err =
            validate_plane_fits(planes[1], height_u / 2U, width_u, tensor_bytes, "CbCr plane")) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, std::move(*err)});
    }
    // NPP's nppiNV12ToRGB_..._P2C3R takes a single rSrcStep for both Y and CbCr planes.
    if (planes[0].row_stride_bytes != planes[1].row_stride_bytes) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported,
                "TensorPreprocessorOp: NV12 flat path requires equal Y and CbCr row strides "
                "(NPP nppiNV12ToRGB expects one rSrcStep). Rewrap the frame with equal pitches "
                "upstream or use input_kind=image with per-plane stride metadata."});
    }
  } else {
    // I420 Cb / Cr planes: H/2 rows, W/2 bytes/row each.
    if (auto err =
            validate_plane_fits(planes[1], height_u / 2U, width_u / 2U, tensor_bytes, "Cb plane")) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, std::move(*err)});
    }
    if (auto err =
            validate_plane_fits(planes[2], height_u / 2U, width_u / 2U, tensor_bytes, "Cr plane")) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, std::move(*err)});
    }
  }
  if (!fits_in_npp_int(planes[0].row_stride_bytes) ||
      !fits_in_npp_int(planes[1].row_stride_bytes) ||
      (pix_fmt == InputPixFmt::kI420 && !fits_in_npp_int(planes[2].row_stride_bytes)) ||
      !fits_in_npp_int(width_u) || !fits_in_npp_int(height_u)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 "TensorPreprocessorOp: input geometry / row strides exceed "
                                 "NPP's int-sized argument range"});
  }
  const auto* base = static_cast<const std::uint8_t*>(tensor.data());
  if (base == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kFailure, "TensorPreprocessorOp: input Tensor has null device pointer"});
  }
  InputView view{
      .y_ptr = base + planes[0].offset_bytes,
      .cbcr_plane = base + planes[1].offset_bytes,
      .y_step_bytes = static_cast<int>(planes[0].row_stride_bytes),
      .cbcr_step_bytes = static_cast<int>(planes[1].row_stride_bytes),
      .src_width = static_cast<int64_t>(config.input_width),
      .src_height = static_cast<int64_t>(config.input_height),
      .pix_fmt = pix_fmt,
      .yuv_matrix = yuv_matrix,
      .yuv_range = yuv_range,
  };
  if (pix_fmt == InputPixFmt::kI420) {
    view.cbcr_plane_v = base + planes[2].offset_bytes;
    view.cbcr_step_bytes_v = static_cast<int>(planes[2].row_stride_bytes);
  }
  return view;
}

// Keep the supported rank/layout cases and their stride checks together so the accepted
// device view can be audited against one contiguous validation sequence.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<InputView, Error> parse_tensor_input(
    const Tensor& tensor, const TensorPreprocessorOpParams& config, InputPixFmt pix_fmt,
    YuvMatrix yuv_matrix, YuvRange yuv_range) {
  if (tensor.dtype().code != kDLUInt || tensor.dtype().bits != 8U || tensor.dtype().lanes != 1U) {
    return make_unexpected(
        Error{ErrorCode::kFailure, "TensorPreprocessorOp: expected a scalar uint8 input Tensor"});
  }
  // Flat NV12/I420 branch: rank-1 byte buffer + input_planes carrying plane geometry.
  if (pix_fmt == InputPixFmt::kNv12 || pix_fmt == InputPixFmt::kI420) {
    if (config.input_planes.empty()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                "TensorPreprocessorOp: NV12/I420 with input_kind=tensor requires input_planes plus "
                "input_width/input_height; use input_kind=image for ImageT-carried plane_layouts"});
    }
    return parse_flat_yuv_tensor(tensor, config, pix_fmt, yuv_matrix, yuv_range);
  }

  const int64_t required_channels = pix_fmt == InputPixFmt::kYuyv ? 2 : 3;
  const std::vector<int64_t>& in_shape = tensor.shape();
  int64_t src_height = 0;
  int64_t src_width = 0;
  int64_t src_channels = required_channels;
  bool rank2_yuyv = false;

  if (pix_fmt == InputPixFmt::kYuyv && in_shape.size() == 2U) {
    // V4L2-style rank-2 [H, W_in_bytes]. Each YUYV group covers 4 bytes (2 pixels).
    const int64_t height = in_shape[0];
    const int64_t row_bytes = in_shape[1];
    if (!fits_in_npp_int(height) || !fits_in_npp_int(row_bytes) || (row_bytes % 4) != 0) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument,
                "TensorPreprocessorOp: rank-2 YUYV input must have positive [H, row_bytes] with "
                "row_bytes divisible by 4 (each macropixel = 2 pixels × 2 bytes)"});
    }
    src_height = height;
    src_width = row_bytes / 2;  // Two bytes per pixel in YUYV.
    rank2_yuyv = true;
  } else if (in_shape.size() == 3U) {
    src_height = in_shape[0];
    src_width = in_shape[1];
    src_channels = in_shape[2];
  } else if (in_shape.size() == 4U && in_shape[0] == 1) {
    src_height = in_shape[1];
    src_width = in_shape[2];
    src_channels = in_shape[3];
  } else {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        "TensorPreprocessorOp: expected rank-3 [H, W, C], rank-4 [1, H, W, C], or (YUYV only) "
        "rank-2 [H, W_in_bytes] input Tensor"});
  }
  if (!rank2_yuyv && src_channels != required_channels) {
    return make_unexpected(Error{ErrorCode::kFailure,
                                 std::string("TensorPreprocessorOp: expected a ") +
                                     std::to_string(required_channels) +
                                     "-channel input Tensor for the configured input_pix_fmt"});
  }
  if (pix_fmt == InputPixFmt::kYuyv && (src_width % 2) != 0) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              "TensorPreprocessorOp: input_pix_fmt='yuyv' requires an even pixel width"});
  }
  if (!fits_in_npp_int(src_width) || !fits_in_npp_int(src_height)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 "TensorPreprocessorOp: input dimensions exceed NPP int range"});
  }
  const void* src_device = tensor.data();
  if (src_device == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kFailure, "TensorPreprocessorOp: input Tensor has null device pointer"});
  }
  const auto& in_strides = tensor.strides();
  const int64_t bytes_per_pixel = pix_fmt == InputPixFmt::kYuyv ? 2 : 3;
  const int64_t expected_row_stride = src_width * bytes_per_pixel;
  if (in_strides.size() == in_shape.size() && !in_strides.empty()) {
    const std::size_t h_axis = !rank2_yuyv && in_shape.size() == 4U ? 1U : 0U;
    if (in_strides[h_axis] != expected_row_stride) {
      return make_unexpected(Error{
          ErrorCode::kFailure,
          "TensorPreprocessorOp: input Tensor must be row-contiguous (row stride " +
              std::to_string(in_strides[h_axis]) + " != expected " +
              std::to_string(expected_row_stride) + "); strided/padded inputs are not supported"});
    }
  }
  return InputView{
      .y_ptr = src_device,
      .y_step_bytes = static_cast<int>(expected_row_stride),
      .src_width = src_width,
      .src_height = src_height,
      .pix_fmt = pix_fmt,
      .yuv_matrix = yuv_matrix,
      .yuv_range = yuv_range,
  };
}

// Parse an ImageT: pick pix_fmt/matrix from the metadata, honor plane_layouts for
// multi-plane encodings; fall back to config overrides for CUSTOM/UNSPECIFIED fields.
[[nodiscard]] expected<InputView, Error> parse_image_input(const schema::ImageT& image,
                                                           InputPixFmt fallback_pix_fmt,
                                                           YuvMatrix fallback_matrix,
                                                           YuvRange yuv_range) {
  if (!image.data) {
    return make_unexpected(
        Error{ErrorCode::kFailure, "TensorPreprocessorOp: ImageT.data (Tensor) is null"});
  }
  const Tensor& tensor = *image.data;
  if (tensor.dtype().code != kDLUInt || tensor.dtype().bits != 8U) {
    return make_unexpected(
        Error{ErrorCode::kFailure, "TensorPreprocessorOp: ImageT.data must be a uint8 Tensor"});
  }
  const void* src_device = tensor.data();
  if (src_device == nullptr) {
    return make_unexpected(Error{
        ErrorCode::kFailure, "TensorPreprocessorOp: ImageT.data Tensor has null device pointer"});
  }
  const InputPixFmt pix_fmt = pix_fmt_from_encoding(image.encoding).value_or(fallback_pix_fmt);
  YuvMatrix matrix = fallback_matrix;
  if (auto derived_matrix = matrix_from_color_space(image.color_space)) {
    matrix = *derived_matrix;
  }
  const int64_t src_width = image.width;
  const int64_t src_height = image.height;
  if (src_width <= 0 || src_height <= 0) {
    return make_unexpected(Error{ErrorCode::kFailure,
                                 "TensorPreprocessorOp: ImageT.width / .height must be positive"});
  }
  if (is_yuv(pix_fmt) && (src_width % 2) != 0) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              "TensorPreprocessorOp: YUV inputs require an even width; ImageT.width was odd"});
  }

  InputView view{
      .src_width = src_width,
      .src_height = src_height,
      .pix_fmt = pix_fmt,
      .yuv_matrix = matrix,
      .yuv_range = yuv_range,
  };
  const auto* base = static_cast<const std::uint8_t*>(src_device);
  const auto& planes = image.plane_layouts;
  switch (pix_fmt) {
    case InputPixFmt::kRgb24:
      view.y_ptr = base;
      view.y_step_bytes = planes.empty() ? static_cast<int>(src_width * 3)
                                         : static_cast<int>(planes[0].row_stride_bytes());
      break;
    case InputPixFmt::kYuyv:
      view.y_ptr = base;
      view.y_step_bytes = planes.empty() ? static_cast<int>(src_width * 2)
                                         : static_cast<int>(planes[0].row_stride_bytes());
      break;
    case InputPixFmt::kNv12:
      if (planes.size() < 2U) {
        return make_unexpected(Error{
            ErrorCode::kFailure,
            "TensorPreprocessorOp: NV12 ImageT must declare exactly 2 plane_layouts (Y, CbCr)"});
      }
      view.y_ptr = base + planes[0].offset_bytes();
      view.y_step_bytes = static_cast<int>(planes[0].row_stride_bytes());
      view.cbcr_plane = base + planes[1].offset_bytes();
      view.cbcr_step_bytes = static_cast<int>(planes[1].row_stride_bytes());
      break;
    case InputPixFmt::kI420:
      if (planes.size() < 3U) {
        return make_unexpected(
            Error{ErrorCode::kFailure,
                  "TensorPreprocessorOp: I420 ImageT must declare exactly 3 plane_layouts (Y, Cb, "
                  "Cr)"});
      }
      view.y_ptr = base + planes[0].offset_bytes();
      view.y_step_bytes = static_cast<int>(planes[0].row_stride_bytes());
      view.cbcr_plane = base + planes[1].offset_bytes();
      view.cbcr_plane_v = base + planes[2].offset_bytes();
      view.cbcr_step_bytes = static_cast<int>(planes[1].row_stride_bytes());
      break;
  }
  return view;
}

// Dispatch the YUV -> interleaved RGB conversion into `d_rgb_u8`. The caller sizes and owns
// the destination buffer.
[[nodiscard]] expected<void, Error> convert_yuv_to_rgb(const InputView& view, Npp8u* d_rgb_u8,
                                                       int dst_step_u8,
                                                       const NppStreamContext& npp_ctx) {
  const NppiSize size{static_cast<int>(view.src_width), static_cast<int>(view.src_height)};
  // Range-vs-matrix admissibility. NPP's YUYV and I420 CSCs are BT.601 full-range only;
  // NV12 supports BT.601 default and BT.709 via the _709CSC variant, both nominally
  // full-range. Reject limited-range explicitly rather than silently producing shifted
  // colors, and reject BT.709 for the YUYV/I420 paths where no NPP variant exists.
  if (view.yuv_range == YuvRange::kLimited && view.pix_fmt != InputPixFmt::kRgb24) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "TensorPreprocessorOp: yuv_range='limited' not yet supported for YUV inputs "
              "(NPP dispatch is currently full-range only)"});
  }

  switch (view.pix_fmt) {
    case InputPixFmt::kYuyv: {
      if (view.yuv_matrix != YuvMatrix::kBt601) {
        return make_unexpected(Error{
            ErrorCode::kNotSupported,
            "TensorPreprocessorOp: YUYV path currently supports BT.601 only (BT.709 pending)"});
      }
      const NppStatus status = nppiYCbCr422ToRGB_8u_C2C3R_Ctx(static_cast<const Npp8u*>(view.y_ptr),
                                                              view.y_step_bytes,
                                                              d_rgb_u8,
                                                              dst_step_u8,
                                                              size,
                                                              npp_ctx);
      if (status != NPP_SUCCESS) {
        return make_unexpected(npp_error("nppiYCbCr422ToRGB_8u_C2C3R_Ctx failed", status));
      }
      return {};
    }
    case InputPixFmt::kNv12: {
      std::array<const Npp8u*, 2> planes{static_cast<const Npp8u*>(view.y_ptr),
                                         static_cast<const Npp8u*>(view.cbcr_plane)};
      // NV12 supports both BT.601 (default) and BT.709 (via the _709CSC_ variant).
      const NppStatus status =
          view.yuv_matrix == YuvMatrix::kBt709
              ? nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(
                    planes.data(), view.y_step_bytes, d_rgb_u8, dst_step_u8, size, npp_ctx)
              : nppiNV12ToRGB_8u_P2C3R_Ctx(
                    planes.data(), view.y_step_bytes, d_rgb_u8, dst_step_u8, size, npp_ctx);
      if (status != NPP_SUCCESS) {
        return make_unexpected(npp_error(view.yuv_matrix == YuvMatrix::kBt709
                                             ? "nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx failed"
                                             : "nppiNV12ToRGB_8u_P2C3R_Ctx failed",
                                         status));
      }
      return {};
    }
    case InputPixFmt::kI420: {
      if (view.yuv_matrix != YuvMatrix::kBt601) {
        return make_unexpected(Error{
            ErrorCode::kNotSupported,
            "TensorPreprocessorOp: I420 path currently supports BT.601 only (BT.709 pending)"});
      }
      std::array<const Npp8u*, 3> planes{static_cast<const Npp8u*>(view.y_ptr),
                                         static_cast<const Npp8u*>(view.cbcr_plane),
                                         static_cast<const Npp8u*>(view.cbcr_plane_v)};
      // Cr may have a different row stride than Cb under Argus/SIPL plane layouts. When the
      // Cr stride is unset (e.g. the ImageT path, where the schema only publishes one stride
      // for the chroma pair) fall back to the Cb stride so the NPP call still receives a
      // valid three-entry array.
      const int cr_step =
          view.cbcr_step_bytes_v != 0 ? view.cbcr_step_bytes_v : view.cbcr_step_bytes;
      std::array<int, 3> steps{view.y_step_bytes, view.cbcr_step_bytes, cr_step};
      const NppStatus status = nppiYCbCr420ToRGB_8u_P3C3R_Ctx(
          planes.data(), steps.data(), d_rgb_u8, dst_step_u8, size, npp_ctx);
      if (status != NPP_SUCCESS) {
        return make_unexpected(npp_error("nppiYCbCr420ToRGB_8u_P3C3R_Ctx failed", status));
      }
      return {};
    }
    case InputPixFmt::kRgb24:
      return make_unexpected(
          Error{ErrorCode::kFailure, "convert_yuv_to_rgb called with pix_fmt=rgb24"});
  }
  return make_unexpected(Error{ErrorCode::kFailure, "unknown InputPixFmt in convert_yuv_to_rgb"});
}

// Keep receive ownership, NPP submission, output-loan commit, and publication in order;
// each format branch uses the same prepared scratch and admitted CUDA stream.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> TensorPreprocessorOp::compute(ExecutionContext& context) {
  const TensorPreprocessorOpParams& config = params();
  const InputKind input_kind = parse_input_kind(config.input_kind);
  const InputPixFmt configured_pix_fmt = parse_input_pix_fmt(config.input_pix_fmt);
  const YuvMatrix configured_matrix = parse_yuv_matrix(config.yuv_matrix);
  const YuvRange configured_range = parse_yuv_range(config.yuv_range);

  SampleMetadata in_metadata{};
  InputView view{};
  if (input_kind == InputKind::kTensor) {
    auto sample = input_.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    in_metadata = sample->metadata;
    auto parsed = parse_tensor_input(
        sample->data, config, configured_pix_fmt, configured_matrix, configured_range);
    if (!parsed) {
      return make_unexpected(std::move(parsed).error());
    }
    view = *parsed;
  } else {
    auto sample = image_input_.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    in_metadata = sample->metadata;
    auto parsed =
        parse_image_input(sample->data, configured_pix_fmt, configured_matrix, configured_range);
    if (!parsed) {
      return make_unexpected(std::move(parsed).error());
    }
    view = *parsed;
  }
  const int64_t src_width = view.src_width;
  const int64_t src_height = view.src_height;

  // Bind this frame's stream. Every NPP `_Ctx` call below dispatches asynchronously onto it.
  auto* const stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "TensorPreprocessorOp: no CUDA stream available for device work"});
  }
  auto& npp_ctx = *static_cast<NppStreamContext*>(npp_ctx_);
  npp_ctx.hStream = stream;
  unsigned int stream_flags = 0U;
  if (cudaStreamGetFlags(stream, &stream_flags) == cudaSuccess) {
    npp_ctx.nStreamFlags = stream_flags;
  }

  const bool out_is_float = is_float_out(config.output_pix_fmt);
  const bool nchw = is_nchw(config.output_layout);
  const NppiSize src_size{static_cast<int>(src_width), static_cast<int>(src_height)};
  const NppiSize dst_size{config.output_width, config.output_height};
  const NppiRect src_roi{0, 0, src_size.width, src_size.height};
  const NppiRect dst_roi{0, 0, dst_size.width, dst_size.height};
  const int dst_step_u8 = dst_size.width * 3;
  const int dst_step_f32 = dst_size.width * 3 * static_cast<int>(sizeof(float));
  const int plane_step_u8 = dst_size.width;
  const int plane_step_f32 = dst_size.width * static_cast<int>(sizeof(float));

  // 0. YUV -> RGB. Lazily allocated (grown, never shrunk) so the operator handles a resolution
  //    bump without churning cudaMalloc on the hot path. For RGB24 input we bypass the CSC and
  //    resize directly from the input pointer.
  const void* resize_src = view.y_ptr;
  int resize_src_step_u8 = view.y_step_bytes;
  if (is_yuv(view.pix_fmt)) {
    const std::size_t rgb_bytes =
        static_cast<std::size_t>(src_size.width) * static_cast<std::size_t>(src_size.height) * 3U;
    if (d_yuv_rgb_u8_bytes_ < rgb_bytes) {
      if (d_yuv_rgb_u8_ != nullptr) {
        cudaFree(d_yuv_rgb_u8_);
        d_yuv_rgb_u8_ = nullptr;
        d_yuv_rgb_u8_bytes_ = 0;
      }
      const cudaError_t alloc_err = cudaMalloc(&d_yuv_rgb_u8_, rgb_bytes);
      if (alloc_err != cudaSuccess) {
        return make_unexpected(cuda_error("cudaMalloc(d_yuv_rgb_u8_) failed", alloc_err));
      }
      d_yuv_rgb_u8_bytes_ = rgb_bytes;
    }
    const int rgb_step_u8 = src_size.width * 3;
    auto converted =
        convert_yuv_to_rgb(view, static_cast<Npp8u*>(d_yuv_rgb_u8_), rgb_step_u8, npp_ctx);
    if (!converted) {
      return make_unexpected(std::move(converted).error());
    }
    resize_src = d_yuv_rgb_u8_;
    resize_src_step_u8 = rgb_step_u8;
  }

  // 1. Resize (u8 interleaved, bilinear).
  {
    const NppStatus status = nppiResize_8u_C3R_Ctx(static_cast<const Npp8u*>(resize_src),
                                                   resize_src_step_u8,
                                                   src_size,
                                                   src_roi,
                                                   static_cast<Npp8u*>(d_resized_u8_),
                                                   dst_step_u8,
                                                   dst_size,
                                                   dst_roi,
                                                   NPPI_INTER_LINEAR,
                                                   npp_ctx);
    if (status != NPP_SUCCESS) {
      return make_unexpected(npp_error("nppiResize_8u_C3R_Ctx failed", status));
    }
  }

  // Allocate the destination Tensor now; all further NPP calls write into it (via a planar
  // C3->P3 split when NCHW is requested) or via a device-device memcpy when NHWC is requested.
  const DLDataType out_dtype =
      out_is_float ? DLDataType{kDLFloat, 32U, 1U} : DLDataType{kDLUInt, 8U, 1U};
  auto loan = output_.allocate_tensor(
      TensorLoanRequest{.shape = TensorShapeView{output_shape_}, .dtype = out_dtype});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }
  if (writer->byte_size() != output_frame_bytes_) {
    return make_unexpected(Error{ErrorCode::kFailure,
                                 "TensorPreprocessorOp: allocated Tensor byte span does not "
                                 "match computed output byte span"});
  }
  void* const dst_device = writer->data();

  if (!out_is_float) {
    // uint8 output: either write straight to dst as interleaved (NHWC), or split into 3 planes
    // (NCHW). Both are single NPP calls; no dtype conversion, no normalization.
    if (nchw) {
      std::array<Npp8u*, 3> planes{
          static_cast<Npp8u*>(dst_device) + 0U * static_cast<std::size_t>(dst_size.width) *
                                                static_cast<std::size_t>(dst_size.height),
          static_cast<Npp8u*>(dst_device) + 1U * static_cast<std::size_t>(dst_size.width) *
                                                static_cast<std::size_t>(dst_size.height),
          static_cast<Npp8u*>(dst_device) + 2U * static_cast<std::size_t>(dst_size.width) *
                                                static_cast<std::size_t>(dst_size.height)};
      const NppStatus status = nppiCopy_8u_C3P3R_Ctx(static_cast<const Npp8u*>(d_resized_u8_),
                                                     dst_step_u8,
                                                     planes.data(),
                                                     plane_step_u8,
                                                     dst_size,
                                                     npp_ctx);
      if (status != NPP_SUCCESS) {
        return make_unexpected(npp_error("nppiCopy_8u_C3P3R_Ctx failed", status));
      }
    } else {
      if (const cudaError_t status = cudaMemcpyAsync(
              dst_device, d_resized_u8_, d_resized_u8_bytes_, cudaMemcpyDeviceToDevice, stream);
          status != cudaSuccess) {
        return make_unexpected(cuda_error("cudaMemcpyAsync (u8 NHWC) failed", status));
      }
    }
  } else {
    // float32 output: dtype-convert into d_resized_f32_, then apply (x/255 - mean) / std in
    // three fused NPP arithmetic passes, then either write straight to dst (NHWC) or split
    // into 3 planes (NCHW).
    {
      const NppStatus status = nppiConvert_8u32f_C3R_Ctx(static_cast<const Npp8u*>(d_resized_u8_),
                                                         dst_step_u8,
                                                         static_cast<Npp32f*>(d_resized_f32_),
                                                         dst_step_f32,
                                                         dst_size,
                                                         npp_ctx);
      if (status != NPP_SUCCESS) {
        return make_unexpected(npp_error("nppiConvert_8u32f_C3R_Ctx failed", status));
      }
    }
    {
      constexpr std::array<Npp32f, 3> kInv255{1.0F / 255.0F, 1.0F / 255.0F, 1.0F / 255.0F};
      const NppStatus status = nppiMulC_32f_C3IR_Ctx(
          kInv255.data(), static_cast<Npp32f*>(d_resized_f32_), dst_step_f32, dst_size, npp_ctx);
      if (status != NPP_SUCCESS) {
        return make_unexpected(npp_error("nppiMulC_32f_C3IR_Ctx failed", status));
      }
    }
    {
      const std::array<Npp32f, 3> mean{
          config.normalize_mean[0], config.normalize_mean[1], config.normalize_mean[2]};
      const NppStatus status = nppiSubC_32f_C3IR_Ctx(
          mean.data(), static_cast<Npp32f*>(d_resized_f32_), dst_step_f32, dst_size, npp_ctx);
      if (status != NPP_SUCCESS) {
        return make_unexpected(npp_error("nppiSubC_32f_C3IR_Ctx failed", status));
      }
    }
    {
      const std::array<Npp32f, 3> stdv{
          config.normalize_std[0], config.normalize_std[1], config.normalize_std[2]};
      const NppStatus status = nppiDivC_32f_C3IR_Ctx(
          stdv.data(), static_cast<Npp32f*>(d_resized_f32_), dst_step_f32, dst_size, npp_ctx);
      if (status != NPP_SUCCESS) {
        return make_unexpected(npp_error("nppiDivC_32f_C3IR_Ctx failed", status));
      }
    }

    if (nchw) {
      std::array<Npp32f*, 3> planes{
          static_cast<Npp32f*>(dst_device) + 0U * static_cast<std::size_t>(dst_size.width) *
                                                 static_cast<std::size_t>(dst_size.height),
          static_cast<Npp32f*>(dst_device) + 1U * static_cast<std::size_t>(dst_size.width) *
                                                 static_cast<std::size_t>(dst_size.height),
          static_cast<Npp32f*>(dst_device) + 2U * static_cast<std::size_t>(dst_size.width) *
                                                 static_cast<std::size_t>(dst_size.height)};
      const NppStatus status = nppiCopy_32f_C3P3R_Ctx(static_cast<const Npp32f*>(d_resized_f32_),
                                                      dst_step_f32,
                                                      planes.data(),
                                                      plane_step_f32,
                                                      dst_size,
                                                      npp_ctx);
      if (status != NPP_SUCCESS) {
        return make_unexpected(npp_error("nppiCopy_32f_C3P3R_Ctx failed", status));
      }
    } else {
      if (const cudaError_t status = cudaMemcpyAsync(
              dst_device, d_resized_f32_, d_resized_f32_bytes_, cudaMemcpyDeviceToDevice, stream);
          status != cudaSuccess) {
        return make_unexpected(cuda_error("cudaMemcpyAsync (f32 NHWC) failed", status));
      }
    }
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }

  EmitOptions emit_options;
  if (in_metadata.source_clock_id.valid()) {
    emit_options.capture_time = TimePoint{.timestamp_ns = in_metadata.capture_timestamp_ns,
                                          .clock_id = in_metadata.source_clock_id};
  }
  if (in_metadata.frame_id != 0U) {
    emit_options.frame_id = in_metadata.frame_id;
  }
  auto emitted = output_.emit(std::move(*loan), std::move(emit_options));
  if (!emitted) {
    return make_unexpected(std::move(emitted).error());
  }

  return {};
}

}  // namespace holoscan::ops
