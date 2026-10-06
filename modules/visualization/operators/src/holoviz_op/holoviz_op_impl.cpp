// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HolovizOp::Impl: the start()/compute()/stop() frame loop over the RenderBackend /
// PresentBackend seam. The public class (holoviz_op.cpp) forwards its lifecycle overrides here.

#include "holoviz_op/holoviz_op_impl.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <cuda_runtime_api.h>
#include <fmt/format.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>

#include "common/holoviz_conversions.hpp"
#include "holoviz_op/backends/cuda_present_backend.hpp"
#include "holoviz_op/backends/display_surface.hpp"
#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/backends/holoviz_render_backend.hpp"
#include "holoviz_op/backends/image_layout.hpp"
#include "holoviz_op/backends/window_surface.hpp"
#include "holoviz_op/error_translation.hpp"
#include "holoviz_op/exclusive_display.hpp"
#include "holoviz_op/holoviz_op_window_desc.hpp"
#include "holoviz_op/image_input.hpp"
#include "holoviz_op/text_input.hpp"

namespace holoscan::visualization {

// ---- ImageT wire-semantics validation ------------------------------------------

namespace {

/// The single-plane, non-YUV ImageFormat a wire schema::ImageEncoding corresponds to, or
/// nullopt for an encoding with no such mapping (multi-plane/YUV encodings, encodings this
/// backend does not support, or CUSTOM). Kept local to this file rather than
/// common/holoviz_conversions.* so the sensor_io dependency (Decision 2 in the ImageT port
/// alignment design doc) stays confined to the operator target, not the HoloViz backends object
/// library.
std::optional<ImageFormat> image_format_for_encoding(schema::ImageEncoding encoding) noexcept {
  switch (encoding) {
    case schema::ImageEncoding_MONO8:
      return ImageFormat::kR8Unorm;
    case schema::ImageEncoding_RGB8:
      return ImageFormat::kR8G8B8Unorm;
    case schema::ImageEncoding_RGBA8:
      return ImageFormat::kR8G8B8A8Unorm;
    case schema::ImageEncoding_BGRA8:
      return ImageFormat::kB8G8R8A8Unorm;
    case schema::ImageEncoding_MONO16:
      return ImageFormat::kR16Unorm;
    case schema::ImageEncoding_NV12:
      return ImageFormat::kNV12;
    case schema::ImageEncoding_YUYV:
      return ImageFormat::kYUYV422;
    case schema::ImageEncoding_RGBA16:
      return ImageFormat::kR16G16B16A16Unorm;
    default:
      // No ImageFormat for this encoding: BGR8, RGB16 (no three-channel 16-bit format yet),
      // the planar RGB and *32F float encodings, UYVY, I420, P010, RGBA1010102, the Bayer
      // variants, or CUSTOM.
      return std::nullopt;
  }
}

const char* reason_name(ImageRejectReason reason) noexcept {
  switch (reason) {
    case ImageRejectReason::kAccepted:
      return "accepted";
    case ImageRejectReason::kNoData:
      return "no data Tensor";
    case ImageRejectReason::kEncodingMismatch:
      return "encoding mismatch";
    case ImageRejectReason::kUnmappedEncoding:
      return "unsupported encoding";
    case ImageRejectReason::kColorSpaceIncompatible:
      return "color space incompatible";
    case ImageRejectReason::kPlaneLayoutsOnCustom:
      return "plane layouts on a CUSTOM frame";
    case ImageRejectReason::kPlaneLayoutsInvalid:
      return "invalid plane layouts";
    case ImageRejectReason::kGeometryMismatch:
      return "geometry mismatch";
    case ImageRejectReason::kUnsupportedMemory:
      return "unsupported memory";
  }
  return "unknown";
}

const char* color_space_name(ColorSpace color_space) noexcept {
  switch (color_space) {
    case ColorSpace::kPassThrough:
      return "kPassThrough";
    case ColorSpace::kSRGBNonlinear:
      return "kSRGBNonlinear";
    case ColorSpace::kBT709Linear:
      return "kBT709Linear";
  }
  return "unknown";
}

const char* yuv_model_name(YUVModelConversion model) noexcept {
  switch (model) {
    case YUVModelConversion::kBT601:
      return "kBT601";
    case YUVModelConversion::kBT709:
      return "kBT709";
    case YUVModelConversion::kBT2020:
      return "kBT2020";
  }
  return "unknown";
}

/// The YUV matrix a wire color-space label names, or nullopt for labels that describe none
/// (UNSPECIFIED, SRGB, and the HDR transfer functions).
std::optional<YUVModelConversion> yuv_model_for_color_space(schema::ColorSpace space) noexcept {
  switch (space) {
    case schema::ColorSpace_REC601:
      return YUVModelConversion::kBT601;
    case schema::ColorSpace_REC709:
      return YUVModelConversion::kBT709;
    case schema::ColorSpace_REC2020:
      return YUVModelConversion::kBT2020;
    default:
      return std::nullopt;
  }
}

/// Check a wire frame's own claims (encoding, color space, plane records) against the layer it
/// arrived on. ImageLayerDesc stays the authoritative contract (Decision 5 in the ImageT port
/// alignment design doc): the wire value is validated against it per frame, never adopted.
///
/// Encoding: must map to the declared format. ImageEncoding_CUSTOM is exempt -- the schema defines
/// it as "states no organization", for producers that do not classify their pixel layout, so it
/// defers to the tensor dtype/shape check -- but a CUSTOM frame that declares plane records is
/// asserting a multi-plane organization the single-plane path never consults and is rejected
/// rather than read as the wrong plane.
///
/// Color space: UNSPECIFIED is the absence of a claim, not a wrong one, and is accepted into any
/// layer -- the declared color_space is the authority, and producers that never stamp a color
/// space (raw sensor frames, capture drivers) keep rendering. A labelled value must agree with the
/// layer: SRGB requires kSRGBNonlinear (an sRGB frame sampled by a kPassThrough layer is a
/// silently wrong image, exactly what rejection exists to prevent); REC601/REC709/REC2020 on a
/// YUV format require the matching yuv_model, two independent statements of the same matrix.
/// REC709 names primaries and a matrix, not the linear-transfer claim kBT709Linear makes, so no
/// wire label maps to kBT709Linear and only UNSPECIFIED frames reach such a layer. The HDR
/// transfer functions (REC2020_PQ, REC2020_HLG) have no counterpart here and are rejected.
ImageRejectReason check_image_semantics(const schema::ImageT& image,
                                        const ImageLayerDesc& desc) noexcept {
  if (image.encoding == schema::ImageEncoding_CUSTOM) {
    if (!image.plane_layouts.empty()) {
      return ImageRejectReason::kPlaneLayoutsOnCustom;
    }
  } else {
    const auto encoded_format = image_format_for_encoding(image.encoding);
    if (!encoded_format) {
      return ImageRejectReason::kUnmappedEncoding;
    }
    if (*encoded_format != desc.format) {
      return ImageRejectReason::kEncodingMismatch;
    }
  }
  switch (image.color_space) {
    case schema::ColorSpace_UNSPECIFIED:
      return ImageRejectReason::kAccepted;
    case schema::ColorSpace_SRGB:
      return desc.color_space == ColorSpace::kSRGBNonlinear
                 ? ImageRejectReason::kAccepted
                 : ImageRejectReason::kColorSpaceIncompatible;
    default: {
      const auto model = yuv_model_for_color_space(image.color_space);
      if (!model || !backends::is_yuv(desc.format) || !desc.yuv_model ||
          *desc.yuv_model != *model) {
        return ImageRejectReason::kColorSpaceIncompatible;
      }
      return ImageRejectReason::kAccepted;
    }
  }
}

/// Bytes the Tensor can address from its first logical element, honoring byte strides: a padded
/// backing is bounded by what it can address, not by the product of its extents.
std::size_t tensor_byte_span(const holoscan::Tensor& tensor) noexcept {
  // Span accessors, not shape()/strides(): those materialize vectors on every admitted frame.
  const std::span<const std::int64_t> shape = tensor.shape_span();
  const std::span<const std::int64_t> strides = tensor.dlpack_strides_span();  // in elements
  if (shape.empty()) {
    return 0;
  }
  if (strides.size() != shape.size()) {
    return static_cast<std::size_t>(std::max<std::int64_t>(tensor.nbytes(), 0));
  }
  constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
  const std::size_t item = static_cast<std::size_t>(tensor.itemsize());
  std::size_t span = item;
  for (std::size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] <= 0) {
      return 0;
    }
    const auto steps = static_cast<std::size_t>(shape[axis] - 1);
    const auto stride = static_cast<std::size_t>(std::max<std::int64_t>(strides[axis], 0));
    // Overflow means the Tensor cannot describe a real extent; 0 makes every plane fail
    // containment instead of letting a wrapped value pass it.
    if (stride != 0 && steps > kMax / stride) {
      return 0;
    }
    const std::size_t reach = steps * stride;
    if (item != 0 && reach > kMax / item) {
      return 0;
    }
    const std::size_t bytes = reach * item;
    if (bytes > kMax - span) {
      return 0;
    }
    span += bytes;
  }
  return span;
}

/// Where one source plane starts in the backing Tensor and how far apart its rows are.
struct SourcePlane {
  std::size_t offset_bytes = 0;
  std::size_t row_stride_bytes = 0;
};

/// An admitted frame's geometry: the tight layout the backend/staging path wants, plus where
/// each plane actually lives in the producer's Tensor.
struct ImageSource {
  backends::ImageLayout layout;
  std::array<SourcePlane, 2> planes{};
};

/// Why resolve_image_source() could not place a frame, with the specifics for the log line.
struct ImageRejection {
  ImageRejectReason reason;
  std::string detail;
};

/// Resolve the frame's plane placement, or say why it cannot be rendered.
///
/// Single-plane formats keep the Tensor as the layout authority (image_input_layout()) and
/// require ImageT::width/height to agree with it: the schema lets the backing Tensor be larger
/// than the declared image, but this path has no way to address a sub-region, so a disagreement
/// would render pixels outside the declared geometry.
///
/// NV12 is the reverse. The schema makes ImageT::width/height the geometry authority and
/// plane_layouts the plane authority (required, one record per plane), and the backing Tensor is
/// just the byte store the offsets index into -- a flat byte buffer is as valid as {H + H/2, W},
/// and the chroma plane need not follow the luma plane. Core already checks the records against
/// the encoding and the Tensor on the wire path (validate_image_facts); the synthetic test fixture
/// bypasses that path and the check is cheap, so it is repeated here before any byte is read.
std::optional<ImageRejection> resolve_image_source(const schema::ImageT& image,
                                                   const holoscan::Tensor& tensor,
                                                   const ImageLayerDesc& desc, ImageSource& out) {
  if (desc.format != ImageFormat::kNV12) {
    const auto input = image_input_layout(tensor, desc);
    if (!input) {
      return ImageRejection{ImageRejectReason::kGeometryMismatch,
                            fmt::format("invalid {} tensor layout: {}",
                                        common::image_format_name(desc.format),
                                        input.error().message)};
    }
    if (image.width != static_cast<std::int32_t>(input->width) ||
        image.height != static_cast<std::int32_t>(input->height)) {
      return ImageRejection{ImageRejectReason::kGeometryMismatch,
                            fmt::format("ImageT declares {}x{} but the backing Tensor is {}x{}",
                                        image.width,
                                        image.height,
                                        input->width,
                                        input->height)};
    }
    out.layout = input->layout;
    out.planes[0] = SourcePlane{0U, input->pitch};
    return std::nullopt;
  }

  if (image.width <= 0 || image.height <= 0 ||
      static_cast<std::uint32_t>(image.width) > desc.max_width ||
      static_cast<std::uint32_t>(image.height) > desc.max_height) {
    return ImageRejection{ImageRejectReason::kGeometryMismatch,
                          fmt::format("ImageT declares {}x{}; the layer accepts up to {}x{}",
                                      image.width,
                                      image.height,
                                      desc.max_width,
                                      desc.max_height)};
  }
  const auto layout = backends::image_layout(desc.format,
                                             static_cast<std::uint32_t>(image.width),
                                             static_cast<std::uint32_t>(image.height));
  if (!layout) {
    return ImageRejection{
        ImageRejectReason::kGeometryMismatch,
        fmt::format(
            "NV12 requires even dimensions; ImageT declares {}x{}", image.width, image.height)};
  }
  if (image.plane_layouts.size() != layout->plane_count) {
    return ImageRejection{
        ImageRejectReason::kPlaneLayoutsInvalid,
        fmt::format("NV12 requires {} plane_layouts entries (Y, UV); the frame carries {}",
                    layout->plane_count,
                    image.plane_layouts.size())};
  }
  const DLDataType dtype = tensor.dtype();
  if (dtype.code != kDLUInt || dtype.bits != 8U || dtype.lanes != 1U) {
    return ImageRejection{ImageRejectReason::kGeometryMismatch,
                          "NV12 requires a uint8 backing Tensor"};
  }
  if (tensor.data() == nullptr) {
    return ImageRejection{ImageRejectReason::kUnsupportedMemory,
                          "the backing Tensor has a null data pointer"};
  }
  const std::size_t extent = tensor_byte_span(tensor);
  out.layout = *layout;
  for (std::uint32_t plane = 0; plane < layout->plane_count; ++plane) {
    const schema::ImagePlaneLayout& record = image.plane_layouts[plane];
    const std::size_t stride = record.row_stride_bytes();
    const auto span =
        backends::image_plane_span(layout->row_bytes[plane], layout->rows[plane], stride);
    const bool contained = span && *span <= extent && record.offset_bytes() <= extent - *span &&
                           record.size_bytes() >= *span;
    if (!contained) {
      return ImageRejection{
          ImageRejectReason::kPlaneLayoutsInvalid,
          fmt::format("plane {} (offset {}, row stride {}, size {}) does not cover {} rows of "
                      "{} bytes inside the {}-byte Tensor",
                      plane,
                      record.offset_bytes(),
                      stride,
                      record.size_bytes(),
                      layout->rows[plane],
                      layout->row_bytes[plane],
                      extent)};
    }
    out.planes[plane] = SourcePlane{static_cast<std::size_t>(record.offset_bytes()), stride};
  }
  return std::nullopt;
}

}  // namespace

// ---- RenderList construction -------------------------------------------------

void HolovizOp::Impl::register_stream_ordered_image_release(FrameDataStorage& storage,
                                                            CUstream stream) {
  if (stream == nullptr) {
    return;
  }
  for (holoscan::Tensor& tensor : storage.asynchronous_image_tensors) {
    if (tensor.data() != nullptr && tensor.set_deallocation_stream(stream)) {
      // record() has already queued the final source read on stream. Dropping this handle lets
      // capable allocators recycle the backing as soon as that work completes.
      tensor = holoscan::Tensor{};
    }
  }
}

// build_render_list returns expected<> so hard failures (copy/upload) propagate to compute()
// rather than silently recording/presenting a partial frame. Soft failures (bad shape, unsupported
// format) still log and skip the affected layer — they are not frame-level failures.
holoscan::expected<HolovizOp::Impl::BuildResult, holoscan::Error>
HolovizOp::Impl::build_render_list(holoscan::ExecutionContext& context, std::uint32_t image_index) {
  const LayerStack& layers = op.layer_stack_;
  BuildResult result;
  std::optional<holoscan::Error> fatal_error;
  const auto cuda_stream = context.cuda_stream().get();

  // Image-layer diagnostics are edge-triggered: a rejection is logged when its reason first
  // appears and an acceptance when the layer recovers. Under a LayerTrigger every activation
  // still pays a full render/present for a dropped frame, so a producer that never matches its
  // layer would otherwise emit one error per frame, forever.
  const auto reject = [this](std::size_t layer, ImageRejectReason reason, std::string_view detail) {
    if (last_image_reject_reason_[layer] == reason) {
      return;
    }
    last_image_reject_reason_[layer] = reason;
    HOLOSCAN_LOG_ERROR(
        "HolovizOp::build_render_list: image layer {} rejected ({}): {}; dropping its samples "
        "(logged once per reason).",
        layer,
        reason_name(reason),
        detail);
  };
  const auto accepted = [this](std::size_t layer) {
    if (last_image_reject_reason_[layer] == ImageRejectReason::kAccepted) {
      return;
    }
    last_image_reject_reason_[layer] = ImageRejectReason::kAccepted;
    HOLOSCAN_LOG_INFO("HolovizOp::build_render_list: image layer {} admits frames again.", layer);
  };

  // Reserve before the loop so no emplace_back/push_back on the outer vectors can
  // reallocate and invalidate the std::span pointers stored in geometry_frames entries.
  const std::size_t n = layers.layers().size();
  result.storage.geometry_batches.reserve(n);
  result.storage.geometry_frames.reserve(n);
  result.storage.text_runs.reserve(n);
  result.storage.text_frames.reserve(n);
  result.storage.image_frames.reserve(n);

  for (std::size_t i = 0; i < layers.layers().size(); ++i) {
    const std::size_t slot = op.pool_slot_for_layer_[i];
    const bool is_image_layer = std::holds_alternative<ImageLayerDesc>(layers.layers()[i].desc);
    const bool missing = is_image_layer ? image_layer_input(slot).is_missing()
                                        : tensor_layer_input(slot).is_missing();

    std::visit(
        [&](auto& desc) {
          using T = std::decay_t<decltype(desc)>;

          if constexpr (std::is_same_v<T, ImageLayerDesc>) {
            // contract() gives non-trigger image ports LatestWithin + ReuseLast, both bounded by
            // max_input_age_: with a producer slower than the trigger (e.g. 30 fps decode under a
            // 60 fps LayerTrigger or ClockTrigger), most activations legitimately have no *new*
            // sample, and Runtime reuses the last admitted one instead of reporting missing — so
            // is_missing() here means the bound has genuinely elapsed, not just "nothing new this
            // activation". Only then is the layer dropped from the frame; every other activation
            // (fresh or Runtime-reused) re-derives and re-uploads from receive_data() below, so
            // the operator never holds its own unbounded, independently-aged copy. The
            // LayerTrigger layer is pinned by OnEach and is never missing.
            if (missing) {
              return;
            }
            auto& cache = image_layer_cache(i);
            auto payload = image_layer_input(slot).receive_data();
            if (!payload) {
              return;  // receive error on a non-missing port is unexpected but non-fatal
            }
            // Check the frame's own claims (encoding, color space, plane records) against the
            // declared layer before touching the Tensor: a mismatch means the producer and this
            // layer disagree about what is being rendered.
            schema::ImageT& image = *payload;
            if (!image.data) {
              reject(i, ImageRejectReason::kNoData, "the ImageT carries no data Tensor");
              return;
            }
            if (const auto reason = check_image_semantics(image, desc);
                reason != ImageRejectReason::kAccepted) {
              if (last_image_reject_reason_[i] == reason) {
                return;  // already logged; skip building the detail string every activation
              }
              reject(i,
                     reason,
                     fmt::format("frame declares encoding {}, color_space {} and {} plane "
                                 "record(s); the layer declares {} with color_space {}{}",
                                 schema::EnumNameImageEncoding(image.encoding),
                                 schema::EnumNameColorSpace(image.color_space),
                                 image.plane_layouts.size(),
                                 common::image_format_name(desc.format),
                                 color_space_name(desc.color_space),
                                 desc.yuv_model ? fmt::format(" and yuv_model {}",
                                                              yuv_model_name(*desc.yuv_model))
                                                : std::string{}));
              return;
            }
            if (image.color_space == schema::ColorSpace_UNSPECIFIED &&
                desc.color_space != ColorSpace::kPassThrough &&
                !unspecified_color_space_noted_[i]) {
              // Not a rejection -- the declared color space is the authority. Said once so a
              // wrong-looking image can be traced to a producer that does not stamp its frames.
              unspecified_color_space_noted_[i] = true;
              HOLOSCAN_LOG_INFO(
                  "HolovizOp::build_render_list: image layer {} frames declare no color space; "
                  "rendering per the layer's configured {}.",
                  i,
                  color_space_name(desc.color_space));
            }
            holoscan::Tensor& tensor = *image.data;
            ImageSource source;
            if (const auto rejection = resolve_image_source(image, tensor, desc, source)) {
              reject(i, rejection->reason, rejection->detail);
              return;
            }
            const backends::ImageLayout& layout = source.layout;
            const auto new_w = static_cast<std::uint32_t>(image.width);
            const auto new_h = static_cast<std::uint32_t>(image.height);
            // Branch on the tensor's actual device residency. setup() leaves image ports
            // unconstrained so both CUDA-resident and host-resident producers are accepted
            // without forcing an unnecessary round-trip copy.
            //
            // CUDA-resident path: use the device pointer directly. Runtime has already
            // queued the producer's readiness event as a cudaStreamWaitEvent on the
            // activation stream (context.cuda_stream()), so passing that stream in
            // ImageFrameData.stream is sufficient to serialize the backend's GPU reads
            // after the producer's writes.
            //
            // Host/pinned path: wait for producer readiness, then pack active rows into
            // the host upload cache. Runtime alone controls the age of reused samples.
            const DLDeviceType dev_type = tensor.device().device_type;
            const bool is_cuda_resident = (dev_type == kDLCUDA || dev_type == kDLCUDAManaged);

            if (is_cuda_resident) {
              // Direct device-pointer path: no D2H copy, no staging buffer. Each plane keeps
              // the producer's own placement and row stride.
              // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
              const auto dev_ptr =
                  static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(tensor.data()));
              if (dev_ptr == 0U) {
                reject(i,
                       ImageRejectReason::kUnsupportedMemory,
                       "the CUDA tensor has a null data pointer");
                return;
              }
              // Retain the backing until record() queues its asynchronous copy. compute() then
              // registers stream-ordered release when the backing supports it, or transfers the
              // handle to the bounded frame-slot fallback.
              result.storage.asynchronous_image_tensors[i] = tensor;
              backends::ImageFrameData frame{
                  .width = new_w,
                  .height = new_h,
                  .plane_count = layout.plane_count,
                  .stream = cuda_stream,
              };
              for (std::uint32_t plane = 0; plane < layout.plane_count; ++plane) {
                frame.planes[plane] = {dev_ptr + source.planes[plane].offset_bytes,
                                       source.planes[plane].row_stride_bytes};
              }
              result.storage.image_frames.push_back(frame);
              cache.bytes.clear();
              cache.width = new_w;
              cache.height = new_h;
              cache.pitch = source.planes[0].row_stride_bytes;
              cache.last_frame = result.storage.image_frames.back();
              result.list.layers.push_back(
                  {static_cast<uint32_t>(i), &result.storage.image_frames.back()});
              accepted(i);
            } else {
              if (dev_type != kDLCPU && dev_type != kDLCUDAHost) {
                reject(i,
                       ImageRejectReason::kUnsupportedMemory,
                       fmt::format("unsupported image memory device type {}",
                                   static_cast<int>(dev_type)));
                return;
              }
              // copy_to_host() only accepts contiguous tensors. Wait explicitly, then pack each
              // plane's active rows tightly into the staging layout without touching unowned
              // padding at the end of any row.
              auto ready = tensor.synchronize_for_host();
              if (!ready) {
                fatal_error = ready.error();
                return;
              }
              std::vector<std::byte> next_bytes(layout.bytes);
              const auto* base = static_cast<const std::byte*>(tensor.data());
              for (std::uint32_t plane = 0; plane < layout.plane_count; ++plane) {
                const std::byte* src = base + source.planes[plane].offset_bytes;
                std::byte* dst = next_bytes.data() + layout.offsets[plane];
                for (std::size_t row = 0; row < layout.rows[plane]; ++row) {
                  std::memcpy(dst + row * layout.row_bytes[plane],
                              src + row * source.planes[plane].row_stride_bytes,
                              layout.row_bytes[plane]);
                }
              }
              cache.bytes = std::move(next_bytes);
              cache.width = new_w;
              cache.height = new_h;
              cache.pitch = layout.row_bytes[0];

              if (cache.bytes.empty()) {
                return;
              }

              // Upload to the per-swapchain-image staging buffer. Per-image_index buffers
              // are required: with multiple in-flight swapchain images the GPU may still be
              // sampling a previous frame's staging buffer on a concurrent activation.
              const CUdeviceptr ptr = image_staging_devptr(i, image_index);
              cudaError_t upload_err =
                  cudaMemcpyAsync(  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
                      reinterpret_cast<void*>(ptr),
                      cache.bytes.data(),
                      cache.bytes.size(),
                      cudaMemcpyHostToDevice,
                      cuda_stream);
              if (upload_err != cudaSuccess) {
                // Hard failure: propagate so compute() can recover the open frame.
                fatal_error = holoscan::Error{
                    holoscan::ErrorCode::kFailure,
                    "HolovizOp::build_render_list: image layer " + std::to_string(i) +
                        " H2D staging upload failed: " + cudaGetErrorString(upload_err)};
                return;
              }
              backends::ImageFrameData frame{
                  .width = new_w,
                  .height = new_h,
                  .plane_count = layout.plane_count,
                  .stream = cuda_stream,
              };
              for (std::uint32_t plane = 0; plane < layout.plane_count; ++plane) {
                frame.planes[plane] = {ptr + layout.offsets[plane], layout.row_bytes[plane]};
              }
              result.storage.image_frames.push_back(frame);
              cache.last_frame = result.storage.image_frames.back();
              result.list.layers.push_back(
                  {static_cast<uint32_t>(i), &result.storage.image_frames.back()});
              accepted(i);
            }

          } else if constexpr (std::is_same_v<T, GeometryLayerDesc>) {
            if (missing) {
              return;
            }
            auto payload = tensor_layer_input(slot).receive_data();
            if (!payload) {
              return;  // receive error on a non-missing port is unexpected but non-fatal
            }
            holoscan::Tensor& tensor = *payload;
            if (desc.topologies.empty()) {
              return;  // mis-declared layer: no topology to render
            }
            if (desc.topologies.size() > 1U) {
              // Only topologies[0] is rendered; additional entries are silently ignored.
              HOLOSCAN_LOG_WARN(
                  "HolovizOp::build_render_list: geometry layer {} declares {} topologies; "
                  "only topologies[0] is rendered. Multi-topology batches are not yet supported.",
                  i,
                  desc.topologies.size());
            }
            auto synchronized = tensor.synchronize_for_host();
            if (!synchronized) {
              fatal_error = holoscan::Error{
                  synchronized.error().code,
                  "HolovizOp::build_render_list: geometry layer " + std::to_string(i) +
                      " failed to synchronize Tensor readiness for host access: " +
                      synchronized.error().message};
              return;
            }
            // Host CPU float32 tensor: {ptr, n_floats} span consumed during record(). setup()
            // constrains this port to rank-1 float32, but data_as<float>() can still return null
            // for a producer that bypasses that contract, so this is checked here rather than
            // trusted — a null ptr must never reach the span constructed below.
            const auto* ptr = tensor.data_as<float>();
            if (ptr == nullptr) {
              HOLOSCAN_LOG_ERROR(
                  "HolovizOp::build_render_list: geometry layer {} received a non-float32 "
                  "or misaligned Tensor; dropping this activation's sample.",
                  i);
              return;
            }
            const auto nbytes = static_cast<std::size_t>(tensor.nbytes());
            if (nbytes % sizeof(float) != 0U) {
              HOLOSCAN_LOG_ERROR(
                  "HolovizOp::build_render_list: geometry layer {} byte size {} is not a "
                  "whole number of float32 elements.",
                  i,
                  nbytes);
              return;
            }
            const auto n_float = nbytes / sizeof(float);
            // Floats per primitive varies by topology:
            //   kRectangleList: x0,y0,x1,y1            → 4 floats/prim
            //   kOvalList:      cx,cy,rx,ry             → 4 floats/prim
            //   kLineList:      x0,y0,x1,y1            → 4 floats/prim
            //   kCrossList:     cx,cy,size              → 3 floats/prim
            //   kLineStrip:     (x,y) per point, N pts → N-1 segments
            std::size_t prim_count = 0;
            bool exact = false;
            switch (desc.topologies[0]) {
              case GeometryTopology::kLineStrip:
                exact = n_float >= 4U && n_float % 2U == 0U;
                prim_count = exact ? n_float / 2U - 1U : 0U;
                break;
              case GeometryTopology::kCrossList:
                exact = n_float % 3U == 0U;
                prim_count = n_float / 3U;
                break;
              default:
                exact = n_float % 4U == 0U;
                prim_count = n_float / 4U;
                break;
            }
            if (!exact) {
              HOLOSCAN_LOG_ERROR(
                  "HolovizOp::build_render_list: geometry layer {} float count {} does not "
                  "evenly divide into primitives for this topology.",
                  i,
                  n_float);
              return;
            }
            if (prim_count > desc.max_primitives) {
              HOLOSCAN_LOG_ERROR(
                  "HolovizOp::build_render_list: geometry layer {} primitive count {} "
                  "exceeds declared max_primitives {}.",
                  i,
                  prim_count,
                  desc.max_primitives);
              return;
            }

            // GeometryFrameData borrows the Tensor pointer through render_->record(). Keep an
            // owning Tensor handle in BuildResult for that complete interval.
            result.storage.record_tensors[i] = tensor;

            result.storage.geometry_batches.emplace_back(
                std::vector<backends::GeometryBatch>{backends::GeometryBatch{
                    .topology = desc.topologies[0],
                    .coords = {ptr, n_float},
                    .primitive_count = static_cast<uint32_t>(prim_count),
                    .color = {1.0F, 1.0F, 1.0F, 1.0F},
                    .line_width = desc.default_line_width,
                }});
            result.storage.geometry_frames.push_back(backends::GeometryFrameData{
                .batches = result.storage.geometry_batches.back(),
            });
            result.list.layers.push_back(
                {static_cast<uint32_t>(i), &result.storage.geometry_frames.back()});

          } else if constexpr (std::is_same_v<T, TextLayerDesc>) {
            if (missing) {
              return;
            }
            auto payload = tensor_layer_input(slot).receive_data();
            if (!payload) {
              return;  // receive error on a non-missing port is unexpected but non-fatal
            }
            holoscan::Tensor& tensor = *payload;
            auto synchronized = tensor.synchronize_for_host();
            if (!synchronized) {
              fatal_error =
                  holoscan::Error{synchronized.error().code,
                                  "HolovizOp::build_render_list: text layer " + std::to_string(i) +
                                      " failed to synchronize Tensor readiness for host access: " +
                                      synchronized.error().message};
              return;
            }
            // Host uint8 {N, 16 + L} records (see TextLayerDesc): decoded into TextRuns whose
            // string_views borrow the Tensor's bytes. setup() constrains dtype, rank and
            // residency; row width, packing, header values and the glyph budget are per-sample
            // properties checked here, and a bad sample drops only this layer's activation.
            std::vector<backends::TextRun> runs;
            const auto decoded = decode_text_records(tensor, desc, runs);
            if (!decoded) {
              HOLOSCAN_LOG_ERROR(
                  "HolovizOp::build_render_list: text layer {} rejected its sample ({}); "
                  "dropping this activation's sample.",
                  i,
                  decoded.error().message);
              return;
            }
            if (runs.empty()) {
              return;  // nothing to draw this frame (N == 0, or every row blank)
            }

            // TextFrameData borrows the Tensor bytes through render_->record(). Keep an owning
            // Tensor handle in BuildResult for that complete interval.
            result.storage.record_tensors[i] = tensor;

            result.storage.text_runs.push_back(std::move(runs));
            result.storage.text_frames.push_back(backends::TextFrameData{
                .runs = result.storage.text_runs.back(),
            });
            result.list.layers.push_back(
                {static_cast<uint32_t>(i), &result.storage.text_frames.back()});

          } else if constexpr (std::is_same_v<T, LUTLayerDesc>) {
            // LUT frame path is not yet implemented.
            (void)desc;
          }
        },
        layers.layers()[i].desc);
  }
  if (fatal_error) {
    return holoscan::make_unexpected(std::move(*fatal_error));
  }
  return result;
}

// ---- kImageCountChanged handling ---------------------------------------------

bool HolovizOp::Impl::adopt_swapchain_info_change(const backends::SwapchainInfo& info) noexcept {
  if (info.image_count > kMaxSwapchainImages) {
    HOLOSCAN_LOG_ERROR(
        "HolovizOp: swapchain recreation grew the image count to {}, exceeding "
        "kMaxSwapchainImages ({}); the render slot and staging rings cannot cover it.",
        info.image_count,
        kMaxSwapchainImages);
    return false;
  }
  swapchain_info_ = info;
  return true;
}

// ---- start() ----------------------------------------------------------------

void HolovizOp::Impl::start() {
  if (terminal_failed_) {
    throw holoscan::RuntimeError{
        holoscan::ErrorCode::kFailure,
        "HolovizOp: stop() left GPU resources live after a quiescence failure; restart is not "
        "supported."};
  }

  // Step 1: sink-specific front end — the presentable surface (window or display) or, for
  // the CUDA sink, no surface at all — plus the device and present backend for it.
  // Everything from Step 3 on is sink-agnostic.
  backends::Extent2D preferred_extent{};
  holoviz::PresentMode preferred_mode = holoviz::PresentMode::kAuto;
  bool framebuffer_srgb = false;

  const auto record_cuda_device = [this] {
    // Record which CUDA device device_ actually landed on: the device factories pick the first
    // capable adapter with no Holoscan-side device preference, so on a multi-GPU host this is
    // not guaranteed to be the CUDA device the Runtime later admits for this operator's
    // activation stream. compute() rejects a mismatched stream instead of silently mixing a
    // staging pointer, stream, and HoloViz device from different CUDA contexts.
    auto cuda_dev = device_.cuda_device();
    if (!cuda_dev) {
      throw_holoviz_error(cuda_dev.error(), "Device::cuda_device");
    }
    cuda_device_ordinal_ = static_cast<int>(*cuda_dev);
  };

  const auto create_device_for_surface_handle = [this] {
    auto r = holoviz::create_device_for_surface(
        {}, *std::static_pointer_cast<holoviz::Surface>(surface_handle_.impl));
    if (!r) {
      throw_holoviz_error(r.error(), "create_device_for_surface");
    }
    device_ = std::move(*r);
  };

  const auto create_upload_stream_and_renderer = [this] {
    auto context = device_.cuda_context();
    if (!context) {
      throw_holoviz_error(context.error(), "Device::cuda_context");
    }
    auto stream = create_cuda_stream(*context);
    if (!stream) {
      throw holoscan::RuntimeError{stream.error()};
    }
    image_upload_stream_ = *stream;

    auto r = backends::create_holoviz_render_backend(
        device_, {.stream = image_upload_stream_, .debug_name = "HolovizOp image upload"});
    if (!r) {
      if (auto destroyed = destroy_cuda_stream(*context, image_upload_stream_); !destroyed) {
        HOLOSCAN_LOG_ERROR(destroyed.error().message);
      }
      image_upload_stream_ = nullptr;
      throw_holoviz_error(r.error(), "create_holoviz_render_backend");
    }
    render_ = std::move(*r);
  };

  // Undo create_upload_stream_and_renderer() when a later step of start() fails: start() throws,
  // stop() never runs, and the render backend only borrows image_upload_stream_, so the raw
  // handle would leak. Renderer first (it is the stream's user), then the stream.
  const auto destroy_upload_stream_and_renderer = [this] {
    if (render_) {
      if (auto destroyed = render_->destroy(std::chrono::seconds{5}); !destroyed) {
        HOLOSCAN_LOG_ERROR(translate_error(destroyed.error(), "start: render_->destroy").message);
      }
      render_.reset();
    }
    if (image_upload_stream_ != nullptr) {
      auto context = device_.cuda_context();
      if (!context) {
        HOLOSCAN_LOG_ERROR(translate_error(context.error(), "start: Device::cuda_context").message);
      } else if (auto destroyed = destroy_cuda_stream(*context, image_upload_stream_); !destroyed) {
        HOLOSCAN_LOG_ERROR(destroyed.error().message);
      }
      image_upload_stream_ = nullptr;
    }
  };

  std::visit(
      [&](const auto& sink) {
        using T = std::decay_t<decltype(sink)>;
        if constexpr (std::is_same_v<T, WindowSinkDesc>) {
#if !defined(_WIN32)
          // The NVIDIA X11 driver paces FIFO presents on the primary output's vblank unless
          // __GL_SYNC_DISPLAY_DEVICE names the target monitor. The ICD reads it once, when the
          // process-wide Vulkan instance is first created, and setenv() races getenv() on other
          // scheduler threads, so the operator never writes it: the application exports it
          // before any Vulkan use. Only hint when it is missing for a named fullscreen monitor.
          if (sink.fullscreen && !sink.display_name.empty()) {
            constexpr const char* kSyncDisplayEnv = "__GL_SYNC_DISPLAY_DEVICE";
            if (std::getenv(kSyncDisplayEnv) == nullptr) {
              HOLOSCAN_LOG_INFO(
                  "HolovizOp: {} is unset; if '{}' is not the primary output, export {}={} "
                  "before launching so the NVIDIA X11 driver paces presents on that monitor",
                  kSyncDisplayEnv,
                  sink.display_name,
                  kSyncDisplayEnv,
                  sink.display_name);
            }
          }
#endif
          auto r = holoviz::create_window(detail::window_desc_from_sink(sink));
          if (!r) {
            throw_holoviz_error(r.error(), "create_window");
          }
          window_ = std::move(*r);

          auto h = backends::make_window_surface_handle(window_);
          if (!h) {
            throw_holoviz_error(h.error(), "make_window_surface_handle");
          }
          surface_handle_ = std::move(*h);
          create_device_for_surface_handle();
          record_cuda_device();
          create_upload_stream_and_renderer();

          auto p = backends::create_window_surface(device_);
          if (!p) {
            destroy_upload_stream_and_renderer();
            throw_holoviz_error(p.error(), "create_window_surface");
          }
          present_ = std::move(*p);

          // HoloViz sizes a window swapchain from the surface's live extent; this is advisory.
          auto extent_result = window_.extent();
          preferred_extent = extent_result
                                 ? backends::Extent2D{extent_result->width, extent_result->height}
                                 : backends::Extent2D{sink.width, sink.height};
          preferred_mode = common::to_holoviz_present_mode(sink.present_mode);
          framebuffer_srgb = sink.framebuffer_srgb;

        } else if constexpr (std::is_same_v<T, ExclusiveDisplaySinkDesc>) {
          auto display = resolve_exclusive_display(sink.display_name, "HolovizOp");
          if (!display) {
            HOLOSCAN_LOG_ERROR(holoscan::runtime_error_message(display.error()));
            throw holoscan::RuntimeError{display.error()};
          }
          auto h = backends::make_display_surface_handle(
              *display,
              {.mode = {.resolution = {sink.width, sink.height},
                        .refresh_rate_mhz = sink.refresh_hz * 1000U}});
          if (!h) {
            throw_holoviz_error(h.error(), "make_display_surface_handle");
          }
          surface_handle_ = std::move(*h);
          create_device_for_surface_handle();
          record_cuda_device();
          create_upload_stream_and_renderer();

          auto p = backends::create_display_surface(device_);
          if (!p) {
            destroy_upload_stream_and_renderer();
            throw_holoviz_error(p.error(), "create_display_surface");
          }
          present_ = std::move(*p);
          // Display surfaces are fixed-size (the chosen mode); the backend reports the actual
          // extent through swapchain_info().
          preferred_extent = backends::Extent2D{sink.width, sink.height};
          preferred_mode = common::to_holoviz_present_mode(sink.present_mode);
          framebuffer_srgb = sink.framebuffer_srgb;

        } else {
          static_assert(std::is_same_v<T, CudaSinkDesc>);
          // Validate before any GPU resource exists: these are the caller's declaration errors.
          if (sink.width == 0U || sink.height == 0U) {
            throw holoscan::RuntimeError{
                holoscan::ErrorCode::kInvalidArgument,
                "HolovizOp: CudaSinkDesc needs a nonzero width and height."};
          }
          if (!cuda_sink_element(sink.format) || !common::to_holoviz_format(sink.format)) {
            throw holoscan::RuntimeError{
                holoscan::ErrorCode::kInvalidArgument,
                std::string("HolovizOp: CudaSinkDesc needs a packed single-plane ImageFormat with "
                            "a render-target lowering and a wire encoding; got ") +
                    std::string(common::image_format_name(sink.format)) + "."};
          }

          // No surface to pick a device for: take the first adapter backed by a CUDA device.
          // compute() still rejects an activation stream from another device (see there); on a
          // multi-GPU host the deployment's bind_tensor_output_device must name this one.
          auto adapters = holoviz::enumerate_adapters();
          if (!adapters) {
            throw_holoviz_error(adapters.error(), "enumerate_adapters");
          }
          const auto adapter = std::find_if(
              adapters->begin(), adapters->end(), [](const holoviz::Adapter& candidate) {
                return candidate.cuda_device_index >= 0;
              });
          if (adapter == adapters->end()) {
            throw holoscan::RuntimeError{
                holoscan::ErrorCode::kNotFound,
                "HolovizOp: no HoloViz adapter backed by a CUDA device is available for the "
                "CUDA sink."};
          }
          auto d = adapter->create_device({});
          if (!d) {
            throw_holoviz_error(d.error(), "Adapter::create_device");
          }
          device_ = std::move(*d);
          record_cuda_device();
          create_upload_stream_and_renderer();

          auto p = backends::create_cuda_present_backend(device_, {.format = sink.format});
          if (!p) {
            destroy_upload_stream_and_renderer();
            throw_holoviz_error(p.error(), "create_cuda_present_backend");
          }
          cuda_present_ = p->get();
          present_ = std::move(*p);

          // The ring is sized from the declaration; the format alone decides sRGB encoding.
          preferred_extent = backends::Extent2D{sink.width, sink.height};
          preferred_mode = holoviz::PresentMode::kFIFO;
          framebuffer_srgb = false;
        }
      },
      op.sink_);

  // Step 3: swapchain first so we know the actual image_count before allocate(). The seam
  // itself allows allocate → swapchain → build; this operator derives max_frames_in_flight from
  // the real swapchain image count, which requires the swapchain to exist before allocate() is
  // called.
  {
    auto r = present_->create_swapchain(backends::SwapchainDesc{
        .surface = surface_handle_,
        .preferred_extent = preferred_extent,
        .preferred_mode = preferred_mode,
        .image_count = std::nullopt,  // let backend choose
        .framebuffer_srgb = framebuffer_srgb,
    });
    if (!r) {
      throw_holoviz_error(r.error(), "create_swapchain");
    }
  }
  {
    auto r = present_->swapchain_info();
    if (!r) {
      throw_holoviz_error(r.error(), "swapchain_info");
    }
    swapchain_info_ = *r;
    // kMaxSwapchainImages is a hard compile-time upper bound: the render slot ring and the
    // staging ring below are both sized to it once, not to this initial count, so a later
    // kImageCountChanged growth never needs a live ring rebuild — see kMaxSwapchainImages'
    // doc comment. That only holds if the initial count itself is within the bound.
    if (swapchain_info_.image_count > kMaxSwapchainImages) {
      throw holoscan::RuntimeError{
          holoscan::ErrorCode::kNotImplemented,
          "HolovizOp: swapchain image count " + std::to_string(swapchain_info_.image_count) +
              " exceeds kMaxSwapchainImages (" + std::to_string(kMaxSwapchainImages) +
              "); increase the compile-time limit.",
      };
    }
  }

  // Step 4: allocate the render slot ring to kMaxSwapchainImages (not the initial image count —
  // see kMaxSwapchainImages' doc comment), then build pipelines.
  {
    backends::RenderConfig cfg{
        .max_frames_in_flight = kMaxSwapchainImages,
    };
    auto r = render_->allocate(op.layer_stack_, cfg);
    if (!r) {
      throw_holoviz_error(r.error(), "backends::RenderBackend::allocate");
    }
  }
  {
    auto r = render_->build(swapchain_info_.format);
    if (!r) {
      throw_holoviz_error(r.error(), "backends::RenderBackend::build");
    }
  }
  {
    // Pre-register present targets so the renderer can warm per-image framebuffer state before
    // the first frame. Backends without enumerable targets return an empty SyncToken and the
    // register_targets() default implementation is a no-op.
    auto t = present_->targets();
    if (!t) {
      throw_holoviz_error(t.error(), "backends::PresentBackend::targets");
    }
    auto reg = render_->register_targets(*t);
    if (!reg) {
      throw_holoviz_error(reg.error(), "backends::RenderBackend::register_targets");
    }
  }

  // Step 5: device staging buffers for host-resident ImageLayerDesc layers, one full set per
  // swapchain image slot up to kMaxSwapchainImages. See image_staging_devptr()'s doc comment
  // for why per-image buffering is required, and kMaxSwapchainImages' doc comment for why every
  // slot up to the bound is pre-allocated here. Only needed for host-resident tensors (CUDA-
  // resident tensors are handed to the backend as-is via the direct device-pointer path in
  // build_render_list()); CUDA-resident layers' slots are left as zero here.
  //
  // Push device_.cuda_context() so cudaMalloc/cudaFree target the correct device, matching the
  // same scope used by create_cuda_stream() above. Errors are collected rather than thrown while
  // the context is pushed so the pop always runs before the exception propagates.
  {
    auto ctx_r = device_.cuda_context();
    if (!ctx_r) {
      throw holoscan::RuntimeError{
          translate_error(ctx_r.error(), "start: Device::cuda_context (staging alloc)")};
    }
    const CUcontext cuda_ctx = *ctx_r;
    CUcontext prev = nullptr;
    (void)cuCtxGetCurrent(&prev);
    const bool pushed = prev != cuda_ctx;
    if (pushed) {
      if (const CUresult r = cuCtxPushCurrent(cuda_ctx); r != CUDA_SUCCESS) {
        throw holoscan::RuntimeError{
            translate_cuda_error(r, "start: cuCtxPushCurrent (staging alloc)")};
      }
    }

    std::vector<std::array<CUdeviceptr, kMaxLayerPorts>> staging(
        kMaxSwapchainImages, std::array<CUdeviceptr, kMaxLayerPorts>{});
    std::optional<holoscan::RuntimeError> staging_error;

    const auto& layers = op.layer_stack_.layers();
    for (std::uint32_t image_index = 0; image_index < kMaxSwapchainImages && !staging_error;
         ++image_index) {
      for (std::size_t i = 0; i < layers.size() && !staging_error; ++i) {
        const auto* image_desc = std::get_if<ImageLayerDesc>(&layers[i].desc);
        if (image_desc == nullptr) {
          continue;
        }
        const auto layout = backends::image_layout(
            image_desc->format, image_desc->max_width, image_desc->max_height);
        if (!layout) {
          staging_error =
              holoscan::RuntimeError{holoscan::ErrorCode::kInvalidArgument,
                                     "HolovizOp: invalid image staging format or maximum extent"};
          break;
        }
        const std::size_t bytes = layout->bytes;
        const cudaError_t err = cudaMalloc(  // NOLINT(cppcoreguidelines-owning-memory)
            reinterpret_cast<void**>(&staging[image_index][i]),  // NOLINT
            bytes);
        if (err != cudaSuccess) {
          staging_error = holoscan::RuntimeError{
              holoscan::ErrorCode::kFailure,
              std::string("HolovizOp: cudaMalloc for image staging buffer failed: ") +
                  cudaGetErrorString(err)};
        }
      }
    }

    // Roll back any partial allocations while still under the pushed context.
    if (staging_error) {
      for (auto& per_image : staging) {
        for (auto& ptr : per_image) {
          if (ptr != 0U) {
            (void)cudaFree(reinterpret_cast<void*>(ptr));  // NOLINT
            ptr = 0U;
          }
        }
      }
    }

    if (pushed) {
      CUcontext popped = nullptr;
      (void)cuCtxPopCurrent(&popped);
    }

    if (staging_error) {
      throw *staging_error;
    }
    image_staging_devptrs_ = std::move(staging);
  }
}

// ---- compute() --------------------------------------------------------------

holoscan::expected<void, holoscan::Error> HolovizOp::Impl::compute(
    holoscan::ExecutionContext& context) {
  // Under a LayerTrigger this runs once per trigger sample, so events (and the close request
  // below) are serviced at the producer's rate. Under a ClockTrigger it runs at the period only
  // while input admission lets the tick compute: once every image layer's ReuseLast hold has
  // expired with no SafeDefault sibling, the runtime skips compute() (kAbsent -> kNoFire) and
  // this poll with it. compute() is not an independent event pump under either trigger.
  if (window_) {
    holoviz::poll_window_events();
  }

  // Check for cooperative stop from framework first.
  if (context.cancellation().stop_requested()) {
    return {};
  }

  // Reject an activation stream from a CUDA device other than the one device_ (start()) was
  // created on. Staging pointers, the H2D copy, and the backend's CUDA/Vulkan interop handoff
  // are all bound to cuda_device_ordinal_; silently mixing a stream from a different device would
  // produce invalid-resource errors or, worse, wrong-context reads/writes rather than a clear
  // failure here.
  const int activation_device = context.cuda_stream().device().get();
  if (activation_device != cuda_device_ordinal_) {
    const std::string msg = "HolovizOp::compute: activation stream is on CUDA device " +
                            std::to_string(activation_device) + ", but the HoloViz device was " +
                            "created on CUDA device " + std::to_string(cuda_device_ordinal_) +
                            "; cross-device HolovizOp activation is not supported.";
    HOLOSCAN_LOG_ERROR(msg);
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kIncompatible, msg});
  }

  // Window close: signal the session to stop cooperatively (window sinks only).
  if (window_) {
    auto closed = window_.close_requested();
    if (!closed) {
      const holoscan::Error err = translate_error(closed.error(), "compute: close_requested()");
      HOLOSCAN_LOG_ERROR(err.message);
      return holoscan::make_unexpected(err);
    }
    if (*closed) {
      // Normal window closure is a cooperative stop, not a run failure.
      // ExecutionContext::request_stop() signals the session to stop cleanly; returning {} here
      // lets the current activation complete before the session exits instead of returning
      // kInterrupted which RunSession reports as RUNTIME_OPERATOR_CALLBACK_FAILED.
      HOLOSCAN_LOG_INFO("HolovizOp::compute: window close requested — signalling cooperative stop");
      context.request_stop();
      return {};
    }
  }

  // Outputs parked by an earlier failed CUDA-sink frame are released once their copy is proven
  // complete; checked every activation so a transient failure does not pin a pool slot for long.
  release_completed_retained_outputs();

  // Acquire the next swapchain image. Transient failures return success without a frame: the
  // next activation retries, which under a LayerTrigger means this sample's frame is dropped.
  auto acquired = present_->acquire(std::chrono::milliseconds{100});
  if (!acquired) {
    const auto code = acquired.error().code;
    if (code == holoviz::ErrorCode::kBackpressured || code == holoviz::ErrorCode::kOutOfDate) {
      return {};  // transient: retry next activation
    }
    if (code == holoviz::ErrorCode::kImageCountChanged) {
      auto info = present_->swapchain_info();
      if (!info) {
        const holoscan::Error err = translate_error(info.error(), "compute: swapchain_info");
        HOLOSCAN_LOG_ERROR(err.message);
        return holoscan::make_unexpected(err);
      }
      if (!adopt_swapchain_info_change(*info)) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kResourceExhausted,
            "HolovizOp::compute: swapchain image count exceeds kMaxSwapchainImages"});
      }
      return {};  // within kMaxSwapchainImages: rings already cover it, retry next activation
    }
    const holoscan::Error err = translate_error(acquired.error(), "compute: acquire");
    HOLOSCAN_LOG_ERROR(err.message);
    return holoscan::make_unexpected(err);
  }

  // CUDA sink: loan this frame's output Tensor now, before begin_frame() — the resolve target
  // must be registered while the image is acquired, and a pool-exhausted tick is better skipped
  // before the frame path's one bounded wait. Every error exit below abandons the loan (guard
  // first: cancelling under a live guard tombstones the pool slot) before the empty present
  // that releases the image.
  std::optional<holoscan::TensorOutputLoan> output_loan;
  std::optional<holoscan::TensorOutputWriteGuard> output_guard;
  const auto abandon_output = [&] {
    output_guard.reset();
    if (output_loan) {
      (void)std::move(*output_loan).cancel();
      output_loan.reset();
    }
  };
  if (const auto* cuda_sink = std::get_if<CudaSinkDesc>(&op.sink_); cuda_sink != nullptr) {
    auto prepared =
        prepare_cuda_output(*cuda_sink, acquired->image_index, context, output_loan, output_guard);
    if (!prepared) {
      abandon_output();
      (void)present_->present(backends::PresentArgs{.image_index = acquired->image_index,
                                                    .acquire_issuance = acquired->issuance});
      const auto code = prepared.error().code;
      if (code == holoscan::ErrorCode::kResourceExhausted ||
          code == holoscan::ErrorCode::kBackpressured) {
        return {};  // transient: the consumer still holds every pooled frame; skip this tick
      }
      HOLOSCAN_LOG_ERROR(prepared.error().message);
      return holoscan::make_unexpected(prepared.error());
    }
  }

  // Begin the frame slot for this swapchain image.
  auto cmds_result = render_->begin_frame(*acquired, acquired->image_index);
  if (!cmds_result) {
    abandon_output();
    // present_backend.hpp requires an empty present for every image returned by acquire(),
    // including after begin_frame() failure, so the swapchain can reclaim the slot. Omitting
    // this turns transient slot pressure (kBackpressured) into a stuck swapchain.
    (void)present_->present(backends::PresentArgs{.image_index = acquired->image_index,
                                                  .acquire_issuance = acquired->issuance});
    if (cmds_result.error().code == holoviz::ErrorCode::kBackpressured) {
      return {};  // transient: previous slot still in use, retry next activation
    }
    const holoscan::Error err = translate_error(cmds_result.error(), "compute: begin_frame");
    HOLOSCAN_LOG_ERROR(err.message);
    return holoscan::make_unexpected(err);
  }
  backends::FrameCommands cmds = std::move(*cmds_result);

  // A successful begin_frame() proves the previous frame in this slot has finished every
  // asynchronous FrameData read, so its retained Tensor handles can now be released.
  retained_image_tensors_[acquired->image_index] = {};

  // Build the render list from non-missing input ports.
  // Storage outlives record() and submit() via BuildResult lifetime.
  // Hard failures in build_render_list (copy_to_host failure, H2D upload failure) propagate
  // here and are routed through frame recovery rather than silently recording a partial frame.
  auto build_result = build_render_list(context, acquired->image_index);
  if (!build_result) {
    HOLOSCAN_LOG_ERROR(build_result.error().message);
    abandon_output();
    (void)render_->abort_frame(cmds);
    (void)present_->present(backends::PresentArgs{.image_index = acquired->image_index,
                                                  .acquire_issuance = acquired->issuance});
    return holoscan::make_unexpected(build_result.error());
  }
  auto& build = *build_result;

  // Record draw commands.
  auto rec = render_->record(cmds, build.list, backends::CompositeSpec{});
  if (!rec) {
    const holoscan::Error err = translate_error(rec.error(), "compute: record");
    HOLOSCAN_LOG_ERROR(err.message);
    abandon_output();
    auto aborted = render_->abort_frame(cmds);
    if (!aborted) {
      register_stream_ordered_image_release(build.storage, image_upload_stream_);
      // A failed abort proves nothing about partially queued image uploads. Preserve their
      // non-stream-aware backing until a later successful begin_frame() or renderer destruction.
      retained_image_tensors_[acquired->image_index] =
          std::move(build.storage.asynchronous_image_tensors);
      HOLOSCAN_LOG_ERROR(
          "HolovizOp::compute: abort_frame failed after record; retaining direct image Tensor "
          "backing for frame slot {}: {}",
          acquired->image_index,
          translate_error(aborted.error(), "compute: abort_frame after record").message);
    }
    (void)present_->present(backends::PresentArgs{.image_index = acquired->image_index,
                                                  .acquire_issuance = acquired->issuance});
    return holoscan::make_unexpected(err);
  }

  // The HoloViz backend has now queued every direct Tensor read on image_upload_stream_. Native
  // Holoscan/HoloMQ backing can retire at copy completion; only unsupported backing remains in the
  // fallback array for frame-slot retention below.
  register_stream_ordered_image_release(build.storage, image_upload_stream_);

  // Submit to GPU.
  auto done_result = render_->submit(cmds, backends::SubmitContext{acquired->image_index});
  if (!done_result) {
    const holoscan::Error err = translate_error(done_result.error(), "compute: submit");
    HOLOSCAN_LOG_ERROR(err.message);
    abandon_output();
    // abort_frame() after a failed submit(): RenderBackend::abort_frame() is documented
    // as a no-op on already-submitted/aborted cmds (render_backend.hpp line 164).
    // That no-op does not prove whether queued uploads have drained, so retain fallback backing
    // until the frame slot is recycled even when abort_frame() reports success.
    auto aborted = render_->abort_frame(cmds);
    retained_image_tensors_[acquired->image_index] =
        std::move(build.storage.asynchronous_image_tensors);
    if (!aborted) {
      HOLOSCAN_LOG_ERROR(
          "HolovizOp::compute: abort_frame failed after submit; retaining direct image Tensor "
          "backing for frame slot {}: {}",
          acquired->image_index,
          translate_error(aborted.error(), "compute: abort_frame after submit").message);
    }
    (void)present_->present(backends::PresentArgs{.image_index = acquired->image_index,
                                                  .acquire_issuance = acquired->issuance});
    return holoscan::make_unexpected(err);
  }

  // submit() is nonblocking. Any source backing that could not register stream-ordered release
  // must remain alive until this frame slot is successfully recycled.
  retained_image_tensors_[acquired->image_index] =
      std::move(build.storage.asynchronous_image_tensors);

  // Present.
  auto presented = present_->present(backends::PresentArgs{
      .image_index = acquired->image_index,
      .acquire_issuance = acquired->issuance,
      .render_done = *done_result,
      .present_id = ++present_id_,
  });
  if (!presented) {
    // A present() that failed after queuing its copy (an event record) still writes the loaned
    // Tensor; only the backend knows whether the copy was enqueued.
    if (cuda_present_ != nullptr &&
        cuda_present_->resolve_may_be_in_flight(acquired->image_index)) {
      retain_output(acquired->image_index, output_guard, output_loan);
    } else {
      abandon_output();
    }
    const auto code = presented.error().code;
    if (code == holoviz::ErrorCode::kFrameDropped ||
        code == holoviz::ErrorCode::kImageCountChanged) {
      if (code == holoviz::ErrorCode::kImageCountChanged) {
        auto info = present_->swapchain_info();
        if (info && !adopt_swapchain_info_change(*info)) {
          return holoscan::make_unexpected(holoscan::Error{
              holoscan::ErrorCode::kResourceExhausted,
              "HolovizOp::compute: swapchain image count exceeds kMaxSwapchainImages"});
        }
      }
      return {};  // transient; continue on next activation
    }
    const holoscan::Error err = translate_error(presented.error(), "compute: present");
    HOLOSCAN_LOG_ERROR(err.message);
    return holoscan::make_unexpected(err);
  }

  // CUDA sink: the resolve into the loaned Tensor is queued on the backend stream. Join the
  // activation stream to it so the readiness event commit() records there covers the copy, then
  // publish the frame. From here on the copy is in flight, so a failure must park the loan
  // (retain_output) rather than abandon it: cancelling would let the pool recycle the memory
  // under the copy.
  if (cuda_present_ != nullptr) {
    auto ready = cuda_present_->ready_event(acquired->image_index);
    if (!ready) {
      retain_output(acquired->image_index, output_guard, output_loan);
      const holoscan::Error err = translate_error(ready.error(), "compute: ready_event");
      HOLOSCAN_LOG_ERROR(err.message);
      return holoscan::make_unexpected(err);
    }
    if (const CUresult r = cuStreamWaitEvent(context.cuda_stream().get(), *ready, 0);
        r != CUDA_SUCCESS) {
      retain_output(acquired->image_index, output_guard, output_loan);
      const holoscan::Error err = translate_cuda_error(r, "compute: cuStreamWaitEvent");
      HOLOSCAN_LOG_ERROR(err.message);
      return holoscan::make_unexpected(err);
    }
    auto committed = std::move(*output_guard).commit();
    output_guard.reset();
    if (!committed) {
      retain_output(acquired->image_index, output_guard, output_loan);
      HOLOSCAN_LOG_ERROR("HolovizOp::compute: committing the output Tensor failed: {}",
                         committed.error().message);
      return holoscan::make_unexpected(committed.error());
    }
    // The frame's wire semantics travel with it: the sink's declared extent, and the encoding /
    // color space a consumer validates against its own declared format. start() admitted the
    // format, so the wire lookup cannot fail here. emit_tensor() takes the Tensor from the loan;
    // the descriptor's own data field is ignored.
    const auto* cuda_sink = std::get_if<CudaSinkDesc>(&op.sink_);
    const auto wire = cuda_sink_wire(cuda_sink->format);
    schema::ImageT descriptor{};
    descriptor.height = static_cast<std::int32_t>(cuda_sink->height);
    descriptor.width = static_cast<std::int32_t>(cuda_sink->width);
    descriptor.encoding = wire->encoding;
    descriptor.color_space = wire->color_space;
    auto emitted =
        op.output_.emit_tensor(std::move(*output_loan),
                               descriptor,
                               holoscan::EmitOptions{.capture_time = context.activation_time()});
    output_loan.reset();
    if (!emitted) {
      HOLOSCAN_LOG_ERROR("HolovizOp::compute: emitting the output ImageT failed: {}",
                         emitted.error().message);
      return holoscan::make_unexpected(emitted.error());
    }
  }

  return {};
}

std::optional<HolovizOp::Impl::CudaSinkWire> HolovizOp::Impl::cuda_sink_wire(
    ImageFormat format) noexcept {
  using schema::ColorSpace_SRGB;
  using schema::ColorSpace_UNSPECIFIED;
  switch (format) {
    case ImageFormat::kR8Unorm:
      return CudaSinkWire{schema::ImageEncoding_MONO8, ColorSpace_UNSPECIFIED};
    case ImageFormat::kR8G8B8A8Unorm:
      return CudaSinkWire{schema::ImageEncoding_RGBA8, ColorSpace_UNSPECIFIED};
    case ImageFormat::kB8G8R8A8Unorm:
      return CudaSinkWire{schema::ImageEncoding_BGRA8, ColorSpace_UNSPECIFIED};
    case ImageFormat::kR8G8B8A8Srgb:
      return CudaSinkWire{schema::ImageEncoding_RGBA8, ColorSpace_SRGB};
    case ImageFormat::kB8G8R8A8Srgb:
      return CudaSinkWire{schema::ImageEncoding_BGRA8, ColorSpace_SRGB};
    case ImageFormat::kR16Unorm:
      return CudaSinkWire{schema::ImageEncoding_MONO16, ColorSpace_UNSPECIFIED};
    case ImageFormat::kR16G16B16A16Unorm:
      return CudaSinkWire{schema::ImageEncoding_RGBA16, ColorSpace_UNSPECIFIED};
    case ImageFormat::kR8G8B8Unorm:  // no render-target lowering
    case ImageFormat::kNV12:         // multi-planar
    case ImageFormat::kYUYV422:      // subsampled
    case ImageFormat::kUnknown:
      return std::nullopt;
  }
  return std::nullopt;
}

std::optional<HolovizOp::Impl::CudaSinkElement> HolovizOp::Impl::cuda_sink_element(
    ImageFormat format) noexcept {
  constexpr DLDataType kU8{kDLUInt, 8U, 1U};
  constexpr DLDataType kU16{kDLUInt, 16U, 1U};
  switch (format) {
    case ImageFormat::kR8Unorm:
      return CudaSinkElement{kU8, 2U, 1};
    case ImageFormat::kR8G8B8A8Unorm:
    case ImageFormat::kB8G8R8A8Unorm:
    case ImageFormat::kR8G8B8A8Srgb:
    case ImageFormat::kB8G8R8A8Srgb:
      return CudaSinkElement{kU8, 3U, 4};
    case ImageFormat::kR16Unorm:
      return CudaSinkElement{kU16, 2U, 1};
    case ImageFormat::kR16G16B16A16Unorm:
      return CudaSinkElement{kU16, 3U, 4};
    case ImageFormat::kR8G8B8Unorm:
    case ImageFormat::kNV12:
    case ImageFormat::kYUYV422:
    case ImageFormat::kUnknown:
      return std::nullopt;
  }
  return std::nullopt;
}

holoscan::expected<void, holoscan::Error> HolovizOp::Impl::prepare_cuda_output(
    const CudaSinkDesc& sink, std::uint32_t image_index, holoscan::ExecutionContext& context,
    std::optional<holoscan::TensorOutputLoan>& loan,
    std::optional<holoscan::TensorOutputWriteGuard>& guard) {
  // start() admitted the format, so the element lookup cannot fail here. The Tensor is the
  // format's native element in {H, W} or {H, W, C}; the resolve copy below is still byte-based
  // (row pitch = width * bytes_per_pixel), which the element table agrees with.
  const auto element = cuda_sink_element(sink.format);
  const std::uint32_t bpp = common::bytes_per_pixel(sink.format);
  const std::array<std::int64_t, 3> shape{
      static_cast<std::int64_t>(sink.height),
      static_cast<std::int64_t>(sink.width),
      element->channels,
  };
  auto allocated = op.output_.allocate_tensor(holoscan::TensorLoanRequest{
      .shape = holoscan::TensorShapeView{shape.data(), element->rank}, .dtype = element->dtype});
  if (!allocated) {
    return holoscan::make_unexpected(holoscan::Error{
        allocated.error().code,
        "HolovizOp::compute: allocating the output Tensor failed: " + allocated.error().message});
  }
  loan.emplace(std::move(*allocated));
  // The pool requires the write stream to be this operator's runtime compute stream; the GPU
  // copy itself runs on the present backend's stream and is joined to this one before commit().
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(
        holoscan::Error{writer.error().code,
                        "HolovizOp::compute: opening the output Tensor for writing failed (is the "
                        "output port's device bound to this operator's CUDA device?): " +
                            writer.error().message});
  }
  guard.emplace(std::move(*writer));
  const std::size_t row_pitch = static_cast<std::size_t>(sink.width) * bpp;
  if (guard->data() == nullptr || guard->byte_size() < row_pitch * sink.height) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure,
                        "HolovizOp::compute: the loaned output Tensor is smaller than one frame."});
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  const auto dst = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(guard->data()));
  // The writer stream orders the backend's copy after any consumer read of a recycled
  // allocation still queued on the activation stream (same-stream readiness elision retires
  // pool slots logically, without a GPU event).
  auto target = cuda_present_->set_resolve_target(
      image_index,
      backends::CudaResolveTarget{.device_ptr = dst, .row_pitch_bytes = row_pitch},
      context.cuda_stream().get());
  if (!target) {
    return holoscan::make_unexpected(
        translate_error(target.error(), "compute: set_resolve_target"));
  }
  return {};
}

void HolovizOp::Impl::retain_output(std::uint32_t image_index,
                                    std::optional<holoscan::TensorOutputWriteGuard>& guard,
                                    std::optional<holoscan::TensorOutputLoan>& loan) {
  if (!guard && !loan) {
    return;
  }
  retained_outputs_.push_back(std::make_unique<RetainedOutput>(RetainedOutput{
      .image_index = image_index, .guard = std::move(guard), .loan = std::move(loan)}));
  guard.reset();
  loan.reset();
  HOLOSCAN_LOG_WARN(
      "HolovizOp::compute: retaining the output Tensor of frame slot {} until its resolve is "
      "proven complete ({} retained).",
      image_index,
      retained_outputs_.size());
}

void HolovizOp::Impl::release_completed_retained_outputs() {
  if (retained_outputs_.empty() || cuda_present_ == nullptr) {
    return;
  }
  // ready_event() is re-recorded by every later present of the same image, and those resolves
  // are stream-ordered after the retained frame's, so its completion proves the retained copy
  // complete. Destroying the guard/loan here runs the pool's ordinary abandon path, which is
  // now safe: nothing writes the allocation anymore.
  // Query under device_'s context (same push/pop discipline as the staging allocation in
  // start()); the operator thread may have another context, or none, current.
  auto context = device_.cuda_context();
  if (!context) {
    return;
  }
  CUcontext previous = nullptr;
  if (cuCtxGetCurrent(&previous) != CUDA_SUCCESS) {
    return;
  }
  const bool pushed = previous != *context;
  if (pushed && cuCtxPushCurrent(*context) != CUDA_SUCCESS) {
    return;
  }
  std::erase_if(retained_outputs_, [&](const std::unique_ptr<RetainedOutput>& retained) {
    auto ready = cuda_present_->ready_event(retained->image_index);
    return ready && cuEventQuery(*ready) == CUDA_SUCCESS;
  });
  if (pushed) {
    CUcontext popped = nullptr;
    (void)cuCtxPopCurrent(&popped);
  }
}

// ---- stop() -----------------------------------------------------------------

void HolovizOp::Impl::stop() {
  // Tear-down is an ownership transaction, not a fixed sequence of independent steps: the
  // staging buffers and every HoloViz object below are only provably safe to free once the renderer
  // has quiesced (render_->destroy() succeeded), because that is the one call that proves no
  // outstanding GPU read/write against them remains in flight. A bare render_.reset() (the
  // previous approach) does not prove that — it just runs the destructor's best-effort bounded
  // fallback and discards whatever it reports.
  //
  // So: quiesce the renderer first, while device_/present_ are still alive for it to quiesce
  // against. If that fails, this function returns immediately and retains every GPU resource
  // (staging, render_, present_, device_, window_) for the remainder of the process rather than
  // risk freeing memory or destroying a device the GPU may still be using — restart is not
  // supported after this path. Only on success do we proceed to free staging and tear down the
  // rest in dependency order. The present backend's destroy() is the second such proof (its
  // output-side work) and fails the same way.
  if (render_) {
    auto r = render_->destroy(std::chrono::seconds{5});
    if (!r) {
      HOLOSCAN_LOG_ERROR(translate_error(r.error(), "stop: render_->destroy").message);
      HOLOSCAN_LOG_ERROR(
          "HolovizOp::stop: renderer did not prove quiescence; retaining all GPU resources "
          "for process lifetime instead of risking a use-after-free. This operator cannot be "
          "restarted.");
      terminal_failed_ = true;
      return;
    }
  }

  // Successful renderer destruction is the other proof that asynchronous FrameData reads have
  // stopped. Release retained input backing only after that proof.
  retained_image_tensors_ = {};

  if (image_upload_stream_ != nullptr) {
    auto context = device_.cuda_context();
    if (!context) {
      HOLOSCAN_LOG_ERROR(translate_error(context.error(), "stop: Device::cuda_context").message);
      // Fall through: stream leaked, but continue best-effort teardown of remaining resources.
    } else {
      auto destroyed = destroy_cuda_stream(*context, image_upload_stream_);
      if (!destroyed) {
        HOLOSCAN_LOG_ERROR(destroyed.error().message);
        // Fall through: stream leaked, but continue best-effort teardown.
      } else {
        image_upload_stream_ = nullptr;
      }
    }
  }

  for (auto& per_image : image_staging_devptrs_) {
    for (auto& devptr : per_image) {
      if (devptr != 0U) {
        cudaError_t err =
            cudaFree(reinterpret_cast<void*>(devptr));  // NOLINT(cppcoreguidelines-owning-memory)
        if (err != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("HolovizOp::stop: cudaFree failed: {}", cudaGetErrorString(err));
        }
        devptr = 0U;
      }
    }
  }
  image_staging_devptrs_.clear();

  if (present_) {
    auto r = present_->destroy(std::chrono::seconds{5});
    if (!r) {
      // The present side did not prove its output work complete (for the CUDA sink: a resolve
      // still in flight against the ring and a loaned Tensor). Same terminal policy as the
      // renderer above: keep every GPU resource for the process lifetime rather than free
      // memory the copy may still target. Retained outputs stay parked with them.
      HOLOSCAN_LOG_ERROR(translate_error(r.error(), "stop: present_->destroy").message);
      HOLOSCAN_LOG_ERROR(
          "HolovizOp::stop: present backend did not prove quiescence; retaining all GPU "
          "resources for process lifetime instead of risking a use-after-free. This operator "
          "cannot be restarted.");
      terminal_failed_ = true;
      return;
    }
  }
  // A successful present destroy drained every resolve, so parked outputs are safe to release.
  retained_outputs_.clear();
  render_.reset();
  cuda_present_ = nullptr;
  present_.reset();
  surface_handle_.impl.reset();

  if (device_) {
    auto r = device_.destroy(std::chrono::seconds{5});
    if (!r) {
      HOLOSCAN_LOG_ERROR(translate_error(r.error(), "stop: device_.destroy").message);
    }
  }

  if (window_) {
    auto r = window_.destroy();
    if (!r) {
      HOLOSCAN_LOG_ERROR(translate_error(r.error(), "stop: window_.destroy").message);
    }
  }

  // Reset present_id_ so a restart begins a fresh strictly-increasing sequence.
  present_id_ = 0;

  for (auto& cache : image_layer_caches_) {
    cache.bytes.clear();
    cache.width = 0;
    cache.height = 0;
    cache.pitch = 0;
    cache.last_frame = {};
  }
  last_image_reject_reason_.fill(ImageRejectReason::kAccepted);
  unspecified_color_space_noted_.fill(false);
}

}  // namespace holoscan::visualization
