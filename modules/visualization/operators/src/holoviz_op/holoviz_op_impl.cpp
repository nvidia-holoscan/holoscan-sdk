// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HolovizOp::Impl: the start()/compute()/stop() frame loop over the RenderBackend /
// PresentBackend seam. The public class (holoviz_op.cpp) forwards its lifecycle overrides here.

#include "holoviz_op/holoviz_op_impl.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <cuda_runtime_api.h>
#include <fmt/format.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/logger/logger.hpp>

#include "common/holoviz_conversions.hpp"
#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/backends/holoviz_render_backend.hpp"
#include "holoviz_op/backends/wsi_present_backend.hpp"
#include "holoviz_op/error_translation.hpp"
#include "holoviz_op/holoviz_op_window_desc.hpp"

namespace holoscan::visualization {

// ---- Format helpers ------------------------------------------------------------

/// Bytes per pixel for the single-plane, non-YUV formats this image-layer upload path supports.
/// Returns 0 for multi-plane/YUV formats (kNV12, kYUYV422) — not supported by this contiguous,
/// single-plane staging-buffer/upload path (matches ImageFrameData::plane_count == 1 elsewhere
/// in this file).
static std::uint32_t bytes_per_pixel(ImageFormat format) noexcept {
  switch (format) {
    case ImageFormat::kR8Unorm:
      return 1U;
    case ImageFormat::kR16Unorm:
      return 2U;
    case ImageFormat::kR8G8B8Unorm:
      return 3U;
    case ImageFormat::kR8G8B8A8Unorm:
    case ImageFormat::kB8G8R8A8Unorm:
      return 4U;
    case ImageFormat::kR16G16B16A16Unorm:
      return 8U;
    case ImageFormat::kUnknown:
    case ImageFormat::kNV12:
    case ImageFormat::kYUYV422:
    default:
      return 0U;
  }
}

namespace {

/// Element type and channel count a single-plane ImageFormat expects from its HWC tensor.
struct FormatElement {
  std::uint8_t bits = 0;
  std::int64_t channels = 0;
};

std::optional<FormatElement> format_element(ImageFormat format) noexcept {
  switch (format) {
    case ImageFormat::kR8Unorm:
      return FormatElement{8U, 1};
    case ImageFormat::kR8G8B8Unorm:
      return FormatElement{8U, 3};
    case ImageFormat::kR8G8B8A8Unorm:
    case ImageFormat::kB8G8R8A8Unorm:
      return FormatElement{8U, 4};
    case ImageFormat::kR16Unorm:
      return FormatElement{16U, 1};
    case ImageFormat::kR16G16B16A16Unorm:
      return FormatElement{16U, 4};
    case ImageFormat::kUnknown:
    case ImageFormat::kNV12:
    case ImageFormat::kYUYV422:
      return std::nullopt;
  }
  return std::nullopt;
}

/// True when a tensor's dtype and trailing channel dimension agree with the declared format.
/// Accepts the natural element type with `{H, W}` (single channel only) or `{H, W, channels}`,
/// and a packed byte view `uint8 {H, W, bytes_per_pixel}` for producers that emit raw bytes.
/// Without this check a mismatched sample (e.g. uint16 on a kR8Unorm layer) uploads with the
/// wrong pitch and renders silently garbled.
bool tensor_matches_format(const holoscan::Tensor& tensor, const std::vector<std::int64_t>& shape,
                           ImageFormat format) noexcept {
  const auto element = format_element(format);
  if (!element || shape.size() < 2U || shape.size() > 3U) {
    return false;
  }
  const DLDataType dtype = tensor.dtype();
  if (dtype.code != kDLUInt || dtype.lanes != 1U) {
    return false;
  }
  const std::int64_t channels = shape.size() == 3U ? shape[2] : 1;
  if (dtype.bits == element->bits) {
    return channels == element->channels;
  }
  return dtype.bits == 8U && channels == static_cast<std::int64_t>(bytes_per_pixel(format));
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

  // Reserve before the loop so no emplace_back/push_back on the outer vectors can
  // reallocate and invalidate the std::span pointers stored in geometry_frames entries.
  const std::size_t n = layers.layers().size();
  result.storage.geometry_batches.reserve(n);
  result.storage.geometry_frames.reserve(n);
  result.storage.image_frames.reserve(n);

  for (std::size_t i = 0; i < layers.layers().size(); ++i) {
    const bool missing = op.layer_input(i).is_missing();

    std::visit(
        [&](auto& desc) {
          using T = std::decay_t<decltype(desc)>;

          if constexpr (std::is_same_v<T, ImageLayerDesc>) {
            // contract() gives image ports ReuseLast{.max_age = max_input_age_}: with HolovizOp's
            // OnClock running faster than a slower producer (e.g. 60 Hz display vs. 30 fps
            // decode), most activations legitimately have no *new* sample, and Runtime reuses the
            // last admitted one (bounded by max_input_age_) instead of reporting missing — so
            // is_missing() here means the bound has genuinely elapsed, not just "nothing new this
            // tick". Only then is the layer dropped from the frame; every other activation
            // (fresh or Runtime-reused) re-derives and re-uploads from receive_data() below, so
            // the operator never holds its own unbounded, independently-aged copy.
            if (missing) {
              return;
            }
            auto& cache = image_layer_cache(i);
            {
              auto payload = op.layer_input(i).receive_data();
              if (!payload) {
                return;  // receive error on a non-missing port is unexpected but non-fatal
              }
              holoscan::Tensor& tensor = *payload;
              const auto strides = tensor.strides();  // byte strides
              const auto shape = tensor.shape();
              const std::uint32_t bpp = bytes_per_pixel(desc.format);
              // Validate before any arithmetic on shape/stride values: a negative or overflowing
              // signed shape/stride cast straight to size_t (the previous approach) can produce a
              // huge "bytes" figure that both over-allocates the host cache and, more seriously,
              // exceeds the fixed max_width * max_height * bpp CUDA staging buffer sized once in
              // start() — writing past it during the H2D upload below. So every dimension is
              // checked against the layer's own declared bounds before it is used at all.
              if (bpp == 0U || shape.size() < 2U || shape[0] <= 0 || shape[1] <= 0) {
                HOLOSCAN_LOG_ERROR(
                    "HolovizOp::build_render_list: image layer {} received a Tensor with an "
                    "unsupported format or invalid shape; dropping this activation's sample.",
                    i);
                return;
              }
              if (!tensor_matches_format(tensor, shape, desc.format)) {
                const DLDataType dtype = tensor.dtype();
                HOLOSCAN_LOG_ERROR(
                    "HolovizOp::build_render_list: image layer {} Tensor (dtype code {} bits {} "
                    "lanes {}, rank {}, channels {}) does not match the declared ImageFormat {}; "
                    "dropping this activation's sample.",
                    i,
                    static_cast<int>(dtype.code),
                    static_cast<int>(dtype.bits),
                    static_cast<int>(dtype.lanes),
                    shape.size(),
                    shape.size() == 3U ? shape[2] : 1,
                    common::image_format_name(desc.format));
                return;
              }
              const auto new_h = static_cast<std::size_t>(shape[0]);
              const auto new_w = static_cast<std::size_t>(shape[1]);
              if (new_w > desc.max_width || new_h > desc.max_height) {
                HOLOSCAN_LOG_ERROR(
                    "HolovizOp::build_render_list: image layer {} Tensor is {}x{}, exceeding "
                    "the declared max {}x{}; dropping this activation's sample.",
                    i,
                    new_w,
                    new_h,
                    desc.max_width,
                    desc.max_height);
                return;
              }
              const std::size_t min_pitch = new_w * bpp;
              std::size_t new_pitch = min_pitch;
              if (!strides.empty()) {
                if (strides[0] <= 0 || static_cast<std::size_t>(strides[0]) < min_pitch) {
                  HOLOSCAN_LOG_ERROR(
                      "HolovizOp::build_render_list: image layer {} row stride is negative or "
                      "shorter than the format's minimum row size; dropping this activation's "
                      "sample.",
                      i);
                  return;
                }
                new_pitch = static_cast<std::size_t>(strides[0]);
              }
              // Active byte span: the last row ends after its final pixel, not after the
              // full trailing pitch. Copying new_pitch * new_h can read past the tensor's
              // valid span when the tensor is a strided view without owned padding on the
              // final row (e.g. a slice of a wider buffer). new_pitch * new_h is used only
              // for the staging-capacity bound check (conservative); the D2H copy uses the
              // tighter safe_bytes so we never read unowned memory.
              const std::size_t safe_bytes = (new_h - 1U) * new_pitch + min_pitch;
              const std::size_t max_bytes =
                  static_cast<std::size_t>(desc.max_width) * desc.max_height * bpp;
              if (safe_bytes > max_bytes) {
                HOLOSCAN_LOG_ERROR(
                    "HolovizOp::build_render_list: image layer {} active byte span {} "
                    "exceeds the staging buffer's allocated size {}; dropping this "
                    "activation's sample.",
                    i,
                    safe_bytes,
                    max_bytes);
                return;
              }
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
              // Host/pinned path: Tensor::copy_to_host() handles producer readiness before
              // the D2H copy. Commit width/height/pitch/bytes to cache atomically (via
              // move into a local buffer first) so a copy failure leaves cache intact for
              // the next activation's hold-last re-upload.
              const DLDeviceType dev_type = tensor.device().device_type;
              const bool is_cuda_resident = (dev_type == kDLCUDA || dev_type == kDLCUDAManaged);

              if (is_cuda_resident) {
                // Direct device-pointer path: no D2H copy, no staging buffer.
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                const auto dev_ptr =
                    static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(tensor.data()));
                if (dev_ptr == 0U) {
                  HOLOSCAN_LOG_ERROR(
                      "HolovizOp::build_render_list: image layer {} CUDA tensor has a null "
                      "data pointer; dropping this activation's sample.",
                      i);
                  return;
                }
                // Retain the backing until record() queues its asynchronous copy. compute() then
                // registers stream-ordered release when the backing supports it, or transfers the
                // handle to the bounded frame-slot fallback.
                result.storage.asynchronous_image_tensors[i] = tensor;
                result.storage.image_frames.push_back(backends::ImageFrameData{
                    .width = static_cast<uint32_t>(new_w),
                    .height = static_cast<uint32_t>(new_h),
                    .planes = {{dev_ptr, new_pitch}},
                    .plane_count = 1U,
                    .stream = cuda_stream,
                });
                result.list.layers.push_back(
                    {static_cast<uint32_t>(i), &result.storage.image_frames.back()});
              } else {
                // Host/pinned path: copy_to_host() waits on the producer readiness event then
                // memcpys. NOTE: use the untyped data() accessor, not data_as<std::byte>() —
                // Tensor::dtype_matches<T>() has no specialization for std::byte so
                // data_as<std::byte>() always returns nullptr regardless of actual data.
                std::vector<std::byte> next_bytes(safe_bytes);
                auto copied = tensor.copy_to_host(next_bytes.data(), safe_bytes, cuda_stream);
                if (!copied) {
                  // Hard failure: propagate so compute() can recover the open frame.
                  fatal_error = holoscan::Error{
                      holoscan::ErrorCode::kFailure,
                      "HolovizOp::build_render_list: image layer " + std::to_string(i) +
                          " copy_to_host failed: " + copied.error().message};
                  return;
                }
                cache.bytes = std::move(next_bytes);
                cache.width = new_w;
                cache.height = new_h;
                cache.pitch = new_pitch;

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
                result.storage.image_frames.push_back(backends::ImageFrameData{
                    .width = static_cast<uint32_t>(cache.width),
                    .height = static_cast<uint32_t>(cache.height),
                    .planes = {{ptr, cache.pitch}},
                    .plane_count = 1U,
                    .stream = cuda_stream,
                });
                result.list.layers.push_back(
                    {static_cast<uint32_t>(i), &result.storage.image_frames.back()});
              }
            }

          } else if constexpr (std::is_same_v<T, GeometryLayerDesc>) {
            if (missing) {
              return;
            }
            auto payload = op.layer_input(i).receive_data();
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
            // Text overlay path is not yet implemented.
            (void)desc;

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

  // Only WindowSinkDesc is implemented; the remaining sink kinds are declared API
  // but their present paths are not yet wired.
  const auto* window_desc = std::get_if<WindowSinkDesc>(&op.sink_);
  if (window_desc == nullptr) {
    std::visit(
        [](auto& d) {
          using T = std::decay_t<decltype(d)>;
          const char* kind = std::is_same_v<T, ExclusiveDisplaySinkDesc>  ? "ExclusiveDisplay"
                             : std::is_same_v<T, CudaPitchLinearSinkDesc> ? "CudaPitchLinear"
                             : std::is_same_v<T, CudaArraySinkDesc>       ? "CudaArray"
                                                                          : "unknown";
          throw holoscan::RuntimeError{
              holoscan::ErrorCode::kNotImplemented,
              std::string("HolovizOp: SinkDesc kind '") + kind +
                  "' is not yet implemented; only WindowSinkDesc is supported.",
          };
        },
        op.sink_);
    return;  // unreachable: visit always throws
  }
  const WindowSinkDesc& w = *window_desc;

  // Step 1: window + surface + device
#if !defined(_WIN32)
  // The NVIDIA X11 driver paces FIFO presents on the primary output's vblank unless
  // __GL_SYNC_DISPLAY_DEVICE names the target monitor. The ICD reads it once, when the
  // process-wide Vulkan instance is first created, and setenv() races getenv() on other
  // scheduler threads, so the operator never writes it: the application exports it before any
  // Vulkan use. Only hint when it is missing for a named fullscreen monitor.
  if (w.fullscreen && !w.display_name.empty()) {
    constexpr const char* kSyncDisplayEnv = "__GL_SYNC_DISPLAY_DEVICE";
    if (std::getenv(kSyncDisplayEnv) == nullptr) {
      HOLOSCAN_LOG_INFO(
          "HolovizOp: {} is unset; if '{}' is not the primary output, export {}={} before "
          "launching so the NVIDIA X11 driver paces presents on that monitor",
          kSyncDisplayEnv,
          w.display_name,
          kSyncDisplayEnv,
          w.display_name);
    }
  }
#endif
  {
    auto r = holoviz::create_window(detail::window_desc_from_sink(w));
    if (!r) {
      throw_holoviz_error(r.error(), "create_window");
    }
    window_ = std::move(*r);
  }

  {
    auto r = backends::make_surface_handle(window_);
    if (!r) {
      throw_holoviz_error(r.error(), "make_surface_handle");
    }
    surface_handle_ = std::move(*r);
  }

  {
    auto r = holoviz::create_device_for_surface(
        {}, *std::static_pointer_cast<holoviz::Surface>(surface_handle_.impl));
    if (!r) {
      throw_holoviz_error(r.error(), "create_device_for_surface");
    }
    device_ = std::move(*r);

    // Record which CUDA device device_ actually landed on: create_device_for_surface() picks
    // the first surface-capable adapter with no Holoscan-side device preference, so on a
    // multi-GPU host this is not guaranteed to be the CUDA device the Runtime later admits for
    // this operator's activation stream. compute() rejects a mismatched stream instead of
    // silently mixing a staging pointer, stream, and HoloViz device from different CUDA contexts.
    auto cuda_dev = device_.cuda_device();
    if (!cuda_dev) {
      throw_holoviz_error(cuda_dev.error(), "Device::cuda_device");
    }
    cuda_device_ordinal_ = static_cast<int>(*cuda_dev);
  }

  // Step 2: backends
  {
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
  }
  {
    auto r = backends::create_wsi_present_backend(device_);
    if (!r) {
      throw_holoviz_error(r.error(), "create_wsi_present_backend");
    }
    present_ = std::move(*r);
  }

  // Step 3: swapchain first so we know the actual image_count before allocate(). The seam
  // itself allows allocate → swapchain → build; this operator derives max_frames_in_flight from
  // the real swapchain image count, which requires the swapchain to exist before allocate() is
  // called.
  {
    // HoloViz sizes a window swapchain from the surface's live extent; this is advisory.
    auto extent_result = window_.extent();
    const backends::Extent2D preferred_extent =
        extent_result ? backends::Extent2D{extent_result->width, extent_result->height}
                      : backends::Extent2D{w.width, w.height};

    auto r = present_->create_swapchain(backends::SwapchainDesc{
        .surface = surface_handle_,
        .preferred_extent = preferred_extent,
        .preferred_mode = common::to_holoviz_present_mode(w.present_mode),
        .image_count = std::nullopt,  // let backend choose
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
        const std::uint32_t bpp = bytes_per_pixel(image_desc->format);
        if (bpp == 0U) {
          staging_error = holoscan::RuntimeError{
              holoscan::ErrorCode::kNotImplemented,
              "HolovizOp: image layer format is not a supported single-plane upload format "
              "(multi-plane/YUV formats are not yet supported by this staging-buffer path)"};
          break;
        }
        const std::size_t bytes =
            static_cast<std::size_t>(image_desc->max_width) * image_desc->max_height * bpp;
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

  // Acquire the next swapchain image.
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

  // Begin the frame slot for this swapchain image.
  auto cmds_result = render_->begin_frame(*acquired, acquired->image_index);
  if (!cmds_result) {
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

  return {};
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
  // rest in dependency order.
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
      HOLOSCAN_LOG_ERROR(translate_error(r.error(), "stop: present_->destroy").message);
    }
  }
  render_.reset();
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
  }
}

}  // namespace holoscan::visualization
