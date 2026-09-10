// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Standard C++
#include <array>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// Holoscan Core
#include <holoscan/operators/holoviz/holoscan_operator_holoviz_export.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>

// HoloViz module — public types
#include <holoviz/holoviz_export.h>
#include <holoviz/layer_desc.hpp>
#include <holoviz/sink_desc.hpp>

// HoloViz backend seam
#include <holoviz/backends/rhi_render_backend.hpp>
#include <holoviz/backends/wsi_present_backend.hpp>
#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>
#include <holoviz/frame_data.hpp>
#include <holoviz/handles.hpp>
#include <holoviz/present_backend.hpp>
#include <holoviz/render_backend.hpp>
#include <holoviz/rhi.hpp>

namespace holoscan::holoviz {

/// Compile-time maximum number of layer input ports on one HolovizOp instance.
inline constexpr std::size_t kMaxLayerPorts = 8;

/// Compile-time upper bound on the swapchain image count HolovizOp supports. The render slot
/// ring (RenderConfig::max_frames_in_flight) and the per-image staging buffer ring are both
/// sized to this bound once in start(), not to the swapchain's initial actual image count — a
/// present-mode-driven recreation can grow the image count later (kImageCountChanged), and a
/// ring sized only to the initial count would then be indexed out of bounds. 8 comfortably
/// covers the present modes and adapters this operator targets (observed swapchains request at
/// most a handful of images); start() rejects an initial count above this bound, and compute()
/// rejects a later grown count above it the same way, rather than attempt a live ring rebuild.
inline constexpr std::uint32_t kMaxSwapchainImages = 8;

/// @brief Holoscan operator that drives HoloViz rendering at a fixed display rate.
///
/// HolovizOp is a sink operator with up to @ref kMaxLayerPorts typed Tensor inputs, one per
/// declared layer. The rendering cadence is governed by an OnClock trigger so the pipeline runs
/// even when upstream data is temporarily absent (late or missing layers receive a SafeDefault
/// selection and are skipped in compute()).
///
/// ### Usage
///
/// ```cpp
/// ::holoviz::LayerStack stack;
/// stack.add_image_layer("video", ::holoviz::ImageLayerDesc{...});
/// stack.add_geometry_layer("overlay", ::holoviz::GeometryLayerDesc{...});
///
/// auto op = graph.op<holoscan::holoviz::HolovizOp>(
///     "holoviz", std::move(stack), ::holoviz::WindowSinkDesc{.title = "Demo"});
/// graph.partition("holoviz").add(op);   // graphs require explicit partition membership
/// ```
///
/// @note HoloViz backend errors are translated into the Holoscan Core error domain at the
/// Operator lifecycle boundary.
class HOLOSCAN_OPERATOR_HOLOVIZ_EXPORT HolovizOp : public holoscan::Operator<> {
 public:
  /// @brief Construct HolovizOp with a fixed layer stack, sink declaration, and display period.
  ///
  /// @param layer_stack  Immutable layer declarations consumed by setup() and start(). The number
  ///                     of layers must not exceed @ref kMaxLayerPorts.
  /// @param sink         Declared display or capture sink (window, exclusive display, CUDA buffer).
  /// @param display_rate Rendering period; defaults to approximately 16.67 ms (60 Hz).
  /// @param max_input_age Freshness window for each layer's `LatestWithin` selector (see
  ///                     contract()). Defaults to @p display_rate, which is only appropriate when
  ///                     every producer feeding a layer publishes at least as fast as the display
  ///                     rate; a slower producer (e.g. 30 fps video decode into a 60 Hz display)
  ///                     needs a wider window here or its layer is missing on most activations.
  explicit HolovizOp(
      ::holoviz::LayerStack layer_stack, ::holoviz::SinkDesc sink,
      std::chrono::nanoseconds display_rate = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>{1.0 / 60.0}),
      std::optional<std::chrono::nanoseconds> max_input_age = std::nullopt);

  // ---- port accessor -------------------------------------------------------

  /// @brief Return the immutable input port declaration for layer @p index.
  /// @param index Zero-based index; must be less than kMaxLayerPorts.
  [[nodiscard]] const holoscan::Input<holoscan::Tensor>& layer_input(
      std::size_t index) const noexcept {
    switch (index) {
      case 0:
        return layer_input_0_;
      case 1:
        return layer_input_1_;
      case 2:
        return layer_input_2_;
      case 3:
        return layer_input_3_;
      case 4:
        return layer_input_4_;
      case 5:
        return layer_input_5_;
      case 6:
        return layer_input_6_;
      case 7:
        return layer_input_7_;
      default:
        std::terminate();  // unreachable: index >= kMaxLayerPorts
    }
  }

  /// @brief Return the mutable input port declaration for layer @p index.
  /// @param index Zero-based index; must be less than kMaxLayerPorts.
  [[nodiscard]] holoscan::Input<holoscan::Tensor>& layer_input(std::size_t index) noexcept {
    // Delegate to the const overload; safe because `this` is non-const here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<holoscan::Input<holoscan::Tensor>&>(
        static_cast<const HolovizOp*>(this)->layer_input(index));
  }

  // ---- Operator lifecycle --------------------------------------------------

  /// @brief Release retained GPU-backed tensors if stop() left GPU resources live.
  ///
  /// If stop()'s render_->destroy() quiescence check failed, render_/present_/device_ were kept
  /// alive for the process lifetime. In that case the destructor body (which runs before member
  /// destructors) must clear retained_image_tensors_ while those objects are still valid, because
  /// the default member destruction order would free tensor backing first (retained_image_tensors_
  /// is declared before render_/present_ and is therefore destroyed earlier in the reverse order).
  ~HolovizOp();

  /// @brief Bind each declared layer to a typed Tensor input port with memory-kind constraints.
  void setup(holoscan::OperatorSpec& spec) override {
    const auto& layers = layer_stack_.layers();
    for (std::size_t i = 0; i < layers.size(); ++i) {
      std::visit(
          [&](auto& desc) {
            using T = std::decay_t<decltype(desc)>;
            if constexpr (std::is_same_v<T, ::holoviz::GeometryLayerDesc>) {
              // build_render_list() treats this Tensor as a flat float32 coordinate array
              // (rank 1, float32, host-resident). A mismatched producer is rejected by
              // Runtime contract enforcement before reaching data_as<float>().
              holoscan::TensorInputSpec ts{};
              ts.representation.memory_kind = holoscan::MemoryKind::kHost;
              ts.representation.dtype = DLDataType{kDLFloat, 32U, 1U};
              ts.representation.rank = 1U;
              spec.input(layer_input(i), layers[i].port_name).queue_depth(8U).expects_tensor(ts);
            } else {
              // ImageLayerDesc / TextLayerDesc / LutLayerDesc: unconstrained — no
              // expects_tensor() call. A default-constructed TensorInputSpec is rejected by
              // Core as an empty contract; unconstrained ports must omit the call entirely.
              spec.input(layer_input(i), layers[i].port_name).queue_depth(8U);
            }
          },
          layers[i].desc);
    }
  }

  /// @brief Declare the clock-driven temporal contract for all layer ports.
  [[nodiscard]] holoscan::Contract contract() const override {
    // OnClock drives rendering at the display refresh rate.
    // Alternative: OnEach{layer_input_0_} triggers on primary layer arrival but
    // requires the application to call graph.partition().add() for every operator
    // in the graph to satisfy PARTITION_MANUAL_AUTHORING_REQUIRED.
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = display_rate_});
    const auto& layers = layer_stack_.layers();
    for (std::size_t i = 0; i < layers.size(); ++i) {
      result.input(layer_input(i), holoscan::LatestWithin{.max_age = max_input_age_});
      if (std::holds_alternative<::holoviz::ImageLayerDesc>(layers[i].desc)) {
        // Image layers hold the last frame across activations where a slower producer has
        // nothing new (see build_render_list()'s doc comment), but that hold must be bounded —
        // an indefinite freeze would let a stopped or failed camera keep rendering as if it were
        // still live. ReuseLast expresses that bound in the temporal contract itself instead of
        // a private, unbounded cache: is_missing() only becomes true once max_input_age_ has
        // actually elapsed since the last real sample, exactly like the LatestWithin bound.
        result.on_missing(layer_input(i), holoscan::ReuseLast{.max_age = max_input_age_});
      } else {
        // Geometry (and future text/LUT) layers are skipped on a missing activation — no
        // implicit hold-last for these layer kinds.
        result.on_missing(layer_input(i), holoscan::SafeDefault{});
      }
    }
    return result;
  }

  /// @brief Return the device staging buffer for layer @p index's host-resident image data.
  ///
  /// One buffer exists per swapchain image (see start()'s Step 5) so that a new frame's upload
  /// never overwrites a buffer the GPU may still be sampling for a previous, still-in-flight
  /// frame — the swapchain image fence alone does not protect this side-channel buffer.
  ///
  /// @param index       Zero-based layer index; must be less than kMaxLayerPorts.
  /// @param image_index Swapchain image index for the frame currently being built (see
  ///                     PresentBackend::acquire()'s AcquiredImage::image_index).
  /// @return A CUDA device pointer sized to that layer's declared max_width*max_height*4 bytes,
  ///         valid only between start() and stop(); zero for non-ImageLayerDesc layers.
  [[nodiscard]] CUdeviceptr image_staging_devptr(std::size_t index,
                                                 std::uint32_t image_index) const noexcept {
    if (index >= kMaxLayerPorts || image_index >= image_staging_devptrs_.size()) {
      return 0;
    }
    return image_staging_devptrs_[image_index][index];
  }

  /// @brief Host-side cache of layer @p index's most recently received image sample.
  ///
  /// build_render_list() re-uploads this on activations where the layer's input is missing (see
  /// its usage for why: kLatestWithin selects a sample at most once, so a producer slower than
  /// the display clock legitimately has no new sample on most activations — without holding the
  /// last frame here the layer would flicker between real content and nothing every such tick).
  /// Empty (bytes.empty() == true) until the first real sample for that layer arrives.
  struct ImageLayerCache {
    std::vector<std::byte> bytes;
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::size_t pitch{0};
  };

  /// @param index Zero-based layer index; must be less than kMaxLayerPorts.
  [[nodiscard]] ImageLayerCache& image_layer_cache(std::size_t index) noexcept {
    return image_layer_caches_[index];
  }

  /// @brief Create the window, device, and backends; allocate GPU resources; build pipelines.
  void start() override;

  /// @brief Consume available layer tensors and drive one rendered frame through the backend.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  /// @brief Release GPU resources in strict reverse-allocation order.
  void stop() override;

 private:
  /// @brief Adopt a post-recreation SwapchainInfo reported via kImageCountChanged.
  ///
  /// The render slot ring and staging ring are both pre-sized to kMaxSwapchainImages in start()
  /// (see its doc comment), so a grown count within that bound needs no ring rebuild — this just
  /// updates swapchain_info_. A grown count exceeding the bound cannot be served by either ring;
  /// unlike kBackpressured/kOutOfDate, that is not safely retryable, so this reports false rather
  /// than let the caller treat it as transient.
  /// @return true if `info` was adopted; false if `info.image_count` exceeds kMaxSwapchainImages.
  [[nodiscard]] bool adopt_swapchain_info_change(const ::holoviz::SwapchainInfo& info) noexcept;

  ::holoviz::LayerStack layer_stack_;
  ::holoviz::SinkDesc sink_;
  std::chrono::nanoseconds display_rate_;
  std::chrono::nanoseconds max_input_age_;

  // Runtime backend handles — all valid only between start() and stop().
  // rhi::Window and rhi::Device expose operator bool() (impl_ != nullptr) so bare
  // value members suffice; stop() guards each step with the bool predicate.
  ::holoviz::rhi::Window window_;
  ::holoviz::SurfaceHandle surface_handle_;
  ::holoviz::rhi::Device device_;
  CUstream image_upload_stream_{nullptr};
  std::unique_ptr<::holoviz::RenderBackend> render_;
  std::unique_ptr<::holoviz::PresentBackend> present_;
  ::holoviz::SwapchainInfo swapchain_info_{};
  ::holoviz::PresentId present_id_{0};
  // Set to true if stop() returns early because render_->destroy() could not prove quiescence.
  // Prevents restart (start() throws) and signals the destructor to clear retained_image_tensors_
  // before member destructors release GPU-backing memory.
  bool terminal_failed_{false};

  // CUDA ordinal device_ was created against (start()), used to reject an activation stream from
  // a different device at the top of compute() — see the doc comment there.
  int cuda_device_ordinal_{-1};

  // Device staging buffers for ImageLayerDesc layers: one array of kMaxLayerPorts buffers per
  // swapchain image (outer index == swapchain image_index), so uploads for frame N+1 cannot
  // race a still-in-flight GPU read of frame N's buffer. Host-resident image tensors are
  // uploaded here in compute() before being handed to the render backend, which requires a
  // real CUDA device pointer (ImagePlane::device_ptr). Zero for layers that are not
  // ImageLayerDesc, or before start() runs. Sized to swapchain_info_.image_count in start().
  std::vector<std::array<CUdeviceptr, kMaxLayerPorts>> image_staging_devptrs_;

  // Direct CUDA image inputs whose backing cannot register image_upload_stream_ for stream-ordered
  // release remain here until the next successful begin_frame() proves the previous reads have
  // completed. Native Holoscan and HoloMQ-backed tensors normally take the stream-aware fast path;
  // producer-owned DLPack tensors use this bounded fallback.
  std::array<std::array<holoscan::Tensor, kMaxLayerPorts>, kMaxSwapchainImages>
      retained_image_tensors_;

  // Last-received-sample cache per layer, used to hold the last frame on activations where a
  // layer's LatestWithin selector has nothing new (see image_layer_cache()'s doc comment).
  std::array<ImageLayerCache, kMaxLayerPorts> image_layer_caches_;

  // Input port bank — kMaxLayerPorts = 8 stable instance members.
  // Input<T> is noncopyable and nonmovable, so these cannot live in a std::array.
  // Port at index i corresponds to layer_stack_.layers()[i] after setup() is called.
  // Ports at index >= layer_stack_.layers().size() are never registered and remain unbound.
  holoscan::Input<holoscan::Tensor> layer_input_0_;
  holoscan::Input<holoscan::Tensor> layer_input_1_;
  holoscan::Input<holoscan::Tensor> layer_input_2_;
  holoscan::Input<holoscan::Tensor> layer_input_3_;
  holoscan::Input<holoscan::Tensor> layer_input_4_;
  holoscan::Input<holoscan::Tensor> layer_input_5_;
  holoscan::Input<holoscan::Tensor> layer_input_6_;
  holoscan::Input<holoscan::Tensor> layer_input_7_;
};

}  // namespace holoscan::holoviz
