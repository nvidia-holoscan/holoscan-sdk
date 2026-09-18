// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The public face of HolovizOp: construction-time validation, port declaration, the temporal
// contract, and the lifecycle overrides forwarding into HolovizOp::Impl (holoviz_op_impl.cpp).

#include <holoscan/visualization/holoviz_op.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#include <holoscan/logger/logger.hpp>

#include "holoviz_op/holoviz_op_impl.hpp"

namespace holoscan::visualization {

HolovizOp::HolovizOp(LayerStack layer_stack, SinkDesc sink, std::chrono::nanoseconds display_rate,
                     std::optional<std::chrono::nanoseconds> max_input_age)
    : layer_stack_(std::move(layer_stack)),
      sink_(std::move(sink)),
      display_rate_(display_rate),
      max_input_age_(max_input_age.value_or(display_rate)) {
  // Validate display_rate before layer-count so a bad cadence fails immediately, not after
  // an otherwise-valid layer stack has already been accepted.
  if (display_rate_ <= std::chrono::nanoseconds::zero()) {
    HOLOSCAN_LOG_ERROR("HolovizOp: display_rate must be > 0 ns; got {} ns.", display_rate_.count());
    std::terminate();
  }

  // Validate compile-time port capacity at construction so the failure surfaces early,
  // not mid-setup() when individual layer_input() accesses start terminating.
  if (layer_stack_.layers().size() > kMaxLayerPorts) {
    HOLOSCAN_LOG_ERROR(
        "HolovizOp: layer count {} exceeds kMaxLayerPorts ({}); increase the "
        "compile-time limit or reduce the declared layer count.",
        layer_stack_.layers().size(),
        kMaxLayerPorts);
    std::terminate();
  }

  impl_ = std::make_unique<Impl>(*this);
}

HolovizOp::~HolovizOp() {
  // If stop()'s render_->destroy() quiescence check failed, the renderer, present backend, and
  // device were kept alive for the process lifetime. In that case the retained input tensors
  // must be released here while those objects are still valid: Impl declares
  // retained_image_tensors_ before render_/present_, so the default member destruction order
  // would free the tensor backing after the backends are gone.
  if (impl_ && impl_->terminal_failed_) {
    impl_->retained_image_tensors_ = {};
  }
}

// ---- setup() ----------------------------------------------------------------

void HolovizOp::setup(holoscan::OperatorSpec& spec) {
  const auto& layers = layer_stack_.layers();
  for (std::size_t i = 0; i < layers.size(); ++i) {
    std::visit(
        [&](auto& desc) {
          using T = std::decay_t<decltype(desc)>;
          if constexpr (std::is_same_v<T, GeometryLayerDesc>) {
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

// ---- contract() -------------------------------------------------------------

holoscan::Contract HolovizOp::contract() const {
  // OnClock drives rendering at the display refresh rate.
  // Alternative: OnEach{layer_input_0_} triggers on primary layer arrival but
  // requires the application to call graph.partition().add() for every operator
  // in the graph to satisfy PARTITION_MANUAL_AUTHORING_REQUIRED.
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = display_rate_});
  const auto& layers = layer_stack_.layers();
  for (std::size_t i = 0; i < layers.size(); ++i) {
    result.input(layer_input(i), holoscan::LatestWithin{.max_age = max_input_age_});
    if (std::holds_alternative<ImageLayerDesc>(layers[i].desc)) {
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

// ---- lifecycle forwarders ---------------------------------------------------

void HolovizOp::start() {
  impl_->start();
}

holoscan::expected<void, holoscan::Error> HolovizOp::compute(holoscan::ExecutionContext& context) {
  return impl_->compute(context);
}

void HolovizOp::stop() {
  impl_->stop();
}

}  // namespace holoscan::visualization
