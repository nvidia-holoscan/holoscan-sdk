// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The public face of HolovizOp: construction-time validation, port declaration, the temporal
// contract, and the lifecycle overrides forwarding into HolovizOp::Impl (holoviz_op_impl.cpp).

#include <holoscan/visualization/holoviz_op.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/logger/logger.hpp>

#include "common/holoviz_conversions.hpp"
#include "holoviz_op/holoviz_op_impl.hpp"

namespace holoscan::visualization {

// Every legal stack (at most kMaxLayerPorts layers of any kind mix) must fit its kind pools, so
// the per-kind assignment below can never run out of slots once the aggregate cap has passed.
static_assert(kMaxImageLayerPorts >= kMaxLayerPorts && kMaxTensorLayerPorts >= kMaxLayerPorts);

/// Input queue depth declared for every layer port; per-edge ConnectionOptions tune individual
/// edges (the class documentation recommends a depth-1 mailbox for the trigger edge).
constexpr std::uint32_t kLayerInputQueueDepth = 8U;

namespace {

/// Name of the captured-frame output port a CudaSinkDesc sink declares.
constexpr const char* kOutputPortName = "output";

}  // namespace

HolovizOp::HolovizOp(LayerStack layer_stack, SinkDesc sink, FrameTrigger trigger,
                     std::chrono::nanoseconds max_input_age)
    : layer_stack_(std::move(layer_stack)),
      sink_(std::move(sink)),
      trigger_(trigger),
      max_input_age_(max_input_age) {
  // Validate the temporal parameters before the layer count so a bad cadence fails immediately,
  // not after an otherwise-valid layer stack has already been accepted.
  if (max_input_age_ <= std::chrono::nanoseconds::zero()) {
    HOLOSCAN_LOG_ERROR("HolovizOp: max_input_age must be > 0 ns; got {} ns.",
                       max_input_age_.count());
    std::terminate();
  }
  if (const auto* clock = std::get_if<ClockTrigger>(&trigger_);
      clock != nullptr && clock->period <= std::chrono::nanoseconds::zero()) {
    HOLOSCAN_LOG_ERROR("HolovizOp: ClockTrigger period must be > 0 ns; got {} ns.",
                       clock->period.count());
    std::terminate();
  }

  // Validate compile-time port capacity at construction so the failure surfaces early,
  // not mid-setup() when individual port accesses start terminating.
  if (layer_stack_.layers().size() > kMaxLayerPorts) {
    HOLOSCAN_LOG_ERROR(
        "HolovizOp: layer count {} exceeds kMaxLayerPorts ({}); increase the "
        "compile-time limit or reduce the declared layer count.",
        layer_stack_.layers().size(),
        kMaxLayerPorts);
    std::terminate();
  }

  // The OnEach driver must be a declared layer; contract() would otherwise reference an unbound
  // port. An empty stack has nothing to drive rendering and needs a ClockTrigger.
  if (const auto* layer = std::get_if<LayerTrigger>(&trigger_);
      layer != nullptr && layer->layer >= layer_stack_.layers().size()) {
    HOLOSCAN_LOG_ERROR(
        "HolovizOp: LayerTrigger layer {} is out of range for {} declared layer(s); name a "
        "declared layer, or use ClockTrigger for a layer-less operator.",
        layer->layer,
        layer_stack_.layers().size());
    std::terminate();
  }

  // Assign each declared layer a slot in its kind's port pool (image or tensor), independent of
  // the layer's position in the declared LayerStack. This is deliberate: pinning slot assignment
  // to declaration order would force an app to reorder its layer list to satisfy the physical
  // port typing (image ports are Input<schema::ImageT>, tensor ports are Input<Tensor>). Each
  // pool holds kMaxLayerPorts slots (static_assert above), so the aggregate check is the only
  // capacity limit an application can hit.
  const auto& layers = layer_stack_.layers();
  std::size_t next_image_slot = 0;
  std::size_t next_tensor_slot = 0;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    pool_slot_for_layer_[i] = std::holds_alternative<ImageLayerDesc>(layers[i].desc)
                                  ? next_image_slot++
                                  : next_tensor_slot++;
  }

  // The CUDA sink binds a fixed-name output port; a layer input of the same name would make
  // the operator's port table ambiguous at compile time with a far less direct diagnostic.
  if (std::holds_alternative<CudaSinkDesc>(sink_)) {
    for (const LayerDecl& layer : layer_stack_.layers()) {
      if (layer.port_name == kOutputPortName) {
        HOLOSCAN_LOG_ERROR(
            "HolovizOp: layer port name '{}' collides with the CudaSinkDesc output port; rename "
            "the layer.",
            layer.port_name);
        std::terminate();
      }
    }
  }

  impl_ = std::make_unique<Impl>(*this);
}

HolovizOp::~HolovizOp() {
  // If stop()'s quiescence checks failed, the renderer, present backend, and device were kept
  // alive for the process lifetime. In that case the retained input tensors must be released
  // here while those objects are still valid: Impl declares retained_image_tensors_ before
  // render_/present_, so the default member destruction order would free the tensor backing
  // after the backends are gone. Parked CUDA-sink outputs are the opposite case — a resolve
  // that was never proven complete may still target them — so they are leaked deliberately.
  if (impl_ && impl_->terminal_failed_) {
    impl_->retained_image_tensors_ = {};
    if (!impl_->retained_outputs_.empty()) {
      // Process-wide, `new`ed and never deleted (see CudaPresentBackend::quarantine_owned_state);
      // independent sessions may destroy their operators concurrently, hence the mutex.
      static auto* quarantine_mutex = new std::mutex();
      static auto* leaked_outputs = new std::vector<std::unique_ptr<Impl::RetainedOutput>>();
      const std::lock_guard<std::mutex> lock(*quarantine_mutex);
      for (auto& retained : impl_->retained_outputs_) {
        leaked_outputs->push_back(std::move(retained));
      }
      impl_->retained_outputs_.clear();
    }
  }
}

// ---- graph-wiring accessors --------------------------------------------------

const holoscan::Input<holoscan::schema::ImageT>& HolovizOp::image_layer_input(
    std::size_t index) const noexcept {
  const auto& layers = layer_stack_.layers();
  if (index >= layers.size() || !std::holds_alternative<ImageLayerDesc>(layers[index].desc)) {
    std::terminate();  // out of range, or the declared layer at this index is not image-kind
  }
  return impl_->image_layer_input(pool_slot_for_layer_[index]);
}
holoscan::Input<holoscan::schema::ImageT>& HolovizOp::image_layer_input(
    std::size_t index) noexcept {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<holoscan::Input<holoscan::schema::ImageT>&>(
      static_cast<const HolovizOp*>(this)->image_layer_input(index));
}

const holoscan::Input<holoscan::Tensor>& HolovizOp::tensor_layer_input(
    std::size_t index) const noexcept {
  const auto& layers = layer_stack_.layers();
  if (index >= layers.size() || std::holds_alternative<ImageLayerDesc>(layers[index].desc)) {
    std::terminate();  // out of range, or the declared layer at this index is image-kind
  }
  return impl_->tensor_layer_input(pool_slot_for_layer_[index]);
}
holoscan::Input<holoscan::Tensor>& HolovizOp::tensor_layer_input(std::size_t index) noexcept {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<holoscan::Input<holoscan::Tensor>&>(
      static_cast<const HolovizOp*>(this)->tensor_layer_input(index));
}

// ---- setup() ----------------------------------------------------------------

void HolovizOp::setup(holoscan::OperatorSpec& spec) {
  const auto& layers = layer_stack_.layers();
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const std::size_t slot = pool_slot_for_layer_[i];
    std::visit(
        [&](auto& desc) {
          using T = std::decay_t<decltype(desc)>;
          if constexpr (std::is_same_v<T, ImageLayerDesc>) {
            // Unconstrained — no expects_tensor()-style call. A default-constructed
            // TensorInputSpec is rejected by Core as an empty contract; unconstrained ports
            // must omit the call entirely. Frame semantics (encoding/color_space) are
            // validated against desc in build_render_list(), not declared here.
            spec.input(impl_->image_layer_input(slot), layers[i].port_name)
                .queue_depth(kLayerInputQueueDepth);
          } else if constexpr (std::is_same_v<T, GeometryLayerDesc>) {
            // build_render_list() treats this Tensor as a flat float32 coordinate array
            // (rank 1, float32, host-resident). A mismatched producer is rejected by
            // Runtime contract enforcement before reaching data_as<float>().
            holoscan::TensorInputSpec ts{};
            ts.representation.memory_kind = holoscan::MemoryKind::kHost;
            ts.representation.dtype = DLDataType{kDLFloat, 32U, 1U};
            ts.representation.rank = 1U;
            spec.input(impl_->tensor_layer_input(slot), layers[i].port_name)
                .queue_depth(kLayerInputQueueDepth)
                .expects_tensor(ts);
          } else if constexpr (std::is_same_v<T, TextLayerDesc>) {
            // build_render_list() decodes this Tensor as fixed-width text records (rank 2,
            // uint8 {N, 16 + L}, host-resident; see TextLayerDesc). Row width and content are
            // validated per sample; dtype, rank, and residency by Runtime contract enforcement.
            holoscan::TensorInputSpec ts{};
            ts.representation.memory_kind = holoscan::MemoryKind::kHost;
            ts.representation.dtype = DLDataType{kDLUInt, 8U, 1U};
            ts.representation.rank = 2U;
            spec.input(impl_->tensor_layer_input(slot), layers[i].port_name)
                .queue_depth(kLayerInputQueueDepth)
                .expects_tensor(ts);
          } else {
            // LUTLayerDesc: unconstrained, same rationale as the image case.
            spec.input(impl_->tensor_layer_input(slot), layers[i].port_name)
                .queue_depth(kLayerInputQueueDepth);
          }
        },
        layers[i].desc);
  }

  // CudaSinkDesc: every presented frame leaves on "output" as one schema::ImageT whose data
  // Tensor comes from a Runtime pool (the deployment binds its device): the format's native
  // element type in {H, W} or {H, W, 4}, encoding/color_space per Impl::cuda_sink_wire(). The
  // display sinks declare no output. A malformed sink leaves the port unbound here so the
  // operator still constructs; start() then rejects the sink with kInvalidArgument.
  if (const auto* cuda = std::get_if<CudaSinkDesc>(&sink_)) {
    // Same gate as start(): the element table admits exactly the formats with a wire encoding,
    // and to_holoviz_format() the ones with a render-target lowering (RGB8 has neither).
    const auto element = Impl::cuda_sink_element(cuda->format);
    if (!element || !common::to_holoviz_format(cuda->format) || cuda->width == 0U ||
        cuda->height == 0U) {
      HOLOSCAN_LOG_ERROR(
          "HolovizOp: CudaSinkDesc needs a nonzero width/height and a packed single-plane "
          "format with a render-target lowering and a wire encoding; the output port is left "
          "unbound and start() will reject the sink.");
      return;
    }
    const std::array<std::int64_t, 3> max_shape{static_cast<std::int64_t>(cuda->height),
                                                static_cast<std::int64_t>(cuda->width),
                                                element->channels};
    spec.output(output_, kOutputPortName)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{.memory_kind = holoscan::MemoryKind::kCudaDevice,
                                               .dtype = element->dtype,
                                               .rank = element->rank},
            .bounds =
                holoscan::tensor_bounds(holoscan::TensorShapeView{max_shape.data(), element->rank}),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }
}

// ---- contract() -------------------------------------------------------------

holoscan::Contract HolovizOp::contract() const {
  holoscan::Contract result;

  // LayerTrigger: one activation per admitted sample on the trigger layer. OnEach pins that
  // sample by readiness, and the contract normalizer rejects both a selector and an on_missing
  // policy on the driver, so the loop below skips that port. ClockTrigger: a fixed cadence with
  // every layer sampled by freshness.
  std::optional<std::size_t> trigger_layer;
  std::visit(
      [&](const auto& trigger) {
        using T = std::decay_t<decltype(trigger)>;
        if constexpr (std::is_same_v<T, LayerTrigger>) {
          trigger_layer = trigger.layer;
          // The driver port's type follows the trigger layer's kind, so resolve it through the
          // same declared-index -> pool-slot map the loop below uses rather than assuming a
          // uniform port bank. The constructor has already rejected an out-of-range layer.
          const std::size_t slot = pool_slot_for_layer_[trigger.layer];
          if (std::holds_alternative<ImageLayerDesc>(layer_stack_.layers()[trigger.layer].desc)) {
            result.trigger(holoscan::OnEach{impl_->image_layer_input(slot)});
          } else {
            result.trigger(holoscan::OnEach{impl_->tensor_layer_input(slot)});
          }
        } else {
          result.trigger(holoscan::OnClock{.period = trigger.period});
        }
      },
      trigger_);

  const auto& layers = layer_stack_.layers();
  for (std::size_t i = 0; i < layers.size(); ++i) {
    if (trigger_layer == i) {
      continue;
    }
    const std::size_t slot = pool_slot_for_layer_[i];
    if (std::holds_alternative<ImageLayerDesc>(layers[i].desc)) {
      result.input(impl_->image_layer_input(slot),
                   holoscan::LatestWithin{.max_age = max_input_age_});
      // Image layers hold the last frame across activations where a producer slower than the
      // trigger has nothing new (see build_render_list()'s doc comment), but that hold must be
      // bounded — an indefinite freeze would let a stopped or failed camera keep rendering as if
      // it were still live. ReuseLast expresses that bound in the temporal contract itself
      // instead of a private, unbounded cache: is_missing() only becomes true once
      // max_input_age_ has actually elapsed since the last real sample, exactly like the
      // LatestWithin bound.
      //
      // Under a ClockTrigger this bound also decides when a stalled operator stops activating:
      // an exhausted ReuseLast yields kAbsent, and with no SafeDefault sibling (e.g. a lone
      // image layer) the runtime aggregates that to kNoFire and skips compute(), including the
      // window-event poll inside it. The header therefore does not promise event servicing
      // through a producer stall in clock mode.
      result.on_missing(impl_->image_layer_input(slot),
                        holoscan::ReuseLast{.max_age = max_input_age_});
    } else {
      result.input(impl_->tensor_layer_input(slot),
                   holoscan::LatestWithin{.max_age = max_input_age_});
      // Geometry (and future text/LUT) layers are skipped on a missing activation — no
      // implicit hold-last for these layer kinds.
      result.on_missing(impl_->tensor_layer_input(slot), holoscan::SafeDefault{});
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
