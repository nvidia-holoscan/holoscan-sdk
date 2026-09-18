// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/backends/holoviz_render_backend.hpp"
#include "detail/destroy_timeout.hpp"
#include "detail/device_impl.hpp"
#include "detail/render_pass_impl.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <holoviz/adapter.hpp>
#include <holoviz/logger.hpp>
#include "holoviz_op/types.hpp"

#include "common/holoviz_conversions.hpp"
#include "cuda/cu_errors.hpp"
#include "cuda/cu_scoped_context.hpp"
#include "holoviz_op/backends/backend_tokens.hpp"

namespace holoscan::visualization::backends {

namespace {

/// Headless targets have a fixed format: nothing above the seam names a HoloViz format for the
/// offscreen path, and RGBA8 is what the readback-based tests and pitch-linear sinks expect.
constexpr holoviz::Format kHeadlessFormat = holoviz::Format::kR8G8B8A8Unorm;

using holoviz::ScopedCudaContext;

/// Bounded replacement for cuStreamSynchronize: poll `event` (already recorded on the
/// stream to drain) until it retires or `timeout` elapses. kBackpressured on timeout —
/// nothing the event covers has been released, so the wait is retryable. The caller holds
/// the CUDA context.
expected<void> wait_event_bounded(CUevent event, std::chrono::nanoseconds timeout) {
  const auto now = std::chrono::steady_clock::now();
  const bool unbounded = timeout > std::chrono::steady_clock::time_point::max() - now;
  const auto deadline = unbounded ? std::chrono::steady_clock::time_point::max() : now + timeout;
  while (true) {
    const CUresult query = cuEventQuery(event);
    if (query == CUDA_SUCCESS) {
      return {};
    }
    if (query != CUDA_ERROR_NOT_READY) {
      VIZ_CU_FORWARD_ERROR(query);
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return make_unexpected(
          Error{.code = ErrorCode::kBackpressured,
                .message = "[holoviz] Timed out waiting for enqueued CUDA uploads to retire."});
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

}  // namespace

HolovizRenderBackend::~HolovizRenderBackend() {
  const expected<void> result = destroy(holoviz::detail::kDestructorDestroyTimeout);
  if (result) {
    return;
  }
  if (stage_ != Stage::kDestroyed) {
    // A Phase A failure: the bounded drain timed out with pending work still referencing
    // every owned resource, and destroy() correctly returned with everything intact. In
    // the destructor there is no retry, and letting member destruction run would free
    // those resources anyway (the per-handle destructors and vk_buffer/vk_image destroy
    // unconditionally) — a use-after-free, not a leak. Quarantine instead: deliberate
    // terminal retention, with the device poisoned so its teardown leaks too.
    quarantine_owned_state();
    try {
      HOLOVIZ_LOG_ERROR(
          "[holoviz] HolovizRenderBackend: destroy timed out in the destructor ({}); "
          "quarantining the owned GPU/CUDA state instead of freeing resources that pending "
          "work still references. The device will be leaked at its teardown.",
          result.error().message);
    } catch (...) {
    }
    return;
  }
  // A Phase B partial failure: teardown ran to completion, best-effort; log only.
  try {
    HOLOVIZ_LOG_ERROR("[holoviz] HolovizRenderBackend: destroy failed in destructor: {}",
                      result.error().message);
  } catch (...) {
  }
}

void HolovizRenderBackend::quarantine_owned_state() noexcept {
  // Process-lifetime quarantine list: `new`ed and never deleted, so no destructor ever
  // runs on the quarantined handles (their own destructors would retry destroy() and then
  // free unconditionally) and the memory stays reachable rather than lost. Static-local so
  // static-destruction order cannot touch it either. The list is process-wide while each
  // backend is single-threaded, and independent backend/device pairs on different threads
  // may time out concurrently — the (equally leaky, for the same destruction-order reason)
  // mutex serializes their push_backs. Allocation policy: this path allocates inside
  // noexcept by design — if even the bundle allocation fails here, terminating is the
  // deliberate outcome, because every fallback frees resources that pending GPU/CUDA work
  // still references (a use-after-free, strictly worse than abort()).
  static auto* quarantine_mutex = new std::mutex();
  static auto* quarantine = new std::vector<std::unique_ptr<QuarantinedState>>();
  auto bundle = std::make_unique<QuarantinedState>();
  bundle->slots = std::move(slots_);
  bundle->render_pass = std::move(render_pass_);
  bundle->layer_states = std::move(layer_states_);
  bundle->upload_semaphore = std::move(upload_semaphore_);
  bundle->drain_event = drain_event_;
  drain_event_ = nullptr;
  if (owns_stream_) {
    bundle->stream = stream_;
    owns_stream_ = false;
  }
  stream_ = nullptr;  // borrowed streams stay the caller's; ours rides the bundle
  {
    const std::lock_guard<std::mutex> lock(*quarantine_mutex);
    quarantine->push_back(std::move(bundle));
  }
  stage_ = Stage::kDestroyed;
  if (device_ != nullptr) {
    device_->impl().report_leaked_resources();
  }
}

expected<void> HolovizRenderBackend::destroy(std::chrono::nanoseconds timeout) {
  if (stage_ == Stage::kDestroyed || device_ == nullptr) {
    stage_ = Stage::kDestroyed;
    return {};
  }
  // Phase A — bounded waits on backend-owned work only, before anything is freed. A
  // kBackpressured here returns with every resource intact, so destroy() is retryable.
  // The drain event covers CUDA uploads record() enqueued but never submitted (which a
  // device wait cannot see) on owned and borrowed streams alike; the per-slot reset
  // waits each slot's last graphics submit. Together they cover exactly this backend's
  // outstanding work without waiting on unrelated users of a borrowed device or stream.
  if (stream_ != nullptr && drain_event_ != nullptr) {
    ScopedCudaContext scoped_context(cuda_context_);
    VIZ_CU_FORWARD_ERROR(cuEventRecord(drain_event_, stream_));
    VIZ_FORWARD_ERROR(wait_event_bounded(drain_event_, timeout));
  }
  for (FrameSlot& slot : slots_) {
    if (slot.cmd) {
      // Discards any open recording and waits the slot's last submit (bounded).
      VIZ_FORWARD_ERROR(slot.cmd.reset(timeout));
    }
  }

  // Phase B — free everything, accumulating the first error (best effort).
  expected<void> result;
  for (FrameSlot& slot : slots_) {
    if (slot.cmd) {
      if (expected<void> r = slot.cmd.destroy(timeout); !r && result) {
        result = r;
      }
    }
    if (slot.headless_target) {
      if (expected<void> r = slot.headless_target.destroy(timeout); !r && result) {
        result = r;
      }
    }
  }
  slots_.clear();
  if (render_pass_) {
    if (expected<void> r = render_pass_.destroy(timeout); !r && result) {
      result = r;
    }
  }
  {
    ScopedCudaContext scoped_context(cuda_context_);
    for (LayerState& layer_state : layer_states_) {
      expected<void> r = std::visit(
          [&](auto& state) -> expected<void> {
            using State = std::decay_t<decltype(state)>;
            if constexpr (std::is_same_v<State, HolovizTextLayerState>) {
              return state.renderer.destroy(timeout);
            } else if constexpr (std::is_same_v<State, HolovizGeometryLayerState>) {
              return destroy_holoviz_geometry_layer(state, timeout);
            } else if constexpr (std::is_same_v<State, HolovizImageLayerState>) {
              return destroy_holoviz_image_layer(state, timeout);
            } else if constexpr (std::is_same_v<State, HolovizLUTLayerState>) {
              return destroy_holoviz_lut_layer(state, timeout);
            }
          },
          layer_state);
      if (!r && result) {
        result = r;
      }
    }
    layer_states_.clear();

    if (upload_semaphore_) {
      if (expected<void> r = upload_semaphore_.destroy(timeout); !r && result) {
        result = r;
      }
    }
    if (drain_event_ != nullptr) {
      VIZ_CU_ACCUMULATE_ERROR(result, cuEventDestroy(drain_event_));
      drain_event_ = nullptr;
    }
    if (owns_stream_ && stream_ != nullptr) {
      // Phase A already drained the stream; cuStreamDestroy defers actual destruction
      // until any residual work completes, so no synchronize is needed here.
      VIZ_CU_ACCUMULATE_ERROR(result, cuStreamDestroy(stream_));
      stream_ = nullptr;
      owns_stream_ = false;
    }
  }
  stage_ = Stage::kDestroyed;
  return result;
}

expected<void> HolovizRenderBackend::allocate(const LayerStack& layers,
                                              const RenderConfig& config) {
  if (stage_ != Stage::kCreated) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] allocate() must be called exactly once, before build() "
                         "(a failed allocate() is terminal — create a fresh backend)."});
  }
  // Precondition checks mutate nothing: these argument errors stay retryable.
  if (config.max_frames_in_flight == 0) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] max_frames_in_flight must be at least 1."});
  }
  if (config.headless && (config.headless_width == 0 || config.headless_height == 0)) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Headless mode requires headless_width/headless_height."});
  }

  expected<void> allocated = allocate_impl(layers, config);
  if (!allocated) {
    // allocate_impl() appends to the owned-state members as it goes, so the failure left
    // partial state behind (earlier layers' pools, the owned stream/semaphore, some frame
    // slots). Retrying over those leftovers would append a second copy and misalign
    // layer_states_ with the declared bundle — record() indexes it by declaration order
    // and casts erased frame data by the (then wrong) alternative. Make the failure
    // terminal instead: release everything through the normal teardown (nothing has been
    // submitted, so the bounded drain is immediate) and refuse further use.
    if (expected<void> cleanup = destroy(std::chrono::nanoseconds(config.backpressure_timeout_ns));
        !cleanup) {
      HOLOVIZ_LOG_ERROR(
          "[holoviz] HolovizRenderBackend: releasing partial state after a failed allocate() "
          "failed too: {}",
          cleanup.error().message);
    }
    stage_ = Stage::kDestroyed;  // terminal even when the cleanup itself failed (leak > UAF)
    return allocated;
  }
  stage_ = Stage::kAllocated;
  return {};
}

expected<void> HolovizRenderBackend::allocate_impl(const LayerStack& layers,
                                                   const RenderConfig& config) {
  config_ = config;
  declared_layer_count_ = static_cast<uint32_t>(layers.layers().size());
  // Pre-size the composite scratch here so record()'s same-size assign() never
  // reallocates — not even on the first frame.
  composite_seen_.assign(declared_layer_count_, 0);

  // The CUDA upload path exists only when a declared layer stages frame data through
  // device memory; a pure-text scene allocates no stream and no semaphore.
  bool needs_uploads = false;
  for (const LayerDecl& decl : layers.layers()) {
    needs_uploads |= !std::holds_alternative<TextLayerDesc>(decl.desc);
  }
  if (needs_uploads) {
    VIZ_UNWRAP(CUcontext, cuda_context, device_->cuda_context());
    cuda_context_ = cuda_context;
    // One scope across stream creation AND create_external_semaphore(): the semaphore
    // import (cuImportExternalSemaphore) binds to the current context, so it must run
    // under the device's context regardless of the caller's ambient state — including
    // the borrowed-stream branch, where nothing else would push one.
    ScopedCudaContext scoped_context(cuda_context_);
    if (desc_.stream != nullptr) {
      stream_ = desc_.stream;
    } else {
      // CU_STREAM_DEFAULT, not CU_STREAM_NON_BLOCKING: on at least one driver
      // (595.71.05 open kernel module, RTX A4500, hybrid iGPU+dGPU Linux), async copies
      // into Vulkan-imported mipmapped arrays enqueued on a NON_BLOCKING stream land only
      // partially (~13 KB prefix) even though cuStreamSynchronize() reports the stream
      // idle — the sampled image then shows a horizontal cut of stale texels. The same
      // copies on a DEFAULT-flag stream (or the legacy null stream) are byte-exact. The
      // cost is legacy-null-stream serialization, which callers avoid by passing their
      // own stream in HolovizRenderBackendDesc (whose stream doc mirrors this trade-off).
      VIZ_CU_FORWARD_ERROR(cuStreamCreate(&stream_, CU_STREAM_DEFAULT));
      owns_stream_ = true;
    }
    VIZ_UNWRAP(holoviz::ExternalSemaphore, semaphore, device_->create_external_semaphore(stream_));
    upload_semaphore_ = std::move(semaphore);
    VIZ_CU_FORWARD_ERROR(cuEventCreate(&drain_event_, CU_EVENT_DISABLE_TIMING));
  }

  // Per-declared-layer GPU state, index-aligned with the bundle. Allocation is
  // all-or-nothing: a layer kind this backend cannot serve yet fails the whole allocate()
  // instead of leaving a hole the frame path would have to skip around.
  layer_states_.reserve(layers.layers().size());
  for (const LayerDecl& decl : layers.layers()) {
    expected<LayerState> allocated = std::visit(
        [&](const auto& spec) -> expected<LayerState> {
          using Spec = std::decay_t<decltype(spec)>;
          if constexpr (std::is_same_v<Spec, TextLayerDesc>) {
            VIZ_UNWRAP(HolovizTextLayerState, state, allocate_holoviz_text_layer(*device_, spec));
            return LayerState{std::move(state)};
          } else if constexpr (std::is_same_v<Spec, GeometryLayerDesc>) {
            // The pool buffer's CUDA export import needs the device context current, like
            // the image/LUT branches below.
            ScopedCudaContext scoped_context(cuda_context_);
            VIZ_UNWRAP(
                HolovizGeometryLayerState,
                state,
                allocate_holoviz_geometry_layer(*device_, spec, config.max_frames_in_flight));
            stats_.geometry_pool_bytes +=
                static_cast<size_t>(state.slot_stride_bytes) * config.max_frames_in_flight;
            return LayerState{std::move(state)};
          } else if constexpr (std::is_same_v<Spec, ImageLayerDesc>) {
            ScopedCudaContext scoped_context(cuda_context_);
            VIZ_UNWRAP(HolovizImageLayerState,
                       state,
                       allocate_holoviz_image_layer(*device_, spec, config.max_frames_in_flight));
            stats_.texture_pool_bytes += static_cast<size_t>(spec.max_width) * spec.max_height *
                                         state.bytes_per_pixel * config.max_frames_in_flight;
            return LayerState{std::move(state)};
          } else if constexpr (std::is_same_v<Spec, LUTLayerDesc>) {
            ScopedCudaContext scoped_context(cuda_context_);
            VIZ_UNWRAP(HolovizLUTLayerState,
                       state,
                       allocate_holoviz_lut_layer(*device_, spec, config.max_frames_in_flight));
            stats_.texture_pool_bytes +=
                static_cast<size_t>(spec.max_width) * spec.max_height * config.max_frames_in_flight;
            return LayerState{std::move(state)};
          }
        },
        decl.desc);
    if (!allocated) {
      return forward_error(allocated);
    }
    layer_states_.push_back(std::move(*allocated));
  }

  // The frame-slot ring: one transient command list (and, headless, one offscreen target)
  // per slot. This is the last allocation the frame path ever sees.
  slots_.reserve(config.max_frames_in_flight);
  for (uint32_t i = 0; i < config.max_frames_in_flight; ++i) {
    FrameSlot slot;
    auto cmd = device_->create_command_list(holoviz::CommandListDesc{
        .type = holoviz::CommandListType::kTransient,
        .debug_name = "holoviz render backend frame slot",
    });
    if (!cmd) {
      return forward_error(cmd);
    }
    slot.cmd = std::move(*cmd);
    if (config.headless) {
      auto target = device_->create_image(holoviz::ImageDesc{
          .extent = {config.headless_width, config.headless_height},
          .format = kHeadlessFormat,
          .usage = holoviz::ImageUsage::kRenderTarget,
          .debug_name = "holoviz render backend headless target",
      });
      if (!target) {
        return forward_error(target);
      }
      slot.headless_target = std::move(*target);
      stats_.texture_pool_bytes +=
          static_cast<size_t>(config.headless_width) * config.headless_height * 4;
    }
    slot.state = std::make_shared<FrameState>();
    slot.state->slot = i;
    slot.state->waits.reserve(layers.layers().size() + 1);
    slot.submit_payload = std::make_shared<SubmitPayload>(device_identity(*device_));
    slot.tickets = {std::make_shared<FrameTicket>(), std::make_shared<FrameTicket>()};
    slots_.push_back(std::move(slot));
  }

  return {};
}

expected<void> HolovizRenderBackend::build(ImageFormat output_format) {
  if (stage_ != Stage::kAllocated) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] build() must be called exactly once, after allocate()."});
  }

  holoviz::Format target_format = kHeadlessFormat;
  if (!config_.headless) {
    VIZ_UNWRAP_NODECL(holoviz::Format, target_format, common::to_holoviz_format(output_format));
    VIZ_UNWRAP(holoviz::CapabilityReport, device_caps, device_->capabilities());
    if (!holoviz::has_flag(device_caps.format_capabilities(target_format),
                           holoviz::FormatCapability::kRenderTarget)) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message =
                                       "[holoviz] The device cannot render to the requested output "
                                       "format."});
    }
  }

  VIZ_UNWRAP_NODECL(holoviz::RenderPass,
                    render_pass_,
                    device_->create_render_pass(holoviz::RenderPassDesc{
                        .render_targets = {{.format = target_format}},
                    }));

  // All pipeline compilation happens here (the build() contract): compile the layer
  // pipelines and pre-bake their descriptors.
  for (LayerState& layer_state : layer_states_) {
    expected<void> built = std::visit(
        [&](auto& state) -> expected<void> {
          using State = std::decay_t<decltype(state)>;
          if constexpr (std::is_same_v<State, HolovizTextLayerState>) {
            return {};  // the text renderer created its pipeline at allocate()
          } else if constexpr (std::is_same_v<State, HolovizGeometryLayerState>) {
            return build_holoviz_geometry_layer(*device_, state, config_.max_frames_in_flight);
          } else if constexpr (std::is_same_v<State, HolovizImageLayerState>) {
            return build_holoviz_image_layer(*device_, state, config_.max_frames_in_flight);
          } else if constexpr (std::is_same_v<State, HolovizLUTLayerState>) {
            return build_holoviz_lut_layer(*device_, state, config_.max_frames_in_flight);
          }
        },
        layer_state);
    VIZ_FORWARD_ERROR(std::move(built));
  }

  // Warm every lazily-compiled pipeline x pass pair now, so the first real frame records
  // against already-compiled variants (HoloViz compiles a pass-specific pipeline on the
  // first bind_pipeline inside that pass — see command_list.hpp). The warm list is recorded
  // and discarded, never submitted; that is enough to drive the compilation. The same list
  // then pre-creates the framebuffer of every real headless target (get_framebuffer caches
  // per target inside begin_render), keeping begin_frame() on its no-allocation contract.
  const bool has_warmable_layer = !layer_states_.empty();
  if (has_warmable_layer || config_.headless) {
    auto warm_cmd = device_->create_command_list(holoviz::CommandListDesc{
        .type = holoviz::CommandListType::kTransient,
        .debug_name = "holoviz render backend pipeline warm list",
    });
    if (!warm_cmd) {
      return forward_error(warm_cmd);
    }
    VIZ_FORWARD_ERROR(warm_cmd->begin());
    holoviz::Image warm_image;
    if (has_warmable_layer) {
      constexpr holoviz::Extent2D kWarmExtent{64, 64};
      auto created = device_->create_image(holoviz::ImageDesc{
          .extent = kWarmExtent,
          .format = target_format,
          .usage = holoviz::ImageUsage::kRenderTarget,
          .debug_name = "holoviz render backend pipeline warm target",
      });
      if (!created) {
        return forward_error(created);
      }
      warm_image = std::move(*created);
      VIZ_FORWARD_ERROR(render_pass_.set_render_image(warm_image,
                                                      0,
                                                      {config_.clear_color[0],
                                                       config_.clear_color[1],
                                                       config_.clear_color[2],
                                                       config_.clear_color[3]}));
      VIZ_FORWARD_ERROR(warm_cmd->begin_render(
          render_pass_, holoviz::Rect2D{0, 0, kWarmExtent.width, kWarmExtent.height}));
      for (LayerState& layer_state : layer_states_) {
        expected<void> warmed = std::visit(
            [&](auto& state) -> expected<void> {
              using State = std::decay_t<decltype(state)>;
              if constexpr (std::is_same_v<State, HolovizTextLayerState>) {
                // One throwaway glyph run drives bind_pipeline against this pass.
                // Slot 0 with a default token: the warm list is never submitted, so no
                // slot has a prior use to gate on.
                VIZ_FORWARD_ERROR(state.renderer.begin_frame(kWarmExtent, 0, {}));
                VIZ_FORWARD_ERROR(state.renderer.add_text({.position = {0.0F, 0.0F}}, " "));
                VIZ_FORWARD_ERROR(state.renderer.end_frame());
                return state.renderer.record(*warm_cmd, kWarmExtent);
              } else if constexpr (std::is_same_v<State, HolovizGeometryLayerState>) {
                return warm_holoviz_geometry_layer(state, *warm_cmd);
              } else if constexpr (std::is_same_v<State, HolovizImageLayerState>) {
                return warm_holoviz_image_layer(state, *warm_cmd);
              } else if constexpr (std::is_same_v<State, HolovizLUTLayerState>) {
                return warm_holoviz_lut_layer(state, *warm_cmd);
              }
            },
            layer_state);
        VIZ_FORWARD_ERROR(std::move(warmed));
      }
      VIZ_FORWARD_ERROR(warm_cmd->end_render());
    }
    VIZ_FORWARD_ERROR(warm_cmd->end());
    VIZ_FORWARD_ERROR(warm_cmd->destroy());
    if (warm_image) {
      VIZ_FORWARD_ERROR(warm_image.destroy());
    }
  }

  if (config_.headless) {
    // Pre-create every real headless target's framebuffer on the setup path, keeping
    // begin_frame() on its no-allocation contract from the first frame. Windowed targets
    // arrive later, through register_targets().
    for (FrameSlot& slot : slots_) {
      VIZ_FORWARD_ERROR(render_pass_.prepare_render_target(slot.headless_target));
    }
  }

  stage_ = Stage::kBuilt;
  return {};
}

expected<void> HolovizRenderBackend::register_targets(const SyncToken& targets) {
  if (stage_ != Stage::kBuilt) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] register_targets() before build()."});
  }
  if (config_.headless) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Headless backends own their targets (pre-created at "
                         "build()); there is nothing to register."});
  }
  const TargetSetLookup lookup = target_set_payload(targets, device_identity(*device_));
  if (lookup.payload == nullptr) {
    if (lookup.check == TokenRegistry::Check::kWrongDevice) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] The target set was minted by a PresentBackend on a "
                           "different device."});
    }
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] register_targets() needs the token from this module's "
                         "PresentBackend::targets()."});
  }
  const std::shared_ptr<const AcquireSource> source = lookup.payload->source.lock();
  if (source == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] The target set outlived its PresentBackend/swapchain."});
  }
  for (const AcquireSource::Target& target : source->targets) {
    if (target.image != nullptr) {
      VIZ_FORWARD_ERROR(render_pass_.prepare_render_target(*target.image));
    }
  }
  // The strong source reference is dropped on return: the backend never pins the
  // producer's liveness cell, it only remembers which generation is warm.
  warmed_generation_ = source->current_generation;
  return {};
}

uint64_t HolovizRenderBackend::render_targets_created_for_test() const {
  return render_pass_ ? render_pass_.impl().render_targets_created() : 0;
}

expected<HolovizRenderBackend::TicketLookup> HolovizRenderBackend::find_ticket(
    const FrameCommands& cmds) const {
  // Validate by pointer identity against this backend's ticket registry: a foreign
  // shared_ptr<void> payload (another backend instance, another type entirely) is compared
  // by address and never dereferenced, and the registry hit pins the slot index in range.
  const void* raw = cmds.impl.get();
  if (raw != nullptr) {
    for (const FrameSlot& slot : slots_) {
      for (const std::shared_ptr<FrameTicket>& ticket : slot.tickets) {
        if (static_cast<const void*>(ticket.get()) == raw) {
          return TicketLookup{ticket.get(), ticket->epoch != slot.current_epoch};
        }
      }
    }
  }
  return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                               .message =
                                   "[holoviz] FrameCommands was not produced by this backend's "
                                   "begin_frame()."});
}

expected<HolovizRenderBackend::FrameState*> HolovizRenderBackend::frame_state(
    const FrameCommands& cmds) const {
  VIZ_UNWRAP(TicketLookup, lookup, find_ticket(cmds));
  if (lookup.stale) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message =
                                     "[holoviz] FrameCommands is stale: a later begin_frame() "
                                     "re-issued this frame slot."});
  }
  return lookup.ticket->state.get();
}

expected<FrameCommands> HolovizRenderBackend::begin_frame(const AcquireToken& acquired,
                                                          uint32_t frame_index) {
  if (stage_ != Stage::kBuilt) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] begin_frame() before build()."});
  }
  if (frame_index >= slots_.size()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] frame_index exceeds RenderConfig::frames_in_flight."});
  }
  FrameSlot& slot = slots_[frame_index];
  if (slot.state->open) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] begin_frame() called twice on one slot without submit()."});
  }

  const holoviz::Image* target_image = nullptr;
  holoviz::Extent2D extent{};
  if (config_.headless) {
    if (acquired.image_ready.impl != nullptr) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message =
                                       "[holoviz] Headless backends render to their own targets; "
                                       "AcquireToken must be null."});
    }
    target_image = &slot.headless_target;
    extent = {config_.headless_width, config_.headless_height};
  } else {
    const AcquireLookup lookup = acquire_payload(acquired.image_ready, device_identity(*device_));
    if (lookup.check == TokenRegistry::Check::kWrongDevice) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message =
                                       "[holoviz] AcquireToken was minted by a PresentBackend on a "
                                       "different device."});
    }
    const AcquirePayload* payload = lookup.payload;
    if (payload == nullptr || payload->image == nullptr) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message =
                                       "[holoviz] AcquireToken must carry the AcquireToken from "
                                       "this module's PresentBackend."});
    }
    // The borrowed image pointer and extent are only valid for the swapchain generation
    // that minted the payload; validate liveness and generation before dereferencing
    // anything (a recreation may have destroyed the image wrapper outright).
    const std::shared_ptr<const AcquireSource> source = payload->source.lock();
    if (source == nullptr) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidState,
                .message = "[holoviz] AcquireToken outlived its PresentBackend/swapchain."});
    }
    if (payload->generation != source->current_generation) {
      return make_unexpected(
          Error{.code = ErrorCode::kFrameDropped,
                .message = "[holoviz] AcquireToken predates a swapchain recreation; re-acquire."});
    }
    if (payload->generation != warmed_generation_) {
      // One-frame cold path after a swapchain recreation (or when register_targets() was
      // never called): re-create every current target's framebuffer now — the recreation
      // itself already allocated, and steady state is this single integer compare.
      for (const AcquireSource::Target& target : source->targets) {
        if (target.image != nullptr) {
          VIZ_FORWARD_ERROR(render_pass_.prepare_render_target(*target.image));
        }
      }
      warmed_generation_ = payload->generation;
    }
    target_image = payload->image;
    extent = payload->extent;
  }

  // Ticket-exhaustion check before the FIRST slot mutation (and after token validation, so
  // every rejected call leaves the slot fully untouched): this issuance would re-stamp the
  // ticket from two issuances ago, and if the caller still retains that handle, minting
  // would either alias it or allocate on the frame path. Refuse instead — retryable after
  // the caller releases the older handle, epoch and slot state unchanged.
  const std::shared_ptr<FrameTicket>& next_ticket =
      slot.tickets[(slot.current_epoch + 1) % slot.tickets.size()];
  if (next_ticket.use_count() > 1) {
    return make_unexpected(
        Error{.code = ErrorCode::kBackpressured,
              .message = "[holoviz] Both FrameCommands handles for this slot are still "
                         "retained; release the handle from the earlier issuance and retry "
                         "begin_frame()."});
  }

  // The frame path's one bounded block: recycling this slot's command list waits for the
  // slot's previous submit (kBackpressured on timeout — drop or retry, the caller's policy).
  VIZ_FORWARD_ERROR(slot.cmd.reset(std::chrono::nanoseconds(config_.backpressure_timeout_ns)));

  VIZ_FORWARD_ERROR(render_pass_.set_render_image(*target_image,
                                                  0,
                                                  {config_.clear_color[0],
                                                   config_.clear_color[1],
                                                   config_.clear_color[2],
                                                   config_.clear_color[3]}));

  VIZ_FORWARD_ERROR(slot.cmd.begin());
  VIZ_FORWARD_ERROR(
      slot.cmd.begin_render(render_pass_, holoviz::Rect2D{0, 0, extent.width, extent.height}));
  VIZ_FORWARD_ERROR(slot.cmd.set_viewport(holoviz::Rect2D{0, 0, extent.width, extent.height}));
  VIZ_FORWARD_ERROR(slot.cmd.set_scissor(holoviz::Rect2D{0, 0, extent.width, extent.height}));

  slot.state->target = target_image;
  slot.state->extent = extent;
  slot.state->acquire = acquired.image_ready.impl;
  slot.state->waits.clear();
  slot.state->open = true;

  // Mint this issuance's ticket. Reuse alternates between the slot's two preallocated
  // tickets; the exhaustion check above guaranteed exclusivity, so the re-stamp can never
  // alias a retained handle and begin_frame() performs no allocation.
  const uint64_t epoch = ++slot.current_epoch;
  std::shared_ptr<FrameTicket>& ticket = slot.tickets[epoch % slot.tickets.size()];
  ticket->slot = frame_index;
  ticket->epoch = epoch;
  ticket->state = slot.state;
  return FrameCommands{ticket};
}

expected<void> HolovizRenderBackend::record(FrameCommands& cmds, const RenderList& layers,
                                            const CompositeSpec& composite) {
  VIZ_UNWRAP(FrameState*, state, frame_state(cmds));
  if (!state->open) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] record() outside begin_frame()/submit()."});
  }

  // Bottom-most first (painter's algorithm); default is declaration order.
  const auto layer_frame_for = [&layers](uint32_t layer_index) -> const LayerFrame* {
    for (const LayerFrame& frame : layers.layers) {
      if (frame.layer_index == layer_index) {
        return &frame;
      }
    }
    return nullptr;
  };

  const size_t draw_count = composite.order.empty() ? layers.layers.size() : composite.order.size();
  // Reused across frames (assign, not construct) — record() is on the no-allocation path
  // after the first frame sizes it.
  composite_seen_.assign(declared_layer_count_, 0);
  for (size_t i = 0; i < draw_count; ++i) {
    const uint32_t layer_index =
        composite.order.empty() ? layers.layers[i].layer_index : composite.order[i];
    if (layer_index >= declared_layer_count_) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] RenderList/CompositeSpec references a layer index "
                           "that was never declared."});
    }
    if (composite_seen_[layer_index] != 0) {
      // A duplicate would record the same layer twice in one frame — for the CUDA-upload
      // layers that is a second async copy into a slot whose first copy is still pending.
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] RenderList/CompositeSpec references the same layer "
                           "index twice."});
    }
    composite_seen_[layer_index] = 1;
    const LayerFrame* frame = layer_frame_for(layer_index);
    if (frame == nullptr || frame->frame_data == nullptr) {
      continue;  // a declared layer with no data this cycle simply draws nothing
    }

    FrameSlot& slot = slots_[state->slot];
    expected<void> recorded = std::visit(
        [&](auto& layer_state) -> expected<void> {
          using State = std::decay_t<decltype(layer_state)>;
          if constexpr (std::is_same_v<State, HolovizTextLayerState>) {
            const auto* data = static_cast<const TextFrameData*>(frame->frame_data);
            VIZ_FORWARD_ERROR(record_holoviz_text_layer(layer_state,
                                                        slot.cmd,
                                                        state->extent,
                                                        state->slot,
                                                        slot.last_submit,
                                                        *data,
                                                        state->waits));
            stats_.text_glyph_high_water =
                std::max(stats_.text_glyph_high_water, layer_state.glyph_high_water);
            return {};
          } else if constexpr (std::is_same_v<State, HolovizGeometryLayerState>) {
            const auto* data = static_cast<const GeometryFrameData*>(frame->frame_data);
            ScopedCudaContext scoped_context(cuda_context_);
            VIZ_FORWARD_ERROR(record_holoviz_geometry_layer(layer_state,
                                                            slot.cmd,
                                                            state->extent,
                                                            state->slot,
                                                            slot.last_submit,
                                                            *data,
                                                            stream_,
                                                            upload_semaphore_,
                                                            state->waits));
            return {};
          } else if constexpr (std::is_same_v<State, HolovizImageLayerState>) {
            const auto* data = static_cast<const ImageFrameData*>(frame->frame_data);
            ScopedCudaContext scoped_context(cuda_context_);
            VIZ_FORWARD_ERROR(record_holoviz_image_layer(layer_state,
                                                         slot.cmd,
                                                         state->slot,
                                                         slot.last_submit,
                                                         *data,
                                                         stream_,
                                                         upload_semaphore_,
                                                         state->waits));
            return {};
          } else if constexpr (std::is_same_v<State, HolovizLUTLayerState>) {
            const auto* data = static_cast<const LutFrameData*>(frame->frame_data);
            ScopedCudaContext scoped_context(cuda_context_);
            VIZ_FORWARD_ERROR(record_holoviz_lut_layer(layer_state,
                                                       slot.cmd,
                                                       state->slot,
                                                       slot.last_submit,
                                                       *data,
                                                       stream_,
                                                       upload_semaphore_,
                                                       state->waits));
            return {};
          }
        },
        layer_states_[layer_index]);
    VIZ_FORWARD_ERROR(std::move(recorded));

    // The text renderer's record() clobbers viewport/scissor; restore the frame's.
    VIZ_FORWARD_ERROR(
        slot.cmd.set_viewport(holoviz::Rect2D{0, 0, state->extent.width, state->extent.height}));
    VIZ_FORWARD_ERROR(
        slot.cmd.set_scissor(holoviz::Rect2D{0, 0, state->extent.width, state->extent.height}));
  }

  return {};
}

expected<SyncToken> HolovizRenderBackend::submit(const FrameCommands& cmds,
                                                 const SubmitContext& ctx) {
  VIZ_UNWRAP(FrameState*, state, frame_state(cmds));
  if (!state->open) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] submit() without begin_frame()."});
  }
  if (ctx.frame_index != state->slot) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] SubmitContext::frame_index does not match the slot these "
                         "FrameCommands were begun on."});
  }
  FrameSlot& slot = slots_[state->slot];

  VIZ_FORWARD_ERROR(slot.cmd.end_render());
  VIZ_FORWARD_ERROR(slot.cmd.end());

  holoviz::SubmitDesc submit_desc{
      .wait_submit_sync_tokens = {&slot.last_submit, 1},  // {0} is skipped by the device
      .wait_external_sync_tokens = state->waits,
      .command_lists = {&slot.cmd, 1},
  };
  // Defensive re-validation of the token begin_frame() already vetted: one registry find
  // (~100 ns) buys "no cast without a live registry hit" as a local invariant instead of
  // an action-at-a-distance argument. Cache the vetted pointer in FrameState instead if
  // this ever shows up in a profile.
  AcquirePayload* acquire =
      acquire_payload(SyncToken{state->acquire}, device_identity(*device_)).payload;
  if (acquire != nullptr) {
    submit_desc.acquire_gates = {&acquire->token.acquire_gate, 1};
    submit_desc.present_gates = {&acquire->token.present_gate, 1};
  }

  VIZ_UNWRAP(holoviz::SubmitSyncToken, token, device_->submit(submit_desc));

  slot.last_submit = token;
  slot.submit_payload->token = token;
  state->open = false;
  state->acquire.reset();
  state->target = nullptr;
  state->waits.clear();

  return SyncToken{slot.submit_payload};
}

expected<void> HolovizRenderBackend::abort_frame(FrameCommands& cmds) {
  VIZ_UNWRAP(TicketLookup, lookup, find_ticket(cmds));
  if (lookup.stale) {
    // An owned handle from a superseded issuance: that frame was already submitted or
    // aborted (a later begin_frame() re-issued the slot), so the abort is a no-op success
    // per the interface contract — and it must never touch the LIVE issuance. Foreign
    // handles still fail (kInvalidArgument, from find_ticket above).
    return {};
  }
  FrameState* state = lookup.ticket->state.get();
  if (!state->open) {
    return {};  // Never begun, or already submitted/aborted: no-op success.
  }
  FrameSlot& slot = slots_[state->slot];

  // Uploads already enqueued by record() (possibly gated on producer streams)
  // must not outlive the frame: FrameData device memory is only pledged until
  // abort_frame() returns (frame_data.hpp). A bounded stall on the error path,
  // never on the frame loop's happy path: the drain event marks the stream's
  // current tail (covering a record() that failed mid-layer, too) and is polled
  // under the backpressure timeout — kBackpressured leaves the frame open and
  // abort retryable. The orphaned upload-semaphore signals in state->waits are
  // timeline signals nobody will wait on — harmless for a monotonic timeline —
  // and the drain flushes them.
  if (stream_ != nullptr && drain_event_ != nullptr) {
    ScopedCudaContext scoped_context(cuda_context_);
    VIZ_CU_FORWARD_ERROR(cuEventRecord(drain_event_, stream_));
    VIZ_FORWARD_ERROR(wait_event_bounded(
        drain_event_, std::chrono::nanoseconds(config_.backpressure_timeout_ns)));
  }

  // Discard the partial recording (reset() handles mid-render-scope lists).
  // slot.last_submit is deliberately untouched: this frame never submitted, so
  // the next begin_frame()'s reset() waits on the same, older token.
  VIZ_FORWARD_ERROR(slot.cmd.reset(std::chrono::nanoseconds(config_.backpressure_timeout_ns)));

  // A record() that failed mid-layer can leave a layer's per-frame recorder open (a text
  // begin_frame whose end_frame never ran); cancel is idempotent, so sweep them all.
  for (LayerState& layer_state : layer_states_) {
    if (auto* text = std::get_if<HolovizTextLayerState>(&layer_state)) {
      VIZ_FORWARD_ERROR(text->renderer.cancel_frame());
    }
  }

  state->open = false;
  state->acquire.reset();
  state->target = nullptr;
  state->waits.clear();
  return {};
}

expected<BackendInfo> HolovizRenderBackend::info() const {
  return BackendInfo{"holoviz"};
}

expected<AllocationStats> HolovizRenderBackend::stats() const {
  return stats_;
}

expected<std::unique_ptr<RenderBackend>> create_holoviz_render_backend(
    holoviz::Device& device, const HolovizRenderBackendDesc& desc) {
  if (!device) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] create_holoviz_render_backend requires a valid device."});
  }
  return std::unique_ptr<RenderBackend>(new HolovizRenderBackend(device, desc));
}

}  // namespace holoscan::visualization::backends
