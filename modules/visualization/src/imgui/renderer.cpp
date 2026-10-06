/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/imgui/renderer.hpp"
#include "detail/destroy_timeout.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <GLFW/glfw3.h>

#include <holoviz/logger.hpp>
#include <holoviz/window.hpp>

#include "cuda/cu_errors.hpp"
#include "cuda/cu_scoped_context.hpp"
#include "detail/device_impl.hpp"
#include "detail/handle_errors.hpp"
#include "detail/window_impl.hpp"
#include "imgui/detail/context_guard.hpp"
#include "imgui/detail/glfw_keymap.hpp"
#include "imgui/renderer_impl.hpp"
#include "imgui/shader_interface.hpp"

#include "imgui/shader.slang.hpp"  // generated from imgui/shader.slang

// The Slang vertex layout in imgui/shader.slang mirrors ImDrawVert; the ImGui headers bundled
// under <holoviz/imgui/vendor/> must stay the version and configuration these checks pin.
static_assert(IMGUI_VERSION_NUM == 18930,
              "bundled Dear ImGui version changed; revalidate the renderer and swipat clearance");
static_assert(sizeof(ImDrawVert) == 20, "ImDrawVert layout must match the embedded Slang shader");
static_assert(offsetof(ImDrawVert, pos) == 0 && offsetof(ImDrawVert, uv) == 8 &&
                  offsetof(ImDrawVert, col) == 16,
              "ImDrawVert layout must match the embedded Slang shader");
static_assert(sizeof(ImDrawIdx) == 2, "the renderer binds index buffers as IndexType::kUint16");

namespace holoviz::imgui {

namespace detail {

namespace {

constexpr std::chrono::seconds kGrowthWaitTimeout{5};
constexpr std::chrono::microseconds kDestroyFencePollInterval{100};

// The error every entry point returns once the backend went terminal (quarantine_backend()).
Error terminal_backend_error() {
  return Error{
      .code = ErrorCode::kCudaError,
      .message =
          "[holoviz] Renderer: the backend is terminal — CUDA retirement of "
          "enqueued stream work could not be proved; destroy the backend",
  };
}

}  // namespace

expected<std::unique_ptr<RendererImpl>> RendererImpl::create(Device& device,
                                                             const RendererDesc& desc) {
  if (desc.frames_in_flight == 0) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] create_renderer: frames_in_flight must be at least 1",
    });
  }
  if (desc.stream == nullptr) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] create_renderer: stream must be a non-null stream on the "
                   "device's CUDA context (the CUDA legacy default stream is not supported)",
    });
  }
  if (desc.font_atlas != nullptr && desc.ttf_path != nullptr) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] create_renderer: ttf_path and font_atlas are exclusive — "
                   "load fonts into the shared atlas before creation instead",
    });
  }

  auto impl = std::make_unique<RendererImpl>();
  impl->device_ = &device;
  impl->stream_ = desc.stream;

  const ContextGuard context_guard;
  impl->context_ = ImGui::CreateContext(desc.font_atlas);
  if (impl->context_ == nullptr) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message = "[holoviz] create_renderer: ImGui::CreateContext failed",
    });
  }
  ImGui::SetCurrentContext(impl->context_);
  ImGuiIO& io = ImGui::GetIO();
  // A library must not write imgui.ini/imgui_log.txt into the process working directory.
  io.IniFilename = nullptr;
  io.LogFilename = nullptr;
  io.BackendRendererName = "holoviz";
  // record_draws honors ImDrawCmd::VtxOffset, so let ImGui split draw lists that outgrow
  // the 16-bit ImDrawIdx range instead of asserting/wrapping.
  io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
  io.DisplaySize = ImVec2(1.0F, 1.0F);  // placeholder until the first new_frame()

  if (desc.ttf_path != nullptr) {
    if (io.Fonts->AddFontFromFileTTF(desc.ttf_path, desc.ttf_size_px) == nullptr) {
      return make_unexpected(Error{
          .code = ErrorCode::kNotFound,
          .message =
              std::string("[holoviz] create_renderer: failed to load font '") + desc.ttf_path + "'",
      });
    }
  } else if (io.Fonts->Fonts.empty()) {
    io.Fonts->AddFontDefault();
  }

  // Rasterize the atlas and upload it once; per-frame data never touches this path.
  unsigned char* atlas_pixels = nullptr;
  int atlas_width = 0;
  int atlas_height = 0;
  io.Fonts->GetTexDataAsRGBA32(&atlas_pixels, &atlas_width, &atlas_height);
  if (atlas_pixels == nullptr || atlas_width <= 0 || atlas_height <= 0) {
    return make_unexpected(Error{
        .code = ErrorCode::kFailure,
        .message = "[holoviz] create_renderer: font atlas rasterization failed",
    });
  }

  const ImageDesc font_image_desc{
      .extent = {static_cast<uint32_t>(atlas_width), static_cast<uint32_t>(atlas_height)},
      .format = Format::kR8G8B8A8Unorm,
      .usage = ImageUsage::kSampled,
      .debug_name = "imgui font atlas",
  };
  auto font_image = device.create_image(font_image_desc);
  if (!font_image) {
    return forward_error(font_image);
  }
  impl->font_image_ = std::move(*font_image);
  const size_t atlas_row_pitch = static_cast<size_t>(atlas_width) * 4;
  const std::span<const std::byte> atlas_bytes{reinterpret_cast<const std::byte*>(atlas_pixels),
                                               atlas_row_pitch * static_cast<size_t>(atlas_height)};
  VIZ_FORWARD_ERROR(device.upload_to_image(impl->font_image_, 0, atlas_bytes, atlas_row_pitch));
  if (desc.font_atlas == nullptr) {
    // Only drop the CPU pixels of an atlas this context owns: a caller-provided shared
    // atlas may still be read through previously returned pixel pointers or by another
    // renderer built over the same atlas.
    io.Fonts->ClearTexData();
  }
  // Stamp the atlas so ImGui writes a real id into every ImDrawCmd; draws whose id is
  // unset (0) also fall back to the font descriptor in record_draws().
  io.Fonts->SetTexID(reinterpret_cast<ImTextureID>(RendererImpl::kFontTextureId));

  // shader.slang is compiled to SPIR-V at build time; the generated header carries the bytes
  // plus the reflected interface (attributes mirroring ImDrawVert, bindings, push block).
  ShaderBlob shader_blob = detail::shader::make_shader_blob();

  // ImGui outputs straight (non-premultiplied) alpha; the BlendState defaults compose
  // premultiplied, so set the factors explicitly.
  const BlendState straight_alpha_blend{
      .enable = true,
      .src_color = BlendFactor::kSrcAlpha,
      .dst_color = BlendFactor::kOneMinusSrcAlpha,
      .src_alpha = BlendFactor::kOne,
      .dst_alpha = BlendFactor::kOneMinusSrcAlpha,
  };
  PipelineDesc pipeline_desc{
      .shader_blob = std::move(shader_blob),
      .topology = PrimitiveTopology::kTriangleList,
      .raster = {.cull = CullMode::kNone},
      .depth = {.test = false, .write = false},
      .blend = straight_alpha_blend,
      .debug_name = desc.debug_name != nullptr ? desc.debug_name : "imgui backend",
  };
  auto pipeline = device.create_pipeline(pipeline_desc);
  if (!pipeline) {
    return forward_error(pipeline);
  }
  impl->pipeline_ = std::move(*pipeline);
  auto font_descriptor = impl->pipeline_.create_descriptor();
  if (!font_descriptor) {
    return forward_error(font_descriptor);
  }
  VIZ_FORWARD_ERROR(font_descriptor->bind_image(impl->font_image_, "fontTexture"));
  VIZ_FORWARD_ERROR(font_descriptor->bind_sampler(device, {}, "fontSampler"));
  impl->textures_.emplace(RendererImpl::kFontTextureId, std::move(*font_descriptor));

  auto semaphore = device.create_external_semaphore(desc.stream);
  if (!semaphore) {
    return forward_error(semaphore);
  }
  impl->semaphore_ = std::move(*semaphore);

  impl->slots_.resize(desc.frames_in_flight);
  const size_t initial_vertex_bytes =
      std::max<size_t>(desc.initial_vertex_capacity, 1) * sizeof(ImDrawVert);
  const size_t initial_index_bytes =
      std::max<size_t>(desc.initial_index_capacity, 1) * sizeof(ImDrawIdx);
  for (Slot& slot : impl->slots_) {
    VIZ_FORWARD_ERROR(impl->ensure_slot_capacity(slot, initial_vertex_bytes, initial_index_bytes));
  }

  return impl;
}

RendererImpl::~RendererImpl() {
  try {
    expected<void> result = destroy(holoviz::detail::kDestructorDestroyTimeout);
    if (!result && result.error().code == ErrorCode::kBackpressured) {
      // The destructor cannot backpressure: the members are about to RAII-destroy under
      // the still-pending stream work, and no bound this destructor may hold can prove it
      // retired. Go terminal and leak instead of freeing under possibly-live DMA.
      quarantine_backend();
      result = destroy(std::chrono::nanoseconds::zero());
    }
    if (!result) {
      HOLOVIZ_LOG_ERROR("~RendererImpl: destroy failed: {}", result.error().message);
    }
  } catch (...) {  // destructors must not throw; formatting/sink code can
  }
}

expected<void> RendererImpl::destroy(std::chrono::nanoseconds timeout) {
  expected<void> result{};
  auto accumulate = [&result](expected<void> step) {
    if (!step && result) {
      result = std::move(step);
    }
  };

  // Phase 1 — prove retirement of every slot's pending stream work within the caller's
  // timeout BEFORE anything is torn down. cuEventSynchronize would block unboundedly on a
  // wedged stream — the very state teardown must survive — so poll each fence against one
  // shared deadline instead. Not retired in time: return kBackpressured with the backend
  // fully intact (the handle layer keeps the impl alive on kBackpressured, so the caller
  // can drain the stream and retry). A hard query failure means retirement is unprovable:
  // go terminal and leak below.
  if (!stream_retirement_unknown_) {
    const ScopedCudaContext cuda_scope(
        device_ != nullptr ? device_->cuda_context().value_or(nullptr) : nullptr);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (Slot& slot : slots_) {
      if (slot.staging_free == nullptr) {
        continue;
      }
      CUresult query = event_query_fn_(slot.staging_free);
      while (query == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(kDestroyFencePollInterval);
        query = event_query_fn_(slot.staging_free);
      }
      if (query == CUDA_ERROR_NOT_READY) {
        return make_unexpected(Error{
            .code = ErrorCode::kBackpressured,
            .message = "[holoviz] Renderer::destroy: pending staging copies did not "
                       "retire within the timeout; drain the stream (or pass a longer "
                       "timeout) and retry",
        });
      }
      if (query != CUDA_SUCCESS) {
        // The event is the only proof the DMA out of this slot's staging retired; with the
        // query failed, freeing or destroying anything the copies touch would race live DMA.
        quarantine_backend();
        accumulate(make_unexpected(
            Error{.code = ErrorCode::kCudaError,
                  .message = "[holoviz] Renderer::destroy: staging fence query failed (" +
                             std::to_string(static_cast<int>(query)) +
                             "); the backend's CUDA-referenced resources are leaked, not "
                             "destroyed"}));
        break;
      }
    }
  }

  if (input_window_impl_ != nullptr) {
    accumulate(uninstall_input_callbacks());
  }
  {
    // Free the pinned staging under the device's CUDA context — teardown may run with no
    // context current. Phase 1 proved each slot's pending copy retired (or went terminal).
    const ScopedCudaContext cuda_scope(
        device_ != nullptr ? device_->cuda_context().value_or(nullptr) : nullptr);
    for (Slot& slot : slots_) {
      if (stream_retirement_unknown_) {
        // Terminal: retirement of the stream work targeting this slot was never proved, so
        // leak the complete dependent set — fence event, pinned staging, both buffers.
        // leak() and pointer resets cannot allocate or throw; freeing or syncing here
        // would either deadlock behind the wedged stream or race live DMA.
        slot.staging_free = nullptr;
        slot.staging = nullptr;
        slot.vertex_buffer.leak();
        slot.index_buffer.leak();
        continue;
      }
      if (slot.staging_free != nullptr) {
        cuEventDestroy(slot.staging_free);
        slot.staging_free = nullptr;
      }
      if (slot.staging != nullptr) {
        host_free_fn_(slot.staging);
        slot.staging = nullptr;
      }
      accumulate(slot.vertex_buffer.destroy(timeout));
      accumulate(slot.index_buffer.destroy(timeout));
    }
  }
  slots_.clear();
  for (auto& [id, descriptor] : textures_) {
    accumulate(descriptor.destroy(timeout));
  }
  textures_.clear();
  accumulate(pipeline_.destroy(timeout));
  accumulate(font_image_.destroy(timeout));
  if (stream_retirement_unknown_) {
    // The unproved stream work includes queued semaphore signals; destroying the import
    // under them is a use-after-destroy, so it joins the leaked set. An already-leaked
    // (empty) semaphore means a destructor re-run after a reported terminal destroy:
    // nothing is left to leak or report.
    if (semaphore_) {
      accumulate(make_unexpected(
          Error{.code = ErrorCode::kCudaError,
                .message = "[holoviz] Renderer::destroy: the backend was terminal; its "
                           "draw buffers, pinned staging, and semaphore import were leaked "
                           "(unprovable CUDA retirement)"}));
      semaphore_.leak();
    }
  } else {
    accumulate(semaphore_.destroy(timeout));
  }
  if (context_ != nullptr) {
    // No ContextGuard here: if the caller left OUR context current, restoring it
    // after DestroyContext would leave a dangling current pointer — null it instead.
    if (ImGui::GetCurrentContext() == context_) {
      ImGui::SetCurrentContext(nullptr);
    }
    ImGui::DestroyContext(context_);
    context_ = nullptr;
  }
  device_ = nullptr;
  return result;
}

expected<ImGuiContext*> RendererImpl::context() const {
  return context_;
}

expected<void> RendererImpl::begin_frame(uint32_t renderer_slot, SubmitSyncToken slot_last_use) {
  if (slots_.empty()) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidState,
        .message = "[holoviz] Renderer::begin_frame called on a destroyed backend",
    });
  }
  if (stream_retirement_unknown_) {
    return make_unexpected(terminal_backend_error());
  }
  const uint32_t slot = renderer_slot % static_cast<uint32_t>(slots_.size());
  // Stream-side wait: CUDA defers the upcoming copies until the GPU is done reading the slot's
  // buffers; the host never blocks here. Wait before committing any state, so a failed call
  // leaves the backend unchanged and is retryable with the same arguments.
  VIZ_FORWARD_ERROR(semaphore_.wait(slot_last_use));
  current_slot_ = slot;
  slots_[slot].reuse_token = slot_last_use;
  frame_begun_ = true;
  return {};
}

expected<void> RendererImpl::cancel_frame() {
  // The begun frame uploaded nothing yet; clearing the flag is the whole rollback (the
  // slot rotation and stream-side semaphore wait from begin_frame are harmless).
  frame_begun_ = false;
  return {};
}

expected<void> RendererImpl::ensure_slot_capacity(Slot& slot, size_t vertex_bytes,
                                                  size_t index_bytes) {
  // A terminal backend never touches its slots again: stream work targeting their buffers
  // or staging may still be in flight, so neither reuse nor the rebuild below (which
  // destroys the old resources) is ever safe — and the fence syncs here and in the upload
  // path would hang unboundedly behind the wedged stream.
  if (stream_retirement_unknown_) {
    return make_unexpected(terminal_backend_error());
  }
  // Fast path. The staging check keeps this from ever accepting a half-grown slot: such a
  // slot must go through the full rebuild below, never straight to the flatten-into-staging
  // step.
  if (slot.vertex_capacity_bytes >= vertex_bytes && slot.index_capacity_bytes >= index_bytes &&
      slot.staging != nullptr) {
    return {};
  }

  const bool growing = slot.vertex_capacity_bytes != 0 || slot.index_capacity_bytes != 0;
  if (growing) {
    // Cold path: exported-memory recreation.
    HOLOVIZ_LOG_DEBUG("Renderer: growing draw buffers to {}+{} bytes", vertex_bytes, index_bytes);
  }

  const auto grow = [](size_t current, size_t needed) {
    return std::max(needed + needed / 2, current * 2);
  };
  const size_t new_vertex_bytes = slot.vertex_capacity_bytes >= vertex_bytes
                                      ? slot.vertex_capacity_bytes
                                      : grow(slot.vertex_capacity_bytes, vertex_bytes);
  const size_t new_index_bytes = slot.index_capacity_bytes >= index_bytes
                                     ? slot.index_capacity_bytes
                                     : grow(slot.index_capacity_bytes, index_bytes);

  // Phase A — build the complete replacement in locals. The slot is not touched until every
  // fallible step has succeeded, so any failure below leaves the old coherent slot in place
  // and the call retryable. (Failed buffer locals self-destroy on return.)
  auto vertex_buffer_result = device_->create_buffer({
      .size_bytes = new_vertex_bytes,
      .share = ShareMode::kExported,
      .debug_name = "imgui vertices",
  });
  if (!vertex_buffer_result) {
    return forward_error(vertex_buffer_result);
  }
  Buffer new_vertex_buffer = std::move(*vertex_buffer_result);
  auto index_buffer_result = device_->create_buffer({
      .size_bytes = new_index_bytes,
      .share = ShareMode::kExported,
      .debug_name = "imgui indices",
  });
  if (!index_buffer_result) {
    return forward_error(index_buffer_result);
  }
  Buffer new_index_buffer = std::move(*index_buffer_result);
  VIZ_UNWRAP(CUdeviceptr, new_vertex_ptr, new_vertex_buffer.cuda_device_ptr());
  VIZ_UNWRAP(CUdeviceptr, new_index_ptr, new_index_buffer.cuda_device_ptr());

  std::byte* new_staging = nullptr;
  CUevent new_staging_free = nullptr;
  const auto discard_replacement_staging = [&] {
    const ScopedCudaContext cuda_scope(device_->cuda_context().value_or(nullptr));
    if (new_staging_free != nullptr) {
      cuEventDestroy(new_staging_free);
    }
    if (new_staging != nullptr) {
      host_free_fn_(new_staging);
    }
  };
  {
    // Allocate the replacement pinned block BEFORE the old one is freed: the transient
    // double allocation (UI-scale, KBs..MBs) buys the all-or-nothing commit below.
    const ScopedCudaContext cuda_scope(device_->cuda_context().value_or(nullptr));
    void* staging = nullptr;
    if (const CUresult alloc = host_alloc_fn_(&staging, new_vertex_bytes + new_index_bytes, 0);
        alloc != CUDA_SUCCESS) {
      return make_unexpected(
          Error{.code = ErrorCode::kCudaError,
                .message = "[holoviz] Renderer: failed to allocate pinned draw staging (" +
                           std::to_string(new_vertex_bytes + new_index_bytes) + " bytes)"});
    }
    new_staging = static_cast<std::byte*>(staging);
    // A fresh fence for the fresh block; a never-recorded event synchronizes immediately,
    // so no state needs to carry over from the old one.
    if (const CUresult create =
            cuEventCreate(&new_staging_free, CU_EVENT_DISABLE_TIMING | CU_EVENT_BLOCKING_SYNC);
        create != CUDA_SUCCESS) {
      discard_replacement_staging();
      return make_unexpected(
          Error{.code = ErrorCode::kCudaError,
                .message = "[holoviz] Renderer: failed to create the staging reuse fence (" +
                           std::to_string(static_cast<int>(create)) + ")"});
    }
  }

  // Phase B — retire all outstanding use of the old resources; still nothing destroyed, so a
  // failed wait leaves the slot intact.
  if (growing) {
    // Host-wait for the submit that last consumed the outgrown buffers before destroying
    // them; begin_frame's stream-side wait does not cover host-visible destruction.
    if (expected<void> consumed = device_->wait_on_submit(slot.reuse_token, kGrowthWaitTimeout);
        !consumed) {
      discard_replacement_staging();
      return consumed;
    }
  }
  if (slot.staging_free != nullptr) {
    // The consuming-submit wait above does not cover copies that never reached a consuming
    // submit (a mid-upload error, an abandoned frame); the staging fence does. Retire them
    // before the buffers they target are destroyed.
    const ScopedCudaContext cuda_scope(device_->cuda_context().value_or(nullptr));
    if (const CUresult sync = event_sync_fn_(slot.staging_free); sync != CUDA_SUCCESS) {
      discard_replacement_staging();
      return make_unexpected(
          Error{.code = ErrorCode::kCudaError,
                .message = "[holoviz] Renderer: failed to retire the slot's pending copies "
                           "before growth (" +
                           std::to_string(static_cast<int>(sync)) + ")"});
    }
  }

  // Phase C — commit by swap, then retire the old resources. From here the slot is fully the
  // new resources (staging and vertex_capacity_bytes publish together, preserving the
  // [vertex | index @ vertex_capacity_bytes] staging layout); retirement failures are
  // reported but can no longer leave the slot half-grown.
  Buffer old_vertex_buffer = std::move(slot.vertex_buffer);
  Buffer old_index_buffer = std::move(slot.index_buffer);
  std::byte* const old_staging = slot.staging;
  CUevent const old_staging_free = slot.staging_free;
  slot.vertex_buffer = std::move(new_vertex_buffer);
  slot.index_buffer = std::move(new_index_buffer);
  slot.vertex_ptr = new_vertex_ptr;
  slot.index_ptr = new_index_ptr;
  slot.vertex_capacity_bytes = new_vertex_bytes;
  slot.index_capacity_bytes = new_index_bytes;
  slot.staging = new_staging;
  slot.staging_free = new_staging_free;

  expected<void> result{};
  auto accumulate = [&result](expected<void> step) {
    if (!step && result) {
      result = std::move(step);
    }
  };
  {
    const ScopedCudaContext cuda_scope(device_->cuda_context().value_or(nullptr));
    if (old_staging_free != nullptr) {
      if (const CUresult destroyed = cuEventDestroy(old_staging_free); destroyed != CUDA_SUCCESS) {
        HOLOVIZ_LOG_WARN("Renderer: failed to destroy the outgrown staging fence ({})",
                         static_cast<int>(destroyed));
      }
    }
    if (old_staging != nullptr) {
      // Both Phase-B waits proved the DMA out of this block retired. If the free itself
      // still fails, leak the block: a leak beats freeing pinned memory the driver may
      // track, and the slot stays coherent either way.
      if (const CUresult freed = host_free_fn_(old_staging); freed != CUDA_SUCCESS) {
        accumulate(make_unexpected(
            Error{.code = ErrorCode::kCudaError,
                  .message = "[holoviz] Renderer: failed to free the outgrown staging block "
                             "(leaked, " +
                             std::to_string(static_cast<int>(freed)) + ")"}));
      }
    }
  }
  accumulate(old_vertex_buffer.destroy());
  accumulate(old_index_buffer.destroy());
  return result;
}

void RendererImpl::quarantine_backend() noexcept {
  // The no-throw terminal transition (callers build diagnostics only after this returns):
  // an unprovable retirement is a property of the shared stream, so the whole backend goes
  // terminal, freezing every slot's buffers and staging plus the semaphore import until
  // destroy() leaks them. Those leaked children keep referencing the device's native
  // CUDA/Vulkan parents, so poison the device too — its teardown then leaks native state
  // instead of destroying the parents under the queued copies/signals.
  stream_retirement_unknown_ = true;
  if (device_ != nullptr) {
    device_->impl().report_leaked_resources();
  }
}

expected<void> RendererImpl::refresh_staging_fence(Slot& slot) {
  if (slot.staging_free == nullptr) {
    return {};  // no pinned staging yet, so nothing depends on the fence
  }
  const CUresult record = event_record_fn_(slot.staging_free, stream_);
  if (record == CUDA_SUCCESS) {
    return {};
  }
  // The fence could not be refreshed; drain the stream so the enqueued work retires now. A
  // successful drain leaves the slot coherent (the never-moved fence is now conservative)
  // and the frame retryable.
  if (stream_sync_fn_(stream_) == CUDA_SUCCESS) {
    return make_unexpected(Error{
        .code = ErrorCode::kCudaError,
        .message = "[holoviz] Renderer: failed to record the staging reuse fence (" +
                   std::to_string(static_cast<int>(record)) + "); the stream was drained instead"});
  }
  // Neither the fence nor a drain proves the enqueued work retired. Go terminal: the
  // backend's complete dependent set (staging blocks, events, draw buffers, the semaphore)
  // is frozen, every entry point rejects new frames, and destroy() leaks the set — a leak
  // beats rewriting or freeing memory under a possibly-live DMA or signal.
  quarantine_backend();
  return make_unexpected(
      Error{.code = ErrorCode::kCudaError,
            .message = "[holoviz] Renderer: failed to record the staging reuse fence (" +
                       std::to_string(static_cast<int>(record)) +
                       ") and to drain the stream; the backend is terminal and its resources "
                       "are quarantined"});
}

expected<FrameSync> RendererImpl::upload_draw_data(const ImDrawData* draw_data) {
  if (!frame_begun_) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidState,
        .message = "[holoviz] Renderer::upload_draw_data called without begin_frame",
    });
  }
  frame_begun_ = false;
  draw_commands_.clear();
  frame_vertex_bytes_ = 0;
  frame_index_bytes_ = 0;

  if (draw_data == nullptr || !draw_data->Valid) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] Renderer::upload_draw_data: draw_data is null or invalid",
    });
  }

  Slot& slot = slots_[current_slot_];
  if (stream_retirement_unknown_) {
    return make_unexpected(terminal_backend_error());
  }

  const size_t vertex_bytes = static_cast<size_t>(draw_data->TotalVtxCount) * sizeof(ImDrawVert);
  const size_t index_bytes = static_cast<size_t>(draw_data->TotalIdxCount) * sizeof(ImDrawIdx);
  if (vertex_bytes == 0 || index_bytes == 0) {
    // Nothing to draw; still signal so the caller's token handling stays uniform. The signal
    // is stream work, so refresh the slot's fence over it: teardown and the growth path sync
    // staging_free as their proof this backend's stream work retired, and a zero-draw frame
    // must not leave that proof pointing at an older stream position. The shared
    // record-or-drain helper keeps this path's error handling identical to the nonempty
    // path's: a failed record drains or quarantines instead of returning stale evidence.
    VIZ_UNWRAP(ExternalSyncToken, empty_token, semaphore_.signal());
    VIZ_FORWARD_ERROR(refresh_staging_fence(slot));
    return FrameSync{.upload_done = empty_token};
  }

  const ImVec2 display_pos = draw_data->DisplayPos;
  const ImVec2 display_size = draw_data->DisplaySize;
  const ImVec2 framebuffer_scale = draw_data->FramebufferScale;
  if (display_size.x <= 0.0F || display_size.y <= 0.0F || framebuffer_scale.x <= 0.0F ||
      framebuffer_scale.y <= 0.0F) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] Renderer::upload_draw_data: DisplaySize and FramebufferScale "
                   "must be positive for a non-empty frame",
    });
  }
  frame_display_pos_ = display_pos;
  frame_display_size_ = display_size;

  VIZ_FORWARD_ERROR(ensure_slot_capacity(slot, vertex_bytes, index_bytes));

  // Host fence for staging reuse: begin_frame's semaphore wait is stream-side only and does
  // not prove the F-frames-old async HtoD out of this pinned block retired before the host
  // rewrites it. By the time a slot comes around again the copy has almost always drained,
  // so this is a near-free no-op — except under a backlogged stream, exactly the case it
  // must guard. (A never-recorded event synchronizes immediately.)
  VIZ_CU_FORWARD_ERROR(cuEventSynchronize(slot.staging_free));

  // Flatten every command list into one vertex and one index range, staged host-side so the
  // caller may invalidate draw_data (e.g. the next NewFrame) as soon as this returns: the
  // copy OUT of draw_data is this synchronous flatten; only the staging block stays live
  // afterwards, fenced by staging_free.
  std::byte* const vertex_staging = slot.staging;
  std::byte* const index_staging = slot.staging + slot.vertex_capacity_bytes;
  size_t vertex_write_offset = 0;
  size_t index_write_offset = 0;
  uint32_t global_vertex_offset = 0;
  uint32_t global_index_offset = 0;
  for (int list_index = 0; list_index < draw_data->CmdListsCount; ++list_index) {
    const ImDrawList* draw_list = draw_data->CmdLists[list_index];
    const size_t list_vertex_bytes =
        static_cast<size_t>(draw_list->VtxBuffer.Size) * sizeof(ImDrawVert);
    const size_t list_index_bytes =
        static_cast<size_t>(draw_list->IdxBuffer.Size) * sizeof(ImDrawIdx);
    std::memcpy(vertex_staging + vertex_write_offset, draw_list->VtxBuffer.Data, list_vertex_bytes);
    std::memcpy(index_staging + index_write_offset, draw_list->IdxBuffer.Data, list_index_bytes);
    vertex_write_offset += list_vertex_bytes;
    index_write_offset += list_index_bytes;

    for (const ImDrawCmd& command : draw_list->CmdBuffer) {
      if (command.UserCallback != nullptr) {
        HOLOVIZ_LOG_WARN("Renderer: ImDrawList user callbacks are not supported; skipping");
        continue;
      }
      if (command.ElemCount == 0) {
        continue;
      }
      // Rebase the clip rect to the display origin and scale it into framebuffer pixels,
      // clamped to non-negative origin here; record_draws intersects it with the
      // framebuffer, where the extent is known.
      const float x1 = std::max((command.ClipRect.x - display_pos.x) * framebuffer_scale.x, 0.0F);
      const float y1 = std::max((command.ClipRect.y - display_pos.y) * framebuffer_scale.y, 0.0F);
      const float x2 = (command.ClipRect.z - display_pos.x) * framebuffer_scale.x;
      const float y2 = (command.ClipRect.w - display_pos.y) * framebuffer_scale.y;
      if (x2 <= x1 || y2 <= y1) {
        continue;
      }
      draw_commands_.push_back({
          .scissor = {static_cast<int32_t>(x1),
                      static_cast<int32_t>(y1),
                      static_cast<uint32_t>(x2 - x1),
                      static_cast<uint32_t>(y2 - y1)},
          .index_count = command.ElemCount,
          .first_index = global_index_offset + command.IdxOffset,
          .vertex_offset = static_cast<int32_t>(global_vertex_offset + command.VtxOffset),
          .texture_id = reinterpret_cast<uint64_t>(command.GetTexID()),
      });
    }
    global_vertex_offset += static_cast<uint32_t>(draw_list->VtxBuffer.Size);
    global_index_offset += static_cast<uint32_t>(draw_list->IdxBuffer.Size);
  }

  // Pinned source: cuMemcpyHtoDAsync is truly asynchronous here, so the block must not be
  // rewritten until the copy retires — that is staging_free's job (refreshed below, synced
  // at the top of this function on the slot's next use). From the first successful enqueue
  // on, EVERY exit must refresh staging_free: slot reuse, the growth path, and destroy()
  // all trust it as the only proof the DMA out of this pinned block retired before the host
  // rewrites or frees it, so an early error return must not leave it at an older stream
  // position. refresh_staging_fence records the fence or, failing that, drains the stream —
  // and quarantines the slot's complete resource set when neither can be proved.
  bool stream_work_enqueued = false;
  expected<ExternalSyncToken> upload_token = [&]() -> expected<ExternalSyncToken> {
    VIZ_CU_FORWARD_ERROR(host_memcpy_fn_(slot.vertex_ptr, vertex_staging, vertex_bytes, stream_));
    stream_work_enqueued = true;
    VIZ_CU_FORWARD_ERROR(host_memcpy_fn_(slot.index_ptr, index_staging, index_bytes, stream_));
    return semaphore_.signal();
  }();
  if (stream_work_enqueued) {
    // A copy/signal failure inside the lambda is the earlier, more specific error; report
    // it in preference to a fence refresh failure that drained and stayed retryable. A
    // quarantining refresh is different: the terminal transition must dominate the report,
    // with the earlier failure preserved as the root cause.
    if (expected<void> fenced = refresh_staging_fence(slot); !fenced) {
      if (upload_token) {
        return forward_error(fenced);
      }
      if (stream_retirement_unknown_) {
        fenced.error().message += "; earlier upload failure: " + upload_token.error().message;
        return forward_error(fenced);
      }
    }
  }
  VIZ_UNWRAP(ExternalSyncToken, token, std::move(upload_token));

  frame_vertex_bytes_ = vertex_bytes;
  frame_index_bytes_ = index_bytes;
  return FrameSync{.upload_done = token};
}

expected<void> RendererImpl::record_draws(CommandList& command_list, Extent2D framebuffer_extent) {
  if (draw_commands_.empty()) {
    return {};  // nothing was drawn this frame
  }
  if (framebuffer_extent.width == 0 || framebuffer_extent.height == 0) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] Renderer::record_draws: framebuffer extent is empty",
    });
  }

  const Slot& slot = slots_[current_slot_];
  VIZ_FORWARD_ERROR(
      command_list.set_viewport({0, 0, framebuffer_extent.width, framebuffer_extent.height}));
  VIZ_FORWARD_ERROR(command_list.bind_pipeline(pipeline_));
  const auto vertex_slice = slot.vertex_buffer.slice(0, frame_vertex_bytes_);
  if (!vertex_slice) {
    return forward_error(vertex_slice);
  }
  VIZ_FORWARD_ERROR(command_list.bind_vertex_buffer(0, *vertex_slice));
  const auto index_slice = slot.index_buffer.slice(0, frame_index_bytes_);
  if (!index_slice) {
    return forward_error(index_slice);
  }
  VIZ_FORWARD_ERROR(command_list.bind_index_buffer(*index_slice, IndexType::kUint16));

  // Maps ImGui display coordinates (y-down, origin at DisplayPos) to clip space; Vulkan's
  // y-down framebuffer means no flip. The viewport spans the full framebuffer, so
  // FramebufferScale is applied implicitly by the NDC->pixel mapping (scissors were
  // pre-scaled at upload time). frame_display_size_ is validated positive whenever
  // draw_commands_ is non-empty.
  const float scale_x = 2.0F / frame_display_size_.x;
  const float scale_y = 2.0F / frame_display_size_.y;
  const PushConstants push_constants{
      float2(scale_x, scale_y),
      float2(-1.0F - frame_display_pos_.x * scale_x, -1.0F - frame_display_pos_.y * scale_y),
  };
  VIZ_FORWARD_ERROR(command_list.set_push_data(
      pipeline_,
      std::span{reinterpret_cast<const std::byte*>(&push_constants), sizeof(push_constants)}));

  uint64_t bound_id = 0;  // ids start at 1, so 0 means "nothing bound yet"
  for (const DrawCommand& command : draw_commands_) {
    // Bind the command's texture (0 = the font atlas) when it changes. An id that was
    // never registered skips the draw with a warning instead of aborting the frame:
    // aborting would leave the command list half-recorded over one bad ImGui::Image().
    const uint64_t texture_id = command.texture_id != 0 ? command.texture_id : kFontTextureId;
    if (texture_id != bound_id) {
      const auto texture_it = textures_.find(texture_id);
      if (texture_it == textures_.end()) {
        HOLOVIZ_LOG_WARN(
            "Renderer: draw references unregistered ImTextureID {:#x}; skipping "
            "(register_image() mints valid ids)",
            texture_id);
        continue;  // deliberately does NOT update bound_id
      }
      VIZ_FORWARD_ERROR(command_list.bind_descriptor(texture_it->second));
      bound_id = texture_id;
    }
    // Intersect the (already non-negative) clip rect with the framebuffer.
    const uint32_t x0 =
        std::min(static_cast<uint32_t>(command.scissor.x), framebuffer_extent.width);
    const uint32_t y0 =
        std::min(static_cast<uint32_t>(command.scissor.y), framebuffer_extent.height);
    const uint32_t x1 = std::min(static_cast<uint32_t>(command.scissor.x) + command.scissor.width,
                                 framebuffer_extent.width);
    const uint32_t y1 = std::min(static_cast<uint32_t>(command.scissor.y) + command.scissor.height,
                                 framebuffer_extent.height);
    if (x1 <= x0 || y1 <= y0) {
      continue;
    }
    VIZ_FORWARD_ERROR(command_list.set_scissor(
        {static_cast<int32_t>(x0), static_cast<int32_t>(y0), x1 - x0, y1 - y0}));
    VIZ_FORWARD_ERROR(command_list.draw_indexed(
        command.index_count, 1, command.first_index, command.vertex_offset, 0));
  }
  return {};
}

// Deliberately NOT context-guarded: this is the internal "make our context current"
// primitive — render_ui()'s build() callback and TextRenderer::add_text() rely on the
// context staying current after it returns. Every public entry point that reaches this
// (render_ui, TextRenderer::begin_frame) holds an ContextGuard at its own level.
expected<void> RendererImpl::new_frame(Extent2D framebuffer_extent) {
  if (context_ == nullptr) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidState,
        .message = "[holoviz] Renderer::new_frame called on a destroyed backend",
    });
  }
  ImGui::SetCurrentContext(context_);
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(static_cast<float>(framebuffer_extent.width),
                          static_cast<float>(framebuffer_extent.height));
  const auto now = std::chrono::steady_clock::now();
  float delta_seconds = 1.0F / 60.0F;
  if (last_new_frame_time_.time_since_epoch().count() != 0) {
    delta_seconds = std::chrono::duration<float>(now - last_new_frame_time_).count();
  }
  io.DeltaTime = std::max(delta_seconds, 1.0e-6F);  // NewFrame asserts DeltaTime > 0
  last_new_frame_time_ = now;
  ImGui::NewFrame();
  return {};
}

expected<FrameSync> RendererImpl::render_ui(Extent2D framebuffer_extent,
                                            const std::function<void()>& build) {
  if (!build) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] Renderer::render_ui: build callback is empty",
    });
  }
  const ContextGuard context_guard;
  VIZ_FORWARD_ERROR(new_frame(framebuffer_extent));
  try {
    build();
  } catch (...) {
    ImGui::EndFrame();  // keep the context balanced before propagating
    throw;
  }
  ImGui::Render();
  return upload_draw_data(ImGui::GetDrawData());
}

expected<void> RendererImpl::forward_input(const Window& window) {
  if (context_ == nullptr) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidState,
        .message = "[holoviz] Renderer::forward_input called on a destroyed backend",
    });
  }
  if (input_window_impl_ != nullptr) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidState,
        .message = "[holoviz] Renderer::forward_input: install_input_callbacks is active; "
                   "polling would double-feed button edges",
    });
  }
  VIZ_UNWRAP(WindowSystem, window_system, window.system());
  if (window_system != WindowSystem::kGlfw) {
    return make_unexpected(Error{
        .code = ErrorCode::kNotImplemented,
        .message = "[holoviz] Renderer::forward_input: only GLFW windows are supported",
    });
  }
  VIZ_UNWRAP_NATIVE_PTR(GLFWwindow, glfw_window, window);

  const ContextGuard context_guard;
  ImGui::SetCurrentContext(context_);
  ImGuiIO& io = ImGui::GetIO();

  // Cursor position arrives in window coordinates; ImGui runs in framebuffer pixels.
  double cursor_x = 0.0;
  double cursor_y = 0.0;
  glfwGetCursorPos(glfw_window, &cursor_x, &cursor_y);
  int window_width = 0;
  int window_height = 0;
  glfwGetWindowSize(glfw_window, &window_width, &window_height);
  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(glfw_window, &framebuffer_width, &framebuffer_height);
  const float scale_x =
      window_width > 0 ? static_cast<float>(framebuffer_width) / window_width : 1.0F;
  const float scale_y =
      window_height > 0 ? static_cast<float>(framebuffer_height) / window_height : 1.0F;
  io.AddMousePosEvent(static_cast<float>(cursor_x) * scale_x,
                      static_cast<float>(cursor_y) * scale_y);

  for (int button = 0; button < 3; ++button) {
    const bool down = glfwGetMouseButton(glfw_window, button) == GLFW_PRESS;
    if (down != mouse_down_prev_[button]) {
      io.AddMouseButtonEvent(button, down);
      mouse_down_prev_[button] = down;
    }
  }
  return {};
}

expected<void*> RendererImpl::register_image(const Image& image, const SamplerDesc& sampler) {
  if (context_ == nullptr || device_ == nullptr) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidState,
        .message = "[holoviz] Renderer::register_image called on a destroyed backend",
    });
  }
  auto descriptor = pipeline_.create_descriptor();
  if (!descriptor) {
    return forward_error(descriptor);
  }
  VIZ_FORWARD_ERROR(descriptor->bind_image(image, "fontTexture"));
  VIZ_FORWARD_ERROR(descriptor->bind_sampler(*device_, sampler, "fontSampler"));
  const uint64_t id = next_texture_id_++;
  textures_.emplace(id, std::move(*descriptor));
  return reinterpret_cast<void*>(id);
}

expected<void> RendererImpl::unregister_image(void* texture_id, std::chrono::nanoseconds timeout) {
  const uint64_t id = reinterpret_cast<uint64_t>(texture_id);
  if (id == kFontTextureId) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidArgument,
        .message = "[holoviz] Renderer::unregister_image: the font-atlas texture is "
                   "owned by the backend and cannot be unregistered",
    });
  }
  const auto texture_it = textures_.find(id);
  if (texture_it == textures_.end()) {
    return make_unexpected(Error{
        .code = ErrorCode::kNotFound,
        .message = "[holoviz] Renderer::unregister_image: unknown ImTextureID",
    });
  }
  VIZ_FORWARD_ERROR(texture_it->second.destroy(timeout));
  textures_.erase(texture_it);
  return {};
}

void RendererImpl::on_cursor_pos(void* user, double x, double y) {
  auto* self = static_cast<RendererImpl*>(user);
  if (self->context_ == nullptr) {
    return;
  }
  const ContextGuard context_guard;
  ImGui::SetCurrentContext(self->context_);
  // Cursor position arrives in window coordinates; ImGui runs in framebuffer pixels.
  float scale_x = 1.0F;
  float scale_y = 1.0F;
  if (self->input_glfw_window_ != nullptr) {
    int window_width = 0;
    int window_height = 0;
    glfwGetWindowSize(self->input_glfw_window_, &window_width, &window_height);
    int framebuffer_width = 0;
    int framebuffer_height = 0;
    glfwGetFramebufferSize(self->input_glfw_window_, &framebuffer_width, &framebuffer_height);
    scale_x = window_width > 0 ? static_cast<float>(framebuffer_width) / window_width : 1.0F;
    scale_y = window_height > 0 ? static_cast<float>(framebuffer_height) / window_height : 1.0F;
  }
  ImGui::GetIO().AddMousePosEvent(static_cast<float>(x) * scale_x, static_cast<float>(y) * scale_y);
}

void RendererImpl::on_mouse_button(void* user, int button, int action, int mods) {
  auto* self = static_cast<RendererImpl*>(user);
  if (self->context_ == nullptr) {
    return;
  }
  const ContextGuard context_guard;
  ImGui::SetCurrentContext(self->context_);
  ImGuiIO& io = ImGui::GetIO();
  io.AddKeyEvent(ImGuiMod_Ctrl, (mods & GLFW_MOD_CONTROL) != 0);
  io.AddKeyEvent(ImGuiMod_Shift, (mods & GLFW_MOD_SHIFT) != 0);
  io.AddKeyEvent(ImGuiMod_Alt, (mods & GLFW_MOD_ALT) != 0);
  io.AddKeyEvent(ImGuiMod_Super, (mods & GLFW_MOD_SUPER) != 0);
  if (button >= 0 && button < ImGuiMouseButton_COUNT) {
    io.AddMouseButtonEvent(button, action == GLFW_PRESS);
  }
}

void RendererImpl::on_scroll(void* user, double dx, double dy) {
  auto* self = static_cast<RendererImpl*>(user);
  if (self->context_ == nullptr) {
    return;
  }
  const ContextGuard context_guard;
  ImGui::SetCurrentContext(self->context_);
  ImGui::GetIO().AddMouseWheelEvent(static_cast<float>(dx), static_cast<float>(dy));
}

void RendererImpl::on_key(void* user, int key, int /*scancode*/, int action, int mods) {
  if (action == GLFW_REPEAT) {
    return;  // ImGui synthesizes key repeats itself
  }
  auto* self = static_cast<RendererImpl*>(user);
  if (self->context_ == nullptr) {
    return;
  }
  const ContextGuard context_guard;
  ImGui::SetCurrentContext(self->context_);
  ImGuiIO& io = ImGui::GetIO();
  io.AddKeyEvent(ImGuiMod_Ctrl, (mods & GLFW_MOD_CONTROL) != 0);
  io.AddKeyEvent(ImGuiMod_Shift, (mods & GLFW_MOD_SHIFT) != 0);
  io.AddKeyEvent(ImGuiMod_Alt, (mods & GLFW_MOD_ALT) != 0);
  io.AddKeyEvent(ImGuiMod_Super, (mods & GLFW_MOD_SUPER) != 0);
  // Drop keys the map cannot name (GLFW_KEY_UNKNOWN, WORLD keys, F-keys past what this
  // ImGui version defines): AddKeyEvent is a named-key API and must not see ImGuiKey_None.
  const ImGuiKey imgui_key = glfw_key_to_imgui_key(key);
  if (imgui_key != ImGuiKey_None) {
    io.AddKeyEvent(imgui_key, action == GLFW_PRESS);
  }
}

void RendererImpl::on_char(void* user, unsigned int codepoint) {
  auto* self = static_cast<RendererImpl*>(user);
  if (self->context_ == nullptr) {
    return;
  }
  const ContextGuard context_guard;
  ImGui::SetCurrentContext(self->context_);
  ImGui::GetIO().AddInputCharacter(codepoint);
}

void RendererImpl::on_focus(void* user, int focused) {
  auto* self = static_cast<RendererImpl*>(user);
  if (self->context_ == nullptr) {
    return;
  }
  const ContextGuard context_guard;
  ImGui::SetCurrentContext(self->context_);
  ImGui::GetIO().AddFocusEvent(focused != 0);
}

expected<void> RendererImpl::install_input_callbacks(Window& window) {
  if (context_ == nullptr) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidState,
        .message = "[holoviz] Renderer::install_input_callbacks on a destroyed backend",
    });
  }
  if (input_window_impl_ != nullptr) {
    return make_unexpected(Error{
        .code = ErrorCode::kInvalidState,
        .message = "[holoviz] Renderer::install_input_callbacks: callbacks are already "
                   "installed; call uninstall_input_callbacks() first",
    });
  }
  VIZ_UNWRAP(WindowSystem, window_system, window.system());
  if (window_system != WindowSystem::kGlfw) {
    return make_unexpected(Error{
        .code = ErrorCode::kNotImplemented,
        .message = "[holoviz] Renderer::install_input_callbacks: only GLFW windows are supported",
    });
  }
  VIZ_UNWRAP_NATIVE_PTR(GLFWwindow, glfw_window, window);
  VIZ_FORWARD_ERROR(window.set_input_callbacks({
      .user_data = this,
      .cursor_pos = &RendererImpl::on_cursor_pos,
      .mouse_button = &RendererImpl::on_mouse_button,
      .scroll = &RendererImpl::on_scroll,
      .key = &RendererImpl::on_key,
      .char_input = &RendererImpl::on_char,
      .focus = &RendererImpl::on_focus,
  }));
  // Retain the heap-stable WindowImpl, not &window: the wrapper may be moved while
  // callbacks are installed, and uninstall must still reach the live registration.
  input_window_impl_ = &window.impl();
  input_glfw_window_ = glfw_window;
  return {};
}

expected<void> RendererImpl::uninstall_input_callbacks() {
  if (input_window_impl_ == nullptr) {
    return {};  // nothing installed: no-op success so teardown can call unconditionally
  }
  // Keep the registration state on failure: dropping it while the window still holds
  // callbacks targeting this backend would leave a dangling event target.
  VIZ_FORWARD_ERROR(input_window_impl_->set_input_callbacks({}));
  input_window_impl_ = nullptr;
  input_glfw_window_ = nullptr;
  return {};
}

}  // namespace detail

Renderer::Renderer(std::unique_ptr<detail::RendererImpl> impl) : impl_(std::move(impl)) {}
Renderer::Renderer() = default;
Renderer::Renderer(Renderer&&) noexcept = default;
Renderer& Renderer::operator=(Renderer&&) noexcept = default;
Renderer::~Renderer() = default;

expected<void> Renderer::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<ImGuiContext*> Renderer::context() const {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::context");
  }
  return impl_->context();
}

expected<void> Renderer::begin_frame(uint32_t renderer_slot, SubmitSyncToken slot_last_use) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::begin_frame");
  }
  return impl_->begin_frame(renderer_slot, slot_last_use);
}

expected<void> Renderer::cancel_frame() {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::cancel_frame");
  }
  return impl_->cancel_frame();
}

expected<FrameSync> Renderer::upload_draw_data(const ImDrawData* draw_data) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::upload_draw_data");
  }
  return impl_->upload_draw_data(draw_data);
}

expected<void> Renderer::record_draws(CommandList& command_list, Extent2D framebuffer_extent) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::record_draws");
  }
  return impl_->record_draws(command_list, framebuffer_extent);
}

expected<FrameSync> Renderer::render_ui(Extent2D framebuffer_extent,
                                        const std::function<void()>& build) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::render_ui");
  }
  return impl_->render_ui(framebuffer_extent, build);
}

expected<void> Renderer::forward_input(const Window& window) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::forward_input");
  }
  return impl_->forward_input(window);
}

expected<void*> Renderer::register_image(const Image& image, const SamplerDesc& sampler) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::register_image");
  }
  return impl_->register_image(image, sampler);
}

expected<void> Renderer::unregister_image(void* texture_id, std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::unregister_image");
  }
  return impl_->unregister_image(texture_id, timeout);
}

expected<void> Renderer::install_input_callbacks(Window& window) {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::install_input_callbacks");
  }
  return impl_->install_input_callbacks(window);
}

expected<void> Renderer::uninstall_input_callbacks() {
  if (!impl_) {
    return holoviz::detail::null_impl_error("Renderer::uninstall_input_callbacks");
  }
  return impl_->uninstall_input_callbacks();
}

expected<Renderer> create_renderer(Device& device, const RendererDesc& desc) {
  VIZ_UNWRAP(
      std::unique_ptr<detail::RendererImpl>, impl, detail::RendererImpl::create(device, desc));
  return Renderer{std::move(impl)};
}

}  // namespace holoviz::imgui
