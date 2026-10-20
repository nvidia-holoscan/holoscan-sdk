// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/backends/cuda_present_backend.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoviz/logger.hpp>

#include "common/holoviz_conversions.hpp"
#include "cuda/cu_errors.hpp"
#include "cuda/cu_scoped_context.hpp"
#include "detail/destroy_timeout.hpp"
#include "detail/device_impl.hpp"
#include "holoviz_op/backends/backend_tokens.hpp"
#include "holoviz_op/backends/swap_red_blue.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

using holoviz::ScopedCudaContext;

namespace {

/// Bounded completion poll without an unbounded driver block for a finite timeout: \p query is
/// a CUDA query call (CUDA_SUCCESS once complete, CUDA_ERROR_NOT_READY while pending), polled at a
/// coarse cadence until the deadline; kWaitForever falls through to \p synchronize. Reports true
/// when complete, false on timeout; CUDA failures are forwarded.
template <typename QueryFn, typename SynchronizeFn>
expected<bool> poll_until_ready(std::chrono::nanoseconds timeout, const QueryFn& query,
                                const SynchronizeFn& synchronize) {
  constexpr auto kPollInterval = std::chrono::microseconds{200};
  const auto deadline = timeout == kWaitForever ? std::chrono::steady_clock::time_point::max()
                                                : std::chrono::steady_clock::now() + timeout;
  for (;;) {
    const CUresult queried = query();
    if (queried == CUDA_SUCCESS) {
      return true;
    }
    if (queried != CUDA_ERROR_NOT_READY) {
      VIZ_CU_FORWARD_ERROR(queried);
    }
    if (timeout == kWaitForever) {
      VIZ_CU_FORWARD_ERROR(synchronize());
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(kPollInterval);
  }
}

/// Bounded wait on a CUDA event (see poll_until_ready()).
expected<bool> wait_event(CUevent event, std::chrono::nanoseconds timeout) {
  return poll_until_ready(
      timeout,
      [event] { return cuEventQuery(event); },
      [event] { return cuEventSynchronize(event); });
}

/// Bounded wait for everything queued on \p stream to complete (see poll_until_ready()).
expected<bool> wait_for_stream_idle(CUstream stream, std::chrono::nanoseconds timeout) {
  return poll_until_ready(
      timeout,
      [stream] { return cuStreamQuery(stream); },
      [stream] { return cuStreamSynchronize(stream); });
}

/// The format the ring images are created in for output format \p format: CUDA arrays have no
/// BGRA layout, so the BGRA kinds render into their RGBA sibling and the resolve swaps the red
/// and blue bytes; every other format is its own render target.
ImageFormat render_target_format(ImageFormat format) {
  switch (format) {
    case ImageFormat::kB8G8R8A8Unorm:
      return ImageFormat::kR8G8B8A8Unorm;
    case ImageFormat::kB8G8R8A8Srgb:
      return ImageFormat::kR8G8B8A8Srgb;
    default:
      return format;
  }
}

}  // namespace

CudaPresentBackend::~CudaPresentBackend() {
  const expected<void> result = destroy(holoviz::detail::kDestructorDestroyTimeout);
  if (result) {
    return;
  }
  if (!slots_.empty()) {
    // A quiescence proof failed (resolve drain, render-side semaphore, or an unprovable
    // resolve): destroy() released nothing, and GPU work may still reference the ring's
    // images. Member destruction would free them anyway (Image::destroy() does not track
    // in-use state) — quarantine instead.
    quarantine_owned_state();
    try {
      HOLOVIZ_LOG_ERROR(
          "[holoviz] {}: destroy timed out in the destructor ({}); quarantining the ring, "
          "semaphore import, and stream instead of freeing memory an in-flight resolve may still "
          "target. The device will be leaked at its teardown.",
          kName,
          result.error().message);
      // Logging can throw during teardown; cleanup must continue without escaping.
      // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
    }
    return;
  }
  try {
    HOLOVIZ_LOG_ERROR(
        "[holoviz] {}: destroy failed in destructor: {}", kName, result.error().message);
    // Logging can throw during teardown; cleanup must continue without escaping.
    // NOLINTNEXTLINE(bugprone-empty-catch)
  } catch (...) {
  }
}

void CudaPresentBackend::quarantine_owned_state() noexcept {
  // Process-lifetime quarantine, `new`ed and never deleted so no handle destructor runs on the
  // bundled objects and static-destruction order cannot reach them; the mutex serializes
  // independent backends timing out on different threads. Allocating inside noexcept is
  // deliberate: every fallback frees memory pending GPU work still references.
  // Allocation failure terminates: freeing resources still in GPU use is unsafe.
  // NOLINTNEXTLINE(bugprone-unhandled-exception-at-new)
  static auto* quarantine_mutex = new std::mutex();
  // Allocation failure terminates: freeing resources still in GPU use is unsafe.
  // NOLINTNEXTLINE(bugprone-unhandled-exception-at-new)
  static auto* quarantine = new std::vector<std::unique_ptr<QuarantinedState>>();
  auto bundle = std::make_unique<QuarantinedState>();
  bundle->ring.slots = std::move(slots_);
  bundle->ring.records = std::move(records_);
  bundle->semaphore = std::move(semaphore_);
  bundle->stream = stream_;
  slots_.clear();
  records_.clear();
  stream_ = nullptr;
  frame_table_.clear();
  retired_acquires_.clear();
  if (acquire_source_ != nullptr) {
    acquire_source_->targets.clear();
    acquire_source_->current_generation += 1;
  }
  acquire_source_.reset();
  payload_pool_.clear();
  info_ = SwapchainInfo{};
  {
    const std::lock_guard<std::mutex> lock(*quarantine_mutex);
    quarantine->push_back(std::move(bundle));
  }
  if (device_ != nullptr) {
    device_->impl().report_leaked_resources();
  }
}

// ---- ring lifecycle -----------------------------------------------------------------------

expected<CudaPresentBackend::Ring> CudaPresentBackend::build_ring(Extent2D extent,
                                                                  uint32_t image_count) {
  VIZ_UNWRAP(holoviz::Format,
             holoviz_format,
             common::to_holoviz_format(render_target_format(desc_.format)));
  const ScopedCudaContext scoped_context(cuda_context_);
  Ring ring;
  ring.slots.reserve(image_count);
  // The +2 mirrors the tracker's eviction bound: image_count + 2 tagged presents stay queryable.
  ring.records.reserve(static_cast<size_t>(image_count) + 2);
  // Every object is pushed before the next allocation so a failure mid-way leaves it reachable
  // for release_ring() rather than half-owned by a local.
  const auto build = [&]() -> expected<void> {
    for (uint32_t i = 0; i < image_count; ++i) {
      VIZ_UNWRAP(holoviz::Image,
                 image,
                 device_->create_image(holoviz::ImageDesc{
                     .extent = {extent.width, extent.height},
                     .format = holoviz_format,
                     .usage = holoviz::ImageUsage::kRenderTarget,
                     .share = holoviz::ShareMode::kExported,
                     .debug_name = "cuda present backend render target",
                 }));
      Slot& slot = ring.slots.emplace_back();
      slot.image = std::move(image);
      VIZ_UNWRAP(CUarray, array, slot.image.cuda_mipmapped_array_get_level(0));
      slot.array = array;
      VIZ_CU_FORWARD_ERROR(cuEventCreate(&slot.ready, CU_EVENT_DISABLE_TIMING));
      VIZ_CU_FORWARD_ERROR(cuEventCreate(&slot.target_ready, CU_EVENT_DISABLE_TIMING));
    }
    for (size_t i = 0; i < static_cast<size_t>(image_count) + 2; ++i) {
      PresentRecord& record = ring.records.emplace_back();
      VIZ_CU_FORWARD_ERROR(cuEventCreate(&record.done, CU_EVENT_DISABLE_TIMING));
    }
    return {};
  };
  if (const expected<void> built = build(); !built) {
    // Nothing in a ring that was never published has GPU work against it: release immediately.
    if (const expected<void> released = release_ring(ring, std::chrono::nanoseconds::zero());
        !released) {
      HOLOVIZ_LOG_ERROR("[holoviz] {}: releasing a partially built ring failed: {}",
                        kName,
                        released.error().message);
    }
    return make_unexpected(built.error());
  }
  return ring;
}

void CudaPresentBackend::commit_ring(Ring&& ring, Extent2D extent) {
  slots_ = std::move(ring.slots);
  records_ = std::move(ring.records);
  records_next_ = 0;
  const uint32_t image_count = static_cast<uint32_t>(slots_.size());

  ++generation_;
  info_ = SwapchainInfo{
      .extent = extent,
      .image_count = image_count,
      .active_mode = holoviz::PresentMode::kFIFO,
      .format = render_target_format(desc_.format),  // what the renderer builds for
  };
  next_slot_ = 0;
  frame_table_.reset(image_count);
  tracker_.reset(image_count);

  if (acquire_source_ == nullptr) {
    acquire_source_ = std::make_shared<AcquireSource>();
  }
  acquire_source_->current_generation = generation_;
  publish_targets();
  // image_count payloads can sit in the frame table plus in-flight keep-alive; the +2
  // margin plus lease_payload()'s exclusivity check makes the exact size non-load-bearing.
  payload_pool_.resize(static_cast<size_t>(image_count) + 2);
  for (std::shared_ptr<AcquirePayload>& entry : payload_pool_) {
    if (entry == nullptr) {
      entry = std::make_shared<AcquirePayload>(device_identity(*device_));
    }
  }
  payload_pool_next_ = 0;
}

// Keep the bounded retirement proof and dependent ring release in one ownership transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void> CudaPresentBackend::release_ring(Ring& ring, std::chrono::nanoseconds timeout) {
  expected<void> result{};
  const ScopedCudaContext scoped_context(cuda_context_);
  for (Slot& slot : ring.slots) {
    if (slot.ready != nullptr) {
      VIZ_CU_ACCUMULATE_ERROR(result, cuEventDestroy(slot.ready));
      slot.ready = nullptr;
    }
    if (slot.target_ready != nullptr) {
      VIZ_CU_ACCUMULATE_ERROR(result, cuEventDestroy(slot.target_ready));
      slot.target_ready = nullptr;
    }
    if (slot.image) {
      if (const expected<void> destroyed = slot.image.destroy(timeout); !destroyed && result) {
        result = destroyed;
      }
    }
  }
  for (PresentRecord& record : ring.records) {
    if (record.done != nullptr) {
      VIZ_CU_ACCUMULATE_ERROR(result, cuEventDestroy(record.done));
      record.done = nullptr;
    }
  }
  ring.slots.clear();
  ring.records.clear();
  return result;
}

expected<void> CudaPresentBackend::destroy_ring(std::chrono::nanoseconds timeout) {
  Ring current;
  current.slots = std::move(slots_);
  current.records = std::move(records_);
  slots_.clear();
  records_.clear();
  return release_ring(current, timeout);
}

expected<void> CudaPresentBackend::drain(std::chrono::nanoseconds timeout) {
  // Everything the output side queues lives on stream_: the resolves and their ready events,
  // the render-completion semaphore waits (including one a present() left behind when its copy
  // then failed to enqueue), the writer-stream edges, and any consumer work ordered there. An
  // idle stream proves all of it at once; the ready events alone would miss a wait with no copy
  // behind it. The render side (submits that wait on render_wait tokens) is the semaphore's
  // destroy() to prove.
  const ScopedCudaContext scoped_context(cuda_context_);
  for (const Slot& slot : slots_) {
    if (slot.resolve_unproven) {
      // No event was recorded behind this copy, so the caller can never join it: terminal.
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = "[holoviz] CUDA present backend: a resolve's completion event was "
                           "never recorded; its completion cannot be proven."});
    }
  }
  if (stream_ == nullptr) {
    return {};
  }
  VIZ_UNWRAP(bool, idle, wait_for_stream_idle(stream_, timeout));
  if (!idle) {
    return make_unexpected(
        Error{.code = ErrorCode::kBackpressured,
              .message = "[holoviz] CUDA present backend: output work is still in flight."});
  }
  return {};
}

void CudaPresentBackend::publish_targets() {
  if (acquire_source_ == nullptr) {
    return;
  }
  acquire_source_->targets.assign(slots_.size(), AcquireSource::Target{});
  for (size_t i = 0; i < slots_.size(); ++i) {
    acquire_source_->targets[i] =
        AcquireSource::Target{&slots_[i].image, {info_.extent.width, info_.extent.height}};
  }
}

expected<CreatedSwapchainInfo> CudaPresentBackend::create_swapchain(const SwapchainDesc& desc) {
  if (!slots_.empty()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message =
                                     "[holoviz] create_swapchain() called twice; use resize() to "
                                     "recreate."});
  }
  if (desc.preferred_extent.width == 0 || desc.preferred_extent.height == 0) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] The CUDA sink has no surface to size from; "
                                            "SwapchainDesc::preferred_extent must be nonzero."});
  }
  // Every render-target format this admits has a CUDA array equivalent (the BGRA kinds were
  // mapped to RGBA above); the exported image's import is the oracle, and the unit test
  // AdmitsEveryPublishedFormat pins the agreement for the whole CudaSinkDesc table.
  if (common::bytes_per_pixel(desc_.format) == 0 ||
      !common::to_holoviz_format(render_target_format(desc_.format))) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] The CUDA sink requires a packed single-plane ImageFormat "
                         "with a render-target lowering."});
  }
  const uint32_t image_count = std::max(1U, desc.image_count.value_or(2));

  VIZ_UNWRAP(CUcontext, cuda_context, device_->cuda_context());
  cuda_context_ = cuda_context;
  {
    // One scope across stream creation AND create_external_semaphore(): the semaphore import
    // binds to the current context (see HolovizRenderBackend::allocate_impl for the same rule).
    // A default-flag stream, for the same driver reason the render backend documents for its
    // upload stream.
    const ScopedCudaContext scoped_context(cuda_context_);
    VIZ_CU_FORWARD_ERROR(cuStreamCreate(&stream_, CU_STREAM_DEFAULT));
    VIZ_UNWRAP_NODECL(
        holoviz::ExternalSemaphore, semaphore_, device_->create_external_semaphore(stream_));
  }
  tracker_.reset_ids();

  expected<Ring> ring = build_ring(desc.preferred_extent, image_count);
  if (!ring) {
    if (expected<void> cleanup = destroy(std::chrono::seconds{1}); !cleanup) {
      HOLOVIZ_LOG_ERROR("[holoviz] {}: cleanup after a failed create_swapchain() failed: {}",
                        kName,
                        cleanup.error().message);
    }
    return make_unexpected(ring.error());
  }
  commit_ring(std::move(*ring), desc.preferred_extent);
  return CreatedSwapchainInfo{
      .extent = info_.extent,
      .image_count = info_.image_count,
      .present_mode = info_.active_mode,
  };
}

expected<void> CudaPresentBackend::destroy(std::chrono::nanoseconds timeout) {
  if (slots_.empty() && !semaphore_ && stream_ == nullptr) {
    return {};
  }
  VIZ_FORWARD_ERROR(drain(timeout));
  // The render side: every submit that waited on an acquire token (render_wait) and every
  // signal enqueued by acquire() must have completed before an image can go. The semaphore's
  // destroy() is exactly that bounded proof and releases nothing on kBackpressured, so a
  // failure here leaves the whole backend intact and retryable.
  if (semaphore_) {
    const ScopedCudaContext scoped_context(cuda_context_);
    VIZ_FORWARD_ERROR(semaphore_.destroy(timeout));
  }

  // Expire the liveness cell first so any caller-retained AcquireToken is rejectable at the
  // render backend instead of dangling; the target table is cleared and the generation
  // bumped so even a strong holder of the cell sees only inert state.
  frame_table_.clear();
  retired_acquires_.clear();
  if (acquire_source_ != nullptr) {
    acquire_source_->targets.clear();
    acquire_source_->current_generation += 1;
  }
  acquire_source_.reset();
  payload_pool_.clear();

  expected<void> result = destroy_ring(timeout);
  {
    const ScopedCudaContext scoped_context(cuda_context_);
    if (stream_ != nullptr) {
      VIZ_CU_ACCUMULATE_ERROR(result, cuStreamDestroy(stream_));
    }
  }
  stream_ = nullptr;
  info_ = SwapchainInfo{};
  return result;
}

expected<void> CudaPresentBackend::resize(Extent2D new_extent) {
  if (slots_.empty()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] resize() before create_swapchain()."});
  }
  if (new_extent.width == 0 || new_extent.height == 0) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] resize() requires a nonzero extent."});
  }
  // The ring images are borrowed by in-flight renders: drain the output side here; the render
  // side is the render backend's to prove (its begin_frame() gates on the previous submit).
  VIZ_FORWARD_ERROR(drain(std::chrono::seconds{5}));
  // Build the replacement first: a failed allocation leaves the live ring — and every token,
  // target, and cursor into it — untouched, so the caller keeps a coherent backend.
  VIZ_UNWRAP(Ring, replacement, build_ring(new_extent, info_.image_count));
  // Frames acquired before the recreation die with it; remember their issuances so
  // presenting one still answers the consumed-frame contract (kFrameDropped, id burned).
  retired_acquires_.remember(frame_table_.pending_issuances());
  // A partially failed release still leaves the old ring gone (release_ring() clears it), so
  // the new generation is committed regardless and the release error reported afterwards.
  const expected<void> released = destroy_ring(std::chrono::seconds{5});
  commit_ring(std::move(replacement), new_extent);
  // Waits on pre-recreation queued ids can no longer be resolved (cross-recreation waits are
  // unsupported, as on the swapchain backends).
  tracker_.retire_all();
  return released;
}

// ---- frame protocol -----------------------------------------------------------------------

expected<std::shared_ptr<AcquirePayload>> CudaPresentBackend::lease_payload() {
  for (size_t i = 0; i < payload_pool_.size(); ++i) {
    std::shared_ptr<AcquirePayload>& entry =
        payload_pool_[(payload_pool_next_ + i) % payload_pool_.size()];
    if (entry.use_count() == 1) {
      payload_pool_next_ = (payload_pool_next_ + i + 1) % payload_pool_.size();
      entry->image = nullptr;
      entry->extent = {};
      entry->token = holoviz::AcquireSyncToken{};
      entry->generation = 0;
      entry->issuance = 0;
      entry->render_wait = holoviz::ExternalSyncToken{};
      entry->source.reset();
      return entry;
    }
  }
  return make_unexpected(
      Error{.code = ErrorCode::kBackpressured,
            .message = "[holoviz] All image_count + 2 acquire payloads are still retained "
                       "(held AcquireTokens or render keep-alive); release one and retry."});
}

expected<AcquireToken> CudaPresentBackend::acquire(
    [[maybe_unused]] std::chrono::nanoseconds timeout) {
  if (slots_.empty()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] acquire() before create_swapchain()."});
  }
  // First free image from the round-robin cursor: presents may release images out of acquire
  // order, and a held image must not mask capacity elsewhere in the ring.
  const uint32_t image_count = static_cast<uint32_t>(slots_.size());
  std::optional<uint32_t> free_index;
  for (uint32_t i = 0; i < image_count; ++i) {
    if (const uint32_t candidate = (next_slot_ + i) % image_count;
        !frame_table_.occupied(candidate)) {
      free_index = candidate;
      break;
    }
  }
  if (!free_index) {
    // Every image is acquired-but-unpresented (the ring is exhausted): the caller must
    // present or release one first. Never blocks — there is no presentation engine to wait
    // for; the GPU-side ordering lives in render_wait.
    return make_unexpected(
        Error{.code = ErrorCode::kBackpressured,
              .message = "[holoviz] Every CUDA output image is acquired and unpresented; present "
                         "or release one before acquiring again."});
  }
  const uint32_t image_index = *free_index;
  VIZ_UNWRAP(std::shared_ptr<AcquirePayload>, payload, lease_payload());

  // GPU-side reuse gating: the signal is enqueued after everything already on stream_ — the
  // previous frame's resolve and any consumer reads ordered there — so the render submit for
  // this frame (which waits on the token) cannot overwrite pixels still being read.
  holoviz::ExternalSyncToken render_wait{};
  {
    const ScopedCudaContext scoped_context(cuda_context_);
    VIZ_UNWRAP_NODECL(holoviz::ExternalSyncToken, render_wait, semaphore_.signal());
  }

  const Slot& slot = slots_[image_index];
  payload->image = &slot.image;
  payload->extent = {info_.extent.width, info_.extent.height};
  payload->generation = generation_;
  payload->issuance = ++next_acquire_issuance_;
  payload->source = acquire_source_;
  payload->render_wait = render_wait;
  VIZ_FORWARD_ERROR(frame_table_.store(image_index, payload));
  next_slot_ = (image_index + 1) % static_cast<uint32_t>(slots_.size());

  const uint64_t issuance = payload->issuance;
  return AcquireToken{image_index, SyncToken{std::move(payload)}, issuance};
}

expected<void> CudaPresentBackend::present(const PresentArgs& args) {
  if (slots_.empty()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] present() before create_swapchain()."});
  }
  if (!tracker_.is_monotonic(args.present_id)) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Nonzero present ids must be strictly increasing."});
  }
  if (args.acquire_issuance == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] PresentArgs::acquire_issuance must carry the presented "
                         "frame's AcquireToken::issuance."});
  }

  holoviz::SubmitSyncToken render_done{};
  if (args.render_done.impl != nullptr) {
    const SubmitLookup lookup = submit_payload(args.render_done, device_identity(*device_));
    if (lookup.payload == nullptr) {
      if (lookup.check == TokenRegistry::Check::kWrongDevice) {
        return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                     .message =
                                         "[holoviz] PresentArgs::render_done was produced by a "
                                         "RenderBackend on a different device."});
      }
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] PresentArgs::render_done was not produced by this "
                           "module's RenderBackend."});
    }
    render_done = lookup.payload->token;
  }

  if (retired_acquires_.consume(args.acquire_issuance)) {
    tracker_.record(args.present_id, PresentTracker::Outcome::kDropped, {});
    return make_unexpected(
        Error{.code = ErrorCode::kFrameDropped,
              .message = "[holoviz] The acquire behind this present was retired by a ring "
                         "recreation; the frame is gone — re-acquire and render again."});
  }

  auto payload_try = frame_table_.peek(args.image_index, generation_, args.acquire_issuance);
  if (!payload_try) {
    if (payload_try.error().code == ErrorCode::kFrameDropped) {
      if (frame_table_.holds(args.image_index, args.acquire_issuance)) {
        frame_table_.erase(args.image_index);
        slots_[args.image_index].has_target = false;  // the registered destination dies with it
      }
      tracker_.record(args.present_id, PresentTracker::Outcome::kDropped, {});
    }
    return forward_error(payload_try);
  }

  Slot& slot = slots_[args.image_index];
  if (args.render_done.impl == nullptr) {
    // Empty present: nothing was rendered; release the image without producing output. A
    // tagged empty present never reaches a consumer, so its id is a known drop.
    frame_table_.erase(args.image_index);
    slot.has_target = false;
    if (args.present_id != 0) {
      tracker_.record(args.present_id, PresentTracker::Outcome::kDropped, {});
    }
    return {};
  }

  if (!slot.has_target) {
    // Checked before anything is queued: a rejected present must leave stream_ untouched.
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] The CUDA sink's present() needs a destination: call "
                         "set_resolve_target() for this image first (or release the image via "
                         "an empty present)."});
  }

  {
    const ScopedCudaContext scoped_context(cuda_context_);
    // Any failure below leaves the frame pending and its id unconsumed (retryable or
    // releasable via an empty present), per the present() contract. A failure that leaves the
    // semaphore wait queued without a copy is covered by drain(), which proves the whole stream.
    slot.resolve_enqueued = false;
    VIZ_FORWARD_ERROR(semaphore_.wait(render_done));
    // resolve() marks the slot's copy as enqueued (and unproven) the moment it is queued: a
    // failure after that point still leaves the frame pending, but the caller must keep the
    // destination alive (resolve_may_be_in_flight()).
    VIZ_FORWARD_ERROR(resolve(slot));
    const CUresult ready_recorded = faults_.fail_ready_record
                                        ? (faults_.fail_ready_record = false, CUDA_ERROR_UNKNOWN)
                                        : cuEventRecord(slot.ready, stream_);
    VIZ_CU_FORWARD_ERROR(ready_recorded);
    slot.resolve_unproven = false;  // slot.ready now covers the copy
    slot.has_frame = true;
    VIZ_FORWARD_ERROR(record_present(args.present_id));
  }
  frame_table_.erase(args.image_index);
  tracker_.record(args.present_id, PresentTracker::Outcome::kQueued, render_done);
  return {};
}

expected<void> CudaPresentBackend::resolve(Slot& slot) {
  // Caller holds the CUDA context scope and validated slot.has_target.
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  copy.srcArray = slot.array;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = slot.target.device_ptr;
  copy.dstPitch = slot.target.row_pitch_bytes;
  copy.WidthInBytes =
      static_cast<size_t>(info_.extent.width) * common::bytes_per_pixel(info_.format);
  copy.Height = info_.extent.height;
  const CUresult copied = faults_.fail_resolve_enqueue
                              ? (faults_.fail_resolve_enqueue = false, CUDA_ERROR_UNKNOWN)
                              : cuMemcpy2DAsync(&copy, stream_);
  // A copy that never entered the stream leaves the destination registered for a retry.
  VIZ_CU_FORWARD_ERROR(copied);
  slot.resolve_enqueued = true;
  slot.resolve_unproven = true;
  slot.has_target = false;  // one resolve per registered destination
  if (desc_.format != info_.format) {
    // BGRA output from an RGBA render target: swap the bytes in place behind the copy. The copy
    // is already queued, so a launch failure follows the "enqueued but unprovable" contract.
    const cudaError_t launched = launch_swap_red_blue_inplace(slot.target.device_ptr,
                                                              slot.target.row_pitch_bytes,
                                                              info_.extent.width,
                                                              info_.extent.height,
                                                              stream_);
    if (launched != cudaSuccess) {
      return make_unexpected(
          Error{.code = ErrorCode::kFailure,
                .message = std::string("[holoviz] CUDA present backend: the BGRA channel swap "
                                       "failed to launch: ") +
                           cudaGetErrorString(launched)});
    }
  }
  return {};
}

expected<void> CudaPresentBackend::record_present(PresentId id) {
  if (id == 0 || records_.empty()) {
    return {};
  }
  // Caller holds the CUDA context scope. The recycled record's previous id is at least
  // image_count + 2 presents old, which the tracker already reports as evicted.
  PresentRecord& record = records_[records_next_];
  const CUresult recorded = faults_.fail_present_record
                                ? (faults_.fail_present_record = false, CUDA_ERROR_UNKNOWN)
                                : cuEventRecord(record.done, stream_);
  VIZ_CU_FORWARD_ERROR(recorded);
  record.id = id;
  record.completed = false;
  records_next_ = (records_next_ + 1) % records_.size();
  return {};
}

CudaPresentBackend::PresentRecord* CudaPresentBackend::record_for_id(PresentId id) {
  for (PresentRecord& record : records_) {
    if (record.id == id && record.done != nullptr) {
      return &record;
    }
  }
  return nullptr;
}

expected<WaitResult> CudaPresentBackend::wait_for_present(PresentId id,
                                                          std::chrono::nanoseconds timeout) {
  if (id == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] wait_for_present() requires the nonzero present id the "
                         "frame was presented with."});
  }
  if (id > tracker_.last_id()) {
    return make_unexpected(Error{.code = ErrorCode::kNotFound,
                                 .message = "[holoviz] Present id was never presented."});
  }
  const PresentTracker::LookupResult outcome = tracker_.lookup(id);
  switch (outcome.state) {
    case PresentTracker::Lookup::kDropped:
      return WaitResult::kDropped;
    case PresentTracker::Lookup::kQueuedRetired:
      return make_unexpected(
          Error{.code = ErrorCode::kNotFound,
                .message = "[holoviz] The frame's outcome crossed a ring recreation and can no "
                           "longer be resolved (waits across recreations are unsupported)."});
    case PresentTracker::Lookup::kNeverIssued:
      return make_unexpected(Error{.code = ErrorCode::kNotFound,
                                   .message = "[holoviz] Present id was never presented."});
    case PresentTracker::Lookup::kEvicted:
      return make_unexpected(
          Error{.code = ErrorCode::kNotFound,
                .message = "[holoviz] Present id's outcome is no longer tracked (older than "
                           "the last image_count + 2 tagged presents)."});
    case PresentTracker::Lookup::kQueuedLive:
      break;
  }
  PresentRecord* record = record_for_id(id);
  if (record == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kNotFound,
                                 .message = "[holoviz] Present id's completion record is no "
                                            "longer tracked."});
  }
  if (record->completed) {
    return WaitResult::kDisplayed;  // latched: a later frame's pending resolve cannot revoke it
  }
  const ScopedCudaContext scoped_context(cuda_context_);
  VIZ_UNWRAP(bool, completed, wait_event(record->done, timeout));
  if (!completed) {
    return WaitResult::kTimeout;
  }
  record->completed = true;
  return WaitResult::kDisplayed;
}

expected<SwapchainInfo> CudaPresentBackend::swapchain_info() const {
  if (slots_.empty()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] swapchain_info() before create_swapchain()."});
  }
  return info_;
}

expected<SyncToken> CudaPresentBackend::targets() const {
  if (slots_.empty()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] targets() before create_swapchain()."});
  }
  auto payload = std::make_shared<TargetSetPayload>(device_identity(*device_));
  payload->source = acquire_source_;
  return SyncToken{std::move(payload)};
}

PresentInfo CudaPresentBackend::info() const {
  // "Displayed" means "resolve complete on the backend stream" for a capture sink; both
  // present ids and waits on them are fully supported.
  return PresentInfo{
      .name = kName,
      .has_present_id = true,
      .has_present_wait = true,
  };
}

// ---- typed extension ----------------------------------------------------------------------

expected<CUstream> CudaPresentBackend::stream() const {
  if (stream_ == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] stream() before create_swapchain()."});
  }
  return stream_;
}

expected<void> CudaPresentBackend::set_resolve_target(uint32_t image_index,
                                                      CudaResolveTarget target,
                                                      CUstream writer_stream) {
  if (image_index >= slots_.size()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] set_resolve_target(): image index out of range."});
  }
  if (!frame_table_.occupied(image_index)) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] set_resolve_target() applies to an acquired image; call it "
                         "between acquire() and present()."});
  }
  const size_t min_pitch =
      static_cast<size_t>(info_.extent.width) * common::bytes_per_pixel(info_.format);
  if (target.device_ptr == 0 || target.row_pitch_bytes < min_pitch) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] set_resolve_target(): the destination must be non-null with "
                         "a row pitch of at least width * bytes_per_pixel."});
  }
  if (writer_stream == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] set_resolve_target(): writer_stream must be the stream the "
                         "destination's write was opened on."});
  }
  Slot& slot = slots_[image_index];
  {
    // Writer-stream → backend-stream edge: whatever the destination's owner already queued on
    // the writer stream (a consumer's read of this very allocation before the pool recycled
    // it, under same-stream readiness elision) completes before this image's resolve. The
    // reverse edge (resolve → writer) is the caller's ready_event() join before commit.
    const ScopedCudaContext scoped_context(cuda_context_);
    VIZ_CU_FORWARD_ERROR(cuEventRecord(slot.target_ready, writer_stream));
    VIZ_CU_FORWARD_ERROR(cuStreamWaitEvent(stream_, slot.target_ready, 0));
  }
  slot.target = target;
  slot.has_target = true;
  return {};
}

expected<CUevent> CudaPresentBackend::ready_event(uint32_t image_index) const {
  if (image_index >= slots_.size()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] ready_event(): image index out of range."});
  }
  if (slots_[image_index].resolve_unproven) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] ready_event(): the latest resolve of this image has no "
                         "completion event; its completion cannot be proven."});
  }
  if (!slots_[image_index].has_frame) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotFound,
              .message = "[holoviz] ready_event(): this image has not been presented yet."});
  }
  return slots_[image_index].ready;
}

bool CudaPresentBackend::resolve_may_be_in_flight(uint32_t image_index) const {
  return image_index < slots_.size() && slots_[image_index].resolve_enqueued;
}

// ---- factory ------------------------------------------------------------------------------

expected<std::unique_ptr<CudaPresentBackend>> create_cuda_present_backend(
    holoviz::Device& device, const CudaPresentBackendDesc& desc) {
  if (!device) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] create_cuda_present_backend requires a valid device."});
  }
  return std::make_unique<CudaPresentBackend>(device, desc);
}

}  // namespace holoscan::visualization::backends
