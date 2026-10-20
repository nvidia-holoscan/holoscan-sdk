// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/backends/swapchain_present_backend.hpp"
#include "detail/destroy_timeout.hpp"

#include <algorithm>
#include <memory>
#include <utility>

#include <magic_enum/magic_enum.hpp>

#include <holoviz/adapter.hpp>
#include <holoviz/logger.hpp>
#include "holoviz_op/types.hpp"

#include "common/holoviz_conversions.hpp"
#include "holoviz_op/backends/backend_tokens.hpp"
#include "holoviz_op/backends/srgb_surface_format.hpp"

namespace holoscan::visualization::backends {

SwapchainPresentBackend::~SwapchainPresentBackend() {
  if (expected<void> result =
          SwapchainPresentBackend::destroy(holoviz::detail::kDestructorDestroyTimeout);
      !result) {
    try {
      HOLOVIZ_LOG_ERROR("[holoviz] SwapchainPresentBackend: destroy failed in destructor: {}",
                        result.error().message);
      // Logging can throw during teardown; cleanup must continue without escaping.
      // NOLINTNEXTLINE(bugprone-empty-catch)
    } catch (...) {
    }
  }
}

expected<void> SwapchainPresentBackend::create_swapchain_internal() {
  VIZ_UNWRAP(holoviz::CapabilityReport, device_caps, device_->capabilities());
  has_present_feedback_ = device_caps.present_completion;

  VIZ_UNWRAP(holoviz::SurfaceCapabilities, caps, device_->surface_capabilities(*surface_));
  VIZ_UNWRAP(holoviz::Format,
             format,
             framebuffer_srgb_
                 ? resolve_srgb_surface_format(caps.formats, device_caps.mutable_swapchain_format)
                 : resolve_unorm_surface_format(caps.formats));
  VIZ_UNWRAP(holoviz::PresentMode,
             resolved_mode,
             resolve_present_mode(preferred_mode_, caps.present_modes));
  if (resolved_mode != preferred_mode_) {
    HOLOVIZ_LOG_INFO("[holoviz] Present mode {} resolved to {} for this surface.",
                     magic_enum::enum_name(preferred_mode_),
                     magic_enum::enum_name(resolved_mode));
  }

  // Image-count bounds are per present mode (a driver may need more images for e.g.
  // FIFO_LATEST_READY than its mode-agnostic range says), so clamp against the bounds of
  // the mode the swapchain is actually created with; HoloViz then sees a valid request.
  VIZ_UNWRAP(holoviz::SurfaceCapabilities,
             mode_caps,
             device_->surface_capabilities(*surface_, resolved_mode));
  uint32_t image_count = std::max(requested_image_count_, mode_caps.min_image_count);
  if (mode_caps.max_image_count != 0) {
    image_count = std::min(image_count, mode_caps.max_image_count);
  }
  if (image_count != requested_image_count_) {
    HOLOVIZ_LOG_INFO(
        "[holoviz] Swapchain image count {} clamped to {} (surface range [{}, {}] for {}).",
        requested_image_count_,
        image_count,
        mode_caps.min_image_count,
        mode_caps.max_image_count,
        magic_enum::enum_name(resolved_mode));
  }

  // HoloViz sizes the swapchain from the surface's live extent, clamped to the surface's
  // current bounds (the window system governs a window surface's size; a display surface is
  // fixed to its mode); SwapchainDesc::preferred_extent is advisory only.
  VIZ_UNWRAP(holoviz::Swapchain,
             swapchain,
             device_->create_swapchain(holoviz::SwapchainDesc{
                 .surface = std::cref(*surface_),
                 .format = format,
                 .present_mode = resolved_mode,
                 .image_count = image_count,
                 .debug_name = "swapchain present backend swapchain",
                 .srgb_attachment = framebuffer_srgb_,
             }));
  swapchain_ = std::move(swapchain);

  VIZ_UNWRAP(holoviz::Extent2D, actual_extent, swapchain_.extent());
  VIZ_UNWRAP(uint32_t, actual_count, swapchain_.image_count());
  VIZ_UNWRAP(holoviz::Format, actual_format, swapchain_.format());
  // Fails for HoloViz swapchain formats with no public spelling; HDR10's kA2B10G10R10Unorm
  // needs an ImageFormat entry appended before an HDR10 swapchain can be reported here.
  VIZ_UNWRAP(ImageFormat, swapchain_format, common::to_image_format(actual_format));

  swapchain_info_ = SwapchainInfo{
      .extent = {actual_extent.width, actual_extent.height},
      .image_count = actual_count,
      .active_mode = resolved_mode,
      .format = swapchain_format,
  };
  VIZ_UNWRAP(uint64_t, generation, swapchain_.generation());
  last_seen_generation_ = generation;
  frame_table_.reset(actual_count);
  tracker_.reset(actual_count);

  // Publish the generation to token consumers (render backend) through the liveness cell,
  // and size the acquire-payload pool: up to image_count payloads can sit in the frame
  // table plus in-flight keep-alive; the +2 margin plus lease_payload()'s exclusivity
  // check makes the exact size non-load-bearing.
  if (acquire_source_ == nullptr) {
    acquire_source_ = std::make_shared<AcquireSource>();
  }
  acquire_source_->current_generation = last_seen_generation_;
  publish_targets(actual_count, actual_extent);
  payload_pool_.resize(static_cast<size_t>(actual_count) + 2);
  for (std::shared_ptr<AcquirePayload>& entry : payload_pool_) {
    if (entry == nullptr) {
      entry = std::make_shared<AcquirePayload>(device_identity(*device_));
    }
  }
  payload_pool_next_ = 0;

  return {};
}

expected<void> SwapchainPresentBackend::refresh_swapchain_state() {
  // Transactional: read everything fallible first, then commit — a failed read must not
  // leave the generation published over unreconciled metadata (all three reads are plain
  // member getters at the impl; the handle-level failure mode is an empty handle, which
  // every caller guards).
  VIZ_UNWRAP(uint64_t, generation, swapchain_.generation());
  if (generation == last_seen_generation_) {
    return {};
  }
  VIZ_UNWRAP(holoviz::Extent2D, extent, swapchain_.extent());
  VIZ_UNWRAP(uint32_t, image_count, swapchain_.image_count());

  last_seen_generation_ = generation;
  if (acquire_source_ != nullptr) {
    acquire_source_->current_generation = last_seen_generation_;
  }
  // Image wrappers are per-generation: republish the target table for consumers (the
  // render backend's register_targets()/re-warm). A failed recreation publishes an empty
  // table (image_count 0).
  publish_targets(image_count, extent);
  // Only extent and image count: the mode and format are contractually stable across
  // recreations (see SwapchainInfo's doc; holoviz::Swapchain::resize() preserves the format).
  // After a FAILED recreation image_count() is 0 — a consistent "lost" state the tables
  // size down to; the next successful acquire()/resize() sizes them back.
  swapchain_info_.extent = {extent.width, extent.height};
  swapchain_info_.image_count = image_count;

  // Frames acquired before the recreation died with it; the table keeps their payloads so
  // presenting one reports kFrameDropped through the generation check (their dead gates are
  // never touched) — a fresh acquire that lands on the same image index replaces such an
  // entry in store(). When the image count changed the table must be resized, which drops
  // them wholesale — their issuances are remembered first so presenting one still answers
  // the consumed-frame contract (kFrameDropped, id burned) instead of a retryable-looking
  // "not acquired"/"out of range" for a frame that can never be redeemed. The tracker reset
  // also forgets pre-recreation outcomes (waits across an image-count change report
  // kNotFound; cross-recreation waits are documented as unsupported). Live payload holders
  // pin their old pool entries, so reusing/regrowing the pool cannot alias them.
  if (frame_table_.size() != image_count) {
    retired_acquires_.remember(frame_table_.pending_issuances());
    frame_table_.reset(image_count);
    tracker_.reset(image_count);
    payload_pool_.assign(static_cast<size_t>(image_count) + 2, nullptr);
    for (std::shared_ptr<AcquirePayload>& entry : payload_pool_) {
      entry = std::make_shared<AcquirePayload>(device_identity(*device_));
    }
    payload_pool_next_ = 0;
  }

  // HoloViz drained this swapchain's in-flight rendering before recreating, so waits on
  // pre-recreation queued ids resolve immediately — the documented approximation (their
  // frames may still be queued on the retired swapchain's handover, but wait_for_present()
  // across a recreation is defined as unsupported). Known drops stay drops.
  tracker_.retire_all();
  return {};
}

void SwapchainPresentBackend::reconcile_after_frame_error(const char* where) {
  if (!swapchain_) {
    return;
  }
  if (expected<void> refreshed = refresh_swapchain_state(); !refreshed) {
    HOLOVIZ_LOG_ERROR(
        "[holoviz] SwapchainPresentBackend: reconciliation after a {} error failed: {}",
        where,
        refreshed.error().message);
  }
}

void SwapchainPresentBackend::publish_targets(uint32_t image_count, holoviz::Extent2D extent) {
  if (acquire_source_ == nullptr) {
    return;
  }
  acquire_source_->targets.assign(image_count, AcquireSource::Target{});
  for (uint32_t i = 0; i < image_count; ++i) {
    const auto image = swapchain_.image(i);
    if (image) {
      acquire_source_->targets[i] = AcquireSource::Target{&image->get(), extent};
    }
  }
}

void SwapchainPresentBackend::release_unpresented_acquire(holoviz::AcquireSyncToken&& token) {
  if (expected<void> released = swapchain_.present(std::move(token), holoviz::SubmitSyncToken{}, 0);
      !released) {
    HOLOVIZ_LOG_ERROR(
        "[holoviz] SwapchainPresentBackend: releasing an unusable acquired image failed ({}); "
        "resize() or destroy() recovers.",
        released.error().message);
  }
}

expected<CreatedSwapchainInfo> SwapchainPresentBackend::create_swapchain(
    const SwapchainDesc& desc) {
  if (swapchain_) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message =
                                     "[holoviz] create_swapchain() called twice; use resize() to "
                                     "recreate."});
  }
  std::shared_ptr<holoviz::Surface> surface = surface_from_handle(desc.surface);
  if (surface == nullptr || !*surface) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] SwapchainDesc::surface is not a valid surface handle "
                         "(create it with backends::make_window_surface_handle or "
                         "make_display_surface_handle)."});
  }

  surface_ = std::move(surface);
  preferred_mode_ = desc.preferred_mode;
  framebuffer_srgb_ = desc.framebuffer_srgb;
  requested_image_count_ = desc.image_count.value_or(default_image_count());
  tracker_.reset_ids();

  expected<void> created = create_swapchain_internal();
  if (!created) {
    surface_.reset();
    return make_unexpected(created.error());
  }
  return CreatedSwapchainInfo{
      .extent = swapchain_info_.extent,
      .image_count = swapchain_info_.image_count,
      .present_mode = swapchain_info_.active_mode,
  };
}

expected<void> SwapchainPresentBackend::destroy(std::chrono::nanoseconds timeout) {
  if (!swapchain_) {
    surface_.reset();
    return {};
  }
  // Pending payloads hold gates into the swapchain being destroyed; drop them first (gates
  // are non-owning views into swapchain-recycled slots, so dropping frees nothing).
  // Expiring the liveness cell makes any caller-retained AcquireToken rejectable at the
  // render backend instead of dangling; the target table is cleared and the generation
  // bumped first so even a strong holder of the cell sees only inert state.
  frame_table_.clear();
  retired_acquires_.clear();  // destroy() is the terminal boundary; cross-destroy tokens
                              // are out of contract, so their tombstones die here too
  if (acquire_source_ != nullptr) {
    acquire_source_->targets.clear();
    acquire_source_->current_generation += 1;
  }
  acquire_source_.reset();
  payload_pool_.clear();
  // No device-wide idle wait: holoviz::Swapchain::destroy() performs bounded, retryable waits
  // on everything the swapchain owns (per-slot submit drains, gate fences, pending present
  // fences); waiting for unrelated device work is the render backend's job, and an
  // unbounded vkDeviceWaitIdle here would defeat this method's timeout. On kBackpressured
  // the swapchain is left intact and destroy() is retryable.
  VIZ_FORWARD_ERROR(swapchain_.destroy(timeout));
  surface_.reset();
  swapchain_info_ = SwapchainInfo{};
  return {};
}

expected<void> SwapchainPresentBackend::resize(Extent2D new_extent) {
  if (!swapchain_) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] resize() before create_swapchain()."});
  }
  // In-place through holoviz::Swapchain::resize(): the handle, format, and image indices
  // survive, and the presentation engine hands over smoothly (oldSwapchain). Resizing is
  // normally automatic inside acquire()/present(); this explicit form exists for surfaces
  // that report no fixed extent ({0,0} = "match the surface" elsewhere) and callers that
  // want to pick the moment. Outstanding AcquireTokens become stale either way — the
  // generation reconciliation below (and in acquire()) turns presenting one into
  // kFrameDropped instead of undefined behavior.
  const expected<void> resized = swapchain_.resize({new_extent.width, new_extent.height});
  // Reconcile even when the resize FAILED: a failed recreation still retired the old
  // swapchain and bumped HoloViz generation (HoloViz invalidates before its fallible
  // successor work), and skipping the refresh would leave acquire_source_ publishing the
  // dead generation over destroyed image wrappers. A kBackpressured resize left the
  // swapchain untouched, so the refresh no-ops on the unchanged generation.
  const expected<void> refreshed = refresh_swapchain_state();
  if (!resized) {
    return resized;  // the resize error dominates the (best-effort) refresh outcome
  }
  return refreshed;
}

expected<AcquireToken> SwapchainPresentBackend::acquire(std::chrono::nanoseconds timeout) {
  if (!swapchain_) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] acquire() before create_swapchain()."});
  }

  // Lease the payload BEFORE HoloViz acquire: pool exhaustion (kBackpressured) must
  // consume nothing — the caller releases a retained token and retries with HoloViz's
  // single outstanding acquire still available.
  VIZ_UNWRAP(std::shared_ptr<AcquirePayload>, payload, lease_payload());

  // The HoloViz swapchain self-heals on resizable surfaces: a stale or out-of-date swapchain is
  // recreated inside this call. kBackpressured (busy or minimized) and kImageCountChanged
  // (per-image resources must be rebuilt) propagate to the caller as retry/rebuild signals;
  // the reconciliation is best-effort so those signals are never masked by a refresh error.
  auto acquired = swapchain_.acquire(timeout);
  if (!acquired) {
    reconcile_after_frame_error("acquire");
    return forward_error(acquired);
  }

  // Reconcile after an extent-only recreation HoloViz performed invisibly (an image-count
  // change surfaces as the error above instead). From here on, every failure exit must
  // hand the acquired image back — abandoning the move-only token would strand HoloViz's
  // single outstanding acquire until a resize()/destroy().
  if (const expected<void> refreshed = refresh_swapchain_state(); !refreshed) {
    release_unpresented_acquire(std::move(*acquired));
    return forward_error(refreshed);
  }

  const uint32_t image_index = acquired->image_index;
  const auto image = swapchain_.image(image_index);
  if (!image) {
    release_unpresented_acquire(std::move(*acquired));
    return forward_error(image);
  }

  payload->image = &image->get();
  payload->extent = {swapchain_info_.extent.width, swapchain_info_.extent.height};
  payload->generation = last_seen_generation_;
  payload->issuance = ++next_acquire_issuance_;
  payload->source = acquire_source_;

  if (const expected<void> stored = frame_table_.store(image_index, payload); !stored) {
    // De-facto unreachable in-protocol (store() replaces stale-generation occupants and a
    // same-generation collision is a double-acquire HoloViz already rejects), kept as a
    // belt: the release keeps HoloViz's outstanding acquire recoverable.
    release_unpresented_acquire(std::move(*acquired));
    return forward_error(stored);
  }
  payload->token = std::move(*acquired);  // last, infallible — no exit may abandon it

  const uint64_t issuance = payload->issuance;
  return AcquireToken{image_index, SyncToken{std::move(payload)}, issuance};
}

expected<std::shared_ptr<AcquirePayload>> SwapchainPresentBackend::lease_payload() {
  for (size_t i = 0; i < payload_pool_.size(); ++i) {
    std::shared_ptr<AcquirePayload>& entry =
        payload_pool_[(payload_pool_next_ + i) % payload_pool_.size()];
    if (entry.use_count() == 1) {
      payload_pool_next_ = (payload_pool_next_ + i + 1) % payload_pool_.size();
      // Exclusive: no stale holder can observe the re-stamp (the generation check at
      // present/begin_frame stays the semantic staleness guard regardless).
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
  // Every pool entry is still referenced: the caller retains more than image_count + 2
  // payloads (AcquireTokens or render keep-alive). Refuse without allocating — the
  // steady-state acquire path stays allocation-free by contract.
  return make_unexpected(
      Error{.code = ErrorCode::kBackpressured,
            .message = "[holoviz] All image_count + 2 acquire payloads are still retained "
                       "(held AcquireTokens or render keep-alive); release one and retry."});
}

expected<void> SwapchainPresentBackend::present(const PresentArgs& args) {
  if (!swapchain_) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] present() before create_swapchain()."});
  }
  if (!tracker_.is_monotonic(args.present_id)) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Nonzero present ids must be strictly increasing."});
  }
  if (args.acquire_issuance == 0) {
    // Issuances are nonzero by construction; a zero here is a caller that never copied
    // AcquireToken::issuance — reject loudly instead of misreporting it as a stale frame.
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] PresentArgs::acquire_issuance must carry the presented "
                         "frame's AcquireToken::issuance."});
  }

  holoviz::SubmitSyncToken render_done{};
  if (args.render_done.impl != nullptr) {
    const SubmitLookup render_done_lookup =
        submit_payload(args.render_done, device_identity(*device_));
    if (render_done_lookup.payload == nullptr) {
      if (render_done_lookup.check == TokenRegistry::Check::kWrongDevice) {
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
    render_done = render_done_lookup.payload->token;
  }

  // A changed-count recreation retired this acquire wholesale (its table slot is gone —
  // possibly its whole index range), so the resized table below would misclassify it as
  // retryable. Checked first: tombstoned issuances all predate the reset, so none can
  // collide with a live occupant. Single-shot, mirroring the drop paths below: the
  // nonzero id is burned and recorded so wait_for_present(id) answers kDropped. No
  // reconcile here — the tombstone exists because refresh_swapchain_state() already ran
  // for that recreation, so no tracker reset can evict the outcome recorded now.
  if (retired_acquires_.consume(args.acquire_issuance)) {
    tracker_.record(args.present_id, PresentTracker::Outcome::kDropped, {});
    return make_unexpected(
        Error{.code = ErrorCode::kFrameDropped,
              .message = "[holoviz] The acquire behind this present was retired by a "
                         "swapchain recreation that changed the image count; the frame is "
                         "gone — re-acquire and render again."});
  }

  auto payload_try =
      frame_table_.peek(args.image_index, last_seen_generation_, args.acquire_issuance);
  if (!payload_try) {
    if (payload_try.error().code == ErrorCode::kFrameDropped) {
      // The caller's frame is gone — its acquire predates a recreation, or a fresh
      // acquire superseded it on the same (recyclable) image index. Mirror the
      // post-present drop path below: the nonzero id is consumed by the tagging (ids stay
      // strictly increasing), and the frame never reached the display — record it as a
      // known drop so wait_for_present(id) answers kDropped instead of kNotFound or a
      // blocked wait. Retire the table entry (peek is non-consuming) only when the
      // occupant IS this dead frame; a superseding occupant is someone else's live
      // pending frame and must stay redeemable. Reconcile BEFORE recording the drop: the
      // best-effort refresh may reset the tracker on an image-count change, which must
      // not evict the outcome just promised to this frame's waiter.
      if (frame_table_.holds(args.image_index, args.acquire_issuance)) {
        frame_table_.erase(args.image_index);
      }
      reconcile_after_frame_error("present (stale token)");
      tracker_.record(args.present_id, PresentTracker::Outcome::kDropped, {});
    }
    return forward_error(payload_try);
  }
  // peek() copied the shared_ptr out of the table (one refcount round-trip per present,
  // versus the old move-out take()): the deliberate price of keeping the entry — and the
  // move-only token inside it — redeemable when swapchain_.present() below fails without
  // consuming the frame.
  const std::shared_ptr<AcquirePayload>& payload = *payload_try;

  holoviz::PresentConsumed consumed{};
  auto presented =
      swapchain_.present(std::move(payload->token), render_done, args.present_id, &consumed);
  if (!presented) {
    if (!consumed.image) {
      // Nothing was consumed (a pre-queue failure inside HoloViz): the table entry, the
      // move-only token inside it, and both id watermarks are exactly as before this call.
      // The caller may retry the same present, release the image via an empty present, or
      // resize — the frame loop is never stranded.
      return presented;
    }
    // The frame is finished (queued then dropped, or dead with its generation) and any
    // tagged id was burned by HoloViz. Burn the id here too, even when HoloViz never
    // tagged it (consumed.id == false): the backend watermark may only run AHEAD of the
    // HoloViz's — harmless in that direction, HoloViz accepts any id above its own watermark —
    // never behind, which would let a later present() reuse an id HoloViz then rejects.
    // Reconcile BEFORE recording the drop: the best-effort refresh resets the tracker on
    // an image-count change, which must not evict the outcome just promised to this
    // frame's waiter (kImageCountChanged's contract is wait_for_present(id) == kDropped).
    frame_table_.erase(args.image_index);
    reconcile_after_frame_error("present");
    tracker_.record(args.present_id, PresentTracker::Outcome::kDropped, {});
    const ErrorCode cause = presented.error().code;
    if (cause == ErrorCode::kFrameDropped || cause == ErrorCode::kImageCountChanged) {
      return presented;  // already the consumed-designated public results
    }
    // Every other lower error that consumed the frame (a recovery drain timing out, a
    // generic post-queue rejection, a lost device, ...) would read as retryable under the
    // present() contract — but retrying a consumed frame can never succeed. Report the
    // accurate continue signal, preserving the native cause in the message (a dead device
    // simply fails the next acquire() too).
    return make_unexpected(
        Error{.code = ErrorCode::kFrameDropped,
              .message = "[holoviz] This frame was consumed by a failed present (" +
                         presented.error().message + "); continue with the next acquire()."});
  }

  frame_table_.erase(args.image_index);
  tracker_.record(args.present_id, PresentTracker::Outcome::kQueued, render_done);

  return {};
}

expected<WaitResult> SwapchainPresentBackend::wait_for_present(PresentId id,
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

  // Consult the exact outcome first: a known drop answers kDropped, a sparse never-issued
  // id or one whose outcome is no longer tracked answers kNotFound — never a defaulted
  // kDisplayed, and never a native wait that a higher id's counter would falsely satisfy.
  const PresentTracker::LookupResult outcome = tracker_.lookup(id);
  switch (outcome.state) {
    case PresentTracker::Lookup::kDropped:
      // Definite bookkeeping knowledge, independent of display feedback.
      return WaitResult::kDropped;
    case PresentTracker::Lookup::kQueuedRetired:
      // Queued before the latest swapchain recreation. The recreation drained RENDERING,
      // which proves nothing about what the presentation engine displayed; kDisplayed
      // exclusively means display-confirmed, so the honest answer is unresolvable.
      return make_unexpected(Error{.code = ErrorCode::kNotFound,
                                   .message =
                                       "[holoviz] The frame's present outcome crossed a swapchain "
                                       "recreation and can no longer be resolved (waits across "
                                       "recreations are unsupported)."});
    case PresentTracker::Lookup::kNeverIssued:
      return make_unexpected(Error{.code = ErrorCode::kNotFound,
                                   .message = "[holoviz] Present id was never presented."});
    case PresentTracker::Lookup::kEvicted:
      return make_unexpected(
          Error{.code = ErrorCode::kNotFound,
                .message = "[holoviz] Present id's outcome is no longer tracked (older than "
                           "the last image_count + 2 tagged presents)."});
    case PresentTracker::Lookup::kQueuedLive:
      break;  // resolved below
  }

  if (!has_present_feedback_) {
    // No display-completion API (info().has_present_wait == false): never approximate
    // kDisplayed from render completion — a fence/timeline wait proves GPU submission
    // finished, not that the presentation engine displayed the frame. The caller holds
    // the frame's render_done SyncToken and can apply that weaker policy explicitly.
    return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                 .message =
                                     "[holoviz] Display completion is unavailable on this device "
                                     "(info().has_present_wait is false); wait on the frame's "
                                     "render_done SyncToken instead."});
  }

  // Native display-done wait (VK_KHR_present_wait through HoloViz).
  auto present_result = swapchain_.wait_for_present(id, timeout);
  if (!present_result) {
    if (present_result.error().code == ErrorCode::kOutOfDate) {
      // A recreation happened during the wait: it proves neither display nor drop for
      // this queued frame — same unresolvable outcome as kQueuedRetired above.
      return make_unexpected(Error{.code = ErrorCode::kNotFound,
                                   .message =
                                       "[holoviz] The frame's present outcome crossed a swapchain "
                                       "recreation and can no longer be resolved (waits across "
                                       "recreations are unsupported)."});
    }
    if (present_result.error().code == ErrorCode::kBackpressured) {
      return WaitResult::kTimeout;
    }

    return forward_error(present_result);
  }

  return WaitResult::kDisplayed;
}

expected<SwapchainInfo> SwapchainPresentBackend::swapchain_info() const {
  if (!swapchain_) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] swapchain_info() before create_swapchain()."});
  }
  return swapchain_info_;
}

expected<SyncToken> SwapchainPresentBackend::targets() const {
  if (!swapchain_) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                 .message = "[holoviz] targets() before create_swapchain()."});
  }
  // Minted on demand (a setup-path allocation): the payload only carries a weak reference
  // to the liveness cell, so the handle tracks the live swapchain across recreations and
  // goes inert with it at destroy().
  auto payload = std::make_shared<TargetSetPayload>(device_identity(*device_));
  payload->source = acquire_source_;
  return SyncToken{std::move(payload)};
}

PresentInfo SwapchainPresentBackend::info() const {
  bool present_feedback = has_present_feedback_;
  if (!swapchain_) {
    // Before create_swapchain() report straight from the device (the capability is a
    // device-creation fact, not a swapchain one).
    const auto caps = device_->capabilities();
    present_feedback = caps.has_value() && caps->present_completion;
  }
  return PresentInfo{
      .name = backend_name(),
      .has_present_id = present_feedback,
      .has_present_wait = present_feedback,
  };
}

}  // namespace holoscan::visualization::backends
