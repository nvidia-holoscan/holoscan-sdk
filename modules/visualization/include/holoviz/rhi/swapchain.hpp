/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Swapchain and the gate model. Presentation engines only speak one-shot
// binary semaphores, which have no timeline equivalent and the sharpest
// validity rules in the API. Instead of exposing those objects, acquire()
// hands out two single-use gates per frame (pre-allocated at swapchain
// creation and recycled — acquire() allocates nothing):
//
//   AcquireSignalGate — "the image is ready": exactly one submit must wait on
//     it (SubmitDesc::acquire_gates) before writing the image.
//   PresentWaitGate — "rendering is done": exactly one submit signals it
//     (SubmitDesc::present_gates); present() then waits on it.
//
// The present handshake thus rides the caller's real work submission — no
// hidden shim submits, hand-written-Vulkan latency. Exactly-once consumption
// is tracked by the swapchain; consuming a gate twice is an error, and
// present() rejects a frame whose two gates are in inconsistent states.

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include <holoviz/holoviz_export.h>

#include "common.hpp"
#include "image.hpp"
#include "surface.hpp"

namespace holoviz::rhi {

/// Earliest pipeline stage at which a submit actually needs the acquired
/// image. kColorAttachmentOutput (the default) lets vertex work start before
/// the image is available; kAllCommands is the conservative fallback.
enum class GateWaitStage : uint8_t {
  kColorAttachmentOutput,
  kAllCommands,
};

namespace detail {
class AcquireSignalGateImpl;
class PresentWaitGateImpl;
class SwapchainImpl;
}  // namespace detail

/// Token for the image-acquired edge of one frame. Normal use: put it in
/// SubmitDesc::acquire_gates of the submit that first touches the image,
/// which consumes it.
class AcquireSignalGate {
 public:
  // We recycle the gate objects, so they are move-only. The impl pointer stays valid for
  // the lifetime of the swapchain *generation* that minted it: any recreation (an explicit
  // resize() or the automatic recovery inside acquire()/present()) invalidates outstanding
  // gates. Gates minted by acquire() carry a generation stamp: every method on a gate
  // whose generation has been superseded returns kInvalidState instead of touching the
  // recycled impl. Gates must still not outlive the Swapchain object itself.
  HOLOVIZ_EXPORT explicit AcquireSignalGate(detail::AcquireSignalGateImpl* impl);
  /// Backend-internal: a gate minted by Swapchain::acquire(), stamped with the swapchain
  /// generation that minted it. `origin` is borrowed and read only to compare generations.
  HOLOVIZ_EXPORT AcquireSignalGate(detail::AcquireSignalGateImpl* impl,
                                   const detail::SwapchainImpl* origin, uint64_t generation);
  AcquireSignalGate() = default;
  AcquireSignalGate(AcquireSignalGate&& other) noexcept
      : impl_(std::exchange(other.impl_, nullptr)),
        origin_(std::exchange(other.origin_, nullptr)),
        generation_(other.generation_) {}
  AcquireSignalGate& operator=(AcquireSignalGate&& other) noexcept {
    impl_ = std::exchange(other.impl_, nullptr);
    origin_ = std::exchange(other.origin_, nullptr);
    generation_ = other.generation_;
    return *this;
  }
  AcquireSignalGate(const AcquireSignalGate&) = delete;
  AcquireSignalGate& operator=(const AcquireSignalGate&) = delete;

  explicit operator bool() const { return impl_ != nullptr; }

  /// Consume the gate on the CPU instead of in a submit: block until the
  /// image is available. Costs an extra flush-submit — a fallback for frames
  /// that render nothing, not the fast path. kBackpressured on timeout: the
  /// gate is consumed queue-side but the drain is still in flight — retry
  /// wait() to finish it.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> wait(std::chrono::nanoseconds timeout);
  [[nodiscard]] HOLOVIZ_EXPORT expected<bool> consumed() const;

  /// Mark the gate consumed without waiting on it. Only for callers that have
  /// satisfied the acquire wait through native means; normally submit() does
  /// this bookkeeping.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> consume();

  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Stage the consuming submit blocks at; set before submitting.
  [[nodiscard]] HOLOVIZ_EXPORT expected<GateWaitStage> wait_stage() const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> set_wait_stage(GateWaitStage stage);

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty AND current-generation (these return
  /// references and cannot report staleness).
  detail::AcquireSignalGateImpl& impl() { return *impl_; }
  const detail::AcquireSignalGateImpl& impl() const { return *impl_; }

 private:
  detail::AcquireSignalGateImpl* impl_ = nullptr;
  /// Swapchain that minted this gate (borrowed; null for unguarded gates). Read only to
  /// compare generations; must not outlive the Swapchain object.
  const detail::SwapchainImpl* origin_ = nullptr;
  uint64_t generation_ = 0;
};

/// Token for the rendering-finished edge of one frame. Normal use: put it in
/// SubmitDesc::present_gates of the last submit that writes the image; the
/// binary present-wait signal then rides that submission.
class PresentWaitGate {
 public:
  enum class State : uint8_t { kEmpty, kPending, kArmed };

  // We recycle the gate objects, so they are move-only. The impl pointer stays valid for
  // the lifetime of the swapchain *generation* that minted it: any recreation (an explicit
  // resize() or the automatic recovery inside acquire()/present()) invalidates outstanding
  // gates. Gates minted by acquire() carry a generation stamp: every method on a gate
  // whose generation has been superseded returns kInvalidState instead of touching the
  // recycled impl. Gates must still not outlive the Swapchain object itself.
  HOLOVIZ_EXPORT explicit PresentWaitGate(detail::PresentWaitGateImpl* impl);
  /// Backend-internal: a gate minted by Swapchain::acquire(), stamped with the swapchain
  /// generation that minted it. `origin` is borrowed and read only to compare generations.
  HOLOVIZ_EXPORT PresentWaitGate(detail::PresentWaitGateImpl* impl,
                                 const detail::SwapchainImpl* origin, uint64_t generation);
  PresentWaitGate() = default;
  PresentWaitGate(PresentWaitGate&& other) noexcept
      : impl_(std::exchange(other.impl_, nullptr)),
        origin_(std::exchange(other.origin_, nullptr)),
        generation_(other.generation_) {}
  PresentWaitGate& operator=(PresentWaitGate&& other) noexcept {
    impl_ = std::exchange(other.impl_, nullptr);
    origin_ = std::exchange(other.origin_, nullptr);
    generation_ = other.generation_;
    return *this;
  }
  PresentWaitGate(const PresentWaitGate&) = delete;
  PresentWaitGate& operator=(const PresentWaitGate&) = delete;

  explicit operator bool() const { return impl_ != nullptr; }

  /// Arm the gate from the CPU with an empty submit — for frames that present
  /// without any rendering work to attach the signal to.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> signal();
  [[nodiscard]] HOLOVIZ_EXPORT expected<bool> consumed() const;

  /// Mark the gate consumed without signaling; see AcquireSignalGate::consume.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> consume();

  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty AND current-generation (these return
  /// references and cannot report staleness).
  detail::PresentWaitGateImpl& impl() { return *impl_; }
  const detail::PresentWaitGateImpl& impl() const { return *impl_; }

 private:
  detail::PresentWaitGateImpl* impl_ = nullptr;
  /// Swapchain that minted this gate (borrowed; null for unguarded gates). Read only to
  /// compare generations; must not outlive the Swapchain object.
  const detail::SwapchainImpl* origin_ = nullptr;
  uint64_t generation_ = 0;
};

/// What the presentation engine does when the swapchain's extent no longer matches the
/// surface (mid-resize, before the caller recreates). kNone keeps the strict behavior:
/// the mismatch surfaces as ErrorCode::kOutOfDate. The other modes keep presents working
/// across the mismatch, letting the caller coalesce recreations (e.g. resize once when a
/// drag settles instead of once per size tick). Support is surface-dependent; an
/// unsupported mode fails swapchain creation with kInvalidArgument.
enum class SwapchainScaling : uint8_t {
  kNone,                ///< No scaling; extent mismatch reports kOutOfDate (default).
  kOneToOne,            ///< Present unscaled (centered); the surface crops or letterboxes.
  kStretch,             ///< Stretch to fill the surface, ignoring aspect ratio.
  kAspectRatioStretch,  ///< Stretch to fit, preserving aspect ratio (centered letterbox).
};

/// Format and color space are requests validated against the surface: an
/// unsupported format fails creation, while an unsupported color space falls
/// back with a warning. The surface is borrowed and must outlive the swapchain.
struct SwapchainDesc {
  std::optional<std::reference_wrapper<const Surface>> surface = std::nullopt;

  Format format = Format::kB8G8R8A8Unorm;
  ColorSpace color_space = ColorSpace::kSrgbNonlinear;
  PresentMode present_mode = PresentMode::kFIFO;
  uint32_t image_count = 3;
  SwapchainScaling scaling = SwapchainScaling::kNone;

  const char* debug_name = nullptr;
};

/// One acquired frame: which image to render (image_index), which
/// frame-in-flight slot it occupies (frame_index), and the two gates minted
/// for it. Move-only; give the whole object back to present(), unmodified:
/// present() validates the fields and rejects an out-of-range image_index
/// with kInvalidArgument and a mismatched frame_index with kFailure.
struct AcquireSyncToken {
  uint32_t image_index = 0;
  uint32_t frame_index = 0;
  /// Swapchain generation (Swapchain::generation()) at acquire time. present() rejects a
  /// token from a superseded generation with kFrameDropped; the gates individually return
  /// kInvalidState. Compare against Swapchain::generation() to probe staleness directly.
  uint64_t generation = 0;
  AcquireSignalGate acquire_gate;
  PresentWaitGate present_gate;
};

/// What present() consumed, reported through its optional out-parameter on success and
/// failure alike. `image` = the acquired frame is finished — queued, eaten by the
/// presentation engine, or dead with a retired generation — never present its token again.
/// `id` = the nonzero present_id entered the swapchain's strictly increasing sequence.
/// Neither set: nothing was consumed; the token is still live and the same present may be
/// retried (or the image released via an empty present).
struct PresentConsumed {
  bool image = false;
  bool id = false;
};

/// Presentable image ring for one window. Frame loop:
/// acquire() -> record/submit with both gates -> present().
/// The swapchain runs image_count + 1 frames-in-flight slots internally;
/// acquire() blocks while all slots are pending.
class Swapchain {
 public:
  HOLOVIZ_EXPORT explicit Swapchain(std::unique_ptr<detail::SwapchainImpl> impl);
  HOLOVIZ_EXPORT Swapchain();
  HOLOVIZ_EXPORT Swapchain(Swapchain&&) noexcept;
  HOLOVIZ_EXPORT Swapchain& operator=(Swapchain&&) noexcept;
  Swapchain(const Swapchain&) = delete;
  Swapchain& operator=(const Swapchain&) = delete;
  HOLOVIZ_EXPORT ~Swapchain();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying backend resources now instead of at destruction.
  /// \p timeout bounds how long to wait for the object to stop being in use:
  /// zero (the default) fails immediately with ErrorCode::kBackpressured if
  /// the object is in use; a non-zero timeout waits up to that duration before
  /// returning ErrorCode::kBackpressured. Swapchains are in use while in-flight
  /// rendering or queued presents are outstanding (the presentation engine
  /// still owns their resources); kBackpressured leaves the swapchain fully
  /// intact and retryable. On success (and on any error other than
  /// kBackpressured) the handle becomes empty. Safe to call on an empty handle
  /// (no-op success). The destructor calls this automatically and logs any
  /// error. Destroying the swapchain also destroys its Images and the gate
  /// objects it handed out. Unlike a resize() — where a superseded gate
  /// survives and reports kInvalidState — destruction frees the storage the
  /// gates' staleness check itself reads, so outstanding gates,
  /// AcquireSyncTokens, and Image references must not be used at all afterwards
  /// (mirroring native Vulkan object lifetime); drop them first.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  /// Recreate the presentation resources, in place. Resizing is normally automatic —
  /// acquire()/present() detect a stale or out-of-date swapchain and call this internally —
  /// so the explicit form exists for window systems whose surfaces do not report a size
  /// (e.g. Wayland, where the automatic path cannot know the new extent) and for callers
  /// that want to resize at a moment of their choosing. Compared to destroy() + a new
  /// create_swapchain(), the handle stays valid, the format/color space/present mode are
  /// preserved, and the presentation engine hands over from the old frames to the new ones
  /// without the surface ever being swapchain-less (Vulkan: VkSwapchainCreateInfoKHR::
  /// oldSwapchain), so a live window resize does not flash or stall on a full device drain.
  ///
  /// A zero \p new_extent means "match the surface"; pass the window's framebuffer size
  /// explicitly on window systems whose surfaces do not report one (e.g. Wayland). The
  /// extent is clamped to the surface's supported range either way.
  ///
  /// Blocks until in-flight rendering against the current images has retired (bounded by
  /// \p timeout — kBackpressured on expiry, and the swapchain is left unchanged).
  /// Invalidated by a successful resize: all outstanding AcquireSyncTokens and gates, and
  /// Image references from image() — re-fetch them, they remain keyed by the same indices.
  /// An acquired-but-unpresented frame is released back to the presentation engine
  /// internally; drop its token, do not present it. image_count() is normally unchanged
  /// (the same count is re-requested) but the driver may return a different number of
  /// images; re-query it after resizing. Present ids keep increasing across the resize,
  /// but wait_for_present() for ids presented before it is no longer supported.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> resize(
      Extent2D new_extent = {}, std::chrono::nanoseconds timeout = std::chrono::seconds(5));

  [[nodiscard]] HOLOVIZ_EXPORT expected<Extent2D> extent() const;

  /// Return the format actually chosen, which is the one render passes
  /// targeting this swapchain must be created with.
  [[nodiscard]] HOLOVIZ_EXPORT expected<Format> format() const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<ColorSpace> color_space() const;

  /// Return the present mode frames are currently presented with.
  [[nodiscard]] HOLOVIZ_EXPORT expected<PresentMode> present_mode() const;

  /// Switch the present mode without recreating the swapchain; takes effect from the next
  /// present(). Only modes the surface reports as compatible with the current swapchain can
  /// be switched to (the mode it was created with always is); an incompatible mode fails
  /// with kInvalidArgument and changes nothing — destroy and recreate the swapchain to get
  /// such a mode. The new mode also carries across subsequent resize() calls.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> set_present_mode(PresentMode mode);

  /// Return the swapchain-owned Image for an index. Identity-stable: index i
  /// always maps to the same wrapper, so per-image resources (command lists,
  /// render-target setups) can be keyed by index.
  [[nodiscard]] HOLOVIZ_EXPORT expected<std::reference_wrapper<const Image>> image(
      uint32_t index) const;

  [[nodiscard]] HOLOVIZ_EXPORT expected<uint32_t> image_count() const;

  /// Return the recreation generation: a counter that advances on every (re)creation of the
  /// presentation resources, whether from an explicit resize() or the automatic recovery
  /// inside acquire()/present(). Code that holds AcquireSyncTokens or Image references
  /// across frames can compare generations to detect an in-between recreation and drop the
  /// stale objects without touching them.
  [[nodiscard]] HOLOVIZ_EXPORT expected<uint64_t> generation() const;

  /// Acquire the next image, blocking until a frame slot frees up (timeout
  /// applies to that wait). Image indices arrive in driver order — do not
  /// assume round-robin.
  ///
  /// At most one acquire may be outstanding: calling acquire() again before the previous
  /// token was present()ed returns kInvalidState (the previous token stays valid). To give
  /// an acquired frame back without rendering it, present it unrendered (both gates
  /// unconsumed — one stale frame may flash); to drop it without presenting at all,
  /// resize() or destroy() the swapchain — both release it back to the presentation engine
  /// internally.
  ///
  /// Self-healing (only for swapchains on a Surface::resizable() surface — a fixed-size
  /// window or a direct-to-display output runs none of this and keeps the classic
  /// kOutOfDate contract): a swapchain that no longer matches its surface (the window
  /// resized, the driver reported out-of-date, or a previous recreation failed) is
  /// recreated here automatically at the surface's current size before an image is handed
  /// out — a resize needs no caller code on surfaces that report their extent. The
  /// caller-visible outcomes:
  ///  - kBackpressured — the wait timed out, or the surface currently has no presentable
  ///    area (minimized window); retry.
  ///  - kImageCountChanged — an internal recreation changed image_count(); rebuild
  ///    per-image resources, re-query, then re-acquire. Reported exactly once.
  ///  - kOutOfDate — automatic recreation itself failed (e.g. the surface reports no fixed
  ///    extent — Wayland); call resize() with an explicit extent.
  /// Any recreation invalidates outstanding AcquireSyncTokens/gates and Image references
  /// (same rules as resize()); per-frame image()/extent() re-fetches — the normal frame-loop
  /// pattern — always observe the current state.
  [[nodiscard]] HOLOVIZ_EXPORT expected<AcquireSyncToken> acquire(std::chrono::nanoseconds timeout);

  /// Queue the frame for presentation and recycle its slot. Both gates must
  /// be consumed together or not at all (an unrendered frame may present
  /// directly). A nonzero present_id tags the frame for wait_for_present()
  /// when the device reports CapabilityReport::present_completion (ids must be
  /// strictly increasing per swapchain); without the capability, or with id 0,
  /// the frame is untagged.
  ///
  /// Ownership on every outcome: when `consumed` is non-null it reports, on success and
  /// failure alike, exactly what this call took (see PresentConsumed). A failure that
  /// consumed nothing (a pre-queue rejection — out-of-range index, non-monotonic id, gate
  /// inconsistency, or a fence-allocation failure — or a non-enqueued out-of-memory result
  /// from the presentation engine, which the Vulkan spec guarantees leaves every referenced
  /// resource and synchronization primitive unaffected) leaves the passed token valid, the
  /// acquire outstanding, and the id sequence unadvanced — the same present may be retried,
  /// or the image released via an empty present. A failure that consumed the frame must
  /// never be retried with the same token.
  ///
  /// Self-healing (resizable surfaces only, like acquire()): when the presentation engine
  /// reports the swapchain out of date, the frame is unavoidably consumed without being
  /// displayed; the swapchain is then recreated automatically and present() returns
  /// kFrameDropped (or kImageCountChanged when the recreation also changed image_count() —
  /// that frame was dropped too). Both are
  /// continue-with-the-next-frame conditions, not failures; a tagged present_id of a
  /// dropped frame never reaches the display, so do not wait_for_present() on it. If the
  /// automatic recreation itself fails, that error is returned instead (kBackpressured =
  /// the recovery drain timed out with the frame already consumed and the swapchain left
  /// as-is) and the next acquire() retries the recovery.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> present(AcquireSyncToken&& acquired,
                                                      SubmitSyncToken submit_sync_token,
                                                      uint64_t present_id = 0,
                                                      PresentConsumed* consumed = nullptr);

  /// Block (event-driven) until the frame tagged `present_id` has been presented to the
  /// display. Requires CapabilityReport::present_completion — kNotImplemented otherwise,
  /// with no hidden fallback: a render-done approximation is the caller's policy decision,
  /// not this primitive's. kBackpressured on timeout.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> wait_for_present(uint64_t present_id,
                                                               std::chrono::nanoseconds timeout);

  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::SwapchainImpl& impl() { return *impl_; }
  const detail::SwapchainImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::SwapchainImpl> impl_;
};

}  // namespace holoviz::rhi
