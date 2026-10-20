// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include <holoscan/visualization/holoviz_op.hpp>  // ImageFormat
#include <holoviz/expected.hpp>
#include <holoviz/present_mode.hpp>
#include "holoviz_op/backends/handles.hpp"  // AcquireToken, FrameCommands, SyncToken, ...
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

/// Monotonically increasing per-frame present identifier.
using PresentId = uint64_t;

/// Result of \c PresentBackend::wait_for_present().
enum class WaitResult : uint8_t {
  kDisplayed,  ///< the frame with the requested \c PresentId was displayed before the deadline
  kTimeout,    ///< the deadline elapsed before display was confirmed
  kDropped,    ///< the frame with the requested \c PresentId was dropped (never displayed)
};

/// Pass as \p timeout to \c PresentBackend::wait_for_present() to wait without a deadline.
inline constexpr auto kWaitForever = std::chrono::nanoseconds::max();

// SurfaceHandle is defined in "holoviz_op/backends/handles.hpp", shared with
// RenderBackend so the opaque handle types live in one place instead of being redeclared per
// seam.

/// Plain width/height pair; avoids pulling a concrete graphics-API extent type into this
/// interface.
struct Extent2D {
  uint32_t width = 0;
  uint32_t height = 0;
};

/// Actual swapchain properties after creation — returned by
/// \c PresentBackend::create_swapchain() so the renderer can size its frame ring correctly.
///
/// A backend may clamp \p extent, select a different \p present_mode, choose a
/// format/color-space from what the surface supports, and allocate more or fewer images than
/// requested. Callers must use these values, not the values in \p SwapchainDesc.
struct CreatedSwapchainInfo {
  Extent2D extent;           ///< actual pixel dimensions (may differ from preferred)
  uint32_t image_count = 0;  ///< actual number of swapchain images allocated
  holoviz::PresentMode present_mode =
      holoviz::PresentMode::kAuto;  ///< actually selected present mode
};

struct SwapchainDesc {
  SurfaceHandle surface;
  Extent2D preferred_extent;
  holoviz::PresentMode preferred_mode = holoviz::PresentMode::kAuto;

  /// Requested image count. \c nullopt lets the backend choose the appropriate default
  /// for the sink kind (typically 2 for exclusive display, 3 for windowed). Backends clamp
  /// to the surface's reported min/max image count regardless.
  std::optional<uint32_t> image_count;
  /// Require a native sRGB attachment with the sRGB nonlinear presentation color space.
  bool framebuffer_srgb = false;
};

/// Arguments to present(). `render_done` is the SyncToken returned by
/// RenderBackend::submit() for this frame; null presents an empty frame — nothing was
/// rendered into the image this cycle, or the frame was aborted
/// (RenderBackend::abort_frame()). The empty present is also the release path for an
/// acquired image that will not be rendered: it frees the image's pending-frame slot so
/// the image can be re-acquired. The image is presented with whatever contents it had
/// (one stale/undefined frame may flash), which is acceptable for an error path. (A
/// flash-free release API over vkReleaseSwapchainImagesEXT is a possible follow-up if
/// the cosmetic ever matters.)
struct PresentArgs {
  uint32_t image_index = 0;
  /// The exact acquire being presented: copy \c AcquireToken::issuance from this frame's
  /// acquire(). The image index alone is recyclable (a fresh acquire after a recreation
  /// can reuse it), so backends that issue nonzero issuances require a match here —
  /// presenting a superseded acquire fails with \c kFrameDropped without touching the
  /// fresh acquire's pending frame, and a zero/missing issuance is rejected with
  /// \c kInvalidArgument.
  uint64_t acquire_issuance = 0;
  SyncToken render_done;
  /// 0 = untagged. Nonzero ids must be strictly increasing across present() calls.
  PresentId present_id = 0;
  /// 0 = present as soon as possible; nonzero = a deadline hint for future deadline-based
  /// pacing work (e.g. VK_EXT_present_timing). The design must not preclude this field being
  /// used, even though nothing populates it yet.
  uint64_t target_present_ns = 0;
};

/// Swapchain state as last observed by the backend. Recreation is normally automatic
/// (inside acquire()/present() on resizable surfaces), so the extent can change from one
/// frame to the next without any call of yours in between; treat it as last-known, re-read
/// after acquire() when it matters. The image count only changes with kImageCountChanged
/// (or an explicit resize()), and the mode/format are stable across recreations.
struct SwapchainInfo {
  Extent2D extent;
  uint32_t image_count = 0;
  /// The concrete mode the backend resolved (never kAuto).
  holoviz::PresentMode active_mode = holoviz::PresentMode::kFIFO;

  ImageFormat format = ImageFormat::kUnknown;  ///< native format of the swapchain images
};

/// Measured present-timing feedback. This is a placeholder pending a real contract:
/// clock domain, epoch, and whether zero is a valid timestamp are currently undefined.
/// PresentId correlation is also missing — a future revision should add an identified
/// timing result. Backends that don't support timing never return a value from
/// \c query_timing().
struct PresentTiming {
  uint64_t actual_present_ns = 0;    ///< placeholder; clock domain TBD
  uint64_t earliest_present_ns = 0;  ///< placeholder; clock domain TBD
};

/// Static backend identity and feature-availability summary, for logging/diagnostics.
struct PresentInfo {
  std::string name;
  bool has_present_id = false;
  bool has_present_wait = false;
};

/// \c PresentBackend owns the acquire → render → present loop: the HoloViz operator never
/// delegates this to an external helper (e.g. \c nvvk::SwapChain). The per-frame call
/// sequence is:
///
/// \code
/// AcquireToken acquired = present_backend.acquire();        // event-driven; never busy-polls
/// FrameCommands cmds    = render_backend.begin_frame(acquired, frame_slot);
/// render_backend.record(cmds, render_list, composite);
/// SyncToken render_done = render_backend.submit(cmds, SubmitContext{frame_slot});
/// present_backend.present(
///     PresentArgs{acquired.image_index, acquired.issuance, render_done, present_id});
/// \endcode
///
/// On a per-frame error after acquire() (a failed record() or submit()), abort the frame
/// and release the image via an empty present so its slot can be re-acquired:
///
/// \code
/// render_backend.abort_frame(cmds);
/// present_backend.present(
///     PresentArgs{acquired.image_index, acquired.issuance, /*render_done=*/{}, 0});
/// \endcode
///
/// If begin_frame() itself fails (e.g. kBackpressured), there is no frame to abort — release
/// the image via the same empty present and keep the slot's previous producer buffers alive:
/// only a successful begin_frame()/abort_frame()/destroy() grants recycle permission (see the
/// FrameData lifetime contract in frame_data.hpp).
///
/// \c acquire() and \c wait_for_present() must be event-driven and must never busy-poll.
/// Backends must gracefully fall back when a preferred present mode or sync mechanism is
/// unavailable — e.g. \c kFIFOLatestReady → \c kFIFO.
///
/// \note **Scope**: these interfaces are currently in-tree only (no third-party backend plugin
/// ABI is defined). A future revision will add capability negotiation, export visibility
/// policy, and ABI versioning if third-party backends become a supported use case.
class PresentBackend {
 public:
  virtual ~PresentBackend() = default;

  /// Create the swapchain. Returns the actual properties after backend clamping so the
  /// renderer can size its frame-in-flight ring correctly before \c allocate() is called.
  [[nodiscard]] virtual expected<CreatedSwapchainInfo> create_swapchain(
      const SwapchainDesc& desc) = 0;
  [[nodiscard]] virtual expected<void> resize(Extent2D new_extent) = 0;

  /// Tear down the swapchain and release the surface reference. Blocks (bounded by
  /// `timeout`) until the device is done with the swapchain images. The destructor calls
  /// this and logs failures.
  [[nodiscard]] virtual expected<void> destroy(std::chrono::nanoseconds timeout) = 0;

  /// Acquire an image slot and its readiness dependency. Blocks (event-driven) if none is
  /// available; never busy-polls. The returned \p AcquireToken must be passed to
  /// \c RenderBackend::begin_frame() before recording commands for this frame.
  ///
  /// Backends may bound how many acquire payloads the caller retains concurrently (the
  /// swapchain backends: image_count + 2, counting live AcquireTokens and render-backend
  /// keep-alive). Exceeding the bound fails with \c kBackpressured without acquiring
  /// anything — release a retained token and retry.
  [[nodiscard]] virtual expected<AcquireToken> acquire(std::chrono::nanoseconds timeout) = 0;

  /// Queue the acquired frame identified by `args.image_index` + `args.acquire_issuance`
  /// for presentation, tagged with `args.present_id`. Returns immediately; the frame is
  /// queued to the display.
  ///
  /// Ownership on failure: \c kFrameDropped and \c kImageCountChanged mean the frame was
  /// consumed (it never reaches the display) and its nonzero id was burned —
  /// \c wait_for_present() on it answers \c kDropped; continue with the next acquire().
  /// Presenting an acquire that predates a swapchain recreation answers this consumed
  /// signal too, even when the recreation changed the image count.
  /// Any other error left the frame pending and its id unconsumed: the same present may
  /// be retried, or the image released via an empty present (null \c render_done, id 0).
  /// Backends never return a retryable error for a consumed frame: a lower-level failure
  /// that consumed the frame (including a lost device, which the next acquire() reports
  /// too) is reported as \c kFrameDropped with the native cause in the message, never as
  /// its raw retryable-looking code.
  [[nodiscard]] virtual expected<void> present(const PresentArgs& args) = 0;

  /// Event-driven wait until \p id has been displayed. Never busy-polls. Pass
  /// \c kWaitForever for \p timeout to wait without a deadline.
  /// (No default argument: a default on a virtual binds to the caller's static type, so
  /// an override with a different default would silently change the blocking behavior.)
  ///
  /// \return \c WaitResult::kDisplayed if — and only if — the presentation engine
  ///         confirmed the frame reached the display before \p timeout elapsed,
  ///         \c WaitResult::kTimeout if the deadline passed first, or
  ///         \c WaitResult::kDropped if the frame is known to have never reached the
  ///         display (e.g. it was consumed by a swapchain recreation). \c kDropped is
  ///         definite bookkeeping knowledge and is reported regardless of display
  ///         feedback.
  /// A \p id that was never presented — including a gap in a sparse increasing sequence —
  /// fails with \c kNotFound, as does an id old enough that its outcome is no longer
  /// tracked (backends keep exact outcomes for at least the last image_count + 2 tagged
  /// presents) or a queued id whose outcome crossed a swapchain recreation (waits across a
  /// recreation are unsupported — the recreation drains rendering, which proves nothing
  /// about display).
  /// \note Backends without \c VK_KHR_present_wait or an equivalent display-completion
  /// API report \c has_present_wait = false in \c info() and fail queued-frame waits with
  /// \c kNotImplemented — never a render-completion approximation dressed up as
  /// \c kDisplayed (a fence/timeline wait proves GPU submission completion, NOT display).
  /// Callers wanting that weaker signal wait on the frame's \c render_done \c SyncToken
  /// themselves.
  [[nodiscard]] virtual expected<WaitResult> wait_for_present(PresentId id,
                                                              std::chrono::nanoseconds timeout) = 0;

  /// Opaque handle enumerating the current swapchain images, for
  /// \c RenderBackend::register_targets(). Stays valid across recreations (it tracks the
  /// live swapchain, not one generation of it). Default: an empty token — backends
  /// without enumerable targets have nothing to pre-register.
  [[nodiscard]] virtual expected<SyncToken> targets() const { return SyncToken{}; }

  /// Query the most recently available present-timing measurement, if any.
  ///
  /// \note This is a placeholder. The current \c PresentTiming has no \c PresentId for
  /// correlation and no defined clock domain — callers cannot safely interpret the
  /// timestamps. A future revision will add an identified, clocked timing result. Default
  /// returns \c nullopt for backends that don't support timing.
  virtual std::optional<PresentTiming> query_timing() { return std::nullopt; }

  [[nodiscard]] virtual expected<SwapchainInfo> swapchain_info() const = 0;

  /// \return static backend identity and feature-availability summary.
  virtual PresentInfo info() const = 0;
};

}  // namespace holoscan::visualization::backends
