// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoviz {

/// How a rendered frame is presented to the display.
///
/// \c kAuto selects the best tear-free mode the platform supports: \c kFIFOLatestReady if
/// available, otherwise \c kFIFO. \c kAuto never selects \c kImmediate; that mode must be
/// requested explicitly. See \c PresentBackend::create_swapchain().
enum class PresentMode : uint8_t {
  kAuto,             ///< backend picks the best available tear-free mode (never kImmediate)
  kFIFO,             ///< tear-free; waits for the next vertical blank
  kFIFOLatestReady,  ///< tear-free; drops stale queued frames, presents the newest one
  kMailbox,  ///< tear-free; drops stale queued frames, presents the newest one (May skip present
             ///< IDs)
  /// No vsync wait — benchmarking / explicit opt-in only. On fixed-refresh displays this
  /// may tear; on a direct-display surface where the driver engages variable refresh
  /// (G-Sync / Adaptive-Sync; Display::vrr_capable is only the advertised capability),
  /// immediate presents drive the display's refresh from the present rate within its VRR
  /// range instead.
  kImmediate,
  /// vsync when a frame is ready in time; tears instead of stalling when it is late
  /// (Vulkan: VK_PRESENT_MODE_FIFO_RELAXED_KHR, a core VK_KHR_surface mode — no extension).
  /// Explicit opt-in only — \c kAuto never selects it; when unsupported it degrades
  /// tear-free like the other FIFO-family modes. Appended after kImmediate to keep
  /// existing enum values stable.
  kFIFORelaxed,
  /// One shared image the presentation engine reads on demand: each present() is a
  /// refresh request. Explicit opt-in at the swapchain level (VK_KHR_shared_presentable_image,
  /// CapabilityReport::shared_present); never selected by \c kAuto and rejected by
  /// \c PresentBackend. After the first frame's normal acquire/present handshake,
  /// acquire() returns pre-consumed gates — do not pass them to Device::submit().
  /// Render passes targeting a shared swapchain must be created with
  /// holoviz::RenderTargetDesc::shared_present (the image lives permanently in its
  /// shared-present layout), and the caller owns frame pacing: acquire() never blocks
  /// on the display in shared modes (pace with Device::wait_for_display_event() or a
  /// timer). Because those gate-less submits leave no per-frame record, retiring the
  /// swapchain (Swapchain::resize() / set_format() / destroy()) in a shared mode waits for
  /// every submission issued on the device so far, not just this swapchain's frames.
  /// \note On NVIDIA Linux drivers the present-timing feedback for shared images has been
  /// observed at a fixed 60 Hz regardless of the display mode; whether the scanout itself is
  /// capped is not yet confirmed. Until it is, prefer a non-shared swapchain with
  /// content-preserving passes for refresh-rate-critical output.
  kSharedDemandRefresh,
  /// Like \c kSharedDemandRefresh, but the presentation engine refreshes continuously
  /// from the shared image; presents after the first are harmless refresh hints.
  kSharedContinuousRefresh,
};

}  // namespace holoviz
