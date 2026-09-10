// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

namespace holoviz {

/// Handle and token types that flow across the \c RenderBackend / \c PresentBackend seam.
///
/// The opaque handle types (\c FrameCommands, \c SyncToken, \c SurfaceHandle) are
/// distinct one-field structs rather than aliases of \c std::shared_ptr<void>: each kind
/// is a semantically different token flowing between different calls, so passing one where
/// another is expected is a compile error rather than a silent mix-up. Nothing above the seam
/// may depend on a concrete backend type — what \p impl wraps is a backend-internal concern.
///
/// \c AcquireToken is also defined here because it flows from
/// \c PresentBackend::acquire() directly into \c RenderBackend::begin_frame(), crossing
/// the seam boundary; placing it in this shared header avoids a dependency between the two
/// backend headers.

/// Per-frame command-recording context, from \c RenderBackend::begin_frame().
struct FrameCommands {
  std::shared_ptr<void> impl;
};

/// Submit-completion signal (e.g. a Vulkan timeline-semaphore wait value). Produced by
/// \c RenderBackend::submit(), consumed by \c PresentBackend::present() — neither side
/// needs the concrete type.
struct SyncToken {
  std::shared_ptr<void> impl;
};

/// Backend-owned display-surface handle (e.g. a \c VkSurfaceKHR for the Vulkan WSI backend).
struct SurfaceHandle {
  std::shared_ptr<void> impl;
};

/// Result of \c PresentBackend::acquire(): the swapchain image slot to render into and the
/// dependency the renderer must observe before writing to it. Passed into
/// \c RenderBackend::begin_frame() so the renderer can select the correct framebuffer and
/// wait on the acquire semaphore. The \p image_index and \p issuance are also forwarded to
/// \c PresentBackend::present() to identify which acquired frame to queue for display —
/// the index alone is recyclable (a fresh acquire after a recreation can land on the same
/// index), so present() validates the exact acquire by its issuance, never by index alone.
struct AcquireToken {
  uint32_t image_index = 0;  ///< swapchain slot to render into this frame
  SyncToken image_ready;     ///< signal: the presentation engine has released the image
  /// Identity of this exact acquire, unique per acquire() call for the backend's lifetime
  /// (nonzero for backends that validate it). Copy into
  /// \c PresentArgs::acquire_issuance for this frame's present().
  uint64_t issuance = 0;
};

}  // namespace holoviz
