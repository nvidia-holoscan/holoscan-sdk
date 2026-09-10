// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Factory for the RHI render backend: the graphics-substrate implementation of the
// holoviz::RenderBackend seam, built on holoviz::rhi (pools and images at allocate(),
// pipelines at build(), record and submit per frame). Deliberately named for the RHI and
// not for any graphics API — the RHI hides which API drives the graphics hardware, and a
// future pure-CUDA backend compositing with kernels would be this backend's sibling behind
// the same seam.

#pragma once

#include <memory>

#include <cuda.h>

#include <holoviz/holoviz_export.h>

#include <holoviz/expected.hpp>
#include <holoviz/render_backend.hpp>
#include <holoviz/rhi/device.hpp>

namespace holoviz::backends {

struct RhiRenderBackendDesc {
  /// Stream the backend queues its per-frame CUDA work on (frame-data uploads, producer
  /// ordering). nullptr = the backend creates a private DEFAULT-flag stream at allocate()
  /// time if any declared layer needs one. Default-flag deliberately, not non-blocking: on
  /// at least one driver, async copies into Vulkan-imported mipmapped arrays land only
  /// partially when enqueued on a CU_STREAM_NON_BLOCKING stream (see the workaround note
  /// at the creation site). A default-flag stream serializes with the legacy null stream —
  /// applications that need concurrency with legacy-null-stream work must pass their own
  /// stream here instead of relying on the fallback.
  CUstream stream = nullptr;
  const char* debug_name = nullptr;
};

/// Create the RHI render backend on `device`. The device is borrowed and must outlive the
/// returned backend; destroy the backend before the device.
[[nodiscard]] HOLOVIZ_EXPORT expected<std::unique_ptr<RenderBackend>> create_rhi_render_backend(
    rhi::Device& device, const RhiRenderBackendDesc& desc);

}  // namespace holoviz::backends
