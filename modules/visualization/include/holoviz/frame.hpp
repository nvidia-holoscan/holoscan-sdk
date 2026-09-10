// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace holoviz {

/// The active sub-rectangle of an image layer's declared maximum extent for one frame.
///
/// An \p ImageLayerDesc declares a maximum \p max_width x \p max_height (the pool size).
/// Each frame binds a \p Subregion of that maximum — the region actually populated by the
/// source this frame. A variable-resolution source (e.g. a camera) varies its \p Subregion
/// while the underlying pool stays fixed, so no runtime reallocation occurs as long as
/// \p x + \p width <= \p max_width and \p y + \p height <= \p max_height.
///
/// The full-extent sentinel is exactly \c {0, 0, 0, 0} (all fields zero): it means "use the
/// complete declared maximum". Any other combination where \p width or \p height is zero but
/// \p x or \p y is nonzero is invalid and must be rejected by the backend.
struct Subregion {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t width = 0;   ///< 0 only when all four fields are zero (full-extent sentinel)
  uint32_t height = 0;  ///< 0 only when all four fields are zero (full-extent sentinel)
};

/// Maximum number of image planes in an \p ImageFrame.
///
/// The \p ImageFormat declared in \p ImageLayerDesc determines the active plane count:
/// - 1 plane: \c kR8Unorm, \c kR8G8B8Unorm, \c kR8G8B8A8Unorm, \c kYUYV422 (packed)
/// - 2 planes: \c kNV12 (plane 0 = Y, plane 1 = interleaved UV)
inline constexpr std::size_t kMaxImagePlanes = 3;

/// Per-plane device-memory binding for one \p ImageFrame.
///
/// Absolute plane pointers cover both contiguous and discontiguous allocation models without
/// a variant: for a contiguous GXF-style buffer, set each plane's \p device_ptr to
/// \c base + color_planes[i].offset; for separately addressed planes, supply them directly.
/// Do not infer the UV plane offset as \c y_pitch * visible_height: decoded surfaces may
/// use an aligned allocation height. \p capacity_bytes enables bounds validation; set it to
/// the backing allocation's total byte size for the plane.
struct ImagePlaneBinding {
  const void* device_ptr = nullptr;
  std::size_t row_pitch_bytes = 0;  ///< bytes per row; 0 = tightly packed
  std::size_t capacity_bytes = 0;   ///< backing allocation size in bytes; 0 = unspecified
};

/// Per-frame binding for one image layer: the device-memory planes plus the sub-rectangle
/// of the declared maximum they occupy this frame.
///
/// The pixel format (and thus the active plane count) is declared once in \p ImageLayerDesc
/// and is not repeated here. The backend maps \p planes[0..n-1] according to the declared
/// format: plane 0 is always Y/luma or the sole packed plane; plane 1 is UV/chroma for
/// two-plane formats (NV12). Planes beyond the active count are ignored.
///
/// **Stream and lifetime contract (HolovizOp path):** In the intended Holoscan 5.x
/// \c HolovizOp integration, the runtime prepares every received \c Tensor before entering
/// \c compute(): it enqueues the producer readiness dependency and registers stream-aware
/// deallocation on \c ExecutionContext::cuda_stream(). An \p ImageFrame derived from a
/// received \c Sample<Tensor> is therefore valid for all device reads enqueued on that
/// stream within the same activation — no additional ready token is needed. The backend must
/// enqueue all source reads on \c ExecutionContext::cuda_stream() before \c compute()
/// returns and must not retain the pointer for a later host-side submission. A standalone
/// caller, or any path that uses a different CUDA stream or imports the source into Vulkan
/// directly, must supply explicit CUDA/Vulkan semaphore ordering and a completion lease.
struct ImageFrame {
  std::array<ImagePlaneBinding, kMaxImagePlanes> planes{};
  Subregion subregion;  ///< active sub-rect within the declared maximum; {0,0,0,0} = full extent
};

}  // namespace holoviz
