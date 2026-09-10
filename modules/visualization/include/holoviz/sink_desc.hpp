// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <variant>

#include <holoviz/image_format.hpp>
#include <holoviz/present_mode.hpp>

namespace holoviz {

/// Where a HoloViz operator sends its rendered output. \c *SinkDesc types are the
/// caller-facing *declaration surface*: the application states its presentation intent by
/// constructing one of these. A \c PresentBackend implementation (see present_backend.hpp) is
/// the internal seam that *executes* that intent — sinks describe what, the backend does how.
///
/// \c WindowSinkDesc is implemented (the windowed Vulkan WSI present path).
/// \c ExclusiveDisplaySinkDesc, \c CudaPitchLinearSinkDesc, and \c CudaArraySinkDesc are part
/// of the stable declaration surface so callers can write code against them, but their present
/// paths are wired in a later release; a backend rejects an unimplemented sink at construction
/// time (not here in the header). Declaring all four now avoids call-site changes when those
/// present paths are wired. Note that \c SinkDesc is a closed \c std::variant: adding a new
/// alternative is an API- and ABI-breaking change, not an additive one.

/// Present into an application/compositor-managed window (windowed Vulkan WSI).
struct WindowSinkDesc {
  std::string title;
  uint32_t width = 0;
  uint32_t height = 0;
  PresentMode present_mode = PresentMode::kAuto;
};

/// Present fullscreen to a specific display via an exclusive (direct-to-display) swapchain —
/// lowest-latency path, no compositor. Declaration surface only for now (present path wired in
/// a later release).
struct ExclusiveDisplaySinkDesc {
  uint32_t display_index = 0;
  uint32_t width = 0;       ///< 0 = use the display's current mode
  uint32_t height = 0;      ///< 0 = use the display's current mode
  uint32_t refresh_hz = 0;  ///< 0 = use the display's current mode
  PresentMode present_mode = PresentMode::kAuto;
};

/// Capture the rendered frame into caller-provided pitch-linear CUDA device memory.
/// Declaration surface only for now (present path wired in a later release).
struct CudaPitchLinearSinkDesc {
  uint32_t width = 0;
  uint32_t height = 0;
  ImageFormat format = ImageFormat::kR8G8B8A8Unorm;
};

/// Capture the rendered frame into a CUDA array (e.g. for a CUDA-interop consumer).
/// Declaration surface only for now (present path wired in a later release).
struct CudaArraySinkDesc {
  uint32_t width = 0;
  uint32_t height = 0;
  ImageFormat format = ImageFormat::kR8G8B8A8Unorm;
};

/// The declared sink for an operator — exactly one kind. Closed set → variant (same rationale
/// as LayerDesc): compile-time exhaustiveness for every consumer that dispatches on it.
using SinkDesc = std::variant<WindowSinkDesc, ExclusiveDisplaySinkDesc, CudaPitchLinearSinkDesc,
                              CudaArraySinkDesc>;

}  // namespace holoviz
