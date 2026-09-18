// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Where a visualization operator sends its rendered frames: the sink descriptors HolovizOp
// (<holoscan/visualization/holoviz_op.hpp>) accepts as any alternative of SinkDesc. Only
// WindowSinkDesc has a present path today; the other kinds are declaration surface only (present
// paths wired in a later release) and HolovizOp::start() rejects them with kNotImplemented.

#pragma once

#include <cstdint>
#include <string>
#include <variant>

#include <holoscan/visualization/image_format.hpp>

namespace holoscan::visualization {

/// How a presented frame reaches the display.
///
/// \c kAuto selects the best tear-free mode the platform supports: \c kFIFOLatestReady if
/// available, otherwise \c kFIFO. \c kAuto never selects \c kImmediate; that mode must be
/// requested explicitly. HolovizOp translates this to the renderer's own vocabulary internally;
/// the public API carries no renderer headers.
enum class PresentMode : uint8_t {
  kAuto,             ///< best available tear-free mode (never kImmediate)
  kFIFO,             ///< tear-free; waits for the next vertical blank
  kFIFOLatestReady,  ///< tear-free; drops stale queued frames, presents the newest one
  kMailbox,          ///< tear-free; drops stale queued frames, presents the newest one (may skip
                     ///< present IDs)
  kImmediate,        ///< no vsync wait; may tear — benchmarking / explicit opt-in only
};

/// Present into a window. For a decorated window each zero dimension defaults independently
/// (\p width to 1280, \p height to 720); for a fullscreen window a zero \p width or \p height
/// selects the monitor's current video mode.
///
/// \note HolovizOp does not set \c __GL_SYNC_DISPLAY_DEVICE. When \p display_name is not the
/// primary output, export that variable before any Vulkan use in the process so the NVIDIA X11
/// driver paces presents on the named monitor (see the module README, "Fullscreen windows and
/// G-Sync/VRR").
struct WindowSinkDesc {
  std::string title;
  uint32_t width = 0;
  uint32_t height = 0;
  PresentMode present_mode = PresentMode::kAuto;
  bool fullscreen = false;   ///< undecorated, fixed-size; original video mode restored on close
  std::string display_name;  ///< fullscreen monitor (exact name); empty = primary
  uint32_t refresh_hz = 0;   ///< fullscreen refresh rate; 0 keeps the current one
  bool close_on_escape = false;
};

/// Present directly to a display the desktop does not own (lowest latency, no compositor).
/// Declaration surface only for now (present path wired in a later release).
struct ExclusiveDisplaySinkDesc {
  uint32_t display_index = 0;
  uint32_t width = 0;       ///< 0 = use the display's current mode
  uint32_t height = 0;      ///< 0 = use the display's current mode
  uint32_t refresh_hz = 0;  ///< 0 = use the display's current mode
  PresentMode present_mode = PresentMode::kAuto;
};

/// Capture into pitch-linear CUDA memory. Declaration surface only for now (present path wired in
/// a later release).
struct CudaPitchLinearSinkDesc {
  uint32_t width = 0;
  uint32_t height = 0;
  ImageFormat format = ImageFormat::kR8G8B8A8Unorm;
};

/// Capture into a CUDA array (e.g. for a CUDA-interop consumer). Declaration surface only for now
/// (present path wired in a later release).
struct CudaArraySinkDesc {
  uint32_t width = 0;
  uint32_t height = 0;
  ImageFormat format = ImageFormat::kR8G8B8A8Unorm;
};

/// Exactly one sink kind. Closed set: adding an alternative is an API/ABI break.
using SinkDesc = std::variant<WindowSinkDesc, ExclusiveDisplaySinkDesc, CudaPitchLinearSinkDesc,
                              CudaArraySinkDesc>;

}  // namespace holoscan::visualization
