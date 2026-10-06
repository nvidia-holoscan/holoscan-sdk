// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Where a visualization operator sends its rendered frames: the sink descriptors HolovizOp
// (<holoscan/visualization/holoviz_op.hpp>) accepts as any alternative of SinkDesc.
// WindowSinkDesc and ExclusiveDisplaySinkDesc present to a display; CudaSinkDesc captures every
// frame into CUDA device memory and publishes it on HolovizOp::output().

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
  kFIFORelaxed,      ///< vsync when a frame is ready in time, tears instead of stalling when it is
                     ///< late; explicit opt-in only (never selected by kAuto)
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
  /// Use a native sRGB attachment: filter and blend linear RGB, then encode on write.
  /// All layer colors (including geometry, text and LUT colors) must be linear, or image
  /// layers must declare a supported encoded ColorSpace. Alpha is always linear.
  /// False preserves the legacy UNORM framebuffer. Uses an sRGB swapchain format or a
  /// compatible native sRGB view of UNORM scanout; fails if neither is supported.
  bool framebuffer_srgb = false;
};

/// Present directly to a display no display server owns (lowest latency, no compositor).
/// Linux only (DRM acquisition): the process must run where no X11/Wayland session holds the
/// GPU, e.g. from a text console with the display manager stopped.
struct ExclusiveDisplaySinkDesc {
  /// Case-sensitive substring of the display name as enumerated by holoviz::enumerate_displays()
  /// (Linux: the DRM connector name, e.g. "DP-2", "HDMI-A-1"); empty = the first display no
  /// compositor owns.
  std::string display_name;
  /// Requested mode. 0 width/height = the largest supported resolution; 0 refresh_hz = the
  /// fastest refresh at that resolution. The closest supported mode wins (scored by width, then
  /// height, then refresh) and an inexact match logs the alternatives.
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t refresh_hz = 0;
  PresentMode present_mode = PresentMode::kAuto;
  /// Same color contract as WindowSinkDesc::framebuffer_srgb, without a window compositor.
  bool framebuffer_srgb = false;
};

/// Capture into CUDA device memory: every rendered frame is published on HolovizOp::output()
/// as one `holoscan::schema::ImageT` whose `data` Tensor is drawn from a Runtime tensor pool, so a
/// downstream operator (another HolovizOp's image layer, an encoder, an inference stage) consumes
/// it with zero copies. The frame carries its wire semantics: `width`/`height` are the declared
/// extent, and `encoding`/`color_space` follow \p format. The Tensor is CUDA-resident, tightly
/// packed (row pitch `width * bytes_per_pixel`), and typed by the format's native element:
///
/// | format                       | encoding | color_space | data Tensor          |
/// |------------------------------|----------|-------------|----------------------|
/// | kR8Unorm                     | MONO8    | UNSPECIFIED | uint8 `{H, W}`       |
/// | kR8G8B8A8Unorm / BGRA        | RGBA8 / BGRA8 | UNSPECIFIED | uint8 `{H, W, 4}` |
/// | kR8G8B8A8Srgb / BGRA         | RGBA8 / BGRA8 | SRGB   | uint8 `{H, W, 4}`    |
/// | kR16Unorm                    | MONO16   | UNSPECIFIED | uint16 `{H, W}`      |
/// | kR16G16B16A16Unorm           | RGBA16   | UNSPECIFIED | uint16 `{H, W, 4}`   |
///
/// The two sRGB kinds are stamped `ColorSpace_SRGB`, so a consuming HolovizOp admits them only
/// into an image layer declared with ColorSpace::kSRGBNonlinear (see ImageLayerDesc). The two
/// BGRA kinds are rendered into an RGBA target (CUDA arrays have no BGRA layout) and the resolve
/// swaps the red and blue bytes in place: one extra pass over the frame, no extra allocation.
///
/// \p width and \p height must be nonzero; \p format must be one of the formats above (a packed
/// single-plane format with a HoloViz render-target lowering and a wire encoding). Other formats
/// are rejected at HolovizOp::start() with kInvalidArgument. The pool is CUDA-resident, so the
/// deployment must bind the port's device (CompileOptions::deployment.bind_tensor_output_device
/// for `<operator>/output`) to the operator's HoloViz device: the first HoloViz adapter backed by
/// a CUDA device (the operator has no surface to pick an adapter from), which is not necessarily
/// CUDA device 0 on a multi-GPU host.
///
/// Frame pacing: there is no presentation engine. Each activation renders into a small ring of
/// exported render targets, resolves the finished frame into the loaned Tensor on the operator's
/// own CUDA stream — ordered after everything already queued on the activation stream, so a
/// pooled allocation recycled under a consumer's still-queued read is never overwritten early —
/// joins the activation stream to that work, and emits. When the consumer still holds every
/// pooled Tensor the activation is skipped (the frame is dropped), mirroring how the display
/// sinks treat a backpressured swapchain.
///
/// \note A zero-copy alternative that handed the render target's CUDA array to a caller
/// callback (a "CUDA array sink") was considered and removed for now to keep one CUDA path and
/// one delivery mechanism (the ImageT port); the pitch-linear resolve costs one device copy.
struct CudaSinkDesc {
  uint32_t width = 0;
  uint32_t height = 0;
  ImageFormat format = ImageFormat::kR8G8B8A8Unorm;
};

/// Exactly one sink kind. Closed set: adding an alternative is an API/ABI break.
using SinkDesc = std::variant<WindowSinkDesc, ExclusiveDisplaySinkDesc, CudaSinkDesc>;

}  // namespace holoscan::visualization
