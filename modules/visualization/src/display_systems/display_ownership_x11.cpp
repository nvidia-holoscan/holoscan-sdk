/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// X11/RandR classification + vkAcquireXlibDisplayEXT acquisition (compiled on
// Linux when libX11/libXrandr are available). The only TU that includes X
// headers; see display_ownership_x11.hpp for the macro-quarantine rationale.

#include "display_ownership_x11.hpp"

#include <cstdio>
#include <string>
#include <vector>

#include <holoviz/logger.hpp>

#include "vulkan/vk_display_util.hpp"
#include "vulkan/vk_errors.hpp"

// After every Vulkan/holoviz header: Xlib defines None/Bool/Status macros that
// would otherwise mangle them. volk's Display/RROutput typedefs (under
// VK_USE_PLATFORM_XLIB_XRANDR_EXT) are identical to the real ones below.
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>

namespace holoviz::display_systems::x11 {

namespace {

/// The X type, disambiguated from holoviz::Display.
using XDisplay = ::Display;

expected<XDisplay*> open_x_connection() {
  XDisplay* connection = XOpenDisplay(nullptr);
  if (connection == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Cannot open the X display (is DISPLAY valid?)."});
  }
  return connection;
}

/// The RandR "non-desktop" output property (RandR 1.6): set by the server for
/// outputs excluded from the desktop (VR headsets, or a user opt-out). Read for
/// logging only; classification keys on CRTC state.
bool read_non_desktop_property(XDisplay* connection, RROutput output) {
  const Atom non_desktop_atom = XInternAtom(connection, "non-desktop", True);
  if (non_desktop_atom == None) {
    return false;
  }
  Atom actual_type = None;
  int actual_format = 0;
  unsigned long item_count = 0;
  unsigned long bytes_after = 0;
  unsigned char* value = nullptr;
  const int status = XRRGetOutputProperty(connection,
                                          output,
                                          non_desktop_atom,
                                          0,
                                          1,
                                          False,
                                          False,
                                          AnyPropertyType,
                                          &actual_type,
                                          &actual_format,
                                          &item_count,
                                          &bytes_after,
                                          &value);
  bool non_desktop = false;
  if (status == Success && value != nullptr) {
    if (actual_type == XA_INTEGER && actual_format == 32 && item_count >= 1) {
      non_desktop = *reinterpret_cast<long*>(value) != 0;
    }
    XFree(value);
  }
  return non_desktop;
}

/// Scanout timing of the output's CRTC; zeroed when the output drives no CRTC.
DisplayMode read_current_mode(XDisplay* connection, XRRScreenResources* resources,
                              const XRROutputInfo& output_info, bool* has_active_crtc) {
  *has_active_crtc = false;
  if (output_info.crtc == None) {
    return {};
  }
  XRRCrtcInfo* crtc_info = XRRGetCrtcInfo(connection, resources, output_info.crtc);
  if (crtc_info == nullptr) {
    return {};
  }
  DisplayMode mode{};
  if (crtc_info->mode != None) {
    *has_active_crtc = true;
    for (int i = 0; i < resources->nmode; ++i) {
      const XRRModeInfo& mode_info = resources->modes[i];
      if (mode_info.id != crtc_info->mode) {
        continue;
      }
      mode.resolution = {static_cast<uint32_t>(mode_info.width),
                         static_cast<uint32_t>(mode_info.height)};
      if (mode_info.hTotal > 0 && mode_info.vTotal > 0) {
        mode.refresh_rate_mhz =
            static_cast<uint32_t>(mode_info.dotClock * 1000ULL /
                                  (static_cast<uint64_t>(mode_info.hTotal) * mode_info.vTotal));
      }
      break;
    }
  }
  XRRFreeCrtcInfo(crtc_info);
  return mode;
}

}  // namespace

expected<std::vector<NativeDisplayInfo>> classify_outputs(VkPhysicalDevice physical_device) {
  if (vkGetRandROutputDisplayEXT == nullptr) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotImplemented,
              .message = "[holoviz] vkGetRandROutputDisplayEXT is unavailable."});
  }

  VIZ_UNWRAP(XDisplay*, connection, open_x_connection());

  int randr_major = 0;
  int randr_minor = 0;
  if (XRRQueryVersion(connection, &randr_major, &randr_minor) == 0 || randr_major < 1 ||
      (randr_major == 1 && randr_minor < 2)) {
    XCloseDisplay(connection);
    return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                 .message =
                                     "[holoviz] The X server lacks RandR >= 1.2; X11 display "
                                     "classification disabled."});
  }

  const Window root = DefaultRootWindow(connection);
  // 1.3's Current variant skips a full probe; the server's cached state is
  // exactly what the compositor acts on.
  XRRScreenResources* resources = (randr_major > 1 || randr_minor >= 3)
                                      ? XRRGetScreenResourcesCurrent(connection, root)
                                      : XRRGetScreenResources(connection, root);
  if (resources == nullptr) {
    XCloseDisplay(connection);
    return make_unexpected(Error{.code = ErrorCode::kFailure,
                                 .message = "[holoviz] XRRGetScreenResources failed; X11 "
                                            "display classification disabled."});
  }

  std::vector<NativeDisplayInfo> displays;
  int unresolved_connected_outputs = 0;
  for (int i = 0; i < resources->noutput; ++i) {
    const RROutput output = resources->outputs[i];
    XRROutputInfo* output_info = XRRGetOutputInfo(connection, resources, output);
    if (output_info == nullptr) {
      continue;
    }
    if (output_info->connection != RR_Connected) {
      XRRFreeOutputInfo(output_info);
      continue;
    }

    // The authoritative join: the driver resolves which VkDisplayKHR this
    // RandR output is. Fails for outputs on another GPU (PRIME) — skip those,
    // they keep their DRM classification.
    VkDisplayKHR vk_display = VK_NULL_HANDLE;
    if (vkGetRandROutputDisplayEXT(physical_device, connection, output, &vk_display) !=
            VK_SUCCESS ||
        vk_display == VK_NULL_HANDLE) {
      ++unresolved_connected_outputs;
      XRRFreeOutputInfo(output_info);
      continue;
    }

    bool has_active_crtc = false;
    const DisplayMode current_mode =
        read_current_mode(connection, resources, *output_info, &has_active_crtc);
    const bool non_desktop = read_non_desktop_property(connection, output);

    NativeDisplayInfo info{};
    info.vk_display = reinterpret_cast<uint64_t>(vk_display);
    info.ownership = vulkan::classify_x11_output(true, has_active_crtc, non_desktop);
    info.current_mode = current_mode;
    std::snprintf(info.name.data(), info.name.size(), "%s", output_info->name);
    displays.push_back(info);

    XRRFreeOutputInfo(output_info);
  }

  XRRFreeScreenResources(resources);
  XCloseDisplay(connection);

  if (displays.empty()) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotFound,
              .message = std::string("[holoviz] No RandR output resolves to a Vulkan display of "
                                     "this adapter (") +
                         std::to_string(unresolved_connected_outputs) +
                         " connected but unresolved — XWayland or a different GPU?)."});
  }
  if (unresolved_connected_outputs > 0) {
    HOLOVIZ_LOG_DEBUG(
        "[holoviz] {} connected RandR output(s) did not resolve to this "
        "adapter's Vulkan displays; they keep their DRM classification.",
        unresolved_connected_outputs);
  }
  return displays;
}

expected<Acquisition> acquire(VkPhysicalDevice physical_device, uint64_t vk_display) {
  if (vkAcquireXlibDisplayEXT == nullptr || vkGetRandROutputDisplayEXT == nullptr) {
    return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                 .message = "[holoviz] vkAcquireXlibDisplayEXT is unavailable."});
  }

  VIZ_UNWRAP(XDisplay*, connection, open_x_connection());

  // Re-resolve the target display through this connection before acquiring:
  // the driver associates the RROutput with the connection in
  // vkGetRandROutputDisplayEXT, and acquisition needs that association (a
  // handle resolved on an earlier, closed connection acquires with
  // VK_ERROR_UNKNOWN on NVIDIA).
  bool resolved_on_connection = false;
  if (XRRScreenResources* resources =
          XRRGetScreenResourcesCurrent(connection, DefaultRootWindow(connection));
      resources != nullptr) {
    for (int i = 0; i < resources->noutput && !resolved_on_connection; ++i) {
      VkDisplayKHR resolved = VK_NULL_HANDLE;
      if (vkGetRandROutputDisplayEXT(
              physical_device, connection, resources->outputs[i], &resolved) == VK_SUCCESS &&
          reinterpret_cast<uint64_t>(resolved) == vk_display) {
        resolved_on_connection = true;
      }
    }
    XRRFreeScreenResources(resources);
  }
  if (!resolved_on_connection) {
    XCloseDisplay(connection);
    return make_unexpected(
        Error{.code = ErrorCode::kNotFound,
              .message = "[holoviz] The display no longer maps to any RandR output of this X "
                         "server; re-enumerate displays."});
  }

  const VkResult result = vkAcquireXlibDisplayEXT(
      physical_device, connection, reinterpret_cast<VkDisplayKHR>(vk_display));
  if (result != VK_SUCCESS) {
    XCloseDisplay(connection);
    return make_unexpected(
        Error{.code = ErrorCode::kVulkanError,
              .message = std::string("[holoviz] vkAcquireXlibDisplayEXT failed (") +
                         vulkan::vk_error_string(result, __FILE__, __LINE__) +
                         "). Detach the output from the desktop first (xrandr --output <name> "
                         "--off, or turn the monitor off in the desktop display settings), then "
                         "retry."});
  }

  return Acquisition{.x11_connection = connection};
}

void close_connection(void* x11_connection) {
  if (x11_connection != nullptr) {
    XCloseDisplay(static_cast<XDisplay*>(x11_connection));
  }
}

}  // namespace holoviz::display_systems::x11
