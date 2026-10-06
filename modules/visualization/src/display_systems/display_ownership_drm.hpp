/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <drm/drm_mode.h>

#include <cstdint>

namespace holoviz::display_systems::drm {

/// Connector naming as libdrm's drmModeGetConnectorTypeName spells it. HoloViz does
/// not link libdrm (see drm_ioctl.hpp), so the table lives here.
[[nodiscard]] constexpr const char* connector_type_name(uint32_t connector_type) noexcept {
  switch (connector_type) {
    case DRM_MODE_CONNECTOR_Unknown:
      return "Unknown";
    case DRM_MODE_CONNECTOR_VGA:
      return "VGA";
    case DRM_MODE_CONNECTOR_DVII:
      return "DVI-I";
    case DRM_MODE_CONNECTOR_DVID:
      return "DVI-D";
    case DRM_MODE_CONNECTOR_DVIA:
      return "DVI-A";
    case DRM_MODE_CONNECTOR_Composite:
      return "Composite";
    case DRM_MODE_CONNECTOR_SVIDEO:
      return "SVIDEO";
    case DRM_MODE_CONNECTOR_LVDS:
      return "LVDS";
    case DRM_MODE_CONNECTOR_Component:
      return "Component";
    case DRM_MODE_CONNECTOR_9PinDIN:
      return "DIN";
    case DRM_MODE_CONNECTOR_DisplayPort:
      return "DP";
    case DRM_MODE_CONNECTOR_HDMIA:
      return "HDMI-A";
    case DRM_MODE_CONNECTOR_HDMIB:
      return "HDMI-B";
    case DRM_MODE_CONNECTOR_TV:
      return "TV";
    case DRM_MODE_CONNECTOR_eDP:
      return "eDP";
    case DRM_MODE_CONNECTOR_VIRTUAL:
      return "Virtual";
    case DRM_MODE_CONNECTOR_DSI:
      return "DSI";
#ifdef DRM_MODE_CONNECTOR_DPI
    case DRM_MODE_CONNECTOR_DPI:
      return "DPI";
#endif
#ifdef DRM_MODE_CONNECTOR_WRITEBACK
    case DRM_MODE_CONNECTOR_WRITEBACK:
      return "Writeback";
#endif
#ifdef DRM_MODE_CONNECTOR_SPI
    case DRM_MODE_CONNECTOR_SPI:
      return "SPI";
#endif
#ifdef DRM_MODE_CONNECTOR_USB
    case DRM_MODE_CONNECTOR_USB:
      return "USB";
#endif
    default:
      return nullptr;
  }
}

}  // namespace holoviz::display_systems::drm
