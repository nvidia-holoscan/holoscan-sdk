/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoviz/display.hpp>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "holoviz/adapter.hpp"
#include "holoviz/common.hpp"
#include "holoviz/expected.hpp"

#include "detail/surface_impl.hpp"  // IWYU pragma: keep
#include "detail/window_impl.hpp"
#include "holoviz/window.hpp"
#include "vulkan/vk_display.hpp"
#include "window_systems/window_util.hpp"

namespace holoviz {

expected<std::vector<Display>> enumerate_displays(const Adapter& adapter) {
  switch (adapter.backend) {
    case Backend::kVulkan:
      return vulkan::enumerate_displays(adapter);
    default:
      return make_unexpected(Error{
          .code = ErrorCode::kNotImplemented,
          .message = "[holoviz] Only the Vulkan backend is implemented",
      });
  }
}

expected<Display> get_display(const std::string& name) {
  VIZ_UNWRAP(std::vector<Adapter>, adapters, enumerate_adapters());

  Display selected{};  // Display is a plain value type; copy the first match.
  bool found = false;
  // Display enumeration is best-effort per adapter: an adapter without a display path (or with
  // a transient enumeration failure) must not hide a matching display on a later adapter. The
  // skipped adapters are reported only when nothing matches, so the failure stays diagnosable.
  std::string skipped;
  for (const Adapter& adapter : adapters) {
    auto displays = enumerate_displays(adapter);
    if (!displays) {
      skipped += std::string(" \"") + adapter.name.data() + "\": " + displays.error().message + ";";
      continue;
    }
    for (const Display& display : *displays) {
      if (name.empty() || std::string(display.name.data()).find(name) != std::string::npos) {
        selected = display;
        found = true;
        break;
      }
    }
    if (found) {
      break;
    }
  }
  if (!found) {
    return make_unexpected(Error{
        .code = ErrorCode::kNotFound,
        .message = "[holoviz] No display matches \"" + name +
                   "\"; enumerate_displays() lists the connected displays." +
                   (skipped.empty() ? std::string{}
                                    : " Adapters skipped after enumeration errors:" + skipped),
    });
  }

  return selected;
}

expected<Display> display_for_window(const Window& window, const Adapter& adapter) {
  if (!window) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "[holoviz] display_for_window: empty window handle"});
  }
  VIZ_UNWRAP(window_systems::NativeMonitorInfo, monitor, window.impl().current_monitor());
  VIZ_UNWRAP(std::vector<Display>, displays, enumerate_displays(adapter));

  const std::optional<size_t> match = window_systems::match_display(displays, monitor);
  if (!match) {
    return make_unexpected(Error{
        .code = ErrorCode::kNotFound,
        .message = std::string("[holoviz] No unambiguous display match for the window's "
                               "monitor \"") +
                   monitor.name.data() + "\" on this adapter.",
    });
  }
  return displays[*match];
}

expected<std::vector<DisplayMode>> Display::modes() const {
  switch (backend) {
    case Backend::kVulkan:
      return vulkan::display_modes(*this);
    default:
      return make_unexpected(Error{
          .code = ErrorCode::kNotImplemented,
          .message = "[holoviz] Only the Vulkan backend is implemented",
      });
  }
}

expected<Surface> Display::create_surface(const DisplaySurfaceDesc& desc) const {
  switch (backend) {
    case Backend::kVulkan:
      return vulkan::display_create_surface(*this, desc).map([](auto impl) {
        return Surface{std::move(impl)};
      });
    default:
      return make_unexpected(Error{
          .code = ErrorCode::kNotImplemented,
          .message = "[holoviz] Only the Vulkan backend is implemented",
      });
  }
}

expected<NativeObject> Display::native_object(NativeObjectType type) const {
  switch (backend) {
    case Backend::kVulkan:
      return vulkan::display_native_object(*this, type);
    default:
      return make_unexpected(Error{
          .code = ErrorCode::kNotImplemented,
          .message = "[holoviz] Only the Vulkan backend is implemented",
      });
  }
}

}  // namespace holoviz
