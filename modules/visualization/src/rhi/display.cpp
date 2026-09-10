/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoviz/rhi/display.hpp>

#include <utility>

#include "holoviz/expected.hpp"
#include "holoviz/rhi/adapter.hpp"
#include "holoviz/rhi/common.hpp"

#include "detail/surface_impl.hpp"  // IWYU pragma: keep
#include "vulkan/vk_display.hpp"

namespace holoviz::rhi {

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

}  // namespace holoviz::rhi
