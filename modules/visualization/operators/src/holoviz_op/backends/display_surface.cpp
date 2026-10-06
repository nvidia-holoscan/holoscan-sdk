// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/backends/display_surface.hpp"

#include <memory>
#include <utility>

#include "holoviz_op/types.hpp"

#include "holoviz_op/backends/backend_tokens.hpp"

namespace holoscan::visualization::backends {

expected<void> DisplaySurface::resize([[maybe_unused]] Extent2D new_extent) {
  return make_unexpected(
      Error{.code = ErrorCode::kInvalidArgument,
            .message = "[holoviz] Exclusive-display surfaces are fixed-size (the chosen display "
                       "mode); resize() is not supported by the display present backend."});
}

expected<SurfaceHandle> make_display_surface_handle(const holoviz::Display& display,
                                                    const holoviz::DisplaySurfaceDesc& desc) {
  VIZ_UNWRAP(holoviz::Surface, surface, display.create_surface(desc));
  return wrap_surface_handle(std::move(surface));
}

expected<std::unique_ptr<PresentBackend>> create_display_surface(holoviz::Device& device) {
  if (!device) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] create_display_surface requires a valid device."});
  }
  return std::unique_ptr<PresentBackend>(new DisplaySurface(device));
}

}  // namespace holoscan::visualization::backends
