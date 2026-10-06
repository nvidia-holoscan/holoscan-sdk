// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/backends/window_surface.hpp"

#include <memory>
#include <utility>

#include <holoviz/window.hpp>
#include "holoviz_op/types.hpp"

#include "holoviz_op/backends/backend_tokens.hpp"

namespace holoscan::visualization::backends {

expected<SurfaceHandle> make_window_surface_handle(const holoviz::Window& window) {
  VIZ_UNWRAP(holoviz::Surface, surface, window.create_surface());
  return wrap_surface_handle(std::move(surface));
}

expected<std::unique_ptr<PresentBackend>> create_window_surface(holoviz::Device& device) {
  if (!device) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] create_window_surface requires a valid device."});
  }
  return std::unique_ptr<PresentBackend>(new WindowSurface(device));
}

}  // namespace holoscan::visualization::backends
